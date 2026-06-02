from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ASTEVOLVE_DATA_ROOT", PROJECT_ROOT / "data"))
DEFAULT_MANIFEST = DATA_ROOT / "external_kb" / "embedding_manifest_esm2_t6_8M_all.json"
DEFAULT_RECORDS = DATA_ROOT / "external_kb" / "substructure_records.jsonl"
DEFAULT_EMBEDDINGS = DATA_ROOT / "external_kb" / "embeddings_esm2_t6_8M_all.npy"
DEFAULT_METADATA = DATA_ROOT / "external_kb" / "embedding_metadata_esm2_t6_8M_all.jsonl"
DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"


AA = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = set("AILMFWVY")
CHARGED = set("KRDE")
POLAR = set("STNQY")
AROMATIC = set("FWYH")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_sequence(seq: str) -> str:
    allowed = set(AA)
    return "".join(aa if aa in allowed else "X" for aa in seq.upper())


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def normalize_vector(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    return x / max(float(norm), 1e-12)


def load_manifest(path: Path | None) -> dict:
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_records_by_id(path: Path) -> dict[str, dict]:
    return {rec["record_id"]: rec for rec in iter_jsonl(path)}


def summarize_hits(hits: list[dict]) -> dict:
    class_counts = Counter(hit["metadata"].get("substructure_class") for hit in hits)
    aa_counts = Counter()
    lengths = []

    for hit in hits:
        rec = hit.get("record") or {}
        seq = rec.get("sequence_fragment") or ""
        aa_counts.update(aa for aa in seq if aa in AA)
        if rec.get("span_length") is not None:
            lengths.append(float(rec["span_length"]))

    aa_total = sum(aa_counts.values())
    aa_frequency = {aa: aa_counts[aa] / aa_total if aa_total else 0.0 for aa in AA}

    return {
        "class_counts": dict(class_counts),
        "length_prior": {
            "min": min(lengths) if lengths else None,
            "median": float(np.median(lengths)) if lengths else None,
            "mean": float(np.mean(lengths)) if lengths else None,
            "max": max(lengths) if lengths else None,
        },
        "residue_class_bias": {
            "hydrophobic": sum(aa_frequency[aa] for aa in HYDROPHOBIC),
            "charged": sum(aa_frequency[aa] for aa in CHARGED),
            "polar": sum(aa_frequency[aa] for aa in POLAR),
            "aromatic": sum(aa_frequency[aa] for aa in AROMATIC),
        },
        "favored_residues": [aa for aa, _ in aa_counts.most_common(8)],
    }


class MagnetonExternalKB:
    def __init__(
        self,
        records: Path = DEFAULT_RECORDS,
        embeddings: Path = DEFAULT_EMBEDDINGS,
        metadata: Path = DEFAULT_METADATA,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        max_length: int = 1022,
    ) -> None:
        self.records_path = records
        self.embeddings_path = embeddings
        self.metadata_path = metadata
        self.model_name = model_name
        self.device_arg = device
        self.max_length = max_length

        raw_embeddings = np.load(self.embeddings_path).astype(np.float32)
        self.embeddings = normalize_rows(raw_embeddings)
        self.metadata = list(iter_jsonl(self.metadata_path))
        if len(self.metadata) != self.embeddings.shape[0]:
            raise ValueError(
                f"Metadata count {len(self.metadata)} != embeddings rows {self.embeddings.shape[0]}"
            )
        self.records = load_records_by_id(self.records_path)

        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None

    @classmethod
    def from_manifest(
        cls,
        manifest: Path = DEFAULT_MANIFEST,
        device: str = "auto",
        max_length: int = 1022,
    ) -> "MagnetonExternalKB":
        data = load_manifest(manifest)
        return cls(
            records=Path(data.get("records", DEFAULT_RECORDS)),
            embeddings=Path(data.get("embeddings", DEFAULT_EMBEDDINGS)),
            metadata=Path(data.get("metadata", DEFAULT_METADATA)),
            model_name=data.get("model", DEFAULT_MODEL),
            device=device,
            max_length=max_length,
        )

    def _load_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device_arg == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device_arg

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model.to(device)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def embed_sequence(self, sequence: str) -> np.ndarray:
        seq = normalize_sequence(sequence)
        if not seq:
            raise ValueError("Cannot embed an empty sequence.")
        if len(seq) > self.max_length:
            raise ValueError(f"Sequence length {len(seq)} exceeds max_length={self.max_length}.")

        self._load_model()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        with self._torch.no_grad():
            encoded = self._tokenizer(
                [seq],
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=True,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            out = self._model(**encoded)
            hidden = out.last_hidden_state.detach().cpu().numpy()
            residue_embeddings = hidden[0, 1 : 1 + len(seq), :]
            return residue_embeddings.mean(axis=0).astype(np.float32)

    def query_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        class_filter: str = "",
        exclude_index: int | None = None,
    ) -> list[dict]:
        query = normalize_vector(query_embedding.astype(np.float32))
        scores = self.embeddings @ query
        order = np.argsort(-scores)

        hits = []
        for idx in order:
            idx = int(idx)
            if exclude_index is not None and idx == exclude_index:
                continue
            row = self.metadata[idx]
            if class_filter and row.get("substructure_class") != class_filter:
                continue
            record = self.records.get(row["record_id"])
            hits.append(
                {
                    "rank": len(hits) + 1,
                    "index": idx,
                    "score": float(scores[idx]),
                    "metadata": row,
                    "record": record,
                }
            )
            if len(hits) >= top_k:
                break
        return hits

    def query_sequence(
        self,
        sequence: str,
        top_k: int = 20,
        class_filter: str = "",
    ) -> dict:
        embedding = self.embed_sequence(sequence)
        hits = self.query_embedding(embedding, top_k=top_k, class_filter=class_filter)
        return {
            "query": {
                "sequence_length": len(normalize_sequence(sequence)),
                "class_filter": class_filter or None,
                "model": self.model_name,
            },
            "hits": hits,
            "prior": summarize_hits(hits),
        }

    def query_index(
        self,
        index: int,
        top_k: int = 20,
        class_filter: str = "",
    ) -> dict:
        if not (0 <= index < self.embeddings.shape[0]):
            raise IndexError(index)
        hits = self.query_embedding(
            self.embeddings[index],
            top_k=top_k,
            class_filter=class_filter,
            exclude_index=index,
        )
        return {
            "query": self.metadata[index],
            "hits": hits,
            "prior": summarize_hits(hits),
        }


def compact_hit(hit: dict) -> dict:
    row = hit["metadata"]
    return {
        "rank": hit["rank"],
        "score": hit["score"],
        "index": hit["index"],
        "class": row.get("substructure_class"),
        "id": row.get("substructure_id"),
        "name": row.get("substructure_name"),
        "span_length": row.get("span_length"),
        "record_id": row.get("record_id"),
    }


def print_result(result: dict) -> None:
    print("[query]")
    print(json.dumps(result["query"], ensure_ascii=False, indent=2))
    print("\n[hits]")
    for hit in result["hits"]:
        row = hit["metadata"]
        print(
            f"{hit['rank']:>2}. score={hit['score']:.4f} "
            f"class={row.get('substructure_class')} "
            f"id={row.get('substructure_id')} "
            f"len={row.get('span_length')} "
            f"name={row.get('substructure_name')}"
        )
    print("\n[prior]")
    print(json.dumps(result["prior"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the ASTevolve Magneton external KB by sequence or by embedding index."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--records", type=Path, default=None)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--sequence", default="")
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument("--query-index", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--class-filter", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    records = args.records or Path(manifest.get("records", DEFAULT_RECORDS))
    embeddings = args.embeddings or Path(manifest.get("embeddings", DEFAULT_EMBEDDINGS))
    metadata = args.metadata or Path(manifest.get("metadata", DEFAULT_METADATA))
    model_name = args.model or manifest.get("model", DEFAULT_MODEL)

    kb = MagnetonExternalKB(
        records=records,
        embeddings=embeddings,
        metadata=metadata,
        model_name=model_name,
        device=args.device,
        max_length=args.max_length,
    )

    if args.sequence_file:
        sequence = "".join(
            line.strip()
            for line in args.sequence_file.read_text(encoding="utf-8").splitlines()
            if not line.startswith(">")
        )
        result = kb.query_sequence(
            sequence,
            top_k=args.top_k,
            class_filter=args.class_filter,
        )
    elif args.sequence:
        result = kb.query_sequence(
            args.sequence,
            top_k=args.top_k,
            class_filter=args.class_filter,
        )
    elif args.query_index is not None:
        result = kb.query_index(
            args.query_index,
            top_k=args.top_k,
            class_filter=args.class_filter,
        )
    else:
        raise SystemExit("Provide --sequence, --sequence-file, or --query-index.")

    if args.json:
        compact = {
            "query": result["query"],
            "hits": [compact_hit(hit) for hit in result["hits"]],
            "prior": result["prior"],
        }
        print(json.dumps(compact, ensure_ascii=False, indent=2))
    else:
        print_result(result)


if __name__ == "__main__":
    main()
