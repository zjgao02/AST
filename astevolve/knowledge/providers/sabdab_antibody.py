from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


AA = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = set("AILMFWVY")
CHARGED = set("KRHDE")
POLAR = set("STNQY")
AROMATIC = set("FWYH")


def _read_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _merge_dicts(*items: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        for key, value in (item or {}).items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = _merge_dicts(out[key], value)
            else:
                out[key] = value
    return out


def _top_residues(freq: Dict[str, Any], n: int = 8) -> list[str]:
    pairs = []
    for aa, value in freq.items():
        try:
            pairs.append((str(aa), float(value)))
        except (TypeError, ValueError):
            continue
    return [aa for aa, _ in sorted(pairs, key=lambda x: x[1], reverse=True)[:n]]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _normalize_sequence(seq: str) -> str:
    return "".join(aa if aa in AA else "X" for aa in str(seq).upper())


def _normalize_rows(x: Any) -> Any:
    import numpy as np

    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _normalize_vector(x: Any) -> Any:
    import numpy as np

    norm = np.linalg.norm(x)
    return x / max(float(norm), 1e-12)


def _aa_frequency_from_records(records: List[Dict[str, Any]]) -> Dict[str, float]:
    counts = {aa: 0 for aa in AA}
    total = 0
    for rec in records:
        seq = str(rec.get("sequence_fragment") or "")
        for aa in seq:
            if aa in counts:
                counts[aa] += 1
                total += 1
    if total <= 0:
        return {}
    return {aa: counts[aa] / total for aa in AA}


def _class_bias_from_freq(freq: Dict[str, float]) -> List[str]:
    if not freq:
        return []
    classes = []
    aromatic = sum(freq.get(aa, 0.0) for aa in AROMATIC)
    polar = sum(freq.get(aa, 0.0) for aa in POLAR)
    charged = sum(freq.get(aa, 0.0) for aa in CHARGED)
    hydro = sum(freq.get(aa, 0.0) for aa in HYDROPHOBIC)
    if aromatic >= 0.12:
        classes.append("aromatic")
    if polar >= 0.25:
        classes.append("polar_uncharged")
    if charged >= 0.15:
        classes.append("contextual_charge")
    if hydro >= 0.40:
        classes.append("hydrophobic")
    return classes


def _as_external_prior_cache(data: Dict[str, Any]) -> Dict[str, Any]:
    """Accept native prior-cache files and Magneton-style summary files."""
    if any(k in data for k in ("nodes", "by_kind", "defaults")):
        return data

    classes = data.get("classes", {})
    if not isinstance(classes, dict):
        return {}

    binding = classes.get("Binding_site") or classes.get("Active_site") or {}
    domain = classes.get("Domain") or {}
    binding_freq = binding.get("aa_frequency", {}) if isinstance(binding, dict) else {}
    domain_freq = domain.get("aa_frequency", {}) if isinstance(domain, dict) else {}

    cache: Dict[str, Any] = {
        "metadata": {
            "provider_schema": "astevolve_external_prior_v1",
            "source_format": "magneton_substructure_summary",
            "note": "Automatically adapted from substructure_summary.json with conservative confidence.",
        },
        "defaults": {
            "priority_boost": 1.0,
            "confidence": 0.20,
        },
        "by_kind": {},
    }

    if binding_freq:
        cache["by_kind"]["cdr"] = {
            "priority_boost": 1.08,
            "confidence": 0.25,
            "aa_weights": binding_freq,
            "favored_residues": _top_residues(binding_freq, n=8),
            "source": "magneton:Binding_site",
        }
        cache["by_kind"]["epitope"] = {
            "priority_boost": 1.0,
            "confidence": 0.20,
            "aa_weights": binding_freq,
            "source": "magneton:Binding_site",
        }

    if domain_freq:
        cache["by_kind"]["framework"] = {
            "priority_boost": 0.95,
            "confidence": 0.15,
            "aa_weights": domain_freq,
            "source": "magneton:Domain",
        }

    return cache


def _as_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)]


def _unique_items(*values: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        for item in _as_list(value):
            if item not in seen:
                out.append(item)
                seen.add(item)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_manifest_file(manifest_path: Optional[Path], value: Any) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    if manifest_path is not None:
        return manifest_path.parent / path
    return path


def _matches_any(value: Any, allowed: List[str]) -> bool:
    if not allowed:
        return True
    return str(value or "") in set(allowed)


def _matches_keyword(value: Any, keywords: List[str]) -> bool:
    if not keywords:
        return True
    text = str(value or "").lower()
    return any(keyword.lower() in text for keyword in keywords)


@dataclass
class ExternalKnowledgeProvider:
    """Small interface seen by the inner loop.

    Implementations should return abstract priors, never raw copied sequences.
    """

    path: Optional[Path] = None
    data: Dict[str, Any] = field(default_factory=dict)
    embedding_manifest_path: Optional[Path] = None
    retrieval_top_k: int = 20
    retrieval_weight: float = 0.6
    device: str = "auto"
    max_length: int = 128
    _embedding_manifest: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _embeddings: Any = field(default=None, init=False, repr=False)
    _metadata: Optional[List[Dict[str, Any]]] = field(default=None, init=False, repr=False)
    _records: Optional[Dict[str, Dict[str, Any]]] = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _resolved_device: Optional[str] = field(default=None, init=False, repr=False)
    _retrieval_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    @property
    def enabled(self) -> bool:
        return bool(self.data)

    def get_node_prior(
        self,
        node_name: str,
        node_kind: str = "",
        chain_id: str = "",
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        defaults = self.data.get("defaults", {})
        by_kind = self.data.get("by_kind", {})
        nodes = self.data.get("nodes", {})

        kind_prior = {}
        if isinstance(by_kind, dict):
            kind_prior = by_kind.get(node_kind, {}) or by_kind.get(str(node_kind).lower(), {}) or {}

        node_prior = {}
        if isinstance(nodes, dict):
            for key in (node_name, node_name.lower(), node_name.casefold()):
                if isinstance(nodes.get(key), dict):
                    node_prior = nodes[key]
                    break

        return _merge_dicts(
            defaults if isinstance(defaults, dict) else {},
            kind_prior if isinstance(kind_prior, dict) else {},
            node_prior if isinstance(node_prior, dict) else {},
            {"node_name": node_name, "node_kind": node_kind, "chain_id": chain_id},
        )

    @property
    def retrieval_enabled(self) -> bool:
        return self.embedding_manifest_path is not None

    def _load_embedding_index(self) -> None:
        if not self.retrieval_enabled or self._embeddings is not None:
            return

        import numpy as np

        manifest = _read_mapping(self.embedding_manifest_path) if self.embedding_manifest_path else {}
        if not manifest:
            return

        embeddings_path = _resolve_manifest_file(self.embedding_manifest_path, manifest.get("embeddings", ""))
        metadata_path = _resolve_manifest_file(self.embedding_manifest_path, manifest.get("metadata", ""))
        records_path = _resolve_manifest_file(self.embedding_manifest_path, manifest.get("records", ""))
        if not embeddings_path.exists() or not metadata_path.exists() or not records_path.exists():
            return

        raw_embeddings = np.load(embeddings_path, mmap_mode="r").astype(np.float32)
        self._embeddings = _normalize_rows(raw_embeddings)
        self._metadata = list(_iter_jsonl(metadata_path))
        self._records = {rec["record_id"]: rec for rec in _iter_jsonl(records_path)}
        self._embedding_manifest = manifest

        if len(self._metadata) != self._embeddings.shape[0]:
            raise ValueError(
                f"Embedding metadata rows {len(self._metadata)} != embedding rows {self._embeddings.shape[0]}"
            )

    def _load_embedding_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._load_embedding_index()
        if not self._embedding_manifest:
            return

        model_name = self._embedding_manifest.get("model", "facebook/esm2_t6_8M_UR50D")
        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = device

    def embed_sequence(self, sequence: str) -> Optional[Any]:
        seq = _normalize_sequence(sequence)
        if not seq or len(seq) > self.max_length:
            return None

        self._load_embedding_model()
        if self._model is None or self._tokenizer is None or self._torch is None:
            return None

        with self._torch.no_grad():
            encoded = self._tokenizer(
                [seq],
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=True,
            )
            encoded = {key: value.to(self._resolved_device) for key, value in encoded.items()}
            out = self._model(**encoded)
            hidden = out.last_hidden_state.detach().cpu().numpy()
            residue_embeddings = hidden[0, 1 : 1 + len(seq), :]
            return residue_embeddings.mean(axis=0).astype("float32")

    def query_sequence(
        self,
        sequence: str,
        node_name: str = "",
        top_k: Optional[int] = None,
        class_filter: Any = "Antibody_CDR",
        substructure_filter: Any = None,
        match_node_name: Optional[bool] = None,
        name_keywords: Any = None,
    ) -> Dict[str, Any]:
        seq = _normalize_sequence(sequence)
        if not seq:
            return {"hits": [], "prior": {}}

        k = int(top_k or self.retrieval_top_k)
        class_filters = _as_list(class_filter)
        substructure_filters = _as_list(substructure_filter)
        keywords = _as_list(name_keywords)
        if match_node_name is None:
            match_node_name = class_filters == ["Antibody_CDR"]

        cache_key = (
            f"{node_name}|{','.join(class_filters)}|{','.join(substructure_filters)}|"
            f"{','.join(keywords)}|{bool(match_node_name)}|{k}|{seq}"
        )
        if cache_key in self._retrieval_cache:
            return self._retrieval_cache[cache_key]

        self._load_embedding_index()
        query_embedding = self.embed_sequence(seq)
        if query_embedding is None or self._embeddings is None or self._metadata is None:
            return {"hits": [], "prior": {}}

        import numpy as np

        query = _normalize_vector(query_embedding.astype(np.float32))
        scores = self._embeddings @ query
        order = np.argsort(-scores)

        hits = []
        wanted_node = str(node_name or "")
        for idx in order:
            idx = int(idx)
            row = self._metadata[idx]
            if not _matches_any(row.get("substructure_class"), class_filters):
                continue
            if substructure_filters and not (
                _matches_any(row.get("substructure_id"), substructure_filters)
                or _matches_any(row.get("match_id"), substructure_filters)
            ):
                continue
            if keywords and not _matches_keyword(row.get("substructure_name"), keywords):
                continue
            if match_node_name and wanted_node and row.get("substructure_id") != wanted_node:
                continue
            record = (self._records or {}).get(row["record_id"])
            if record and record.get("sequence_fragment") == seq:
                continue
            hits.append(
                {
                    "rank": len(hits) + 1,
                    "index": idx,
                    "score": float(scores[idx]),
                    "metadata": row,
                    "record": record,
                }
            )
            if len(hits) >= k:
                break

        prior = self._prior_from_hits(hits, node_name=node_name)
        result = {
            "query": {
                "sequence_length": len(seq),
                "node_name": node_name or None,
                "class_filter": class_filters or None,
                "substructure_filter": substructure_filters or None,
                "name_keywords": keywords or None,
                "match_node_name": bool(match_node_name),
                "top_k": k,
            },
            "hits": hits,
            "prior": prior,
        }
        self._retrieval_cache[cache_key] = result
        return result

    def _retrieval_filter_for_node(self, node_name: str, node_kind: str) -> Dict[str, Any]:
        retrieval = self.data.get("retrieval", {}) if isinstance(self.data, dict) else {}
        if not isinstance(retrieval, dict):
            retrieval = {}

        filters: Dict[str, Any] = {}
        kind_filters = retrieval.get("node_kind_filters", {})
        if isinstance(kind_filters, dict):
            filters = _merge_dicts(filters, kind_filters.get(str(node_kind), {}) or {})
            filters = _merge_dicts(filters, kind_filters.get(str(node_kind).lower(), {}) or {})

        node_filters = retrieval.get("node_filters", {})
        if isinstance(node_filters, dict):
            for key in (node_name, node_name.lower(), node_name.casefold()):
                filters = _merge_dicts(filters, node_filters.get(key, {}) or {})

        if filters:
            return filters

        if node_kind == "cdr":
            return {"class_filter": "Antibody_CDR", "match_node_name": True}
        return {}

    def get_sequence_prior(
        self,
        node_name: str,
        node_kind: str,
        chain_id: str,
        sequence: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        static_prior = self.get_node_prior(node_name, node_kind, chain_id)
        if not self.retrieval_enabled:
            return static_prior

        retrieval_filter = self._retrieval_filter_for_node(node_name, node_kind)
        if not retrieval_filter:
            return static_prior

        retrieval = self.query_sequence(
            sequence=sequence,
            node_name=node_name,
            top_k=top_k,
            class_filter=retrieval_filter.get("class_filter", retrieval_filter.get("class_filters")),
            substructure_filter=retrieval_filter.get(
                "substructure_filter",
                retrieval_filter.get("substructure_filters"),
            ),
            match_node_name=retrieval_filter.get("match_node_name"),
            name_keywords=retrieval_filter.get("name_keywords"),
        )
        dynamic_prior = retrieval.get("prior", {})
        if not dynamic_prior:
            return static_prior

        merged = _merge_dicts(static_prior, dynamic_prior)
        for key in (
            "favored_residues",
            "favored_residue_classes",
            "disfavored_residues",
            "disfavored_residue_classes",
        ):
            merged[key] = _unique_items(static_prior.get(key), dynamic_prior.get(key))
        merged["confidence"] = max(
            _safe_float(static_prior.get("confidence"), 0.0),
            _safe_float(dynamic_prior.get("confidence"), 0.0),
        )
        if static_prior.get("source") and dynamic_prior.get("source"):
            merged["source"] = f"{static_prior.get('source')}+{dynamic_prior.get('source')}"
        merged["retrieval"] = {
            "enabled": True,
            "top_k": len(retrieval.get("hits", [])),
            "mean_similarity": dynamic_prior.get("mean_similarity"),
            "top_hits": [
                {
                    "score": hit.get("score"),
                    "record_id": hit.get("metadata", {}).get("record_id"),
                    "substructure_id": hit.get("metadata", {}).get("substructure_id"),
                    "sequence_length": len(str((hit.get("record") or {}).get("sequence_fragment") or "")),
                }
                for hit in retrieval.get("hits", [])[:5]
            ],
        }
        return merged

    def _prior_from_hits(self, hits: List[Dict[str, Any]], node_name: str = "") -> Dict[str, Any]:
        records = [hit.get("record") for hit in hits if isinstance(hit.get("record"), dict)]
        if not records:
            return {}

        freq = _aa_frequency_from_records(records)
        scores = [float(hit.get("score", 0.0)) for hit in hits]
        retrieval_strength = max(0.0, min(1.0, float(self.retrieval_weight)))
        return {
            "priority_boost": 1.0,
            "confidence": min(0.85, 0.35 + 0.03 * len(records)) * retrieval_strength,
            "aa_weights": freq,
            "favored_residues": _top_residues(freq, n=10),
            "favored_residue_classes": _class_bias_from_freq(freq),
            "disfavored_residues": ["C"],
            "source": f"embedding:{node_name or 'Antibody_CDR'}",
            "mean_similarity": float(sum(scores) / max(1, len(scores))),
            "support_count": len(records),
        }

    def describe(self) -> Dict[str, Any]:
        metadata = self.data.get("metadata", {}) if isinstance(self.data, dict) else {}
        nodes = self.data.get("nodes", {}) if isinstance(self.data, dict) else {}
        by_kind = self.data.get("by_kind", {}) if isinstance(self.data, dict) else {}
        retrieval = self.data.get("retrieval", {}) if isinstance(self.data, dict) else {}
        return {
            "enabled": self.enabled,
            "path": str(self.path) if self.path else None,
            "metadata": metadata if isinstance(metadata, dict) else {},
            "node_count": len(nodes) if isinstance(nodes, dict) else 0,
            "kind_count": len(by_kind) if isinstance(by_kind, dict) else 0,
            "retrieval_config": retrieval if isinstance(retrieval, dict) else {},
            "embedding_retrieval": {
                "enabled": self.retrieval_enabled,
                "manifest": str(self.embedding_manifest_path) if self.embedding_manifest_path else None,
                "top_k": self.retrieval_top_k,
                "weight": self.retrieval_weight,
                "device": self.device,
                "loaded": self._embeddings is not None,
                "shape": list(self._embeddings.shape) if self._embeddings is not None else None,
                "model": self._embedding_manifest.get("model") if self._embedding_manifest else None,
            },
        }


def load_external_knowledge_provider(
    path: Optional[str],
    embedding_manifest: Optional[str] = None,
    retrieval_top_k: int = 20,
    retrieval_weight: float = 0.6,
    device: str = "auto",
    max_length: int = 128,
) -> ExternalKnowledgeProvider:
    if not path:
        return ExternalKnowledgeProvider(
            embedding_manifest_path=Path(embedding_manifest) if embedding_manifest else None,
            retrieval_top_k=int(retrieval_top_k),
            retrieval_weight=float(retrieval_weight),
            device=device,
            max_length=int(max_length),
        )

    p = Path(path)
    data = _read_mapping(p)
    return ExternalKnowledgeProvider(
        path=p,
        data=_as_external_prior_cache(data),
        embedding_manifest_path=Path(embedding_manifest) if embedding_manifest else None,
        retrieval_top_k=int(retrieval_top_k),
        retrieval_weight=float(retrieval_weight),
        device=device,
        max_length=int(max_length),
    )
