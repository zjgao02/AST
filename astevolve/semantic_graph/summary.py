from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Set


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _node_id(item: Any, fallback: str) -> str:
    if isinstance(item, dict):
        return str(item.get("id") or item.get("name") or fallback)
    return str(item or fallback)


def _normalize_node_map(raw: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(raw, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for key, value in raw.items():
            node = dict(value) if isinstance(value, dict) else {"value": value}
            node.setdefault("id", str(key))
            out[str(key)] = node
        return out
    if isinstance(raw, list):
        out = {}
        for index, value in enumerate(raw):
            node = dict(value) if isinstance(value, dict) else {"value": value}
            key = _node_id(node, f"node_{index}")
            node.setdefault("id", key)
            out[key] = node
        return out
    return {}


def _normalize_edges(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        edges: List[Dict[str, Any]] = []
        for key, value in raw.items():
            edge = dict(value) if isinstance(value, dict) else {"value": value}
            edge.setdefault("id", str(key))
            edges.append(edge)
        return edges
    if isinstance(raw, list):
        out = []
        for index, value in enumerate(raw):
            edge = dict(value) if isinstance(value, dict) else {"value": value}
            edge.setdefault("id", edge.get("name") or f"edge_{index}")
            out.append(edge)
        return out
    return []


def _listify(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, dict)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _segment_summary(compiled: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not compiled:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for seg in compiled.get("segments", []) or []:
        name = str(getattr(seg, "name", "") or "")
        if not name:
            continue
        out[name] = {
            "chain_id": str(getattr(seg, "chain_id", "")),
            "kind": str(getattr(seg, "kind", "")),
            "spans": getattr(seg, "spans", None),
            "start": getattr(seg, "start", None),
            "end": getattr(seg, "end", None),
            "total_length": getattr(seg, "total_length", None),
        }
    return out


def _constraint_sets(state: Mapping[str, Any]) -> tuple[Set[str], Set[str]]:
    constraints = _as_dict(state.get("design_constraints"))
    mutable = set(_listify(constraints.get("mutable_nodes")))
    frozen = set(_listify(constraints.get("frozen_nodes")))
    policy = _as_dict(state.get("mutation_policy"))
    mutable.update(_listify(policy.get("always_open_segments")))
    mutable.update(_listify(policy.get("conditionally_open_segments")))
    frozen.update(_listify(policy.get("generally_frozen")))
    return mutable, frozen


def _node_confidence(node_plddt: Any, node_id: str) -> Dict[str, Any]:
    if not isinstance(node_plddt, dict):
        return {}
    value = node_plddt.get(node_id)
    if isinstance(value, dict):
        return {k: value.get(k) for k in ("mean", "min", "max", "count") if k in value}
    if isinstance(value, (int, float)):
        return {"mean": float(value)}
    return {}


def _mapping_table(graph: Mapping[str, Any], functional_nodes: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for fn_id, node in functional_nodes.items():
        values = node.get("maps_to") or node.get("structural_nodes") or node.get("realization")
        if values:
            mapping[fn_id] = _listify(values)
    raw = graph.get("mappings") or graph.get("functional_to_structural")
    if isinstance(raw, dict):
        for key, value in raw.items():
            mapping[str(key)] = _listify(value)
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            fn = item.get("functional_node") or item.get("source") or item.get("from") or item.get("id")
            structural = item.get("structural_nodes") or item.get("targets") or item.get("to") or item.get("maps_to")
            if fn:
                mapping[str(fn)] = _listify(structural)
    return mapping


def build_semantic_graph_summary(
    state: Optional[Mapping[str, Any]],
    compiled: Optional[Mapping[str, Any]] = None,
    node_plddt: Any = None,
) -> Dict[str, Any]:
    """Return a compact, LLM-friendly Protein Semantic Graph summary."""
    state = state or {}
    graph = _as_dict(state.get("semantic_graph"))
    if not graph:
        return {"enabled": False, "warnings": ["design_state.semantic_graph is absent"]}

    structural_graph = _as_dict(graph.get("structural_graph"))
    functional_graph = _as_dict(graph.get("functional_graph"))
    structural_nodes = _normalize_node_map(structural_graph.get("nodes"))
    functional_nodes = _normalize_node_map(functional_graph.get("nodes"))
    structural_edges = _normalize_edges(structural_graph.get("edges"))
    functional_edges = _normalize_edges(functional_graph.get("edges"))
    segment_info = _segment_summary(compiled)
    mutable, frozen = _constraint_sets(state)
    warnings: List[str] = []

    for node_id, info in segment_info.items():
        structural_nodes.setdefault(node_id, {"id": node_id})
        structural_nodes[node_id].setdefault("kind", info.get("kind"))
        structural_nodes[node_id].setdefault("spans", info.get("spans"))
        structural_nodes[node_id].setdefault("chain_id", info.get("chain_id"))
        structural_nodes[node_id].setdefault("start", info.get("start"))
        structural_nodes[node_id].setdefault("end", info.get("end"))
        structural_nodes[node_id].setdefault("total_length", info.get("total_length"))

    for node_id, node in structural_nodes.items():
        node.setdefault("id", node_id)
        node["editable"] = node_id in mutable and node_id not in frozen
        node["frozen"] = node_id in frozen
        confidence = _node_confidence(node_plddt, node_id)
        if confidence:
            node["confidence"] = confidence

    mapping = _mapping_table(graph, functional_nodes)
    structural_ids = set(structural_nodes)
    functional_ids = set(functional_nodes)
    for fn_id, mapped in mapping.items():
        functional_nodes.setdefault(fn_id, {"id": fn_id})
        functional_nodes[fn_id]["maps_to"] = mapped
        missing = [name for name in mapped if name not in structural_ids]
        if missing:
            warnings.append(f"functional node {fn_id} maps to unknown structural nodes: {missing}")

    for edge in functional_edges:
        src = str(edge.get("source") or edge.get("from") or "")
        dst = str(edge.get("target") or edge.get("to") or "")
        if src and src not in functional_ids and src not in mapping:
            warnings.append(f"functional edge {edge.get('id')} has unknown source {src}")
        if dst and dst not in functional_ids and dst not in mapping:
            warnings.append(f"functional edge {edge.get('id')} has unknown target {dst}")
        edge.setdefault("edge_semantics", "functional_coupling")

    return {
        "enabled": True,
        "schema_version": graph.get("schema_version", "protein_semantic_graph_v1"),
        "structural_node_count": len(structural_nodes),
        "functional_node_count": len(functional_nodes),
        "structural_edge_count": len(structural_edges),
        "functional_edge_count": len(functional_edges),
        "structural_nodes": structural_nodes,
        "functional_nodes": functional_nodes,
        "functional_to_structural": mapping,
        "structural_edges": structural_edges,
        "functional_edges": functional_edges,
        "outer_loop_contract": graph.get("outer_loop_contract", {}),
        "warnings": sorted(set(warnings)),
    }
