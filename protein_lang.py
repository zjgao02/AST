from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterable, Tuple


@dataclass
class ConstraintSpec:
    kind: str
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    kind: str
    name: str = ""
    length: Optional[int] = None
    repeat: int = 1
    children: List["Node"] = field(default_factory=list)
    constraints: List[ConstraintSpec] = field(default_factory=list)
    props: Dict[str, Any] = field(default_factory=dict)
    # ---- 新增：显式指定残基区间（可不连续） ----
    residue_spans: Optional[List[Tuple[int, int]]] = None

    def iter_nodes(self) -> Iterable["Node"]:
        yield self
        for c in self.children:
            yield from c.iter_nodes()

    def has_explicit_spans(self) -> bool:
        """检查自身或任何后代是否使用了显式 residue_spans。"""
        if self.residue_spans is not None:
            return True
        return any(c.has_explicit_spans() for c in self.children)


@dataclass
class Segment:
    """
    一个 segment 可以覆盖多段不连续的残基区间。
    spans: [(start, end), ...] 半开区间。
    backward-compat: .start / .end 属性仍可用（返回第一段起点和最后一段终点）。
    """
    kind: str
    name: str
    chain_id: str
    spans: List[Tuple[int, int]]
    props: Dict[str, Any] = field(default_factory=dict)

    # ---- backward-compat 属性 ----
    @property
    def start(self) -> int:
        return self.spans[0][0] if self.spans else 0

    @property
    def end(self) -> int:
        return self.spans[-1][1] if self.spans else 0

    # ---- 新增核心方法 ----
    def indices(self) -> List[int]:
        """返回该 segment 覆盖的所有残基索引（有序）。"""
        idx: List[int] = []
        for s, e in self.spans:
            idx.extend(range(s, e))
        return idx

    def extract(self, seq: str) -> str:
        """从序列中提取该 segment 对应的子序列。"""
        return "".join(seq[i] for i in self.indices())

    def write_into(self, seq_list: List[str], fragment: str) -> None:
        """将 fragment 写回 seq_list 对应的索引位置。"""
        idxs = self.indices()
        if len(fragment) != len(idxs):
            raise ValueError(
                f"Fragment length {len(fragment)} != segment index count {len(idxs)}"
            )
        for i, idx in enumerate(idxs):
            seq_list[idx] = fragment[i]

    @property
    def total_length(self) -> int:
        return sum(e - s for s, e in self.spans)

    @property
    def is_contiguous(self) -> bool:
        if len(self.spans) <= 1:
            return True
        for i in range(len(self.spans) - 1):
            if self.spans[i][1] != self.spans[i + 1][0]:
                return False
        return True


@dataclass
class Blueprint:
    root: Node
    constraints: List[ConstraintSpec] = field(default_factory=list)

    def compile(self) -> Dict[str, Any]:
        """
        支持两种布局模式（per chain 自动判断）：
        1. sequential — 原有 cursor 递增模式（所有叶节点无 residue_spans）
        2. explicit  — 叶节点通过 residue_spans 显式声明覆盖区间
        """
        segments: List[Segment] = []
        chain_lengths: Dict[str, int] = {}
        chain_order: List[str] = []

        def emit_chain(chain_node: Node):
            if chain_node.kind != "chain":
                raise ValueError("Children of complex must be chain nodes")
            chain_id = chain_node.props.get("chain_id", chain_node.name)
            if not chain_id:
                raise ValueError("chain_id is required")

            chain_order.append(chain_id)

            # 判断该链使用哪种模式
            uses_explicit = any(
                n.residue_spans is not None
                for n in chain_node.iter_nodes()
                if n is not chain_node
            )

            if uses_explicit:
                new_segs = _emit_chain_explicit(chain_node, chain_id)
                segments.extend(new_segs)
                # chain length: 优先用 chain_node.length，否则从 spans 推断
                if chain_node.length is not None:
                    chain_lengths[chain_id] = chain_node.length
                else:
                    max_end = 0
                    for seg in new_segs:
                        for _, e in seg.spans:
                            max_end = max(max_end, e)
                    chain_lengths[chain_id] = max_end
            else:
                new_segs, cursor = _emit_chain_sequential(chain_node, chain_id)
                segments.extend(new_segs)
                chain_lengths[chain_id] = cursor

        if self.root.kind == "chain":
            emit_chain(self.root)
        elif self.root.kind == "complex":
            for ch in self.root.children:
                emit_chain(ch)
        else:
            raise ValueError("Blueprint.root.kind must be 'chain' or 'complex'")

        return {
            "segments": segments,
            "chain_lengths": chain_lengths,
            "chain_order": chain_order,
            "blueprint": self,
        }


# ---------------------------------------------------------------------------
# 内部编译函数
# ---------------------------------------------------------------------------

def _emit_chain_sequential(
    chain_node: Node, chain_id: str
) -> Tuple[List[Segment], int]:
    """原有 cursor 递增布局。返回 (segments, chain_length)。"""
    segments: List[Segment] = []
    cursor = 0

    def emit_subtree(node: Node, repeat_group: Optional[str] = None):
        nonlocal cursor
        for r in range(node.repeat):
            if node.children:
                for ch in node.children:
                    emit_subtree(
                        ch,
                        repeat_group=repeat_group or node.props.get("repeat_group"),
                    )
            else:
                if node.length is None or node.length <= 0:
                    raise ValueError(f"Leaf node must have positive length: {node}")
                seg = Segment(
                    kind=node.kind,
                    name=node.name or node.kind,
                    chain_id=chain_id,
                    spans=[(cursor, cursor + node.length)],
                    props={
                        **node.props,
                        "repeat_group": repeat_group,
                        "repeat_index": r,
                    },
                )
                segments.append(seg)
                cursor += node.length

    for ch in chain_node.children:
        emit_subtree(ch)

    return segments, cursor


def _emit_chain_explicit(
    chain_node: Node, chain_id: str
) -> List[Segment]:
    """
    显式 spans 布局：每个叶节点必须有 residue_spans。
    父节点可以设置 residue_spans 作为"继承默认"，仅当子节点自身
    没有 residue_spans 时使用。
    """
    segments: List[Segment] = []

    def emit_subtree(
        node: Node,
        inherited_spans: Optional[List[Tuple[int, int]]] = None,
        repeat_group: Optional[str] = None,
    ):
        for r in range(node.repeat):
            # 本节点的 spans：优先自身，其次继承
            effective_spans = (
                node.residue_spans if node.residue_spans is not None else inherited_spans
            )

            if node.children:
                for ch in node.children:
                    emit_subtree(
                        ch,
                        inherited_spans=effective_spans,
                        repeat_group=repeat_group or node.props.get("repeat_group"),
                    )
            else:
                # 叶节点
                if effective_spans is None:
                    raise ValueError(
                        f"In explicit mode, leaf node must have residue_spans "
                        f"(directly or inherited): {node}"
                    )
                seg = Segment(
                    kind=node.kind,
                    name=node.name or node.kind,
                    chain_id=chain_id,
                    spans=list(effective_spans),
                    props={
                        **node.props,
                        "repeat_group": repeat_group,
                        "repeat_index": r,
                    },
                )
                segments.append(seg)

    for ch in chain_node.children:
        emit_subtree(ch)

    # 校验：同一链内不允许 index 重叠
    all_indices: set = set()
    for seg in segments:
        for idx in seg.indices():
            if idx in all_indices:
                raise ValueError(
                    f"Overlapping spans in chain {chain_id}: index {idx} "
                    f"appears in multiple segments"
                )
            all_indices.add(idx)

    return segments