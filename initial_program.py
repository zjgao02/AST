# EVOLVE-BLOCK-START
import numpy as np
from protein_lang import Node, Blueprint


def _hotspot_window(seq: str, motif: str = "GLGFNI", left: int = 3, right: int = 6):
    idx = seq.index(motif)
    start = max(0, idx - left)
    end = min(len(seq), idx + len(motif) + right)
    return start, end


def propose_blueprint_and_config():
    # ---- Fixed Targets ----
    target_A = (
        "GSPEFLGEEDIPREPRRIVIHRGSTGLGFNIIGGEDGEGIFISFILAGGPADLSGELRKGDQILSVNGVDLRNASHEQAAIALKNAGQTVTIIAQYKPEEYSRFEANSRVNSSGRIVTN"
    )
    target_B = (
        "EDIPREPRRIVIHRGSTGLGFNIVGGEDGEGIFISFILAGGPADLSGELRKGDQILSVNGVDLRNASHEQAAIALKNAGQTVTIIAQYKPEEYSRFEAK"
    )

    # ---- Hotspot windows ----
    A_start, A_end = _hotspot_window(target_A, "GLGFNI", left=3, right=6)
    B_start, B_end = _hotspot_window(target_B, "GLGFNI", left=3, right=6)

    # ---- Target chains with epitope segment ----
    # 这里仍使用 sequential 模式（叶节点无 residue_spans），完全向后兼容
    def make_target_chain(seq: str, chain_id: str, ep_start: int, ep_end: int):
        children = []
        if ep_start > 0:
            children.append(Node(kind="domain", name="nterm", length=ep_start))
        children.append(Node(kind="epitope", name="epitope", length=ep_end - ep_start))
        if ep_end < len(seq):
            children.append(Node(kind="domain", name="cterm", length=len(seq) - ep_end))

        return Node(
            kind="chain", name=f"target_{chain_id}", props={"chain_id": chain_id},
            children=children
        )

    chain_TA = make_target_chain(target_A, "TA", A_start, A_end)
    chain_TB = make_target_chain(target_B, "TB", B_start, B_end)

    # ---- Binder chain (A-selective only) ----
    chain_BB = Node(
        kind="chain", name="binder_BB", props={"chain_id": "BB"},
        children=[Node(kind="iface", name="iface", length=5)]
    )

    bp = Blueprint(
        root=Node(
            kind="complex", name="complex",
            children=[chain_TA, chain_TB, chain_BB]
        )
    )

    # ---- Masks ----
    masks = {
        "TA": [False] * len(target_A),
        "TB": [False] * len(target_B),
        "BB": [True] * 5,
    }

    # ---- Constraints ----
    no_cys = set("ADEFGHIKLMNPQRSTVWY")

    constraint_specs = [
        {"kind": "alphabet", "weight": 1.0, "params": {"allowed": no_cys}},
        {"kind": "fixed_chain_sequence", "weight": 1.0, "params": {"chain_id": "TA", "sequence": target_A}},
        {"kind": "fixed_chain_sequence", "weight": 1.0, "params": {"chain_id": "TB", "sequence": target_B}},
        {"kind": "max_run", "weight": 1.0, "params": {
            "aa_set": "AILMFWVY",
            "max_run": 2,
            "segment_filter": {"chain_id": "BB", "name": "iface"}
        }},
        {"kind": "max_run", "weight": 1.0, "params": {
            "aa_set": "KRDE",
            "max_run": 2,
            "segment_filter": {"chain_id": "BB", "name": "iface"}
        }},
        {"kind": "segment_composition", "weight": 1.5, "params": {
            "aa_set": "DE",
            "min_frac": 0.20,
            "max_frac": 0.40,
            "segment_filter": {"chain_id": "BB", "name": "iface"}
        }},
        {"kind": "segment_composition", "weight": 1.0, "params": {
            "aa_set": "AILMFWVY",
            "min_frac": 0.10,
            "max_frac": 0.30,
            "segment_filter": {"chain_id": "BB", "name": "iface"}
        }},
        {"kind": "chai_plddt_delta", "weight": 10.0, "stage": "chai", "params": {
            "chains_A": ["TA", "BB"],
            "chains_B": ["TB", "BB"],
            "metric": "iptm",               # ← 改为 iptm
            "direction": "A_gt_B",           # ← 希望 A 的 iptm 更高
            "delta_threshold": 0.15,         # ← iptm 范围 0-1，阈值要调小
            "scale": 10,                     # ← 可根据需要调整
            "device": None
        }},
    ]

    # ---- Search config ----
    sa_config = {
        "iterations": 1000,
        "init_temp": 2.0,
        "cooling": 0.995,
        "mutation_rate": 0.20,
        "resample_segment_prob": 0.10,

        "progen_weight": 1.0,
        "progen_chains": ["BB"],
        "progen_reduce": "length_weighted",

        "chai1_enabled": True,
        "chai1_top_frac": 0.02,
        "chai1_min_candidates": 1,
        "chai1_max_candidates": 2,
        "chai1_num_trunk_recycles": 3,
        "chai1_num_diffn_timesteps": 100,

        "mutation_ops": {
            "point": 0.7,
            "block": 0.15,
            "segment_resample": 0.10,
            "swap": 0.05,
        },
        "history_size": 30,
    }

    score_config = {
        "weight_fast": 1,
        "weight_plddt": 5,
        "plddt_scale": 1.0,
        "fast_loss_nonneg": True,
    }

    templates = {
        "TA": target_A,
        "TB": target_B,
        "BB": "AAAAV",
    }
    fixed_residues = {}

    return bp, constraint_specs, sa_config, masks, templates, fixed_residues, score_config
# EVOLVE-BLOCK-END


# ---------------------------------------------------------------------------
# 不连续域示例
# ---------------------------------------------------------------------------
#
def propose_discontinuous_example():
    """
    示例：一条 50 残基的链，包含一个不连续结构域和一个插入域。

    序列布局:
      [0, 15)  -> domain_A part 1
      [15, 35) -> domain_B (插入)
      [35, 50) -> domain_A part 2

    结构上 domain_A 是一个整体，由 [0,15) 和 [35,50) 组成。
    """
    chain = Node(
        kind="chain", name="my_chain", length=50,
        props={"chain_id": "X"},
        children=[
            Node(kind="domain", name="domA",
                 residue_spans=[(0, 15), (35, 50)]),   # 不连续！
            Node(kind="domain", name="domB",
                 residue_spans=[(15, 35)]),
        ],
    )
    bp = Blueprint(root=chain)
    compiled = bp.compile()

    # 验证
    for seg in compiled["segments"]:
        print(f"{seg.name}: spans={seg.spans}, total_length={seg.total_length}, "
              f"contiguous={seg.is_contiguous}")
    # 输出:
    # domA: spans=[(0, 15), (35, 50)], total_length=30, contiguous=False
    # domB: spans=[(15, 35)], total_length=20, contiguous=True

    # 测试 extract
    test_seq = "A" * 15 + "B" * 20 + "C" * 15
    for seg in compiled["segments"]:
        print(f"{seg.name}: '{seg.extract(test_seq)}'")
    # domA: 'AAAAAAAAAAAAAAACCCCCCCCCCCCCCC'  (15 A + 15 C)
    # domB: 'BBBBBBBBBBBBBBBBBBBB'            (20 B)

    return bp, compiled


# ---- 固定部分 ----
import numpy as np
from inner_opt import optimize_multichain, SAConfig


def run_search(seed: int | None = None):
    bp, constraint_specs, sa_cfg, masks, templates, fixed_residues, score_cfg = (
        propose_blueprint_and_config()
    )
    # bp, constraint_specs, sa_cfg, masks, templates, fixed_residues, score_cfg = (
    #     propose_discontinuous_example()
    # )
    compiled = bp.compile()
    masks_np = {k: np.array(v, dtype=bool) for k, v in masks.items()}
    cfg = SAConfig(**sa_cfg, seed=seed)

    out = optimize_multichain(
        compiled,
        constraint_specs,
        cfg,
        masks=masks_np,
        template_seqs=templates,
        fixed_residues=fixed_residues,
    )
    out["chain_lengths"] = compiled["chain_lengths"]
    # ---- 更新：序列化 segment 时包含完整 spans 信息 ----
    out["segments"] = [
        {
            "chain_id": s.chain_id,
            "kind": s.kind,
            "name": s.name,
            "spans": s.spans,
            "total_length": s.total_length,
            "is_contiguous": s.is_contiguous,
            "start": s.start,   # backward compat
            "end": s.end,       # backward compat
            "props": s.props,
        }
        for s in compiled["segments"]
    ]
    out["blueprint_summary"] = {
        "chain_order": compiled["chain_order"],
        "chain_lengths": compiled["chain_lengths"],
    }
    out["score_config"] = score_cfg
    return out


if __name__ == "__main__":
    r = run_search(seed=0)
    print("Fast loss:", r["fast_loss"])