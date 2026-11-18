# -*- coding: utf-8 -*-
"""四类电容形状库：直接用 qiskit-metal 官方组件实例化。

A. 双岛电容   → TransmonPocket（pad_width/height/gap 变化 + pocket）
B. 单岛多臂   → TransmonCross / TransmonStar（臂宽/长/数变化）
C. 同心圆     → TransmonConcentric（内外径/间隙变化）
D. 交指电容   → TransmonInterdigitated（手指宽/高/间距/梳齿变化）

每类 N 个参数采样点 → 直接实例化为 Metal 组件 → Q3D 仿真。
不自行造多边形——用官方组件保证几何正确性（尤其 D 类交指方向）。
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.configs import frozen

# ── 各类的参数空间 ─────────────────────────────────────────

def gen_class_A(n, rng):
    """双岛电容：pad 尺寸与间隙变化。"""
    return [dict(
        pad_width=f"{rng.uniform(400, 600):.0f}um",
        pad_height=f"{rng.uniform(80, 180):.0f}um",
        pad_gap=f"{rng.uniform(20, 40):.0f}um",
        pocket_width="800um", pocket_height="800um",
        connection_pads=dict(
            readout=dict(loc_W=+1, loc_H=-1, pad_width="200um"),
            bus1=dict(loc_W=-1, loc_H=+1, pad_height="30um"),
            bus2=dict(loc_W=-1, loc_H=-1, pad_height="50um"),
        ),
    ) for _ in range(n)]


def gen_class_B_cross(n, rng):
    """Xmon：十字臂宽/长变化。"""
    return [dict(
        cross_width=f"{rng.uniform(40, 80):.0f}um",
        cross_length=f"{rng.uniform(150, 300):.0f}um",
        cross_gap=f"{rng.uniform(20, 50):.0f}um",
        connection_pads=dict(
            readout=dict(conn_type="1", claw_length="30um"),
            bus1=dict(conn_type="2", claw_length="30um"),
            bus2=dict(conn_type="3", claw_length="30um"),
        ),
    ) for _ in range(n)]


def gen_class_B_star(n, rng):
    """星形：臂数/宽/长变化。"""
    return [dict(
        num_points=int(rng.integers(3, 8)),
        arm_width=f"{rng.uniform(30, 60):.0f}um",
        arm_length=f"{rng.uniform(150, 250):.0f}um",
        arm_gap=f"{rng.uniform(20, 50):.0f}um",
        junction_arm_index=0,
        connection_pads=dict(
            readout=dict(conn_type="1", claw_length="30um"),
        ),
    ) for _ in range(n)]


def gen_class_C(n, rng):
    """同心圆：内外径/间隙变化。"""
    return [dict(
        rad_o=f"{rng.uniform(180, 250):.0f}um",
        rad_i=f"{rng.uniform(100, 160):.0f}um",
        gap=f"{rng.uniform(20, 50):.0f}um",
        jj_w="20um",
    ) for _ in range(n)]


def gen_class_D_corrected(n, rng):
    """交指电容（修正版 CorrectIDC）：手指在结区两侧交错。

    junction_gap 自动满足 jg ≥ 2*fh + 2*fg 约束。
    """
    from .correct_idc import gen_idc_variant
    return gen_idc_variant(n, rng)


# ── 汇总 ─────────────────────────────────────────────────

FAMILIES = {
    "A_dual_pad":   ("TransmonPocket",         gen_class_A),
    "B_cross":      ("TransmonCross",           gen_class_B_cross),
    "B_star":       ("TransmonStar",            gen_class_B_star),
    "C_concentric": ("TransmonConcentric",      gen_class_C),
    "D_interdigit": ("CorrectIDC",               gen_class_D_corrected),
}


def make_library(n_per_family=8, seed=20261026):
    """生成全部五族的参数列表。"""
    lib = {}
    import zlib
    for name, (comp_name, gen) in FAMILIES.items():
        # hash(name) 按进程加盐导致跨进程变体不可复现，改用稳定摘要
        rng = np.random.default_rng(seed + zlib.crc32(name.encode("utf-8")) % 10000)
        lib[name] = dict(component=comp_name, variants=gen(n_per_family, rng))
    return lib


def build_design_for(variant, comp_name, name="Q1"):
    """实例化一个 Metal 组件。"""
    from qiskit_metal import designs

    mod_name = {
        "TransmonPocket": "transmon_pocket",
        "TransmonCross": "transmon_cross",
        "TransmonStar": "transmon_star",
        "TransmonConcentric": "transmon_concentric",
        "TransmonInterdigitated": "Transmon_Interdigitated",
        "CorrectIDC": "correct_idc",
        "NovelTwoPadTransmon": "novel_structures",
    }[comp_name]

    if mod_name in ("correct_idc", "novel_structures"):
        mod = __import__(f"robust_design_lab.geometry.{mod_name}",
                         fromlist=[comp_name])
    else:
        mod = __import__(f"qiskit_metal.qlibrary.qubits.{mod_name}",
                         fromlist=[comp_name])
    cls = getattr(mod, comp_name)

    design = designs.design_planar.DesignPlanar()
    design.overwrite_enabled = True
    cls(design, name, options=variant)
    design.rebuild()
    return design
