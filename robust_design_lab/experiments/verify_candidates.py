# -*- coding: utf-8 -*-
"""第二步：四对象复验——修正同步地刻蚀下的结构优势判定。

对象与来源（docs/plans/new_structures_next_steps_2026-11-02.md 第二步）：
- vgap_vlm：变间隙 v1 VLM 终选（v1 旧固定地 D=0.00245）
- annulus_cma：环形 v1 CMA 终选（v1 旧固定地 D=0.00216）
- ellipse_idc_ref：资格基准 q_ellipse_idc 默认参数（资格旧口径 D=0.0157）
- native_ref：官方 TransmonPocket 无连接垫（纯形状比较；组件参数平移
  近似刻蚀——每边 δ + 口袋同步四边外退，与 buffer 模型同表时标注
  模型差异；带读出耦合的器件比较引用 step2_q3d_baseline 既有记录）

两种工艺假设同表报告（不混为一个机制结论）：
- 表 A：修正同步地（本驱动新解；独立名义口袋四边各退 δ）
- 表 B：旧固定地近似（引用 v1/资格记录，口袋未随刻蚀移动）

附加检查：
- C_Σ 变化分解：互电容 C12 vs 对地串联项（机制归因，独立重算并与
  run_q3d 汇总值对质，差 >1e-9 fF 即报错）
- 代表对象加密档（max_passes=12）三场景复算——可分辨性由逐轮数据
  判断，不以"加密本身"当通过

结果目录：results/verify_candidates_v2_2026-11-03/（逐对象落盘可续跑，
--only 选择对象，已完成对象自动跳过）。
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance

OUT = ROOT / "results" / "verify_candidates_v2_2026-11-03"
OUT.mkdir(parents=True, exist_ok=True)

ETCH_DELTA_UM = 0.5
EJ_GHZ, EC_GHZ_PER_F = 13.28, 19.36
CJ_FF = 2.0
REFINE_PASSES = 12  # 加密档（基准档 6 与历史口径一致）

OBJECTS = {
    "vgap_vlm": dict(
        kind="novel", family="m_variable_gap",
        kw=dict(H=480.0, depth=290.0, g0=32.0, g_edge=88.0),
        source="v1 vlm 终选（mainline_compare_2026-11-02）",
        legacy_fixed_pocket_D=0.002449),
    "annulus_cma": dict(
        kind="novel", family="q_annulus",
        kw=dict(r_out=253.9, r_in=121.5, gap=33.8),
        source="v1 cma_es 终选（终选参数原为字符串，已转 float）",
        legacy_fixed_pocket_D=0.002160),
    "ellipse_idc_ref": dict(
        kind="novel", family="q_ellipse_idc", kw={},
        source="资格基准默认参数（mainline_qualification_2026-11-01）",
        legacy_fixed_pocket_D=0.015676),
    "native_ref": dict(
        kind="native",
        kw=dict(pad_width="560um", pad_height="176um", pad_gap="34um",
                pocket_width="860um", pocket_height="690um"),
        source="官方 TransmonPocket 无连接垫（纯形状参照）。尺寸经 4 次"
               "名义校准选定，与环形参照名义条件对齐（C_Σ 90.07 vs 90.04"
               "fF、f01 4.564 vs 4.5645GHz、EJ/EC 61.8）；口袋边距 150μm"
               " 对齐新结构名义环境。校准轨迹见 verify.json "
               "native_calibration。官方几何 8 轮 last2diag=1.34% 收敛慢于"
               "新结构（6 轮即收敛），加密档 12 轮补收敛证据。",
        legacy_fixed_pocket_D=None),  # 无旧同口径记录，表 B 留空
}
# 加密档对象：环形（v1 最小差值 1.5e-5 出现处）+ native（基准档收敛慢）
REFINE_OBJECTS = ("annulus_cma", "native_ref")

NATIVE_CALIBRATION = [
    dict(pad_width=620, pad_height=220, pad_gap=30, pocket=650,
         C_sigma_fF=118.11, f01_GHz=4.009, EJ_over_EC=81.0,
         converged=False, note="默认口袋 650：C 超窗、EJ/EC 出窗"),
    dict(pad_width=560, pad_height=180, pad_gap=30, pocket="860x590",
         C_sigma_fF=93.58, f01_GHz=4.481, EJ_over_EC=64.2,
         converged=False, note="y 边距仅 100μm，f01 差 0.019 出窗"),
    dict(pad_width=560, pad_height=180, pad_gap=34, pocket="860x590",
         max_passes=8, C_sigma_fF=93.05, f01_GHz=4.493, EJ_over_EC=63.8,
         converged=False, note="8 轮 last2diag 0.905%，f01 差 0.007"),
    dict(pad_width=560, pad_height=176, pad_gap=34, pocket="860x690",
         max_passes=8, C_sigma_fF=90.07, f01_GHz=4.564, EJ_over_EC=61.8,
         converged=False, note="终选：名义条件与环形参照对齐；"
         "8 轮 last2diag 1.337%，收敛慢于新结构（如实记录）"),
]


def gen_novel(family):
    from robust_design_lab.geometry import novel_structures as geo
    return getattr(geo, family)


def um_shift(val, delta):
    """字符串 um 参数平移（与 q3d_shape_sweep._shift_um 同规则）。"""
    num = float(str(val).replace("um", ""))
    return f"{max(num + delta, 5.0):.2f}um"


def native_etch(opts, delta_um):
    """官方组件参数平移刻蚀（每边 δ 语义，对齐 buffer 模型）：
    pad_width/height 为全宽/全高 → ∓2δ（两侧各 δ）；pad_gap 两侧各退
    δ → +2δ；pocket 四边外退 → ±2δ（地边界同步蚀刻）。
    与 shapely buffer 的差别（直角平移 vs 连续偏移）在报告中标注。"""
    d = 2 * delta_um
    o = dict(opts)
    o["pad_width"] = um_shift(opts["pad_width"], -d)
    o["pad_height"] = um_shift(opts["pad_height"], -d)
    o["pad_gap"] = um_shift(opts.get("pad_gap", "30um"), d)
    o["pocket_width"] = um_shift(opts.get("pocket_width", "650um"), d)
    o["pocket_height"] = um_shift(opts.get("pocket_height", "650um"), d)
    return o


def scenarios_for(name, spec, delta=ETCH_DELTA_UM, max_passes=6,
                  percent_error=None):
    """三场景（名义/刻蚀±）求解；novel 走 buffer+独立口袋，native 走
    参数平移。返回 {tag: run_q3d 完整输出}。

    percent_error：真加密须降低误差目标（默认 0.5%）——只提高
    max_passes 时求解器达到原目标即停（2026-11-04 审阅反例）。"""
    from robust_design_lab.geometry.novel_structures import (
        struct_to_options, etch_variant)
    from robust_design_lab.experiments.q3d_shape_sweep import run_q3d
    scen = {}
    if spec["kind"] == "novel":
        s0 = gen_novel(spec["family"])(**spec["kw"])
        variants = [("nom", s0), ("ep", etch_variant(s0, +delta)),
                    ("em", etch_variant(s0, -delta))]
        for tag, st in variants:
            scen[tag] = run_q3d(struct_to_options(st), "NovelTwoPadTransmon",
                                "A_dual_pad", max_passes=max_passes,
                                percent_error=percent_error)
    elif spec["kind"] == "edited":
        variants = [("nom", spec["struct"]), ("ep",
                   etch_variant(spec["struct"], +delta)),
                   ("em", etch_variant(spec["struct"], -delta))]
        for tag, st in variants:
            scen[tag] = run_q3d(struct_to_options(st), "NovelTwoPadTransmon",
                                "A_dual_pad", max_passes=max_passes,
                                percent_error=percent_error)
    else:
        o0 = dict(spec["kw"])
        for tag, o in (("nom", o0), ("ep", native_etch(o0, +delta)),
                       ("em", native_etch(o0, -delta))):
            scen[tag] = run_q3d(dict(o), "TransmonPocket", "A_dual_pad",
                                max_passes=max_passes,
                                percent_error=percent_error)
    return scen


def c_sigma_breakdown(mat_fF, nets):
    """C_Σ 分解（与 compute_c_sigma A_dual_pad 分支同式，独立重算对质）：
    C_Σ = C12 + Cj + Σ_k C_1k·C_2k/(C_1k+C_2k)。"""
    mat = [list(map(float, row)) for row in mat_fF]
    n = len(nets)
    pad_idx = [i for i, net in enumerate(nets) if "pad" in net.lower()
               and not any(w in net.lower() for w in
                           ("connector", "readout", "bus"))]
    assert len(pad_idx) == 2, f"pad 网数异常: {nets}"
    p1, p2 = pad_idx
    c12 = -mat[p1][p2]
    ground_series = 0.0
    for k in range(n):
        if k in (p1, p2):
            continue
        c1k, c2k = max(-mat[p1][k], 0.0), max(-mat[k][p2], 0.0)
        if c1k + c2k > 0:
            ground_series += c1k * c2k / (c1k + c2k)
    return dict(C12_fF=c12, ground_series_fF=ground_series, Cj_fF=CJ_FF,
                C_sigma_fF=c12 + CJ_FF + ground_series,
                extra_nets=[nets[k] for k in range(n) if k not in (p1, p2)])


def evaluate_object(name, spec, delta=ETCH_DELTA_UM, max_passes=6,
                    refine=False, percent_error=None):
    """三场景评价：逐场景检查收敛——任一 converged=False 即拒收
    （保存失败状态与已得矩阵后抛异常，不生成有限 D；2026-11-06 复核
    R1：交指 nom/em 未收敛仍被写入 qualified=true 的反例）。"""
    t0 = time.perf_counter()
    scen = scenarios_for(name, spec, delta=delta, max_passes=max_passes,
                         percent_error=percent_error)
    not_conv = [t for t, s in scen.items()
                if s.get("converged") is not True]  # None/缺失同样拒收
    if not_conv:
        npass = {t: len(s.get("capacitance_all_passes_fF") or {})
                 for t, s in scen.items()}
        raise RuntimeError(
            "%s: 场景收敛标志非 True %s（max_passes=%s, 轮数 %s）——拒收，"
            "已得矩阵随本异常前的场景记录保存" % (name, not_conv,
                                                    max_passes, npass))

    def f01(c):
        ec = EC_GHZ_PER_F / c
        return math.sqrt(8 * EJ_GHZ * ec) - ec

    cs = {t: s["C_sigma_fF"] for t, s in scen.items()}
    f0 = f01(cs["nom"])
    d_ep = abs(math.log(f01(cs["ep"]) / f0))
    d_em = abs(math.log(f01(cs["em"]) / f0))
    row = dict(
        object=name, kind=spec["kind"], source=spec["source"],
        kw=spec.get("kw"), delta_um=delta, max_passes=max_passes,
        percent_error=(percent_error if percent_error is not None else 0.5),
        refine=refine, scenarios=scen,
        C_sigma_fF=cs["nom"], f01_GHz=f0,
        EJ_over_EC=EJ_GHZ / (EC_GHZ_PER_F / cs["nom"]),
        alpha_MHz=-EC_GHZ_PER_F / cs["nom"] * 1e3,
        D=max(d_ep, d_em), d_ep=d_ep, d_em=d_em,
        wall_s=round(time.perf_counter() - t0, 1),
        provenance=provenance(),
    )
    # 分解 + 对质（互电容 vs 对地分项的刻蚀变化）
    bd = {t: c_sigma_breakdown(s["matrix_fF"], s["nets"])
          for t, s in scen.items()}
    for t, b in bd.items():
        assert abs(b["C_sigma_fF"] - cs[t]) < 1e-9, \
            f"{name}/{t}: 分解重算 {b['C_sigma_fF']} ≠ 汇总 {cs[t]}"
    row["breakdown"] = bd
    row["breakdown_delta"] = dict(
        dC12_ep=bd["ep"]["C12_fF"] - bd["nom"]["C12_fF"],
        dC12_em=bd["em"]["C12_fF"] - bd["nom"]["C12_fF"],
        dG_ep=bd["ep"]["ground_series_fF"] - bd["nom"]["ground_series_fF"],
        dG_em=bd["em"]["ground_series_fF"] - bd["nom"]["ground_series_fF"],
        dC_ep=cs["ep"] - cs["nom"], dC_em=cs["em"] - cs["nom"])
    return row


def obj_path(name, refine=False):
    return OUT / (f"verify_{name}{'_refine' if refine else ''}.json")


def load_done(name, refine=False):
    p = obj_path(name, refine)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def run_all(only=None, skip_refine=False):
    print("== 第二步复验：修正同步地刻蚀（表 A）==", flush=True)
    rows = {}
    for name, spec in OBJECTS.items():
        if only and name not in only:
            continue
        done = load_done(name)
        if done:
            print(f"[跳过] {name}（已完成 D={done['D']:.5f}）", flush=True)
            rows[name] = done
            continue
        print(f"[求解] {name}: {spec['source']}", flush=True)
        try:
            row = evaluate_object(name, spec)
        except Exception as e:
            row = dict(object=name, failed=f"{type(e).__name__}: {e}"[:300])
        obj_path(name).write_text(
            json.dumps(row, ensure_ascii=False, default=str, indent=1),
            encoding="utf-8")
        if "D" in row:
            print(f"  CΣ={row['C_sigma_fF']:.2f}fF f01={row['f01_GHz']:.3f}"
                  f" D={row['D']:.5f} (ep {row['d_ep']:.5f} / em {row['d_em']:.5f})"
                  f" [{row['wall_s']}s]", flush=True)
            bd = row["breakdown_delta"]
            print(f"  ΔC分解 ep: dC12={bd['dC12_ep']:+.3f} dG={bd['dG_ep']:+.3f}"
                  f" (dCΣ={bd['dC_ep']:+.3f}fF)", flush=True)
        else:
            print(f"  失败: {row['failed']}", flush=True)
        rows[name] = row

    if skip_refine:
        return rows
    # 加密档对象（环形 + native）：可分辨性与收敛补检
    for name in REFINE_OBJECTS:
        done = load_done(name, refine=True)
        if done is not None:
            continue
        if only and name not in only:
            continue
        print(f"[加密档 max_passes={REFINE_PASSES}] {name}", flush=True)
        try:
            row = evaluate_object(name, OBJECTS[name],
                                  max_passes=REFINE_PASSES, refine=True)
            obj_path(name, refine=True).write_text(
                json.dumps(row, ensure_ascii=False, default=str, indent=1),
                encoding="utf-8")
            base = rows.get(name, {}).get("D")
            print(f"  D={row['D']:.5f} (基准档 D={base})", flush=True)
        except Exception as e:
            print(f"  加密档失败: {e}", flush=True)
    return rows


def summarize(rows):
    """两表汇总：表 A 修正同步地（新解）；表 B 旧固定地（引用记录）。"""
    table_a, table_b = [], []
    for name, spec in OBJECTS.items():
        r = rows.get(name) or {}
        if "D" in r:
            table_a.append(dict(
                object=name, model=("buffer+独立口袋" if spec["kind"] == "novel"
                                    else "参数平移+口袋同步(近似)"),
                C_sigma_fF=round(r["C_sigma_fF"], 2),
                f01_GHz=round(r["f01_GHz"], 3),
                EJ_over_EC=round(r["EJ_over_EC"], 1),
                D=round(r["D"], 6), d_ep=round(r["d_ep"], 6),
                d_em=round(r["d_em"], 6),
                converged_all=all(bool(s["converged"])
                                  for s in r["scenarios"].values()),
                breakdown=r["breakdown_delta"]))
        if spec.get("legacy_fixed_pocket_D") is not None:
            table_b.append(dict(
                object=name, model="旧固定地近似（口袋未随刻蚀移动）",
                D=spec["legacy_fixed_pocket_D"],
                source=spec["source"]))
    summary = dict(
        table_a_corrected_sync_pocket=table_a,
        table_b_legacy_fixed_pocket=table_b,
        note="表 A 与表 B 是两种工艺假设，排名不跨表比较；native_ref 的"
             "参数平移近似与 buffer 模型同表时按 model 字段区分。",
        native_calibration=NATIVE_CALIBRATION,
        refine_rows=[],
        provenance=provenance())
    for name in REFINE_OBJECTS:
        r = load_done(name, refine=True)
        if r and "D" in r:
            base = rows.get(name, {}).get("D")
            summary["refine_rows"].append(dict(
                object=name, max_passes=REFINE_PASSES,
                D=round(r["D"], 6),
                base_D=(round(base, 6) if isinstance(base, (int, float))
                        else None),
                dD_vs_base=(round(r["D"] - base, 6)
                            if isinstance(base, (int, float)) else None),
                converged_all=all(bool(s["converged"])
                                  for s in r["scenarios"].values())))
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, default=str, indent=1),
        encoding="utf-8")
    print("\n表 A（修正同步地）：")
    for t in table_a:
        print(f"  {t['object']:16s} {t['model'][:14]:14s} CΣ={t['C_sigma_fF']}"
              f" D={t['D']:.6f}")
    print("表 B（旧固定地，引用）：")
    for t in table_b:
        print(f"  {t['object']:16s} D={t['D']:.6f}")
    return summary


def refine2_objects():
    """真加密复验对象（2026-11-04 计划第一步）：环形起点、CMA 编辑
    冠军（0.00209）、有图终选（0.00225）、变间隙代表。编辑结构从
    pilot runs.jsonl 的采纳链精确重建。"""
    from robust_design_lab.geometry import boundary_edit as be
    pilot = ROOT / "results" / "region_edit_pilot_2026-11-03" / "runs.jsonl"
    final_action = {}
    if pilot.exists():
        for line in pilot.open(encoding="utf-8"):
            r = json.loads(line)
            final_action[(r["start"], r["arm"])] = r.get("final_action")

    def rebuilt(start_name, arm, base_struct, label):
        act = final_action.get((start_name, arm))
        op, seg, mag, width = be.action_from_kw(act)
        s = be.apply_edit(base_struct, op, seg, mag, width)
        return dict(kind="edited", struct=s, kw={},
                    source=f"pilot {arm} 终选重建: {label} "
                           f"({be.action_describe(act)})")

    annulus = gen_novel("q_annulus")(r_out=253.9, r_in=121.5, gap=33.8)
    vgap = gen_novel("m_variable_gap")(H=480.0, depth=290.0, g0=32.0,
                                       g_edge=88.0)
    return {
        "annulus_start": dict(kind="novel", family="q_annulus",
                              kw=dict(r_out=253.9, r_in=121.5, gap=33.8),
                              source="环形 v1 CMA 终选（起点）"),
        "annulus_cma_edit": rebuilt("A_annulus", "cma_es", annulus,
                                    "D=0.00209 待复验"),
        "annulus_vlm_edit": rebuilt("A_annulus", "vlm_img", annulus,
                                    "D=0.00225 待复验"),
        "vgap_start": dict(kind="novel", family="m_variable_gap",
                           kw=dict(H=480.0, depth=290.0, g0=32.0,
                                   g_edge=88.0),
                           source="变间隙 v1 VLM 终选（代表）"),
    }


REFINE2_OUT = ROOT / "results" / "refine2_verify_2026-11-04"
REFINE2_PCT = 0.1      # 误差目标 0.5% → 0.1%（真加密，非提高轮数上限）
REFINE2_PASSES = 20


def run_refine2(only=None):
    REFINE2_OUT.mkdir(parents=True, exist_ok=True)
    objs = refine2_objects()
    print(f"== 真加密复验：percent_error={REFINE2_PCT}% "
          f"max_passes={REFINE2_PASSES} ==", flush=True)
    for name, spec in objs.items():
        if only and name not in only:
            continue
        p = REFINE2_OUT / f"refine2_{name}.json"
        if p.exists():
            print(f"[跳过] {name}（已完成）", flush=True)
            continue
        print(f"[求解] {name}: {spec['source']}", flush=True)
        t0 = time.perf_counter()
        try:
            row = evaluate_object(name, spec, max_passes=REFINE2_PASSES,
                                  refine=True, percent_error=REFINE2_PCT)
            p.write_text(json.dumps(row, ensure_ascii=False, default=str,
                                    indent=1), encoding="utf-8")
            conv = {t: s["converged"] for t, s in row["scenarios"].items()}
            npass = {t: len(s["capacitance_all_passes_fF"] or {})
                     for t, s in row["scenarios"].items()}
            print(f"  CΣ={row['C_sigma_fF']:.2f} D={row['D']:.6f} "
                  f"converged={conv} passes={npass} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  失败: {type(e).__name__}: {e}", flush=True)
            p.write_text(json.dumps(dict(object=name, failed=str(e)[:300]),
                                    ensure_ascii=False), encoding="utf-8")


def simple_dual_pad(w=560.0, h=178.0, gap=30.0):
    """简单双岛参照：直边矩形对置平板（最朴素的平板电容机制，
    buffer 刻蚀与环形/交指同模型——2026-11-06 计划 B 的合格参照）。"""
    from shapely.geometry import LineString, box
    pl = box(-gap / 2 - w, -h / 2, -gap / 2, h / 2)
    pr = box(gap / 2, -h / 2, gap / 2 + w, h / 2)
    return dict(pad_left=pl, pad_right=pr, islands=[],
                junction=LineString([(-gap / 2, 0), (gap / 2, 0)]),
                label=f"simple_pad({w:.0f}x{h:.0f},gap{gap:.0f})")


REFS_B_OUT = ROOT / "results" / "refs_b_2026-11-06"
REFS_B_TARGET = (91.0, 92.0)  # 共同名义电容带（0.1% 口径，环形实测 91.61）


def _nominal_c(struct, pct=0.1, max_passes=None):
    """单场景名义校准（0.1%）：只解名义，返回 C_Σ。

    收敛标志必须显式 True（None/缺失同样拒收——2026-11-07 复核）。
    交指类网格收敛慢（20 轮内不达 0.1% 目标，第二轮校准放宽 30 轮
    并公开记账——与官方参照当年 12 轮才收敛同类）。"""
    from robust_design_lab.geometry.novel_structures import struct_to_options
    from robust_design_lab.experiments.q3d_shape_sweep import run_q3d
    if max_passes is None:
        max_passes = 20 if pct < 0.5 else 6
    r = run_q3d(struct_to_options(struct), "NovelTwoPadTransmon",
                "A_dual_pad", max_passes=max_passes, percent_error=pct)
    if r.get("converged") is not True:
        raise RuntimeError(f"校准场景收敛标志非 True（max_passes="
                           f"{max_passes}, 实际 "
                           f"{len(r.get('capacitance_all_passes_fF') or {})} 轮）")
    return r["C_sigma_fF"]


def run_refs_b(calibrate_only=False):
    """B 阶段：交指与简单双岛定尺寸至共同名义电容带（91-92 fF，
    0.1% 口径），合格后各做三场景。校准迭代公开记账（第一轮 6 次未
    入带：交指 20 轮不收敛×3、双岛 3 步偏低——第二轮按斜率外推并放
    宽交指轮数，全程留档）。"""
    REFS_B_OUT.mkdir(parents=True, exist_ok=True)
    from robust_design_lab.geometry.novel_structures import q_ellipse_idc
    plans = {
        # 第三轮：第二轮插值（idc: 172→90.98, 180→92.94；pad:
        # 720x215→89.77, 740x220→92.49），各取带内点
        "idc_ref": [dict(a=175.0), dict(a=177.0), dict(a=173.0)],
        "pad_ref": [dict(w=734.0, h=218.0), dict(w=738.0, h=219.0),
                    dict(w=730.0, h=217.0)],
    }
    makers = {
        "idc_ref": lambda kw: q_ellipse_idc(
            **{**dict(b=140.0, D=140.0, n_f=4, f_w=16.0, f_gap=32.0,
                      f_len=88.0, jg=40.0), **kw}),
        "pad_ref": lambda kw: simple_dual_pad(**kw),
    }
    results = {}
    for name, steps in plans.items():
        log = []
        chosen = None
        mp = 30 if name == "idc_ref" else None  # 交指第二轮放宽（记账）
        for i, kw in enumerate(steps):
            struct = makers[name](kw)
            try:
                c = _nominal_c(struct, max_passes=mp)
            except Exception as e:
                log.append(dict(step=i, kw=kw, failed=str(e)[:150]))
                continue
            log.append(dict(step=i, kw=kw, C_sigma_fF=c,
                            max_passes=mp or 20,
                            in_band=REFS_B_TARGET[0] <= c <= REFS_B_TARGET[1]))
            print(f"[校准] {name} step{i} {kw} → C={c:.2f}", flush=True)
            if REFS_B_TARGET[0] <= c <= REFS_B_TARGET[1]:
                chosen = (kw, c, struct)
                break
        if chosen is None:
            results[name] = dict(calibration=log, qualified=False)
            print(f"  {name}: 3 次校准未入带，如实记录", flush=True)
            continue
        if calibrate_only:
            results[name] = dict(calibration=log, qualified=True,
                                 kw=chosen[0], C_nominal=chosen[1])
            continue
        # 正式三场景与校准同配置（交指 mp=30；其余按 0.1% 档推导 20）；
        # 任一场景未收敛即拒收，qualified 保持 False（R1 修复）。
        mp3 = mp if mp is not None else 20
        try:
            row = evaluate_object(
                name, dict(kind="edited", struct=chosen[2], kw=chosen[0],
                           source=f"refs_b 校准合格 {chosen[0]}"),
                max_passes=mp3, refine=True, percent_error=0.1)
        except RuntimeError as e:
            results[name] = dict(calibration=log, qualified=False,
                                 failed=str(e)[:250])
            print(f"  {name}: 三场景未收敛，拒收（{str(e)[:90]}）",
                  flush=True)
            continue
        (REFS_B_OUT / f"refs_b_{name}.json").write_text(
            json.dumps(dict(calibration=log, row=row), ensure_ascii=False,
                       default=str, indent=1), encoding="utf-8")
        # 同源汇总：C/f/D 全部来自本次合格评价（不与校准轮混表）
        results[name] = dict(calibration=log, qualified=True,
                             kw=chosen[0], C_nominal=row["C_sigma_fF"],
                             f01=row["f01_GHz"], D=row["D"],
                             C_calib=chosen[1])
        print(f"  {name}: C={row['C_sigma_fF']:.2f}(校准 {chosen[1]:.2f}) "
              f"三场景 D={row['D']:.6f} f01={row['f01_GHz']:.4f}",
              flush=True)
    (REFS_B_OUT / "refs_b.json").write_text(
        json.dumps(dict(target=REFS_B_TARGET, results=results,
                        provenance=provenance()), ensure_ascii=False,
                   default=str, indent=1), encoding="utf-8")
    return results


MECH_C_OUT = ROOT / "results" / "mech_c_2026-11-06"


def _mech_segments(struct):
    """结果前固定规则选段：互电容主导段=外环 gap_min 最小；对地主导
    段=外环 gap_to_ground 最小（2026-11-06 计划 C：看结果前确定区域）。"""
    from robust_design_lab.geometry import spatial_feedback as sf
    rows = [r for r in sf.segment_metrics(struct)
            if r.get("valid") and r["ring"] == "outer"]
    by_gap = min(rows, key=lambda r: r["gap_min"])
    by_g2g = min(rows, key=lambda r: r["gap_to_ground"])
    return by_gap, by_g2g


def run_mech_c():
    """C 阶段：两起点 × （互电容区/对地区 × 增/减）+ 成对补偿组合，
    共 10 候选 30 场景（0.1%），分项符号与局部分析预测对照。"""
    from robust_design_lab.geometry import boundary_edit as be
    from robust_design_lab.geometry.novel_structures import q_annulus
    MECH_C_OUT.mkdir(parents=True, exist_ok=True)
    # 起点 2 优先 B 合格的 pad_ref；未合格时回退变间隙终选（0.1% 合格）
    starts = {"s1_annulus": q_annulus(r_out=253.9, r_in=121.5, gap=33.8)}
    try:
        rb = json.loads((REFS_B_OUT / "refs_b.json").read_text(
            encoding="utf-8"))
        pad = rb["results"]["pad_ref"]
        if pad.get("qualified"):
            starts["s2_pad_ref"] = simple_dual_pad(**pad["kw"])
    except Exception:
        pass
    if len(starts) < 2:
        starts["s2_vgap"] = gen_novel("m_variable_gap")(
            H=480.0, depth=290.0, g0=32.0, g_edge=88.0)

    results = []
    for sname, struct in starts.items():
        seg_c, seg_g = _mech_segments(struct)
        print(f"[C] {sname}: 互电容段=seg{seg_c['seg']}(gap={seg_c['gap_min']}"
              f"μm) 对地段=seg{seg_g['seg']}(g2g={seg_g['gap_to_ground']}μm)",
              flush=True)
        combos = [
            (f"{sname}_C12_tab", [("tab", seg_c["seg"], 0.5, 0.5)]),
            (f"{sname}_C12_notch", [("notch", seg_c["seg"], 0.5, 0.5)]),
            (f"{sname}_G_tab", [("tab", seg_g["seg"], 0.5, 0.5)]),
            (f"{sname}_G_notch", [("notch", seg_g["seg"], 0.5, 0.5)]),
            (f"{sname}_pair_comp", [("notch", seg_c["seg"], 0.5, 0.5),
                                    ("tab", seg_g["seg"], 0.5, 0.5)]),
        ]
        for cname, actions in combos:
            p = MECH_C_OUT / f"mech_{cname}.json"
            if p.exists():
                continue
            s2 = struct
            try:
                for op, seg, mag, w in actions:
                    s2 = be.apply_edit(s2, op, seg, mag, w)
                row = evaluate_object(cname, dict(kind="edited", struct=s2,
                                                   kw=dict(
                                                       actions=[list(a) for a
                                                                in actions]),
                                                   source=f"mech_c {sname}"),
                                      max_passes=20, refine=True,
                                      percent_error=0.1)
            except be.InvalidEditError as e:
                row = dict(object=cname, failed=f"无效编辑: {e}"[:200])
            except Exception as e:
                row = dict(object=cname,
                           failed=f"{type(e).__name__}: {e}"[:200])
            row["actions"] = [list(a) for a in actions]
            p.write_text(json.dumps(dict(row=row), ensure_ascii=False,
                                    default=str, indent=1), encoding="utf-8")
            if "D" in row:
                bd = row["breakdown_delta"]
                print(f"  {cname}: D={row['D']:.6f} "
                      f"dC12(ep)={bd['dC12_ep']:+.3f} "
                      f"dG(ep)={bd['dG_ep']:+.3f}", flush=True)
            else:
                print(f"  {cname}: {row['failed']}", flush=True)
            results.append(dict(row=row))
        (MECH_C_OUT / "mech_c.json").write_text(
            json.dumps(dict(starts={k: str(v['label']) for k, v in
                                    starts.items()}, results=results),
                       ensure_ascii=False, default=str, indent=1),
            encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑指定对象（如 --only native_ref）")
    ap.add_argument("--skip-refine", action="store_true")
    ap.add_argument("--refine2", action="store_true",
                    help="真加密复验（percent_error 0.1%%，四对象）")
    ap.add_argument("--refs-b", action="store_true",
                    help="B 阶段：交指/简单双岛定尺寸至 91-92fF 并三场景")
    ap.add_argument("--refs-b-calibrate-only", action="store_true")
    ap.add_argument("--mech-c", action="store_true",
                    help="C 阶段：机制区分编辑（两起点 10 候选）")
    args = ap.parse_args()
    t0 = time.perf_counter()
    if args.mech_c:
        run_mech_c()
        print(f"\n总耗时 {time.perf_counter() - t0:.0f}s → {MECH_C_OUT}")
        return
    if args.refs_b or args.refs_b_calibrate_only:
        run_refs_b(calibrate_only=args.refs_b_calibrate_only)
        print(f"\n总耗时 {time.perf_counter() - t0:.0f}s → {REFS_B_OUT}")
        return
    if args.refine2:
        run_refine2(only=args.only)
        print(f"\n总耗时 {time.perf_counter() - t0:.0f}s → {REFINE2_OUT}")
        return
    rows = run_all(only=args.only, skip_refine=args.skip_refine)
    summarize(rows)
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s → {OUT}")


if __name__ == "__main__":
    main()


# ── E 阶段：锁定候选的未见扰动验证（预注册，2026-11-08） ──────
E_OUT = ROOT / "results" / "unseen_stress_E_2026-11-08"
E_SCENARIOS = [
    ("U+0.25", "uniform", 0.25),
    ("U-0.25", "uniform", -0.25),
    ("NU-left+0.25", "left_only", 0.25),   # 仅左岛内缩（掩模左偏代理）
    ("NU-right+0.25", "right_only", 0.25),  # 仅右岛内缩
]


def nonuniform_etch(struct, mode, delta):
    """岛级非均匀刻蚀（预注册代理）：left_only/right_only 仅对一侧
    岛施加 buffer，另一岛不动——掩模左右偏移的一阶代理。"""
    from robust_design_lab.geometry.novel_structures import etch_variant
    if mode == "uniform":
        return etch_variant(struct, delta)
    out = dict(struct)
    key = "pad_left" if mode == "left_only" else "pad_right"
    out[key] = struct[key].buffer(-delta, quad_segs=8)
    return out


def run_unseen_E():
    """E：3 对象（冠军/合格交指/双岛）× 4 预注册未见场景，0.1%。"""
    from shapely import wkt as W
    from robust_design_lab.geometry.novel_structures import (
        q_annulus, q_ellipse_idc, struct_to_options)
    from robust_design_lab.geometry import boundary_edit as be
    from robust_design_lab.experiments.q3d_shape_sweep import run_q3d
    E_OUT.mkdir(parents=True, exist_ok=True)
    champ = be.apply_edit(q_annulus(r_out=253.9, r_in=121.5, gap=33.8),
                          "bulge", be.seg_encode("R", "inner", 3),
                          0.36, 0.30)
    objs = {
        "champion": champ,
        "idc_ref": q_ellipse_idc(a=177.0, b=140.0, D=140.0, n_f=4,
                                 f_w=16.0, f_gap=32.0, f_len=88.0, jg=40.0),
        "pad_ref": simple_dual_pad(w=734.0, h=218.0),
    }
    import math
    rows = {}
    for oname, struct in objs.items():
        p = E_OUT / f"E_{oname}.json"
        if p.exists():
            rows[oname] = json.loads(p.read_text(encoding="utf-8"))
            continue
        rec = dict(object=oname, scenarios={})
        base = None
        for tag, mode, delta in E_SCENARIOS:
            st = nonuniform_etch(struct, mode, delta)
            r = run_q3d(struct_to_options(st), "NovelTwoPadTransmon",
                        "A_dual_pad", max_passes=30, percent_error=0.1)
            rec["scenarios"][tag] = dict(C=r["C_sigma_fF"],
                                         converged=r["converged"],
                                         passes=len(r["capacitance_all_passes_fF"]
                                                    or {}),
                                         matrix=r["matrix_fF"], nets=r["nets"])
            if tag == "U+0.25":
                base = r["C_sigma_fF"]
        if any(v["converged"] is not True for v in rec["scenarios"].values()):
            rec["rejected"] = "存在非 True 收敛"
        else:
            def f01(c):
                ec = 19.36 / c
                return math.sqrt(8 * 13.28 * ec) - ec
            f0 = f01(base)
            rec["drift"] = {t: abs(math.log(f01(v["C"]) / f0))
                            for t, v in rec["scenarios"].items()}
            rec["D_unseen"] = max(rec["drift"].values())
        p.write_text(json.dumps(rec, ensure_ascii=False, default=str,
                                indent=1), encoding="utf-8")
        rows[oname] = rec
        print(f"{oname}: D_unseen={rec.get('D_unseen')} "
              f"drift={rec.get('drift')}", flush=True)
    return rows


# ── E2：原冠军的未见扰动复验（程序化载入+固定地界+未刻蚀基准） ──
E2_OUT = ROOT / "results" / "unseen_stress_E2_2026-11-07"


def load_champion_action():
    """从 pilot 原始记录程序化载入冠军动作（2026-11-07 复核 P1：E
    曾手写 0.36/0.30，实际为 0.3634/0.3001——边界差 0.136μm）。"""
    src = (ROOT / "results" / "region_edit_pilot_2026-11-03"
           / "runs.jsonl")
    for line in src.open(encoding="utf-8"):
        r = json.loads(line)
        if (r.get("arm"), r.get("start")) == ("cma_es", "A_annulus"):
            return r["final_action"], r["final_D"]
    raise FileNotFoundError("冠军动作记录未找到")


def island_etch_fixed_pocket(struct, side, delta):
    """岛级尺寸偏差（复核术语：非'掩模偏移代理'）：仅单侧岛 buffer，
    地口袋固定为**未刻蚀名义口袋**（塞入 struct['pocket']——struct_
    to_options 优先取该字段，不再随电极外接框重算）。"""
    from robust_design_lab.geometry.novel_structures import nominal_pocket
    out = dict(struct)
    out["pocket"] = nominal_pocket(struct)  # 名义地界，扰动下不动
    key = "pad_left" if side == "left" else "pad_right"
    out[key] = struct[key].buffer(-delta, quad_segs=8)
    return out


def run_unseen_E2():
    """补 8 场景：原冠军 4（U±0.25 均匀 + 左/右岛级，固定地界）+
    双岛 2 + 交指 2（左右岛级，固定地界）。基准=各自同几何未刻蚀
    名义态存档（冠军 refine2；双岛/交指 refs_b）。"""
    import math
    from shapely import wkt as W
    from robust_design_lab.geometry.novel_structures import (
        q_annulus, q_ellipse_idc, struct_to_options)
    from robust_design_lab.geometry import boundary_edit as be
    from robust_design_lab.experiments.q3d_shape_sweep import run_q3d
    E2_OUT.mkdir(parents=True, exist_ok=True)
    act, champ_D01 = load_champion_action()
    op, seg, mag, width = be.action_from_kw(act)
    champ = be.apply_edit(q_annulus(r_out=253.9, r_in=121.5, gap=33.8),
                          op, seg, mag, width)

    def f01(c):
        ec = 19.36 / c
        return math.sqrt(8 * 13.28 * ec) - ec

    plans = {
        "champion_exact": dict(
            struct=champ,
            nom_C=json.loads((ROOT / "results" / "refine2_verify_2026-11-04"
                              / "refine2_annulus_cma_edit.json").read_text(
                                  encoding="utf-8"))["C_sigma_fF"],
            scenes=[("Us+0.25", "uniform_sync", 0.25),
                    ("Us-0.25", "uniform_sync", -0.25),
                    ("IL+0.25", "left", 0.25), ("IR+0.25", "right", 0.25)]),
        "pad_ref": dict(
            struct=simple_dual_pad(w=734.0, h=218.0),
            nom_C=json.loads((REFS_B_OUT / "refs_b_pad_ref.json").read_text(
                encoding="utf-8"))["row"]["C_sigma_fF"],
            scenes=[("IL+0.25", "left", 0.25), ("IR+0.25", "right", 0.25)]),
        "idc_ref": dict(
            struct=q_ellipse_idc(a=177.0, b=140.0, D=140.0, n_f=4,
                                 f_w=16.0, f_gap=32.0, f_len=88.0, jg=40.0),
            nom_C=json.loads((REFS_B_OUT / "refs_b_idc_ref_30p.json")
                             .read_text(encoding="utf-8"))["row"]
                ["C_sigma_fF"],
            scenes=[("IL+0.25", "left", 0.25), ("IR+0.25", "right", 0.25)]),
    }
    summary = {}
    for name, spec in plans.items():
        p = E2_OUT / f"E2_{name}.json"
        if p.exists():
            summary[name] = json.loads(p.read_text(encoding="utf-8"))
            continue
        rec = dict(object=name, scenes={}, champion_action=None,
                   champion_D01=champ_D01 if name == "champion_exact"
                   else None)
        if name == "champion_exact":
            rec["champion_action"] = dict(op=op, seg=seg, mag=mag,
                                          width=width)
        for tag, mode, delta in spec["scenes"]:
            if mode == "uniform_sync":
                # 均匀±0.25 且地边界同步刻蚀（etch_variant 自带独立
                # 口袋同步偏移）——与双岛/交指复用的均匀场景同口径
                from robust_design_lab.geometry.novel_structures import \
                    etch_variant
                st = etch_variant(spec["struct"], delta)
            elif mode == "uniform":
                from robust_design_lab.geometry.novel_structures import \
                    etch_variant
                st = etch_variant(spec["struct"], delta)
                st = dict(st, pocket=nominal_pocket_safe(spec["struct"]))
            else:
                st = island_etch_fixed_pocket(spec["struct"], mode, delta)
            r = run_q3d(struct_to_options(st), "NovelTwoPadTransmon",
                        "A_dual_pad", max_passes=30, percent_error=0.1)
            assert r["converged"] is True, f"{name}/{tag} 未收敛，拒收"
            drift = abs(math.log(f01(r["C_sigma_fF"])
                                 / f01(spec["nom_C"])))
            rec["scenes"][tag] = dict(
                C=r["C_sigma_fF"],
                drift_vs_unetched_nominal=drift,
                # Δf = f0·(1−e^{−drift})，f0 名义 4.53 GHz（旧版误除
                # ln10——2026-11-07 复核更正，冠军正向应为 6.8522 MHz）
                MHz=4530.0 * (1 - math.exp(-drift)))
            print(f"  {name}/{tag}: C={r['C_sigma_fF']:.3f} "
                  f"drift={drift:.6f}", flush=True)
        # 双岛/交指：合并复用的同步地界均匀场景（旧 E json 矩阵按
        # 未刻蚀名义基准重算——2026-11-07 复核：汇总不得只取单岛）
        if name in ("pad_ref", "idc_ref"):
            old = json.loads((ROOT / "results" / "unseen_stress_E_2026-11-08"
                              / f"E_{name}.json").read_text(encoding="utf-8"))
            for tag in ("U+0.25", "U-0.25"):
                c = old["scenarios"][tag]["C"]
                rec["scenes"][tag + "(sync,复用)"] = dict(
                    C=c, drift_vs_unetched_nominal=abs(
                        math.log(f01(c) / f01(spec["nom_C"]))),
                    MHz=4530.0 * (1 - math.exp(-abs(
                        math.log(f01(c) / f01(spec["nom_C"]))))), reused=True)
        rec["D_unseen_vs_nominal"] = max(
            v["drift_vs_unetched_nominal"] for v in rec["scenes"].values())
        p.write_text(json.dumps(rec, ensure_ascii=False, default=str,
                                indent=1), encoding="utf-8")
        summary[name] = rec
        print(f"{name}: D_unseen={rec['D_unseen_vs_nominal']:.6f}",
              flush=True)
    (E2_OUT / "E2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    return summary


def nominal_pocket_safe(struct):
    from robust_design_lab.geometry.novel_structures import nominal_pocket
    return nominal_pocket(struct)


B005_OUT = ROOT / "results" / "b_refine_005_2026-11-07"


def run_b_refine_005():
    """B 加严（0.05%/30 轮）：环形起点、环形冠军、合格交指三对象
    各三场景（双岛 0.05% 已有 pct005_ 记录，核配置后复用）。判定：
    冠军对起点约 3% 的差值跨档可分辨性。"""
    import math
    from robust_design_lab.geometry.novel_structures import (
        q_annulus, q_ellipse_idc)
    from robust_design_lab.geometry import boundary_edit as be
    B005_OUT.mkdir(parents=True, exist_ok=True)
    act, _ = load_champion_action()
    op, seg, mag, width = be.action_from_kw(act)
    objs = {
        "annulus_start": dict(kind="novel", family="q_annulus",
                              kw=dict(r_out=253.9, r_in=121.5, gap=33.8),
                              source="环形终选参数起点"),
        "champion": dict(kind="edited",
                         struct=be.apply_edit(
                             q_annulus(r_out=253.9, r_in=121.5, gap=33.8),
                             op, seg, mag, width),
                         kw={}, source=f"冠军程序化载入 {mag}/{width}"),
        "idc_ref": dict(kind="edited",
                        struct=q_ellipse_idc(a=177.0, b=140.0, D=140.0,
                                             n_f=4, f_w=16.0, f_gap=32.0,
                                             f_len=88.0, jg=40.0),
                        kw={}, source="合格交指 a=177"),
    }
    for name, spec in objs.items():
        p = B005_OUT / f"b005_{name}.json"
        if p.exists():
            print(f"[跳过] {name}", flush=True)
            continue
        print(f"[0.05%] {name}", flush=True)
        t0 = time.perf_counter()
        try:
            row = evaluate_object(name, spec, max_passes=30, refine=True,
                                  percent_error=0.05)
        except RuntimeError as e:
            print(f"  拒收: {str(e)[:120]}", flush=True)
            continue
        p.write_text(json.dumps(dict(row=row), ensure_ascii=False,
                                default=str, indent=1), encoding="utf-8")
        print(f"  D={row['D']:.7f} C={row['C_sigma_fF']:.4f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
