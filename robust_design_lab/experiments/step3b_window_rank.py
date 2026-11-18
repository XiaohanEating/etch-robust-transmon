# -*- coding: utf-8 -*-
"""第三步(b)：衔接位置敏感性 + 跨形状排序证据（纯后处理 + 2D 参考重算）。

1. 窗口敏感性：矩形三场景两档在 W_IN ∈ {15, 20, 30} μm 下重算
   p_SM/p_SA 与有符号漂移（20 = 现行值）。2D 参考随窗口重算；
   检验"衔接位置变化时总量与漂移是否稳定"（不得以最差误差选窗）。
2. 跨形状排序：10 代表名义 p_SM/p_SA/drift_geom 的两档 Spearman 相关。
3. MA 影响界限已量化于 g1_repost.json（≤0.13% T1），此处汇总引用。

输出 results/step3b_window_rank_2026-10-24/。
"""

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

OUT = ROOT / "results" / "step3b_window_rank_2026-10-24"
OUT.mkdir(parents=True, exist_ok=True)
ARCHIVES = ROOT / "results" / "finite_alignment_v1" / "rect_reference"
TRANSPORT = ROOT / "results" / "finite_alignment_transport_v1"


def refs_for_window(s3, w_in):
    """以指定 W_IN 重算 2D 参考（outer+gap）。"""
    old = s3.W_IN
    s3.W_IN = w_in
    try:
        return dict(outer=s3.two_d_reference(), gap=s3.gap_reference())
    finally:
        s3.W_IN = old


def rectangle_at_window(s3, w_in, lj_nH):
    import numpy as np
    from robust_design_lab.experiments.field_audit import load_solution
    from robust_design_lab.evaluator.lom import transmon_metrics

    refs = refs_for_window(s3, w_in)
    folders = {
        ("0.01", "nominal"): ARCHIVES / "h0p01" / "nominal",
        ("0.01", "etch_plus"): ARCHIVES / "h0p01" / "etch_plus",
        ("0.01", "etch_minus"): ARCHIVES / "h0p01" / "etch_minus",
        ("0.005", "nominal"): ARCHIVES / "h0p005" / "nominal",
        ("0.005", "etch_plus"): TRANSPORT / "etch_plus",
        ("0.005", "etch_minus"): TRANSPORT / "etch_minus",
    }
    out = {}
    for (level, scene), folder in folders.items():
        fs, rec, _ = load_solution(folder)
        with np.load(folder / "surfaces.npz", allow_pickle=False) as s:
            surfaces = {k: s[k] for k in s.files}
        pr = s3.corrected_participation(fs, surfaces, rec, lj_nH, refs)
        tm = transmon_metrics(fs.C_q_fF * 1e-15, lj_nH)
        rate = 2 * math.pi * tm["f01_GHz"] * 1e9 * (
            pr["p_SM"] + pr["p_SA"] + pr["p_MA"])
        out[(level, scene)] = dict(p_SM=pr["p_SM"], p_SA=pr["p_SA"],
                                   p_MA=pr["p_MA"], rate=rate)
    return out


def main():
    from robust_design_lab.experiments import g1_step3_fast_rectangle as s3

    t0 = time.perf_counter()
    report = dict(provenance=provenance(), windows={})

    # 固定 Lj：现行矩形报告的 pinned 值
    base = json.loads((ROOT / "results" / "g1_step3_rectangle_2026-10-23"
                       / "step3_fast_rectangle.json").read_text(encoding="utf-8"))
    lj_nH = base["lj_pinned_nH"]

    for w in (15.0, 20.0, 30.0):
        if w == 20.0:
            res = None     # 现行值直接引用 base 场景表换算
            rows = {}
            for level, tag in (("h0p01", "h0p01"), ("h0p005", "h0p005")):
                for scene in ("nominal", "etch_plus", "etch_minus"):
                    r = base["scenarios"][tag][scene]
                    rows[(level, scene)] = dict(
                        p_SM=r["p_SM_corrected"], p_SA=r["p_SA_corrected"],
                        p_MA=r["p_MA_corrected"], rate=None)
        else:
            rows = rectangle_at_window(s3, w, lj_nH)
        # 漂移与档位差
        drifts = {}
        for level in ("0.01", "0.005"):
            key_lvl = "h0p01" if level == "0.01" else "h0p005"
            n = rows.get((level, "nominal")) or rows.get((key_lvl, "nominal"))
            for scene in ("etch_plus", "etch_minus"):
                r = (rows.get((level, scene)) or rows.get((key_lvl, scene)))
                if w == 20.0:
                    # 现行表无 rate 字段——由 p 与 f01 重算
                    rate_n = 2 * math.pi * base["scenarios"][key_lvl]["nominal"]["f01_GHz"] * 1e9 * (
                        n["p_SM"] + n["p_SA"] + n["p_MA"])
                    rate_s = 2 * math.pi * base["scenarios"][key_lvl][scene]["f01_GHz"] * 1e9 * (
                        r["p_SM"] + r["p_SA"] + r["p_MA"])
                else:
                    rate_n, rate_s = n["rate"], r["rate"]
                drifts[(level, scene)] = math.log(rate_n / rate_s) * -1
        diffs = {sc: abs(drifts[("0.01", sc)] - drifts[("0.005", sc)])
                 for sc in ("etch_plus", "etch_minus")}
        nom01 = rows.get(("0.01", "nominal")) or rows.get(("h0p01", "nominal"))
        nom005 = rows.get(("0.005", "nominal")) or rows.get(("h0p005", "nominal"))
        report["windows"][str(w)] = dict(
            p_SM=dict(h0p01=nom01["p_SM"], h0p005=nom005["p_SM"],
                      rel=abs(nom005["p_SM"] / nom01["p_SM"] - 1)),
            p_SA=dict(h0p01=nom01["p_SA"], h0p005=nom005["p_SA"],
                      rel=abs(nom005["p_SA"] / nom01["p_SA"] - 1)),
            drift_level_diff=diffs,
            drifts={f"{a}_{b}": v for (a, b), v in drifts.items()})
        print(f"[W_IN={w}] SM rel={report['windows'][str(w)]['p_SM']['rel']*100:.2f}% "
              f"SA rel={report['windows'][str(w)]['p_SA']['rel']*100:.2f}% "
              f"drift_diff={diffs}", flush=True)
        write_json(OUT / "window_rank.json", report)

    # 跨形状排序（来自 g1_repost）
    repost = json.loads((ROOT / "results" / "step3_repost_g1_2026-10-24"
                         / "g1_repost.json").read_text(encoding="utf-8"))
    tags = list(repost["reps"])
    # 名义两档 p 排序 + 每代表 drift（取档间独立量的均值代表形状值）
    import scipy.stats as st
    import numpy as np
    sm01, sm005, sa01, sa005, dr = [], [], [], [], []
    crit_rows = repost["reps"]
    # 名义 p 从 criteria 拿不到绝对值——重读矩阵输出里的逐场景行
    rows = [json.loads(l) for l in (ROOT / "results" / "g1_step4_matrix_2026-10-24"
                                    / "matrix.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # 用 g1_repost 的原始结果不可得绝对值 → 由 step3_repost 内部表重算会重复；
    # 改为直接从 repost 报告补：本脚本重跑 post_one 太贵，改从矩阵 jsonl 的
    # p_SM_corrected（旧公式）不可用——用 repost 内 criteria 无绝对值。
    # 结论：排序证据需要绝对值，补存：重读 60 存档太重，改为在 repost 报告中
    # 已存 per-rep criteria + drifts + ma_*，但没有名义绝对值。
    # 因此这里从窗口脚本自身补：直接调用 step3_repost 的 post_one 一次/代表
    # （仅名义两档，20 个轻量后处理）。
    from robust_design_lab.experiments.step3_repost_g1 import post_one
    pin = repost["pinned_Lj_nH"]
    for tag in tags:
        a = post_one((tag, "0.01", "nominal", pin[tag] * 1e-9))
        b = post_one((tag, "0.005", "nominal", pin[tag] * 1e-9))
        sm01.append(a["p_SM"]); sm005.append(b["p_SM"])
        sa01.append(a["p_SA"]); sa005.append(b["p_SA"])
        # drift：档间平均（形状级排序量）
        d_a = crit_rows[tag]["drifts"]
        dr.append(np.mean([abs(v) for v in d_a.values()]))
    rank = dict(
        p_SM_spearman=dict(rho=st.spearmanr(sm01, sm005).statistic,
                           p=st.spearmanr(sm01, sm005).pvalue),
        p_SA_spearman=dict(rho=st.spearmanr(sa01, sa005).statistic,
                           p=st.spearmanr(sa01, sa005).pvalue),
        note="名义参与比两档排序相关（10 代表）；drift 档间均值为形状级参考量")
    report["cross_shape_ranking"] = rank
    report["tags"] = tags
    report["wall_s"] = round(time.perf_counter() - t0, 1)
    write_json(OUT / "window_rank.json", report)
    print(f"ranking: {rank}  wall={report['wall_s']}s")


if __name__ == "__main__":
    main()
