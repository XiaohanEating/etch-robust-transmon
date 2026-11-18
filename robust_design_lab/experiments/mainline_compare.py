# -*- coding: utf-8 -*-
"""主线对比实验 v2：共同评价协议下的 VLM vs 强数值基线。

v1（results/mainline_compare_2026-11-02，保留不动）经 2026-11-02 审阅
证实的问题：地口袋同步刻蚀被电极外接框抵消（已在
geometry/novel_structures.py 修复为独立名义口袋偏移）、VLM 含起点十次
评价 vs 数值九次、可行性口径不一致、空回复标完成、CMA seed=0 随机、
终选参数字符串、三场景矩阵/收敛状态/工程路径未存档。

v2 全部经 search/common_protocol.py：
- 家族级共享起点实测一次（start_<family>.json），四臂注入复用；
- 每臂恰好 9 次新增候选求解；重复点/起点复用去重（duplicate 记账，
  不耗预算）；求解失败消耗预算并罚分；
- objective = D（可行）/ 罚分（出窗或失败）；接受判据与终选统一；
- 每条记录含三场景完整 Q3D 输出（电容矩阵、网名、逐轮电容、收敛
  状态、工程路径）；VLM 调用完整落盘（提示、图像哈希、原始响应）；
- 协议身份写 protocol.json（参数箱/起点/预算/窗口/罚分/种子表/
  口袋模型/结参数）；结果 runs.jsonl 逐臂落盘、可续跑。

用法：
  python mainline_compare.py                 # 全部家族 × 全部臂
  python mainline_compare.py --start-only    # 只做共享起点实测（小批验证）
  python mainline_compare.py --families m_variable_gap --arms cma_es
"""
import argparse
import base64
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance
from robust_design_lab.search import common_protocol as cp

OUT = ROOT / "results" / "mainline_compare_v2_2026-11-02"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = OUT / "runs.jsonl"
VLM_CALLS = OUT / "vlm_calls"
VLM_CALLS.mkdir(exist_ok=True)

BUDGET = 9
ETCH_DELTA_UM = 0.5
EJ_GHZ = 13.28          # Lj=12.31nH 固定（与资格记录一致）
EC_GHZ_PER_F = 19.36    # EC[GHz] = 19.36 / C_Σ[fF]
ARMS = ("vlm", "coordinate_descent", "bayesian_opt", "cma_es")

PARAM_BOXES = {
    "m_variable_gap": {"H": (420, 540), "depth": (250, 330),
                       "g0": (20, 34), "g_edge": (80, 115)},
    "q_annulus": {"r_out": (230, 270), "r_in": (105, 125), "gap": (24, 34)},
}
STARTS = {
    "m_variable_gap": dict(H=480.0, depth=290.0, g0=26.0, g_edge=100.0),
    "q_annulus": dict(r_out=250.0, r_in=115.0, gap=28.0),
}
DESIGN_NOTES = {
    "m_variable_gap": "平滑变间隙双岛：对置间隙 g(y)=g0+(g_edge-g0)(2y/H)^2，"
                      "结区窄两端宽——敏感度沿边界分散（资格实测 D=0.0050）",
    "q_annulus": "环形垫对：两个完整圆环并置，内缘为额外工作边（资格实测 D=0.0033）",
}


def gen(family):
    from robust_design_lab.geometry.novel_structures import (
        MAINLINE_FAMILIES, q_annulus)
    return q_annulus if family == "q_annulus" else MAINLINE_FAMILIES[family]


def eval_kwargs(family, kwargs):
    """三场景 Q3D → 指标 + 每场景完整求解记录；任一场景异常即整体失败。

    完整记录含：网名、电容矩阵、C_Σ、逐轮电容、收敛状态、工程路径
    （run_q3d v2 输出）——不再只留 C_Σ 汇总（审阅 §3 证据链要求）。
    """
    from robust_design_lab.geometry.novel_structures import (
        struct_to_options, etch_variant)
    from robust_design_lab.experiments.q3d_shape_sweep import run_q3d
    s = gen(family)(**kwargs)
    scen = {}
    for tag, st in (("nom", s), ("ep", etch_variant(s, +ETCH_DELTA_UM)),
                    ("em", etch_variant(s, -ETCH_DELTA_UM))):
        scen[tag] = run_q3d(struct_to_options(st), "NovelTwoPadTransmon",
                            "A_dual_pad")

    def f01(c):
        ec = EC_GHZ_PER_F / c
        return math.sqrt(8 * EJ_GHZ * ec) - ec

    f0 = f01(scen["nom"]["C_sigma_fF"])
    D = max(abs(math.log(f01(scen["ep"]["C_sigma_fF"]) / f0)),
            abs(math.log(f01(scen["em"]["C_sigma_fF"]) / f0)))
    return dict(D=D, C_sigma_fF=scen["nom"]["C_sigma_fF"], f01_GHz=f0,
                alpha_MHz=-EC_GHZ_PER_F / scen["nom"]["C_sigma_fF"] * 1e3,
                scenarios=scen)


def eval_kwargs_retry(family, kwargs, retries=1, wait_s=180.0):
    """COM 忙（遗留求解占用 Desktop）时等待重试一次，其余异常直抛。

    重试是运行时恢复：失败尝试不产生记录、不消耗预算（预算由
    ProtocolSession 按成功返回的求解计）。"""
    for attempt in range(retries + 1):
        try:
            return eval_kwargs(family, kwargs)
        except Exception as e:
            busy = ("com_error" in str(type(e)).lower() or "COM" in str(e)
                    or "发生意外" in str(e))
            if busy and attempt < retries:
                print("  [COM忙] 等 %.0fs 重试" % wait_s, flush=True)
                time.sleep(wait_s)
                continue
            raise


def make_evaluator(family):
    def evaluator(kw):
        return eval_kwargs_retry(family, kw)
    return evaluator


def render(family, kwargs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = gen(family)(**kwargs)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for p in (s["pad_left"], s["pad_right"]):
        ax.fill(*p.exterior.xy, color="#2196F3", alpha=0.85,
                edgecolor="#0D47A1", linewidth=0.8, zorder=3)
        for h in p.interiors:
            ax.fill(*h.xy, color="white", zorder=4)
    x, y = s["junction"].xy
    ax.plot(x, y, color="#F44336", linewidth=2.5, zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def shared_start(family):
    """家族级共享起点：实测一次落盘，所有臂注入同一条记录。"""
    cache = OUT / ("start_%s.json" % family)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["metrics"]
    t0 = time.perf_counter()
    m = eval_kwargs_retry(family, dict(STARTS[family]))
    cache.write_text(json.dumps(
        dict(family=family, kw=STARTS[family], metrics=m,
             wall_s=round(time.perf_counter() - t0, 1),
             provenance=provenance()),
        ensure_ascii=False, default=str), encoding="utf-8")
    return m


# ── VLM 提示与调用（完整落盘） ──────────────────────────────────

def build_prompt(family, cur, cur_rec, history):
    box = PARAM_BOXES[family]
    rules = ("设计规则：目标 C_Σ∈[70,105]fF 且 f01∈[4.5,5.5]GHz 下最小化"
             "刻蚀漂移 D。EJ/EC=0.686·C_Σ[fF]，甜点≈50 即 C_Σ≈73fF。"
             "出窗参数组合按罚分处理，不会因 D 低被采纳。结构："
             + DESIGN_NOTES[family] + "。可调参数与范围：" + json.dumps(box)
             + "。Lj=12.31nH 固定。")
    hist = ""
    if history:
        lines = ["  %s → D=%.5f%s" % (
            json.dumps(h["kw"]),
            h["D"] if h["D"] is not None else float("nan"),
            "" if h["feasible"] else "（出窗，罚分）")
            for h in history[-6:]]
        hist = "\n历史提案实测：\n" + "\n".join(lines)
    return (rules + "\n当前参数：" + json.dumps(cur)
            + "\n当前实测：C_Σ=%.1ffF f01=%.2fGHz D=%.5f%s"
            % (cur_rec["metrics"]["C_sigma_fF"],
               cur_rec["metrics"]["f01_GHz"], cur_rec["metrics"]["D"],
               "" if cur_rec["feasible"] else "（出窗，罚分）")
            + hist
            + "\n提出 3 个参数调整（每个只改 1-2 个参数，范围合规）。"
            '只返回 JSON 数组：[{"param":"名","value":数值,'
            '"reason":"简短理由"}]')


def make_vlm_chat(family):
    """真实模型调用 + 完整落盘（提示全文、图像 sha256、原始响应）。"""
    from robust_design_lab.vlm.aliyun import load_credentials, chat
    creds = load_credentials()

    def chat_fn(rnd, cur, cur_rec, history):
        img = OUT / ("vlm_%s_r%d.png" % (family, rnd))
        render(family, cur, img)
        prompt = build_prompt(family, cur, cur_rec, history)
        img_sha = hashlib.sha256(img.read_bytes()).hexdigest()
        b64 = base64.b64encode(img.read_bytes()).decode()
        msg = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + b64}}]}]
        t0 = time.perf_counter()
        r = chat(creds, "qwen3-vl-plus", msg, max_tokens=600, timeout=90)
        raw = r.get("content", "") if isinstance(r, dict) else str(r)
        (VLM_CALLS / ("%s_r%d.json" % (family, rnd))).write_text(json.dumps(
            dict(family=family, round=rnd, model="qwen3-vl-plus",
                 prompt=prompt, image_path=str(img), image_sha256=img_sha,
                 request=msg, response_raw=r, parsed_ok=None,
                 wall_s=round(time.perf_counter() - t0, 1)),
            ensure_ascii=False, default=str), encoding="utf-8")
        return raw
    return chat_fn


# ── 臂执行与落盘 ────────────────────────────────────────────────

def run_arm(arm, family):
    t0 = time.perf_counter()
    start = shared_start(family)
    session = cp.ProtocolSession(
        family, PARAM_BOXES[family], STARTS[family],
        evaluator=make_evaluator(family), budget=BUDGET)
    session.attach_start(start, source="shared_family_start")
    seed = cp.arm_seed(family, arm)
    entry = dict(arm=arm, family=family, seed=seed,
                 protocol="mainline-compare-v2", complete=False)
    try:
        if arm == "vlm":
            out = cp.run_vlm(session, make_vlm_chat(family))
        else:
            out = cp.run_numeric_arm(arm, session, seed)
    except Exception as e:
        entry.update(failed=f"{type(e).__name__}: {e}"[:300],
                     counts=dict(n_new_evals=session.n_new_evals,
                                 n_duplicates=session.n_duplicates,
                                 n_failures=session.n_failures),
                     records=session.records)
        return entry
    entry.update(
        final_kw=out.get("final_kw"), final_metrics=out.get("final_metrics"),
        final_D=out.get("final_D"), final_feasible=out.get("final_feasible"),
        complete=out.get("complete", False),
        counts=dict(n_new_evals=out.get("n_new_evals"),
                    n_duplicates=out.get("n_duplicates"),
                    n_failures=out.get("n_failures"),
                    n_feasible=out.get("n_feasible"), budget=BUDGET),
        rounds=out.get("rounds"), history=out.get("history"),
        cma_actual_seed=out.get("cma_actual_seed"),
        records=session.records, wall_s=round(time.perf_counter() - t0, 1))
    return entry


def write_runs(entry, replace_key=None):
    """落盘；同 (arm, family) 重复写时去重保留最新。"""
    lines = []
    if RUNS.exists():
        for line in RUNS.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                if (r.get("arm"), r.get("family")) != replace_key:
                    lines.append(line.rstrip("\n"))
            except Exception:
                continue
    lines.append(json.dumps(dict(provenance=provenance(), **entry),
                            ensure_ascii=False, default=str))
    RUNS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def done_keys():
    if not RUNS.exists():
        return set()
    ks = set()
    for line in RUNS.open(encoding="utf-8"):
        try:
            r = json.loads(line)
            if r.get("complete"):
                ks.add((r["arm"], r["family"]))
        except Exception:
            continue
    return ks


def write_protocol():
    from robust_design_lab.geometry import novel_structures as geo
    p = dict(
        version="mainline-compare-v2", date="2026-11-02", budget=BUDGET,
        etch_delta_um=ETCH_DELTA_UM,
        scenario_order=["nom", "etch+%.1f" % ETCH_DELTA_UM,
                        "etch-%.1f" % ETCH_DELTA_UM],
        fail_penalty=cp.FAIL_PENALTY,
        feasibility_window=cp.FEASIBILITY_WINDOW,
        objective="min objective; objective=D if feasible else FAIL_PENALTY",
        dedup="same canonical point within arm reuses record, no budget cost;"
              " start point shared at family level (measured once)",
        pocket_model="independent nominal pocket (bounds+margin) buffered "
                     "+/-delta (mitre); electrodes buffered from nominal "
                     "shapes; pocket anchored at nominal center",
        pocket_margin_um=geo.POCKET_MARGIN_UM,
        junction=dict(Lj_nH=12.31, EJ_GHz=EJ_GHZ, EC_GHz_per_fF=EC_GHZ_PER_F,
                      f01_formula="sqrt(8*EJ*EC)-EC"),
        evaluator="q3d_shape_sweep.run_q3d (Medium, max_passes=6, solved "
                  "project saved after solve); C_sigma via A_dual_pad branch",
        families={f: dict(param_box=PARAM_BOXES[f], start=STARTS[f],
                          note=DESIGN_NOTES[f]) for f in STARTS},
        seeds={"%s/%s" % (f, a): cp.arm_seed(f, a)
               for f in STARTS for a in ARMS},
        provenance=provenance(),
    )
    (OUT / "protocol.json").write_text(
        json.dumps(p, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")


def summarize():
    import collections
    agg = collections.defaultdict(list)
    for line in RUNS.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        d = r.get("final_D", (r.get("final_metrics") or {}).get("D"))
        if d is not None and r.get("complete"):
            agg[(r["family"], r["arm"])].append(d)
    print("\n汇总（可行终选 D）:")
    for (fam, arm), ds in sorted(agg.items()):
        print("  %-16s %-18s D=%.5f" % (fam, arm, min(ds)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=list(STARTS))
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--start-only", action="store_true",
                    help="只做共享起点实测（小批物理验证，不跑臂）")
    args = ap.parse_args()
    write_protocol()
    t0 = time.perf_counter()
    if args.start_only:
        for family in args.families:
            t1 = time.perf_counter()
            m = shared_start(family)
            print("start/%s: CΣ=%.1ffF f01=%.3fGHz D=%.5f (%.0fs)" % (
                family, m["C_sigma_fF"], m["f01_GHz"], m["D"],
                time.perf_counter() - t1), flush=True)
        print("总耗时 %.0fs" % (time.perf_counter() - t0))
        return
    for family in args.families:
        for arm in ARMS:
            if arm not in args.arms:
                continue
            if (arm, family) in done_keys():
                print("跳过 %s/%s" % (arm, family))
                continue
            print("===== %s / %s =====" % (arm, family), flush=True)
            entry = run_arm(arm, family)
            write_runs(entry, replace_key=(arm, family))
            print("→ %s/%s D=%s complete=%s (%.0fs)" % (
                arm, family, entry.get("final_D"), entry.get("complete"),
                entry.get("wall_s", 0)), flush=True)
    summarize()
    print("总耗时 %.0fs" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
