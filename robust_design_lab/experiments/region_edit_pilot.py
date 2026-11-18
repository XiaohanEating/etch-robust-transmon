# -*- coding: utf-8 -*-
"""第三步小试：区域边界编辑上的五组同权限对比（两起点）。

协议（docs/plans/new_structures_next_steps_2026-11-02.md 第三步）：
- 动作空间 = boundary_edit（op/seg/mag/width，三类机制不同的算子、
  32 段编码、结区保护、结构检查对所有方法相同）；
- 两起点（v1 终选，第二步已实测注入共享起点）× 五组：vlm_img /
  vlm_text / coordinate_descent / bayesian_opt / cma_es；
- 每组 3 轮 × 3 次新增评价（轮内提案作用于轮前结构快照，轮后统一
  按 objective 移动）——有图、无图、数值同预算同权限；
- 错配图探针（预指定状态：起点 B vlm_img 第 2 轮提示下喂起点 A 的
  图）与图像正控制（读段号位置）单独调用、费用另记；
- 本小试只验证接口、成本与候选多样性，不做视觉增益或方法优势的
  统计结论。

结果：results/region_edit_pilot_2026-11-03/（runs.jsonl 逐臂可续跑、
vlm_calls/、protocol.json、探针文件、summary.json）。
"""
import argparse
import base64
import hashlib
import json
import math
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance
from robust_design_lab.search import common_protocol as cp
from robust_design_lab.geometry import boundary_edit as be
from robust_design_lab.geometry import novel_structures as geo

OUT = ROOT / "results" / "region_edit_pilot_2026-11-03"
OUT.mkdir(parents=True, exist_ok=True)
VLM_CALLS = OUT / "vlm_calls"
VLM_CALLS.mkdir(exist_ok=True)
RUNS = OUT / "runs.jsonl"

BUDGET = 9
ROUNDS, PER_ROUND = 3, 3
ETCH_DELTA_UM = 0.5
EJ_GHZ, EC_GHZ_PER_F = 13.28, 19.36
MODEL = "qwen3-vl-plus"
ARMS = ("vlm_img", "vlm_text", "coordinate_descent", "bayesian_opt", "cma_es")

# 动作空间（int 维吸附）
EDIT_BOX = {"op": (0, 2, "int"), "seg": (0, be.N_SEG - 1, "int"),
            "mag": be.MAG_RANGE, "width": be.WIDTH_RANGE}

def _load_verified_metrics(path):
    """从第二步复验记录读全精度起点指标（不手填舍入值——2026-11-04
    审阅要求）；文件缺失或字段缺失直接报错。"""
    p = ROOT / "results" / "verify_candidates_v2_2026-11-03" / path
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d["metrics"] if "metrics" in d else d  # 兼容行结构
    return {k: float(m[k]) for k in ("D", "C_sigma_fF", "f01_GHz",
                                     "alpha_MHz")}


def _start_metrics():
    return {
        "A_annulus": _load_verified_metrics("verify_annulus_cma.json"),
        "B_vgap": _load_verified_metrics("verify_vgap_vlm.json"),
    }


STARTS = {
    "A_annulus": dict(
        make=lambda: geo.q_annulus(r_out=253.9, r_in=121.5, gap=33.8),
        note="v1 cma_es 终选（第二步复验记录全精度注入）",
        source="verify_candidates_v2_2026-11-03/verify_annulus_cma.json"),
    "B_vgap": dict(
        make=lambda: geo.m_variable_gap(H=480.0, depth=290.0, g0=32.0,
                                        g_edge=88.0),
        note="v1 vlm 终选（第二步复验记录全精度注入）",
        source="verify_candidates_v2_2026-11-03/verify_vgap_vlm.json"),
}


# ── 物理：结构 → 三场景指标 ──────────────────────────────────

def struct_fp(struct) -> str:
    def wkt(p):
        g = p.simplify(0.05)
        return g.wkt
    return "%08x" % zlib.crc32(
        (wkt(struct["pad_left"]) + "|" + wkt(struct["pad_right"])).encode())


def eval_struct(struct, delta=ETCH_DELTA_UM, max_passes=6,
                percent_error=None):
    """编辑后结构的三场景 Q3D（修正同步地）→ 指标；场景异常整体失败。

    未收敛场景按失败处理（不产生可用于终选的有效漂移——2026-11-04
    审阅约束：converged=False 的记录不得带有效 D 入库）。"""
    from robust_design_lab.geometry.novel_structures import (
        struct_to_options, etch_variant)
    from robust_design_lab.experiments.q3d_shape_sweep import run_q3d
    scen = {}
    for tag, st in (("nom", struct), ("ep", etch_variant(struct, +delta)),
                    ("em", etch_variant(struct, -delta))):
        scen[tag] = run_q3d(struct_to_options(st), "NovelTwoPadTransmon",
                            "A_dual_pad", max_passes=max_passes,
                            percent_error=percent_error)
    not_conv = [t for t, s in scen.items() if s.get("converged") is False]
    if not_conv:
        raise RuntimeError("场景未收敛: %s（按失败处理，不入有效漂移）"
                           % ",".join(not_conv))

    def f01(c):
        ec = EC_GHZ_PER_F / c
        return math.sqrt(8 * EJ_GHZ * ec) - ec

    f0 = f01(scen["nom"]["C_sigma_fF"])
    D = max(abs(math.log(f01(scen["ep"]["C_sigma_fF"]) / f0)),
            abs(math.log(f01(scen["em"]["C_sigma_fF"]) / f0)))
    return dict(D=D, C_sigma_fF=scen["nom"]["C_sigma_fF"], f01_GHz=f0,
                alpha_MHz=-EC_GHZ_PER_F / scen["nom"]["C_sigma_fF"] * 1e3,
                scenarios=scen,
                area_um2=dict(L=round(struct["pad_left"].area),
                              R=round(struct["pad_right"].area)))


def make_evaluator(cur_state):
    """evaluator 闭包：动作作用于**当前结构快照**（轮前由 context_fn
    固化）；无效编辑抛 InvalidEditError（协议层罚分、无求解）。"""

    def evaluator(kw):
        op, seg, mag, width = be.action_from_kw(kw)
        s2 = be.apply_edit(cur_state["struct"], op, seg, mag, width)
        m = eval_struct(s2)
        m["action"] = be.action_describe(kw)
        m["struct_fp"] = struct_fp(s2)
        return m
    return evaluator


# ── 图与提示 ──────────────────────────────────────────────────

def render_segments(struct, path):
    """当前结构图：岛、孔、结线、外/内环段中点编号（白底半透明框）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    for isl, poly in (("L", struct["pad_left"]), ("R", struct["pad_right"])):
        ax.fill(*poly.exterior.xy, color="#2196F3", alpha=0.85,
                edgecolor="#0D47A1", linewidth=0.8, zorder=3)
        for hole in poly.interiors:
            ax.fill(*hole.xy, color="white", zorder=4)
        for ring, line in (("outer", poly.exterior),) + tuple(
                ("inner", h) for h in poly.interiors):
            pts = be._densified_ccw(line)
            for region, seg in enumerate(be._ring_segments(pts)):
                mx, my = seg["mid"]
                ax.plot([mx], [my], "o", ms=3, color="#FF9800", zorder=6)
                ax.annotate(f"{isl}{ring[0]}{region}", (mx, my),
                            fontsize=6.5, zorder=7,
                            bbox=dict(boxstyle="round,pad=0.15",
                                      fc="white", ec="none", alpha=0.75))
    x, y = struct["junction"].xy
    ax.plot(x, y, color="#F44336", linewidth=2.5, zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


PROMPT_RULES = """设计规则：目标 C_Σ∈[70,105]fF 且 f01∈[4.5,5.5]GHz 下最小化刻蚀漂移 D（名义+刻蚀±0.5μm 三场景 Q3D 实测的 max|log(f_δ/f_0)|）。出窗或无效动作按罚分 10.0 计，不会因 D 低被采纳。
当前为双岛电容结构（两片内部各自连通的电极，中间结线相连，红色为结线，其上下 50μm 为保护区不可编辑）。
每次编辑动作选择一个岛的一段边界施加一个算子：
- op："bulge"=局部法向外凸/内凹（改曲率）；"notch"=边界开圆槽（金属减少，增加工作边）；"tab"=边界外伸矩形舌（金属增加，增加对置工作边）。
- island："L"/"R"；ring："outer"（外缘）/"inner"（内孔缘；无孔岛该环无效）；region：0-7（该环按弧长等分的段号）。
- mag：0.2-1.0 幅度比例；width：0.3-1.0 影响宽度（bulge/tab）。
单次动作面积变化上限 30%，且编辑后须通过连通/最小特征/短接检查，否则罚分。"""


def build_prompt(struct, cur_rec, history, seg_rows):
    hist = ""
    if history:
        lines = ["  %s → D=%s%s%s" % (
            be.action_describe(h["kw"]),
            ("%.5f" % h["D"]) if h["D"] is not None else "无效/失败",
            "" if h["feasible"] else "（出窗，罚分）",
            "" if not h["error"] else f" [{h['error'][:40]}]")
            for h in history[-6:]]
        hist = "\n历史提案实测：\n" + "\n".join(lines)
    m = cur_rec["metrics"]
    return (PROMPT_RULES
            + "\n当前结构各段位置（段中点坐标 μm，坐标系与图一致）：\n"
            + "\n".join("  " + r for r in seg_rows)
            + "\n当前实测：C_Σ=%.1ffF f01=%.3fGHz D=%.5f%s" % (
                m["C_sigma_fF"], m["f01_GHz"], m["D"],
                "" if cur_rec["feasible"] else "（出窗，罚分）")
            + hist
            + "\n提出 3 个编辑动作（可不同岛不同算子；优先尝试低漂移方向）。"
            '\n只返回 JSON 数组：[{"op":"bulge","island":"L","ring":'
            '"outer","region":3,"mag":0.6,"width":0.5,"reason":"简短理由"}]')


def parse_vlm_actions(raw):
    """模型回复 → 动作 kw 列表（非法项跳过）。"""
    import re
    text = re.sub(r"```json?\n?", "", raw or "").replace("```", "").strip()
    arr = cp.json_loads_lenient(text)
    if isinstance(arr, dict):
        arr = [arr]
    out = []
    for p in arr if isinstance(arr, list) else []:
        try:
            out.append(be.action_kw(str(p["op"]), str(p["island"]),
                                    str(p["ring"]), int(p["region"]),
                                    float(p["mag"]), float(p["width"])))
        except Exception:
            continue
    if not out:
        raise ValueError("回复中无合法动作")
    return out


def make_vlm_proposer(start_name, arm, cur_state, creds, round_override=None):
    """VLM proposer（有图/无图共用；arm=vlm_img 时附图）。

    错图探针（2026-11-04 审阅修正）：round_override 提供的错图文件
    **只读**——不重新渲染（旧版把正确结构渲染到错图路径，覆盖了
    本要发送的图片）；发送前记录实发字节哈希与身份标注。"""
    from robust_design_lab.vlm.aliyun import chat

    def propose(rnd, cur_kw, cur_rec, history):
        struct = cur_state["struct"]
        seg_rows = be.segment_table(struct)
        prompt = build_prompt(struct, cur_rec, history, seg_rows)
        img_path = None
        img_note = None
        msg_content = [{"type": "text", "text": prompt}]
        img_sha = None
        if arm == "vlm_img":
            img_override = round_override and round_override.get(rnd)
            if img_override:
                img_path = Path(img_override)  # 错图只读，不覆盖
                img_note = "MISMATCH_PROBE(错图，勿渲染)"
                if not img_path.exists():
                    raise FileNotFoundError(f"错图不存在: {img_path}")
            else:
                img_path = OUT / ("%s_%s_r%d.png" % (start_name, arm, rnd))
                render_segments(struct, img_path)
            img_sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            msg_content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + b64}})
        msg = [{"role": "user", "content": msg_content}]
        t0 = time.perf_counter()
        r = chat(creds, MODEL, msg, max_tokens=900, timeout=120)
        raw = r.get("content", "") if isinstance(r, dict) else str(r)
        parsed = None
        try:
            parsed = parse_vlm_actions(raw)
        except Exception:
            parsed = None
        (VLM_CALLS / ("%s_%s_r%d.json" % (start_name, arm, rnd))).write_text(
            json.dumps(dict(start=start_name, arm=arm, round=rnd,
                            model=MODEL, prompt=prompt,
                            image=str(img_path) if img_path else None,
                            image_note=img_note,
                            image_sha256=img_sha,
                            sent_bytes_sha256=img_sha,  # 实发字节=文件字节
                            request=msg,
                            response_raw=r, n_parsed=len(parsed) if parsed
                            else 0,
                            wall_s=round(time.perf_counter() - t0, 1)),
                       ensure_ascii=False, default=str), encoding="utf-8")
        if not parsed:
            raise ValueError("VLM 无合法动作提案")
        return parsed
    return propose


# ── 臂执行 ────────────────────────────────────────────────────

def run_arm(arm, start_name, mismatch=False):
    t0 = time.perf_counter()
    spec = STARTS[start_name]
    cur_state = dict(struct=spec["make"]())
    start_fp = struct_fp(cur_state["struct"])
    session = cp.ProtocolSession(
        "edit_pilot", EDIT_BOX,
        start_kw=dict(op=1, seg=4, mag=0.6, width=0.5),
        evaluator=make_evaluator(cur_state), budget=BUDGET)
    session.attach_start(_start_metrics()[start_name],
                         source="step2_verified_full_precision")
    seed = cp.arm_seed("edit_" + start_name, arm)

    def context_fn(rnd):
        cur_state["struct"] = cur_state.get("next_struct",
                                            cur_state["struct"])
        cur_state["fp"] = struct_fp(cur_state["struct"])
        return cur_state["fp"]

    def on_round_end(rnd, adopted_kw):
        if adopted_kw is not None:
            op, seg, mag, width = be.action_from_kw(adopted_kw)
            cur_state["next_struct"] = be.apply_edit(
                cur_state["struct"], op, seg, mag, width)
        else:
            cur_state["next_struct"] = cur_state["struct"]

    round_override = None
    if mismatch and arm == "vlm_img" and start_name == "B_vgap":
        # 错配探针：第 2 轮喂起点 A 的图（预指定状态，正常执行照旧）
        a_struct = STARTS["A_annulus"]["make"]()
        wrong = OUT / "mismatch_probe_A_structure_r2.png"
        render_segments(a_struct, wrong)
        round_override = {2: wrong}

    if arm in ("vlm_img", "vlm_text"):
        from robust_design_lab.vlm.aliyun import load_credentials
        prop = make_vlm_proposer(start_name, arm, cur_state,
                                 load_credentials(), round_override)
    elif arm == "coordinate_descent":
        import numpy as np
        prop = cp.make_coord_proposer(session, np.random.default_rng(seed))
    elif arm == "cma_es":
        prop = cp.make_cma_proposer(session, seed)
    elif arm == "bayesian_opt":
        prop = cp.make_bo_proposer(session, seed)
    else:
        raise KeyError(arm)

    try:
        out = cp.run_round_based(session, prop, rounds=ROUNDS,
                                 per_round=PER_ROUND, context_fn=context_fn,
                                 on_round_end=on_round_end)
    except Exception as e:
        return dict(arm=arm, start=start_name, complete=False,
                    failed=f"{type(e).__name__}: {e}"[:300],
                    records=session.records,
                    wall_s=round(time.perf_counter() - t0, 1))
    # 终选完整结构：按采纳链从起点重建（不只保存最后一个动作——
    # 2026-11-04 审阅要求候选身份含完整几何）
    final_struct = spec["make"]()
    for a in out["adopted"]:
        op, seg, mag, width = be.action_from_kw(a["kw"])
        final_struct = be.apply_edit(final_struct, op, seg, mag, width)
    entry = dict(
        arm=arm, start=start_name, seed=seed, complete=out["complete"],
        start_struct_fp=start_fp,
        final_action=out["final_action"],
        final_action_desc=be.action_describe(out["final_action"]),
        final_D=out.get("final_D"), final_feasible=out.get("final_feasible"),
        final_metrics=out.get("final_metrics"),
        final_struct_wkt=dict(
            pad_left=final_struct["pad_left"].simplify(0.05).wkt,
            pad_right=final_struct["pad_right"].simplify(0.05).wkt,
            junction=list(final_struct["junction"].coords)),
        n_adopted_struct_rebuild_fp=struct_fp(final_struct),
        adopted=out["adopted"], rounds=out["rounds"], history=out["history"],
        counts=dict(n_new_evals=out["n_new_evals"],
                    n_duplicates=out["n_duplicates"],
                    n_failures=out["n_failures"], n_invalid=out["n_invalid"],
                    n_feasible=out["n_feasible"], budget=BUDGET),
        records=session.records, wall_s=round(time.perf_counter() - t0, 1))
    return entry


def write_runs(entry):
    lines = []
    if RUNS.exists():
        for line in RUNS.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                if (r.get("arm"), r.get("start")) != (entry["arm"],
                                                      entry["start"]):
                    lines.append(line.rstrip("\n"))
            except Exception:
                continue
    lines.append(json.dumps(dict(provenance=provenance(), **entry),
                            ensure_ascii=False, default=str))
    RUNS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def done_keys():
    ks = set()
    if RUNS.exists():
        for line in RUNS.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("complete"):
                    ks.add((r["arm"], r["start"]))
            except Exception:
                continue
    return ks


# ── 探针：图像正控制与错配对比（单独调用，费用另记） ──────────

def probes():
    """正控制：给编辑前后两图，问差异位置（验证模型真的读图）。
    错配对比结果在主循环结束后从落盘调用对里汇总（比较同轮正确图与
    错配图的响应差异）。"""
    from robust_design_lab.vlm.aliyun import load_credentials, chat
    creds = load_credentials()
    results = []
    for start_name in STARTS:
        s0 = STARTS[start_name]["make"]()
        edited = be.apply_edit(s0, "notch", be.seg_encode("R", "outer", 4),
                               0.8, 0.5)
        p0 = OUT / f"pc_{start_name}_orig.png"
        p1 = OUT / f"pc_{start_name}_edited.png"
        render_segments(s0, p0)
        render_segments(edited, p1)
        prompt = ("图1与图2是同一双岛结构的编辑前后图。编辑发生在一个岛"
                  "的一段边界上。请回答：编辑发生在哪个岛（L/R）、哪条环"
                  "（outer/inner）、大约哪几个 region 编号、是什么类型的"
                  "编辑（金属增加/减少）。只给一句话。")
        msg = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(
                    p0.read_bytes()).decode()}},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(
                    p1.read_bytes()).decode()}}]}]
        r = chat(creds, MODEL, msg, max_tokens=200, timeout=90)
        raw = r.get("content", "") if isinstance(r, dict) else str(r)
        truth = "R 岛 outer 环段 4 附近，notch（金属减少）"
        results.append(dict(start=start_name, truth=truth, response=raw,
                            prompt=prompt))
        (VLM_CALLS / f"positive_control_{start_name}.json").write_text(
            json.dumps(results[-1], ensure_ascii=False, default=str),
            encoding="utf-8")
    (OUT / "positive_control.json").write_text(
        json.dumps(results, ensure_ascii=False, default=str, indent=1),
        encoding="utf-8")
    return results


def write_protocol():
    p = dict(
        version="region-edit-pilot-v1", date="2026-11-03",
        budget=BUDGET, rounds=ROUNDS, per_round=PER_ROUND,
        edit_box={k: list(v) for k, v in EDIT_BOX.items()},
        edit_ops=be.EDIT_OPS, n_seg=be.N_SEG,
        protect_r_um=be.PROTECT_R_UM, max_area_change=be.MAX_AREA_CHANGE,
        etch_delta_um=ETCH_DELTA_UM,
        objective="min objective; D if feasible else FAIL_PENALTY",
        starts={k: dict(note=v["note"], source=v["source"],
                        metrics=_start_metrics()[k]) for k, v in
                STARTS.items()},
        seeds={"%s/%s" % (s, a): cp.arm_seed("edit_" + s, a)
               for s in STARTS for a in ARMS},
        probes=dict(positive_control="每起点一次：编辑前后两图问答（读图"
                                     "位置核对）",
                    mismatch="起点 B vlm_img 第 2 轮喂起点 A 的结构图"
                             "（round_override 预指定），响应落盘对比"),
        note="小试只验证接口/成本/候选多样性，不做视觉增益或方法优势"
             "统计结论；数值臂与 VLM 同动作空间同预算同检查。",
        provenance=provenance())
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
        if r.get("complete"):
            agg[(r["start"], r["arm"])].append(
                dict(D=r.get("final_D"),
                     invalid=r["counts"]["n_invalid"],
                     dup=r["counts"]["n_duplicates"],
                     adopted=len(r["adopted"])))
    print("\n汇总（可行终选 D / 无效 / 重复 / 采纳数）:")
    for (s, a), rows in sorted(agg.items()):
        for row in rows:
            print("  %-11s %-17s D=%s invalid=%d dup=%d adopted=%d" % (
                s, a, ("%.5f" % row["D"]) if row["D"] is not None else "—",
                row["invalid"], row["dup"], row["adopted"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", nargs="*", default=list(STARTS))
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--probes-only", action="store_true",
                    help="只跑图像正控制探针（单独调用）")
    args = ap.parse_args()
    write_protocol()
    t0 = time.perf_counter()
    if args.probes_only:
        probes()
        print("探针完成")
        return
    done = done_keys()
    for start_name in args.starts:
        for arm in ARMS:
            if arm not in args.arms:
                continue
            if (arm, start_name) in done:
                print("跳过 %s/%s" % (arm, start_name))
                continue
            print("===== %s / %s =====" % (arm, start_name), flush=True)
            # 错配探针预指定：B_vgap/vlm_img 第 2 轮喂 A 结构图
            # （mismatch=True 现已实际传递——旧版漏传，审阅反例）
            entry = run_arm(arm, start_name,
                            mismatch=(start_name == "B_vgap"))
            write_runs(entry)
            print("→ %s/%s D=%s complete=%s invalid=%d (%.0fs)" % (
                arm, start_name, entry.get("final_D"),
                entry.get("complete"),
                entry.get("counts", {}).get("n_invalid", -1),
                entry.get("wall_s", 0)), flush=True)
    summarize()
    print("总耗时 %.0fs" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
