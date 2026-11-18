# -*- coding: utf-8 -*-
"""第三步：未优化可行起点的受控新形状搜索（decisions 2026-11-04(四)）。

- 两起点（环形默认=内缘工作边机制；交指默认=对置工作边机制，均未
  经搜索优化）× 三方法（vlm_img 空间图+表 / vlm_text 同文字无图 /
  cma_es 混合动作空间编码）× 共享起点+9 次新增评价；
- 搜索用 0.5% 粗筛档（两档策略）；优胜者 0.1% 复验由 --refine-winners
  单独执行；
- 提示含逐段空间表（spatial_feedback，所有方法同信息）与 op×ring
  金属增减语义表（修正旧提示与实现不一致）；
- 记录形状差异（对称差面积比/孔数/间隙分布）与合格独立几何数。

复用 boundary_edit / common_protocol / spatial_feedback；断点续跑。
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance
from robust_design_lab.search import common_protocol as cp
from robust_design_lab.geometry import boundary_edit as be
from robust_design_lab.geometry import novel_structures as geo
from robust_design_lab.geometry import spatial_feedback as sf

OUT = ROOT / "results" / "fresh_start_search_2026-11-04"
OUT.mkdir(parents=True, exist_ok=True)
VLM_CALLS = OUT / "vlm_calls"
VLM_CALLS.mkdir(exist_ok=True)

BUDGET = 9
ROUNDS, PER_ROUND = 3, 3
MODEL = "qwen3-vl-plus"
ARMS = ("vlm_img", "vlm_text", "rule_based", "cma_es")
SEARCH_PCT = [0.5]     # 求解精度档（percent_error），CLI --pct 覆盖
MOVE_POLICY = ["greedy"]  # 移动策略，CLI --policy 覆盖


def passes_for(pct):
    """精度档与自适应轮数上限绑定（2026-11-06 审阅反例：0.1% 目标
    配 6 轮上限必然不收敛，第四步批次 7/7 评价全部白跑）。
    0.1% 用已有 0.1% 复验成功的 20 轮；新增精度档先单点实测成本。"""
    return 20 if pct < 0.5 else 6

EDIT_BOX = {"op": (0, 2, "int"), "seg": (0, be.N_SEG - 1, "int"),
            "mag": be.MAG_RANGE, "width": be.WIDTH_RANGE}

STARTS = {
    "annulus_default": dict(
        make=lambda: geo.q_annulus(),
        note="环形默认参数（250/115/28；内缘工作边机制；未搜索优化）"),
    "ellipse_idc_default": dict(
        make=lambda: geo.q_ellipse_idc(),
        note="椭圆垫交指默认参数（对置工作边机制；未搜索优化）"),
    "branch_comb_default": dict(
        make=lambda: geo.m_branch_comb(),
        note="分支梳默认参数（分支桨工作边机制；资格 91.5fF/4.53GHz；未搜索）"),
    "pad_ref": dict(
        make=lambda: _simple_pad(),
        note="简单双岛参照尺寸（734×218；B 阶段名义校准合格 91.61fF；"
             "未做漂移优化——G_notch 显示有改进空间）"),
}


def _simple_pad():
    from robust_design_lab.experiments.verify_candidates import \
        simple_dual_pad
    return simple_dual_pad(w=734.0, h=218.0)

# op×ring 金属增减语义（修正旧提示与实现不一致——计划第三步要求）
OP_SEMANTICS = """算子语义（金属增减按环明确）：
- bulge：外环=金属外凸（面积增）；内环=孔局部扩大（面积减，内缘向金属内退）。
- notch：外环=外缘开圆槽（面积减）；内环=孔挖圆缺（面积减）。
- tab：外环=外伸舌（面积增）；内环=金属伸入孔（面积增）。"""

PROMPT = """设计规则：目标 C_Σ∈[70,105]fF 且 f01∈[4.5,5.5]GHz 下最小化刻蚀漂移 D（名义+刻蚀±0.5μm 三场景 max|log(f_δ/f_0)|）。出窗或无效动作罚 10.0。
当前为双岛电容结构（两电极+中间结线；结线上下 50μm 保护圆内不可编辑；编辑的几何变化范围不得侵入保护圆）。
{semantics}
逐段空间信息表（局部间隙=到对置电极最近距离；到地边界=到地面口袋；曲率=离散曲率均值；对置≈最近对面段；无内环的段无效）：
{table}
空间提示：刻蚀使全部边界同步内缩 0.5μm——局部间隙小的对置段相对变化大；曲率高/转折多处面积相对损失大；到地边界近的段对地面刻蚀敏感。
可用动作：op∈bulge/notch/tab；island L/R；ring outer/inner；region 0-7；mag 0.2-1.0；width 0.3-1.0。单次面积变化≤30%。
{hist}提出 3 个编辑动作。只返回 JSON 数组：[{{"op":"bulge","island":"L","ring":"outer","region":3,"mag":0.6,"width":0.5,"reason":"引用空间量"}}]"""


def struct_fp(struct):
    """几何指纹用完整 wkt（2026-11-06 审阅：未经登记的几何简化不能
    进入身份判定——simplify 会让不同结构碰撞出同一指纹）。"""
    import zlib
    return "%08x" % zlib.crc32((struct["pad_left"].wkt + "|"
                                + struct["pad_right"].wkt).encode())


class UnconvergedError(RuntimeError):
    """场景未收敛（按失败处理）；已取得的场景转存失败档案。"""

    def __init__(self, msg, partial=None, path=None):
        super().__init__(msg)
        self.partial = partial
        self.path = path


def eval_struct(struct, max_passes=None, percent_error=None):
    """三场景求解。max_passes 缺省由精度档推导（passes_for 绑定）。
    未收敛：已取得场景先落盘再抛 UnconvergedError（2026-11-06 审阅
    A5——失败的求解时间不白花，矩阵留待分析）。"""
    import math
    if max_passes is None:
        max_passes = passes_for(percent_error if percent_error
                                is not None else 0.5)
    from robust_design_lab.geometry.novel_structures import (
        struct_to_options, etch_variant)
    from robust_design_lab.experiments.q3d_shape_sweep import run_q3d
    scen = {}
    try:
        for tag, st in (("nom", struct), ("ep", etch_variant(struct, 0.5)),
                        ("em", etch_variant(struct, -0.5))):
            scen[tag] = run_q3d(struct_to_options(st), "NovelTwoPadTransmon",
                                "A_dual_pad", max_passes=max_passes,
                                percent_error=percent_error)
    finally:
        not_conv = [t for t, s in scen.items()
                    if s.get("converged") is not True]  # None/缺失拒收
        if not_conv:
            import time as _t
            p = OUT / ("failed_eval_%s_%d.json" % (
                struct_fp(struct), int(_t.time())))
            slim = {t: {k: v for k, v in s.items()
                        if k != "capacitance_all_passes_fF"}
                    for t, s in scen.items()}
            p.write_text(json.dumps(dict(fp=struct_fp(struct),
                                         max_passes=max_passes,
                                         percent_error=percent_error,
                                         unconverged=not_conv,
                                         scenarios=slim),
                                    ensure_ascii=False, default=str),
                         encoding="utf-8")
    if not_conv:
        raise UnconvergedError("场景未收敛: %s" % ",".join(not_conv),
                               partial=scen, path=str(p))

    def f01(c):
        ec = 19.36 / c
        return math.sqrt(8 * 13.28 * ec) - ec

    f0 = f01(scen["nom"]["C_sigma_fF"])
    D = max(abs(math.log(f01(scen["ep"]["C_sigma_fF"]) / f0)),
            abs(math.log(f01(scen["em"]["C_sigma_fF"]) / f0)))
    return dict(D=D, C_sigma_fF=scen["nom"]["C_sigma_fF"], f01_GHz=f0,
                alpha_MHz=-19.36 / scen["nom"]["C_sigma_fF"] * 1e3,
                scenarios=scen)


def make_evaluator(cur_state):
    pct = SEARCH_PCT[0]

    def evaluator(kw):
        op, seg, mag, width = be.action_from_kw(kw)
        s2 = be.apply_edit(cur_state["struct"], op, seg, mag, width)
        m = eval_struct(s2, percent_error=pct)  # 轮数上限由精度档绑定
        m["action"] = be.action_describe(kw)
        m["struct_fp"] = struct_fp(s2)
        return m
    return evaluator


RUN_SEED = [None]  # 运行种子（CLI --seed 覆盖；None=arm_seed 推导）


def arm_seed_for(start_name, arm):
    return RUN_SEED[0] if RUN_SEED[0] is not None \
        else cp.arm_seed("fresh_" + start_name, arm)


def runs_path():
    """分批独立落盘，运行身份纳入精度、策略、种子与驱动源码版本
    （自动取本文件 SHA256 前 8 位；2026-11-07 复核：仅驱动哈希时
    不同种子仍互相覆盖）。旧混合文件保留为历史档案。"""
    import hashlib
    src = Path(__file__).read_bytes()
    ver = hashlib.sha256(src).hexdigest()[:8]
    seed = RUN_SEED[0] if RUN_SEED[0] is not None else "dflt"
    return OUT / ("runs_p%s_%s_v%s_s%s.jsonl" % (SEARCH_PCT[0],
                                                 MOVE_POLICY[0], ver, seed))


def shape_delta(base, new):
    """形状差异记录：对称差面积比、孔数、间隙分布。"""
    def profile(struct):
        lg, _ = sf.local_gap_profile(struct)
        g = lg["gap"]
        return dict(gap_min=round(float(g.min()), 1),
                    gap_median=round(float(np_median(g)), 1),
                    gap_max=round(float(g.max()), 1))
    return dict(
        symdiff_area_ratio=round(
            (new["pad_left"].symmetric_difference(base["pad_left"]).area
             + new["pad_right"].symmetric_difference(
                 base["pad_right"]).area)
            / (base["pad_left"].area + base["pad_right"].area), 4),
        holes=dict(base=sum(len(p.interiors) for p in
                            (base["pad_left"], base["pad_right"])),
                   new=sum(len(p.interiors) for p in
                           (new["pad_left"], new["pad_right"]))),
        base=profile(base), new=profile(new))


def np_median(a):
    import numpy as np
    return float(__import__("numpy").median(a))


def build_prompt(struct, cur_rec, history, seg_rows):
    hist = ""
    if history:
        lines = ["  %s → D=%s%s" % (
            be.action_describe(h["kw"]),
            ("%.5f" % h["D"]) if h["D"] is not None else "无效/失败",
            "" if h["feasible"] else "（出窗罚分）")
            for h in history[-6:]]
        hist = "历史提案实测：\n" + "\n".join(lines) + "\n"
    m = cur_rec["metrics"]
    return PROMPT.format(
        semantics=OP_SEMANTICS,
        table="\n".join("  " + r for r in seg_rows),
        hist=hist
    ) + "\n当前实测：C_Σ=%.1ffF f01=%.3fGHz D=%.5f%s" % (
        m["C_sigma_fF"], m["f01_GHz"], m["D"],
        "" if cur_rec["feasible"] else "（出窗罚分）")


def make_vlm_proposer(start_name, arm, cur_state, creds,
                      session=None):
    import base64
    import hashlib
    from robust_design_lab.vlm.aliyun import chat

    def propose(rnd, cur_kw, cur_rec, history):
        struct = cur_state["struct"]
        prompt = build_prompt(struct, cur_rec, history,
                              sf.spatial_table(struct))
        content = [{"type": "text", "text": prompt}]
        img_sha = None
        img_path = None
        if arm == "vlm_img":
            img_path = OUT / f"{start_name}_{arm}_r{rnd}.png"
            sf.render_spatial(struct, img_path, title=start_name)
            img_sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + b64}})
        msg = [{"role": "user", "content": content}]
        t0 = time.perf_counter()
        r = chat(creds, MODEL, msg, max_tokens=900, timeout=120)
        raw = r.get("content", "") if isinstance(r, dict) else str(r)
        (VLM_CALLS / f"{start_name}_{arm}_r{rnd}.json").write_text(
            json.dumps(dict(start=start_name, arm=arm, round=rnd,
                            model=MODEL, prompt=prompt,
                            image=str(img_path) if img_path else None,
                            sent_image_sha256=img_sha, request=msg,
                            response_raw=r,
                            wall_s=round(time.perf_counter() - t0, 1)),
                       ensure_ascii=False, default=str), encoding="utf-8")
        import re
        from robust_design_lab.experiments.spatial_probe import parse_actions
        acts, dropped = parse_actions(raw)
        # 丢弃项显式落盘（不静默；2026-11-07 复核 P1：曾 10 项静默
        # 丢弃、3 项 region 越界被错译到别的位置）
        if dropped:
            (VLM_CALLS / f"{start_name}_{arm}_r{rnd}_dropped.json").write_text(
                json.dumps(dropped, ensure_ascii=False, indent=1),
                encoding="utf-8")
        # 丢弃动作逐个占登记尝试（与非法动作/求解失败同权——
        # 2026-11-07 复核：仅落盘可见不构成统一失败规则）
        if session is not None:
            for i in range(len(dropped)):
                try:
                    session.register_parse_failure(
                        f"{arm}_r{rnd}drop{i}")
                except cp.BudgetExhausted:
                    break
        if not acts:
            raise ValueError("VLM 无合法动作提案（丢弃 %d 项已占预算）"
                             % len(dropped))
        return [a[0] for a in acts]
    return propose


def start_eval(start_name):
    """起点共享评价（与搜索同精度档，落盘复用；兼容已存 0.1% 锚点）。"""
    pct = SEARCH_PCT[0]
    cur_fp = struct_fp(STARTS[start_name]["make"]())
    cache = OUT / f"start_{start_name}_p{pct}.json"
    legacy = OUT / (f"start_fine_{start_name}.json" if pct == 0.1
                    else f"start_{start_name}.json")
    for p in (cache, legacy):
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            # 缓存核对：几何指纹与精度档一致才复用（2026-11-07 复核：
            # 旧缓存不核对，起点参数变更后仍返回旧值）
            if d.get("struct_fp") == cur_fp and d.get("pct") == pct:
                return d["metrics"]  # 指纹与精度都一致才复用
    t0 = time.perf_counter()
    m = eval_struct(STARTS[start_name]["make"](), percent_error=pct)
    cache.write_text(json.dumps(dict(start=start_name, pct=pct,
                                     struct_fp=cur_fp, metrics=m,
                                     wall_s=round(time.perf_counter() - t0, 1),
                                     provenance=provenance()),
                                ensure_ascii=False, default=str),
                     encoding="utf-8")
    return m


RULE_POOL_SIZE = 5  # 规则候选池（轮内名额 3，重复点由缓存顺延）


def make_rule_proposer(cur_state):
    """物理规则 proposer（D 首批第四方法）：与 VLM/数值同信息
    （segment_metrics 空间表）同预算，事前固定规则、无模型——
    检验"简单规则是否已经足够"（计划 D 的退出路径之一）。

    规则（看结果前固定，按优先级）：
    1 最窄局部间隙合法外环段 bulge（扩宽最窄对置间隙）；
    2 最高曲率合法段 notch（削减刻蚀敏感的高曲率区）；
    3 最窄间隙的对置段 bulge（从另一侧扩宽同一间隙）；
    4 到地最近的合法段 notch（对地缘开槽——G_notch 同型动作）；
    5 次窄间隙段 bulge。
    合法段过滤含 apply_edit 试探（结区保护/结构检查/±δ 变体），
    无效换下一名；每轮基于当前结构重算。"""
    from robust_design_lab.geometry import spatial_feedback as sf

    def rule_candidates(struct):
        rows = [r for r in sf.segment_metrics(struct)
                if r.get("valid") and r["ring"] == "outer"]
        by_gap = sorted(rows, key=lambda r: r["gap_min"])
        by_curv = sorted(rows, key=lambda r: -r["curvature_mean"])
        by_g2g = sorted(rows, key=lambda r: r["gap_to_ground"])
        plan = [
            ("bulge", by_gap[0], 0.6, 0.5),
            ("notch", by_curv[0], 0.5, 0.5),
            ("bulge", next((r for r in rows
                            if r["seg"] == by_gap[0]["facing_seg"]),
                           by_gap[0]), 0.6, 0.5),
            ("notch", by_g2g[0], 0.5, 0.5),
            ("bulge", by_gap[1] if len(by_gap) > 1 else by_gap[0], 0.6, 0.5),
        ]
        # 2026-11-07 复核 P1：去掉 apply_edit 预滤——非法动作与其他
        # 方法（模型/CMA）同权在协议层受罚，不享有合法性特权
        out, seen = [], set()
        for op, r, mag, w in plan:
            kw = be.action_kw(op, r["island"], "outer", r["region"],
                              mag, w)
            key = (kw["op"], kw["seg"], kw["mag"], kw["width"])
            if key in seen:
                continue
            seen.add(key)
            out.append(kw)
        return out

    def propose(rnd, cur_kw, cur_rec, history):
        cands = rule_candidates(cur_state["struct"])[:RULE_POOL_SIZE]
        if not cands:
            raise ValueError("规则候选池为空（全部段不合法）")
        return cands
    return propose


def run_arm(arm, start_name):
    t0 = time.perf_counter()
    spec = STARTS[start_name]
    base_struct = spec["make"]()
    cur_state = dict(struct=base_struct)
    session = cp.ProtocolSession(
        "fresh_search", EDIT_BOX, start_kw=dict(op=1, seg=4, mag=0.6,
                                                width=0.5),
        evaluator=make_evaluator(cur_state), budget=BUDGET)
    session.attach_start(start_eval(start_name), source="shared_fresh_start")
    seed = arm_seed_for(start_name, arm)

    snapshots = {}  # 轮前结构快照：终选几何按最优记录的轮前状态重建

    def context_fn(rnd):
        cur_state["struct"] = cur_state.get(
            "next_struct", cur_state["struct"])
        snapshots[rnd] = cur_state["struct"]
        return struct_fp(cur_state["struct"])

    def on_round_end(rnd, adopted_kw):
        if adopted_kw is not None:
            op, seg, mag, width = be.action_from_kw(adopted_kw)
            cur_state["next_struct"] = be.apply_edit(
                cur_state["struct"], op, seg, mag, width)
        else:
            cur_state["next_struct"] = cur_state["struct"]

    if arm in ("vlm_img", "vlm_text"):
        from robust_design_lab.vlm.aliyun import load_credentials
        prop = make_vlm_proposer(start_name, arm, cur_state,
                                 load_credentials(), session=session)
    elif arm == "rule_based":
        prop = make_rule_proposer(cur_state)
    else:
        prop = cp.make_cma_proposer(session, seed)

    try:
        out = cp.run_round_based(session, prop, rounds=ROUNDS,
                                 per_round=PER_ROUND, context_fn=context_fn,
                                 on_round_end=on_round_end,
                                 move_policy=MOVE_POLICY[0])
    except Exception as e:
        return dict(arm=arm, start=start_name, complete=False,
                    failed=f"{type(e).__name__}: {e}"[:300],
                    records=session.records,
                    wall_s=round(time.perf_counter() - t0, 1))
    # 终选几何 = 产生最优记录的几何（按其轮前快照+动作重建）——
    # 不用 forced 累积末态替代最好状态（2026-11-06 审阅 A3 反例）
    fm = out.get("final_metrics")
    src = next((r.get("source") for r in session.records
                if r.get("metrics") is fm), "start")
    final_struct = spec["make"]()
    if fm is not None and str(src).startswith("r"):
        rnd_i = int(str(src)[1:str(src).index("p")])
        op, seg, mag, width = be.action_from_kw(out["final_kw"])
        final_struct = be.apply_edit(snapshots.get(
            rnd_i, base_struct), op, seg, mag, width)
    entry = dict(
        arm=arm, start=start_name, seed=seed, complete=out["complete"],
        search_pct=SEARCH_PCT[0], move_policy=MOVE_POLICY[0],
        final_action=out["final_kw"],
        final_anchor=out["final_action"],
        final_action_desc=be.action_describe(out["final_kw"]),
        final_record_source=src,
        final_D=out.get("final_D"), final_feasible=out.get("final_feasible"),
        final_metrics=out.get("final_metrics"),
        final_struct_wkt=dict(
            pad_left=final_struct["pad_left"].wkt,
            pad_right=final_struct["pad_right"].wkt,
            junction=list(final_struct["junction"].coords)),
        final_struct_fp=struct_fp(final_struct),
        shape_delta=shape_delta(base_struct, final_struct),
        adopted=out["adopted"], rounds=out["rounds"], history=out["history"],
        counts=dict(n_new_evals=out["n_new_evals"],
                    n_duplicates=out["n_duplicates"],
                    n_failures=out["n_failures"], n_invalid=out["n_invalid"],
                    n_feasible=out["n_feasible"], budget=BUDGET),
        records=session.records, wall_s=round(time.perf_counter() - t0, 1))
    return entry


def write_runs(entry):
    # 完成键纳入种子（同起点同方法不同种子共存；2026-11-07 复核）
    lines = []
    rp = runs_path()
    if rp.exists():
        for line in rp.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                if (r.get("arm"), r.get("start"), r.get("seed")) != (
                        entry["arm"], entry["start"], entry["seed"]):
                    lines.append(line.rstrip("\n"))
            except Exception:
                continue
    lines.append(json.dumps(dict(provenance=provenance(), **entry),
                            ensure_ascii=False, default=str))
    rp.write_text("\n".join(lines) + "\n", encoding="utf-8")


def done_keys():
    ks = set()
    rp = runs_path()
    if rp.exists():
        for line in rp.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("complete"):
                    ks.add((r["arm"], r["start"], r.get("seed")))
            except Exception:
                continue
    return ks


def refine_winners():
    """优胜者 0.1% 复验（两档策略）：可行且相对起点有改进的终选
    （不要求 complete——预算未满的臂其终选同样是真实可行记录）。"""
    start_D = {}
    for line in runs_path().open(encoding="utf-8"):
        r = json.loads(line)
        start_D.setdefault(r["start"], None)
    for sname in STARTS:
        p = OUT / f"start_{sname}.json"
        if p.exists():
            start_D[sname] = json.loads(
                p.read_text(encoding="utf-8"))["metrics"]["D"]
    rows = []
    for line in runs_path().open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("search_pct") == 0.1:
            continue  # 已是 0.1% 档终选，无需复验
        if (r.get("final_feasible") and r.get("final_D") is not None
                and start_D.get(r["start"]) is not None
                and r["final_D"] < start_D[r["start"]]):
            rows.append(r)
    for r in rows:
        name = f"{r['start']}_{r['arm']}"
        p = OUT / f"winner01_{name}.json"
        if p.exists():
            continue
        print(f"[0.1%复验] {name}", flush=True)
        from shapely import wkt as _wkt_mod
        from shapely.geometry import LineString
        struct = dict(pad_left=_wkt_mod.loads(r["final_struct_wkt"]["pad_left"]),
                      pad_right=_wkt_mod.loads(
                          r["final_struct_wkt"]["pad_right"]),
                      islands=[],
                      junction=LineString(r["final_struct_wkt"]["junction"]),
                      label=name)
        t0 = time.perf_counter()
        try:
            m = eval_struct(struct, max_passes=20, percent_error=0.1)
            row = dict(name=name, coarse_D=r["final_D"], fine_D=m["D"],
                       fine_C=m["C_sigma_fF"], fine_f01=m["f01_GHz"],
                       scenarios=m["scenarios"],
                       wall_s=round(time.perf_counter() - t0, 1))
        except Exception as e:
            row = dict(name=name, failed=str(e)[:300])
        p.write_text(json.dumps(dict(provenance=provenance(), **row),
                                ensure_ascii=False, default=str, indent=1),
                     encoding="utf-8")
        if "fine_D" in row:
            print("  coarse=%.5f → fine=%.5f (%s%+.1f%%)" % (
                row["coarse_D"], row["fine_D"], "",
                (row["fine_D"] - row["coarse_D"]) / row["coarse_D"] * 100),
                flush=True)


def summarize():
    print("\n汇总（粗筛 0.5%）：")
    for line in runs_path().open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("complete"):
            sd = r.get("shape_delta", {})
            print("  %-19s %-17s D=%-9s feasible=%s 采纳%d 形状差=%.1f%%"
                  " 孔%d→%d" % (
                      r["start"], r["arm"],
                      ("%.5f" % r["final_D"])
                      if r.get("final_D") is not None else "—",
                      r.get("final_feasible"),
                      len(r.get("adopted") or []),
                      100 * sd.get("symdiff_area_ratio", 0),
                      sd.get("holes", {}).get("base", -1),
                      sd.get("holes", {}).get("new", -1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", nargs="*", default=list(STARTS))
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--refine-winners", action="store_true")
    ap.add_argument("--pct", type=float, default=0.5,
                    help="求解精度档 percent_error（默认 0.5=粗筛；"
                         "方向性判定一律 0.1）")
    ap.add_argument("--policy", choices=("greedy", "forced"),
                    default="greedy", help="轮后移动策略（forced=强制累积）")
    ap.add_argument("--seed", type=int, default=None,
                    help="运行种子（默认按起点×方法推导；进运行身份）")
    args = ap.parse_args()
    SEARCH_PCT[0] = args.pct
    MOVE_POLICY[0] = args.policy
    RUN_SEED[0] = args.seed
    t0 = time.perf_counter()
    if args.refine_winners:
        refine_winners()
        print("总耗时 %.0fs" % (time.perf_counter() - t0))
        return
    (OUT / "protocol.json").write_text(json.dumps(
        dict(version="fresh-start-search-v1", date="2026-11-04",
             decision="docs/decisions.md 2026-11-04(四)",
             budget=BUDGET, arms=ARMS, edit_box={k: list(v) for k, v
                                                 in EDIT_BOX.items()},
             coarse_pct=0.5, refine_pct=0.1,
             starts={k: v["note"] for k, v in STARTS.items()},
             seeds={"%s/%s" % (s, a): cp.arm_seed("fresh_" + s, a)
                    for s in STARTS for a in ARMS},
             provenance=provenance()), ensure_ascii=False, indent=1,
        default=str), encoding="utf-8")
    done = done_keys()
    for start_name in args.starts:
        m = start_eval(start_name)
        print("起点 %s: CΣ=%.2f D=%.5f" % (start_name, m["C_sigma_fF"],
                                           m["D"]), flush=True)
        for arm in ARMS:
            if arm not in args.arms:
                continue
            if (arm, start_name, arm_seed_for(start_name, arm)) in done:
                print("跳过 %s/%s" % (arm, start_name))
                continue
            print("===== %s / %s =====" % (arm, start_name), flush=True)
            entry = run_arm(arm, start_name)
            write_runs(entry)
            print("→ %s/%s D=%s complete=%s invalid=%d (%.0fs)" % (
                arm, start_name, entry.get("final_D"),
                entry.get("complete"),
                entry.get("counts", {}).get("n_invalid", -1),
                entry.get("wall_s", 0)), flush=True)
    summarize()
    print("总耗时 %.0fs → %s" % (time.perf_counter() - t0, OUT))


if __name__ == "__main__":
    main()
