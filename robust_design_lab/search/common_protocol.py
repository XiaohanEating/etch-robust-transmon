# -*- coding: utf-8 -*-
"""共同评价协议：VLM 与数值臂走同一入口、同一预算、同一可行性处理。

背景（docs/reviews/new_structures_progress_review_2026-11-02.md 反例）：
- 旧 VLM 臂先完整评价起点再评 9 提案（实际 10 次评价），数值臂只耗
  9 次且拿不到起点指标；BO 又把起点算进 9 次预算——三种口径互不相同；
- VLM 只按 D 接受、数值臂有出窗罚分，可行性处理不一致，不可行的
  低漂移点会被 VLM 接受；
- 三轮均无有效提案仍写 complete=True；
- 重复候选重复求解（坐标组 9 次评价只有 8 个不同参数点）；
- CMA seed=0 实际生成随机种子且未落盘；数值终选参数是字符串，
  直接传回生成器报 TypeError。

本协议统一为：
- ProtocolSession.evaluate 是唯一候选评价入口：所有臂共享同一次起点
  实测（起点参数点复用、不耗预算）；臂内重复候选复用缓存并记
  duplicate（不耗预算）；预算按"新参数点的实际求解次数"计，恰为
  budget；求解失败消耗预算并记 FAIL_PENALTY；
- 原始指标与优化目标分开存：objective = D（可行）/ FAIL_PENALTY
  （出窗或求解失败）；所有臂的接受判据与终选统一用 objective；
- 终选 = 可行记录中 D 最小者（含起点；无可行记录时退化为 objective
  最小并显式标注 final_feasible=False）；
- complete 需满足：轮循环走完、至少一个有效新增评价、预算守恒；
- 种子非零且落盘（CMA 同时记录 requested 与 actual）；
- 参数一律 float，final_kw 可直接传回结构生成器重建。
"""

from __future__ import annotations

import math
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

FAIL_PENALTY = 10.0
# 可行性窗口（与资格/对比实验一致；上限 105 fF 防止 C_Σ 过大退出
# transmon 区）
FEASIBILITY_WINDOW = dict(C_sigma_fF=(70.0, 105.0), f01_GHz=(4.5, 5.5))
SEED_BASE = 20261102  # 非零；cma 对 seed=0 生成随机种子（审阅反例）


class BudgetExhausted(RuntimeError):
    """新增候选评价超出预算。"""


def _box_kind(entry):
    """box 条目可为 (lo, hi) 或 (lo, hi, "int")；返回 (lo, hi, is_int)。"""
    if len(entry) == 3 and entry[2] == "int":
        return float(entry[0]), float(entry[1]), True
    return float(entry[0]), float(entry[1]), False


def canonical(kw: Dict, keys: List[str]) -> Tuple:
    """参数点的规范化键：按 keys 顺序取 float 并圆整（去重判定用）。"""
    return tuple(round(float(kw[k]), 6) for k in keys)


def feasibility(metrics: Optional[dict],
                window: Dict = FEASIBILITY_WINDOW) -> Tuple[bool, str]:
    """统一可行性判据：C_Σ 与 f01 均在窗内。"""
    if not metrics or metrics.get("D") is None:
        return False, "评估失败"
    cs = metrics.get("C_sigma_fF", 0.0)
    f = metrics.get("f01_GHz", 0.0)
    lo, hi = window["C_sigma_fF"]
    if not (lo <= cs <= hi):
        return False, f"C_Σ={cs:.1f}fF 出窗[{lo},{hi}]"
    lo, hi = window["f01_GHz"]
    if not (lo <= f <= hi):
        return False, f"f01={f:.2f}GHz 出窗[{lo},{hi}]"
    return True, ""


class ProtocolSession:
    """一次搜索臂的共同评价会话。

    evaluator: kw → metrics dict（至少 D/C_sigma_fF/f01_GHz；任一必需
    场景缺失时抛异常，由本类统一记为失败——不产生有效漂移）。
    """

    def __init__(self, family: str, param_box: Dict[str, Tuple],
                 start_kw: Dict, evaluator: Callable,
                 budget: int = 9, fail_penalty: float = FAIL_PENALTY,
                 window: Dict = FEASIBILITY_WINDOW):
        self.family = family
        self.box = {k: _box_kind(v) for k, v in param_box.items()}
        self.keys = list(self.box)
        self.budget = int(budget)
        self.fail_penalty = fail_penalty
        self.window = window
        self.evaluator = evaluator
        self.start_kw = self.normalize(start_kw)
        self.records: List[dict] = []
        self.start_record: Optional[dict] = None
        self.n_new_evals = 0      # 实际求解的新参数点数（预算口径）
        self.n_duplicates = 0     # 复用缓存/起点的次数（不耗预算）
        self.n_failures = 0       # 求解异常次数（消耗预算）
        self.n_invalid = 0        # 无求解失败（如无效编辑，消耗预算）
        self._cache: Dict[Tuple, dict] = {}

    # ── 编解码：kw ⇄ [0,1]^d（数值臂用；int 维吸附） ──
    def normalize(self, kw: Dict) -> Dict:
        out = {}
        for k in self.keys:
            lo, hi, is_int = self.box[k]
            v = float(kw[k])
            out[k] = float(int(round(min(max(v, lo), hi)))) if is_int \
                else round(v, 6)
        return out

    def encode(self, kw: Dict) -> np.ndarray:
        x = np.zeros(len(self.keys))
        for i, k in enumerate(self.keys):
            lo, hi, _ = self.box[k]
            x[i] = (float(kw[k]) - lo) / (hi - lo)
        return np.clip(x, 0.0, 1.0)

    def decode(self, x) -> Dict:
        kw = {}
        for k, xi in zip(self.keys, np.asarray(x, dtype=float)):
            lo, hi, is_int = self.box[k]
            v = float(np.clip(lo + float(xi) * (hi - lo), lo, hi))
            kw[k] = float(int(round(v))) if is_int else round(v, 4)
        return kw

    # ── 评价 ──
    def evaluate_start(self) -> dict:
        """评价共享起点（预算外，每个家族只做一次，所有臂读同一条）。"""
        if self.start_record is not None:
            return self.start_record
        self.start_record = self._solve(self.start_kw, role="start",
                                        source="start")
        self._cache[(None, canonical(self.start_kw, self.keys))] = \
            self.start_record
        return self.start_record

    def attach_start(self, metrics: dict, wall_s: Optional[float] = None,
                     source: str = "shared") -> dict:
        """注入驱动层预先实测的家族级共享起点（各臂复用同一次求解，
        不再各自评价起点——修复"VLM 评起点而数值臂不评"的口径差）。"""
        if self.start_record is not None:
            return self.start_record
        if not metrics or metrics.get("D") is None:
            raise ValueError("共享起点指标缺失，不能注入")
        feasible, reason = feasibility(metrics, self.window)
        d = float(metrics["D"])
        rec = dict(role="start", source=source, kw=dict(self.start_kw),
                   metrics=metrics, D_raw=d, feasible=feasible,
                   feasible_reason=reason,
                   objective=d if feasible else self.fail_penalty,
                   error="", wall_s=wall_s)
        self.start_record = rec
        self._cache[(None, canonical(self.start_kw, self.keys))] = rec
        self.records.append(rec)
        return rec

    def register_parse_failure(self, source: str = "") -> dict:
        """解析失败计入预算（2026-11-07 复核：丢弃可见但不占尝试，
        失败规则仍未统一）。每个被丢弃的模型动作占一次登记尝试、
        按罚分记录，与非法动作/求解失败同权。"""
        if self.start_record is None:
            raise RuntimeError("必须先 evaluate_start()")
        if self.n_new_evals >= self.budget:
            raise BudgetExhausted(
                f"{self.family}: 新增候选预算 {self.budget} 已耗尽")
        self.n_new_evals += 1
        rec = dict(role="parse_failure", source=source, kw=None,
                   metrics=None, D_raw=None, feasible=False,
                   feasible_reason="解析失败", objective=self.fail_penalty,
                   error="动作解析失败（丢弃清单见调用记录）",
                   wall_s=0.0, context=None)
        self.records.append(rec)
        return rec

    def evaluate(self, kw: Dict, source: str = "",
                 context: Optional[str] = None) -> dict:
        """评价一个候选参数点（消耗预算；重复点复用缓存不耗预算）。

        context：候选所作用的基准状态指纹（如序列编辑的当前结构）。
        同一动作作用于不同基准是不同候选——cache 键含 context。"""
        if self.start_record is None:
            raise RuntimeError("必须先 evaluate_start()")
        if self.n_new_evals >= self.budget:
            raise BudgetExhausted(
                f"{self.family}: 新增候选预算 {self.budget} 已耗尽")
        kw = self.normalize(kw)
        key = (str(context) if context is not None else None,
               canonical(kw, self.keys))
        if key in self._cache:
            self.n_duplicates += 1
            rec = dict(self._cache[key], role="duplicate", source=source)
            self.records.append(rec)
            return rec
        rec = self._solve(kw, role="candidate", source=source)
        rec["context"] = str(context) if context is not None else None
        self._cache[key] = rec
        return rec

    def _solve(self, kw: Dict, role: str, source: str) -> dict:
        is_new_candidate = role != "start"
        if is_new_candidate:
            self.n_new_evals += 1  # 失败也消耗预算（评价机会已被消耗）
        t0 = time.perf_counter()
        try:
            m = self.evaluator(dict(kw))
            err = ""
        except Exception as e:  # 求解/提案异常：罚分记录，不重试
            m, err = None, f"{type(e).__name__}: {e}"[:200]
            if getattr(e, "no_solve", False):
                self.n_invalid += 1  # 无物理求解的失败（如无效编辑）
        if m is None or m.get("D") is None or not math.isfinite(m["D"]):
            metrics, feasible, reason = None, False, "评估失败"
            objective, d_raw = self.fail_penalty, None
            self.n_failures += 1
        else:
            metrics = m
            feasible, reason = feasibility(m, self.window)
            d_raw = float(m["D"])
            objective = d_raw if feasible else self.fail_penalty
        rec = dict(role=role, source=source, kw=dict(kw),
                   metrics=metrics, D_raw=d_raw, feasible=feasible,
                   feasible_reason=reason, objective=objective, error=err,
                   wall_s=round(time.perf_counter() - t0, 1))
        self.records.append(rec)
        return rec

    # ── 终选与收尾 ──
    def final(self) -> dict:
        """终选 = 可行记录（含起点）中 D 最小者；无可用可行记录时
        退化为 objective 最小并显式标注。"""
        ok = [r for r in self.records if r["metrics"] is not None]
        feas = [r for r in ok if r["feasible"]]
        out = dict(n_new_evals=self.n_new_evals,
                   n_duplicates=self.n_duplicates,
                   n_failures=self.n_failures,
                   n_invalid=self.n_invalid,
                   n_feasible=len(feas), budget=self.budget)
        pool = feas if feas else ok
        if not pool:
            return dict(out, final_kw=None, final_metrics=None,
                        final_D=None, final_feasible=False)
        b = min(pool, key=lambda r: (r["D_raw"] if feas else r["objective"]))
        return dict(out, final_kw=dict(b["kw"]),
                    final_metrics=b["metrics"], final_D=b["D_raw"],
                    final_feasible=b["feasible"], final_role=b["role"])

    def budget_ok(self) -> bool:
        return self.n_new_evals == self.budget


# ── 数值臂（全部走 ProtocolSession，objective 统一含可行性罚分） ──

def run_coordinate_descent(session: ProtocolSession, rng) -> dict:
    """贪心坐标搜索：与 VLM 同支架（单参数局部移动、变好才采纳）。
    倍率梯子轮流坐标；重复点由 session 去重，不再浪费求解。"""
    ladder = [1.15, 0.87, 1.30, 0.70, 1.50, 0.55, 1.80, 0.40, 2.20]
    keys = list(session.keys)
    step = {k: 0 for k in keys}
    order = list(rng.permutation(keys))
    cur = dict(session.start_kw)
    cur_obj = session.start_record["objective"]
    n_skipped = 0
    while session.n_new_evals < session.budget:
        if n_skipped > len(keys) * len(ladder):
            break  # 全坐标贴边，无试探可做（如实停止）
        k = order[0]
        order = order[1:] + [k]
        lo, hi, is_int = session.box[k]
        f = ladder[step[k] % len(ladder)]
        step[k] += 1
        cand = dict(cur)
        if is_int:
            span = hi - lo + 1
            nv = int(round(cur[k] + (f - 1.0) * span / 2))
            nv = int(min(max(nv, lo), hi))
            if abs(nv - cur[k]) < 1:
                n_skipped += 1
                continue
            cand[k] = float(nv)
        else:
            nv = min(max(cur[k] * f, lo), hi)
            if abs(nv - cur[k]) < (hi - lo) * 0.02:
                n_skipped += 1
                continue
            cand[k] = round(nv, 4)
        if cand[k] == cur[k]:
            n_skipped += 1
            continue
        rec = session.evaluate(cand, source="coordinate_descent")
        if rec["metrics"] is not None and rec["objective"] < cur_obj:
            cur, cur_obj = cand, rec["objective"]
    return session.final()


def run_cma_es(session: ProtocolSession, seed: int) -> dict:
    """CMA-ES（LLM 优化文献标准强基线）。seed 必须非零：cma 对 0 生成
    随机种子（审阅反例）；实际种子从 es.opts 读出并落盘。"""
    import cma
    if seed == 0:
        raise ValueError("CMA seed=0 会被 cma 替换为随机种子，须用非零种子")
    es = cma.CMAEvolutionStrategy(session.encode(session.start_kw), 0.25, {
        "popsize": 3, "bounds": [0, 1], "seed": seed,
        "verbose": -9, "CMA_active": True,
    })
    actual_seed = int(es.opts["seed"])
    popsize = 3
    while session.n_new_evals + popsize <= session.budget and not es.stop():
        xs = es.ask()
        recs = [session.evaluate(session.decode(np.asarray(x)),
                                 source="cma_es") for x in xs]
        es.tell(xs, [r["objective"] for r in recs])
    if session.n_new_evals < session.budget:  # 零头预算：评当前均值
        session.evaluate(session.decode(np.asarray(es.mean)),
                         source="cma_es_tail")
    out = session.final()
    out["cma_requested_seed"] = seed
    out["cma_actual_seed"] = actual_seed
    return out


def run_bayesian_opt(session: ProtocolSession, seed: int, rng) -> dict:
    """GP-EI 贝叶斯优化。x0=起点：session 对起点参数点复用共享实测、
    不耗预算；skopt 调用数固定，剩余预算以均匀随机补齐并如实标注。"""
    from skopt import gp_minimize
    from skopt.space import Real
    dims = [Real(*session.box[k][:2]) for k in session.keys]

    def obj(vals):
        kw = {k: round(float(v), 4) for k, v in zip(session.keys, vals)}
        return session.evaluate(kw, source="bayesian_opt")["objective"]

    gp_minimize(obj, dims, n_calls=session.budget,
                n_initial_points=min(4, session.budget - 1),
                x0=[session.start_kw[k] for k in session.keys],
                random_state=seed, noise=1e-8, verbose=False)
    while session.n_new_evals < session.budget:  # 重复点挤占后补齐
        kw = session.decode(rng.random(len(session.keys)))
        session.evaluate(kw, source="bayesian_opt_tail")
    return session.final()


# ── VLM 臂（chat_fn 注入：驱动接真实 API，测试接模拟） ──

def parse_proposals(raw_text: str, box: Dict[str, Tuple],
                    cur: Dict) -> List[Tuple[str, float, str]]:
    """解析模型回复 → [(param, 吸附后数值, 理由)]。

    只接受 box 内的 param；数值截断到箱内（与旧臂同规则）；
    抛 ValueError 表示无法解析。"""
    text = re.sub(r"```json?\n?", "", raw_text or "").replace("```", "").strip()
    props = json_loads_lenient(text)
    if isinstance(props, dict):
        props = [props]
    if not isinstance(props, list):
        raise ValueError("回复不是 JSON 数组")
    out = []
    for p in props:
        if not isinstance(p, dict) or "param" not in p:
            continue
        k = p["param"]
        if k not in box:
            continue
        lo, hi, _ = _box_kind(box[k])
        v = float(min(max(float(p["value"]), lo), hi))
        out.append((k, round(v, 4), str(p.get("reason", ""))[:200]))
    if not out:
        raise ValueError("回复中无合法提案")
    return out


def json_loads_lenient(text: str):
    import json
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def run_vlm(session: ProtocolSession, chat_fn: Callable, rounds: int = 3,
            proposals_per_round: int = 3) -> dict:
    """VLM 臂轮循环。

    chat_fn(round_index, cur_kw, cur_record, history) → 模型原始回复
    文本（提示组装与请求落盘由驱动完成，本函数只管控制流）。
    接受判据与终选统一用 objective（可行性罚分口径）——不可行的
    低漂移点不再被接受。

    完成条件：轮循环走完 且 至少一个有效新增评价 且 预算守恒；
    全部轮无有效提案 → complete=False（旧版空回复也标完成，反例）。"""
    cur = dict(session.start_kw)
    cur_rec = session.start_record
    history: List[dict] = []
    rounds_log: List[dict] = []
    for rnd in range(rounds):
        entry = dict(round=rnd)
        try:
            raw = chat_fn(rnd, dict(cur), dict(cur_rec),
                          [dict(h) for h in history])
            props = parse_proposals(raw, session.box, cur)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"[:200]
            rounds_log.append(entry)
            continue
        entry["n_proposals"] = len(props)
        best = None
        for pi, (k, v, reason) in enumerate(props[:proposals_per_round]):
            cand = dict(cur)
            cand[k] = v
            if cand[k] == cur[k]:
                continue  # 提案与当前值相同：不消耗预算
            try:
                rec = session.evaluate(cand, source=f"vlm_r{rnd}p{pi}")
            except BudgetExhausted:
                entry["budget_exhausted"] = True
                break
            history.append(dict(round=rnd, p=pi, kw=dict(cand),
                                D=rec["D_raw"], objective=rec["objective"],
                                feasible=rec["feasible"], reason=reason))
            if best is None or rec["objective"] < best[1]["objective"]:
                best = (cand, rec)
        if best is not None and best[1]["objective"] < cur_rec["objective"]:
            cur, cur_rec = best[0], best[1]
            entry["moved_to_D"] = cur_rec["D_raw"]
        rounds_log.append(entry)
    out = session.final()
    out["rounds"] = rounds_log
    out["history"] = history
    out["complete"] = (session.n_new_evals >= 1 and session.budget_ok())
    return out


NUMERIC_ARMS = ("coordinate_descent", "cma_es", "bayesian_opt")


# ── 轮制序列编辑协议（第三步区域编辑小试用） ──────────────────
#
# 所有方法（VLM 有图/无图、坐标、BO、CMA）共用：3 轮 × 每轮 3 个提案，
# 提案作用于**轮前结构快照**（驱动层维护），轮后统一按 objective 移动。
# 动作 kw 由驱动定义（如 op/seg/mag/width）；propose_fn 无内部预算控制
# （预算由 session 强制），返回本轮提案 kw 列表。

def run_round_based(session: "ProtocolSession", propose_fn: Callable,
                    rounds: int = 3, per_round: int = 3,
                    context_fn: Optional[Callable] = None,
                    on_round_end: Optional[Callable] = None,
                    move_policy: str = "greedy") -> dict:
    """轮制搜索主循环（与 run_vlm 同规则，提案来源任意）。

    propose_fn(round, cur_kw, cur_rec, history) → list[kw]（候选池，
    轮内按新增评价数取 per_round 个有效名额，重复点跳过）。
    context_fn(round) → 轮前基准状态指纹（序列编辑：当前结构随采纳
    变化，同动作不同基准是不同候选，缓存键含 context）。
    on_round_end(round, adopted_kw_or_None) → 轮后钩子（驱动在此把
    采纳动作应用到自己的当前结构上）。
    move_policy：greedy=改进才移动（默认）；forced=每轮强制采纳轮内
    最优（即使劣于当前）——累积编辑探索用，3 轮产生 3 步累积结构；
    终选规则不变（全部已评价可行点最优）。
    轮内逐个评价（BudgetExhausted 即停）。完成条件与 run_vlm 一致。"""
    cur = dict(session.start_kw)
    cur_rec = session.start_record
    history: List[dict] = []
    rounds_log: List[dict] = []
    adopted: List[dict] = []
    for rnd in range(rounds):
        entry = dict(round=rnd)
        ctx = context_fn(rnd) if context_fn is not None else None
        try:
            # 保留完整候选池（不截断）：轮内按新增评价数取足 per_round
            # 个名额，重复点跳过后继续取池中后续候选（2026-11-04 审阅
            # 反例：截断使坐标组 6 个不同点只评价 3 个、预算用不满）。
            props = list(propose_fn(rnd, dict(cur), dict(cur_rec),
                                    [dict(h) for h in history]))
            if not props:
                raise ValueError("proposer 未给出提案")
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"[:200]
            rounds_log.append(entry)
            continue
        entry["n_proposals"] = len(props)
        best = None
        n_fresh = 0  # 本轮新增评价数（重复点复用缓存不占轮名额）
        pi = 0
        while n_fresh < per_round and pi < len(props):
            kw = session.normalize(props[pi])
            pi += 1
            # 连续编辑模式：动作等于当前锚不代表候选相同——同一动作
            # 作用于新几何（轮前快照）是新候选，由 (context, kw) 缓存
            # 按实际几何去重（2026-11-06 复核 R2：旧版 kw==cur 跳过使
            # 两轮只评价一次）。
            try:
                rec = session.evaluate(kw, source=f"r{rnd}p{pi}",
                                       context=ctx)
            except BudgetExhausted:
                entry["budget_exhausted"] = True
                break
            if rec["role"] != "duplicate":
                n_fresh += 1
            history.append(dict(round=rnd, p=pi, kw=dict(kw),
                                D=rec["D_raw"], objective=rec["objective"],
                                feasible=rec["feasible"], error=rec["error"]))
            # 轮内最优只在有效评价（有 metrics）中选：forced 模式下
            # 全罚分轮没有可采纳结构，不移动
            if rec["metrics"] is not None and (
                    best is None
                    or rec["objective"] < best[1]["objective"]):
                best = (kw, rec)
        if best is not None and (
                (move_policy == "forced")
                or best[1]["objective"] < cur_rec["objective"]):
            cur, cur_rec = best[0], best[1]
            adopted.append(dict(round=rnd, kw=dict(cur),
                                D=cur_rec["D_raw"],
                                objective=cur_rec["objective"],
                                forced=(move_policy == "forced"
                                        and not best[1]["objective"]
                                        < session.start_record["objective"])))
            entry["moved_to_objective"] = cur_rec["objective"]
        rounds_log.append(entry)
        if on_round_end is not None:
            on_round_end(rnd, dict(cur) if adopted and
                         adopted[-1]["round"] == rnd else None)
    out = session.final()
    out.update(rounds=rounds_log, history=history, adopted=adopted,
               final_action=dict(cur))
    out["complete"] = (session.n_new_evals >= 1 and session.budget_ok())
    return out


# ── 数值 proposer 工厂（动作空间 [0,1]^d 编码，int 维由 session 吸附） ──

def make_coord_proposer(session: "ProtocolSession", rng) -> Callable:
    """坐标搜索 proposer：以当前锚出发轮流坐标梯子。

    每轮给出 6 个候选（轮制循环按"新增评价数"取前 3 个有效名额，
    重复点由协议层去重跳过）；离散维试探 ±span/2 步，连续维乘梯子。
    锚=最近被采纳动作。"""
    ladder = [1.15, 0.87, 1.30, 0.70]
    state = dict(step={k: 0 for k in session.keys},
                 order=list(rng.permutation(session.keys)))

    def propose(rnd, cur_kw, cur_rec, history):
        out = []
        guard = 0
        while len(out) < 6 and guard < 48:
            guard += 1
            k = state["order"][0]
            state["order"] = state["order"][1:] + [k]
            lo, hi, is_int = session.box[k]
            f = ladder[state["step"][k] % len(ladder)]
            state["step"][k] += 1
            cand = dict(cur_kw)
            if is_int:
                span = hi - lo + 1
                new = int(round(cur_kw[k] + (f - 1.0) * span / 2))
                new = int(min(max(new, lo), hi))
                if abs(new - cur_kw[k]) < 1:
                    continue
                cand[k] = float(new)
            else:
                new = min(max(cur_kw[k] * f, lo), hi)
                if abs(new - cur_kw[k]) < (hi - lo) * 0.05:
                    continue
                cand[k] = round(new, 4)
            if cand == cur_kw:
                continue
            out.append(cand)
        return out
    return propose


def make_cma_proposer(session: "ProtocolSession", seed: int) -> Callable:
    """CMA proposer：每代 ask 3 个（popsize=3）；下一轮从 session.records
    按 (context, kw) 匹配上一轮 objective 后 tell——序列编辑下同一动作
    作用在不同几何上是不同候选，反馈必须绑定产生它的状态
    （2026-11-04 审阅 §4）；缺失按罚分兜底。"""
    import cma
    if seed == 0:
        raise ValueError("CMA seed=0 会被 cma 替换为随机种子，须用非零种子")
    es = cma.CMAEvolutionStrategy(
        np.full(len(session.keys), 0.5), 0.25,
        dict(popsize=3, bounds=[0, 1], seed=seed, verbose=-9,
             CMA_active=True))
    state = dict(xs=[], last_rnd=None)

    def _objs_for(xs, last_rnd):
        objs = []
        for x in xs:
            kw_i = session.decode(np.asarray(x))
            obj = session.fail_penalty
            pre = f"r{last_rnd}p"
            for r in reversed(session.records):
                if str(r.get("source", "")).startswith(pre) \
                        and r.get("kw") == kw_i:
                    obj = r["objective"]
                    break
            objs.append(obj)
        return objs

    def propose(rnd, cur_kw, cur_rec, history):
        if state["xs"]:
            es.tell(state["xs"], _objs_for(state["xs"], state["last_rnd"]))
        xs = es.ask()
        state["xs"] = [np.asarray(x) for x in xs]
        state["last_rnd"] = rnd
        return [session.decode(x) for x in xs]
    return propose


def make_bo_proposer(session: "ProtocolSession", seed: int) -> Callable:
    """BO proposer：skopt Optimizer 增量接口，每轮 ask(3)；objective 按
    (context, kw) 匹配回填（同上，反馈绑定几何状态）。"""
    from skopt import Optimizer
    from skopt.space import Real
    dims = [Real(0.0, 1.0) for _ in session.keys]
    opt = Optimizer(dims, n_initial_points=4, random_state=seed)
    state = dict(xs=[], last_rnd=None)

    def _ys_for(xs, last_rnd):
        ys = []
        for x in xs:
            kw_i = session.decode(np.asarray(x))
            y = session.fail_penalty
            pre = f"r{last_rnd}p"
            for r in reversed(session.records):
                if str(r.get("source", "")).startswith(pre) \
                        and r.get("kw") == kw_i:
                    y = r["objective"]
                    break
            ys.append(y)
        return ys

    def propose(rnd, cur_kw, cur_rec, history):
        if state["xs"]:
            opt.tell([list(map(float, x)) for x in state["xs"]],
                     _ys_for(state["xs"], state["last_rnd"]))
        xs = opt.ask(n_points=3)
        state["xs"] = [np.asarray(x) for x in xs]
        state["last_rnd"] = rnd
        return [session.decode(x) for x in xs]
    return propose


def run_numeric_arm(name: str, session: ProtocolSession, seed: int) -> dict:
    """数值臂统一入口（种子非零；CMA 实际种子落盘）。"""
    rng = np.random.default_rng(seed)
    if name == "coordinate_descent":
        out = run_coordinate_descent(session, rng)
    elif name == "cma_es":
        out = run_cma_es(session, seed)
    elif name == "bayesian_opt":
        out = run_bayesian_opt(session, seed, rng)
    else:
        raise KeyError(f"未知数值臂 {name}；可用: {NUMERIC_ARMS}")
    out["complete"] = bool(out.get("final_metrics") is not None
                           and session.budget_ok())
    return out


def arm_seed(family: str, arm: str, base: int = SEED_BASE) -> int:
    """每臂确定性非零种子（同族同臂同种子，跨进程可复现）。

    用 zlib.crc32 稳定摘要派生——内建 hash() 进程加盐不可复现
    （error-journal 2026-10-27 教训）。"""
    import zlib
    h = (base + zlib.crc32(f"{family}/{arm}".encode("utf-8"))) % (2 ** 31)
    return h or 1
