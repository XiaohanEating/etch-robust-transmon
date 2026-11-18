# -*- coding: utf-8 -*-
"""强数值基线：与 VLM 探索同协议的对照臂。

背景：旧随机臂每次重生成整套参数（一次改 3-7 个参数的大跳变），是最弱
基线。文献中昂贵黑箱 + 小预算（个位数评估）场景的标准对照是：
贝叶斯优化（GP-EI，小预算最优一族）、CMA-ES（LLM 优化论文的标准强基线，
见 OPRO arXiv:2309.03409）、Nelder-Mead（OPRO 基线之一）、贪心坐标搜索
（与 VLM 臂同支架同动作空间的机制对照）与 Sobol 准随机 best-of-n（升级版
随机下界，见 arXiv:2310.00077 低预算研究）。

公平协议（所有臂一致）：
- 同一起点（base_variant），同一参数箱（PARAM_BOXES，与形状库生成器同范围）；
- 同一预算：恰好 budget 次 eval_fn 调用（每次 = 名义+刻蚀± 3 个 Q3D 场景）；
- 同一目标：最小化 D（与 VLM 臂的轮间接受判据一致）；可行性（C_Σ≥70 fF、
  f01∈[4.5,5.5] GHz）另作透镜报告，不混入目标；
- 报告终选 = 已评估点中的最优点（VLM 臂同规则）；
- 评估失败（求解异常）按大罚分计，不重试、不计为成功。

整数维（n_fingers）在评估前吸附；连续维格式化为 "XXXum"。
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import numpy as np

# ── 参数箱：与 four_shape_families / correct_idc 生成器范围一致 ──
# (lo, hi, kind)；kind 为 "um"（微米字符串）或 "int"（整数）
PARAM_BOXES: Dict[str, Dict[str, tuple]] = {
    "A_dual_pad": {
        "pad_width": (400, 600, "um"),
        "pad_height": (80, 180, "um"),
        "pad_gap": (20, 40, "um"),
    },
    "B_cross": {
        "cross_width": (40, 80, "um"),
        "cross_length": (150, 300, "um"),
        "cross_gap": (20, 50, "um"),
    },
    "B_star": {
        "num_points": (3, 7, "int"),
        "arm_width": (30, 60, "um"),
        "arm_length": (150, 250, "um"),
        "arm_gap": (20, 50, "um"),
    },
    "C_concentric": {
        "rad_o": (180, 250, "um"),
        "rad_i": (100, 160, "um"),
        "gap": (20, 50, "um"),
    },
    "D_interdigit": {
        "pad_width": (250, 400, "um"),
        "pad_height": (60, 120, "um"),
        "finger_width": (10, 25, "um"),
        "finger_height": (40, 80, "um"),
        "finger_gap": (10, 25, "um"),
        "n_fingers": (3, 7, "int"),
        "junction_gap": (15, 30, "um"),
    },
}

FAIL_PENALTY = 10.0  # 评估失败时的目标值（D 正常量级 ~1e-3）


def feasibility(metrics: Optional[dict]) -> dict:
    """可行性透镜：C_Σ ≥ 70 fF 且 f01 ∈ [4.5, 5.5] GHz。"""
    if not metrics or metrics.get("D") is None:
        return dict(feasible=False, reason="评估失败")
    cs, f = metrics.get("C_sigma_fF", 0.0), metrics.get("f01_GHz", 0.0)
    if cs < 70:
        return dict(feasible=False, reason=f"C_Σ={cs:.1f}fF<70")
    if not (4.5 <= f <= 5.5):
        return dict(feasible=False, reason=f"f01={f:.2f}GHz 出窗")
    return dict(feasible=True, reason="")


class _Evaluator:
    """预算计数 + 变体编解码 + 失败罚分。"""

    def __init__(self, family: str, base_variant: dict,
                 eval_fn: Callable[[dict, str], Optional[dict]], budget: int):
        self.family = family
        self.base = dict(base_variant)
        self.eval_fn = eval_fn
        self.budget = budget
        self.n_calls = 0
        self.records: List[dict] = []

    # 编解码：variant ⇄ 归一化向量 x ∈ [0,1]^d
    @property
    def keys(self) -> List[str]:
        return list(PARAM_BOXES[self.family])

    def decode(self, x: np.ndarray) -> dict:
        v = dict(self.base)
        for xi, k in zip(x, self.keys):
            lo, hi, kind = PARAM_BOXES[self.family][k]
            t = float(np.clip(xi, 0.0, 1.0))
            if kind == "int":
                v[k] = int(round(lo + t * (hi - lo)))
            elif kind == "float":
                v[k] = round(lo + t * (hi - lo), 1)
            else:
                v[k] = f"{lo + t * (hi - lo):.0f}um"
        return v

    def encode(self, variant: dict) -> np.ndarray:
        x = np.zeros(len(self.keys))
        for i, k in enumerate(self.keys):
            lo, hi, kind = PARAM_BOXES[self.family][k]
            if kind == "int":
                val = float(variant[k])
            else:
                val = float(str(variant[k]).rstrip("um"))
            x[i] = (val - lo) / (hi - lo)
        return np.clip(x, 0.0, 1.0)

    def evaluate(self, variant: dict) -> dict:
        """消耗 1 次预算；返回 (指标, 目标值) 记录。"""
        if self.n_calls >= self.budget:
            raise RuntimeError("预算已耗尽")
        self.n_calls += 1
        try:
            m = self.eval_fn(dict(variant), self.family)
        except Exception as e:  # 求解异常按罚分，不重试
            m = None
            err = f"{type(e).__name__}: {e}"[:200]
        else:
            err = ""
        if m is None or m.get("D") is None or not math.isfinite(m.get("D", math.inf)):
            obj, m2 = FAIL_PENALTY, None
        else:
            obj, m2 = float(m["D"]), m
        rec = dict(variant={k: str(v) for k, v in variant.items()
                            if k != "connection_pads"},
                   metrics=m2, objective=obj, error=err,
                   feasible=feasibility(m2)["feasible"])
        self.records.append(rec)
        return rec

    def best(self) -> dict:
        ok = [r for r in self.records if r["metrics"] is not None]
        if not ok:
            return dict(final_variant=None, final_metrics=None, best_objective=None)
        b = min(ok, key=lambda r: r["objective"])
        return dict(final_variant=_full_variant(self.base, b["variant"]),
                    final_metrics=b["metrics"],
                    best_objective=b["objective"],
                    n_ok=len(ok), n_failed=len(self.records) - len(ok))


def _full_variant(base: dict, recorded: Optional[dict]) -> Optional[dict]:
    """recorded 只存了非 connection_pads 键；与 base 合并还原完整变体。"""
    if recorded is None:
        return None
    v = dict(base)
    for k, val in (recorded or {}).items():
        v[k] = _maybe_int(val)
    return v


def _maybe_int(val):
    s = str(val)
    try:
        f = float(s.rstrip("um"))
    except ValueError:
        return val
    return int(f) if f.is_integer() and "." not in s and "um" not in s else val


# ── 各基线算法：恰好消耗 budget 次 evaluate ──────────────────

def run_coordinate_descent(family, base_variant, eval_fn, budget=9, seed=0):
    """贪心坐标搜索：与 VLM 臂同支架（单参数局部移动、变好才采纳）。

    每坐标的试探值从当前值出发按倍率梯子（1.15, 0.87, 1.30, 0.70 …）
    生成并截断到箱内；轮流坐标，直到预算耗尽。机制对照：把 VLM 的
    "选哪个参数往哪推" 换成系统性枚举。
    """
    rng = np.random.default_rng(seed)
    ev = _Evaluator(family, base_variant, eval_fn, budget)
    keys = ev.keys
    cur = dict(base_variant)
    ladder = [1.15, 0.87, 1.30, 0.70, 1.50, 0.55, 1.80, 0.40, 2.20]
    step = {k: 0 for k in keys}
    order = list(rng.permutation(keys))
    cur_obj = None
    while ev.n_calls < budget:
        k = order[0]
        order = order[1:] + [k]  # 轮转
        lo, hi, kind = PARAM_BOXES[family][k]
        cur_val = float(str(cur[k]).rstrip("um"))
        f = ladder[step[k] % len(ladder)]
        step[k] += 1
        new_val = float(np.clip(cur_val * f, lo, hi))
        if abs(new_val - cur_val) < (hi - lo) * 0.02:
            continue  # 已贴边，跳过该试探
        cand = dict(cur)
        if kind == "int":
            cand[k] = int(round(new_val))
        elif kind == "float":
            cand[k] = round(new_val, 1)
        else:
            cand[k] = f"{new_val:.0f}um"
        rec = ev.evaluate(cand)
        if rec["metrics"] is not None and (cur_obj is None or rec["objective"] < cur_obj):
            cur, cur_obj = cand, rec["objective"]
    return dict(arm="coordinate_descent", family=family, **ev.best(),
                all_results=ev.records, n_evals=ev.n_calls)


def run_cma_es(family, base_variant, eval_fn, budget=9, seed=0):
    """CMA-ES（cma 4.5.0）：LLM 优化文献的标准强基线。

    popsize=3（9 预算 = 3 代）；小预算下方差更新受限于代数，这是
    诚实的局限，按文献惯例仍列为强基线。
    """
    import cma
    ev = _Evaluator(family, base_variant, eval_fn, budget)
    popsize = 3
    es = cma.CMAEvolutionStrategy(ev.encode(base_variant), 0.25, {
        "popsize": popsize, "bounds": [0, 1], "seed": seed,
        "verbose": -9, "CMA_active": True,
    })
    while ev.n_calls + popsize <= budget and not es.stop():
        xs = es.ask()
        recs = [ev.evaluate(ev.decode(np.asarray(x))) for x in xs]
        es.tell(xs, [r["objective"] for r in recs])
    # 零头预算：单独评估当前均值
    if ev.n_calls < budget:
        ev.evaluate(ev.decode(np.asarray(es.mean)))
    return dict(arm="cma_es", family=family, **ev.best(),
                all_results=ev.records, n_evals=ev.n_calls)


def run_nelder_mead(family, base_variant, eval_fn, budget=9, seed=0):
    """Nelder-Mead（scipy）：OPRO 的经典基线，低维小预算适用。"""
    from scipy.optimize import minimize
    ev = _Evaluator(family, base_variant, eval_fn, budget)
    x0 = ev.encode(base_variant)
    d = len(x0)

    def obj(x):
        if ev.n_calls >= budget:
            raise _BudgetStop
        ev.evaluate(ev.decode(np.asarray(x)))
        return ev.records[-1]["objective"]

    class _BudgetStop(Exception):
        pass

    simplex = np.vstack([x0] + [
        np.clip(x0 + np.eye(d)[i] * 0.12, 0, 1) for i in range(d)])
    try:
        minimize(obj, x0, method="Nelder-Mead", bounds=[(0, 1)] * d,
                 options=dict(maxfev=budget, initial_simplex=simplex,
                              xatol=1e-3, fatol=1e-6))
    except _BudgetStop:
        pass
    # scipy 可能提前收敛而不耗尽预算：剩余预算试探对角方向
    while ev.n_calls < budget:
        x = np.clip(x0 + np.full(d, 0.2 - 0.4 * ((ev.n_calls // max(d, 1)) % 2)), 0, 1)
        ev.evaluate(ev.decode(x))
        x0 = x  # 允许游走，扩大覆盖
    return dict(arm="nelder_mead", family=family, **ev.best(),
                all_results=ev.records, n_evals=ev.n_calls)


def run_sobol_best(family, base_variant, eval_fn, budget=9, seed=0):
    """Sobol 准随机 best-of-n：升级版随机下界（低差异序列覆盖优于 iid）。"""
    from scipy.stats import qmc
    ev = _Evaluator(family, base_variant, eval_fn, budget)
    d = len(ev.keys)
    # 预算 9 非 2 的幂，Sobol 平衡性保证减弱，但覆盖仍远优于 iid 随机
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pts = qmc.Sobol(d, scramble=True, seed=seed).random(budget)
    for p in pts:
        ev.evaluate(ev.decode(p))
    return dict(arm="sobol_best", family=family, **ev.best(),
                all_results=ev.records, n_evals=ev.n_calls)


def run_bayesian_opt(family, base_variant, eval_fn, budget=9, seed=0):
    """GP-EI 贝叶斯优化（scikit-optimize）：小预算昂贵评估的首选基线。

    需要 scikit-optimize；未安装时明确报错（不静默换算法）。
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError as e:
        raise ImportError(
            "bayesian_opt 需要 scikit-optimize（pip install scikit-optimize）；"
            "按项目规则装包前须获得确认") from e
    ev = _Evaluator(family, base_variant, eval_fn, budget)
    keys = ev.keys
    dims = []
    for k in keys:
        lo, hi, kind = PARAM_BOXES[family][k]
        if kind == "int":
            dims.append(Real(lo, hi))  # 整数维按连续处理，decode 时吸附
        else:
            dims.append(Real(lo, hi))

    def obj(vals):
        variant = dict(base_variant)
        for k, v in zip(keys, vals):
            lo, hi, kind = PARAM_BOXES[family][k]
            if kind == "int":
                variant[k] = int(round(np.clip(v, lo, hi)))
            elif kind == "float":
                variant[k] = round(float(np.clip(v, lo, hi)), 1)
            else:
                variant[k] = f"{np.clip(v, lo, hi):.0f}um"
        return ev.evaluate(variant)["objective"]

    gp_minimize(obj, dims, n_calls=budget, n_initial_points=min(4, budget - 1),
                x0=[float(x) for x in
                    [float(str(base_variant[k]).rstrip('um')) for k in keys]],
                random_state=seed, noise=1e-8, verbose=False)
    return dict(arm="bayesian_opt", family=family, **ev.best(),
                all_results=ev.records, n_evals=ev.n_calls)


BASELINES = {
    "coordinate_descent": run_coordinate_descent,
    "cma_es": run_cma_es,
    "nelder_mead": run_nelder_mead,
    "sobol_best": run_sobol_best,
    "bayesian_opt": run_bayesian_opt,
}


def run_strong_baseline(name, family, base_variant, eval_fn,
                        budget=9, seed=0) -> dict:
    if name not in BASELINES:
        raise KeyError(f"未知基线 {name}；可用: {list(BASELINES)}")
    if family not in PARAM_BOXES:
        raise KeyError(f"族 {family} 无参数箱；可用: {list(PARAM_BOXES)}")
    out = BASELINES[name](family, base_variant, eval_fn, budget, seed)
    out["seed"] = seed
    out["budget"] = budget
    assert out["n_evals"] == budget, f"{name}: 预算不守恒 {out['n_evals']}≠{budget}"
    return out
