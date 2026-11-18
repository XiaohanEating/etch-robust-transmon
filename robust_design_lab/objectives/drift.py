# -*- coding: utf-8 -*-
"""漂移幅度指标（方案 v2.0 §2.3）与词典序比较。

D_y^S(x) = max_{δ∈S} |ln(y(x,δ)/y(x,0))|   （正值量：T1、f01、CΣ）
D_α^S(x) = max_{δ∈S} |α(x,δ)−α(x,0)|/|α(x,0)|（有符号非谐性；变号记约束失败）
词典序：D_T1(S_all) → D_T1(S_geom) → D_f01(S_all) → D_α(S_all)。
"""

import math

from ..configs import frozen
from ..evaluator import fast, participation
from ..perturbation import channels


def evaluate_scene(rec, scene: dict, cache: "fast.FieldCache" = None,
                    mesh_kwargs=None, sample=None, lj0_H: float = None):
    """单场景评估 → dict(C_q_fF, f01_GHz, alpha_MHz, p_*, T1_us)。

    lj0_H：该候选的名义结电感（按候选配定协议；缺省用冻结标称值）。
    场景 lj_rel 乘在 lj0 上；ti_rel 走界面层缩放；刻蚀场景重解场。
    """
    new_rec, lj_rel, ti, err = channels.apply_scene(rec, scene)
    if err:
        return dict(failed=err)
    if lj_rel <= 0 or ti <= 0:
        return dict(failed='Nonpositive process multiplier')
    lj = (frozen.LJ0_NH * 1e-9 if lj0_H is None else lj0_H) * lj_rel
    cache = cache or fast.FieldCache()
    fs = cache.get_or_solve(new_rec, mesh_kwargs)
    p = participation.surface_participation(fs, new_rec, sample, lj_H=lj, ti_scale=ti)
    p["C_q_fF"] = fs.C_q_fF
    return p


def drift_metrics(nominal: dict, results: dict) -> dict:
    """nominal + {场景名: 结果} → 漂移向量（含均值/95分位/最坏方向）。"""
    def rel_log(y, y0):
        return abs(math.log(y / y0)) if y > 0 and y0 > 0 else float("inf")

    def rel_diff(a, a0):
        return abs(a - a0) / abs(a0) if a0 else float("inf")

    d_t1, d_f, d_a = [], [], []
    n_fail = 0
    failures = {}
    keys = ('T1_us', 'f01_GHz', 'alpha_MHz')
    def finite(value):
        return isinstance(value,(int,float)) and math.isfinite(value)
    nominal_ok = all(k in nominal and finite(nominal[k]) for k in keys)
    nominal_ok = nominal_ok and nominal.get('T1_us', 0) > 0 and nominal.get('f01_GHz', 0) > 0
    nominal_ok = nominal_ok and nominal.get('alpha_MHz', 0) != 0
    for name, r in results.items():
        reason = r.get('failed')
        if not nominal_ok:
            reason = 'Invalid nominal metrics'
        elif any(k not in r or not finite(r[k]) for k in keys):
            reason = reason or 'Missing or nonfinite scene metrics'
        elif r['T1_us'] <= 0 or r['f01_GHz'] <= 0:
            reason = 'Nonpositive scene metric'
        elif r['alpha_MHz'] * nominal['alpha_MHz'] <= 0:
            reason = 'Anharmonicity changed sign or became zero'
        if reason:
            n_fail += 1
            failures[name] = str(reason)
            for values in (d_t1, d_f, d_a):
                values.append((math.inf, name))
            continue
        d_t1.append((rel_log(r["T1_us"], nominal["T1_us"]), name))
        d_f.append((rel_log(r["f01_GHz"], nominal["f01_GHz"]), name))
        d_a.append((rel_diff(r["alpha_MHz"], nominal["alpha_MHz"]), name))

    def pack(vals):
        if not vals:
            return dict(max=math.inf, mean=math.inf, q95=math.inf, worst=None)
        vs = sorted(v for v, _ in vals)
        return dict(max=vs[-1], mean=sum(vs) / len(vs), q95=vs[math.ceil(0.95 * len(vs)) - 1],
                    worst=max(vals, key=lambda t: t[0])[1])

    return dict(
        D_T1=pack(d_t1), D_f01=pack(d_f), D_alpha=pack(d_a),
        n_scenarios=len(results), n_failed=n_fail, n_completed=len(results) - n_fail,
        valid=bool(nominal_ok and results and not n_fail), failures=failures,
        quantile_method='nearest_rank_ceil',
    )


def geometry_only(scene):
    """Pure etch; joint junction/material perturbations stay in all-channel results."""
    return (abs(scene.get('etch_um', 0)) > 1e-12
            and abs(scene.get('lj_rel', 1) - 1) < 1e-12
            and abs(scene.get('ti_rel', 1) - 1) < 1e-12)


def lex_key(dm: dict, geom=None) -> tuple:
    """All-channel T1, geometry T1, frequency, anharmonicity."""
    if 'drift_all' in dm:
        geom, dm = dm['drift_geom'], dm['drift_all']
    if geom is None:
        raise ValueError('Four-part ranking requires geometry-only drift metrics')
    if dm.get('valid') is False or geom.get('valid') is False:
        return (math.inf,)*4
    return (dm['D_T1']['max'], geom['D_T1']['max'],
            dm['D_f01']['max'], dm['D_alpha']['max'])


def feasible(nominal: dict) -> bool:
    """电路窗口硬约束（方案 §6.2）。"""
    f, a = nominal["f01_GHz"], nominal["alpha_MHz"]
    return (frozen.WINDOW_F01_GHZ[0] <= f <= frozen.WINDOW_F01_GHZ[1]
            and frozen.WINDOW_ALPHA_MHZ[0] <= a <= frozen.WINDOW_ALPHA_MHZ[1])
