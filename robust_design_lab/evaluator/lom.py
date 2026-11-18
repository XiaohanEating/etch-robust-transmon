# -*- coding: utf-8 -*-
"""LOM：电容 + 结电感 → f01 / 非谐性（权威实现 scqubits，避免手写近似）。

两岛单结、无端口的集总模型：C_Σ = C_q + C_j；
EJ = Φ0²/(4π²Lj)，EC = e²/(2C_Σ)，单位 GHz（scqubits 频率单位约定）。
"""

import numpy as np

from ..configs import frozen

_H = 6.62607015e-34
_E = 1.602176634e-19


def transmon_metrics(C_q_F: float, lj_H: float, cj_F: float = None):
    """返回 dict: EJ_GHz, EC_GHz, f01_GHz, alpha_MHz, C_sigma_fF。"""
    from scqubits import Transmon

    cj = frozen.CJ_FF * 1e-15 if cj_F is None else cj_F
    c_sigma = C_q_F + cj
    EJ_GHz = (frozen.PHI0 ** 2 / (4 * np.pi ** 2 * lj_H)) / _H / 1e9
    EC_GHz = _E ** 2 / (2 * c_sigma * _H) / 1e9

    q = Transmon(EJ=EJ_GHz, EC=EC_GHz, ng=0.0, ncut=110)
    f01 = float(q.E01())
    alpha = float(q.anharmonicity()) * 1e3          # GHz → MHz
    return dict(EJ_GHz=float(EJ_GHz), EC_GHz=float(EC_GHz),
                f01_GHz=f01, alpha_MHz=alpha, C_sigma_fF=c_sigma * 1e15)


def match_lj(C_q_F: float, f_target_GHz: float, lj_bounds=(6e-9, 25e-9), tol=1e-4):
    """二分求结电感使 f01 命中目标（按候选配定结，方案 §6.2 窗口重冻结的实现）。

    返回 (lj_H 或 None, info)。电容固定，无需额外场解。越界返回 None。
    """
    lo, hi = lj_bounds
    m_lo, m_hi = transmon_metrics(C_q_F, lo), transmon_metrics(C_q_F, hi)
    f_lo, f_hi = m_lo["f01_GHz"], m_hi["f01_GHz"]
    if not (min(f_lo, f_hi) - tol <= f_target_GHz <= max(f_lo, f_hi) + tol):
        return None, f"f01 target {f_target_GHz} unreachable: Lj∈[{lo*1e9:.0f},{hi*1e9:.0f}]nH → f01∈[{min(f_lo,f_hi):.2f},{max(f_lo,f_hi):.2f}]"
    for _ in range(50):
        mid = (lo + hi) / 2
        f_mid = transmon_metrics(C_q_F, mid)["f01_GHz"]
        if abs(f_mid - f_target_GHz) < tol:
            return mid, "converged"
        if (f_mid - f_target_GHz) * (f_lo - f_target_GHz) > 0:
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return (lo + hi) / 2, "max_iter"
