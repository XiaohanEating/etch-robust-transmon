# -*- coding: utf-8 -*-
"""正式搜索评价器 v3：v2 的结参数/场景/失败处理修正版。

2026-10-25 审阅修正（docs/reviews/progress_review_verdict_2026-10-25.md）：
- **名义结参数固定**：L0(g) 只在名义几何上配定一次（f01→5.10 GHz）；
  刻蚀变体沿用 L0——同一器件受扰动，而非逐变体重配（v2 缺陷，抵消漂移）；
  结电感扰动场景才按 ±5% 改变 L。
- **联合场景**：补 etch_plus 与 lj_plus 的联合场景（原搜索集第 7 项）。
- **严格失败处理**：任一预定场景缺失或非有限 → drift_geom/drift_all 均为
  None 且 feasible=False（v2 的 drift_all 可由部分场景算出，不完整）。
- **v2 缓存迁移**：v2 缓存中的 C_q 与参与比不依赖 Lj，可纯后处理恢复——
  名义变体重配 L0、刻蚀变体用传入 L0 重算 LOM，免重解场。
- **原子写唯一临时文件**（uuid 后缀，消除并发 .tmp 冲突）。
- 有符号漂移 = log(T1_δ/T1_0) = log(rate_0/rate_δ)。

注意：G1 在原容差下仍未通过；本评价器供修复后的探索计算与正式验证准备，
其结果不自动构成 G1 证据。v1/v2 缓存目录不再读取（v2 仅作迁移来源）。
"""

import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path

import numpy as np

from ..configs import frozen
from ..runtime.records import ROOT

VERSION = "corrected-finite-v5"
F_TARGET = 5.10
SEARCH_H = 0.01
FINAL_H = 0.005
CACHE = ROOT / "results" / "cache" / "corrected_finite_v5"

_SOURCES = ("evaluator/corrected.py", "experiments/g1_step3_fast_rectangle.py",
            "experiments/g1_step2_local_edge.py", "evaluator/lom.py",
            "evaluator/finite_film.py")
_REFS_PATH = (ROOT / "results" / "g1_step3_rectangle_2026-10-23"
              / "step3_fast_rectangle.json")

_IDENTITY = None
SCENARIO_NAMES = ("nominal", "etch_plus", "etch_minus",
                  "lj_plus", "lj_minus", "ti_plus", "etch_lj_plus")


def identity_hash():
    """缓存身份：版本+冻结配置+2D 参考+关键源码哈希。任一变更即失效。"""
    global _IDENTITY
    if _IDENTITY is None:
        h = hashlib.sha256()
        h.update(VERSION.encode())
        h.update(frozen.frozen_hash().encode())
        h.update(hashlib.sha256(_REFS_PATH.read_bytes()).hexdigest().encode())
        for rel in _SOURCES:      # ROOT 即 robust_design_lab 包目录
            h.update(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest().encode())
        _IDENTITY = h.hexdigest()[:16]
    return _IDENTITY


def _cache_path(gh, h_edge):
    return CACHE / f"h{h_edge}" / f"{gh}_{identity_hash()}.json"


def _atomic_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _refs():
    return json.loads(_REFS_PATH.read_text(encoding="utf-8"))["two_d_reference"]


def _finite(x):
    return x is not None and math.isfinite(x)


def summarize_scenarios(nominal=None, etch_plus=None, etch_minus=None,
                        lj_plus=None, lj_minus=None, ti_plus=None,
                        etch_lj_plus=None):
    """纯后处理：七场景 → 有符号漂移与主终点（可独立单测）。

    每场景为 dict(p_SM, p_SA, p_MA, f01_GHz) 或 None。
    有符号漂移 = log(T1_δ/T1_0)；D = max|signed|。
    **严格失败**：任一非名义场景缺失或速率非有限 → drift_geom/drift_all=None。
    """
    scenarios = dict(nominal=nominal, etch_plus=etch_plus, etch_minus=etch_minus,
                     lj_plus=lj_plus, lj_minus=lj_minus, ti_plus=ti_plus,
                     etch_lj_plus=etch_lj_plus)
    empty = dict(all_scenarios_finite=False, signed=None, drift_geom=None,
                 drift_all=None, T1_us_nominal=None,
                 missing=[k for k, v in scenarios.items()
                          if k != "nominal" and v is None])
    if nominal is None:
        empty["missing"] = ["nominal"] + empty["missing"]
        return empty

    def t1(m):
        rate = 2 * math.pi * m["f01_GHz"] * 1e9 * sum(
            m["p_" + k] * frozen.TAN_DELTA[k] for k in ("SM", "MA", "SA"))
        return 1.0 / rate if _finite(rate) and rate > 0 else None

    t0 = t1(nominal)
    if t0 is None:
        return empty
    signed, complete = {}, True
    for name, m in scenarios.items():
        if name == "nominal":
            continue
        if m is None:
            signed[name] = None
            complete = False
            continue
        td = t1(m)
        signed[name] = math.log(td / t0) if (td and td > 0) else None
        if signed[name] is None or not _finite(signed[name]):
            complete = False
    out = dict(all_scenarios_finite=bool(complete),
               signed=signed,
               T1_us_nominal=t0 * 1e6,
               missing=[k for k, v in signed.items() if v is None])
    if not complete:
        out["drift_geom"] = None
        out["drift_all"] = None
        return out
    geom_vals = [abs(signed[k]) for k in ("etch_plus", "etch_minus")]
    out["drift_geom"] = max(geom_vals)
    out["drift_all"] = max(abs(v) for v in signed.values())
    return out


def _lom_block(c_q_fF, lj_H):
    from ..evaluator.lom import transmon_metrics
    tm = transmon_metrics(c_q_fF * 1e-15, lj_H)
    return dict(f01_GHz=tm["f01_GHz"], alpha_MHz=tm["alpha_MHz"])


def _variant_metrics(rec, h_edge, lj_pinned_H=None, use_cache=True):
    """单几何变体指标：v3 缓存 → v2 缓存迁移（纯后处理）→ 新解。

    lj_pinned_H=None 仅限名义变体（在此配定 L0）；刻蚀变体必须传入 L0。
    返回 (metrics, source)，source ∈ {v5, fresh}（旧缓存不再迁移）。
    """
    from ..evaluator.lom import match_lj

    gh = rec.geometry_hash
    cpath = _cache_path(gh, h_edge)
    if use_cache and cpath.exists():
        base = json.loads(cpath.read_text(encoding="utf-8"))
        # 缓存只存场量；电路参数按本次调用方 Lj 重算（lj 缺省=名义配定）
        lj = lj_pinned_H
        if lj is None:
            from ..evaluator.lom import match_lj
            lj, info = match_lj(base["C_q_fF"] * 1e-15, F_TARGET)
            if lj is None:
                raise ValueError(f"match_lj unreachable: {info}")
        return dict(base, **_lom_block(base["C_q_fF"], lj), lj_nH=lj * 1e9), "v5"

    # 旧缓存（v2/v4）**不再迁移参与比**：参与比随积分公式与局部修正改变
    # （corrections_v4_review 反例：v2 迁移 SA=4.80e-5 vs 当前公式 9.24e-5，
    # 低 48%）。旧缓存不返回派生量；v5 起参与比一律由当前公式重算。

    # 新解（唯一允许的场求解路径）
    from ..evaluator import finite_film
    from ..experiments import g1_step3_fast_rectangle as s3

    settings = finite_film.MeshSettings(thickness_nm=200.0, h_edge_mm=h_edge,
                                        h_max_mm=0.6, grow_mm=3.0)
    fs, surfaces = finite_film.solve_fields(rec, settings)
    lj = lj_pinned_H
    if lj is None:
        lj, info = match_lj(fs.C_q_fF * 1e-15, F_TARGET)
        if lj is None:
            raise ValueError(f"match_lj unreachable: {info}")
    pr = s3.corrected_participation(fs, surfaces, rec, lj, _refs())
    out = dict(p_SM=pr["p_SM"], p_SA=pr["p_SA"], p_MA=pr["p_MA"],
               C_q_fF=fs.C_q_fF)          # 只存场量；电路参数读取时重算
    if use_cache:
        _atomic_write(cpath, out)
    lj_eff = lj_pinned_H if lj_pinned_H is not None else lj
    return dict(out, **_lom_block(fs.C_q_fF, lj_eff),
                lj_nH=lj_eff * 1e9), "fresh"


def _check_physical(metrics):
    """变体级物理合法性：参与比非负（容差 -1e-12）、正频率、正电容。"""
    bad = []
    for k in ("p_SM", "p_SA", "p_MA"):
        v = metrics.get(k)
        if v is None or not _finite(v) or v < -1e-12:
            bad.append(f"{k}={v}")
    if not (_finite(metrics.get("C_q_fF")) and metrics["C_q_fF"] > 0):
        bad.append(f"C_q={metrics.get('C_q_fF')}")
    if not (_finite(metrics.get("f01_GHz")) and metrics["f01_GHz"] > 0):
        bad.append(f"f01={metrics.get('f01_GHz')}")
    return bad


def _evaluate_raw(rec, h_edge, use_cache, lj0_H=None):
    """名义（配定 L0）→ 刻蚀±（沿用 L0）→ 指标与来源。"""
    from ..perturbation.channels import apply_etch

    solves = {"nominal": rec}
    failures = {}
    for name, delta in (("etch_plus", frozen.ETCH_UM),
                        ("etch_minus", -frozen.ETCH_UM)):
        erec, err = apply_etch(rec, delta)
        if err:
            failures[name] = err
        else:
            solves[name] = erec

    results, sources = {}, {}
    lj0 = lj0_H            # 异常路径也需有定义
    # L0：外部传入优先（跨网格同器件）；否则在本档名义电容上配定
    if lj0_H is not None:
        lj0 = lj0_H
    try:
        nom, src = _variant_metrics(rec, h_edge, lj0 if lj0_H is not None else None,
                                    use_cache)
    except Exception as e:                                     # noqa: BLE001
        return {}, dict(nominal=f"{type(e).__name__}: {e}"[:200]), {}, lj0
    bad = _check_physical(nom)
    if bad:
        return {}, dict(nominal="nonphysical: " + ",".join(bad)), {}, lj0
    results["nominal"], sources["nominal"] = nom, src
    if lj0_H is None:
        lj0 = nom["lj_nH"] * 1e-9
    for name in ("etch_plus", "etch_minus"):
        if name in solves:
            try:
                m, src = _variant_metrics(solves[name], h_edge, lj0, use_cache)
            except Exception as e:                             # noqa: BLE001
                failures[name] = f"{type(e).__name__}: {e}"[:200]
                continue
            bad = _check_physical(m)
            if bad:
                failures[name] = "nonphysical: " + ",".join(bad)
                continue
            results[name], sources[name] = m, src
    return results, failures, sources, lj0


def evaluate_record(rec, h_edge=SEARCH_H, use_cache=True, lj0_H=None):
    """几何 → 七场景全扰动主终点与几何分项（严格失败处理）。

    lj0_H：外部固定的名义结电感（跨网格/终选复评须传入同一值，保持同一器件）。
    feasible = 全场景有限 + 物理合法 + 名义 f01/α 落有限模型窗（待追认探索窗）。
    """
    t0 = time.perf_counter()
    results, failures, sources, lj0 = _evaluate_raw(rec, h_edge, use_cache,
                                                    lj0_H=lj0_H)
    if "nominal" not in results:
        summary = summarize_scenarios(nominal=None)
        return dict(
            version=VERSION, identity=identity_hash(),
            geometry_hash=rec.geometry_hash, h_edge=h_edge,
            feasible=False, failures=failures, sources=sources,
            missing_scenarios=summary["missing"],
            all_scenarios_finite=False,
            drift_geom=None, drift_all=None, signed=None,
            seconds=round(time.perf_counter() - t0, 1))
    nom = results["nominal"]

    # 结电感通道（后处理）：f01 随 L 变化，p 不变
    lj_scenes = {}
    for name, rel in (("lj_plus", 1 + frozen.LJ_REL),
                      ("lj_minus", 1 - frozen.LJ_REL)):
        lj_scenes[name] = dict(nom, **_lom_block(nom["C_q_fF"], lj0 * rel))

    # 界面层通道（后处理）：p_i 整体 ×(1+10%)
    ti_plus = {**nom}
    for k in ("p_SM", "p_SA", "p_MA"):
        ti_plus[k] = nom[k] * (1 + frozen.TI_REL)

    # 联合场景：刻蚀正向几何 × L+5%（同一器件受联合扰动）
    etch_lj = None
    if "etch_plus" in results:
        ep = results["etch_plus"]
        etch_lj = dict(ep, **_lom_block(ep["C_q_fF"], lj0 * (1 + frozen.LJ_REL)))

    summary = summarize_scenarios(
        nominal=nom,
        etch_plus=results.get("etch_plus"),
        etch_minus=results.get("etch_minus"),
        lj_plus=lj_scenes["lj_plus"],
        lj_minus=lj_scenes["lj_minus"],
        ti_plus=ti_plus,
        etch_lj_plus=etch_lj)

    in_window = (frozen.WINDOW_F01_GHZ[0] <= nom["f01_GHz"] <= frozen.WINDOW_F01_GHZ[1]
                 and frozen.WINDOW_ALPHA_MHZ_FINITE[0] <= nom["alpha_MHz"]
                 <= frozen.WINDOW_ALPHA_MHZ_FINITE[1])
    return dict(
        version=VERSION, identity=identity_hash(), geometry_hash=rec.geometry_hash,
        h_edge=h_edge, C_q_fF=nom["C_q_fF"], f01_GHz=nom["f01_GHz"],
        alpha_MHz=nom["alpha_MHz"], lj0_nH=nom["lj_nH"],
        p_SM=nom["p_SM"], p_SA=nom["p_SA"], p_MA=nom["p_MA"],
        T1_us_nominal=summary["T1_us_nominal"],
        drift_geom=summary["drift_geom"], drift_all=summary["drift_all"],
        signed=summary["signed"], missing_scenarios=summary["missing"],
        all_scenarios_finite=summary["all_scenarios_finite"],
        feasible=bool(summary["all_scenarios_finite"] and in_window),
        failures=failures, sources=sources,
        seconds=round(time.perf_counter() - t0, 1),
    )
