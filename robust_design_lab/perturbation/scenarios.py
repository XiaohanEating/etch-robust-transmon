# -*- coding: utf-8 -*-
"""三套扰动场景的生成与冻结（方案 v2.0 §5.2）。

S_search（7）：名义 + 刻蚀± + 结± + 界面+ + 联合（刻蚀+,结+）。
S_dev（9）：未用于排序的方向（界面−、各联合方向）。
S_holdout（27）：种子锁定、连续幅度、含未见联合方向；只用于最终报告。
全部参数与哈希落盘 results/scenarios/scenarios.json；生成后不可改。
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from ..configs import frozen


def _s(etch=0.0, lj=0.0, ti=0.0, **kw):
    d = dict(etch_um=etch, lj_rel=1.0 + lj, ti_rel=1.0 + ti)
    d.update(kw)
    return d


def build_scenarios(master_seed: int = 20261021) -> dict:
    E, L, T = frozen.ETCH_UM, frozen.LJ_REL, frozen.TI_REL

    s_search = [
        _s(),                       # 名义点
        _s(etch=+E), _s(etch=-E),
        _s(lj=+L), _s(lj=-L),
        _s(ti=+T),
        _s(etch=+E, lj=+L),         # 几何/材料联合
    ]

    s_dev = [
        _s(ti=-T),
        _s(etch=+E, ti=-T), _s(etch=-E, ti=+T),
        _s(etch=+E, lj=-L), _s(etch=-E, lj=+L),
        _s(lj=+L, ti=+T), _s(lj=-L, ti=-T),
        _s(etch=-E, lj=+L, ti=+T),
        _s(etch=+E, lj=-L, ti=-T),
    ]

    rng = np.random.default_rng(master_seed)
    s_holdout = []
    for _ in range(27):
        # 连续幅度（± 冻结边界内）+ 随机通道组合，含两两/三联合
        s_holdout.append(_s(
            etch=float(rng.uniform(-E, E)),
            lj=float(rng.uniform(-L, L)),
            ti=float(rng.uniform(-T, T)),
        ))

    return dict(S_search=s_search, S_dev=s_dev, S_holdout=s_holdout)


def freeze(root: "Path" = None) -> dict:
    """生成并落盘场景表 + 哈希。返回 dict 含各集合与 sha256。"""
    root = Path(root or Path(__file__).resolve().parents[1] / "results" / "scenarios")
    root.mkdir(parents=True, exist_ok=True)
    sc = build_scenarios()
    blob = json.dumps(sc, sort_keys=True)
    sc["sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    sc["frozen_config"] = frozen.frozen_hash()
    out = root / "scenarios.json"
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old["sha256"] != sc["sha256"]:
            raise RuntimeError(
                f"场景表已冻结且与新生成不一致（旧 {old['sha256'][:12]} vs 新 {sc['sha256'][:12]}）；"
                "改动属协议变更，须先落档 docs/decisions.md 并手动删除旧文件")
        return sc
    out.write_text(json.dumps(sc, indent=2, ensure_ascii=False), encoding="utf-8")
    return sc


def load(root: "Path" = None) -> dict:
    root = Path(root or Path(__file__).resolve().parents[1] / "results" / "scenarios")
    return json.loads((root / "scenarios.json").read_text(encoding="utf-8"))
