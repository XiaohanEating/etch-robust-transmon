# -*- coding: utf-8 -*-
"""快速层评估器：GeometryRecord → FEM 场解 → 物理指标。

复用 qiskit_metal.analyses.fem.solver（gmsh + scikit-fem，tutorial 4.4x 验证栈）。
关键分离（省掉大批重复场解）：
- 场解只依赖**几何**（含刻蚀偏移后的几何）→ solve_fields，按几何哈希缓存；
- Lj / 界面层厚度只改后处理参数 → metrics(fields, lj, ti_scale) 秒级重算。
"""

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..configs import frozen
from ..geometry import two_pad


@dataclass
class FieldSolution:
    """一次静电场解（几何固定，模式能量归一 1 J）。"""
    geometry_hash: str
    C_matrix_fF: np.ndarray          # Maxwell 电容矩阵 (fF)
    C_q_fF: float                    # 结端口电容（两岛，其余悬浮）
    origin_xy_mm: tuple = (0.0, 0.0)  # 组件局部系原点在网格坐标系中的位置
    est: object = None               # Electrostatics 实例（参与比采样用）
    port_a: str = ""
    port_b: str = ""
    mesh_time_s: float = 0.0
    solve_time_s: float = 0.0
    n_conductors: int = 2
    substrate_mm: float = frozen.SUBSTRATE_MM
    package_mesh: object = None

    @property
    def key(self) -> str:
        return self.geometry_hash


class FieldCache:
    """几何哈希 → FieldSolution 的进程内缓存（搜索循环复用标称场解）。"""

    def __init__(self):
        self._store = {}

    def get_or_solve(self, rec, mesh_kwargs=None, order=None):
        gh = rec.geometry_hash + _mesh_tag(mesh_kwargs, order)
        if gh not in self._store:
            self._store[gh] = solve_fields(rec, mesh_kwargs, order)
        return self._store[gh]


def _mesh_tag(mesh_kwargs, order) -> str:
    blob = repr((sorted((mesh_kwargs or frozen.SEARCH_MESH).items()), order,
                 frozen.frozen_hash(), 'field-v2'))
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def solve_fields(rec, mesh_kwargs=None, order=None):
    """GeometryRecord → 网格 → Electrostatics 场解。

    mesh_kwargs/order 缺省用搜索档冻结配置；返回 FieldSolution。
    """
    from qiskit_metal.analyses.fem.solver import (
        Electrostatics, mesh_package, package_from_design,
    )

    order = frozen.SEARCH_ELEMENT_ORDER if order is None else order
    mk = dict(frozen.SEARCH_MESH if mesh_kwargs is None else mesh_kwargs)

    t0 = time.time()
    design, name = two_pad.build_design(rec, box_mm=frozen.SEARCH_BOX_MM,
                                        t_mm=frozen.SUBSTRATE_MM)
    pkg = package_from_design(design, eps_r=frozen.EPS_R_SUB)
    if design.qgeometry.tables['poly']['subtract'].astype(bool).any():
        raise ValueError('Subtract geometry cannot be discarded by the fast evaluator')
    mesh = mesh_package(pkg, "full", **mk)
    t_mesh = time.time() - t0

    t0 = time.time()
    est = Electrostatics(mesh, order=order)
    t_solve = time.time() - t0

    port_a, port_b = f"{name}.pad_left", f"{name}.pad_right"
    if port_a not in est.names:                    # 防御：名字顺序不稳定时按序取
        port_a, port_b = est.names[0], est.names[1]

    return FieldSolution(
        geometry_hash=rec.geometry_hash,
        C_matrix_fF=est.C * 1e15,
        C_q_fF=float(est.port_capacitance(port_a, port_b) * 1e15),
        origin_xy_mm=(frozen.SEARCH_BOX_MM[0] / 2, frozen.SEARCH_BOX_MM[1] / 2),
        est=est,
        port_a=port_a,
        port_b=port_b,
        mesh_time_s=t_mesh,
        solve_time_s=t_solve,
        n_conductors=len(est.names),
        substrate_mm=frozen.SUBSTRATE_MM,
        package_mesh=mesh,
    )
