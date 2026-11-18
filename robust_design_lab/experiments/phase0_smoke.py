# -*- coding: utf-8 -*-
"""Phase 0 环境冒烟：快速层评估器端到端跑通。

复用 tutorial 4.41–4.45 的验证载体（docs/tut/resources/package_modes）：
TwoPadTransmon（双矩形 pad + 结线）→ package_from_design → gmsh 网格
→ scikit-fem Electrostatics 电容矩阵。粗网格 + 小封装，只验证链路；
论文参考值（C_qubit≈170 fF，30 mm 封装、P2、细网格）留给 Phase A 标定。
成功判据：无异常，C 非负、量级合理（百 fF 量级）。
"""

import sys
import time
from pathlib import Path

t0 = time.time()

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docs" / "tut" / "resources" / "package_modes"))

import numpy as np

import package_modes as pm

from qiskit_metal.analyses.fem.solver import (
    mesh_package,
    Electrostatics,
)

print(f"[{time.time()-t0:6.1f}s] imports ok")

# 单比特、6 mm 小封装（冒烟提速；Phase A 用论文 30 mm 封装对数值）
design = pm.build_design(n=1, Lx=6.0, Ly=6.0, Lz=3.0, t=0.5)
pkg = pm.package_from_design(design, eps_r=11.9)
print(f"[{time.time()-t0:6.1f}s] package: box {pkg.Lx:.1f}x{pkg.Ly:.1f}x{pkg.Lz:.1f} mm, "
      f"t={pkg.t} mm, metals={len(pkg.metals)}, junctions={list(pkg.junctions)}")

mesh = mesh_package(pkg, None, h_metal=0.30, h_max=1.0)
print(f"[{time.time()-t0:6.1f}s] mesh ok")

est = Electrostatics(mesh, order=1)
names = est.names
C = est.C
print(f"[{time.time()-t0:6.1f}s] conductors: {names}")
print(f"[{time.time()-t0:6.1f}s] Maxwell C matrix (fF):\n{np.round(C * 1e15, 2)}")

q = names[0].rsplit(".", 1)[0]
C_q = est.port_capacitance(f"{q}.pad_left", f"{q}.pad_right")
diag_fF = np.diag(C) * 1e15
ok = np.all(diag_fF > 0) and np.all(diag_fF < 1000) and 10 < C_q * 1e15 < 1000
print(f"SMOKE {'PASSED' if ok else 'FAILED'}: diag(C) = {np.round(diag_fF, 2)} fF, "
      f"port C_q = {C_q*1e15:.1f} fF (粗网格小封装，论文 170 fF 属 Phase A 标定), "
      f"wall time {time.time()-t0:.1f}s")
