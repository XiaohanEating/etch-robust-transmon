# -*- coding: utf-8 -*-
"""P0：环境版本锁定 → results/environment.json（方案 §11 P0 退出条件）。"""

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))          # 仓库根，使 robust_design_lab 可导入

PACKAGES = [
    "qiskit_metal", "gmsh", "skfem", "numpy", "scipy", "scqubits",
    "shapely", "geopandas", "pandas", "matplotlib", "pyEPR", "pyaedt",
    "zhipuai", "cma", "PIL",
]


def main():
    import importlib

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "env_name": "qmetal09",
        "frozen_config": __import__(
            "robust_design_lab.configs.frozen", fromlist=["frozen_hash"]
        ).frozen_hash(),
        "packages": {},
    }
    for name in PACKAGES:
        try:
            m = importlib.import_module(name)
            info["packages"][name] = getattr(m, "__version__", "unknown")
        except Exception as e:                        # noqa: BLE001
            info["packages"][name] = f"IMPORT_FAIL: {type(e).__name__}"

    out = ROOT / "results" / "environment.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"environment.json → {out}")
    for k, v in info["packages"].items():
        print(f"  {k:12s} {v}")
    fails = [k for k, v in info["packages"].items() if str(v).startswith("IMPORT_FAIL")]
    print(f"ENV LOCK {'FAILED: ' + ', '.join(fails) if fails else 'PASSED'}")


if __name__ == "__main__":
    main()
