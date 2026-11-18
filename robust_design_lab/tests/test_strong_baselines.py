# -*- coding: utf-8 -*-
"""强基线模块测试：全部用代理函数，不做 Q3D 求解。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robust_design_lab.search.strong_baselines import (
    BASELINES, PARAM_BOXES, _Evaluator, feasibility, run_strong_baseline,
)

A_BASE = dict(
    pad_width="545um", pad_height="178um", pad_gap="34um",
    pocket_width="800um", pocket_height="800um",
    connection_pads={"readout": {"loc_W": 1}},
)
D_BASE = dict(
    pad_width="291um", pad_height="84um", finger_width="11um",
    finger_height="51um", finger_gap="16um", n_fingers=3,
    junction_gap="24um", conn_w="100um", conn_h="30um",
    jj_width="20um", pocket_margin="60um",
)


def surrogate_A(variant, family):
    """二次代理：最优在 pad_width=520um, pad_height=120um, pad_gap=30um。"""
    x = np.array([float(variant["pad_width"].rstrip("um")),
                  float(variant["pad_height"].rstrip("um")),
                  float(variant["pad_gap"].rstrip("um"))])
    x_star = np.array([520.0, 120.0, 30.0])
    scale = np.array([200.0, 100.0, 20.0])
    d = float(np.sum(((x - x_star) / scale) ** 2))
    return dict(D=d, C_sigma_fF=80.0, f01_GHz=5.0, alpha_MHz=-300.0)


class TestEvaluatorProtocol(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        ev = _Evaluator("A_dual_pad", A_BASE, surrogate_A, 9)
        x = ev.encode(A_BASE)
        back = ev.decode(x)
        for k in ev.keys:
            self.assertEqual(str(back[k]), str(A_BASE[k]), k)

    def test_decode_snaps_integer_and_box(self):
        ev = _Evaluator("D_interdigit", D_BASE, surrogate_A, 9)
        # 键序: pad_width, pad_height, finger_width, finger_height,
        #       finger_gap, n_fingers, junction_gap
        v = ev.decode(np.array([1.5, -0.2, 0.5, 0.5, 0.5, 1.2, 0.5]))
        self.assertEqual(v["n_fingers"], 7)        # t=1.2 截断到 1 → 上界
        self.assertEqual(v["pad_width"], "400um")  # 上界截断
        self.assertEqual(v["pad_height"], "60um")  # 下界截断

    def test_budget_enforced(self):
        ev = _Evaluator("A_dual_pad", A_BASE, surrogate_A, 2)
        ev.evaluate(A_BASE)
        ev.evaluate(A_BASE)
        with self.assertRaises(RuntimeError):
            ev.evaluate(A_BASE)

    def test_failure_penalty(self):
        def boom(variant, family):
            raise RuntimeError("COM 失败")
        ev = _Evaluator("A_dual_pad", A_BASE, boom, 1)
        rec = ev.evaluate(A_BASE)
        self.assertEqual(rec["objective"], 10.0)
        self.assertIsNone(rec["metrics"])
        self.assertFalse(rec["feasible"])
        best = ev.best()
        self.assertIsNone(best["final_metrics"])

    def test_feasibility_lens(self):
        self.assertTrue(feasibility(dict(D=0.001, C_sigma_fF=75,
                                         f01_GHz=5.0))["feasible"])
        self.assertFalse(feasibility(dict(D=0.001, C_sigma_fF=44,
                                          f01_GHz=6.4))["feasible"])


class TestBaselines(unittest.TestCase):
    """预算守恒 + 代理函数上确实收敛。"""

    def _run(self, name, family=D_BASE if False else A_BASE, seed=0):
        base = A_BASE
        fam = "A_dual_pad"
        return run_strong_baseline(name, fam, base, surrogate_A,
                                   budget=9, seed=seed)

    def test_all_baselines_budget_conserved(self):
        for name in ("coordinate_descent", "cma_es", "nelder_mead", "sobol_best"):
            out = self._run(name)
            self.assertEqual(out["n_evals"], 9, name)
            self.assertGreaterEqual(out["n_evals"],
                                    len(out["all_results"]), name)

    def test_coordinate_descent_beats_start(self):
        # 起点离最优点有距离；坐标搜索应显著降低代理目标
        out = self._run("coordinate_descent")
        self.assertLess(out["best_objective"], 0.35)

    def test_cma_es_and_nm_beat_start(self):
        for name in ("cma_es", "nelder_mead"):
            out = self._run(name)
            self.assertLess(out["best_objective"], 0.5, name)

    def test_sobol_reaches_decent_point(self):
        out = self._run("sobol_best")
        self.assertLess(out["best_objective"], 1.5)

    def test_bayesian_opt_raises_without_skopt(self):
        try:
            import skopt  # noqa: F401
            self.skipTest("skopt 已安装，无法测试缺包分支")
        except ImportError:
            with self.assertRaises(ImportError) as cm:
                self._run("bayesian_opt")
            self.assertIn("scikit-optimize", str(cm.exception))

    def test_seeds_change_sobol_trajectory(self):
        o1 = self._run("sobol_best", seed=1)
        o2 = self._run("sobol_best", seed=2)
        v1 = [r["variant"] for r in o1["all_results"]]
        v2 = [r["variant"] for r in o2["all_results"]]
        self.assertNotEqual(v1, v2)


if __name__ == "__main__":
    unittest.main()
