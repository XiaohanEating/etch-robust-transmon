# -*- coding: utf-8 -*-
"""第一步定义修复的解析参考测试（progress_review_next_steps_2026-10-24 第一步验收）。

覆盖：
1. 两侧等能：SA 密度在纯切向（Et 连续）与纯法向（Dn 连续）下基片/空气两侧相等；
2. 均匀场背景的网格划分无关性（按边长加权均值定义）；
3. 归一化一致性：远区/背景（无 ½）与固体（½ε₀ε·x@Kx）在同一分母约定下的
   系数关系——用解析构造的小问题验证 U_solid×2 与 ε₀t·∫d2 的等值换算；
4. 单侧失败候选不得参与优选（corrected.evaluate_record 的失败处理契约）。
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robust_design_lab.configs import frozen
from robust_design_lab.experiments.g1_step2_local_edge import interface_density


class DensityTwoSideTests(unittest.TestCase):
    def test_tangential_field_same_from_both_sides(self):
        eps_sub = frozen.EPS_R_SUB
        d_sub = interface_density("SA", 2.5, 0.0, eps_sub)
        d_air = interface_density("SA", 2.5, 0.0, 1.0)
        self.assertAlmostEqual(d_sub, d_air, places=12)

    def test_normal_field_same_via_displacement_continuity(self):
        eps_sub = frozen.EPS_R_SUB
        en2_sub = 9.0                       # 场值 3 V/μm 的平方
        en2_air = en2_sub * eps_sub ** 2    # Dn 连续
        d_sub = interface_density("SA", 0.0, en2_sub, eps_sub)
        d_air = interface_density("SA", 0.0, en2_air, 1.0)
        self.assertAlmostEqual(d_sub, d_air, places=9)

    def test_sm_uses_all_component_scaling(self):
        d = interface_density("SM", 2.0, 3.0, frozen.EPS_R_SUB)
        self.assertAlmostEqual(
            d, frozen.EPS_R_SUB ** 2 / frozen.INTERFACE_EPS_R["SM"] * 5.0, places=12)


class BackgroundMeshIndependenceTests(unittest.TestCase):
    def _bg(self, widths, values):
        widths, values = np.asarray(widths), np.asarray(values)
        return float(np.sum(values * widths) / np.sum(widths))

    def test_uniform_field_partition_invariance(self):
        for n in (4, 7, 13, 41):
            bg = self._bg(np.full(n, 20.0 / n), np.full(n, 5.0))
            self.assertAlmostEqual(bg, 5.0, places=12)

    def test_window_excess_partition_invariance(self):
        # 均匀背景 + 未知近缘：窗口超额 = Σd2_win − bg·ΣΔr_win 与划分无关
        bg_true = 4.0
        excess_point = 12.0

        def build(n):
            r = np.linspace(0.0, 20.0, n + 1)
            dr = np.diff(r)
            d2 = np.full(n, bg_true)
            d2[0] += excess_point / dr[0]     # 近缘超额集中在首边
            return dr, d2

        outs = []
        for n in (5, 11, 23):
            dr, d2 = build(n)
            band = np.arange(n) >= n // 2    # 远带作背景（与实现同思路）
            bg = self._bg(dr[band], d2[band])
            outs.append(float(np.sum(d2 * dr) - bg * np.sum(dr)))
        for v in outs:
            self.assertAlmostEqual(v, excess_point, delta=1e-6)


class NormalizationConsistencyTests(unittest.TestCase):
    def test_solid_energy_matches_thin_convention_factor_two(self):
        """解析均匀场一致性（无量纲比值断言，防量纲空洞通过）：

        U_solid = ½ε₀ε_ox·E_ox²·(t·A)  —— 注意体积含膜厚 t（2026-10-25 审阅
        §8：旧测试漏乘 t 且用绝对容差，|x−y|<5e-7 对 ~1e-26 量级恒真）。
        薄层分子（2U 约定）= ε₀·t·(ε_sub²/ε_i)·E²·A。
        由 Dn 连续 E_ox=E_sub·ε_sub/ε_ox，且 ε_i=ε_ox ⇒ 2·U_solid ≡ 薄层分子。
        断言比值相对误差 < 1e-12。"""
        eps_i = frozen.INTERFACE_EPS_R["SM"]
        E2 = 7.5                        # (V/m)²
        A = 3.0e-8                      # m²（窗口面积）
        t = frozen.T_INTERFACE_NM["SM"] * 1e-9
        n_thin_bg = EPS0 * t * frozen.EPS_R_SUB ** 2 / eps_i * E2 * A
        E2_ox = E2 * frozen.EPS_R_SUB ** 2 / eps_i ** 2
        u_solid_bg = 0.5 * EPS0 * eps_i * E2_ox * (t * A)   # 体积 = t·A
        ratio = (2.0 * u_solid_bg) / n_thin_bg
        self.assertAlmostEqual(ratio, 1.0, places=12)
        self.assertGreater(n_thin_bg, 0)   # 防零值再次空洞化


EPS0 = 8.8541878128e-12


class FailureHandlingTests(unittest.TestCase):
    def test_single_side_failure_excludes_from_selection(self):
        """corrected.evaluate_record 的契约：任一场景失败 → drift=None 且
        feasible=False。以一个刻蚀侧检查失败的合成记录驱动纯后处理路径。"""
        from robust_design_lab.evaluator.corrected import summarize_scenarios

        out = summarize_scenarios(
            nominal=dict(p_SM=1e-4, p_SA=1e-4, p_MA=1e-6, f01_GHz=5.1),
            etch_plus=dict(p_SM=1.01e-4, p_SA=1e-4, p_MA=1e-6, f01_GHz=5.101),
            etch_minus=None,
            lj_plus=dict(p_SM=1e-4, p_SA=1e-4, p_MA=1e-6, f01_GHz=5.105),
            lj_minus=dict(p_SM=1e-4, p_SA=1e-4, p_MA=1e-6, f01_GHz=5.095),
            ti_plus=dict(p_SM=1.1e-4, p_SA=1.1e-4, p_MA=1.1e-6, f01_GHz=5.1),
            etch_lj_plus=None)
        self.assertIsNone(out["drift_geom"])
        self.assertIsNone(out["drift_all"])       # 严格失败：部分场景不得给出 D
        self.assertFalse(out["all_scenarios_finite"])
        self.assertIn("etch_minus", out["missing"])
        self.assertIn("etch_lj_plus", out["missing"])


if __name__ == "__main__":
    unittest.main()


class CacheIdentityTests(unittest.TestCase):
    def test_identity_changes_with_version(self):
        from robust_design_lab.evaluator import corrected

        base = corrected.identity_hash()
        corrected._IDENTITY = None
        try:
            corrected.VERSION = corrected.VERSION + "-probe"
            probe = corrected.identity_hash()
        finally:
            corrected.VERSION = corrected.VERSION[:-len("-probe")]
            corrected._IDENTITY = None
            corrected.identity_hash()
        self.assertNotEqual(base, probe)
        # 当前（恢复后）缓存文件名必须不同于 probe 版本的文件名
        self.assertNotEqual(
            corrected._cache_path("deadbeef", 0.01).name,
            (corrected.CACHE / "h0.01" / f"deadbeef_{probe}.json").name)
        self.assertEqual(
            corrected._cache_path("deadbeef", 0.01).name,
            (corrected.CACHE / "h0.01" / f"deadbeef_{base}.json").name)

    def test_v1_v2_cache_directory_not_read(self):
        from robust_design_lab.evaluator import corrected
        self.assertIn("corrected_finite_v5", str(corrected.CACHE))
        self.assertTrue(str(corrected.CACHE).endswith("corrected_finite_v5"))
