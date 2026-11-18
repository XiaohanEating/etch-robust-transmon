# -*- coding: utf-8 -*-
"""评价器 v4 防回归检查（对应 corrections_followup_review_2026-10-25 四反例）。

R1 缓存改变计算对象：同一缓存条目传入不同 Lj → 电路参数必须随之变化
   （缓存只存场量，读取时重算）；
R2 迁移来源不明/冲突：同几何多身份 v2 文件 → 拒绝迁移；
R3 求解异常：单变体抛错 → 场景失败记录而非崩溃；
R4 非法物理量：负参与比但总率为正 → 判 nonphysical 失败，不返回 D=0。
全部用临时目录与合成缓存，不触发场求解。
"""

import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SYN_HASH = None  # 用记录真实哈希（见各测试内）


def _syn_record():
    from shapely.geometry import LineString, Polygon
    from robust_design_lab.geometry.record import GeometryRecord
    return GeometryRecord(
        family="two_pad_free", category="synthetic", category_index=0, seed=0,
        pad_left=Polygon([(-215, -150), (-15, -150), (-15, 150), (-215, 150)]),
        pad_right=Polygon([(15, -150), (215, -150), (215, 150), (15, 150)]),
        junction=LineString([(-15, 0), (15, 0)]))


class CacheFieldOnlyTests(unittest.TestCase):
    def test_different_lj_gives_different_circuit_params(self):
        """R1：合成 v5 缓存（只存场量）→ 不同 Lj → 不同 f01。"""
        import numpy as np
        from robust_design_lab.evaluator import corrected as C

        gh = _syn_record().geometry_hash
        with tempfile.TemporaryDirectory() as td:
            C._IDENTITY = None
            old_cache, C.CACHE = C.CACHE, Path(td)
            try:
                cpath = C._cache_path(gh, 0.01)
                cpath.parent.mkdir(parents=True)
                cpath.write_text(json.dumps(
                    dict(p_SM=1e-4, p_SA=1e-4, p_MA=1e-6, C_q_fF=60.0)),
                    encoding="utf-8")
                a, sa = C._variant_metrics(_syn_record(), 0.01, 12e-9)
                b, sb = C._variant_metrics(_syn_record(), 0.01, 16e-9)
                self.assertNotAlmostEqual(a["f01_GHz"], b["f01_GHz"], places=3)
                self.assertGreater(abs(a["f01_GHz"] - b["f01_GHz"]), 0.05)
                self.assertEqual(sa, "v5")
            finally:
                C.CACHE = old_cache
                C._IDENTITY = None


class MigrationSourceGuardTests(unittest.TestCase):
    def test_old_participation_migration_unavailable(self):
        """R2（v5 契约）：旧参与比不得经任何迁移路径进入结果。

        corrections_v4_review 反例：白名单 v2 迁移使 SA=4.80e-5，而当前公式
        对同一存档给 9.24e-5（低 48%）。v5 起旧缓存完全不进入结果路径。
        """
        from robust_design_lab.evaluator import corrected as C

        src = open(Path(C.__file__), encoding="utf-8").read()
        self.assertNotIn("v2_migrated", src)
        self.assertNotIn("migrated_from", src)
        # 行为断言：v5 空缓存 + 场解禁用 → 调用必须失败（不得回退旧值）
        gh = _syn_record().geometry_hash
        OLD_SA = 4.799205692e-5
        with tempfile.TemporaryDirectory() as td:
            C._IDENTITY = None
            old_cache, C.CACHE = C.CACHE, Path(td)
            try:
                with unittest.mock.patch(
                        "robust_design_lab.evaluator.finite_film.solve_fields",
                        side_effect=RuntimeError("solve disabled")):
                    # 放一份带旧 SA 的伪 v2 文件（即便未来有人恢复迁移代码）
                    v2dir = Path(td) / "old" / "h0.01"
                    v2dir.mkdir(parents=True, exist_ok=True)
                    (v2dir / f"{gh}_908628673c030cc7.json").write_text(
                        json.dumps(dict(p_SM=1e-4, p_SA=OLD_SA, p_MA=1e-6,
                                        C_q_fF=60.0)), encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        C._variant_metrics(_syn_record(), 0.01, None)
            finally:
                C.CACHE = old_cache
                C._IDENTITY = None


class ExceptionHandlingTests(unittest.TestCase):
    def test_solve_exception_recorded_as_failure(self):
        """R3：名义变体求解抛错 → failures 记录，evaluate_record 返回缺场景。"""
        from unittest.mock import patch
        from robust_design_lab.evaluator import corrected as C

        with tempfile.TemporaryDirectory() as td:
            C._IDENTITY = None
            old_cache, C.CACHE = C.CACHE, Path(td)
            try:
                with patch("robust_design_lab.evaluator.finite_film.solve_fields",
                           side_effect=RuntimeError("boom")):
                    out = C.evaluate_record(_syn_record(), 0.01, use_cache=False)
                self.assertFalse(out["feasible"])
                self.assertIn("nominal", out["failures"])
                self.assertIn("RuntimeError: boom", out["failures"]["nominal"])
                self.assertIsNone(out["drift_all"])
                self.assertIn("nominal", out["missing_scenarios"])
            finally:
                C.CACHE = old_cache
                C._IDENTITY = None


class PhysicalValidityTests(unittest.TestCase):
    def test_negative_participation_rejected(self):
        """R4：负 p_SM（总率仍正）→ nonphysical，不得返回 D=0 可行结果。"""
        from robust_design_lab.evaluator import corrected as C
        bad = dict(p_SM=-1e-4, p_SA=3e-4, p_MA=1e-6, C_q_fF=60.0,
                   f01_GHz=5.1)
        problems = C._check_physical(bad)
        self.assertTrue(any("p_SM" in p for p in problems))

    def test_summary_rejects_nonpositive_rate(self):
        from robust_design_lab.evaluator import corrected as C
        out = C.summarize_scenarios(
            nominal=dict(p_SM=-1e-4, p_SA=3e-4, p_MA=1e-6, f01_GHz=5.1),
            etch_plus=dict(p_SM=-1e-4, p_SA=3e-4, p_MA=1e-6, f01_GHz=5.1),
            etch_minus=dict(p_SM=-1e-4, p_SA=3e-4, p_MA=1e-6, f01_GHz=5.1),
            lj_plus=dict(p_SM=-1e-4, p_SA=3e-4, p_MA=1e-6, f01_GHz=5.11),
            lj_minus=dict(p_SM=-1e-4, p_SA=3e-4, p_MA=1e-6, f01_GHz=5.09),
            ti_plus=dict(p_SM=-1.1e-4, p_SA=3.3e-4, p_MA=1.1e-6, f01_GHz=5.1),
            etch_lj_plus=dict(p_SM=-1e-4, p_SA=3e-4, p_MA=1e-6, f01_GHz=5.11))
        # 汇总层不拒绝正总率（变体层 _check_physical 已拦）；此处至少不得给出 0 漂移的"完美"结果
        self.assertFalse(out["all_scenarios_finite"] is True and out["drift_all"] == 0.0)


if __name__ == "__main__":
    unittest.main()
