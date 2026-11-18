# -*- coding: utf-8 -*-
"""共同评价协议测试：全部用代理评价函数，不做 Q3D 求解与模型调用。

覆盖 docs/plans/new_structures_next_steps_2026-11-02.md 第一步验收反例：
- 地口袋独立偏移（四边分别 ±δ，不被扰动电极外接框抵消）；
- 十次评价含起点的预算口径统一（共享起点注入 + 恰 9 次新增求解）；
- 不可行低漂移点不更新最优（objective 含可行性罚分）；
- 空回复不标完成；
- CMA 相同非零实际种子可重现；
- 终选参数为 float、可直接重建结构；
- 重复候选复用缓存不耗预算；求解失败罚分且消耗预算。
"""

import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robust_design_lab.geometry import novel_structures as geo
from robust_design_lab.search import common_protocol as cp


# ── 地掩膜独立偏移（审阅 §1 反例） ────────────────────────────

class TestIndependentPocket(unittest.TestCase):

    def test_pocket_sides_move_independently(self):
        """±0.5μm 扰动下口袋四边各退 δ（宽高变化 ±2δ），中心不动。"""
        for fname in ("q_annulus", "m_variable_gap"):
            s0 = getattr(geo, fname)()
            o0 = geo.struct_to_options(s0)
            n0 = geo.nominal_pocket(s0).bounds
            for d in (0.5, -0.5):
                ev = geo.etch_variant(s0, d)
                op = geo.struct_to_options(ev)
                p = ev["pocket"].bounds
                # 四边分别检查（不只比总面积）
                self.assertAlmostEqual(p[0], n0[0] - d, places=6,
                                       msg=f"{fname} delta={d} 左边")
                self.assertAlmostEqual(p[1], n0[1] - d, places=6,
                                       msg=f"{fname} delta={d} 下边")
                self.assertAlmostEqual(p[2], n0[2] + d, places=6,
                                       msg=f"{fname} delta={d} 右边")
                self.assertAlmostEqual(p[3], n0[3] + d, places=6,
                                       msg=f"{fname} delta={d} 上边")
                # 宽高变化 = ±2δ（旧实现 ≈0，即被外接框抵消）
                self.assertAlmostEqual(op["pocket_w"] - o0["pocket_w"],
                                       2 * d, places=6, msg=fname)
                self.assertAlmostEqual(op["pocket_h"] - o0["pocket_h"],
                                       2 * d, places=6, msg=fname)
                # 口袋中心不动（各向同性同步刻蚀）
                self.assertAlmostEqual(op["pocket_cx"], o0["pocket_cx"],
                                       places=6)
                self.assertAlmostEqual(op["pocket_cy"], o0["pocket_cy"],
                                       places=6)

    def test_nominal_pocket_matches_legacy_formula(self):
        """名义路径不改变旧口径：口袋 = 名义金属外接框 + 2×150 边距。"""
        for fname in ("q_annulus", "m_variable_gap", "q_ellipse_idc",
                      "m_branch_comb"):
            s = getattr(geo, fname)()
            o = geo.struct_to_options(s)
            allg = [s["pad_left"], s["pad_right"]] + list(s["islands"])
            xs0 = min(g.bounds[0] for g in allg)
            xs1 = max(g.bounds[2] for g in allg)
            ys0 = min(g.bounds[1] for g in allg)
            ys1 = max(g.bounds[3] for g in allg)
            self.assertAlmostEqual(o["pocket_w"],
                                   (xs1 - xs0) + 2 * geo.POCKET_MARGIN_UM,
                                   places=6, msg=fname)
            self.assertAlmostEqual(o["pocket_h"],
                                   (ys1 - ys0) + 2 * geo.POCKET_MARGIN_UM,
                                   places=6, msg=fname)
            self.assertAlmostEqual(o["pocket_cx"], (xs0 + xs1) / 2, places=6)
            self.assertAlmostEqual(o["pocket_cy"], (ys0 + ys1) / 2, places=6)

    def test_etched_geometry_still_valid(self):
        for fname in ("q_annulus", "m_variable_gap"):
            s = getattr(geo, fname)()
            for d in (0.5, -0.5):
                errs = geo.check_structure(geo.etch_variant(s, d))
                self.assertEqual(errs, [], f"{fname} delta={d}: {errs}")


# ── 模拟物理：q_annulus 参数箱上的玩具评价器 ───────────────────

BOX = {"r_out": (230.0, 270.0), "r_in": (105.0, 125.0), "gap": (24.0, 34.0)}
START = dict(r_out=250.0, r_in=115.0, gap=28.0)


def f01_of(c_fF):
    ec = 19.36 / c_fF
    return math.sqrt(8 * 13.28 * ec) - ec


def fake_metrics(kw):
    """玩具物理：C_Σ 主随 r_out；陷阱区 r_out≈268 处 D 极低但 C_Σ 出窗。
    起点可行（C≈90、f01∈窗）。"""
    c = 90.0 + 2.0 * (kw["r_out"] - 250.0) + 0.2 * (kw["r_in"] - 115.0) \
        - 0.1 * (kw["gap"] - 28.0)
    d = 0.011 - 0.00006 * (kw["r_in"] - 105.0) \
        + 0.0004 * abs(kw["gap"] - 30.0)
    if abs(kw["r_out"] - 268.0) < 4.0:       # 出窗低漂移陷阱
        d = 0.0005
    return dict(D=d, C_sigma_fF=c, f01_GHz=f01_of(c), alpha_MHz=-19.36 / c * 1e3)


def make_session(evaluator=None, budget=9):
    s = cp.ProtocolSession("q_annulus", BOX, START,
                           evaluator=evaluator or fake_metrics, budget=budget)
    s.attach_start(fake_metrics(dict(START)))
    return s


class TestProtocolSession(unittest.TestCase):

    def test_budget_conservation_and_start_sharing(self):
        for arm in ("coordinate_descent", "cma_es", "bayesian_opt"):
            s = make_session()
            seed = cp.arm_seed("q_annulus", arm)
            out = cp.run_numeric_arm(arm, s, seed)
            self.assertEqual(s.n_new_evals, 9, arm)
            self.assertTrue(out["complete"], arm)
            # 起点未被重复求解（共享注入）
            start_solves = [r for r in s.records
                            if r["role"] == "start"]
            self.assertEqual(len(start_solves), 1, arm)

    def test_infeasible_low_D_never_wins(self):
        """出窗低漂移点（D=0.0005、C_Σ 出窗）objective=罚分，不进终选。"""
        s = make_session()
        seed = cp.arm_seed("q_annulus", "coordinate_descent")
        out = cp.run_numeric_arm("coordinate_descent", s, seed)
        self.assertTrue(out["final_feasible"])
        win = out["final_metrics"]
        lo, hi = cp.FEASIBILITY_WINDOW["C_sigma_fF"]
        self.assertTrue(lo <= win["C_sigma_fF"] <= hi)
        # 全部可行记录的 D 都 ≥ 终选 D；出窗记录即便 D 更低也不获胜
        feas_d = [r["D_raw"] for r in s.records if r["feasible"]]
        self.assertAlmostEqual(min(feas_d), out["final_D"], places=12)
        traps = [r for r in s.records
                 if r["metrics"] and r["D_raw"] < out["final_D"]]
        for t in traps:
            self.assertFalse(t["feasible"], "低 D 陷阱点必须是出窗的")

    def test_duplicate_reuse_does_not_burn_budget(self):
        s = make_session()
        s.evaluate(dict(START), source="t_start_reuse")   # 起点复用
        s.evaluate(dict(r_out=240.0, r_in=115.0, gap=28.0), source="t1")
        s.evaluate(dict(r_out=240.0, r_in=115.0, gap=28.0), source="t1b")
        self.assertEqual(s.n_new_evals, 1)
        self.assertEqual(s.n_duplicates, 2)
        self.assertEqual(len(s.records), 4)  # start + duplicate + new + duplicate
        self.assertTrue(all(isinstance(r["kw"]["r_out"], float)
                            for r in s.records))

    def test_failure_penalized_and_consumes_budget(self):
        def flaky(kw):
            if kw["r_out"] > 245:
                raise RuntimeError("求解失败")
            return fake_metrics(kw)
        s = make_session(evaluator=flaky, budget=3)
        s.evaluate(dict(r_out=260.0, r_in=115.0, gap=28.0))
        self.assertEqual(s.n_failures, 1)
        rec = s.records[-1]
        self.assertIsNone(rec["metrics"])
        self.assertEqual(rec["objective"], cp.FAIL_PENALTY)
        self.assertIn("RuntimeError", rec["error"])
        self.assertEqual(s.n_new_evals, 1)  # 失败也消耗预算
        with self.assertRaises(cp.BudgetExhausted):
            for i in range(5):
                s.evaluate(dict(r_out=240.0 - i, r_in=115.0, gap=28.0))

    def test_final_kw_rebuilds_structure_directly(self):
        s = make_session()
        out = cp.run_numeric_arm("cma_es", s, cp.arm_seed("q_annulus", "cma_es"))
        kw = out["final_kw"]
        self.assertIsNotNone(kw)
        for v in kw.values():
            self.assertIsInstance(v, float)   # 旧版为字符串，重建即 TypeError
        shape = geo.q_annulus(**kw)           # 直接重建，不再报 TypeError
        self.assertEqual(geo.check_structure(shape), [])

    def test_seed_nonzero_deterministic_and_saved(self):
        seed = cp.arm_seed("q_annulus", "cma_es")
        self.assertNotEqual(seed, 0)
        outs = []
        for _ in range(2):
            s = make_session()
            out = cp.run_numeric_arm("cma_es", s, seed)
            outs.append((out["cma_actual_seed"],
                         [r["kw"] for r in s.records
                          if r["role"] == "candidate"]))
        self.assertEqual(outs[0][0], seed)          # 实际种子落盘且非随机
        self.assertEqual(outs[0], outs[1])          # 同种子逐点重现
        with self.assertRaises(ValueError):
            cp.run_cma_es(make_session(), seed=0)   # seed=0 被协议拒绝


class TestVlmControlFlow(unittest.TestCase):

    @staticmethod
    def chat_ok(rnd, cur, cur_rec, history):
        # 每轮 3 个不同合法提案（含非法名与越界值：过滤/吸附）× 3 轮 = 9
        return json.dumps([
            {"param": "r_in", "value": 111.0 - rnd,
             "reason": "缩小内径降 D"},
            {"param": "bad_name", "value": 1.0, "reason": "非法名"},
            {"param": "gap", "value": 30.0 + rnd, "reason": "微调间隙"},
            {"param": "r_out", "value": 235.0 + rnd, "reason": "试探外径"},
        ])

    def test_vlm_budget_and_completion(self):
        s = make_session()
        out = cp.run_vlm(s, self.chat_ok)
        self.assertEqual(s.n_new_evals, 9)
        self.assertTrue(out["complete"])
        self.assertIsNotNone(out["final_feasible"])
        self.assertEqual(len(out["rounds"]), 3)

    def test_empty_reply_not_complete(self):
        s = make_session()
        out = cp.run_vlm(s, lambda *a: "模型超时无有效内容")
        self.assertFalse(out["complete"])           # 旧版空回复也标完成
        self.assertEqual(s.n_new_evals, 0)
        self.assertTrue(all("error" in r for r in out["rounds"]))
        # 无新增评价：终选退回共享起点（已评价最优），但 complete=False
        self.assertEqual(out["final_kw"], dict(START))
        self.assertAlmostEqual(out["final_D"],
                               fake_metrics(dict(START))["D"], places=12)

    def test_vlm_stubborn_repeats_leave_budget_underspent(self):
        """模型固执重复同一提案：重复点复用缓存不耗预算，预算不守恒
        → complete=False（如实标记模型无法提供新点，不虚报完成）。"""
        same = json.dumps([{"param": "r_in", "value": 111.0, "reason": "r"}])
        s = make_session()
        out = cp.run_vlm(s, lambda rnd, *a: same)
        self.assertEqual(s.n_new_evals, 1)   # 第二三轮为重复点（duplicate）
        self.assertEqual(s.n_duplicates, 2)
        self.assertFalse(out["complete"])

    def test_vlm_rejects_infeasible_low_D(self):
        """VLM 提案落到出窗低漂移陷阱（r_out≈268）：objective=罚分，
        不移动当前点，也不进终选。"""
        replies = [json.dumps([{"param": "r_out", "value": 268.0,
                                "reason": "低 D"}]),
                   json.dumps([{"param": "r_in", "value": 106.0,
                                "reason": "微调"}]),
                   json.dumps([{"param": "gap", "value": 30.0,
                                "reason": "对中"}])]
        s = make_session()
        out = cp.run_vlm(s, lambda rnd, *a: replies[rnd])
        infeas = [h for h in out["history"]
                  if h["kw"]["r_out"] == 268.0]
        self.assertTrue(infeas)
        self.assertFalse(infeas[0]["feasible"])
        self.assertEqual(infeas[0]["objective"], cp.FAIL_PENALTY)
        self.assertTrue(out["final_feasible"])
        self.assertGreaterEqual(out["final_D"], 0.0005)

    def test_parse_proposals(self):
        box = BOX
        props = cp.parse_proposals(
            "```json\n[{\"param\":\"r_in\",\"value\":111,\"reason\":\"r\"}]\n```",
            box, dict(START))
        self.assertEqual(props, [("r_in", 111.0, "r")])
        # 单 dict、越界吸附、非法名剔除
        props = cp.parse_proposals(
            json.dumps({"param": "gap", "value": 99.0}), box, dict(START))
        self.assertEqual(props, [("gap", 34.0, "")])
        with self.assertRaises(ValueError):
            cp.parse_proposals("[]", box, dict(START))
        with self.assertRaises(ValueError):
            cp.parse_proposals("完全不是 JSON", box, dict(START))


class TestArmSeed(unittest.TestCase):

    def test_arm_seed_stable_nonzero(self):
        a = cp.arm_seed("q_annulus", "cma_es")
        self.assertNotEqual(a, 0)
        self.assertEqual(a, cp.arm_seed("q_annulus", "cma_es"))
        self.assertNotEqual(a, cp.arm_seed("m_variable_gap", "cma_es"))


if __name__ == "__main__":
    unittest.main()
