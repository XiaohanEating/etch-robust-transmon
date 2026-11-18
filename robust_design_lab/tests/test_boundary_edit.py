# -*- coding: utf-8 -*-
"""区域边界编辑算子与轮制序列编辑协议测试（全部代理评价，无求解/模型）。

覆盖：
- 动作空间全量扫描（环形全有效；变间隙内环段与结区保护段无效）；
- 各算子几何语义（notch/tab 面积增减、内环=孔变化）；
- 结区保护与面积限幅触发 InvalidEditError（no_solve 标记）；
- 轮制 run_round_based：预算守恒、同动作不同结构缓存不误判、
  采纳移动、proposer 失败轮不虚报完成；
- int 维 box 吸附与编码往返。
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robust_design_lab.geometry import boundary_edit as be
from robust_design_lab.geometry import novel_structures as geo
from robust_design_lab.search import common_protocol as cp


def annulus_struct():
    return geo.q_annulus(r_out=253.9, r_in=121.5, gap=33.8)


def vgap_struct():
    return geo.m_variable_gap(H=480.0, depth=290.0, g0=32.0, g_edge=88.0)


class TestBoundaryEdit(unittest.TestCase):

    def test_full_action_space(self):
        """环形：92 有效；4 个动作被全支撑结区保护拒绝（2026-11-04
        审阅反例修复后新增拦截——旧版只查段中点时会放行）。"""
        s = annulus_struct()
        ok = footprint_rej = 0
        for op in be.EDIT_OPS:
            for seg in range(be.N_SEG):
                try:
                    be.apply_edit(s, op, seg, 0.6, 0.5)
                    ok += 1
                except be.InvalidEditError as e:
                    if "保护圆" in str(e):
                        footprint_rej += 1
                    else:
                        self.fail(f"环形意外无效: {op}{seg}: {e}")
        self.assertEqual(ok, 92)
        self.assertEqual(footprint_rej, 4)

    def test_vgap_invalid_segments_accounted(self):
        """变间隙（无内孔）：48 内环段无效 + 4 全支撑保护 + 3 岛间距
        + 1 刻蚀变体失败 = 56 无效；40 有效。分类全部可解释。"""
        s = vgap_struct()
        counts = dict(ok=0, inner=0, footprint=0, spacing=0, etched=0)
        for op in be.EDIT_OPS:
            for seg in range(be.N_SEG):
                try:
                    be.apply_edit(s, op, seg, 0.6, 0.5)
                    counts["ok"] += 1
                except be.InvalidEditError as e:
                    msg = str(e)
                    if "内环" in msg:
                        counts["inner"] += 1
                    elif "保护圆" in msg:
                        counts["footprint"] += 1
                    elif "变体检查失败" in msg:
                        counts["etched"] += 1
                    else:
                        counts["spacing"] += 1
                    self.assertTrue(getattr(e, "no_solve", False))
        self.assertEqual(counts, dict(ok=40, inner=48, footprint=4,
                                       spacing=3, etched=1))

    def test_op_geometry_semantics(self):
        s = annulus_struct()
        a0 = s["pad_left"].area
        notch = be.apply_edit(s, "notch", be.seg_encode("L", "outer", 4),
                              0.6, 0.5)
        tab = be.apply_edit(s, "tab", be.seg_encode("L", "outer", 4),
                            0.6, 0.5)
        self.assertLess(notch["pad_left"].area, a0)   # 开槽=金属减少
        self.assertGreater(tab["pad_left"].area, a0)  # 外伸舌=金属增加
        # 不修改原结构
        self.assertAlmostEqual(s["pad_left"].area, a0)

    def test_inner_ring_edits_change_hole(self):
        s = annulus_struct()
        hole0 = abs(s["pad_left"].interiors[0].length)
        s1 = be.apply_edit(s, "tab", be.seg_encode("L", "inner", 2), 0.5, 0.5)
        # 内环 tab=金属伸入孔=孔周长变化
        self.assertNotAlmostEqual(abs(s1["pad_left"].interiors[0].length),
                                  hole0, places=1)

    def test_junction_protection_direct(self):
        """结区保护：构造距段中点很近的结线，编辑该段必须被拒。"""
        import math
        s = annulus_struct()
        pts = be._densified_ccw(s["pad_left"].exterior)
        seg = be._ring_segments(pts)[2]
        mid = seg["mid"]
        # 结线一端贴住段中点（另一端放到右侧岛），段中点距端点 ~0 < 保护半径
        s2 = dict(s, junction=__import__("shapely.geometry", fromlist=[
            "LineString"]).LineString(
            [mid, (abs(s["pad_right"].bounds[0]) , 0.0)]))
        try:
            be.apply_edit(s2, "bulge", be.seg_encode("L", "outer", 2),
                          0.5, 0.5)
            self.fail("结区保护未触发")
        except be.InvalidEditError as e:
            self.assertIn("保护圆", str(e))
        # 原结构（结线在两岛之间）段 2 距结线远于保护半径，编辑正常
        be.apply_edit(s, "bulge", be.seg_encode("L", "outer", 2), 0.5, 0.5)

    def test_eval_struct_rejects_unconverged(self):
        """未收敛场景按失败拒收（不产生可入终选的有效漂移——
        2026-11-04 审阅约束）。"""
        from unittest.mock import patch
        from robust_design_lab.experiments import region_edit_pilot as rp

        def fake_run_q3d(variant, comp, family, name="Q1", max_passes=6,
                         percent_error=None):
            return dict(nets=["ground_main_plane", "pad_left_Q1",
                              "pad_right_Q1"],
                        matrix_fF=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        C_sigma_fF=90.0, solve_s=1.0, project="p",
                        project_path="p", capacitance_all_passes_fF={},
                        passes_units="fF", percent_error=0.5,
                        converged=False)
        s = annulus_struct()
        with patch("robust_design_lab.experiments.q3d_shape_sweep.run_q3d",
                   fake_run_q3d):
            with self.assertRaises(RuntimeError) as cm:
                rp.eval_struct(s)
            self.assertIn("未收敛", str(cm.exception))

    def test_protection_full_footprint_deterministic(self):
        """全支撑保护（2026-11-04 审阅反例）：长条矩形岛，结端点放在
        上边中段中点正上方 70μm（圆外 20μm）；tab 外伸 80μm 贯穿端点。
        旧版只查段中点（70>50）会放行，新版按几何差分必须拒绝。"""
        import math
        from shapely.geometry import LineString, Polygon as Poly
        pl = Poly([(-300, -20), (-16, -20), (-16, 20), (-300, 20)])
        pr = Poly([(16, -20), (300, -20), (300, 20), (16, 20)])
        pts = be._densified_ccw(pl.exterior)
        segs = be._ring_segments(pts)
        # 上边（y=+20）中部段：在上边段中找中点 x 最接近 -158
        top = [sg for sg in segs if sg["mid"][1] > 0]
        cand = min(top, key=lambda sg: abs(sg["mid"][0] + 158))
        region = segs.index(cand)
        end = (cand["mid"][0], cand["mid"][1] + 70.0)  # 中点正上方 70μm
        s = dict(pad_left=pl, pad_right=pr, islands=[],
                 junction=LineString([end, (300.0, cand["mid"][1] + 70.0)]),
                 label="rect_test")
        self.assertGreater(min(math.dist(cand["mid"], p)
                               for p in s["junction"].coords),
                           be.PROTECT_R_UM)  # 中点在保护圆外
        try:
            be.apply_edit(s, "tab", be.seg_encode("L", "outer", region),
                          1.0, 0.5, check_etched=False)
            self.fail("全支撑保护未触发（旧版只查段中点的缺陷）")
        except be.InvalidEditError as e:
            self.assertIn("保护圆", str(e))

    def test_action_kw_roundtrip(self):
        kw = be.action_kw("notch", "R", "inner", 5, 0.7, 0.45)
        op, seg, mag, width = be.action_from_kw(kw)
        self.assertEqual((op, seg, mag, width),
                         ("notch", be.seg_encode("R", "inner", 5), 0.7, 0.45))
        self.assertEqual(be.seg_decode(seg), ("R", "inner", 5))
        # 越界吸附
        kw2 = be.action_kw("tab", "L", "outer", 0, 99.0, 0.1)
        self.assertEqual(kw2["mag"], be.MAG_RANGE[1])
        self.assertEqual(kw2["width"], be.WIDTH_RANGE[0])


EDIT_BOX = {"op": (0, 2, "int"), "seg": (0, 31, "int"),
            "mag": (0.2, 1.0), "width": (0.3, 1.0)}
START_ACTION = dict(op=1, seg=10, mag=0.6, width=0.5)


def toy_eval_action(kw):
    """玩具评价：D 依赖 (op,seg) 组合与 mag；部分动作无效（no_solve）。"""
    if int(kw["seg"]) == 15:  # 模拟无效编辑段
        raise be.InvalidEditError("模拟无效段")
    d = 0.006 - 0.0004 * ((int(kw["seg"]) % 8) - 4.0) ** 0.5 \
        - 0.001 * int(kw["op"]) * kw["mag"]
    c = 90.0 + 0.5 * (kw["mag"] - 0.6)
    return dict(D=max(d, 0.001), C_sigma_fF=c, f01_GHz=4.7, alpha_MHz=-215.0)


def make_session(evaluator=None, budget=9):
    s = cp.ProtocolSession("edit_pilot", EDIT_BOX, START_ACTION,
                           evaluator=evaluator or toy_eval_action,
                           budget=budget)
    s.attach_start(dict(D=0.0058, C_sigma_fF=90.0, f01_GHz=4.7,
                        alpha_MHz=-215.0))
    return s


class TestRoundProtocol(unittest.TestCase):

    def test_int_box_snap_and_roundtrip(self):
        s = make_session()
        x = s.encode(dict(op=2, seg=31, mag=0.9, width=0.4))
        kw = s.decode(x)
        self.assertEqual(kw["op"], 2.0)
        self.assertEqual(kw["seg"], 31.0)
        self.assertEqual(s.normalize(dict(op=2.4, seg=40, mag=1.0, width=0.3)),
                         dict(op=2.0, seg=31.0, mag=1.0, width=0.3))

    def test_round_based_budget_and_adoption(self):
        for maker in ("coord", "cma", "bo"):
            s = make_session()
            seed = cp.arm_seed("edit", maker)
            if maker == "coord":
                prop = cp.make_coord_proposer(s, __import__("numpy")
                                              .random.default_rng(seed))
            elif maker == "cma":
                prop = cp.make_cma_proposer(s, seed)
            else:
                prop = cp.make_bo_proposer(s, seed)
            out = cp.run_round_based(s, prop)
            self.assertEqual(s.n_new_evals, 9, maker)   # 预算守恒
            self.assertTrue(out["complete"], maker)
            self.assertEqual(len(out["rounds"]), 3, maker)
            self.assertIsNotNone(out["final_action"], maker)

    def test_invalid_action_penalized_no_solve(self):
        """无效段（seg=15）→ 罚分消耗预算、计入 n_invalid、无指标。"""
        s = make_session()
        s.evaluate(dict(op=0, seg=15, mag=0.5, width=0.5), source="t")
        self.assertEqual(s.n_invalid, 1)
        self.assertEqual(s.n_new_evals, 1)
        rec = s.records[-1]
        self.assertIsNone(rec["metrics"])
        self.assertEqual(rec["objective"], cp.FAIL_PENALTY)

    def test_context_cache_separates_structures(self):
        """同一动作在不同结构快照上是不同候选，不误复用缓存。"""
        s = make_session()
        r1 = s.evaluate(dict(op=0, seg=3, mag=0.5, width=0.5),
                        source="t", context="struct_v0")
        r2 = s.evaluate(dict(op=0, seg=3, mag=0.5, width=0.5),
                        source="t", context="struct_v1")
        self.assertEqual(s.n_new_evals, 2)  # 两次都是新评价
        self.assertEqual(s.n_duplicates, 0)
        r3 = s.evaluate(dict(op=0, seg=3, mag=0.5, width=0.5),
                        source="t", context="struct_v1")
        self.assertEqual(s.n_duplicates, 1)  # 同结构同动作才复用
        self.assertEqual(s.n_new_evals, 2)

    def test_empty_proposer_not_complete(self):
        s = make_session()
        out = cp.run_round_based(s, lambda *a: [])
        self.assertFalse(out["complete"])
        self.assertEqual(s.n_new_evals, 0)

    def test_forced_policy_accumulates(self):
        """forced=强制累积：每轮移动到轮内最优（即使劣于当前），
        3 轮产生 3 步累积；全罚分轮不移动；终选仍是可行点最优。"""
        s = make_session()
        # 轮 0 两个提案均劣于起点（toy: seg%8=4 → D=0.006 > 起点 0.0058）；
        # 轮 1 全无效（seg=15）；轮 2 一个劣提案
        props = [
            [dict(op=0, seg=4, mag=0.5, width=0.5),
             dict(op=0, seg=12, mag=0.5, width=0.5)],
            [dict(op=0, seg=15, mag=0.5, width=0.5)],
            [dict(op=0, seg=20, mag=0.5, width=0.5)],
        ]
        it = iter(props)

        def proposer(rnd, *a):
            return next(it)
        out = cp.run_round_based(s, proposer, move_policy="forced")
        self.assertEqual(s.n_new_evals, 4)        # 2 + 1(无效罚分) + 1
        self.assertEqual(s.n_invalid, 1)
        self.assertEqual(len(out["adopted"]), 2)  # 轮 1 全无效不移动
        self.assertTrue(all(a.get("forced") for a in out["adopted"]))
        # 终选 = 全部可行记录最优：全部提案劣于起点 → 终选回退起点
        self.assertAlmostEqual(out["final_D"], 0.0058, places=12)
        self.assertEqual(out["final_kw"], dict(START_ACTION))

    def test_round_based_adopts_improvement(self):
        """固定 proposer 给出改进动作：adopted 记录且锚更新。"""
        s = make_session()
        props = [dict(op=2, seg=20, mag=0.9, width=0.5),
                 dict(op=0, seg=25, mag=0.4, width=0.4),
                 dict(op=1, seg=1, mag=0.5, width=0.6)]
        it = iter(props + [dict(op=2, seg=21, mag=0.9, width=0.5)] * 9)
        out = cp.run_round_based(s, lambda *a: [next(it)])
        self.assertTrue(out["adopted"])
        self.assertEqual(out["final_action"]["op"], 2.0)
        self.assertLess(out["final_D"], 0.0058)


class TestBatchIntegrity(unittest.TestCase):
    """2026-11-06 审阅 A 计划四类离线反例：0.1%配6轮、批次覆盖、
    终选几何错位、无孔分支全链路。"""

    def _fake_run_q3d(self, store):
        def f(variant, comp, family, name="Q1", max_passes=6,
              percent_error=None):
            store.update(max_passes=max_passes, pct=percent_error)
            return dict(nets=["g", "pl", "pr"], matrix_fF=[[1, 0, 0],
                        [0, 1, 0], [0, 0, 1]], C_sigma_fF=90.0, solve_s=1.0,
                        project="p", project_path="p",
                        capacitance_all_passes_fF={}, passes_units="fF",
                        percent_error=percent_error or 0.5, converged=True)
        return f

    def test_pct_passes_binding(self):
        """反例1：0.1% 精度档必须绑定 20 轮上限（旧版传 pct 不传轮数，
        7/7 评价全部不收敛）。"""
        from unittest.mock import patch
        from robust_design_lab.experiments import fresh_start_search as fs
        store = {}
        fs.SEARCH_PCT[0] = 0.1
        try:
            ev = fs.make_evaluator(dict(struct=annulus_struct()))
            with patch("robust_design_lab.experiments.q3d_shape_sweep.run_q3d",
                       self._fake_run_q3d(store)):
                ev(dict(op=1, seg=4, mag=0.5, width=0.5))
            self.assertEqual(store["max_passes"], 20)
            self.assertEqual(store["pct"], 0.1)
        finally:
            fs.SEARCH_PCT[0] = 0.5

    def test_batch_runs_isolation(self):
        """反例2：0.5% 批的完成记录不得让 0.1% 批被跳过。"""
        import tempfile
        from unittest.mock import patch as _patch
        from robust_design_lab.experiments import fresh_start_search as fs
        with _patch.object(fs, "OUT", Path(tempfile.mkdtemp())):
            fs.SEARCH_PCT[0], fs.MOVE_POLICY[0], fs.RUN_SEED[0] = 0.5, "greedy", None
            p05 = fs.runs_path()
            p05.write_text(json.dumps(dict(arm="cma_es",
                                            start="annulus_default",
                                            complete=True)) + "\n",
                           encoding="utf-8")
            self.assertIn(("cma_es", "annulus_default", None),
                          fs.done_keys())
            fs.SEARCH_PCT[0], fs.MOVE_POLICY[0] = 0.1, "forced"
            self.assertNotEqual(fs.runs_path(), p05)
            self.assertEqual(fs.done_keys(), set())  # 跨批不误跳
        fs.SEARCH_PCT[0], fs.MOVE_POLICY[0], fs.RUN_SEED[0] = 0.5, "greedy", None

    def test_final_geometry_matches_best_record(self):
        """反例3：forced 累积末态≠最优记录时，终选几何必须按最优记录
        的轮前快照+动作重建，不得以末态替代。"""
        import tempfile
        from unittest.mock import patch as _patch
        from robust_design_lab.experiments import fresh_start_search as fs

        def fake_eval(struct, max_passes=None, percent_error=None):
            # D 只依赖结构指纹末位数字（可区分）；可行
            d = 0.004 + 0.0001 * (int(struct_fp_local(struct)[-1], 16))
            return dict(D=d, C_sigma_fF=90.0, f01_GHz=4.7, alpha_MHz=-215.0,
                        scenarios={})

        def struct_fp_local(struct):
            import zlib
            return "%x" % (zlib.crc32((struct["pad_left"].wkt
                                       + struct["pad_right"].wkt)
                                      .encode()) % 16)

        props = [
            [dict(op=1, seg=4, mag=0.9, width=0.5),   # 轮0 最优
             dict(op=1, seg=6, mag=0.5, width=0.5)],
            [dict(op=2, seg=8, mag=0.5, width=0.5)],   # forced 移动（劣）
            [dict(op=0, seg=12, mag=0.5, width=0.5)],  # forced 再移动
        ]
        it = iter(props)
        with _patch.object(fs, "OUT", Path(tempfile.mkdtemp())), \
                _patch.object(fs, "eval_struct", fake_eval), \
                _patch.object(fs, "start_eval",
                              lambda s: dict(D=0.005, C_sigma_fF=90.0,
                                             f01_GHz=4.7, alpha_MHz=-215.0)), \
                _patch.object(fs.cp, "make_cma_proposer",
                              lambda session, seed: (
                                  lambda rnd, *a: next(it))):
            fs.SEARCH_PCT[0], fs.MOVE_POLICY[0] = 0.1, "forced"
            entry = fs.run_arm("cma_es", "annulus_default")
        fs.SEARCH_PCT[0], fs.MOVE_POLICY[0] = 0.5, "greedy"
        # forced 下 3 轮全部移动（提案 2+1+1=4 次，预算未满是如实标记）
        self.assertEqual(len(entry["adopted"]), 3)
        recs = [r for r in entry["records"] if r.get("metrics")]
        best = min(recs, key=lambda r: r["objective"])
        # 终选几何 = 最优记录的几何（指纹一致），且来自轮 0 快照而非末态链
        self.assertEqual(entry["final_struct_fp"],
                         best["metrics"]["struct_fp"])
        self.assertTrue(str(entry["final_record_source"]).startswith("r0"))

    def test_branch_comb_full_pipeline(self):
        """反例4：无孔的分支梳起点全链路（空间表+代表性编辑）无未分类
        异常。"""
        from robust_design_lab.geometry import spatial_feedback as sf
        s = geo.m_branch_comb()
        tbl = sf.spatial_table(s)
        self.assertEqual(len(tbl), 32)
        self.assertIn("无内环", tbl[8])
        for op in be.EDIT_OPS:
            for seg in (0, 3, 16, 19):
                try:
                    be.apply_edit(s, op, seg, 0.6, 0.5)
                except be.InvalidEditError:
                    pass  # 分类拒绝是预期行为


class TestFollowupRFixes(unittest.TestCase):
    """2026-11-06 复核 R1/R2 三个反例的修复验证。"""

    def test_r1_unconverged_rejected(self):
        """R1：任一场景 converged=False → evaluate_object 拒收（无 D）。"""
        from unittest.mock import patch
        from robust_design_lab.experiments import verify_candidates as vc

        calls = {"n": 0}

        def fake(variant, comp, family, name="Q1", max_passes=6,
                 percent_error=None):
            calls["n"] += 1
            return dict(nets=["g", "pl", "pr"], matrix_fF=[[1, 0, 0],
                        [0, 1, 0], [0, 0, 1]], C_sigma_fF=90.0, solve_s=1.0,
                        project="p", project_path="p",
                        capacitance_all_passes_fF={}, passes_units="fF",
                        percent_error=percent_error or 0.5,
                        converged=(calls["n"] != 2))  # 第 2 场景不收敛
        with patch("robust_design_lab.experiments.q3d_shape_sweep.run_q3d",
                   fake):
            with self.assertRaises(RuntimeError) as cm:
                vc.evaluate_object("t", dict(kind="novel",
                                             family="q_annulus",
                                             kw=dict(r_out=250.0, r_in=115.0,
                                                     gap=28.0)))
        self.assertIn("非 True", str(cm.exception))

    def test_r2a_same_action_new_state_evaluated(self):
        """R2a：同一动作作用于轮前更新的几何 = 新候选，两轮各评价一次
        （旧版 kw==cur 跳过使两轮只评价一次）。"""
        s = make_session()
        same = dict(op=0, seg=4, mag=0.5, width=0.5)
        ctxs = ["state_v0", "state_v1"]

        def proposer(rnd, *a):
            return [dict(same)]

        out = cp.run_round_based(
            s, proposer, rounds=2, per_round=1,
            context_fn=lambda rnd: ctxs[rnd])
        self.assertEqual(s.n_new_evals, 2)      # 两个状态各评一次
        self.assertEqual(s.n_duplicates, 0)
        recs = [r for r in s.records if r["role"] == "candidate"]
        self.assertEqual({r["context"] for r in recs},
                         {"state_v0", "state_v1"})

    def test_r2b_runs_path_has_source_version(self):
        """R2b：runs 文件名含驱动源码版本哈希（不同版本互不覆盖）。"""
        import re as _re
        import tempfile
        from unittest.mock import patch as _patch
        from robust_design_lab.experiments import fresh_start_search as fs
        with _patch.object(fs, "OUT", Path(tempfile.mkdtemp())):
            p = fs.runs_path()
            self.assertRegex(
                p.name,
                _re.compile(r"runs_p0\.5_greedy_v[0-9a-f]{8}_s(dflt|[0-9]+)"
                            r"\.jsonl"))


class TestRuleProposer(unittest.TestCase):
    """D 首批物理规则 proposer：同信息（空间表）、事前固定规则、
    合法过滤（apply_edit 试探）。"""

    def test_rule_pool_no_prefilter(self):
        """2026-11-07 复核 P1 修正语义：规则池不预滤非法动作——
        与模型/CMA 同权在协议层受罚；池按规则排序、无重复。"""
        from robust_design_lab.experiments import fresh_start_search as fs
        from robust_design_lab.geometry import novel_structures as geo
        for mk in (lambda: geo.q_annulus(), fs._simple_pad):
            struct = mk()
            prop = fs.make_rule_proposer(dict(struct=struct))
            cands = prop(0, None, None, [])
            self.assertTrue(1 <= len(cands) <= fs.RULE_POOL_SIZE)
            self.assertEqual(len({tuple(sorted(k.items()))
                                  for k in cands}), len(cands))  # 无重复
            # 无预滤：池内允许出现会在协议层受罚的非法动作

    def test_rule_first_is_min_gap_bulge(self):
        from robust_design_lab.experiments import fresh_start_search as fs
        from robust_design_lab.geometry import spatial_feedback as sf
        from robust_design_lab.geometry import novel_structures as geo
        struct = geo.q_annulus()
        rows = [r for r in sf.segment_metrics(struct)
                if r.get("valid") and r["ring"] == "outer"]
        want = min(rows, key=lambda r: r["gap_min"])
        prop = fs.make_rule_proposer(dict(struct=struct))
        first = prop(0, None, None, [])[0]
        expect = be.action_kw("bulge", want["island"], "outer",
                              want["region"], 0.6, 0.5)
        self.assertEqual(first["seg"], expect["seg"])


class TestParseStrict(unittest.TestCase):
    """2026-11-07 复核 P1：region 越界必须显式拒绝（旧版静默错译到
    另一位置），丢弃项带原因返回。"""

    def test_region_out_of_range_rejected(self):
        from robust_design_lab.experiments.spatial_probe import parse_actions
        raw = json.dumps([{"op": "notch", "island": "L", "ring": "inner",
                           "region": 15, "mag": 0.5, "width": 0.5}])
        acts, dropped = parse_actions(raw)
        self.assertEqual(len(acts), 0)
        self.assertEqual(len(dropped), 1)
        self.assertIn("越界", dropped[0]["reason"])

    def test_valid_triples_and_seg_passthrough(self):
        from robust_design_lab.experiments.spatial_probe import parse_actions
        raw = json.dumps([
            {"op": "bulge", "island": "R", "ring": "inner", "region": 7,
             "mag": 0.5, "width": 0.5},
            {"op": "notch", "seg": 27, "mag": 0.4, "width": 0.6}])
        acts, dropped = parse_actions(raw)
        self.assertEqual(len(dropped), 0)
        self.assertEqual([int(a[0]["seg"]) for a in acts], [31, 27])

    def test_bad_island_rejected(self):
        from robust_design_lab.experiments.spatial_probe import parse_actions
        raw = json.dumps([{"op": "tab", "island": "X", "ring": "outer",
                           "region": 2, "mag": 0.5, "width": 0.5}])
        acts, dropped = parse_actions(raw)
        self.assertEqual(len(acts), 0)
        self.assertIn("island", dropped[0]["reason"])


class TestParseFailureBudget(unittest.TestCase):
    """2026-11-07 复核：解析失败占登记尝试（与非法动作同权）。"""

    def test_parse_failure_consumes_budget(self):
        s = make_session(budget=3)
        r1 = s.register_parse_failure("t_drop0")
        s.evaluate(dict(op=0, seg=3, mag=0.5, width=0.5), source="t1")
        r2 = s.register_parse_failure("t_drop1")
        self.assertEqual(s.n_new_evals, 3)          # 丢弃两次各占一次
        self.assertEqual(r1["objective"], cp.FAIL_PENALTY)
        self.assertIsNone(r2["metrics"])
        with self.assertRaises(cp.BudgetExhausted):
            s.evaluate(dict(op=0, seg=5, mag=0.5, width=0.5), source="t2")


if __name__ == "__main__":
    unittest.main()
