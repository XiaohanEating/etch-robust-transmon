# -*- coding: utf-8 -*-
"""内孔导出守恒：源 shapely ↔ options ↔ Metal 组件三层面面积一致。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shapely.geometry import Polygon

from robust_design_lab.geometry.novel_structures import (
    q_annulus, struct_to_options, _novel_component)


class TestHoleExport(unittest.TestCase):
    def test_options_area_preserved(self):
        s = q_annulus()
        opts = struct_to_options(s)
        # options 顶点表重建（外环+孔）
        back = Polygon(opts['pad_left'][0], opts['pad_left'][1:])
        self.assertAlmostEqual(back.area * 1e6, s['pad_left'].area, delta=0.5)

    def test_component_poly_has_hole(self):
        s = q_annulus()
        opts = struct_to_options(s)
        cls = _novel_component()
        poly = cls._to_poly(opts['pad_left'])
        self.assertEqual(len(poly.interiors), 1)
        self.assertAlmostEqual(poly.area * 1e6, s['pad_left'].area, delta=0.5)

    def test_flat_list_still_works(self):
        cls = _novel_component()
        poly = cls._to_poly([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
                             [0.0, 0.0]])
        self.assertAlmostEqual(poly.area, 1.0)
        self.assertEqual(len(poly.interiors), 0)


if __name__ == "__main__":
    unittest.main()
