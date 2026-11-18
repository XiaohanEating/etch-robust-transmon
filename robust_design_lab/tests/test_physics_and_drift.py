"""Analytic regression tests, independent of solver output."""
import math
import unittest
import numpy as np
from robust_design_lab.evaluator.participation import EPS0, layer_energy
from robust_design_lab.objectives.drift import drift_metrics, geometry_only, lex_key


class InterfaceEnergyTests(unittest.TestCase):
    def test_metal_normal_displacement(self):
        E = np.array([[50.0], [60.0], [2.0]])
        actual = layer_energy(E, np.array([3.0]), np.array([True]), 4e-9, 10.0, 11.45, True)
        expected = 0.5 * EPS0 * 4e-9 * 3.0 * 11.45 ** 2 / 10.0 * 4.0
        self.assertAlmostEqual(actual / expected, 1.0, places=12)

    def test_sa_includes_tangential_field(self):
        E = np.array([[3.0], [4.0], [0.0]])
        actual = layer_energy(E, np.array([2.0]), np.array([True]), 3e-9, 10.0, 1.0, False)
        self.assertAlmostEqual(actual / (0.5 * EPS0 * 3e-9 * 2.0 * 10.0 * 25.0), 1.0, places=12)

    def test_interface_side_invariance(self):
        air, sub = np.array([[3.0], [4.0], [22.9]]), np.array([[3.0], [4.0], [2.0]])
        args = (np.ones(1), np.ones(1, dtype=bool), 3e-9, 10.0)
        a, b = layer_energy(air, *args, 1.0, False), layer_energy(sub, *args, 11.45, False)
        self.assertAlmostEqual(a / b, 1.0, places=12)


class DriftTests(unittest.TestCase):
    nominal = dict(T1_us=100.0, f01_GHz=5.0, alpha_MHz=-200.0)

    def test_small_sample_q95(self):
        rows = {str(i): dict(self.nominal, T1_us=100.0 * math.exp(i / 10)) for i in range(7)}
        result = drift_metrics(self.nominal, rows)
        self.assertAlmostEqual(result['D_T1']['q95'], 0.6)
        self.assertTrue(result['valid'])

    def test_failed_scene_cannot_win(self):
        result = drift_metrics(self.nominal, {'ok': dict(self.nominal), 'bad': dict(failed='Mesh failed')})
        self.assertEqual(result['n_failed'], 1)
        self.assertEqual(result['n_scenarios'], 2)
        self.assertTrue(math.isinf(result['D_T1']['max']))
        self.assertFalse(result['valid'])

    def test_sign_change(self):
        result = drift_metrics(self.nominal, {'bad': dict(self.nominal, alpha_MHz=10.0)})
        self.assertFalse(result['valid'])
        self.assertTrue(math.isinf(result['D_alpha']['max']))

    def test_missing_metric(self):
        self.assertFalse(drift_metrics(self.nominal, {'bad': {}})['valid'])

    def test_json_null_cannot_enter_ranking(self):
        result=drift_metrics(self.nominal,{'bad':dict(self.nominal,T1_us=None)})
        self.assertFalse(result['valid'])
        self.assertEqual(lex_key(dict(valid=False),dict(valid=False)),(math.inf,)*4)

    def test_geometry_excludes_joint(self):
        self.assertTrue(geometry_only(dict(etch_um=0.25, lj_rel=1.0, ti_rel=1.0)))
        self.assertFalse(geometry_only(dict(etch_um=0.25, lj_rel=1.05, ti_rel=1.0)))

    def test_geometry_is_second_ranking_component(self):
        a = dict(D_T1=dict(max=0.2), D_f01=dict(max=0.01), D_alpha=dict(max=0.03))
        self.assertEqual(lex_key(a, dict(D_T1=dict(max=0.04))), (0.2, 0.04, 0.01, 0.03))
        with self.assertRaises(ValueError):
            lex_key(a)


if __name__ == '__main__':
    unittest.main()
