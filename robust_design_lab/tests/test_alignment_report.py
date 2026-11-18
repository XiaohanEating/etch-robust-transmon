"""Independent drift cases guard against hidden directional errors."""
import copy
import math
import unittest

from robust_design_lab.experiments.alignment_report import compare_triplets, source_issues
from robust_design_lab.configs import frozen


def triplet(plus=0.02, minus=-0.01):
    return {scene: dict(C_q_fF=60, p_SM=1e-4, p_MA=1e-6, p_SA=1e-4,
                        T1_us=100*math.exp(delta)) for scene, delta in
            (('nominal', 0), ('etch_plus', plus), ('etch_minus', minus))}


class AlignmentReportTests(unittest.TestCase):
    def test_missing_scene_cannot_pass(self):
        rows = triplet()
        del rows['etch_minus']
        self.assertFalse(compare_triplets(triplet(), rows)['passed'])

    def test_reversed_drift_directions_fail_even_when_maxima_match(self):
        result = compare_triplets(triplet(0.02, -0.02), triplet(-0.02, 0.02))
        self.assertFalse(result['passed'])
        for row in result['etch_log_drift_by_direction'].values():
            self.assertAlmostEqual(row['absolute_drift_change'], 0)
            self.assertAlmostEqual(row['signed_difference_change'], 0.04)
            self.assertFalse(row['direction_same'])

    def test_etched_participation_failure_cannot_hide_behind_nominal(self):
        rows = triplet()
        rows['etch_plus']['p_MA'] *= 1.06
        self.assertFalse(compare_triplets(triplet(), rows)['passed'])

    def test_identical_triplets_pass_only_the_local_check(self):
        rows = triplet()
        self.assertTrue(compare_triplets(rows, copy.deepcopy(rows))['passed'])

    def test_invalid_physical_values_raise(self):
        for value in (0, -1, float('nan'), float('inf')):
            rows = triplet()
            rows['etch_plus']['T1_us'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare_triplets(triplet(), rows)


class SourceIdentityTests(unittest.TestCase):
    def fast_report(self):
        return dict(identity=dict(geometry_hash='expected', settings=dict(thickness_nm=200.0),
                                  frozen_config=frozen.frozen_hash()),
                    lj_H=14e-9, provenance=dict(frozen_config=frozen.frozen_hash()))

    def hfss_report(self):
        return dict(model_geometry=dict(geometry_hash='expected', thickness_nm=200.0,
                                        lj0_H=14e-9, etch_um=.25,
                                        provenance=dict(frozen_config=frozen.frozen_hash())),
                    provenance=dict(frozen_config=frozen.frozen_hash()),
                    diagnostic_completed=True, frequency_converged=True)

    def test_same_physical_problem_is_accepted(self):
        self.assertEqual(source_issues(self.fast_report(), 'expected', 'nominal', 'fast', 14e-9), [])
        self.assertEqual(source_issues(self.hfss_report(), 'expected', 'etch_plus', 'hfss', 14e-9), [])

    def test_wrong_geometry_and_inductance_are_rejected(self):
        self.assertTrue(source_issues(self.fast_report(), 'other', 'nominal', 'fast', 14e-9))
        self.assertTrue(source_issues(self.fast_report(), 'expected', 'nominal', 'fast', 15e-9))

    def test_wrong_thickness_and_configuration_are_rejected(self):
        for key, value in (('settings', dict(thickness_nm=400.0)), ('frozen_config', 'stale')):
            row = self.fast_report()
            row['identity'][key] = value
            self.assertTrue(source_issues(row, 'expected', 'nominal', 'fast', 14e-9))

    def test_wrong_etch_direction_or_failed_hfss_is_rejected(self):
        self.assertTrue(source_issues(self.hfss_report(), 'expected', 'etch_minus', 'hfss', 14e-9))
        for key in ('diagnostic_completed', 'frequency_converged'):
            row = self.hfss_report()
            row[key] = False
            self.assertTrue(source_issues(row, 'expected', 'etch_plus', 'hfss', 14e-9))
