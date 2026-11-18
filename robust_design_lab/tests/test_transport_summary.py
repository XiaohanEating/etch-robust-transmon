"""Checks for the paired-mesh transport summary and its explicit boundaries."""

import unittest

from robust_design_lab.experiments.transport_summary import _level_comparison


def _row(c, sm, ma, sa, t1):
    return {"C_q_fF": c, "p_SM": sm, "p_MA": ma, "p_SA": sa, "T1_us": t1}


class TransportSummaryTests(unittest.TestCase):
    def test_drift_can_be_stable_while_absolute_participation_fails(self):
        coarse = {
            "nominal": _row(57.4, 7.0e-5, 5.8e-7, 9.9e-5, 92.0),
            "etch_plus": _row(57.7, 7.1e-5, 5.9e-7, 1.0e-4, 91.6),
            "etch_minus": _row(57.1, 6.9e-5, 5.7e-7, 9.8e-5, 92.4),
        }
        fine = {
            "nominal": _row(58.2, 6.0e-5, 4.5e-7, 9.0e-5, 104.8),
            "etch_plus": _row(58.5, 6.1e-5, 4.6e-7, 9.1e-5, 104.4),
            "etch_minus": _row(57.9, 5.9e-5, 4.4e-7, 8.9e-5, 105.2),
        }
        result = _level_comparison(coarse, fine)
        self.assertTrue(result["checks"]["drift_difference_passed"])
        self.assertFalse(result["checks"]["participation_level_alignment_passed"])
        self.assertFalse(result["diagnostic_passed"])

    def test_direction_is_reported_per_etch_sign(self):
        coarse = {
            "nominal": _row(1, 1, 1, 1, 10),
            "etch_plus": _row(1, 1, 1, 1, 9),
            "etch_minus": _row(1, 1, 1, 1, 11),
        }
        fine = {
            "nominal": _row(1, 1, 1, 1, 20),
            "etch_plus": _row(1, 1, 1, 1, 18),
            "etch_minus": _row(1, 1, 1, 1, 22),
        }
        result = _level_comparison(coarse, fine)
        self.assertTrue(result["etch_log_drift_difference"]["etch_plus"]["direction_same"])
        self.assertTrue(result["etch_log_drift_difference"]["etch_minus"]["direction_same"])


if __name__ == "__main__":
    unittest.main()
