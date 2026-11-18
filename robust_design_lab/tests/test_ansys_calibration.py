"""Unit checks for Ansys/pyEPR calibration value conventions."""
import unittest

from robust_design_lab.evaluator.ansys_calibration import (
    eigenmode_frequency_hz,
    plasma_frequency_hz,
)


class AnsysCalibrationTests(unittest.TestCase):
    def test_pyepr_ghz_frequency_is_converted_to_hz(self):
        self.assertAlmostEqual(eigenmode_frequency_hz([5.40297802192]), 5.40297802192e9)
        self.assertAlmostEqual(eigenmode_frequency_hz(5.1), 5.1e9)

    def test_invalid_frequency_is_rejected(self):
        for value in ([], [0.0], [-5.1], [float('nan')]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                eigenmode_frequency_hz(value)

    def test_hfss_eigenmode_compares_to_linear_plasma_frequency(self):
        lom = dict(EJ_GHz=11.20704616105238, EC_GHz=0.3320667077962166,
                   f01_GHz=5.100005602425409)
        f_plasma = plasma_frequency_hz(lom)
        self.assertAlmostEqual(f_plasma, 5.456362834578281e9, places=2)
        self.assertLess(abs(5.40297802192e9-f_plasma)/f_plasma, 0.02)
        # The anharmonic f01 is a different observable from the linear mode.
        self.assertGreater(abs(5.40297802192e9-lom['f01_GHz']*1e9)/
                           (lom['f01_GHz']*1e9), 0.02)


if __name__ == '__main__':
    unittest.main()
