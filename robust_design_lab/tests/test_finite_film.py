"""Physical partition and field checks for finite metal cavities."""

from types import SimpleNamespace
import unittest

import numpy as np

from robust_design_lab.evaluator import finite_film
from robust_design_lab.evaluator.fast import FieldSolution
from robust_design_lab.evaluator.fields import ElectrostaticField
from robust_design_lab.experiments.finite_alignment import mesh_checks
from robust_design_lab.geometry import library


class FiniteFilmSettingsTests(unittest.TestCase):
    def test_invalid_mesh_settings_are_rejected(self):
        for key in ('thickness_nm', 'h_edge_mm', 'h_max_mm', 'grow_mm'):
            for value in (0, -1, float('nan'), float('inf')):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    finite_film.MeshSettings(**{key: value})
        with self.assertRaises(ValueError):
            finite_film.MeshSettings(threads=0)


class FiniteFilmPhysicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = library.rect_record(200, 300, 0)
        cls.settings = finite_film.MeshSettings(h_edge_mm=0.05)
        cls.fs, cls.surfaces = finite_film.solve_fields(cls.rec, cls.settings)

    def test_metal_cavity_thickness_and_complete_interface_coverage(self):
        checks = mesh_checks(self.fs, self.surfaces, self.rec, self.settings)
        self.assertAlmostEqual(checks['measured_thickness_nm'], 200, places=6)
        self.assertAlmostEqual(checks['areas_mm2']['SM'], 0.12, places=10)
        self.assertAlmostEqual(checks['areas_mm2']['MA'], 0.1204, places=10)
        self.assertAlmostEqual(checks['areas_mm2']['SA'], 35.88, places=9)
        self.assertLess(max(checks['volume_relative_errors'].values()), 1e-9)

    def test_cavities_are_equipotential_and_not_grounded(self):
        est = self.fs.est
        from qiskit_metal.analyses.fem.solver import _facet_index
        for k, name in enumerate(est.names):
            facets = _facet_index(est._mesh, self.fs.package_mesh.metal_tris[name])
            dofs = est._basis.get_dofs(facets=facets).flatten()
            np.testing.assert_array_equal(est._owner[dofs], k)
        self.assertGreater(float(np.linalg.eigvalsh(est.C).min()), 0)
        self.assertGreater(self.fs.C_q_fF, 0)

    def test_field_gradient_and_energy_agree_with_native_fem(self):
        fs, est = self.fs, self.fs.est
        field = ElectrostaticField(fs)
        native = -est._basis.interpolate(field.phi).grad
        xyz = np.asarray(est._basis.global_coordinates())
        idx = np.arange(min(128, est._t.shape[1]))
        actual = field.electric_on_elements(xyz[:, idx].reshape(3, -1),
                                            np.repeat(idx, xyz.shape[2]))
        expected = native[:, idx].reshape(3, -1)
        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected), 1e-10)
        eps = np.where(fs.package_mesh.material == 1, fs.package_mesh.package.eps_r, 1)
        energy = 0.5*finite_film.EPS0*1e-3*np.sum(
            np.sum(native**2, axis=0)*est._basis.dx*eps[:, None])
        self.assertAlmostEqual(energy/field.energy, 1, places=10)

    def test_thickness_loss_scaling_and_equation_residual(self):
        a = finite_film.surface_participation(self.fs, self.surfaces, 14.5856e-9)
        b = finite_film.surface_participation(self.fs, self.surfaces, 14.5856e-9, 1.1)
        for name in ('p_SM', 'p_MA', 'p_SA'):
            self.assertGreater(a[name], 0)
            self.assertAlmostEqual(b[name]/a[name], 1.1, places=12)
        self.assertAlmostEqual(a['T1_us']/b['T1_us'], 1.1, places=12)
        self.assertLess(a['residuals']['internal_relative'], 1e-8)


if __name__ == '__main__':
    unittest.main()
