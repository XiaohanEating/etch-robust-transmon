"""Independent polynomial fields and local credential validation."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from skfem import MeshTet, Basis, ElementTetP1, ElementTetP2
from robust_design_lab.evaluator.fields import ElectrostaticField
from robust_design_lab.vlm.aliyun import read_csv, CredentialError


class PolynomialFieldTests(unittest.TestCase):
    def evaluate(self, order, function, gradient):
        from qiskit_metal.analyses.fem.solver import _element_edges
        mesh = MeshTet()
        basis = Basis(mesh, ElementTetP2() if order == 2 else ElementTetP1())
        est = SimpleNamespace(_mesh=mesh, _p=mesh.p, _t=mesh.t,
                              _basis=basis, _order=order)
        field = ElectrostaticField.__new__(ElectrostaticField)
        field.est, field.phi = est, function(basis.doflocs)
        field.edge_map = _element_edges(mesh, mesh.t)
        elements = np.arange(mesh.t.shape[1])
        points = mesh.p[:, mesh.t].mean(axis=1)
        np.testing.assert_allclose(field.electric_on_elements(points, elements),
                                   -gradient(points), atol=2e-12, rtol=2e-12)

    def test_linear_potential_p1(self):
        self.evaluate(1, lambda x: 2*x[0]-3*x[1]+4*x[2]+7,
                      lambda x: np.tile([[2],[-3],[4]], (1,x.shape[1])))

    def test_quadratic_potential_p2(self):
        self.evaluate(2, lambda x: x[0]**2+2*x[1]**2-3*x[2]**2+x[0]*x[2],
                      lambda x: np.array([2*x[0]+x[2], 4*x[1], -6*x[2]+x[0]]))


class CredentialTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.csv'
            path.write_text(text, encoding='utf-8')
            return read_csv(path)

    def test_incomplete_key_is_rejected(self):
        with self.assertRaises(CredentialError):
            self.parse('apiKey,1\nopenAiCompatible,https://dashscope.aliyuncs.com/compatible-mode/v1')

    def test_untrusted_endpoint_is_rejected(self):
        with self.assertRaises(CredentialError):
            self.parse('apiKey,sk-local-test-placeholder\nopenAiCompatible,https://example.com/v1')

    def test_secret_not_in_repr(self):
        result=self.parse('apiKey,sk-local-test-placeholder\nopenAiCompatible,https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.assertNotIn('sk-local-test-placeholder', repr(result))
