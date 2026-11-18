"""Geometry, enclosure, invertibility, and scope of paired rectangle meshes."""

import unittest
import numpy as np

from robust_design_lab.evaluator.mesh_transport import transport_nodes
from robust_design_lab.geometry import library
from robust_design_lab.perturbation.channels import apply_etch


class MeshTransportTests(unittest.TestCase):
    def test_perturbed_corners_match_geometry_and_walls_remain_walls(self):
        source = library.rect_record(200, 300, 0)
        for delta in (-.25, .25):
            target, error = apply_etch(source, delta)
            self.assertIsNone(error)
            for a, b in zip(source.polys, target.polys):
                points = np.array([[a.bounds[0]*1e-3+3, a.bounds[2]*1e-3+3, 0, 6],
                                   [a.bounds[1]*1e-3+3, a.bounds[3]*1e-3+3, 0, 6],
                                   [.5, .5002, 0, 3]])
                moved = transport_nodes(points, source, target)
                np.testing.assert_allclose(moved[:2, :2],
                    [[b.bounds[0]*1e-3+3, b.bounds[2]*1e-3+3],
                     [b.bounds[1]*1e-3+3, b.bounds[3]*1e-3+3]], atol=1e-13, rtol=0)
                np.testing.assert_array_equal(moved[2], points[2])
                np.testing.assert_array_equal(moved[:, 2:], points[:, 2:])

    def test_mapping_is_invertible_and_does_not_mutate_input(self):
        source = library.rect_record(200, 300, 0)
        target, _ = apply_etch(source, .25)
        points = np.random.default_rng(20261023).uniform(0, 1, (3, 256))*[[6], [6], [3]]
        original = points.copy()
        moved = transport_nodes(points, source, target)
        np.testing.assert_allclose(transport_nodes(moved, target, source), original,
                                   atol=1e-13, rtol=0)
        np.testing.assert_array_equal(points, original)

    def test_curved_shape_is_rejected(self):
        curved = library.rect_record(200, 300, 30)
        with self.assertRaises(ValueError):
            transport_nodes(np.zeros((3, 1)), curved, curved)
