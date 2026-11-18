"""Independent geometry checks for the finite-film interface partition."""

import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

from robust_design_lab.experiments.hfss_interfaces import partition_faces, save_diagnostic_report


class FiniteFilmFaceTests(unittest.TestCase):
    def test_bottom_is_separate_from_top_and_vertical_sides(self):
        faces = {
            1: [[0, 0, 0.5], [1, 0, 0.5], [0, 1, 0.5]],
            2: [[0, 0, 0.5002], [1, 0, 0.5002], [0, 1, 0.5002]],
            3: [[0, 0, 0.5], [1, 0, 0.5], [1, 0, 0.5002], [0, 0, 0.5002]],
        }
        self.assertEqual(partition_faces(faces, 0.5, 200), ([1], [2, 3]))

    def test_zero_thickness_sheet_cannot_be_reported_as_separate_interfaces(self):
        with self.assertRaises(ValueError):
            partition_faces({1: [[0, 0, 0.5], [1, 0, 0.5], [0, 1, 0.5]]}, 0.5, 200)

    def test_millimetre_nanometre_confusion_is_rejected(self):
        with self.assertRaises(ValueError):
            partition_faces({1: [[0, 0, 0.5], [1, 0, 0.5]],
                             2: [[0, 0, 0.7], [1, 0, 0.7]]}, 0.5, 200)


class DiagnosticPersistenceTests(unittest.TestCase):
    def test_message_failure_preserves_completed_physical_report(self):
        desktop = Mock()
        desktop.get_messages.side_effect = RuntimeError('COM message read failed')
        project = SimpleNamespace(_project=Mock())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'report.json'
            save_diagnostic_report(path, dict(diagnostic_completed=True, fields=dict(p_SM=1e-4)),
                                   desktop, project)
            row = json.loads(path.read_text(encoding='utf-8'))
        self.assertTrue(row['diagnostic_completed'])
        self.assertEqual(row['fields']['p_SM'], 1e-4)
        self.assertIn('COM message read failed', row['message_collection_error'])

    def test_message_failure_does_not_replace_original_physical_error(self):
        desktop = Mock()
        desktop.get_messages.side_effect = RuntimeError('message failure')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'report.json'
            save_diagnostic_report(path, dict(diagnostic_completed=False, failed='original solve failure'),
                                   desktop, SimpleNamespace(_project=Mock()))
            row = json.loads(path.read_text(encoding='utf-8'))
        self.assertFalse(row['diagnostic_completed'])
        self.assertEqual(row['failed'], 'original solve failure')


if __name__ == '__main__':
    unittest.main()
