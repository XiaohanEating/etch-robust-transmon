"""Fixed operation and paired-arm protocol checks."""
import unittest
from robust_design_lab.geometry import library,operations,checks
from robust_design_lab.vlm.experiment import parse_proposals,OPERATIONS


class SearchContractTests(unittest.TestCase):
    def test_generator_shapes_are_valid(self):
        for category in library.CATEGORY_NAMES:
            record=library.make_record(category,0)
            valid,reasons=checks.check_record(record)
            self.assertTrue(valid,(category,reasons))
            self.assertEqual(library.GENERATOR_VERSION,'library-v4-bounded-semiaxes-junction-neck')

    def test_all_shared_operations_are_recorded_and_checked(self):
        for category in library.CATEGORY_NAMES:
            record=library.make_record(category,0)
            for operation in OPERATIONS:
                try:candidate=operations.apply(record,operation)
                except ValueError:continue
                valid,reasons=checks.check_record(candidate)
                self.assertTrue(valid,(category,operation,reasons))
                self.assertTrue(candidate.operation_history)

    def test_response_contract_snaps_to_preregistered_set(self):
        # v2-snap 合约（docs/decisions.md 2026-10-24）：集合外数值吸附到最近合法点
        self.assertEqual(len(parse_proposals({'proposals':[dict(OPERATIONS[0])]})),1)
        with self.assertRaises(ValueError):
            parse_proposals({'proposals':[{'kind':'polygon','points':[[0,0],[1,1]]}]})
        with self.assertRaises(ValueError):
            parse_proposals({'proposals':[{'kind':'resize','sx':1.0,'sy':1.0,'z':2}]})
        snapped = parse_proposals({'proposals':[{'kind':'resize','sx':1.01,'sy':1.0},
                                                {'kind':'round','radius_um':3.0}]})
        self.assertIn({'kind':'resize','sx':1.1,'sy':1.0}, snapped)
        self.assertIn({'kind':'round','radius_um':2.0}, snapped)


if __name__=='__main__':unittest.main()
