"""Coverage test for run_one.TARGET_RUNS -- same contract as
envs/b3_extend_training.py's own test file would use (that file doesn't
exist in this repo, checked; TARGET_RUNS/resolve_spec here follow the
identical "flat, authoritative, index -> config list" pattern anyway).

Run with: python3 -m pytest test_run_one.py -q  (or plain unittest).
"""
from __future__ import annotations

import itertools
import unittest

from run_one import ENCODERS, GRIDTIMED_INSTANCES, MODESEL_INSTANCES, SEEDS, TARGET_RUNS, resolve_spec


class TestTargetRunsCoverage(unittest.TestCase):
    def test_length_is_48(self):
        self.assertEqual(len(TARGET_RUNS), 48)

    def test_indices_are_0_to_47_contiguous(self):
        indices = [r.index for r in TARGET_RUNS]
        self.assertEqual(indices, list(range(48)))

    def test_every_combination_covered_exactly_once(self):
        expected = set()
        for domain, instances in (("modesel", MODESEL_INSTANCES), ("gridtimed", GRIDTIMED_INSTANCES)):
            for instance, encoder, seed in itertools.product(instances, ENCODERS, SEEDS):
                expected.add((domain, instance, encoder, seed))
        self.assertEqual(len(expected), 48)

        actual = [(r.domain, r.instance, r.encoder, r.seed) for r in TARGET_RUNS]
        actual_set = set(actual)

        # No duplicates.
        self.assertEqual(len(actual), len(actual_set), "TARGET_RUNS contains a duplicate (domain, instance, encoder, seed)")
        # Full coverage.
        self.assertEqual(actual_set, expected)

    def test_run_id_unique_per_entry(self):
        run_ids = [r.run_id for r in TARGET_RUNS]
        self.assertEqual(len(run_ids), len(set(run_ids)))

    def test_resolve_spec_matches_list_lookup(self):
        for i in range(48):
            self.assertIs(resolve_spec(i), TARGET_RUNS[i])

    def test_resolve_spec_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            resolve_spec(-1)
        with self.assertRaises(ValueError):
            resolve_spec(48)

    def test_domain_instance_pairing_is_correct(self):
        for r in TARGET_RUNS:
            if r.domain == "modesel":
                self.assertIn(r.instance, MODESEL_INSTANCES)
            elif r.domain == "gridtimed":
                self.assertIn(r.instance, GRIDTIMED_INSTANCES)
            else:
                self.fail(f"unexpected domain {r.domain!r}")


if __name__ == "__main__":
    unittest.main()
