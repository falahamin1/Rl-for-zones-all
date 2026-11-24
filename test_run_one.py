"""Coverage test for run_one.TARGET_RUNS -- same contract as
envs/b3_extend_training.py's own test file would use (that file doesn't
exist in this repo, checked; TARGET_RUNS/resolve_spec here follow the
identical "flat, authoritative, index -> config list" pattern anyway).

Extended for the v_rep/GNN addition (/tmp/VREP_FEASIBILITY.md): indices
0-47 (the original 4-encoder sweep) must be BYTE-IDENTICAL to before that
extension, and indices 48-71 cover the 2 NEW_ENCODERS the same way.

Extended again for the gridtimed seed extension (/tmp/SWEEP_FINAL.md Task
C): indices 0-71 must be BYTE-IDENTICAL to before this extension, and
indices 72-155 cover EXTRA_SEEDS on gridtimed, all six encoders.

Extended again for the shaping experiment (/tmp/PART1_FINAL.md item 3):
indices 0-155 must be BYTE-IDENTICAL to before this extension, and indices
156-167 cover the 12 shaping-experiment configs.

Extended again for trmgrid (/tmp/TRMGRID_SWEEP.md): indices 0-167 must be
BYTE-IDENTICAL to before this extension -- checked explicitly against a
transcribed snapshot -- and indices 168-287 cover the 120 trmgrid configs
(2 TRMGRID_INSTANCES x 6 encoders x 10 TRMGRID_SEEDS). trmgrid is a brand
NEW domain string, so no run_id here can collide with anything in 0-167 by
construction (every prior run_id starts with "modesel__" or
"gridtimed__"); CHECKED EXPLICITLY anyway, not just reasoned about -- the
shaping experiment nearly resumed and mutated real completed Alpine data
because of exactly this kind of collision, caught only by an explicit
assertion, not by inspection.

Run with: python3 -m pytest test_run_one.py -q  (or plain unittest).
"""
from __future__ import annotations

import itertools
import unittest

from run_one import (
    ENCODERS, EXTRA_SEEDS, GRIDTIMED_INSTANCES, MODESEL_INSTANCES, NEW_ENCODERS, SEEDS, SHAPING_ENCODERS,
    SHAPING_INSTANCE, TARGET_RUNS, TRMGRID_INSTANCES, TRMGRID_SEEDS, resolve_spec,
)

#: The exact original 168-row table (indices 0-167), transcribed from
#: /tmp/PART1_FINAL.md's verified construction -- the ground truth for
#: "unchanged" going into the trmgrid extension, not re-derived from
#: current code. Built the same four-loop way _build_target_runs itself
#: does for 0-167.
_INSTANCES = [("modesel", i) for i in MODESEL_INSTANCES] + [("gridtimed", i) for i in GRIDTIMED_INSTANCES]
_GRIDTIMED_ONLY = [("gridtimed", i) for i in GRIDTIMED_INSTANCES]
_SHAPING_DOMAIN, _SHAPING_INSTANCE_NAME = SHAPING_INSTANCE
_ORIGINAL_168 = (
    [(d, i, e, s, None) for d, i in _INSTANCES for e in ENCODERS for s in SEEDS]
    + [(d, i, e, s, None) for d, i in _INSTANCES for e in NEW_ENCODERS for s in SEEDS]
    + [(d, i, e, s, None) for d, i in _GRIDTIMED_ONLY for e in ENCODERS + NEW_ENCODERS for s in EXTRA_SEEDS]
    + [
        (_SHAPING_DOMAIN, _SHAPING_INSTANCE_NAME, e, s, shaped)
        for e in SHAPING_ENCODERS for s in SEEDS for shaped in (False, True)
    ]
)


class TestTargetRunsCoverage(unittest.TestCase):
    def test_length_is_288(self):
        self.assertEqual(len(TARGET_RUNS), 288)

    def test_indices_are_0_to_287_contiguous(self):
        indices = [r.index for r in TARGET_RUNS]
        self.assertEqual(indices, list(range(288)))

    def test_first_168_unchanged(self):
        first_168 = [(r.domain, r.instance, r.encoder, r.seed, r.shaped) for r in TARGET_RUNS[:168]]
        self.assertEqual(first_168, _ORIGINAL_168, "indices 0-167 must be byte-identical to the pre-trmgrid-extension sweep")

    def test_every_combination_covered_exactly_once_through_167(self):
        expected = set()
        for domain, instances in (("modesel", MODESEL_INSTANCES), ("gridtimed", GRIDTIMED_INSTANCES)):
            for instance, encoder, seed in itertools.product(instances, ENCODERS + NEW_ENCODERS, SEEDS):
                expected.add((domain, instance, encoder, seed, None))
        for instance, encoder, seed in itertools.product(GRIDTIMED_INSTANCES, ENCODERS + NEW_ENCODERS, EXTRA_SEEDS):
            expected.add(("gridtimed", instance, encoder, seed, None))
        for encoder, seed, shaped in itertools.product(SHAPING_ENCODERS, SEEDS, (False, True)):
            expected.add((_SHAPING_DOMAIN, _SHAPING_INSTANCE_NAME, encoder, seed, shaped))
        self.assertEqual(len(expected), 168)

        actual = [(r.domain, r.instance, r.encoder, r.seed, r.shaped) for r in TARGET_RUNS[:168]]
        actual_set = set(actual)
        self.assertEqual(len(actual), len(actual_set), "TARGET_RUNS[:168] contains a duplicate")
        self.assertEqual(actual_set, expected)

    def test_48_to_71_cover_new_encoders_only(self):
        block = TARGET_RUNS[48:72]
        self.assertEqual(len(block), 24)
        for r in block:
            self.assertIn(r.encoder, NEW_ENCODERS)
            self.assertNotIn(r.encoder, ENCODERS)

    def test_72_to_155_are_gridtimed_extra_seeds_all_six_encoders(self):
        block = TARGET_RUNS[72:156]
        self.assertEqual(len(block), 84)
        for r in block:
            self.assertEqual(r.domain, "gridtimed")
            self.assertIn(r.instance, GRIDTIMED_INSTANCES)
            self.assertIn(r.encoder, ENCODERS + NEW_ENCODERS)
            self.assertIn(r.seed, EXTRA_SEEDS)
            self.assertNotIn(r.seed, SEEDS)

    def test_modesel_untouched_by_seed_extension(self):
        modesel_runs = [r for r in TARGET_RUNS if r.domain == "modesel"]
        self.assertEqual(len(modesel_runs), 2 * len(ENCODERS + NEW_ENCODERS) * len(SEEDS))
        for r in modesel_runs:
            self.assertIn(r.seed, SEEDS)
            self.assertNotIn(r.seed, EXTRA_SEEDS)
            self.assertIsNone(r.shaped)

    def test_156_to_167_are_the_shaping_experiment(self):
        block = TARGET_RUNS[156:168]
        self.assertEqual(len(block), 12)
        expected = set()
        for encoder, seed, shaped in itertools.product(SHAPING_ENCODERS, SEEDS, (False, True)):
            expected.add((_SHAPING_DOMAIN, _SHAPING_INSTANCE_NAME, encoder, seed, shaped))
        actual = {(r.domain, r.instance, r.encoder, r.seed, r.shaped) for r in block}
        self.assertEqual(actual, expected)

    def test_shaping_run_ids_distinct_from_original_sweep(self):
        original_ids = {
            r.run_id for r in TARGET_RUNS[:156]
            if r.domain == _SHAPING_DOMAIN and r.instance == _SHAPING_INSTANCE_NAME and r.encoder in SHAPING_ENCODERS
        }
        shaping_ids = {r.run_id for r in TARGET_RUNS[156:168]}
        self.assertEqual(original_ids & shaping_ids, set(), "shaping-experiment run_ids must not collide with the original sweep's")

    def test_168_to_287_are_the_trmgrid_extension(self):
        tail = TARGET_RUNS[168:]
        self.assertEqual(len(tail), 120)
        expected = set()
        for instance, encoder, seed in itertools.product(TRMGRID_INSTANCES, ENCODERS + NEW_ENCODERS, TRMGRID_SEEDS):
            expected.add(("trmgrid", instance, encoder, seed))
        self.assertEqual(len(expected), 120)
        actual = {(r.domain, r.instance, r.encoder, r.seed) for r in tail}
        self.assertEqual(actual, expected)
        for r in tail:
            self.assertIsNone(r.shaped)

    def test_trmgrid_run_ids_do_not_collide_with_anything(self):
        """CRITICAL (per the task's own explicit instruction, after the
        shaping experiment's near-miss): trmgrid's 120 run_ids must not
        collide with ANY of the other 168 entries' run_ids. Checked
        explicitly against the full set, not inferred from "trmgrid" being
        a new domain string."""
        trmgrid_ids = {r.run_id for r in TARGET_RUNS if r.domain == "trmgrid"}
        other_ids = {r.run_id for r in TARGET_RUNS if r.domain != "trmgrid"}
        self.assertEqual(len(trmgrid_ids), 120)
        self.assertEqual(trmgrid_ids & other_ids, set(), "trmgrid run_ids must not collide with any existing run_id")

    def test_run_id_unique_per_entry(self):
        run_ids = [r.run_id for r in TARGET_RUNS]
        self.assertEqual(len(run_ids), len(set(run_ids)))

    def test_resolve_spec_matches_list_lookup(self):
        for i in range(288):
            self.assertIs(resolve_spec(i), TARGET_RUNS[i])

    def test_resolve_spec_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            resolve_spec(-1)
        with self.assertRaises(ValueError):
            resolve_spec(288)

    def test_domain_instance_pairing_is_correct(self):
        for r in TARGET_RUNS:
            if r.domain == "modesel":
                self.assertIn(r.instance, MODESEL_INSTANCES)
            elif r.domain == "gridtimed":
                self.assertIn(r.instance, GRIDTIMED_INSTANCES)
            elif r.domain == "trmgrid":
                self.assertIn(r.instance, TRMGRID_INSTANCES)
            else:
                self.fail(f"unexpected domain {r.domain!r}")


if __name__ == "__main__":
    unittest.main()
