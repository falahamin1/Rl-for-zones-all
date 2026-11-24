"""trmgrid: an NxN grid domain driven by a timed reward machine (TRM) --
visit sub-goals A, then B, then C, in order, subject to timing constraints
on the A->B and B->C transitions plus a global deadline.

Built FRESH for /tmp/TRMGRID.md: reuses only zoneviews/ and pyudbm. Does
not import, fork, or modify gridtimed, modesel, supplementary/frozen_lake,
supplementary/taxi, or Timed-RM -- those were read (not imported) purely
as a reference for proven DBM-operation patterns (the edge-clock uncertain-
elapse technique, the deterministic-concrete-DP-plus-symbolic-zone split,
the backward-induction-by-global-clock DP).
"""
