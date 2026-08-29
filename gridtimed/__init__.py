"""gridtimed: a fresh, minimal timed-automata grid-navigation environment.

Built from scratch for the geometry gate described in the task -- does NOT
fork or import from supplementary/frozen_lake, supplementary/taxi, or
Timed-RM. Reuses only zoneviews/ (Zone, raster, the affine-dim-first v_rep
fix) and pyudbm. Does not modify prob_jobshop_ext, deadline_track, or
gate_track.
"""
