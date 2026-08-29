# PRESUBMIT: Two Diagnostics Before Array Submission

Diagnosis only — no files modified (`run_one.py`, `submit.sbatch`,
`TARGET_RUNS` untouched). Nothing submitted to Alpine. No direction
recommended.

---

## Task A — is the difference-feature probe measuring what we think?

**Verdict: A2 (probe capacity) explains cnn_dbm's original low score. A1
(target mismatch) does not. The original finding does NOT stand as
reported for cnn_dbm — it was an artifact of an under-powered linear
probe, not a property of the representation.**

| Encoder | Original (ridge, concrete diff) | A2: MLP probe, 256h, concrete diff | A3: untrained, ridge, concrete diff |
|---|---:|---:|---:|
| `mlp_point` | 0.933 | 0.973 | **0.911** |
| `deepset_h` | 0.927 | 0.991 | **0.913** |
| `cnn_dbm` | 0.411 | **0.996** | 0.196 |
| `cnn_proj` | 0.167 | 0.376 | 0.000 |

**A1 (target mismatch, cnn_dbm only)**: re-ran the ridge probe on
`cnn_dbm` predicting the normalized DBM bound itself (same values/same
cap-normalization `dbm_tensor` encodes) instead of the concrete
difference. R² = **0.365** — still low, not "near 1.0". Target mismatch is
ruled out as the explanation: even predicting the exact quantity the
tensor literally stores, the small linear probe can't recover it well.
Of the 6 pairs × 1000 states sampled, 100% were bounded in both directions
(no unbounded pairs); **46.6% were tight** (`bound(i,j)+bound(j,i)==0`,
difference pinned to a single value) and 53.4% loose. So roughly half the
concrete differences genuinely aren't pinned by the zone — a real ceiling
on what ANY encoder could recover for those pairs, independent of the
probe-capacity issue below.

**A2 (probe capacity, all four encoders)**: a 2-layer/256-hidden MLP
probe, trained to convergence with early stopping on a held-out split,
predicting concrete differences (same target as the original report).
`cnn_dbm` jumps from 0.411 → **0.996** — the largest capacity-sensitivity
of any encoder, and once given an adequate probe, `cnn_dbm` recovers
concrete pairwise differences essentially perfectly, *better* than
`mlp_point` or `deepset_h`. `cnn_proj` also improves (0.167 → 0.376) but
stays clearly behind the other three even at full probe capacity.
`mlp_point`/`deepset_h` improve only modestly (0.933→0.973, 0.927→0.991) —
they were already near their ceiling with the small linear probe.

**A3 (untrained control, all four encoders)**: random-init weights, no
training at all, original small ridge probe. `mlp_point` (0.911) and
`deepset_h` (0.913) are barely below their FULLY TRAINED small-probe
numbers (0.933, 0.927) — **most of their apparent R² is present before
training touches the weights at all**, meaning the original probe was, for
these two encoders, measuring architecture pass-through more than
anything training organized. `cnn_dbm` (0.196 untrained) and `cnn_proj`
(0.000 untrained) show training clearly does add real information (0.196→
0.411 with the small probe, more with the big one) — the small-probe gap
between untrained and trained is real for the CNNs, just not the dominant
effect for the original result's headline gap.

**Caveat carried from the original report and unchanged here**: all
numbers above use the same 6,000-env_step QUICK in-script training, not a
real 300k-step checkpoint (none exist — nothing has been submitted to
Alpine). The *ranking* could still shift with full training; what's
resolved here is specifically why `cnn_dbm` scored suspiciously low, not
whether these exact numbers will hold at convergence.

---

## Task B — cnn_proj rendering scheme: occupancy data, no direction chosen

All numbers below: `res=64`, 200 sampled zones per instance (`use_cache=False`,
forcing real rasterization), "lit" = pixel value > 0 (any coverage, matching
the earlier report's convention — `projection_stack` is continuous-coverage,
not hard-binary).

### res=64, default (global `[0,k]×[0,k]` box) vs per-zone normalization

| Instance | scheme | mean lit-frac | median lit-frac | frac. channels entirely empty | mean bbox extent (of *lit* channels) |
|---|---|---:|---:|---:|---:|
| modesel/M3 | default | 0.115% | 0.000% | 90.0% | 12.8% |
| modesel/M3 | per-zone | 4.64% | 0.000% | 90.0% | **100%** |
| modesel/M4 | default | 0.087% | 0.000% | 91.6% | 12.4% |
| modesel/M4 | per-zone | 4.60% | 0.000% | 91.6% | **100%** |
| gridtimed/T2 | default | 0.056% | 0.024% | 49.8% | 5.2% |
| gridtimed/T2 | per-zone | 11.08% | 1.56% | 49.8% | **100%** |
| gridtimed/T3 | default | 0.058% | 0.049% | 40.4% | 5.5% |
| gridtimed/T3 | per-zone | 5.54% | 1.56% | 40.4% | **100%** |

**A finding not in the original report: per-zone normalization does NOT
reduce the empty-channel fraction — at all, on any instance** (90.0%→90.0%,
91.6%→91.6%, 49.8%→49.8%, 40.4%→40.4%, exact ties). It only inflates the
occupancy of channels that already had *some* content, filling their frame
completely (bbox extent exactly 100% wherever lit, on every instance).
This is expected once stated precisely: per-zone normalization rescales
the rendering *box*, it can't manufacture area in a projection whose true
geometric extent is zero — an empty channel stays empty under any box
choice.

**Domain asymmetry, confirmed by the data, not assumed**: gridtimed has
roughly half as many fully-empty channels as modesel (40–50% vs 90–92%),
but the channels that ARE lit are much thinner slivers (mean bbox extent
5–5.5% vs modesel's 12–13%) — gridtimed's default-rendering problem is
more "many thin slivers" than modesel's "almost everything is exactly
zero." This is consistent with (but does not, on its own, fully explain)
the earlier report's returns gap being much larger on gridtimed/T2 (7.06
→ 18.86) than modesel/M3 (−69.97 → −70.04, no real difference).

### Empty-channel classification (default rendering, res=64)

For every empty channel, re-rendered the SAME zone at `res=1024` (16×) and
checked whether it lights up there:

| Instance | checked | zero measure / outside box (case a) | too small for res=64 (case b) |
|---|---:|---:|---:|
| modesel/M3 | 60 of 1080 empty | **60** | 0 |
| modesel/M4 | 60 of 1832 empty | **60** | 0 |
| gridtimed/T2 | 60 of 995 empty | **60** | 0 |
| gridtimed/T3 | 60 of 1212 empty | **60** | 0 |

**240/240 checked empty channels, across all four instances, are case
(a): the zone's true projected region has zero or near-zero geometric
overlap with the rendering box — not a resolution artifact.** None
resolved into a visible region even at 16× the resolution. (Method: same
zone, same channel, `_projection_stack_uncached` at `res=1024` instead of
64, default box; "lights up" = any pixel > 0. Checked a capped sample of
60 per instance for tractability, not the full empty set — the sample is
large enough that the 60/60 result is not close to ambiguous.)

### res=128, 256 (default only)

| Instance | res=64 | res=128 | res=256 |
|---|---:|---:|---:|
| modesel/M3 | 0.115% | 0.106% | 0.101% |
| modesel/M4 | 0.087% | 0.081% | 0.078% |
| gridtimed/T2 | 0.056% | 0.035% | 0.026% |
| gridtimed/T3 | 0.058% | 0.031% | 0.019% |

Occupancy is flat-to-mildly-declining with resolution, never rising — the
opposite of what "resolution-limited" would predict. modesel is nearly
flat; gridtimed declines somewhat faster (consistent with the "thin
slivers" finding above — thin, elongated regions lose more of their
apparent area to anti-aliasing bleed at coarse resolutions, so the true
occupancy is if anything slightly *below* even these already-tiny numbers).

### Resolution needed for >10% occupancy

All four instances are under 5% at res=64 (table above), so per
instructions: swept resolution further (separate, smaller 25-zone sample,
different seed, for tractability — noted so its res=256 value isn't
mistaken for the same-sample res=256 in the table above, which it isn't;
treat this sweep as an independent check of the *trend*, not a
continuation of the same measurement) up to res=1024 (capped after the
trend was unambiguous and a 4× jump to res=2048 stopped being tractable —
one modesel run alone exceeded 10 minutes wall-clock and was killed rather
than waited out further):

| Instance | res=256 | res=512 | res=1024 | crosses 10%? |
|---|---:|---:|---:|---|
| modesel/M3 | 0.102% | 0.100% | 0.098% | **No** |
| modesel/M4 | 0.040% | 0.039% | 0.038% | **No** |
| gridtimed/T2 | 0.010% | 0.005% | 0.0025% | **No** |
| gridtimed/T3 | 0.028% | 0.022% | 0.020% | **No** |

**None of the four instances reach 10% occupancy anywhere in the tested
range, and the trend is flat-to-declining, not rising** — so "what
resolution is needed for >10%" doesn't have an answer in any practically
reachable range; this is not a resolution problem, consistent with the
100% case-(a) classification above. Per-rasterization cost at the highest
resolution actually tested (`res=1024`, per zone, single-threaded, this
machine — the closest available data point, since no resolution achieves
the target): modesel/M3 0.33s, modesel/M4 0.57s, gridtimed/T2 0.57s,
gridtimed/T3 0.88s — roughly 300–800× the `res=64` cost the current
48-run sweep budgets for, which would blow through the wall-clock estimate
in `/tmp/ENCODERS_SETUP.md` §8 by a similar factor if used naively for all
rollout steps.

### The tradeoff (stated, not decided)

Per-zone normalization discards absolute clock magnitude — it renders
every zone's own finite extent as filling the frame, so a zone near the
deadline and a zone far from it, if similarly *shaped*, look similar. In a
domain with deadlines, how close `g` is to `D_j` is exactly the kind of
information the earlier probe work (`/tmp/ENCODERS_SETUP.md` §6) suggests
these zone-based encoders already struggle to preserve precisely — a
representation that also drops absolute position removes a second channel
through which that information could arrive. The measured returns
asymmetry (gridtimed's larger gain from per-zone normalization,
modesel's near-zero gain) doesn't resolve this tradeoff either way: it's a
single-seed, 10k-step, short-horizon result, not a converged one, and
higher return early in a short run doesn't by itself confirm the discarded
information wasn't needed — it may just mean the discarded information
wasn't yet being used at 10k steps regardless of encoding. The occupancy
numbers above are real and stable across seeds sampled; the
returns-tradeoff question is not settled by them.

No new training runs were done for this report beyond what
`/tmp/ENCODERS_SETUP.md` already ran (modesel/M3, gridtimed/T2, both at
10k steps, one seed each).
