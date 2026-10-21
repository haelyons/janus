# Spec: tools/sparedview.py (affected-vs-spared view for a unit's most distinctive transform)

Write one new file, `tools/sparedview.py`. Read only this spec and `tools/claude_call.py`, `tools/run_subjects.py`,
`tools/bundlemax.py`, `tools/battery.py`, `tools/openmethod.py`, `tools/lookviews.py` and `bank/img_acts.py`. Do not
open any other file except `authors/battery.reply.json`, and do not run anything.

## Purpose
Collect explanations of units shown (a) the default card, crops and invariance profile (lookviews condition CP) and
(b) the same plus a view of the photos most and least affected by the transform the unit is most unusually sensitive
to. Nothing is scored here.

## Inputs
`--units-file` (default `authors/round23_units.json`): a JSON list of single-channel unit strings. Values, cards,
card text, receptive-field size S and crops, profiles and profile text: exactly as tools/lookviews.py (import it),
with caches under `runs/sparedview/<unit>/`; reuse `runs/lookviews/<unit>/` caches when present (copy, then check as
lookviews does). Compute new values and gradients on CUDA when `torch.cuda.is_available()` (outputs moved to CPU),
else CPU; for any unit whose values you compute, first check the first 64 card-pool photos against openmethod's CPU
computation (np.allclose, rtol 1e-3) and stop if they differ.

## T* and the view
From the unit's profile rows (lookviews.unit_profile): T* = the transform with the lowest percentile, ties broken by
lower retention, then battery.TRANSFORMS order. If its percentile is above 20, the unit gets no view (condition CP+A
is then identical to CP and is not run; record "no T*"). Otherwise: X = the unit's 200 card-pool photos with the
highest value. For each, a = its value, b = the unit's spatial max on battery.TRANSFORMS[T*](photo), drop = 1 - b / a.
Affected = the 10 with the largest drop (ties: higher a). Spared candidates = the third of X with the smallest drop,
excluding affected photos; for each affected photo in order, take the unused candidate with the smallest
|log(a) - log(a_affected)|. Grids `affected.jpg` and `spared.jpg`: the photos' receptive-field crops as lookviews
(side S, 160 px, 5 columns), a white strip under each with 'activation k, loses q%' (k on the card's 0-10 scale of M,
q the drop as a whole percentage). Save `spared.json` (T*, its percentile, the 20 photos, a, b, drop).

## Conditions and readings
- CP: exactly lookviews.condition_contents for CP.
- CP+A: CP's blocks, with these inserted just before run_subjects.TASK: a text block, affected.jpg, spared.jpg. Text,
  verbatim with the fields filled ({T} the transform name, {desc} its description as lookviews.describe, {p} its
  percentile): "The next two grids are about the transformation this unit is most unusually sensitive to: {T}
  (removes {desc}; percentile {p} in the profile above). The first shows the 10 of its 200 strongest photos that lose
  the most activation under it; the second shows 10 photos of similar activation that lose the least. Each is cropped
  as in the third grid; under each are its activation and the share it loses."
3 samples each, bare system prompt, parsed and cached as lookviews (`readings.json`, keys "CP_s1" ... "CPA_s3").
Up to WORKERS = 6 calls in parallel.

## Report
`runs/sparedview/report.txt`, printed: per unit, S, T* and its percentile (or "no T*"), mean drop of affected and
spared photos; one line per reading: id, class, first 160 characters. `runs/sparedview/readings_all.json`: {unit:
{reading id: explanation}}.

## Interface (a separate tester writes tests against these; keep the names and behaviour)
- `pick_tstar(rows)`: rows = list of {"name": str, "u": float, "p": float} in battery order; returns (name, p) or None
  when the lowest percentile is above 20, by the rule above.
- `select_pairs(a, drop, n=10)`: numpy arrays over X; returns (affected_idx, spared_idx), lists of indices into X, by
  the rule above (spared drawn from the third with the smallest drop, matched on |log a| difference, no reuse).
- `drop_of(a, b)`: 1 - b / a, elementwise.
- `strip_text(k, q)`: the caption string, e.g. strip_text(7, 0.634) == 'activation 7, loses 63%'.

## Selftest
`--selftest`, no model or Claude: pick_tstar (threshold, ties), select_pairs on hand arrays (exclusion, matching, no
reuse), drop_of, strip_text rounding, and the CP+A content order. Print `selftest ok`.

## Usage line
    python -I tools/sparedview.py [--units-file authors/round23_units.json] [--selftest]
