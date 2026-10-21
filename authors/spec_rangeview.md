# Spec: tools/rangeview.py (stratified exemplars and same-label near misses, in a 2x2 design)

Write one new file, `tools/rangeview.py`. Read only this spec and `tools/claude_call.py`, `tools/run_subjects.py`,
`tools/bundlemax.py`, `tools/battery.py`, `tools/openmethod.py`, `tools/lookviews.py` and `bank/img_acts.py`. Do not
open any other file except `authors/battery.reply.json`, and do not run anything.

## Purpose
Collect explanations of units shown the default card (lookviews condition CP) plus none, one or both of two views:
S, photos sampled across the unit's activation range; N, the least-active photos of the labels most common among its
strongest photos. Nothing is scored here.

## Inputs
`--units-file` (default `authors/round24_units.json`): a JSON list of single-channel units. Values, M, cards, card text,
receptive-field size and crops, profiles and profile text: exactly as tools/lookviews.py (import it), caches under
`runs/rangeview/<unit>/`, copying `runs/lookviews/<unit>/` or `runs/sparedview/<unit>/` caches when present (then
checked as lookviews does). New values and gradients on CUDA when available (outputs moved to CPU); for any unit whose
values you compute, first check the first 64 card-pool photos against openmethod's CPU computation (np.allclose, rtol
1e-3) and stop if they differ. Labels and class names as tools/nearmiss2.py.

## View S
r = value / M over the card pool C. Excluding the 20 card photos, bins (lower inclusive, upper exclusive, the last
bin's upper bound inclusive of 1.0): [0.6, 1.0], [0.4, 0.6), [0.2, 0.4), [0.05, 0.2). With
`rng = np.random.default_rng(zlib.crc32(unit.encode()))`, from each bin in that order draw min(5, size) photos without
replacement (rng.choice on the bin's photos in ascending id order). Grid `range.jpg`: whole photos with the peak box,
5 columns, one row per bin, as lookviews' boxed card (caption '#n   activation k' on the card's scale, numbered from 1
across the grid). Text block, verbatim: "The next grid shows photos drawn at random from four activation ranges below
the strongest photos, one row each: 60-100%, 40-60%, 20-40% and 5-20% of the strongest activation (fewer when a range
has fewer photos). Same box and scale as the first grid."

## View N
Labels of the 20 card photos; the 5 most common (ties: higher sum of r over those card photos, then lower label
index); fewer if there are fewer distinct labels. For each, in that order: its 2 card-pool photos (C, excluding the 20
card photos) with the lowest value (ties: lower id). Grid `nearmiss.jpg`: whole photos with the peak box, 2 per row
... laid out as 5 columns with each label's two photos adjacent (label 1: #1 #2, label 2: #3 #4, ...), caption
'#n   activation k'. No label names appear anywhere. Text block, verbatim: "The next grid shows near misses: for each of
the 5 kinds of photo most common among the strongest photos, the 2 photos of that kind on which the unit is least
active (#1-#2 are one kind, #3-#4 the next, and so on). Same box and scale as the first grid."

## Conditions and readings
CP = lookviews.condition_contents for CP. CP+S, CP+N and CP+S+N insert, just before run_subjects.TASK, the S blocks
(text, range.jpg), the N blocks (text, nearmiss.jpg), or S then N. Keys "CP_s1".."CP_s3", "CPS_s*", "CPN_s*",
"CPSN_s*"; 3 samples each, bare system prompt, parsed and cached as lookviews (`readings.json`); up to WORKERS = 6
calls in parallel.

## Report
`runs/rangeview/report.txt`, printed: per unit, bin sizes, the N labels' names (report only) with each photo's value
ratio; one line per reading: id, class, first 160 characters. `runs/rangeview/readings_all.json`: {unit: {id: text}}.

## Interface (a separate tester writes tests against these; keep the names and behaviour)
- `bin_of(r)`: index 0-3 of the bin for a ratio r by the rule above, or None below 0.05.
- `stratified_pick(ids, r, exclude, rng, per_bin=5)`: ids and r numpy arrays over C; exclude a set of ids; returns
  a list of 4 lists of ids (one per bin, in bin order) by the rule above.
- `top_labels(card_labels, card_r, k=5)`: lists of the 20 card photos' labels and ratios; returns up to k labels by
  the rule above.
- `least_active(ids, values, labels, label, exclude, n=2)`: the n ids of that label with the lowest value, ties by
  lower id, excluding `exclude`.

## Selftest
`--selftest`, no model or Claude: bin_of at every boundary (0.05, 0.2, 0.4, 0.6, 1.0, just below each), stratified_pick
(determinism, exclusion, short bins), top_labels ties, least_active ties and exclusion, and the content order of each
condition. Print `selftest ok`.

## Usage line
    python -I tools/rangeview.py [--units-file authors/round24_units.json] [--selftest]
