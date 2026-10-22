# Spec: tools/rangeprofile.py (the invariance profile in three activation strata)

Write one new file, `tools/rangeprofile.py`. Read only this spec and `tools/lookviews.py`, `tools/rangeview.py`,
`tools/battery.py`, `tools/bundlemax.py` and `bank/img_acts.py`. Do not open any other file, and do not run anything.
Import from those files wherever this spec says "as" one of them.

## Purpose
The invariance profile (how much of a channel's activation survives each image transform) is measured on each
channel's 20 strongest photos. Measure it also on photos further down each channel's range, and report, for every
channel of each layer, how its retention and its change under each transform compare with the other channels of its
layer, in each stratum.

## Inputs
- `--layers` (default mixed3a,mixed3b,mixed4b,mixed4c,mixed4d,mixed4e,mixed5a) and `--units-file` (default
  `authors/round26_units.json`: a JSON list of unit strings 'mixedNx:c', for the report).
- P0 and the profile values (for each layer, the spatial max of every channel on the P0 photos, originals and under
  each transform of battery.TRANSFORMS), as lookviews.profiles makes them: use the first existing
  `profile_<layer>.npz` among `runs/rangeview/`, `runs/lookviews/`, `runs/sparedview/`, checked as lookviews.profiles
  checks a cache (layer, P0, transform list); otherwise compute it with rangeview.profiles(layers, P0, get_model,
  out=Path('runs/rangeprofile')), on CUDA when available if that function allows it, else as it does.

## Strata (per channel)
Its P0 photos with original value above 0, sorted by original value, highest first (stable on ties). Stratum 1:
ranks 1-20; stratum 2: ranks 21-100; stratum 3: ranks 101-300 (rank 1 the highest). A stratum is missing (NaN
statistics) when the channel has fewer positive photos than the stratum's last rank.

## Statistics (per layer, channel, transform, stratum)
- retention = median over the stratum's photos of t / o (o original, t transformed).
- change = median over the stratum's photos of |t - o| / M, M = the channel's max of o over P0.
- For each statistic, the channel's percentile = 100 x the share of the layer's channels with a finite value strictly
  below its own (NaN when its own is NaN).
- Neutral label on a change percentile: >= 95 CHANGES_MORE, <= 5 CHANGES_LESS, else TYPICAL; NaN MISSING.

## Outputs
- `runs/rangeprofile/<layer>.csv`: one row per channel, columns `unit` ('mixedNx:c'), then for each transform in
  battery.TRANSFORMS order and each stratum s = 1, 2, 3: `<transform>_ret_s<s>`, `<transform>_ret_pct_s<s>`,
  `<transform>_chg_s<s>`, `<transform>_chg_pct_s<s>` (6 decimals; empty for NaN).
- `runs/rangeprofile/report.txt`, printed: per unit of the units file, one line per transform: name, then per stratum
  retention, its percentile, change, its percentile and the label.

## Interface (a separate tester writes tests against these; keep the names and behaviour)
- `strata(o_col, bounds=((1, 20), (21, 100), (101, 300)))`: o_col a 1-d array of one channel's original values;
  returns a list, one per stratum, of index arrays into o_col by the rule above, or None for a missing stratum.
- `stratum_stats(o_col, t_col, idx)`: (retention, change) for one channel and one stratum, M taken from o_col.
- `layer_stats(orig, trans, bounds=...)`: orig and trans (photos x channels); returns (ret, chg), each an array of
  channels x strata.
- `pct_below(x)`: x a 1-d array over channels; returns an array of their percentiles by the rule above.
- `label(p)`: the neutral label.

## Selftest
`--selftest`, no model: strata on a hand case (zeros skipped, ties stable, missing strata), stratum_stats against
hand values, layer_stats shape, pct_below with NaN, label at the boundaries. Print `selftest ok`.

## Usage line
    python -I tools/rangeprofile.py [--layers ...] [--units-file authors/round26_units.json] [--selftest]
