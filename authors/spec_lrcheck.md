# Spec: tools/lrcheck.py (left- vs right-facing within kind, for every channel of a layer)

Write one new file, `tools/lrcheck.py`. Read only this spec, `tools/bundlemax.py` and `bank/img_acts.py`. Do not open
any other file except the data file `runs/views/labels.json`, and do not run anything.

## Purpose
For every channel of the given layers, measure how differently it responds to left-facing and right-facing subjects
within one kind of photo, using labels made by a model that never saw any activation.

## Inputs
- `runs/views/labels.json`: {"photos": [...], "labels": [{"photo": photo index, "kind": "D" or "V", "facing": ...}]}.
  Left = facing in {"left", "toward_left", "away_left"}; right = {"right", "toward_right", "away_right"}; others unused.
- Photos as bundlemax.photo; network input as bundlemax.net_input; model img_acts.load_model; layers `--layers`
  (default mixed4b,mixed4c,mixed4d,mixed4e).

## Values
For each labelled photo and its left-right mirror (PIL.ImageOps.mirror): the spatial max of every channel of each
layer, one forward pass per batch hooking all layers and stopping after the deepest (raise img_acts.Stop), on CUDA when
available (outputs moved to CPU). Cache `runs/lrcheck/values.npz`.

## Statistic
Per layer, channel and kind (D, V), on original photos: v = value / (max of the channel over all labelled photos of
both kinds, originals); d = (mean v over left - mean v over right) / (sd of v over that kind's left and right photos
together, ddof 1); d = 0 when the sd is 0. d_mirror: the same on the mirrored photos, with each photo keeping its
original label. |d| percentile = 100 x share of the layer's channels with |d| below this channel's.

## Report
`runs/lrcheck/report.txt`, printed: counts of left and right photos per kind; per layer the median and 90th and 95th
percentiles of |d| per kind; then for the units in `--named` (default mixed4b:443,431,429,384,409,418,237,373,
mixed4c:254,407,447,mixed4d:409,mixed4e:55), one line: unit, d_D, |d_D| percentile, d_mirror_D, d_V, |d_V| percentile,
d_mirror_V. `runs/lrcheck/table.csv`: every channel, the same columns.

## Interface (a separate tester writes tests against these; keep the names and behaviour)
- `side_of(facing)`: 'left', 'right' or None.
- `d_stat(v, sides)`: v array, sides list of 'left'/'right'/None; returns d by the rule above (None entries ignored).
- `pct_abs(d_all, d)`: 100 x share of entries of d_all with |entry| < |d|.

## Selftest
`--selftest`, no model: side_of on every label value, d_stat on hand cases (zero sd, ignored entries, sign), pct_abs.
Print `selftest ok`.

## Usage line
    python -I tools/lrcheck.py [--layers mixed4b,mixed4c,mixed4d,mixed4e] [--named ...] [--selftest]
