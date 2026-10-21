# Spec: tools/curves.py (synthetic curve stimuli: which channels of a layer respond to curves?)

Write one new file, `tools/curves.py`. Read only this spec, `tools/battery.py` and `bank/img_acts.py`.

## Stimuli (224x224 RGB, white background, black stroke)
Draw at 4x size (896x896) and downsample with LANCZOS, so strokes are antialiased; stroke width 3 px after
downsampling.
- Curves: arcs of a circle spanning 90 degrees, radius r in {16, 32, 48} px. The arc's midpoint sits at the image
  centre; the circle's centre lies at distance r from that midpoint in direction theta (degrees, 0 = toward the
  image's right, increasing counter-clockwise as seen on screen), for theta = 0, 15, ..., 345 (24 orientations).
- Lines: straight segments of length 50 px centred on the image centre, at angles 0, 15, ..., 165 (12).
- Blank: the white image.

## Activations
InceptionV1 as in tools/battery.py; the post-ReLU output of `--layer` (default mixed3b; any `mixedNx` module); per
channel and stimulus, the spatial max.

## Per channel
curve_peak = max over curves; line_peak = max over lines; blank = blank value; ratio = curve_peak / (line_peak + 1e-6);
tuning = responses to the r = 32 curves over the 24 orientations; pref = the theta of the largest; osi = 1 - circular
variance of the tuning, |sum_k t_k exp(i theta_k)| / sum_k t_k (t clipped at 0); best_r = the radius with the
largest response at pref.

## Report (`runs/curves/report.txt`, also printed)
Layer medians of curve_peak, ratio and osi. The 20 channels with the largest ratio among channels whose curve_peak is
above the layer's 75th percentile, each with all numbers. `--report a,b,c`: those channels, each with percentiles of
curve_peak, ratio and osi. `--sheet c`: `runs/curves/sheet_{c}.jpg`, the r = 32 curve stimuli in a row of 24 tiles
(64 px), each labelled with the channel's response divided by its curve_peak (2 decimals). Save all numbers to
`runs/curves/{layer}.npz`.

## Selftest
`--selftest`, model-free: stimulus sizes and value range, for a curve at theta = 0 (circle centre to the right of the midpoint) the leftmost dark pixel column is within
2 px of x = 112 and the mean x of dark pixels is greater than 112; lines are symmetric under 180 degree
rotation, osi is 1 for a single-orientation tuning and 0 for a flat one. Print `selftest ok`.

## Usage
    python -I tools/curves.py [--layer mixed3b] [--report 1,2] [--sheet 3] [--selftest]
