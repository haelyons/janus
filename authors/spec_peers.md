# Spec: tools/peers.py (compare one channel's response to an image edit against other channels)

Write one new file, `tools/peers.py`. Read only this spec, `tools/battery.py` and `bank/img_acts.py`. Do not open
any other file in the repo.

## Purpose
An experimenter edits a photo x into y and sees how much of one channel's activation survives. On its own that
number cannot say whether the drop is typical. This module gives the comparison: among the other channels of the
same layer that respond strongly to x, what share of their activation survives the same edit, and where the given
channel falls among them.

## Definitions
- Layer: InceptionV1 `mixed4e`, 832 channels, post-ReLU. a_c(img) = spatial max of channel c, computed exactly as
  tools/battery.py computes it (reuse its functions; import tools/battery.py, do not copy).
- Strong level s_c = median of a_c over channel c's top-20 photos, from tools/battery.py's caches
  (`runs/battery/orig.npy`, `pool.npy`, `topk.npy`). Load once and keep.
- Peers of x for unit u: channels c != u with a_c(x) >= 0.5 * s_c.
- r_c = a_c(y) / a_c(x) for each peer; unit_ratio = a_u(y) / a_u(x) (NaN if a_u(x) <= 0).

## Functions
- `baseline_from_acts(ax, ay, s, unit)`: pure numpy, ax and ay are length-832 arrays, s the strong levels. Returns a
  dict {"n": number of peers, "peer_median": median r_c over peers (NaN if n == 0), "unit_ratio",
  "unit_pct": 100 x fraction of peers with r_c <= unit_ratio (NaN if n == 0 or unit_ratio is NaN)}.
- `peer_baseline(model, pairs, unit)`: pairs is a list of (x, y) PIL RGB 224x224 images. Runs the layer on all
  2*len(pairs) images in one batched pass (batches of 64 if more) and returns a list of the dicts above.
- `load_model()` convenience wrapper that loads the weights as tools/battery.py does, so callers can share a model.

## Selftest
`python -I tools/peers.py --selftest`, model-free: on synthetic arrays check peer selection (threshold inclusive,
unit excluded), the median, the percentile (including ties counted as <=), and the NaN cases. Print `selftest ok`.
