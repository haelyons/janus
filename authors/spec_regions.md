# Spec: tools/regions.py (do labelled image regions push a channel up or down?)

Write one new file, `tools/regions.py`. Read only this spec, `tools/battery.py`, `tools/claude_call.py` and
`bank/img_acts.py`. Do not open any other file in the repo.

## Purpose
A claim to test: within a photo, regions containing one kind of shape raise a channel's activation and regions
containing another kind lower it, whatever the object is. Regions are labelled by a separate vision model that
never sees any activation. The code must not favour either direction.

## Photos
- names = json `bank/img/names.json`; photo i is `bank/img/thumbs/{i:05d}.jpg` (224x224). Held pool
  H = indices with `zlib.crc32(names[i].encode()) % 10 >= 7`.
- `bank/img/acts.npz` key `maps`: the pre-ReLU 14x14 map of the recorded channel (given as `--unit`, a channel of
  mixed4e) for every photo.
- Selection: 60 photos drawn from H with `np.random.default_rng(0).choice(H, 60, replace=False)`, plus the 20
  photos of H not already drawn with the highest spatial max of `maps`. Record which subset each came from.
  Shuffle the 80 with `np.random.default_rng(1).permutation` and number them 1-80 in that order.

## Labelling (cached in `runs/regions/labels.json`; skip if present)
- Sheet per photo: upscale to 448x448, draw a 7x7 grid (thin lines in a colour visible on any photo, such as
  1 px white over 1 px black), and write the cell names in small text at the top-left of each cell: rows A-G from
  top to bottom, columns 1-7 from left to right (A1 top-left).
- Send batches of 5 sheets per call with `claude_call.converse('', [content])` (a single turn, fresh conversation per
  batch, up to 4 batches in parallel). Content is a list of Anthropic blocks: for each sheet a text block
  "Image {id}:" then an image block {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
  "data": ...}}, then this text block, verbatim:

  Each image has a 7x7 grid. Cells are named by row letter (A-G, top to bottom) and column number (1-7, left to
  right). For each image, list the cells that contain a pointed shape: a tip, a point, a corner, or a V- or
  wedge-shaped part of something. Then list the cells that contain a round shape: a circle, disc, ring, dome or a
  round part of something. A cell can be in both lists or in neither. Reply with only JSON:
  {"images": [{"id": n, "pointed": ["A1", ...], "round": ["B2", ...]}]}

- Parse with `claude_call.parse_json`; retry a batch once if any id is missing or a cell name is invalid. Save the
  raw replies too (`runs/regions/replies.json`).

## Analysis
- Cell value: the mean of the 2x2 block of the 14x14 map that falls in that cell. P = cells labelled pointed only,
  R = round only, O = unlabelled; cells labelled both are dropped.
- For the unit's pre-ReLU map (from acts.npz), for each photo with at least one P and one R cell: d_PR = mean(P) -
  mean(R). For photos with at least one O cell also d_PO = mean(P) - mean(O) and d_RO = mean(R) - mean(O).
- Report for each of d_PR, d_PO, d_RO, for all photos and separately for the random 60 and the top 20: n, median,
  fraction > 0, and a two-sided sign test p (scipy.stats.binomtest, zeros dropped). Also the fraction of R, P and O
  cells whose pre-ReLU value is negative.
- Comparison across the layer: run InceptionV1 (as tools/battery.py does) on the 80 photos and take the post-ReLU
  `mixed4e` 14x14 maps of all 832 channels. For each channel compute d_PR per photo on its post-ReLU map divided by
  the channel's strong level (median activation over its top-20, from tools/battery.py's caches), then the
  channel's median d_PR and fraction of photos with d_PR > 0 (zeros dropped). Report the unit's values on its own
  post-ReLU map and its percentile on both among the 832 channels, and the layer medians.
- Write everything to `runs/regions/report.txt` and print it.

## Selftest
`--selftest`, model-free and without calling Claude: cell parsing (valid and invalid names), 2x2 pooling on a
synthetic 14x14 map, d_PR / d_PO / d_RO on a synthetic map and labels with known answers, and that a cell in both
lists is dropped. Print `selftest ok`.

## Usage line (docstring)
    python -I tools/regions.py --unit N [--selftest]
