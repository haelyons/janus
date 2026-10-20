# Spec: tools/views.py (view labels for vehicle and dog photos; view tuning of every mid-layer channel)

Write one new file, `tools/views.py`. Read only this spec, `tools/battery.py`, `tools/claude_call.py` and
`bank/img_acts.py`. Do not open any other file in the repo.

## Purpose
A claim to test: some channels respond to the view of an object (which way it faces the camera, how it is framed)
in the same way for vehicles and for dogs. The code measures every channel's view tuning without favouring any
channel or any view.

## Photos
- names = json `bank/img/names.json`; photo i is `bank/img/thumbs/{i:05d}.jpg` (224x224); labels:
  `np.load(bank/img/acts.npz)['classes']` (ImageNet-1k indices).
- Vehicles: every photo whose class is in {407, 436, 468, 479, 511, 581, 609, 627, 656, 661, 717, 734, 751, 817}.
  Dogs: `np.random.default_rng(0).choice` of 300 photos among those with class 151-268. Use both pools.
- Shuffle all selected photos together with `np.random.default_rng(1).permutation`; number them 1..n in that order.

## Labelling (cached in `runs/views/labels.json`, raw replies in `runs/views/replies.json`; skip if present)
- Batches of 10 photos per call with `claude_call.converse('', [content])`, a fresh single-turn conversation per
  batch, up to 4 in parallel. Content: for each photo a text block "Image {id}:" then an image block
  {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": ...}} (the 224 px thumbnail),
  then this text block, verbatim:

  For each image, consider the main animal or vehicle. Give "facing": which way its front (face, or the front of
  the vehicle) points, as seen in the image: "toward" (facing the camera), "toward_left" (front visible, turned
  toward the image's left), "left" (side view, front pointing to the image's left), "away_left", "away" (seen from
  behind), "away_right", "right", "toward_right", or "unclear". Give "framing": "head" (only the head or the front
  fills most of the frame), "part" (head and part of the body), "whole" (the whole animal or vehicle is visible),
  or "unclear". Reply with only JSON: {"images": [{"id": n, "facing": "...", "framing": "..."}]}

- Parse with `claude_call.parse_json`; retry a batch once if an id is missing or a value is not in the lists.

## Activations
Run InceptionV1 as tools/battery.py does on the selected photos; take the spatial max of every channel of
`mixed4c` (512) and `mixed4d` (528), post-ReLU (module outputs). Cache in `runs/views/acts.npy`. Normalise each
channel by its maximum over the selected photos (channels whose maximum is 0 are NaN throughout).

## Analysis, per channel, separately for vehicles (V) and dogs (D)
- Folded facing bins (left and right merged): front = {toward}, three_quarter_front = {toward_left, toward_right},
  side = {left, right}, rear = {away, away_left, away_right}. Unclear is dropped. A bin needs at least 5 photos of
  that kind, else NaN.
- Tuning t_V, t_D: mean normalised activation per folded bin. eta2_V, eta2_D: between-bin sum of squares over
  total sum of squares across the photos of that kind in defined bins.
- corr: Pearson correlation of t_V and t_D over the bins defined for both (NaN if fewer than 3).
- pref_V, pref_D: the bin with the highest mean.
- The same three numbers for framing (bins head, part, whole): eta2f_V, eta2f_D, corrf.
- Left-right: mean normalised activation on {toward_left, left, away_left} minus on {toward_right, right,
  away_right}, for V and D (lr_V, lr_D).
- dogshare: share of dogs among the channel's 50 highest-activation selected photos.

## Report
Write `runs/views/table.csv` (one row per channel, both layers) and `runs/views/report.txt`:
- counts of labels by kind, facing and framing;
- for each layer, the medians of eta2_V, eta2_D, corr, eta2f_V, eta2f_D, corrf;
- the 15 channels of each layer with the highest min(eta2_V, eta2_D), with all their numbers;
- `--report 4d:12,4c:7,...`: those channels' numbers, each with its percentile within its layer on
  min(eta2_V, eta2_D), corr, min(eta2f_V, eta2f_D) and corrf.
- `--sheet 4d:12`: `runs/views/sheet_4d_12.jpg`, the channel's 8 top vehicles and 8 top dogs (112 px), each
  captioned with facing/framing and normalised activation to 2 decimals.

## Selftest
`--selftest`, model-free and without calling Claude: label validation, the folding, eta2 and corr on synthetic
data with known answers (including a NaN bin and a constant channel), lr sign. Print `selftest ok`.

## Usage line (docstring)
    python -I tools/views.py [--selftest] [--report 4d:N,...] [--sheet 4d:N]
