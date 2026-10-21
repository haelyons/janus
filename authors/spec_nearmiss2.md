# Spec: tools/nearmiss2.py (blind maximum and near misses on a pool built from each explanation's own examples)

Write one new file, `tools/nearmiss2.py`. Read only this spec, `tools/claude_call.py`, `tools/bundlemax.py`,
`tools/nearmiss.py` and `bank/img_acts.py`. Do not open any other file, and do not run anything.

## Purpose
Score several explanations of the same unit by what they predict, on a pool of photos that no unit activation helped
choose. For each explanation, a blind agent names the ImageNet classes it predicts the unit fires on and near misses
it predicts it does not; the pool is drawn at random from those classes, so every explanation contributes its own
examples. Then, as in tools/nearmiss.py, an agent that sees only one explanation picks 6 fire and 6 near-miss photos
from the pool, and the unit's activations score the picks. A second score needs no picker: how well the unit's
activation separates all held photos of an explanation's fire classes from those of its near-miss classes.

## Inputs
- `authors/explanations/nearmiss2.json`: {unit: {explanation_id: {"text": ..., "source": ...}}}, unit strings as
  tools/bundlemax.py's parse_unit accepts ('mixedNx:c' or 'sum:mixedNx:a+b'). Explanation ids and sources are never
  shown to any agent.
- names = json `bank/img/names.json`; photo i is `bank/img/thumbs/{i:05d}.jpg`; labels = `np.load('bank/img/acts.npz')['classes']`
  (load only that key); class names = `torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1.meta['categories']`, label k
  is class name k. Held pool H = photos with `zlib.crc32(names[i].encode()) % 10 >= 7` (bundlemax.pools).
- The unit's activation on a photo is the "whole" value of tools/bundlemax.py: the spatial max of bundlemax.umap with
  the scales in `runs/bundlemax/<unit>/bundles.json` (key "scales"), computed with bundlemax's forward and no channels
  zeroed. ratio = value / M, M = max of `card_vals` in `runs/bundlemax/<unit>/values.npz`. Compute values for all of
  H, cache them in `runs/nearmiss2/<unit>/held_vals.npy`, and check them against that values.npz: for the ids in its
  `H`, they must match the row of `vH` whose `conditions` entry is "whole" (np.allclose, rtol 1e-3); stop if not.
  Load the model once, only if some unit's cache is missing, and process units one at a time.

## Class lists (one call per explanation)
A fresh single-turn `claude_call.converse('', [content])`, content one text block, verbatim with {names} (the 1000
class names joined by '; ') and {text} filled in:

  Below are the 1000 ImageNet-1k class names, separated by semicolons. A unit (one neuron) in an image
  classification network has been explained as follows:

  {text}

  Using only the explanation, judging each class by what its typical photo shows, list:
  - "fire": 8 to 15 classes on whose typical photo the unit should fire strongly;
  - "near_miss": 8 to 15 classes whose typical photo looks related to what the explanation describes or shares
    obvious features with it, but on which, according to the explanation, the unit should barely fire.
  Copy names exactly as listed and put no class in both lists. Reply with only JSON: {"fire": [...], "near_miss": [...]}

  Class names:
  {names}

Parse the last JSON object in the reply with both keys (do not rely on claude_call.parse_json's fallback). Keep names
that match a class name exactly; drop a name in both lists from both; record dropped names. Valid: at least 4 kept in
each list; otherwise retry once in a fresh conversation, then record the explanation's lists as invalid and leave it
out of the pool and the scores. Cache in `runs/nearmiss2/<unit>/classes.json` keyed by explanation id; skip cached.

## Pool (per unit)
Groups, in the order of the explanations file, two per valid explanation: its fire classes, then its near-miss
classes. A group's photos are the H photos whose label is in the group. With `rng = np.random.default_rng(zlib.crc32(unit.encode()))`,
cycle through the groups in order, each time drawing one photo uniformly (rng.integers) from that group's photos not yet
in the pool, skipping groups with none left, until the pool has 48 photos or no group has any left (then stop with an
error). Shuffle with `np.random.default_rng(1).permutation`, number 1-48. Sheet, tiles, numbering and the files
`runs/nearmiss2/<unit>/pool.json` (number, photo index, label name, ratio, the ids of the groups that drew it) and
`sheet.jpg`: as tools/nearmiss.py. No activation is used before the pool is fixed.

## Calls and scores
Exactly as tools/nearmiss.py: the same two text blocks verbatim around the sheet, 3 samples per explanation, fresh
conversations, up to 4 in parallel, the same reply validation and one retry, the same per-sample fire_hit (ratio >= 0.3),
near_ok (ratio < 0.1) and gap, and the same chance per unit. Cache in `runs/nearmiss2/<unit>/replies.json`.
Class AUC per explanation: the probability that a random H photo of its fire classes has a higher ratio than a random
H photo of its near-miss classes (ties count one half; computed exactly, no sklearn), with both counts, and the mean
ratio of each side.

## Report
`runs/nearmiss2/report.txt`, printed. Per unit: a header with M and the chance line (fire_hit, near_ok); the pool's
count of photos per group; then one line per explanation, in file order: id, source, fire_hit mean (min-max),
near_ok mean, gap mean, number of valid samples, class AUC, n fire / n near-miss photos, mean ratio fire / near-miss
classes; then the explanation ids ranked by fire_hit mean. Also save everything per unit in
`runs/nearmiss2/<unit>/scores.json`.

## Selftest
`--selftest`, without the model or Claude: class-list parsing and filtering (exact match, overlap dropped, too few),
pool drawing on synthetic groups (determinism, round robin, no duplicates, skipping an exhausted group, the error when
all run out), the AUC on small cases with ties against hand values, and the pick scoring as nearmiss.py. Print `selftest ok`.

## Usage line
    python -I tools/nearmiss2.py --units mixed4c:447,mixed4e:55 [--selftest]
