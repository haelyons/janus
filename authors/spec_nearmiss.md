# Spec: tools/nearmiss.py (blind maximum and near misses for image units)

Write one new file, `tools/nearmiss.py`. Read only this spec and `tools/claude_call.py`. Do not open any other file.

## Purpose
Score an explanation of a unit by what it predicts. An agent that sees only the explanation picks, from a fixed
pool of unlabelled photos, the ones the unit should fire on most and near misses it should not fire on; the unit's
recorded activations then score the picks. The same pool is used for every explanation of a unit.

## Inputs
- `authors/explanations/nearmiss.json`: {name: {"unit_card", "text", "source"}}.
- Per unit_card, the unit's maps and scale (act = spatial max of max(map, 0), divided by card_max):
  - U-121: `bank/img/acts.npz` key `maps` (N x 14 x 14), card_max 1527.0
  - U-945: `bank/img/maps_mixed4d_409.npy` (N x 14 x 14), card_max 2288.9
- names = json `bank/img/names.json`; photo i is `bank/img/thumbs/{i:05d}.jpg`; held pool H = indices with
  `zlib.crc32(names[i].encode()) % 10 >= 7`.

## Pool (per unit_card)
Strata over H by act: high >= 0.3, mid 0.05-0.3, low < 0.05. Draw 16 from each with
`np.random.default_rng(0).choice(..., 16, replace=False)` (one generator per unit, strata in the order high, mid, low),
shuffle the 48 with `np.random.default_rng(1).permutation`, number 1-48. Sheet: 8 columns x 6 rows of 112 px tiles,
each with its number in a white strip under it. Save `runs/nearmiss/pool_{unit_card}.json` (number, photo index,
act) and `runs/nearmiss/sheet_{unit_card}.jpg`.

## Calls
For each explanation, 3 samples, each a fresh single-turn `claude_call.converse('', [content])` (up to 4 in
parallel). Content: a text block, the sheet as an image block {"type": "image", "source": {"type": "base64",
"media_type": "image/jpeg", "data": ...}}, then a text block. First text block, verbatim with {text} filled in:

  A unit (one neuron) of an image classification network has been explained as follows:

  {text}

  The attached sheet shows 48 numbered photos. You have not seen how the unit responds to any of them.

Last text block, verbatim:

  Using only the explanation, choose the 6 photos on which the unit should fire most strongly, and 6 near misses:
  photos that look related to what the explanation describes or share obvious features with it, but on which,
  according to the explanation, the unit should barely fire. Reply with only JSON: {"fire": [six numbers],
  "near_miss": [six numbers]}

Parse the last JSON object in the reply that has both keys (claude_call.parse_json returns the wrong object when
there is no 'class', 'predictions' or 'edits' key, so do not rely on its fallback). Valid: 6 distinct integers in
1-48 per list and no overlap; retry once in a fresh conversation, then record the sample as invalid. Cache replies
in `runs/nearmiss/replies.json` keyed by name and sample; skip cached ones.

## Scores
Per sample: fire_hit = share of fire picks with act >= 0.3; near_ok = share of near_miss picks with act < 0.1;
gap = mean act of fire picks - mean act of near_miss picks. Per explanation: mean and min-max over valid samples.
Chance per unit: fire_hit = share of the pool with act >= 0.3, near_ok = share with act < 0.1, gap = 0. Write
`runs/nearmiss/report.txt` (one line per explanation with its unit, source and scores, then the chance lines) and print it.

## Selftest
`--selftest`, without the model or Claude: strata sizes and determinism of the pool on synthetic acts, reply
validation (good, duplicate, out of range, overlap, wrong count), scoring on a synthetic pool. Print `selftest ok`.

## Usage line
    python -I tools/nearmiss.py [--selftest]
