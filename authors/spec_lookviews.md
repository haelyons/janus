# Spec: tools/lookviews.py (two added views of a unit's data, in a 2x2 design)

Write one new file, `tools/lookviews.py`. Read only this spec and `tools/claude_call.py`, `tools/run_subjects.py`,
`tools/bundlemax.py`, `tools/battery.py`, `tools/openmethod.py` and `bank/img_acts.py`. Do not open any other file
except `authors/battery.reply.json` (data it uses), and do not run anything. Import from those files wherever this spec
says "as" one of them.

## Purpose
Collect explanations of a unit from agents shown its data in four conditions: a base card, the base card plus crops
of the top photos to the unit's effective receptive field (C), the base card plus an invariance profile (P), and both
(CP). Nothing is scored here.

## Inputs
- `--units-file` (default `authors/round21_units.json`): a JSON list of unit strings, single channels of mixed3a-mixed5b.
- Photos, pools C and H, unit values (spatial max, its row and col, map size), M, the card `card.jpg` and the random
  row `random.jpg`, and the card text: exactly as tools/openmethod.py (import its functions), with caches under
  `runs/lookviews/<unit>/`.

## Effective receptive field and crops (per unit)
For each of the 20 card photos: input as bundlemax.net_input with gradients enabled; forward to the unit's layer
(stop there); take the output at channel c and the stored (row, col) of its max; backpropagate to the input; g = the
absolute input gradient summed over the 3 colour channels (224 x 224). Centre = the centre of the peak cell in pixels,
((col + 0.5) * 224 / size, (row + 0.5) * 224 / size). s = the smallest side in steps of 2 px from 8 to 224 such that
the square of side s centred there (clipped to the photo) holds at least 80% of the sum of g. S = the median of the 20
values of s, rounded to the nearest multiple of 8 and clamped to [32, 224]. Each crop: the square of side S centred
on the photo's peak centre, shifted (not shrunk) to lie inside the photo, resized to 160 x 160 (bicubic). `crops.jpg`:
5 columns, in card order, a white strip under each with '#n'. Save S and the 20 values of s in `rf.json`.

## Invariance profile (per layer, shared by its units)
P0 = `np.sort(np.random.default_rng(0).choice(C, 1000, replace=False))`. For the original photos and for each
transform in battery.TRANSFORMS (in its order): the spatial max of every channel of every layer that has a unit in the
units file, in one forward pass per batch that hooks all those layers and stops after the deepest. Cache per layer in
`runs/lookviews/profile_<layer>.npz`. Per channel: its top 20 = its 20 P0 photos with the highest original value
(skip photos whose original value is 0); retention = median over them of transformed / original. Per unit and
transform: its retention u, the layer median m, and its percentile p = 100 x the share of the layer's channels with
retention below u. Description of each transform: authors/battery.reply.json's "removes" for that name, cut at the
first ':' and then at the first '.', lower-cased; for rotate_180 "upright orientation", for mirror "left-right
orientation". Profile text block, verbatim with the fields filled (percentages as whole numbers):

  How the unit's activation changes when photos are transformed. For each transformation: the median share of its
  activation the unit keeps on its 20 strongest photos out of a random 1000, the median of the same number over all
  {n} units of its layer (each on its own 20 strongest photos), and this unit's percentile among them (low = this unit
  loses more than most units do).
  - {name} (removes {description}): keeps {u}%; layer median {m}%; percentile {p}

(one "- " line per transform, in battery.TRANSFORMS order; join wrapped lines of the first paragraph with spaces.)

## Conditions and readings
Content per condition, in order (image blocks as run_subjects.img_block):
- base: card text, card.jpg, random.jpg, run_subjects.TASK
- C: card text + crop sentence, card.jpg, random.jpg, crops.jpg, TASK
- P: card text, card.jpg, random.jpg, profile text, TASK
- CP: card text + crop sentence, card.jpg, random.jpg, crops.jpg, profile text, TASK
Crop sentence, appended to the card text after a space, verbatim with {S}: "The third grid shows the same 20 photos, in
the same order, each cropped to a {S} px square of the 224 px photo around the spot where the unit is most active."
For each condition, samples 1-3: a fresh single-turn `claude_call.converse('', [content])`, up to 4 in parallel,
parsed as bundlemax.parse_expl, one retry on a parse failure, an is_error reply not cached. Cache in `readings.json`
keyed "base_s1" ... "CP_s3", with reply text, class and explanation.

## Report
`runs/lookviews/report.txt`, printed: per unit, S; the profile lines (u, m, p); then one line per reading: id, class,
first 160 characters of the explanation. Also `runs/lookviews/readings_all.json`: {unit: {reading id: explanation}}.

## Selftest
`--selftest`, no model or Claude: the receptive-field rule on synthetic g (a point mass gives the minimum; uniform g
gives at least the side holding 80% of the area; a peak at a corner clips), the crop placement (shift inside, size S),
the retention and percentile on a synthetic layer (zero originals skipped), the description rule, the profile text
format, and the content order of each condition. Print `selftest ok`.

## Usage line
    python -I tools/lookviews.py [--units-file authors/round21_units.json] [--selftest]
