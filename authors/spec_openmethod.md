# Spec: tools/openmethod.py (look, then score: one pipeline run the same way on any image unit)

Write one new file, `tools/openmethod.py`. Read only this spec and `tools/claude_call.py`, `tools/run_subjects.py`,
`tools/bundlemax.py`, `tools/nearmiss.py`, `tools/nearmiss2.py`, `tools/simulate.py` and `bank/img_acts.py`. Do not open
any other file, and do not run anything. Import from those files wherever this spec says "as" one of them.

## Purpose
For each unit, collect explanations ("readings") from agents that see a card of the unit's data, then score every
reading by how well an agent that sees only that reading predicts the unit's activation on a fixed pool of photos.

## Inputs
- `--units-file` (default `authors/round20_units.json`): a list of {"unit": "mixedNx:c", "wrong_from": "mixedNx:c"}.
  Units are single channels of mixed3a-mixed5b; `wrong_from` names another unit of the same file.
- Photos, names, labels, class names, card pool C and held pool H: as tools/nearmiss2.py and tools/bundlemax.py.
- Arms: system prompts '' ("bare"), the stripped text of `arms/Q0_shared.txt` ("Q0") and of `arms/F1_forced.txt` ("F1").

## Activations (per unit, one unit at a time, model loaded once and only if a cache is missing)
The unit's value on a photo is the spatial max of channel c of the named layer's output (the module of that name in
the model of bank/img_acts.load_model), input as bundlemax.net_input, batches of 64, stopping the forward pass at
that layer by raising img_acts.Stop from a forward hook. Also keep the (row, col) of the max and the map size. Compute
for all of C and H; cache in `runs/openmethod/<unit>/values.npz`. M = max over C; ratio = value / M.

## Card (per unit)
`card.jpg`: the 20 C photos with the highest value, strongest first, as bundlemax.make_card (whole photos, 5 columns,
'#n   activation k'), plus on each photo a 2 px red rectangle around the cells of the map from (row-1, col-1) to
(row+1, col+1), clipped, each cell being 224 / map size pixels. `random.jpg`: 10 C photos drawn with
`np.random.default_rng(zlib.crc32(unit.encode())).choice` among C photos outside the top 20, in that order, same
layout without rectangles, numbered #1-#10. Card name: bundlemax.card_name(unit, 'whole'). Content, in order: text
block (below, verbatim with {name}), card.jpg, random.jpg (image blocks as run_subjects.img_block), text block
run_subjects.TASK.

  # Unit {name}

  Recorded from one unit (a single neuron) in an image classification network, run over several thousand natural
  photos. The first attached grid shows the 20 photos on which the unit is most active, strongest first; a red box
  marks where in each photo it is most active. Under each is its activation, scaled 0-10 relative to the strongest.
  The second grid shows 10 photos drawn at random from the same collection, with their activation on the same scale.

(Join wrapped lines with spaces; keep the paragraph break.)

## Readings
For each arm and sample 1, 2: a fresh single-turn `claude_call.converse(system_prompt, [content])`, up to 4 in
parallel, parsed as bundlemax.parse_expl; one retry on a parse failure, then the sample is missing; an is_error reply
is not cached. Cache in `runs/openmethod/<unit>/readings.json` keyed "bare_s1" etc., with the reply text, class and
explanation. Then the reading "wrong" = the explanation of `wrong_from`'s bare_s1 (so readings of every unit come
first, then wrong readings). Readings of a unit, in this order: bare_s1, bare_s2, Q0_s1, Q0_s2, F1_s1, F1_s2, wrong.

## Class lists and pool (per unit)
Class lists for each reading exactly as tools/nearmiss2.py (same prompt, parsing, filtering, retry), cached in
`classes.json`. Groups: for each valid reading in the order above, its fire classes then its near-miss classes. Pool:
with `rng = np.random.default_rng(zlib.crc32(unit.encode()))`, cycle through the groups; at each turn draw one class
uniformly (rng.integers) from the group's classes that are not yet in the pool and have at least 2 H photos; add 2 H
photos of it drawn without replacement (rng.choice); skip a group with none left; stop at 24 classes (48 photos), or
stop with an error if all groups run out first. Shuffle the 48 with `np.random.default_rng(1).permutation`, number
1-48; sheet and pool.json as nearmiss2 (pool.json also records the class pair id). No activation is used before the
pool is fixed.

## Scores
For each reading, 3 samples of predicted activation exactly as tools/simulate.py (same prompts, validation, retry,
Spearman, top12, consistency, rho of the mean prediction), cached in `sim.json`. Also per reading: within-pair score =
over the 24 same-class pairs whose two ratios differ by at least 0.1, the share where the mean prediction is higher
for the photo with the higher ratio (ties count one half), with the number of such pairs.

## Report
`runs/openmethod/report.txt`, printed, and per unit `scores.json`. Per unit: header (unit, M, map size); the bare
samples' classes; one line per reading: id, rho mean (min to max), rho of mean prediction, consistency, top12,
within-pair score (n), first 140 characters of the explanation; then ids ranked by mean rho, and "top: <id>". Also
write `runs/openmethod/readings_all.json`: {unit: {reading id: explanation}}.

## Selftest
`--selftest`, no model or Claude: the box geometry (clipping at edges, 14x14 and 7x7 maps), the random-row draw
(determinism, excludes the top 20), the paired pool (determinism, 2 photos per class, classes with < 2 H photos
skipped, exhausted groups skipped, the error), the within-pair score on hand cases with ties, reading order and the
wrong-reading assignment. Print `selftest ok`.

## Usage line
    python -I tools/openmethod.py [--units-file authors/round20_units.json] [--selftest]
