# Spec: tools/contrastloop.py (predictions from contrasts, checked blind on other photos)

Write one new file, `tools/contrastloop.py`. Read only this spec and `tools/proploop.py` (and what it imports:
`tools/claude_call.py`, `tools/run_subjects.py`, `tools/bundlemax.py`, `tools/rangeview.py`, `bank/img_acts.py`), plus
the data files named below for their structure. Do not open any other file, and do not run anything. Reuse
proploop's functions and constants wherever this spec says "as proploop".

## Purpose
Per unit: an agent looks at photos chosen to contrast the unit with itself (photos of one class, from most to least
active) and with similar units of its layer (photos on which they disagree), and writes hypotheses as predictions:
a question about a photo, its answers, and which answer's photos the unit will be most and least active on within a
class. A second agent, which never sees the unit's activity, answers the questions for other photos of the same
classes. Each prediction is checked against every other channel of the layer on the same photos and answers; then an
agent, shown each prediction's outcome (real, or from answers shuffled within class: null), says what the unit
responds to.

## Inputs
- `--units-file` (default `authors/round27_units.json`): unit strings as proploop.parse_unit.
- Photos, names, labels, class names: as proploop. Values: proploop.layer_values with `out` proploop's OUT (the
  cached `runs/proploop/values_<layer>.npy`; the model loaded only if a cache is missing, as proploop's main).
- Outputs in `runs/contrastloop/`.

## Photos (per unit, layer L, channel c, V = L's values over all 10000 photos, m channels)
- S = V with each column divided by its max over all photos (a column with max 0 is all 0). pct(i, j) = 100 x S[i, j]
  as a whole number (proploop._whole).
- Classes: proploop.top_classes(V[:, c], labels, k=24), in that order.
- Split: the photos of those classes with `zlib.crc32(names[i].encode()) % 2 == 0` are the looked-at photos D, those
  with 1 the checked photos T, each in photo index order.
- Range panel: the first 4 classes; for each, its D photos sorted by V[:, c] highest first (ties by lower photo
  index); when more than 6, the 3 highest and the 3 lowest (in that order).
- Peers: class mean of channel j = mean of V[:, j] over the photos of a label, over all 10000 photos and all labels
  present. The peers are the 3 channels j != c whose class-mean vectors have the highest Pearson correlation with
  c's, ties by lower j; a correlation that is undefined (zero variance) counts as lowest.
- Peer panel, per peer p in order: Z = S[D] (columns c and p) minus their mean over the D photos with the same label;
  diff = Z_c - Z_p. "More": the 2 D photos with the highest diff (ties by lower photo index); "less": the 2 with the
  lowest diff (ties by lower photo index). Shown more then less.

## Looking (3 calls per unit, s1-s3, the same content)
Fresh single-turn `claude_call.converse('', content)` calls as proploop's calls. Content, in order (each line below is
one text block, verbatim with the fields filled; paragraphs wrapped here are joined with spaces, paragraph breaks
kept; every image is run_subjects.img_block of the photo's thumb, as proploop):

1. Text:

   Below are photos with the activity of one unit (a single neuron) of an image classification network on each, as
   a percentage of the most it reaches on any of 10000 photos.

   Part 1. For each of the 4 photo classes on which the unit is most active on average, photos of that class, from
   most to least active.

2. Per class j = 1..4: a text block `Class {j}: {class name}`; then per photo k = 1, 2, ...: a text block
   `Photo {j}.{k}: {pct}%` and the image.
3. Text:

   Part 2. Similar units: the 3 other units of the same layer whose average activity over photo classes is most
   like this unit's. For each, photos on which this unit is much more active than the similar unit, and photos on
   which it is much less active, each compared with their own averages on the photo's class.

4. Per peer j = 1..3: a text block `Similar unit {j}`; then per photo k = 1..4 (more, then less): a text block
   `Photo S{j}.{k} ({class name}): this unit {pct_c}%, similar unit {j} {pct_p}%` and the image.
5. Text:

   Whatever separates the more active from the less active photos of one class cannot be the class itself, and
   whatever separates this unit from a similar unit is what is particular to it. From these photos, write hypotheses
   about what this unit's activity depends on, each as a prediction to be checked on new photos of the same classes:
   someone who never sees the unit's activity will answer a question about each new photo, and the unit's activity
   will be compared between photos given different answers, within the same class. For each prediction give the
   question, which must be answerable by looking at a small photo; 2 to 5 short answers that do not overlap, so that
   every photo fits exactly one of them, including one for photos to which the question does not apply; the answer
   on whose photos you predict the unit is most active; and the answer on whose photos you predict it is least
   active. Write up to 4 predictions: the ones you expect to separate the unit's activity most clearly, including
   from the similar units. Also write a short explanation of what the unit responds to. Reply with only JSON:
   {"explanation": "...", "predictions": [{"question": "...", "answers": ["...", "..."], "most_active": "...",
   "least_active": "..."}]}

Parse: the last JSON object with the key "predictions" (bundlemax.last_obj). Valid explanation: "explanation" a
string, non-empty after strip. Keep, in order, the entries of "predictions" (if a list) that are objects whose
"question" is a non-empty string after strip, whose "answers" is a list of 2 to 5 strings, non-empty after strip and
distinct after strip and casefold, and whose "most_active" and "least_active" are strings that, stripped and
casefolded, equal exactly one answer's stripped casefolded text each, and not the same answer. Store question and
answers stripped, most and least as answer indices. Keep the first 4 kept. Valid reply: a valid explanation and at
least 1 kept prediction. Otherwise retry once in a fresh conversation (proploop.attempt); a sample still invalid
contributes nothing. Cache `runs/contrastloop/<unit>/looking.json` keyed s1-s3 (proploop.cached_calls), and save the
panels' photo ids, pct values, class names and peer channels in `runs/contrastloop/<unit>/panels.json`.

## Answers (per unit)
Questions: the kept predictions of s1, then s2, then s3 (question and answers only), Q1, Q2, ... A unit with no valid
sample is recorded as failed and skipped. Batches: proploop.answer_batches(unit, T ids); content
proploop.answers_content; parsing, retry, failure and cache as proploop (`runs/contrastloop/<unit>/answers.json`,
keyed by batch number). Before using a cache, check its batches' photo lists equal the batches; stop if not.

## Checks (per prediction; the same for every channel j of L)
- R = proploop.centre(V[T], labels[T]); ans = each T photo's answer to the prediction's question.
- n_hi, n_lo = the numbers of T photos given the most-active and the least-active answer. Checkable when both are at
  least 5. Over those n_hi + n_lo photos: d_j = (mean of R_j over the most-active photos - mean over the least-active
  photos) / sd of R_j over all of them (ddof 1), 0 when that sd is 0.
- The unit's percentile: proploop.percentile(d, c). Outcome: "held" when the percentile is at least 95 and the unit's
  d is above 0; else "not held"; "too few" when not checkable.
- Null: the same after proploop.permute_within(rows, labels[T], classes, np.random.default_rng(zlib.crc32(unit.encode())))
  on the T photos' answer rows (classes in class order).

## Results text (per sample, real and null)
Verbatim with the fields filled ({n} = number of T photos, {m} = channels of L, {K} = 24):

  The predictions were then checked on new photos: {n} photos of the {K} photo classes on which the unit is most
  active on average, none of them among the photos looked at. Someone who never saw the unit's activity answered
  each question about each photo. For each prediction: the number of photos given each of the two answers it
  compares, and d, how much more active the unit is on the photos given the predicted most-active answer than on
  those given the predicted least-active answer, after removing the differences between classes, in standard
  deviations, with this unit's percentile among all {m} units of its layer on the same photos and answers. A
  prediction held when that percentile is at least 95.

then, separated by blank lines, one block per kept prediction of the sample, j = 1, 2, ...:

  Prediction {j}. {question}
  Answers: {answers joined by '; '}
  Predicted most active: {most-active answer}. Predicted least active: {least-active answer}.
  Result: {n_hi} and {n_lo} photos, d {d with a sign and 2 decimals}, percentile {whole}: {held | did not hold}.

or, when not checkable, the last line `Result: {n_hi} and {n_lo} photos: too few to check (each needs at least 5).`

## Readings (per valid sample s, real and null)
A fresh single-turn call, content two text blocks: first, verbatim with the fields filled,

  Here is an explanation of what one unit (a single neuron) of an image classification network responds to, written
  with the predictions below by someone who looked at photos of the unit's activity:

  {explanation}

  {results text}

  Taking both into account, say what the unit responds to.

then run_subjects.TASK. Parse as bundlemax.parse_expl, one retry. Keys "real_s1" ... "null_s3"; cache
`runs/contrastloop/<unit>/readings.json` as proploop's revisions.

## Calls
As proploop: at most WORKERS = 3 in flight, is_error replies not cached, reruns redo only what is not cached.

## Report
`runs/contrastloop/report.txt`, printed. Per unit: its classes, peers (unit names and correlations), each sample's
explanation (first 160 characters) and predictions with, real and null, n_hi, n_lo, d, percentile and outcome; a
tally of outcomes (real, null); each reading id with its class and the first 160 characters of its explanation. Also
`runs/contrastloop/<unit>/stats.json` (everything above, including all predictions as parsed) and
`runs/contrastloop/readings_all.json`: {unit: {reading id: explanation or null}}.

## Interface (a separate tester writes tests against these; keep the names and behaviour)
- `scaled(V)`: each column divided by its max over the rows (0 column when the max is 0).
- `split(set_ids, names)`: set_ids photo ids in index order; returns (D, T) by the crc32 rule, each a list in order.
- `range_panel(vals, labels, D, classes, k=4, cap=6)`: vals the unit's values over all photos; returns a list, per
  class of the first k, of photo id lists by the rule above.
- `peers(V, labels, c, k=3)`: returns a list of (channel, correlation) by the rule above.
- `peer_photos(S, labels, D, c, p, n=2)`: S scaled values over all photos; returns (more, less), lists of photo ids.
- `parse_predictions(text)`: {"explanation": str, "predictions": [{"question", "answers", "most", "least"}]} by the
  rule above, or None.
- `contrast(R, ans, hi, lo, min_n=5)`: returns (n_hi, n_lo, d), d an array over R's columns, or None when not
  checkable.
- `outcome(d_unit, pct)`: "held" or "not held" by the rule above.
- `results_text(n, m, K, preds, res)`: preds as parse_predictions's list; res one dict per prediction, {"n_hi",
  "n_lo", "d", "pct"} with the unit's d and percentile, both None when the prediction is not checkable; returns the
  text above.
- `looking_content(panel_a, panel_b, block=None)`: panel_a a list per class of (class name, [(photo id, pct)]);
  panel_b a list per peer of [(photo id, class name, pct_c, pct_p)]; block(id) gives an image block (default
  run_subjects.img_block of the thumb); returns the content list above.

## Selftest
`--selftest`, no model or Claude: scaled with a zero column; split on hand names; range_panel order, ties and the
cap; peers order, ties and an undefined correlation; peer_photos within-class centring and ties; parse_predictions
(bad entries dropped, case and spaces in most/least, most equal to least, the 4 cap, no explanation); contrast
against hand values, too few, zero sd; outcome at the boundary; results_text on a hand case with a held, a not held
and a too-few prediction; looking_content block order and image count. Print `selftest ok`.

## Usage line
    python -I tools/contrastloop.py [--units-file authors/round27_units.json] [--selftest]
