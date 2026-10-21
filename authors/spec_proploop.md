# Spec: tools/proploop.py (readings become photo questions, answered blind, checked within class)

Write one new file, `tools/proploop.py`. Read only this spec and `tools/claude_call.py`, `tools/run_subjects.py`,
`tools/bundlemax.py`, `tools/rangeview.py` and `bank/img_acts.py`. Do not open any other file except the data files
named below, and do not run anything. Import from those files wherever this spec says "as" one of them.

## Purpose
For each unit: an agent turns existing explanations of the unit into questions about a photo; a second agent, which
never sees the explanations or any activation, answers them for ordinary photos; the answers are compared with the
unit's activation within photo classes, against every other channel of its layer on the same photos and answers;
then an agent revises each explanation given the resulting table, or given the same table built from answers
shuffled within class.

## Inputs
- `--units-file` (default `authors/round25_units.json`): a JSON list of unit strings, single channels of mixed3a-mixed5b
  ('mixedNx:c').
- Explanations: `runs/rangeview/<unit>/readings.json`, the "explanation" of keys CP_s1, CP_s2, CP_s3 (in that order).
- Photos: 10000, photo i is `bank/img/thumbs/{i:05d}.jpg`; names = json `bank/img/names.json`; labels =
  `np.load('bank/img/acts.npz')['classes']` (load only that key); class names =
  `torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1.meta['categories']`.
- Values: the spatial max of every channel of a layer's output (the module of that name in the model of
  bank/img_acts.load_model), input as bundlemax.net_input, batches of 64, on CUDA when available (outputs moved to
  CPU), for all 10000 photos, one forward pass per batch hooking every layer that has a unit in the units file and
  stopping after the deepest (raise img_acts.Stop). Cache per layer `runs/proploop/values_<layer>.npy` (float32,
  10000 x channels). Load the model only if a cache is missing.

## Photo set (per unit, layer L, channel c)
Class mean of a label = mean of the unit's value over all photos with that label. The set's classes: the 12 labels
with the highest class mean (ties by lower label), in that order. The set: every photo of those labels, in photo
index order (n photos). X = the values of all m channels of L on the set (n x m).

## Questions (one call per unit)
A fresh single-turn `claude_call.converse('', [content])`, content one text block, verbatim with the fields filled
({explanations}: one line per explanation, "Explanation 1: <text>", "Explanation 2: ...", joined by blank lines):

  Below are explanations, written independently, of what one unit (a single neuron) of an image classification
  network responds to.

  {explanations}

  The explanations will be checked on ordinary photos: someone who never sees the unit's activity will answer
  questions about each photo, and the answers will be compared with how strongly the unit responds to it. Write up to
  8 such questions. Cover both whether a photo contains what the explanations describe and the ways in which what
  they describe can differ from one photo that contains it to another. Each question must be answerable by looking at
  a small photo, and needs 2 to 5 short answers that do not overlap, so that every photo fits exactly one of them;
  include an answer for photos to which the question does not apply. Reply with only JSON:
  {"questions": [{"question": "...", "answers": ["...", "..."]}]}

(Join wrapped lines of a paragraph with spaces; keep the paragraph breaks.) Parse the last JSON object in the reply
with the key "questions" (bundlemax.last_obj). Keep, in order, entries that are objects whose "question" is a
non-empty string and whose "answers" is a list of 2 to 5 strings, each non-empty after strip, distinct after strip;
store them stripped; keep the first 8 kept. Valid: at least 3 kept. Otherwise retry once in a fresh conversation,
then record the unit as failed and skip it. Cache `runs/proploop/<unit>/questions.json` (reply text, kept questions).

## Answers (per unit)
Order the set with `np.random.default_rng(zlib.crc32(unit.encode()) + 1).permutation(n)` and cut it into batches of
12 in that order (the last may be shorter). One fresh single-turn call per batch, content: for each photo of the batch,
a text block "Image {k}:" (k = 1, 2, ... within the batch) followed by the photo as run_subjects.img_block; then one
text block, verbatim with the fields filled:

  Answer every question below for each of the {b} images above. For each question choose exactly one of its answers,
  by letter, or "?" if you cannot tell.

  {questions}

  Reply with only JSON: {"images": [{"id": n, "answers": ["A", ...]}]}, one entry per image, each with one answer per
  question, in question order.

{questions}: one line per question, "Q{j}. {question} " followed by its answers as "A) {answer}", "B) {answer}", ...
joined by single spaces (j from 1). Parse: the last JSON object with the key "images"; valid when it is a list whose
"id" values are exactly 1..b, each once, and each "answers" is a list with one entry per question, each entry a string
that, stripped and upper-cased, is "?" or one of that question's letters. Answer of photo i to question q: the index
of its letter (0 for A), or None for "?". On an invalid reply retry once in a fresh conversation; then every photo of
the batch has None for every question, and the batch is recorded as failed. Cache `runs/proploop/<unit>/answers.json`
keyed by batch number (reply text, photo indices, parsed answers).

## Statistics (per unit; the same for every channel j of L)
- v_j = X[:, j] / max of X[:, j] (all 0 if the max is 0); R[:, j] = v_j minus the mean of v_j over the set's photos with
  the same label (centred within class).
- Per question q with k answers: the counts n_a of photos per answer, and the number of None ("could not tell").
  Compared answers: those with n_a >= 5. Kept photos: those whose answer is a compared answer. If fewer than 2 answers
  are compared, the question has no statistics.
- Over the kept photos: eta2_j = sum over compared a of n_a (mean_a R_j - mean R_j)^2 / sum over kept photos of
  (R_ij - mean R_j)^2, 0 when the denominator is 0. For each compared a: d_ja = (mean of R_j over photos with answer a
  - mean of R_j over the other kept photos) / sd of R_j over the kept photos (ddof 1), 0 when that sd is 0.
- Percentile of the unit's statistic = 100 x the share of the layer's m channels whose statistic is strictly below the
  unit's.
- Halves (report only): the kept photos with `zlib.crc32(names[i].encode()) % 2` equal to 0, and those equal to 1. In
  each half, the unit's eta2 percentile over that half's kept photos, using the compared answers that have at least 2
  photos in the half (None if fewer than 2 such answers).
- Null: the same statistics after permuting answers within class: with `rng = np.random.default_rng(zlib.crc32(unit.encode()))`,
  for each of the set's classes in the set's class order, the photos of that class (in set order) receive the answer
  rows (all questions together) of those same photos in the order `rng.permutation` of their count.

## Table text (real, and null from the permuted answers)
Verbatim with the fields filled; percentiles and counts as whole numbers (round half to even, Python round), d with
a sign and 2 decimals; {classes}: the set's class names in set class order, joined by ', ':

  # Checks on ordinary photos

  Someone who never saw the unit's activity answered the questions below about each of {n} photos: all photos of the
  {K} photo classes on which the unit is most active on average ({classes}). For each answer: the number of photos
  given it, and d, how much more (positive) or less (negative) active the unit is on them than on the other photos,
  after removing the differences between classes, in standard deviations. Each question also has a separation score,
  the share of the unit's within-class variation that its answers account for. Each number comes with this unit's
  percentile among all {m} units of its layer, computed on the same photos and answers (above 95 or below 5 is
  unusual for this layer).

then one block per question, in order, blocks separated by blank lines:

  Q{j}. {question}
  separation: percentile {p}
  - {answer}: {n_a} photos, d {d}, percentile {p}

with one "- " line per answer in answer order; a non-compared answer's line is "- {answer}: {n_a} photos (too few to
compare)"; then, if any photo has None for the question, "- could not tell: {count} photos". A question without
statistics has "separation: too few photos per answer to compare" and every answer line in the non-compared form.

## Revisions
For each table (real, null) and each explanation s = 1, 2, 3: a fresh single-turn `claude_call.converse('', [content])`,
content two text blocks: first, verbatim with the fields filled,

  Here is an explanation of what one unit (a single neuron) of an image classification network responds to, written
  by someone who looked at the photos on which it is most active:

  {explanation}

  Later, these checks were run:

  {table}

  Taking both into account, say what the unit responds to.

then run_subjects.TASK. Parse as bundlemax.parse_expl, one retry on a parse failure. Keys "real_s1" ... "null_s3".
Cache `runs/proploop/<unit>/revisions.json` (reply text, class, explanation).

## Calls
At most WORKERS = 3 calls in flight (a module constant). A reply with is_error is not cached and counts as a failure
for that call; reruns redo only what is not cached, as rangeview.py.

## Report
`runs/proploop/report.txt`, printed. Per unit: the set's classes; for each question, its text, its answers with n_a,
d, percentile (real), its eta2 percentile (real; halves; null); then each revision id with its class and the first 160
characters of its explanation. Also `runs/proploop/<unit>/stats.json` (everything above per question, real and null,
including the unit's eta2 and d values), and `runs/proploop/readings_all.json`: {unit: {revision id: explanation}}.

## Interface (a separate tester writes tests against these; keep the names and behaviour)
- `top_classes(unit_vals, labels, k=12)`: numpy arrays over all photos; returns a list of the k labels (ints) with the
  highest class mean, ties by lower label.
- `parse_questions(text)`: the kept questions as a list of {"question": str, "answers": [str, ...]} by the rule above,
  or None when fewer than 3 are kept or no object is found.
- `questions_block(questions)`: the {questions} text of the answers prompt.
- `parse_answers(text, b, questions)`: a list of b lists (image 1 first), each with one entry per question (int answer
  index or None), or None when the reply is invalid.
- `centre(X, labels)`: X n x m, labels length n; returns R (scaling by each column's max, then centring within label).
- `question_stats(R, ans, k)`: R n x m, ans a length-n list of int or None, k answers; returns None when fewer than 2
  answers are compared, else a dict {"compared": [answer indices], "eta2": array of m, "d": {answer index: array of m}}.
- `percentile(values, j)`: 100 x the share of values strictly below values[j].
- `permute_within(rows, set_labels, class_order, rng)`: rows a list of n answer rows; returns the permuted list by the
  null rule above.
- `table_text(classes, n, m, questions, qs)`: classes a list of names, questions as parse_questions returns, qs one
  entry per question: None (no statistics) or {"counts": [n_a per answer], "unclear": int, "compared": [indices],
  "d": {index: float}, "d_pct": {index: float}, "eta2_pct": float}; returns the table text above.

## Selftest
`--selftest`, no model or Claude: top_classes ties; parse_questions (bad entries dropped, duplicates after strip, the
8 cap, too few); questions_block format; parse_answers (missing id, wrong length, bad letter, lower-case and "?");
centre on a hand case and a zero column; question_stats against hand values (eta2, d, an answer with < 5 photos
dropped, None answers dropped, None when < 2 compared, zero sd); percentile; permute_within determinism and that each
class keeps its own rows; table_text on a hand case including a non-compared answer, "could not tell" and a question
without statistics. Print `selftest ok`.

## Usage line
    python -I tools/proploop.py [--units-file authors/round25_units.json] [--selftest]
