# Spec: tools/pairtable.py (a fixed question set answered blind, every answer pair compared, rows read by an agent)

Write one new file, `tools/pairtable.py`. Read only this spec, `tools/proploop.py`, `tools/contrastloop.py` and
`tools/refineloop.py` (and what they import: `tools/claude_call.py`, `tools/run_subjects.py`, `tools/bundlemax.py`,
`tools/rangeview.py`, `bank/img_acts.py`), plus `authors/standing.reply.json` and the data files named below for
their structure. Do not open any other file, and do not run anything. Reuse the named functions and constants of
those modules wherever this spec says "as proploop", "as contrastloop" or "as refineloop".

## Purpose
Per unit: someone who never sees the unit's activity answers a fixed set of 8 questions, the same for every unit,
about every photo of the unit's top photo classes. On the looked-at half of those photos, every two answers of each
question are compared within class, for the unit and every other channel of its layer. An agent reads the resulting
table (no photos), says what the unit responds to and picks up to 4 rows expected to hold on the other half; the same
is done on a table from answers shuffled within class. A fixed rule also picks rows from each table. The picked rows,
and every row of the real table, are checked on the other half.

## Inputs
- `--units-file` (default `authors/round29_units.json`): unit strings as proploop.parse_unit.
- Questions: the list "questions" of `authors/standing.reply.json` (8 entries, each {"question", "answers"}), in
  order, used as given; Q1 is the first.
- Photos, names, labels, class names, values: as contrastloop's main (values from proploop.layer_values with
  `out=proploop.OUT`, the model loaded only if a cache is missing).
- Outputs in `runs/pairtable/`.

## Photos (per unit, layer L, channel c, V = L's values over all 10000 photos, m channels)
- classes, set ids: proploop.photo_set(V[:, c], labels, k=24); D, T = contrastloop.split(set ids, names).
- Peers: contrastloop.peers(V, labels, c) (3 channels, in that order: similar units 1 to 3).

## Answers (one pass, shared by all units, before any unit's table)
- P = the union of all units' set ids, sorted ascending. Batches: proploop.answer_batches('standing', P).
- Content per batch: refineloop.d_content(batch, questions). Parse, one retry, failure and cache as contrastloop's
  answers (proploop.cached_calls with proploop's answer parser and answer fields), at `runs/pairtable/answers.json`,
  keyed "1", "2", ... by batch number. Before using that cache, contrastloop.check_answer_cache(path, batches).
- A photo's row: its 8 answers (answer index, or None for "?"); a photo of a failed batch has 8 None.
- If any batch is not cached after the pass (call errors), print which and stop; a rerun redoes only those.

## Pairs and rows
- pairs(k): every (a, b) with 0 <= a < b < k, ordered by a, then b.
- Rows of a photo set X (D or T, in set order) given one answer row per photo: R = proploop.centre(V[X], labels[X]).
  For each question q (in order) and each pair (a, b) of its answers (in pairs order): ans = each photo's answer to
  q; got = contrastloop.contrast(R, ans, a, b). Skip the pair when got is None. Else d = got[2]; when d[c] < 0, the
  row is (first, second) = (b, a) with d replaced by -d and the counts swapped, else (a, b). pct =
  proploop.percentile(d, c); peer_pct = [proploop.percentile(d, p) for each peer p]; gap = pct minus the mean of the
  3 peer_pct. Rows are numbered 1, 2, ... in this order.
- Row record: {"row", "q" (question index from 0), "first", "second", "n_first", "n_second", "d" (d[c]), "pct",
  "peer_pct", "gap"}.

## Tables (per unit, on D)
- real: D's answer rows. null: proploop.permute_within(D's answer rows, labels[D], classes,
  np.random.default_rng(zlib.crc32(unit.encode()) + 3)) (classes in class order).
- Table text: one text block, verbatim with the fields filled ({n} = number of D photos, {K} = 24, {classes} = the
  class names in class order joined by ", ", {m} = channels of L; whole numbers by proploop._whole; paragraphs
  wrapped here are joined with spaces, paragraph breaks kept):

  Below is what is known about one unit (a single neuron) of an image classification network. Someone who never saw
  the unit's activity answered the questions below about each of {n} photos of the {K} photo classes on which the
  unit is most active on average ({classes}). For each question, every two of its answers that were each given to at
  least 5 photos are compared. d is how much more active the unit is on the photos given the first answer than on
  the photos given the second, after removing the differences between photo classes, in standard deviations; each
  pair is written with the answer on which the unit is more active first. The percentile places d among all {m}
  units of the unit's layer on the same photos and answers (95 or above is unusual for this layer). The same
  percentile is given for the 3 units of the layer whose average activity over photo classes is most like this
  unit's (similar units 1 to 3): a row on which a similar unit is also at 95 or above describes something they
  share, not what is particular to this unit.

  then, after a blank line, one line per row:

  Row {row}. {question} "{first answer}" ({n_first} photos) more than "{second answer}" ({n_second} photos): d {d with a sign and 2 decimals}, percentile {pct}; similar units {peer_pct 1}, {peer_pct 2}, {peer_pct 3}.

  then, after a blank line:

  The rows will be checked on new photos of the same classes, none of them among those above, answered in the same
  way: a row holds when this unit's d, with the same answer first, is again at the 95th percentile of its layer or
  above. Say in a few sentences what this unit responds to, and choose up to 4 rows that you expect to hold for this
  unit on the new photos and that describe what is particular to this unit rather than what it shares with the
  similar units. Reply with only JSON: {"explanation": "...", "rows": [n, ...]}

## Readings (per unit: real_s1, real_s2, real_s3 on the real table; null_s1, null_s2, null_s3 on the null table)
- Fresh single-turn `claude_call.converse('', content)` calls through proploop.cached_calls; content = [the table's
  text block]. Cache `runs/pairtable/<unit>/readings.json`.
- Parse: the last JSON object with the key "rows" (bundlemax.last_obj). Valid: "explanation" a string, non-empty
  after strip, and "rows" a list; keep, in order, its entries that are ints (not bool) from 1 to the table's number
  of rows, each only at its first occurrence; keep the first 4 kept. Valid reply: a valid explanation and at least 1
  kept row; otherwise None (proploop.attempt retries once). Store {"explanation" (stripped), "rows"}.
- A table with no rows: no calls for it; its readings are recorded as not made.

## Picks (arms, per unit)
- "real_s1" ... "null_s3": the kept rows of that reading (rows of that reading's table); none when not valid.
- "rule_real", "rule_null": from that table, the rows with pct >= 95, sorted by gap highest first, ties by lower row
  number; the first 4.
- "every": every row of the real table.

## Checks on T (per picked row, the same for every arm)
- T rows: T's answer rows (real) and proploop.permute_within(T's answer rows, labels[T], classes,
  np.random.default_rng(zlib.crc32(unit.encode()))) (null). R = proploop.centre(V[T], labels[T]).
- For a row (q, first, second) and T rows: got = contrastloop.contrast(R, ans, first, second), ans = each T photo's
  answer to q. Record {"n_first", "n_second", "d", "pct", "peer_pct", "outcome", "distinctive"}: n_first and
  n_second the counts of T photos given first and second; when got is None, d, pct and peer_pct None, outcome
  "too few", distinctive false; else d = got[2][c], pct = proploop.percentile(got[2], c), peer_pct as above,
  outcome contrastloop.outcome(d, pct), distinctive = outcome "held" and min(peer_pct) < 50.
- Each pick: {"row", "q", "first", "second", "question", "first_answer", "second_answer", "D" (the row record from
  its table), "T" (real check), "T_null" (null check)}.

## Calls
main sets proploop.WORKERS = 1 before any call (one call at a time). is_error replies are not cached and reruns redo
only what is not cached (proploop.cached_calls).

## Outputs
- `runs/pairtable/<unit>/stats.json`: unit, layer, chan, m, classes, class_names, peers [{"chan", "unit"
  ("layer:chan"), "corr"}], D, T, questions, tables {"real": [rows], "null": [rows]}, readings {key: {"valid",
  "explanation", "rows"}} (valid false and the others None when not valid or not made), picks {arm: [pick]}, tally
  {arm: {"real": {outcome: count}, "null": {outcome: count}}} over the outcomes "held", "not held", "too few".
- `runs/pairtable/report.txt`, printed: per unit its classes, peers (unit names, correlations), the real table's
  rows (row, question number, first and second answers, counts, d, pct, peer_pct, gap), each reading's explanation
  (first 200 characters) and rows, and for each arm its picks with D pct, T d, pct, outcome, distinctive and T_null
  outcome, then the tally.

## Interface (a separate tester writes tests against these; keep the names and behaviour)
- `pairs(k)`: list of (a, b) as above.
- `table_rows(R, ans_rows, questions, c, peers, min_n=5)`: R a centred n x m array; ans_rows one list of answers
  (int or None) per photo, one per question; peers a list of channels; returns the list of row records above.
- `table_text(n, K, class_names, m, questions, rows)`: the table text above.
- `parse_rows(text, n_rows)`: {"explanation", "rows"} by the rule above, or None.
- `rule_rows(rows, k=4)`: the row numbers picked by the rule above, in order.
- `t_check(R, ans, first, second, c, peers)`: the check record above.

## Selftest
`--selftest`, no model or Claude: pairs for k = 1, 2, 4; table_rows on a hand case (a negative d swapped, a pair
with fewer than 5 photos on one side skipped, None answers ignored, percentiles and gap); table_text line format on
a hand case; parse_rows (bools, duplicates, out of range, non-ints dropped, the 4 cap, no explanation, no kept row);
rule_rows (the 95 cut, gap order, ties); t_check held, not held, too few and distinctive. Print `selftest ok`.

## Usage line
    python -I tools/pairtable.py [--units-file authors/round29_units.json] [--selftest]
