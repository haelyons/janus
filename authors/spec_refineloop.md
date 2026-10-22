# Spec: tools/refineloop.py (conditional refinements of earlier predictions, checked blind on other photos)

Write one new file, `tools/refineloop.py`. Read only this spec, `tools/contrastloop.py` and `tools/proploop.py` (and
what they import: `tools/claude_call.py`, `tools/run_subjects.py`, `tools/bundlemax.py`, `tools/rangeview.py`,
`bank/img_acts.py`), plus the data files named below for their structure. Do not open any other file, and do not run
anything. Reuse their functions and constants wherever this spec says "as contrastloop" or "as proploop".

## Purpose
contrastloop.py left, per unit, predictions (a question, its answers, the answers whose photos the unit will be most
and least active on within a class) checked on the checked photos T. This tool picks one earlier prediction per unit
and arm, shows an agent photos of T given each of that prediction's two compared answers, and asks for conditional
predictions: a new question that splits one of those answers. They are checked on the looked-at photos D, which the
agent never sees, within the photos given the split answer. Two arms:
- F (family): the earlier prediction is picked by comparing the unit with its 3 peers (contrastloop's similar units);
  the agent sees all earlier predictions' results for the unit and the peers, and photos on which the unit and one
  peer disagree.
- R (refinement alone): the earlier prediction is the unit's best one; the agent sees the unit's results only, and the
  photos on which the unit is most and least active.

## Inputs
- `--units-file` (default `authors/round28_units.json`): unit strings as proploop.parse_unit.
- Per unit, contrastloop's outputs in `runs/contrastloop/<unit>/`: `stats.json` (keys used: layer, chan, m, classes,
  peers (list of {chan, unit, corr}), D, T, samples: {"s1".."s3": {"predictions": null or a list of {question,
  answers, most, least, question_index, ...}}}), `answers.json` (contrastloop's answer batches on T: photos, answers,
  failed), `panels.json`. Stop with an error if any is missing or any T batch failed.
- Values, labels, names, class names, photos: as contrastloop (proploop.layer_values with proploop's OUT).
- Outputs in `runs/refineloop/`.

## Earlier predictions on T (per unit, layer L, channel c, peers p1-p3)
- R_T = proploop.centre(V[T], labels[T]); each T photo's answer row from answers.json.
- The earlier predictions: every prediction of s1, s2, s3 in that order (j = 1, 2, ...). For each, ans = the T photos'
  answers to its question_index; contrastloop.contrast(R_T, ans, most, least). Not checkable: record "too few".
- Checkable: the unit's d_u and percentile P_u (proploop.percentile(d, c)); held as contrastloop.outcome. Per peer p:
  d_p, P_p (proploop.percentile(d, p)); rho_p = Pearson correlation of R_T[:, c] and R_T[:, p] over the compared photos
  (those given the most- or least-active answer), 0 when undefined; z_p = (d_u - d_p) / sqrt(2 (1 - rho_p) (1/n_hi +
  1/n_lo)), 0 when 1 - rho_p <= 1e-12.
- Kind of a checkable prediction: "shared" when held and some peer has P_p >= 95 (its peer: the one with the highest
  P_p, ties by peer order); "own" when held otherwise; "differs" when not held and some peer has |z_p| >= 2 (its peer:
  the one with the highest |z_p|, ties by peer order); else "none".

## Triggers (per unit)
- Arm F: if any prediction is "shared", the "shared" one with the highest P_u (ties by lower j), with its peer;
  otherwise, if any is "differs", the "differs" one with the highest |z_p| of its peer (ties by lower j), with that
  peer; otherwise none (no arm-F calls for the unit; recorded).
- Arm R: the checkable prediction with the highest P_u (ties by lower j); none if no prediction is checkable.
- Groups: the trigger's two compared answers, most-active first: the T photos given each.

## Panels (per arm, per group G; ids in photo index order before sorting; ties by lower photo id)
- Arm F (peer k): e = R_T[:, c] - R_T[:, k] over G's photos. "more": the 3 with the highest e; "less": the 3 with the
  lowest e, lowest first. When G has fewer than 6 photos, use floor(|G| / 2) for each instead of 3.
- Arm R: "most": the 3 with the highest R_T[:, c]; "least": the 3 with the lowest, lowest first; the same rule for
  small groups.
- pct(i, j) = 100 x V[i, j] / max of V[:, j] over all photos, as a whole number (as contrastloop's pct).

## Refinement calls (3 per arm: F_s1-F_s3, R_s1-R_s3; the same content within an arm)
Fresh single-turn calls as contrastloop's looking calls. Content, in order (each item one text block, verbatim with the
fields filled; paragraphs wrapped here are joined with single spaces, paragraph breaks kept; images as contrastloop):

1. Arm F text:

   Below is what is known about one unit (a single neuron) of an image classification network.

   Someone who looked at photos of the unit's activity wrote predictions about it, and each was checked on other
   photos. Someone who never saw the unit's activity answered the prediction's question about each photo. d is how
   much more active the unit is on photos given the predicted most-active answer than on photos given the predicted
   least-active answer, after removing the differences between photo classes, in standard deviations; the
   percentile places it among all {m} units of the unit's layer on the same photos and answers. A prediction held
   when its percentile is at least 95. The same was computed for the 3 units of the layer whose average activity
   over photo classes is most like this unit's (similar units 1 to 3). A prediction that held for this unit and also
   for a similar unit describes something they share, not what is particular to this unit. A prediction on which
   this unit and a similar unit clearly differ points to something particular to one of them.

   {table}

   Prediction {j} is examined below, with similar unit {k}. For each of the two answers it compares, there are photos
   given that answer: first some on which this unit is much more active than similar unit {k}, then some on which it
   is much less active, each relative to its own average on the photo's class. The percentages are each unit's
   activity as a share of the most it reaches on any of 10000 photos.

   Arm R text: the same first two paragraphs without their last three sentences (from "The same was computed"), then
   {table} (arm R form), then:

   Prediction {j} is examined below. For each of the two answers it compares, there are photos given that answer:
   first those on which the unit is most active, then those on which it is least active, each relative to its
   average on the photo's class. The percentages are the unit's activity as a share of the most it reaches on any of
   10000 photos.

2. Per group g = 1, 2: a text block `Answer {g}: "{answer}"`; then a text block `This unit much more active than
   similar unit {k}:` (arm R: `Most active:`) and per photo i = 1, 2, ... of "more" (arm R: "most") a text block
   `Photo {g}.{i} ({class name}): this unit {pct_c}%, similar unit {k} {pct_k}%` (arm R: `Photo {g}.{i} ({class
   name}): {pct_c}%`) and the image; then a text block `This unit much less active than similar unit {k}:` (arm R:
   `Least active:`) and the photos of "less" (arm R: "least") the same way, numbering continuing.
3. Arm F text:

   First, in "notes", describe each photo above in a few words. Then, within one of the two answers, find what
   separates the photos on which this unit is more active than similar unit {k} from those on which it is less
   active, and write it as up to 2 predictions. Each prediction splits one of the two answers: a new question to be
   asked about new photos given that answer, with 2 to 5 short answers that do not overlap, so that every photo fits
   exactly one of them, including one for photos to which it does not apply; the answer on whose photos you predict
   this unit will be most active; and the answer on whose photos you predict it will be least active. Each prediction
   will be checked on new photos given the answer it splits, within the same class, against all units of the layer,
   and should hold for this unit but not for similar unit {k}. Reply with only JSON: {"notes": ["..."],
   "predictions": [{"splits": "...", "question": "...", "answers": ["...", "..."], "most_active": "...",
   "least_active": "..."}]}, where "splits" is one of the two answers above, exactly as written.

   Arm R text: the same, with "the photos on which this unit is more active from those on which it is less active"
   in place of "the photos on which this unit is more active than similar unit {k} from those on which it is less
   active", and "and should hold for this unit." in place of "and should hold for this unit but not for similar unit
   {k}."

{table}, one block per earlier prediction j, blocks separated by blank lines:

    Prediction {j}. {question}
    Answers: {answers joined by '; '}
    Predicted most active: {most-active answer}. Predicted least active: {least-active answer}.
    This unit: d {d_u}, percentile {P_u}. Similar unit 1: d {d}, percentile {P}. Similar unit 2: ... Similar unit 3: ...
    {verdict}

d with a sign and 2 decimals, percentiles whole (proploop._whole). {verdict}: shared: `Held for this unit and for
similar unit {k}.`; own: `Held for this unit only.`; differs: `Did not hold for this unit; this unit and similar unit
{k} clearly differ on it.`; none: `Did not hold for this unit, and no similar unit clearly differs from it.` A
prediction that is not checkable has, after its first three lines, only `Too few photos to check.` Arm R form: the
fourth line is `This unit: d {d_u}, percentile {P_u}.` and the verdict `Held.` or `Did not hold.`

Parse: the last JSON object with the key "predictions" (bundlemax.last_obj). Keep, in order, entries of
"predictions" (if a list) that pass contrastloop.parse_predictions's entry rule (question, answers, most_active,
least_active) and whose "splits" is a string that, stripped and casefolded, equals exactly one of the trigger's two
compared answers' stripped casefolded text. Store as contrastloop stores a prediction, plus "splits": that answer's
index in the trigger's answers list. Keep the first 2 kept. Valid: at least 1 kept ("notes" is not checked). Retry
once (proploop.attempt); a sample still invalid contributes nothing. Cache `runs/refineloop/<unit>/refine.json`
(proploop.cached_calls), keys F_s1-F_s3, R_s1-R_s3.

## Answers on D (per unit)
Questions: the distinct trigger questions (arm F's, then arm R's if it is another earlier prediction; one entry when
both arms' triggers have the same question and answers), then the kept refinements of F_s1, F_s2, F_s3, R_s1, R_s2,
R_s3 in that order (question and answers). Batches: proploop.answer_batches(unit, D). Content: as
proploop.answers_content, except that the last text block uses this prompt (verbatim, fields as proploop's):

  Answer every question below for each of the {b} images above. For each question choose exactly one of its
  answers, by letter, or "?" if you cannot tell. Unless a question says otherwise, left and right mean the left and
  right of the photo as you look at it.

  {questions}

  Reply with only JSON: {"images": [{"id": n, "answers": ["A", ...]}]}, one entry per image, each with one answer
  per question, in question order.

Parsing, retry, failure and cache as proploop (`runs/refineloop/<unit>/answers.json`, keyed by batch number). Before
using a cache, check its batches' photo lists equal the batches; stop if not. If the refinement calls are not all
cached, ask nothing yet and report the unit as incomplete.

## Checks on D (per unit; real, and null)
- R_D = proploop.centre(V[D], labels[D]).
- Trigger re-check: contrastloop.contrast on D with the trigger's question's answers, most and least: the unit's d,
  percentile and outcome; each peer's d and percentile.
- Refinement (conditional): ans_i = the photo's answer to the refinement's question when its answer to the trigger's
  question is the split answer, else None; contrastloop.contrast(R_D, ans, most, least). Record n_hi, n_lo, the unit's
  d, percentile and outcome ("too few" when not checkable), each peer's d and percentile, and "distinctive": held and
  the lowest peer percentile below 50.
- Report only: the same refinement contrast without the condition (all D photos), and with the condition but only on
  D photos not shown in round 27 (not in panels.json's range or peer photos).
- Null: the D answer rows permuted with proploop.permute_within(rows, labels[D], classes,
  np.random.default_rng(zlib.crc32(unit.encode()) + 2)), then the same.

## Report
`runs/refineloop/report.txt`, printed. Per unit: peers; each earlier prediction's T statistics and kind; each arm's
trigger (prediction, kind, peer) and panel photo ids; per refinement: arm and sample, the split answer, question,
answers, most and least, real and null n_hi, n_lo, d, percentile, outcome, peers' percentiles, distinctive, and the
report-only checks; the trigger re-checks; a tally per arm of outcomes (real, null). Also `runs/refineloop/<unit>/
stats.json` with all of it.

## Calls
As proploop: at most WORKERS = 3 in flight, is_error replies not cached, reruns redo only what is not cached.

## Interface (a separate tester writes tests against these; keep the names and behaviour)
- `z_score(d_u, d_p, rho, n_hi, n_lo)`: by the rule above.
- `t_record(R, ans, most, least, c, peers)`: peers a list of channels; None when not checkable, else {"n_hi", "n_lo",
  "d_u", "pct_u", "held", "peers": [{"chan", "d", "pct", "rho", "z"}]}.
- `kind(rec)`: (kind, peer channel or None) by the rule above; ("too few", None) for None.
- `trigger_f(recs)`: recs one per earlier prediction (t_record outputs, None allowed); (index into recs, peer
  channel, kind) or None.
- `trigger_r(recs)`: an index into recs or None.
- `panel_f(R, ids, ans, a, c, k, n=3)`: R rows aligned with ids (a list of photo ids), ans aligned with ids, a the
  answer index of the group; returns (more, less) lists of photo ids.
- `panel_r(R, ids, ans, a, c, n=3)`: returns (most, least).
- `parse_refined(text, trigger_answers)`: trigger_answers the trigger question's answer list; returns
  {"notes": the notes as given or None, "predictions": [...]} or None.
- `conditional(parent_ans, child_ans, a)`: lists aligned by photo; the child answer where the parent answer is a,
  else None.
- `table_text(rows, arm)`: rows one per earlier prediction: {"question", "answers", "most", "least", "rec": t_record
  output or None, "kind", "peer_no": 1-3 or None}; arm "F" or "R"; returns {table}.
- `refine_content(arm, m, table, j, k_no, groups, block=None)`: groups a list of 2 entries {"answer": text, "hi":
  [(photo id, class name, pct_c, pct_k)], "lo": [...]} (pct_k ignored for arm R); returns the content list above.

## Selftest
`--selftest`, no model or Claude: z_score (including rho 1); t_record on a hand case (too few; peer d, pct, rho);
kind for each kind and ties; trigger_f preferring shared over differs, ties, none; trigger_r ties and all None;
panel_f and panel_r order, ties and a small group; parse_refined (splits case and spaces, a split that is not a
compared answer, the 2 cap, bad entries); conditional; table_text for every verdict, both arms, a too-few row;
refine_content block order, texts and image count for both arms. Print `selftest ok`.

## Usage line
    python -I tools/refineloop.py [--units-file authors/round28_units.json] [--selftest]
