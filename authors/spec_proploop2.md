# Spec: tools/proploop2.py (round-25 question tables plus a standing question set)

Write one new file, `tools/proploop2.py`. Read only this spec and `tools/proploop.py` (and what it imports:
`tools/claude_call.py`, `tools/run_subjects.py`, `tools/bundlemax.py`, `tools/rangeview.py`, `bank/img_acts.py`), plus
the data files named below for their structure. Do not open any other file, and do not run anything. Reuse
proploop's functions and constants wherever this spec says "as proploop".

## Purpose
proploop.py asked, per unit, 8 questions written from explanations of the unit. Ask the same photos a fixed set of
extra questions, the same for every unit, and redo proploop's statistics, tables and revisions with all questions.

## Inputs
- `--units-file` (default `authors/round25_units.json`).
- Per unit, proploop's caches in `runs/proploop/<unit>/`: `questions.json` (its kept questions), `answers.json` (per
  batch: photo indices and parsed answers); the value caches `runs/proploop/values_<layer>.npy`. Stop with an error if
  any is missing, or if any round-25 batch is recorded as failed.
- Standing questions: `authors/standing.reply.json`, key "questions" (a list in proploop.parse_questions's format).
- Explanations, photos, names, labels, class names, the photo set: as proploop.

## Questions and answers (per unit)
Questions: the unit's round-25 kept questions in order, then the standing questions in order (Q1-Q8 then Q9-...).
Answers to the round-25 questions: from its answers.json. Answers to the standing questions: new calls, the batches
exactly as proploop (proploop.answer_batches on the same set), the answers prompt as proploop with only the standing
questions, the same parsing, retry and failure rule. Cache `runs/proploop2/<unit>/answers.json` keyed by batch number
as proploop. Each photo's answer row = its round-25 row followed by its standing row. Before any call, check that the
round-25 batches' photo lists equal proploop.answer_batches on the set; stop if not.

## Statistics, tables, revisions
Exactly as proploop on the merged rows and the full question list: real and null (the same permutation rule and
seed), proploop.table_text, revisions keyed "real_s1" ... "null_s3" with proploop's prompt, cached in
`runs/proploop2/<unit>/revisions.json`. Calls: at most WORKERS = 3 in flight; is_error replies are not cached; reruns
redo only what is not cached.

## Report
As proploop, in `runs/proploop2/`: `report.txt` (printed), `<unit>/stats.json`, `readings_all.json`. In the report,
mark the standing questions with "(standing)" after their number.

## Interface (a separate tester writes tests against these; keep the names and behaviour)
- `merge_rows(rows_a, rows_b)`: two lists of the same length of answer rows (lists); returns the list of
  concatenated rows; error on a length mismatch.
- `all_questions(own, standing)`: the two question lists; returns own followed by standing (a new list).
- `rows_in_set_order(batches, set_ids)`: batches a list of (photo index list, parsed answers list of rows, or None for
  a failed batch); returns one row per photo of set_ids, in set order (error if a photo is missing or repeated; a
  failed batch's photos get rows of None, one per question, which needs the number of questions: take it as a third
  argument `nq`).

## Selftest
`--selftest`, no model or Claude: merge_rows, all_questions, rows_in_set_order (order, a failed batch, a missing
photo), and that the standing answers prompt holds only the standing questions. Print `selftest ok`.

## Usage line
    python -I tools/proploop2.py [--units-file authors/round25_units.json] [--selftest]
