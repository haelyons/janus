# Spec: tools/r31_tables.py (pairtable's tables and checks on another photo pool, without readings)

Write one new file, `tools/r31_tables.py`. Read only this spec and `tools/pairtable.py`, `tools/proploop.py`,
`tools/contrastloop.py`. Do not open any other file, and do not run anything. No Claude calls.

## Purpose
`pairtable.run_unit` builds, for one unit, the pairwise tables on half D of its photos, three readings by Claude,
and the checks of the picked rows on half T. Do the same on a second photo pool (more photos per class), with no
readings: only the arms that need no call (`rule_real`, `rule_null`, `every`).

## Pools
- The reference pool: the repo's pool, `V` per layer from `proploop.layer_values(layers, None,
  np.arange(proploop.N_PHOTOS))`, labels from `proploop.ACTS` (`classes`), names from `proploop.NAMES`.
- The new pool, a directory `--pool` with `names.json` (list of names), `labels.npy` (int64) and
  `values_<layer>.npy` (float32, pool ids x channels of that layer).
- The new pool's answers, `--answers`: a `tools/vlm_answer.py` output (`{id (str): {"answers", "valid", ...}}`);
  a photo whose record is missing or has `valid` false has a row of None for every question.

## Interface
- `unit_new(u, ref, new, row_of, questions)`: `u` a `proploop.parse_unit` record; `ref` and `new` dicts with keys
  `V` (that unit's layer values, photos x channels), `labels` and `names` for each pool; `row_of` `{new pool id:
  answer row}`. Steps:
  1. classes = `pairtable.unit_photos(u, ref["V"], ref["labels"], ref["names"])[0]` and peers =
     `contrastloop.peers(ref["V"], ref["labels"], c)`: both from the reference pool, as `run_unit` takes them.
  2. The unit's new photos: every new pool id whose label is in classes, in id order; D and T =
     `contrastloop.split(those ids, new["names"])`.
  3. From here exactly as `pairtable.run_unit` does after its photos and peers, on the new pool (its V and labels,
     the same seeds `zlib.crc32(name) + pairtable.NULL_D` and `+ pairtable.NULL_T`): real and null tables on D
     with `pairtable.table_rows`; the arms `rule_real` and `rule_null` (`pairtable.rule_rows` of each table) and
     `every` (every row of the real table); each pick by `pairtable.pick` on T with the real and the null answer
     rows; `pairtable.tally` over these three arms.
  Returns the stats dict with keys `unit`, `layer`, `chan`, `m` (channels), `classes`, `class_names` (from
  torchvision's GoogLeNet categories, as run_unit), `peers` (as run_unit's peer records), `D`, `T`, `tables`,
  `picks` (the three arms), `tally`, `n_unanswered_D` and `n_unanswered_T` (photos whose row is all None).
- `main()`: arguments `--units` (one or more JSON files of unit strings), `--pool`, `--answers`, `--out` (a
  directory). Loads questions with `pairtable.load_questions()`. Writes `<out>/<unit>/stats.json` per unit and
  `<out>/report.txt` with, per unit, the lines `pairtable.section` writes for the head, classes, peers, real table,
  the three arms' picks and their tallies (no reading lines; `section` itself expects all nine arms, so write these
  lines in its format rather than calling it). Loads one layer's values at a time.
- `selftest()` (`--selftest`): checks `unit_new` on a small synthetic pair of pools (a few channels, classes and
  photos, answers made by hand) against the same steps done by hand with the pairtable functions; prints
  "selftest ok".

## Constraints
- Reuse pairtable, proploop and contrastloop functions as named; no new statistics.
- Match the repo's style: module docstring with usage, small functions, plain asserts in the selftest.
