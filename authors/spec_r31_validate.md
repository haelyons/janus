# Spec: tools/r31_validate.py (agreement of two answer sources on the standing questions, and of the tables they give)

Write one new file, `tools/r31_validate.py`. Read only this spec and `tools/pairtable.py`, `tools/proploop.py`,
`tools/contrastloop.py`. Do not open any other file, and do not run anything. No Claude calls, no GPU.

## Purpose
The same photos can be answered by two sources (two runs of one answerer, or two answerers). Measure how often the
two give the same answer to each question, and how much the pairwise tables of `tools/pairtable.py` change when one
source's answers replace the other's.

## Answer sources
An answer source is a dict `{photo id (int): answer row}`, a row being a list with one int or None per question.
- `load_claude(path)`: from a pairtable `answers.json` (a dict of batch records with keys `photos`, `answers`,
  `failed`): every photo of every batch whose `failed` is false, with its row.
- `load_vlm(path)`: from a `tools/vlm_answer.py` output (`{id (str): {"answers", "valid", ...}}`): every record with
  `valid` true, id converted to int.
- `merge(*sources)`: one source; for a photo in several, the last source given wins.

## Split
`half(name)` = `zlib.crc32(("r31:" + name).encode()) % 2`; 0 is the calibration half, 1 the confirmation half.
Names are `json.loads(proploop.NAMES.read_text())`, indexed by photo id.

## Interface
- `question_agreement(a, b, ids, nq)` -> a list of nq dicts, one per question: `n` (photos of ids in both sources
  with a non-None answer to that question in both), `agree` (of those, equal answers), `rate` (agree / n, None when
  n is 0), `none_a` and `none_b` (photos of ids in both sources with None for that question in a, in b).
- `match_rows(rows_a, rows_b)` -> a list of `(row_a, row_b)` pairs for the rows of two `pairtable.table_rows` results
  that compare the same question and the same two answers (in either order); `row_b`'s `d` is negated in the pair
  (a copy) when its `first` is `row_a`'s `second`. Rows in only one are left out. Order: row_a's order.
- `unit_compare(u, V, labels, names, a, b, questions)`: for one unit (`proploop.parse_unit` record) on the pool
  (`V` its layer's values, all photos x channels; `labels`): its classes, D and T as `pairtable.unit_photos`; its
  peers as `contrastloop.peers(V, labels, c)`; then D and T are kept to the photos answered in both a and b
  (absent photos dropped). With `R_D = proploop.centre(V[D], labels[D])` and `R_T` likewise: the real tables of
  `pairtable.table_rows(R_D, rows, questions, c, peers)` with a's rows and with b's rows; `pairs` =
  `match_rows(table_a, table_b)`; for every row of table_a, `pairtable.t_check` on R_T with a's and with b's answers
  to its question (same first, second). Returns `{"unit", "nD", "nT", "pairs": [[d_a, d_b, pct_a, pct_b], ...],
  "t_outcomes": [[outcome_a, outcome_b], ...]}`.
- `summarise(units)` over unit_compare results: Pearson correlation of d_a with d_b over all pairs of all units
  (None when fewer than 3 pairs or a constant side), the same for pct; the share of pairs with both pct >= 95 or
  both below 95; the share of rows with equal T outcome, and of rows whose outcome is 'held' with a and with b.
- `main()`: arguments `--a` and `--b`, each one or more files given as `kind:path` with kind `claude` or `vlm`
  (merged in the order given), `--units` (one or more JSON files of unit strings), `--half` (`0`, `1` or `all`,
  default `all`: which photos count, by `half`), `--out` (a JSON path). Reports, printed and in `--out`: the photo
  counts of a, b and both (within the half); question_agreement on the photos in both and in the half; and for the
  table part, unit_compare for each unit with D and T further kept to the half, and summarise over them. The pool:
  `V` per layer from `proploop.layer_values(layers, None, np.arange(proploop.N_PHOTOS))`, labels from
  `proploop.ACTS` (`classes`), questions from `pairtable.load_questions()`.
- `selftest()` (`--selftest`): checks question_agreement, match_rows and summarise on small hand-made inputs, without
  the pool; prints "selftest ok".

## Constraints
- Use pairtable, proploop and contrastloop functions where this spec names them; no other statistics.
- Match the repo's style: module docstring with usage, small functions, plain asserts in the selftest.
