## Pool, answers, candidates
- Values: `V` per layer from `proploop.layer_values([layer], None, np.arange(proploop.N_PHOTOS))` (all photos x
  channels); labels from `proploop.ACTS` (`classes`), names from `proploop.NAMES`, questions from
  `pairtable.load_questions()`.
- Answers: `r31_validate.merge` of `r31_validate.load_claude` on `runs/pairtable/answers.json`,
  `runs/r30b/pairtable/answers.json`, `runs/r30a/pairtable/answers.json` (that order). A photo without an answer has
  a row of None for every question.
- The sub-pool P: the photos whose label is in `authors/round31_classes.json`, in photo id order. Every table and
  check below uses P only (its values, labels, names); ids inside a candidate are sub-pool indices.
- Layers: those of the units in `authors/round31_units.json` (strings `layer:channel`; parse with
  `proploop.parse_unit`).
- Groups per layer L with m channels:
  - `selected`: the units of that file in L. Name: the unit string.
  - `channel`: `draw(L, m, exclude, k)`'s channels (exclude = the selected channels of L). Name `f"{L}:{c}"`.
  - `direction`: `draw`'s columns of W. Name `f"{L}:dir{j:02d}"` (j = 0..k-1).

## Interface
- `draw(layer, m, exclude, k)` -> `(chans, W)`: `rng = np.random.default_rng(zlib.crc32(("r32:" + layer).encode()))`;
  `chans = sorted(rng.choice(<the channels 0..m-1 not in exclude, ascending>, size=k, replace=False))` (as ints);
  then, from the same rng, `W = rng.standard_normal((m, k))` with each column divided by its L2 norm.
- `project_max(F, W)` -> for F (n x m x h x w) and W (m x k): for each photo and column, the maximum over the h x w
  positions of `sum_c W[c, j] * F[:, c]`; float32, n x k. Accepts numpy arrays (a torch version inside the forward
  pass is fine as long as it gives the same numbers).
- `direction_values(model, layer, W, ids, device)` -> len(ids) x k float32: one forward pass per batch of
  `proploop.BATCH` photos (input `bundlemax.net_input`), hooked on the module of that name as
  `proploop.forward_values` hooks it (stopped after it), each batch's output projected by `project_max`. Cached in
  `<out>/dirvalues_<layer>.npy` (computed over `np.arange(proploop.N_PHOTOS)`; loaded when present with the right
  shape and dtype). The model: `bundlemax._bank().load_model(proploop.WEIGHTS)`, gradients off, eval mode.
- `joint_d(R, ans_rows, questions, q, a, b, min_n=pairtable.MIN_N)` -> a length-m vector, or None. R: n x m centred
  values; ans_rows: one answer row per photo. Design X (n x p): an intercept column of ones, then for every question
  q' and every answer index v of q' (in order) with at least min_n photos answering v, a column that is 1 for those
  photos and 0 otherwise. None when answer a or b of question q has no column. Else
  `beta = np.linalg.lstsq(X, R, rcond=None)[0]` and the result is `(beta[col(q, a)] - beta[col(q, b)])` divided,
  per column of R, by that column's sd over the n photos (ddof 1), 0 where the sd is 0.
- `excess_kurtosis(x)` -> `mean((x - mean)^4) / var^2 - 3` with the population variance (ddof 0); None when the
  variance is 0.
- `candidate(name, group, Vc, c, labels, names, row_of, questions, x_all, joint=False)` -> the candidate's record.
  Vc: the sub-pool values (photos x columns) in which column c is the candidate (for a direction: the layer's m
  channels with its column appended, c = m); labels, names, row_of over the sub-pool; x_all: the candidate's values
  over all N photos. As `pairtable.run_unit` does without readings:
  1. `classes, _, D, T = pairtable.unit_photos({"chan": c}, Vc, labels, names)`; peers =
     `contrastloop.peers(Vc, labels, c)` (channels only).
  2. seed `zlib.crc32(name.encode())`; real and null answer rows on D and T with `proploop.permute_within` and the
     seeds `seed + pairtable.NULL_D`, `seed + pairtable.NULL_T`; `R_D`, `R_T` by `proploop.centre`; the real table
     `pairtable.table_rows(R_D, rows_D, questions, c, peers)`.
  3. Every row of the real table picked with `pairtable.pick(r, questions, R_T, rows_T, null_T, c, peers)`.
  4. When joint: per row, `joint_d` on (R_D, rows_D) and on (R_T, rows_T) with the row's q, first, second; the
     unit's value `jd[c]` and `proploop.percentile(jd, c)` on each; the T outcome `contrastloop.outcome(jd[c], pct)`,
     or `'too few'` when joint_d is None.
  Record: `group`, `unit` (name), `layer`, `n_D`, `n_T`, `classes`, `peers` (channel list), `kurt_all`
  (excess_kurtosis of x_all), `kurt_pool` (of column c of Vc), `rows` (per picked row: `row`, `q`, `first`,
  `second`, `D_d`, `D_pct`, `T_d`, `T_pct`, `T_outcome`, `T_distinctive`, `T_null_outcome`, and when joint
  `jD_d`, `jD_pct`, `jT_d`, `jT_pct`, `jT_outcome`), `n_rows`, `n_D95` (rows with D_pct >= 95), `n_held` (T_outcome
  'held'), `n_distinctive`, `n_held_null` (T_null_outcome 'held').
- `summary(records)` -> per (layer, group) and per group over all layers: `n`, `mean_held`, `share_held` (records
  with n_held >= 1), `mean_distinctive`, `share_distinctive`, `mean_held_null`, `share_held_null`, `mean_D95`,
  `median_kurt_all`.
- `main()`: `--k` (default 40), `--out` (default `runs/r32`), `--device` (default `cuda` when available, else
  `cpu`), `--selftest`. One layer at a time: values, draw, direction values, then every candidate (selected with
  joint=True). Writes `<out>/basis.json` (`{"records": [...], "summary": ...}`, via a temporary file and rename) and
  `<out>/report.txt`: the summary as a table, then one line per selected unit (n_held, n_distinctive, and each held
  row as `q first>second T_pct joint jT_outcome`).
- `selftest()`: without the pool or a model: `draw` (determinism, exclusion, unit norms); `project_max` against a
  loop on a small random F; `joint_d` on a hand-made case where answer x of question 0 always comes with answer y of
  question 1 for most photos and the values depend only on question 1 (question 0's plain contrast large, its joint
  contrast near 0), and on a case where a needed answer has no column (None); `excess_kurtosis` on a known vector;
  `candidate` on a tiny synthetic pool (a few classes, channels and photos, hand-made answers) against the same
  steps done by hand with the pairtable functions. Prints "selftest ok".

