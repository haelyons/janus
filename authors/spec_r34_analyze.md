# Spec: tools/r34_analyze.py (which units and directions carry one meaning across many surface forms)

Write one new file, `tools/r34_analyze.py`. Read only this spec. Do not open any other file, and do not run anything.
numpy and scipy only; CPU.

## Data
- `sentences.json`: a list of records with `sid`, `form` (int), `meaning` (string), `text`, and other fields. Every
  form appears once with each meaning (a full form x meaning grid). One meaning is the target (`--target`, default
  `postponed`); the others are contrasts.
- `record.npz` (row i = sentence i): `hidden_suffix` [n, L, H], `pre_suffix`, `mid_suffix`, `post_suffix` [n, L, d]
  (the residual stream before layer l, after its attention, after the layer), and the same for `_text`. Readout
  `suffix` is primary; `--readout text` switches every array.
- Split of forms: `half(form) = zlib.crc32(f"r34:{form}".encode()) % 2`; 0 = selection half S, 1 = test half T.

## Interface
- `unit_scores(X, y)` -> length-m vector: X [n, m] (rows of S), y boolean (target). (mean over target rows - mean
  over the other rows) / sd over all rows (ddof 1); 0 where the sd is 0.
- `top_units(scores, k, sign=0)` -> indices of the k largest |scores| (sign 0), or largest scores (sign +1), ties by
  lower index.
- `centroid_accuracy(X_S, y_S, X_T, y_T, units)` -> balanced accuracy on T of nearest-centroid classification of
  target vs rest using only the given columns, each standardised by its mean and sd (ddof 1) over S (sd 0 -> 1);
  centroids are the S means of target and rest; Euclidean distance.
- `direction_auc(X_S, y_S, X_T, y_T)` -> ROC AUC on T of the projection onto d = mean_S(target) - mean_S(rest)
  (scipy.stats.mannwhitneyu U / (n_pos n_neg), target scores against rest).
- `top_share(d, frac=0.01)` -> share of sum(d^2) carried by the ceil(frac x len(d)) entries of largest |d|.
- `gaussian_share(m, frac=0.01, draws=20, seed=0)` -> the mean top_share of `draws` standard normal vectors of
  length m (the share for a uniformly random direction, i.e. d after a random rotation).
- `eta2(x, meaning, form)` -> (eta2_meaning, eta2_form): for one unit's values over all sentences, the share of the
  total sum of squares explained by meaning means and by form means (one-way each, over the full grid).
- `write_contributions(pre, mid, post, y, D)` -> list over layers of (attn, mlp): with attn_write = mid - pre and
  mlp_write = post - mid, (mean over target rows - mean over rest rows) of each write, dotted with the unit vector
  D / |D|. Also returns the embedding term the same way for pre[:, 0]; the three parts sum to the same difference of
  post[:, L-1] (an identity; assert it within 1e-3 relative).
- `analyze(sent, rec, target, ks=(1, 10, 100))` -> dict:
  - per layer l: for the MLP hidden space: `acc_k` (centroid_accuracy with top_units(unit_scores on S, k)) for each
    k; `auc_dir_hidden` and `share_top1pct` (top_share of d in the hidden space, d from S) and `share_gauss`; for the
    residual stream `post`: `auc_dir_resid`; `eta2_top10`: mean (eta2_meaning, eta2_form) over the top-10 units.
  - `writes`: write_contributions on T rows with D = mean_S(target) - mean_S(rest) of `post[:, L-1]`; `attn_share` =
    sum of attn terms / (sum of attn terms + sum of mlp terms).
  - `best_layer`: the layer with the highest `acc_10` (ties: lower layer); `selected`: the 8 units of
    top_units(unit_scores on S at best_layer, 8, sign=+1), each with its score.
  - `factors`: for each record field among `stance`, `speech_act`, `voice` present: per layer, acc_10 for that
    field's most common value against the rest, computed the same way (selection on S, test on T) over all meanings.
- `main()`: `--record DIR --target postponed --readout suffix|text --out FILE`. Writes the dict as JSON and a short
  text table (per layer: acc_1, acc_10, acc_100, auc_dir_hidden, auc_dir_resid, share_top1pct, share_gauss;
  then attn_share, best_layer, selected).
- `selftest()` (`--selftest`): synthetic data where one hidden unit carries the target perfectly and the residual
  direction is planted: checks every function above against hand values; prints "selftest ok".

## Constraints
- Match the repo's style: module docstring with usage, small functions, plain asserts. No other statistics.
