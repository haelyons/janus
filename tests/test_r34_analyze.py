"""Black-box tests for tools/r34_analyze.py, written from spec.md only.

No implementation, model, GPU, files on disk, or network is needed. Numerical
oracles below use hand values or NumPy directly, never another r34 function.
The full-grid fixture has separate CRC32-selected forms, a transferable hidden
unit, misleading selection-only units, and a planted residual direction.

The spec does not fix result container layouts or a Gaussian RNG algorithm.
Small readers accept common tuple/dict layouts without requiring extra keys;
Gaussian checks use exact identities and a two-dimensional analytic mean.
Centroid ties, zero-energy shares or normalized write directions, constant eta2
inputs, and k > width are avoided because their behavior is not specified.
"""

import sys
import zlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import r34_analyze  # noqa: E402


# Independent numerical oracles.

def _scores(X, y):
    sd = np.std(X, axis=0, ddof=1)
    difference = np.mean(X[y], axis=0) - np.mean(X[~y], axis=0)
    return np.divide(difference, sd, out=np.zeros_like(difference), where=sd != 0)


def _order(scores, k, positive=False):
    strength = scores if positive else np.abs(scores)
    return np.array(sorted(range(len(scores)), key=lambda j: (-strength[j], j))[:k])


def _accuracy(X_S, y_S, X_T, y_T, units):
    train, test = X_S[:, units], X_T[:, units]
    mean = np.mean(train, axis=0)
    sd = np.std(train, axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    train, test = (train - mean) / sd, (test - mean) / sd
    yes, no = np.mean(train[y_S], axis=0), np.mean(train[~y_S], axis=0)
    yes_dist = np.sum((test - yes) ** 2, axis=1)
    no_dist = np.sum((test - no) ** 2, axis=1)
    # Do not silently choose a policy that the spec leaves unspecified.
    assert np.all(np.abs(yes_dist - no_dist) > 1e-10), "fixture has a centroid tie"
    prediction = yes_dist < no_dist
    return (np.mean(prediction[y_T]) + np.mean(~prediction[~y_T])) / 2


def _auc(X_S, y_S, X_T, y_T):
    direction = np.mean(X_S[y_S], axis=0) - np.mean(X_S[~y_S], axis=0)
    projection = X_T @ direction
    comparisons = projection[y_T, None] - projection[None, ~y_T]
    return np.mean((comparisons > 0) + 0.5 * (comparisons == 0))


def _share(direction, frac=0.01):
    energy = np.square(direction)
    count = int(np.ceil(frac * len(direction)))
    return np.sum(np.sort(energy)[-count:]) / np.sum(energy)


def _eta(x, meaning, form):
    total = np.sum((x - np.mean(x)) ** 2)
    values = []
    for labels in (meaning, form):
        explained = 0.0
        for label in np.unique(labels):
            group = x[labels == label]
            explained += len(group) * (np.mean(group) - np.mean(x)) ** 2
        values.append(explained / total)
    return np.asarray(values)


def _writes(pre, mid, post, y, direction):
    axis = direction / np.linalg.norm(direction)

    def project(values):
        return (np.mean(values[y], axis=0) - np.mean(values[~y], axis=0)) @ axis

    pairs = np.column_stack((project(mid - pre), project(post - mid)))
    return pairs, float(project(pre[:, 0]))


# Container readers: only the named metrics and their values are contractual.

def _walk(value):
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk(child)


def _pair_rows(value, layers):
    if isinstance(value, Mapping):
        if "attn" in value and "mlp" in value:
            arr = np.column_stack((value["attn"], value["mlp"]))
            return arr if arr.shape == (layers, 2) else None
        if len(value) != layers:
            return None
        value = list(value.values())
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != layers:
        return None
    if all(isinstance(row, Mapping) and "attn" in row and "mlp" in row for row in value):
        value = [(row["attn"], row["mlp"]) for row in value]
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return arr if arr.shape == (layers, 2) else None


def _read_writes(value, layers):
    pairs = next((rows for node in _walk(value)
                  if (rows := _pair_rows(node, layers)) is not None), None)
    assert pairs is not None, "missing per-layer (attn, mlp) contributions"
    embedding = [node[key] for node in _walk(value) if isinstance(node, Mapping)
                 for key in ("embedding", "embed", "emb") if key in node]
    if not embedding and isinstance(value, (tuple, list)):
        embedding = [item for item in value if np.isscalar(item)]
    assert len(embedding) == 1, "missing or ambiguous embedding contribution"
    return pairs, float(embedding[0])


def _layer_rows(result, layers):
    rows = [node for node in _walk(result)
            if isinstance(node, Mapping) and "auc_dir_hidden" in node]
    if len(rows) == 1 and np.ndim(rows[0]["auc_dir_hidden"]) == 1:
        columns = rows[0]
        rows = [{key: (value[i] if not isinstance(value, Mapping)
                       else {k: v[i] for k, v in value.items()})
                 for key, value in columns.items()
                 if isinstance(value, (list, tuple, np.ndarray, Mapping))}
                for i in range(layers)]
    assert len(rows) == layers, "missing per-layer analysis metrics"
    if all("layer" in row for row in rows):
        rows = sorted(rows, key=lambda row: int(row["layer"]))
    return rows


def _acc_value(row, k, ks):
    if f"acc_{k}" in row:
        return row[f"acc_{k}"]
    values = row["acc_k"]
    if isinstance(values, Mapping):
        return values[k] if k in values else values[str(k)]
    return values[list(ks).index(k)]


def _selected_pairs(value):
    if isinstance(value, Mapping):
        if "units" in value and "scores" in value:
            value = list(zip(value["units"], value["scores"]))
        else:
            value = list(value.items())
    pairs = []
    for row in value:
        if isinstance(row, Mapping):
            key = next(k for k in ("unit", "index", "idx") if k in row)
            row = (row[key], row["score"])
        unit, score = row
        pairs.append((int(unit), float(score)))
    return pairs


def _factor_accuracies(value, layers):
    rows = [node for node in _walk(value)
            if isinstance(node, Mapping) and "acc_10" in node]
    if rows:
        if all("layer" in row for row in rows):
            rows = sorted(rows, key=lambda row: int(row["layer"]))
        return np.asarray([row["acc_10"] for row in rows], dtype=float).reshape(-1)
    for node in _walk(value):
        if isinstance(node, Mapping) and len(node) == layers:
            if set(map(str, node)) == set(map(str, range(layers))):
                return np.asarray([node[i] if i in node else node[str(i)]
                                   for i in range(layers)], dtype=float)
        if isinstance(node, (list, tuple, np.ndarray)):
            if len(node) == layers and all(np.isscalar(x) and not isinstance(x, str) for x in node):
                return np.asarray(node, dtype=float)
    raise AssertionError("missing per-layer factor acc_10")


# Individual numerical functions.

@pytest.mark.parametrize("y", ([True, True, False, False], [True, False, False, False]))
def test_unit_scores_signed_difference_sample_sd_and_constant(y):
    X = np.array([[1, 10, 7, 0], [3, 8, 7, 4], [5, 6, 7, 8], [7, 4, 7, 12]], dtype=float)
    # Both labelings give differences (-4, 4, 0, -8); sample SDs are
    # sqrt(20/3), sqrt(20/3), 0, sqrt(80/3).
    want = np.array([-4, 4, 0, -4], dtype=float) / np.sqrt(20 / 3)
    got = np.asarray(r34_analyze.unit_scores(X, np.asarray(y)))
    assert got.shape == (4,)
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("sign, k, want", [
    (0, 1, [6]),
    (0, 5, [6, 0, 1, 5, 2]),
    (0, 8, [6, 0, 1, 5, 2, 3, 4, 7]),
    (1, 5, [1, 5, 2, 4, 7]),
    (1, 8, [1, 5, 2, 4, 7, 3, 0, 6]),
])
def test_top_units_orders_signed_or_absolute_scores_and_breaks_ties(sign, k, want):
    scores = np.array([-7, 7, 3, -3, 0, 7, -9, 0], dtype=float)
    np.testing.assert_array_equal(r34_analyze.top_units(scores, k, sign=sign), want)
    if sign == 0:
        np.testing.assert_array_equal(r34_analyze.top_units(scores, k), want)


@pytest.mark.parametrize("units", ([0], [0, 1], [2, 1, 0]))
def test_centroid_accuracy_uses_selection_scaling_subset_and_balanced_accuracy(units):
    X_S = np.array([[0, -100, 9, -1000], [0, 100, 9, -1000],
                    [2, -98, 9, 1000], [2, 102, 9, 1000]], dtype=float)
    y_S = np.array([False, False, True, True])
    X_T = np.array([[1.5, -1000, 10, -10000], [0.2, 1000, -8, 10000],
                    [0.5, 0, 11, -10000], [0.25, 1, -7, 10000],
                    [0, 2, 9, 10000], [-1000, -2, -9, 10000]], dtype=float)
    y_T = np.array([True, False, True, False, False, False])
    # Predictions: target, rest, rest, rest, rest, rest. Sensitivity 1/2,
    # specificity 1, balanced accuracy 3/4 (ordinary accuracy would be 5/6).
    assert _accuracy(X_S, y_S, X_T, y_T, units) == pytest.approx(0.75)
    assert r34_analyze.centroid_accuracy(X_S, y_S, X_T, y_T, units) == pytest.approx(0.75)


@pytest.mark.parametrize("reverse, want", [(False, 7 / 12), (True, 5 / 12)])
def test_direction_auc_selection_direction_orientation_and_half_credit_ties(reverse, want):
    X_S = np.array([[3, 0, 10], [5, 0, 10], [0, 0, 10], [0, 4, 10]], dtype=float)
    y_S = np.array([True, True, False, False])
    X_T = np.array([[0, -1, 100], [1, 0, -100], [0, -0.5, 10],
                    [1, 1, -20], [2, 1.5, 0]], dtype=float)
    y_T = np.array([True, True, False, False, False])
    # d = (4, -2, 0); target projections 2, 4, contrast projections 1, 2, 5.
    # U = 1 + 1/2 + 0 + 1 + 1 + 0 = 3.5 out of six comparisons.
    labels = ~y_S if reverse else y_S
    assert r34_analyze.direction_auc(X_S, labels, X_T, y_T) == pytest.approx(want)


def test_direction_auc_zero_difference_scores_every_pair_as_tied():
    X_S = np.array([[-1, 0], [1, 0], [0, -2], [0, 2]], dtype=float)
    y_S = np.array([True, True, False, False])
    X_T = np.array([[9, -1], [-3, 7], [4, 2]], dtype=float)
    y_T = np.array([True, False, False])
    assert r34_analyze.direction_auc(X_S, y_S, X_T, y_T) == pytest.approx(0.5)


@pytest.mark.parametrize("frac, want", [(0.25, 144 / 169), (0.26, 160 / 169),
                                        (0.51, 1.0), (1.0, 1.0)])
def test_top_share_squared_energy_ceiling_and_scale_invariance(frac, want):
    direction = np.array([3, -4, 0, 12], dtype=float)
    assert r34_analyze.top_share(direction, frac=frac) == pytest.approx(want)
    assert r34_analyze.top_share(-7 * direction, frac=frac) == pytest.approx(want)


def test_top_share_default_one_percent_rounds_up():
    direction = np.arange(1, 102, dtype=float) * (-1.0) ** np.arange(101)
    want = (101 ** 2 + 100 ** 2) / (101 * 102 * 203 / 6)
    assert r34_analyze.top_share(direction) == pytest.approx(want)


def test_gaussian_share_exact_one_dimension_and_full_fraction():
    assert r34_analyze.gaussian_share(1) == pytest.approx(1.0)
    assert r34_analyze.gaussian_share(17, frac=1.0, draws=7, seed=91) == pytest.approx(1.0)


def test_gaussian_share_seed_repeatability_and_analytic_two_dimensional_mean():
    kwargs = dict(frac=0.5, draws=20000, seed=34)
    got = r34_analyze.gaussian_share(2, **kwargs)
    assert got == r34_analyze.gaussian_share(2, **kwargs)
    # A uniform direction in 2D has max(cos(theta)^2, sin(theta)^2)
    # = (1 + |cos(2 theta)|)/2, with mean 1/2 + 1/pi.
    # Broad Monte Carlo tolerance avoids pinning a particular NumPy RNG.
    assert got == pytest.approx(0.5 + 1 / np.pi, abs=0.015)


@pytest.mark.parametrize("scale, offset", [(1.0, 0.0), (-3.0, 17.0)])
def test_eta2_full_grid_meaning_form_and_unexplained_interaction(scale, offset):
    x = np.array([6, 11, 10, 10, 9, 14], dtype=float)
    meaning = np.tile(["postponed", "accepted", "canceled"], 2)
    form = np.repeat([17, -3], 3)
    order = np.array([4, 0, 5, 2, 3, 1])
    # Grand mean 10; meaning SS 16, form SS 6, interaction SS 12, total SS 34.
    got = r34_analyze.eta2((scale * x + offset)[order], meaning[order], form[order])
    np.testing.assert_allclose(got, [8 / 17, 3 / 17], rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("source, want", [("meaning", [1.0, 0.0]), ("form", [0.0, 1.0])])
def test_eta2_pure_crossed_effects(source, want):
    meaning = np.tile(["z", "a", "m"], 2)
    form = np.repeat([5, 9], 3)
    x = np.tile([-2.0, 0.0, 2.0], 2) if source == "meaning" else np.repeat([-1.0, 1.0], 3)
    np.testing.assert_allclose(r34_analyze.eta2(x, meaning, form), want, atol=1e-10)


def _write_fixture():
    y = np.array([True, False, False, True, False, False])
    nuisance = np.array([-1, -1, 0, 1, 0, 1], dtype=float)
    attn = np.array([[2, -1, 7], [-1, 3, 2], [4, 0, -1]], dtype=float)
    mlp = np.array([[-3, 1, -2], [2, -2, 4], [-1, 4, 3]], dtype=float)
    pre, mid, post = (np.empty((6, 3, 3), dtype=float) for _ in range(3))
    state = 10 + y[:, None] * [1, 2, 9] + nuisance[:, None] * [0, 0, 3]
    for layer in range(3):
        pre[:, layer] = state
        mid[:, layer] = state + y[:, None] * attn[layer] + nuisance[:, None] * [7, -3, 0]
        state = mid[:, layer] + y[:, None] * mlp[layer] + nuisance[:, None] * [-2, 1, 4]
        post[:, layer] = state
    return pre, mid, post, y


@pytest.mark.parametrize("multiple", [1.0, 10.0, -1.0])
def test_write_contributions_normalized_projection_embedding_and_identity(multiple):
    pre, mid, post, y = _write_fixture()
    direction = multiple * np.array([3.0, 4.0, 0.0])
    pairs, embedding = _read_writes(r34_analyze.write_contributions(pre, mid, post, y, direction), 3)
    sign = np.sign(multiple)
    want = sign * np.array([[0.4, -1.0], [1.8, -0.4], [2.4, 2.6]])
    np.testing.assert_allclose(pairs, want, rtol=1e-7, atol=1e-9)
    assert embedding == pytest.approx(sign * 2.2)
    # Final target-minus-rest vector is (4, 7, 22), whose projection is 8.
    final = (np.mean(post[y, -1], axis=0) - np.mean(post[~y, -1], axis=0))
    projected = final @ (direction / np.linalg.norm(direction))
    assert projected == pytest.approx(sign * 8.0)
    assert embedding + np.sum(pairs) == pytest.approx(projected, rel=1e-3)


def test_write_contributions_asserts_identity_for_inconsistent_layer_boundary():
    pre, mid, post, y = _write_fixture()
    pre[:, 1] += y[:, None] * [30.0, 40.0, 0.0]
    with pytest.raises(AssertionError):
        r34_analyze.write_contributions(pre, mid, post, y, np.array([3.0, 4.0, 0.0]))


# A shuffled full form x meaning grid, with traps for selection/test leakage.

def _grid():
    groups = [[], []]
    for form in range(1000):
        half = zlib.crc32(f"r34:{form}".encode()) % 2
        if len(groups[half]) < 6:
            groups[half].append(form)
        if all(len(group) == 6 for group in groups):
            break
    assert all(len(group) == 6 for group in groups)
    meanings = ("postponed", "accepted", "canceled")
    form_effects = (-8, -4, -2, 2, 4, 8)
    sent, halves, effect = [], [], []
    for half, forms in enumerate(groups):
        for position, form in enumerate(forms):
            for meaning in meanings:
                sent.append({"sid": f"sentence-{len(sent)}", "form": form,
                             "meaning": meaning, "text": f"Form {form}: {meaning}.",
                             "stance": "plain" if position < 4 else "marked",
                             "speech_act": "statement" if position >= 2 else "question",
                             "voice": "active" if position < 4 else "passive"})
                halves.append(half)
                effect.append(form_effects[position])

    half = np.asarray(halves)
    u = np.asarray(effect, dtype=float)
    t = np.asarray([row["meaning"] == "postponed" for row in sent], dtype=float)
    contrast = np.asarray([0 if row["meaning"] == "postponed" else
                           (-1 if row["meaning"] == "accepted" else 1) for row in sent])
    n, layers, width = len(sent), 3, 100
    hidden = np.empty((n, layers, width), dtype=float)
    hidden[:] = u[:, None, None] * (1 + np.arange(width) % 7)[None, None, :]

    # Layer 0: nine of ten perfect selection units reverse on test forms.
    for j in range(10):
        sign = -1 if j % 2 == 0 else 1
        hidden[:, 0, j] = sign * (2 * t - 1) * np.where(half == 0, 1, -1)
    hidden[:, 0, 7] = 2 * t - 1

    # Layers 1 and 2: unit 7 transfers perfectly; unit 8 is misleading on T.
    hidden[:, 1, 0] = -4 * t + u / 8
    hidden[:, 1, 1] = 2 * t + u / 4
    hidden[:, 1, 2] = t + u / 2
    hidden[:, 1, 3] = contrast + u / 4
    hidden[:, 1, 4] = -2 * t + u / 2
    hidden[:, 1, 5] = t / 2 + u
    hidden[:, 1, 7] = 4 * t
    hidden[:, 1, 8] = np.where(half == 0, 3, -1) * t + u / 8
    hidden[:, 1, 9] = t + u
    hidden[:, 2] = hidden[:, 1] * np.arange(1, width + 1) + 32

    # All nuisance directions are orthogonal to D. Residual streams telescope.
    D = np.array([3.0, 4.0, 0.0, 0.0])
    orth_a, orth_b = np.array([-4, 3, 1, 0]), np.array([4, -3, 0, 1])
    pre, mid, post = (np.empty((n, layers, 4), dtype=float) for _ in range(3))
    embed = np.where(half == 0, 0.25, 0.5)
    attn = np.array([[-0.125, 0.25, 0.375], [0.5, -0.25, 0.75]])
    mlp = np.array([[0.125, -0.125, 0.25], [0.25, 0.5, -0.25]])
    state = (t * embed)[:, None] * D + u[:, None] * orth_a + contrast[:, None] * orth_b
    for layer in range(layers):
        pre[:, layer] = state
        mid[:, layer] = (state + (t * attn[half, layer])[:, None] * D
                         + u[:, None] * orth_a * (layer + 1) / 16)
        state = (mid[:, layer] + (t * mlp[half, layer])[:, None] * D
                 + contrast[:, None] * orth_b * (layer + 1) / 8)
        post[:, layer] = state

    # Exercise sentence/record alignment without relying on grouped row order.
    order = np.random.default_rng(3401).permutation(n)
    sent = [sent[i] for i in order]
    rec = {"hidden_suffix": hidden[order], "pre_suffix": pre[order],
           "mid_suffix": mid[order], "post_suffix": post[order]}
    # analyze has no readout argument; accept both natural input conventions.
    rec.update({key.removesuffix("_suffix"): value for key, value in list(rec.items())})
    return sent, rec


def _grid_reference(sent, rec, target, ks):
    meaning = np.asarray([row["meaning"] for row in sent])
    form = np.asarray([row["form"] for row in sent])
    S = np.asarray([zlib.crc32(f"r34:{f}".encode()) % 2 == 0 for f in form])
    T, y = ~S, meaning == target
    hidden, post = rec["hidden_suffix"], rec["post_suffix"]
    rows, scores = [], []
    for layer in range(hidden.shape[1]):
        X = hidden[:, layer]
        score = _scores(X[S], y[S])
        units = _order(score, 10)
        direction = np.mean(X[S & y], axis=0) - np.mean(X[S & ~y], axis=0)
        rows.append({**{f"acc_{k}": _accuracy(X[S], y[S], X[T], y[T], _order(score, k)) for k in ks},
                     "auc_dir_hidden": _auc(X[S], y[S], X[T], y[T]),
                     "auc_dir_resid": _auc(post[S, layer], y[S], post[T, layer], y[T]),
                     "share_top1pct": _share(direction),
                     "eta2_top10": np.mean([_eta(X[:, j], meaning, form) for j in units], axis=0)})
        scores.append(score)
    best = max(range(len(rows)), key=lambda layer: (rows[layer]["acc_10"], -layer))
    selected = _order(scores[best], 8, positive=True)
    direction = np.mean(post[S & y, -1], axis=0) - np.mean(post[S & ~y, -1], axis=0)
    pairs, embedding = _writes(rec["pre_suffix"][T], rec["mid_suffix"][T], post[T], y[T], direction)
    factors = {}
    for field in ("stance", "speech_act", "voice"):
        labels = np.asarray([row[field] for row in sent])
        values, counts = np.unique(labels, return_counts=True)
        assert np.sum(counts == np.max(counts)) == 1  # no modal-value tie
        factor_y = labels == values[np.argmax(counts)]
        factors[field] = [_accuracy(hidden[S, layer], factor_y[S], hidden[T, layer], factor_y[T],
                                    _order(_scores(hidden[S, layer], factor_y[S]), 10))
                          for layer in range(hidden.shape[1])]
    return {"layers": rows, "best_layer": best,
            "selected": [(int(j), float(scores[best][j])) for j in selected],
            "pairs": pairs, "embedding": embedding,
            "attn_share": np.sum(pairs[:, 0]) / np.sum(pairs), "factors": factors}


def test_grid_fixture_has_planted_unit_direction_and_selection_traps():
    sent, rec = _grid()
    want = _grid_reference(sent, rec, "postponed", (1, 10, 100))
    assert len({row["form"] for row in sent}) == 12
    assert len({(row["form"], row["meaning"]) for row in sent}) == len(sent) == 36
    np.testing.assert_allclose([row["acc_1"] for row in want["layers"]], [0, 1, 1])
    np.testing.assert_allclose([row["acc_10"] for row in want["layers"]], [0, 1, 1])
    np.testing.assert_allclose([row["auc_dir_resid"] for row in want["layers"]], [1, 1, 1])
    assert want["best_layer"] == 1  # layers 1 and 2 tie
    assert want["selected"][0][0] == 7
    assert 8 in [j for j, _ in want["selected"]]
    assert 0 not in [j for j, _ in want["selected"]]  # strong negative score
    all_scores = _scores(rec["hidden_suffix"][:, 1],
                         np.asarray([row["meaning"] == "postponed" for row in sent]))
    assert want["selected"][1][1] != pytest.approx(all_scores[8])
    np.testing.assert_allclose(want["pairs"], [[2.5, 1.25], [-1.25, 2.5], [3.75, -1.25]])
    assert want["embedding"] == pytest.approx(2.5)
    assert want["attn_share"] == pytest.approx(2 / 3)
    for values in want["factors"].values():
        np.testing.assert_allclose(values, [1, 1, 1])


@pytest.mark.parametrize("target, ks", [("postponed", (1, 10, 100)),
                                        ("postponed", (1, 3, 10)),
                                        ("accepted", (1, 10, 100))])
def test_analyze_grid_metrics_selection_writes_and_factors(target, ks):
    sent, rec = _grid()
    want = _grid_reference(sent, rec, target, ks)
    if ks == (1, 10, 100):
        got = r34_analyze.analyze(sent, rec, target)
    else:
        got = r34_analyze.analyze(sent, rec, target, ks=ks)
    assert isinstance(got, dict)
    for row, expected in zip(_layer_rows(got, 3), want["layers"]):
        for k in ks:
            assert _acc_value(row, k, ks) == pytest.approx(expected[f"acc_{k}"])
        for key in ("auc_dir_hidden", "auc_dir_resid", "share_top1pct"):
            assert row[key] == pytest.approx(expected[key], rel=1e-7, abs=1e-9)
        np.testing.assert_allclose(row["eta2_top10"], expected["eta2_top10"], rtol=1e-7, atol=1e-9)
        # With 100 hidden units, a top-1% share must lie in [1/100, 1].
        assert np.isfinite(row["share_gauss"])
        assert 0.01 <= row["share_gauss"] <= 1.0
    assert got["best_layer"] == want["best_layer"]
    selected = _selected_pairs(got["selected"])
    assert [j for j, _ in selected] == [j for j, _ in want["selected"]]
    np.testing.assert_allclose([s for _, s in selected], [s for _, s in want["selected"]],
                               rtol=1e-7, atol=1e-9)
    # Support embedding stored alongside writes, or inside the writes result.
    source = got if any(key in got for key in ("embedding", "embed", "emb")) else got["writes"]
    pairs, embedding = _read_writes(source, 3)
    np.testing.assert_allclose(pairs, want["pairs"], rtol=1e-7, atol=1e-9)
    assert embedding == pytest.approx(want["embedding"])
    assert embedding + np.sum(pairs) == pytest.approx(want["embedding"] + np.sum(want["pairs"]), rel=1e-3)
    assert got["attn_share"] == pytest.approx(want["attn_share"])
    for field, expected in want["factors"].items():
        np.testing.assert_allclose(_factor_accuracies(got["factors"][field], 3), expected,
                                   rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("present", [(), ("stance",), ("speech_act", "voice")])
def test_analyze_factors_include_only_present_optional_fields(present):
    sent, rec = _grid()
    want = _grid_reference(sent, rec, "postponed", (1, 10, 100))
    fields = ("stance", "speech_act", "voice")
    sent = [{key: value for key, value in row.items() if key not in fields or key in present}
            for row in sent]
    got = r34_analyze.analyze(sent, rec, "postponed")
    factors = got.get("factors", {})
    for field in fields:
        if field in present:
            np.testing.assert_allclose(_factor_accuracies(factors[field], 3), want["factors"][field],
                                       rtol=1e-7, atol=1e-9)
        else:
            assert field not in factors
