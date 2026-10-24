"""pairtable's checks on random channels and random directions, and a joint-answer contrast (spec:
authors/spec_r32_basis.md).

The checks tools/pairtable.py makes on every row of a unit's real table (its arm every: the unit's d on the checked
photos T with the same answer first, its percentile among its layer's columns, its 3 similar channels' percentiles, and
the same with the answers permuted within class), without readings, on three groups of candidates per layer of the
units of authors/round31_units.json: selected (those units), channel (k channels of the layer drawn at random, the
selected ones excluded) and direction (k random unit-norm mixtures of the layer's channels; a direction's value on a
photo is the spatial max of the layer's output projected on it, and it is ranked among the layer's channels and
itself). Everything is on the sub-pool of the photos whose label is in authors/round31_classes.json, with the answers
of runs/pairtable, runs/r30b/pairtable and runs/r30a/pairtable (merged in that order, the last winning; a photo
without an answer has None for every question). For the selected units each row also gets a joint contrast: the
difference of its two answers' coefficients in one least-squares fit of the centred values on indicators of every
answer (with at least 5 photos) of every question, over the column's sd. Per candidate also the excess kurtosis of its
values over all photos and over the sub-pool. No Claude calls; the model runs only for the direction values (cached).
Outputs: runs/r32/ (basis.json, report.txt, dirvalues_<layer>.npy).

Decision printed in the report, per checked row (as tools/pairtable.py): the candidate's d on T, with the same answer
first, at a percentile >= 95 among its layer's columns and above 0 -> held, else not held; too few when either answer
has fewer than 5 photos of T; distinctive when held and the lowest of its 3 similar channels' percentiles is below 50.
The joint outcome applies the same cut to the joint contrast on T; too few when either answer has no column. Per layer
and group, and per group: the mean number of held rows per candidate and the share of candidates with at least one.

usage: python -I tools/r32_basis.py [--k 40] [--out runs/r32] [--device cuda|cpu] [--selftest]
"""
import argparse
import gc
import json
import statistics
import sys
import zlib
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))  # -I does not add the script dir
import bundlemax  # noqa: E402
import contrastloop  # noqa: E402
import pairtable  # noqa: E402
import proploop  # noqa: E402
import r31_validate  # noqa: E402

OUT = 'runs/r32'  # relative paths are taken from the repo root
UNITS_FILE = 'authors/round31_units.json'
CLASSES_FILE = 'authors/round31_classes.json'
ANSWERS = ('runs/pairtable/answers.json', 'runs/r30b/pairtable/answers.json', 'runs/r30a/pairtable/answers.json')
SALT = 'r32:'  # draw's seed: crc32(SALT + layer)
K_DRAW = 40  # random channels and random directions per layer
GROUPS = ('selected', 'channel', 'direction')
ROW_KEYS = ('row', 'q', 'first', 'second', 'D_d', 'D_pct', 'T_d', 'T_pct', 'T_outcome', 'T_distinctive',
            'T_null_outcome')
JOINT_KEYS = ('jD_d', 'jD_pct', 'jT_d', 'jT_pct', 'jT_outcome')
RECORD_KEYS = ('group', 'unit', 'layer', 'n_D', 'n_T', 'classes', 'peers', 'kurt_all', 'kurt_pool', 'rows', 'n_rows',
               'n_D95', 'n_held', 'n_distinctive', 'n_held_null')
SUMMARY_KEYS = ('n', 'mean_held', 'share_held', 'mean_distinctive', 'share_distinctive', 'mean_held_null',
                'share_held_null', 'mean_D95', 'median_kurt_all')


# ---------- pool ----------

def load_classes(path):
    """the labels of the JSON list at path (ints 0-999), sorted, each once; stops when it is not such a list"""
    cl = json.loads(Path(path).read_text())
    if not isinstance(cl, list) or not cl or not all(
            isinstance(x, int) and not isinstance(x, bool) and 0 <= x < 1000 for x in cl):
        raise SystemExit(f'{path}: expected a JSON list of class labels (ints 0-999); stopping')
    return sorted(set(cl))


def sub_pool(labels, classes):
    """the photo ids (labels over all photos) whose label is in classes, in id order"""
    return np.flatnonzero(np.isin(np.asarray(labels), list(classes))).astype(np.int64)


# ---------- candidates: random channels, random directions ----------

def draw(layer, m, exclude, k):
    """(chans, W): rng = default_rng(crc32('r32:' + layer)); chans = k of the channels 0..m-1 not in exclude, drawn
    without replacement, sorted, as ints; then from the same rng W = standard normal m x k, each column divided by its
    L2 norm"""
    rng = np.random.default_rng(zlib.crc32((SALT + layer).encode()))
    ex = {int(x) for x in exclude}
    pool = np.array([j for j in range(m) if j not in ex], dtype=np.int64)
    chans = sorted(int(x) for x in rng.choice(pool, size=k, replace=False))
    W = rng.standard_normal((m, k))
    W = W / np.linalg.norm(W, axis=0, keepdims=True)
    return chans, W


def project_max(F, W):
    """F n x m x h x w, W m x k: per photo and column j, the max over the h x w positions of sum_c W[c, j] F[:, c];
    float32, n x k"""
    F = np.asarray(F, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    assert F.ndim == 4 and W.ndim == 2 and F.shape[1] == W.shape[0], (F.shape, W.shape)
    return np.tensordot(F, W, axes=([1], [0])).max(axis=(1, 2)).astype(np.float32)  # n x h x w x k -> n x k


def direction_values(model, layer, W, ids, device, make_input=None):
    """len(ids) x k float32: one forward pass per batch of proploop.BATCH photos (input make_input(ids), default
    bundlemax.net_input, on device), the module named layer hooked as proploop.forward_values hooks it and the pass
    stopped after it (img_acts.Stop); each batch's output projected by project_max"""
    import torch
    make_input = make_input or bundlemax.net_input
    Stop = bundlemax._bank().Stop
    mod = dict(model.named_modules())[layer]
    got = {}

    def hook(_m, _i, out):
        assert out.shape[1] == proploop.NCH[layer] == W.shape[0], (layer, tuple(out.shape), W.shape)
        got['F'] = out.detach().cpu().numpy()
        raise Stop

    res = []
    h = mod.register_forward_hook(hook)
    try:
        for b, s in enumerate(range(0, len(ids), proploop.BATCH)):
            x = make_input(ids[s:s + proploop.BATCH]).to(device)
            got.clear()
            try:
                with torch.no_grad():
                    model(x)
            except Stop:
                pass
            assert 'F' in got, layer
            res.append(project_max(got.pop('F'), W))
            if b % 20 == 19:
                print(f'  directions {min(s + proploop.BATCH, len(ids))}/{len(ids)}', flush=True)
    finally:
        h.remove()
    return np.concatenate(res) if res else np.zeros((0, W.shape[1]), dtype=np.float32)


def cached_direction_values(layer, W, get_model, device, out):
    """out/dirvalues_<layer>.npy, loaded when present with shape (N_PHOTOS, k) and dtype float32; else
    direction_values over np.arange(N_PHOTOS) with get_model(device) (full float32 on CUDA, as proploop.layer_values),
    saved via a temporary file and rename"""
    path = out / f'dirvalues_{layer}.npy'
    want = (proploop.N_PHOTOS, W.shape[1])
    if path.exists():
        v = np.load(path)
        if v.shape == want and v.dtype == np.float32:
            print(f'direction values of {layer}: cached', flush=True)
            return v
        print(f'{path}: shape {v.shape}, dtype {v.dtype}; expected {want} float32; recomputing', flush=True)
    proploop.no_tf32()
    print('direction values of', layer, 'on', device, flush=True)
    v = direction_values(get_model(device), layer, W, np.arange(proploop.N_PHOTOS), device)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f'dirvalues_{layer}.tmp.npy'
    np.save(tmp, np.ascontiguousarray(v, dtype=np.float32))
    tmp.replace(path)
    return v


# ---------- statistics ----------

def joint_d(R, ans_rows, questions, q, a, b, min_n=pairtable.MIN_N):
    """R n x m centred values, ans_rows one answer row per photo. X: a column of ones, then per question q' and answer
    index v of q' (in order) with at least min_n photos answering v, a column 1 on those photos, 0 elsewhere. None when
    answer a or b of question q has no column; else (beta[col(q, a)] - beta[col(q, b)]) / the sd of each column of R
    over the n photos (ddof 1), 0 where that sd is 0; beta = lstsq(X, R)"""
    R = np.asarray(R, dtype=np.float64)
    n = R.shape[0]
    assert R.ndim == 2 and len(ans_rows) == n, (R.shape, len(ans_rows))
    cols, where = [np.ones(n)], {}
    for qq, Q in enumerate(questions):
        codes = proploop._codes([r[qq] for r in ans_rows])
        for v in range(len(Q['answers'])):
            on = codes == v
            if np.count_nonzero(on) >= min_n:
                where[(qq, v)] = len(cols)
                cols.append(on.astype(np.float64))
    if (q, a) not in where or (q, b) not in where:
        return None
    beta = np.linalg.lstsq(np.stack(cols, axis=1), R, rcond=None)[0]
    diff = beta[where[(q, a)]] - beta[where[(q, b)]]
    sd = R.std(axis=0, ddof=1)
    return np.where(sd > 0, diff / np.where(sd > 0, sd, 1.0), 0.0)


def excess_kurtosis(x):
    """mean((x - mean)^4) / var^2 - 3, var the population variance (ddof 0); None when the variance is 0"""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0 or np.ptp(x) == 0:
        return None
    z = x - x.mean()
    var = float((z ** 2).mean())
    if var == 0:
        return None
    return float((z ** 4).mean() / var ** 2 - 3.0)


def _joint(R, rows, questions, q, a, b, c):
    """(column c's joint_d, its percentile among the columns), or (None, None)"""
    jd = joint_d(R, rows, questions, q, a, b)
    return (None, None) if jd is None else (float(jd[c]), proploop.percentile(jd, c))


# ---------- one candidate ----------

def candidate(name, group, Vc, c, labels, names, row_of, questions, x_all, joint=False):
    """the record of one candidate. Vc: sub-pool values (photos x columns), column c the candidate (a direction: the
    layer's m channels with its column appended, c = m); labels, names, row_of (an answer row per photo) over the
    sub-pool; x_all: the candidate's values over all photos. As pairtable.run_unit without readings: photos D and T,
    peers, the answers on T permuted within class (seed crc32(name) + NULL_T), the real table on D and every row of it
    checked on T with pairtable.pick; with joint, each row's joint_d on D and on T"""
    Vc, labels = np.asarray(Vc), np.asarray(labels)
    seed = zlib.crc32(name.encode())

    # photos and peers
    classes, _, D, T = pairtable.unit_photos({'chan': c}, Vc, labels, names)
    peers = [p for p, _ in contrastloop.peers(Vc, labels, c)]

    # answers and values on D and T, the real table on D
    D_ids, T_ids = np.asarray(D, dtype=np.int64), np.asarray(T, dtype=np.int64)
    rows_D, rows_T = [row_of[i] for i in D], [row_of[i] for i in T]
    null_T = proploop.permute_within(rows_T, labels[T_ids], classes,
                                     np.random.default_rng(seed + pairtable.NULL_T))
    R_D, R_T = proploop.centre(Vc[D_ids], labels[D_ids]), proploop.centre(Vc[T_ids], labels[T_ids])
    table = pairtable.table_rows(R_D, rows_D, questions, c, peers)

    # every row checked on T
    rows = []
    for r in table:
        p = pairtable.pick(r, questions, R_T, rows_T, null_T, c, peers)
        x = {'row': p['row'], 'q': p['q'], 'first': p['first'], 'second': p['second'], 'D_d': p['D']['d'],
             'D_pct': p['D']['pct'], 'T_d': p['T']['d'], 'T_pct': p['T']['pct'], 'T_outcome': p['T']['outcome'],
             'T_distinctive': p['T']['distinctive'], 'T_null_outcome': p['T_null']['outcome']}
        if joint:
            jD = _joint(R_D, rows_D, questions, p['q'], p['first'], p['second'], c)
            jT = _joint(R_T, rows_T, questions, p['q'], p['first'], p['second'], c)
            x.update(jD_d=jD[0], jD_pct=jD[1], jT_d=jT[0], jT_pct=jT[1],
                     jT_outcome='too few' if jT[0] is None else contrastloop.outcome(jT[0], jT[1]))
        rows.append(x)

    return {'group': group, 'unit': name, 'layer': name.split(':')[0], 'n_D': len(D), 'n_T': len(T),
            'classes': classes, 'peers': peers, 'kurt_all': excess_kurtosis(x_all),
            'kurt_pool': excess_kurtosis(Vc[:, c]), 'rows': rows, 'n_rows': len(rows),
            'n_D95': sum(x['D_pct'] >= pairtable.PCT_RULE for x in rows),
            'n_held': sum(x['T_outcome'] == 'held' for x in rows),
            'n_distinctive': sum(bool(x['T_distinctive']) for x in rows),
            'n_held_null': sum(x['T_null_outcome'] == 'held' for x in rows)}


# ---------- summary and report ----------

def _stats(recs):
    """the summary numbers of a non-empty list of records"""
    n = len(recs)

    def mean(k):
        return sum(r[k] for r in recs) / n

    def share(k):
        return sum(r[k] >= 1 for r in recs) / n
    ks = [r['kurt_all'] for r in recs if r['kurt_all'] is not None]
    return {'n': n, 'mean_held': mean('n_held'), 'share_held': share('n_held'),
            'mean_distinctive': mean('n_distinctive'), 'share_distinctive': share('n_distinctive'),
            'mean_held_null': mean('n_held_null'), 'share_held_null': share('n_held_null'),
            'mean_D95': mean('n_D95'), 'median_kurt_all': float(statistics.median(ks)) if ks else None}


def summary(records):
    """{'by_layer': {layer: {group: numbers}}, 'by_group': {group: numbers}}: n, the mean of n_held, n_distinctive,
    n_held_null and n_D95, the share of records with n_held, n_distinctive, n_held_null >= 1, and the median kurt_all
    (Nones left out; None when there is none). Layers in network order, groups in GROUPS order, present ones only"""
    layers = sorted({r['layer'] for r in records}, key=proploop.LAYERS.index)
    by_layer = {}
    for L in layers:
        by_layer[L] = {}
        for g in GROUPS:
            rs = [r for r in records if r['layer'] == L and r['group'] == g]
            if rs:
                by_layer[L][g] = _stats(rs)
    by_group = {g: _stats([r for r in records if r['group'] == g]) for g in GROUPS
                if any(r['group'] == g for r in records)}
    return {'by_layer': by_layer, 'by_group': by_group}


def _cell(x):
    if x is None:
        return '-'
    if isinstance(x, int) and not isinstance(x, bool):
        return str(x)
    return f'{x:.3f}'


def report_text(records, sm):
    """the summary as a table (per layer and group, then per group over all layers as layer 'all'), then one line per
    selected unit: n_held, n_distinctive and each held row as 'q first>second T_pct joint jT_outcome'"""
    head = ['layer', 'group'] + list(SUMMARY_KEYS)
    cells = [head]
    cells += [[L, g] + [_cell(s[k]) for k in SUMMARY_KEYS] for L, by in sm['by_layer'].items() for g, s in by.items()]
    cells += [['all', g] + [_cell(s[k]) for k in SUMMARY_KEYS] for g, s in sm['by_group'].items()]
    w = [max(len(row[j]) for row in cells) for j in range(len(head))]
    lines = [' '.join(x.ljust(w[j]) if j < 2 else x.rjust(w[j]) for j, x in enumerate(row)).rstrip()
             for row in cells]
    lines += ['', 'selected units: n_held, n_distinctive; each held row as q first>second T_pct joint jT_outcome '
                  '(q, first, second: indices into the questions and their answers)']
    for r in records:
        if r['group'] != 'selected':
            continue
        held = [f'{x["q"]} {x["first"]}>{x["second"]} {x["T_pct"]:.1f} joint {x["jT_outcome"]}'
                for x in r['rows'] if x['T_outcome'] == 'held']
        lines.append(f'{r["unit"]}  n_held {r["n_held"]}, n_distinctive {r["n_distinctive"]}'
                     + (': ' + '; '.join(held) if held else ''))
    return '\n'.join(lines)


# ---------- selftest ----------

def selftest():
    import tempfile
    from collections import OrderedDict

    assert SALT == 'r32:' and K_DRAW == 40 and GROUPS == ('selected', 'channel', 'direction')
    assert pairtable.MIN_N == 5 and pairtable.PCT_RULE == 95.0 and pairtable.NULL_T == 0

    # pool helpers
    assert sub_pool(np.array([4, 2, 9, 4, 7]), [4, 7]).tolist() == [0, 3, 4]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'classes.json'
        p.write_text('[7, 3, 7]')
        assert load_classes(p) == [3, 7]
        for bad in ('{"classes": [1]}', '[]', '[true]', '["3"]', '[1000]', '[-1]', '[1.0]'):
            p.write_text(bad)
            try:
                load_classes(p)
            except SystemExit:
                continue
            raise AssertionError('accepted ' + bad)

    # draw: the recipe by hand, determinism, exclusion, unit norms
    ex = [3, 7, 100, 7]
    chans, W = draw('mixed4c', 512, ex, 40)
    c2, W2 = draw('mixed4c', 512, ex, 40)
    assert chans == c2 and np.array_equal(W, W2)
    assert len(set(chans)) == 40 and chans == sorted(chans) and all(type(x) is int for x in chans)
    assert not set(chans) & set(ex) and all(0 <= x < 512 for x in chans)
    assert W.shape == (512, 40) and np.allclose(np.linalg.norm(W, axis=0), 1.0, rtol=0, atol=1e-12)
    g = np.random.default_rng(zlib.crc32(b'r32:mixed4c'))
    assert chans == sorted(int(x) for x in g.choice([x for x in range(512) if x not in ex], size=40, replace=False))
    Wh = g.standard_normal((512, 40))
    assert np.allclose(W, Wh / np.linalg.norm(Wh, axis=0), rtol=0, atol=1e-15)
    assert draw('mixed4d', 512, ex, 40)[0] != chans  # another layer, another draw
    assert draw('mixed3a', 10, [0, 2, 4, 6, 8], 5)[0] == [1, 3, 5, 7, 9]
    try:
        draw('mixed3a', 10, list(range(6)), 5)
        raise AssertionError('drew more channels than are left')
    except ValueError:
        pass

    # project_max against a loop on a small random F; a one-hot column is the channel's spatial max
    rng = np.random.default_rng(10)
    F = rng.normal(size=(3, 5, 4, 6)).astype(np.float32)
    Wp = rng.normal(size=(5, 2))
    pm = project_max(F, Wp)
    want = np.empty((3, 2))
    for n in range(3):
        for j in range(2):
            want[n, j] = max(sum(float(Wp[c, j]) * float(F[n, c, h, w]) for c in range(5))
                             for h in range(4) for w in range(6))
    assert pm.shape == (3, 2) and pm.dtype == np.float32 and np.allclose(pm, want, rtol=1e-6, atol=1e-6), (pm, want)
    assert np.array_equal(project_max(F, np.eye(5)[:, [2]]), F[:, 2].max(axis=(1, 2))[:, None])

    # joint_d: column 0 depends only on question 1, column 1 only on question 0, column 2 is constant; question 0's
    # answer 0 comes with question 1's answer 0 for 15 of its 18 photos, its answer 1 with question 1's answer 1 for
    # 15 of 18. Question 2: 4 photos, below min_n; question 0's answer 2: no photo
    q0 = [0] * 18 + [1] * 18 + [None] * 4
    q1 = [0] * 15 + [1] * 3 + [0] * 3 + [1] * 15 + [0, 1, 0, 1]
    q2 = [0] * 4 + [None] * 36
    rows_j = [list(x) for x in zip(q0, q1, q2)]
    qs_j = [{'question': 'A?', 'answers': ['x', 'z', 'w']}, {'question': 'B?', 'answers': ['y', 'v']},
            {'question': 'C?', 'answers': ['u', 't']}]
    col_b = np.array([1.0 if a == 0 else 0.0 for a in q1])
    col_a = np.array([1.0 if a == 0 else 0.0 for a in q0])
    Rj = np.stack([col_b, col_a, np.full(40, 0.25)], axis=1)
    plain = contrastloop.contrast(Rj, q0, 0, 1)[2]
    assert plain[0] > 1.0, plain  # question 0's plain contrast on column 0: 15 of 18 against 3 of 18
    j0 = joint_d(Rj, rows_j, qs_j, 0, 0, 1)
    assert abs(j0[0]) < 1e-9 and abs(j0[1] - 1 / np.std(col_a, ddof=1)) < 1e-9 and j0[2] == 0, j0
    j1 = joint_d(Rj, rows_j, qs_j, 1, 0, 1)
    assert abs(j1[0] - 1 / np.std(col_b, ddof=1)) < 1e-9 and abs(j1[1]) < 1e-9 and j1[2] == 0, j1
    assert abs(j1[0] - contrastloop.contrast(Rj, q1, 0, 1)[2][0]) < 1e-9  # every photo answers question 1
    assert np.allclose(joint_d(Rj, rows_j, qs_j, 0, 1, 0), -j0, rtol=0, atol=1e-12)
    assert joint_d(Rj, rows_j, qs_j, 2, 0, 1) is None and joint_d(Rj, rows_j, qs_j, 0, 0, 2) is None
    assert joint_d(Rj, rows_j, qs_j, 2, 0, 1, min_n=4) is None  # answer 1 of question 2 has no photo
    no_q2 = [[a, b, None] for a, b, _ in rows_j]
    assert np.array_equal(joint_d(Rj, no_q2, qs_j, 0, 0, 1), j0)  # an answer below min_n adds no column

    # excess kurtosis on known vectors, and against a plain-Python computation
    assert excess_kurtosis([1, -1, 1, -1]) == -2.0
    assert abs(excess_kurtosis([0] * 9 + [1]) - 46 / 9) < 1e-12
    assert excess_kurtosis([2.5] * 7) is None and excess_kurtosis(np.zeros(3, dtype=np.float32)) is None
    xr = np.random.default_rng(11).normal(size=50).tolist()
    mu = statistics.fmean(xr)
    want_k = statistics.fmean([(x - mu) ** 4 for x in xr]) / statistics.pvariance(xr) ** 2 - 3
    assert abs(excess_kurtosis(xr) - want_k) < 1e-9

    # candidate on a tiny synthetic pool (6 classes of 20 photos, 24 channels, hand-made answers) against the same
    # steps done by hand; channel 3 is raised on question 0's answer 0
    rng = np.random.default_rng(8)
    n_s, m_s = 120, 24
    labs_s = rng.permutation(np.repeat(np.arange(6), 20))
    nms_s = [f'img/{i}.jpeg' for i in range(n_s)]
    qs_s = [{'question': 'Is it red?', 'answers': ['yes', 'no']},
            {'question': 'Which side?', 'answers': ['left', 'middle', 'right']},
            {'question': 'Is there text?', 'answers': ['yes', 'no']}]
    rows_s = [[None] * 3 if i % 17 == 0 else [i % 2, i % 3, 0 if i % 20 == 1 else None] for i in range(n_s)]
    V_s = rng.random((n_s, m_s)).astype(np.float32)
    V_s[:, 3] += np.array([3.0 if r[0] == 0 else 0.0 for r in rows_s], dtype=np.float32)
    x_s = rng.random(500).astype(np.float32)

    def by_hand(name, Vc, c):
        seed = zlib.crc32(name.encode())
        classes, _, D, T = pairtable.unit_photos({'chan': c}, Vc, labs_s, nms_s)
        peers = [p for p, _ in contrastloop.peers(Vc, labs_s, c)]
        Di, Ti = np.asarray(D, dtype=np.int64), np.asarray(T, dtype=np.int64)
        rD, rT = [rows_s[i] for i in D], [rows_s[i] for i in T]
        nT = proploop.permute_within(rT, labs_s[Ti], classes, np.random.default_rng(seed + pairtable.NULL_T))
        RD, RT = proploop.centre(Vc[Di], labs_s[Di]), proploop.centre(Vc[Ti], labs_s[Ti])
        tab = pairtable.table_rows(RD, rD, qs_s, c, peers)
        return classes, D, T, peers, (RD, rD), (RT, rT), [pairtable.pick(r, qs_s, RT, rT, nT, c, peers) for r in tab]

    def plain_fields(p):
        return {'row': p['row'], 'q': p['q'], 'first': p['first'], 'second': p['second'], 'D_d': p['D']['d'],
                'D_pct': p['D']['pct'], 'T_d': p['T']['d'], 'T_pct': p['T']['pct'], 'T_outcome': p['T']['outcome'],
                'T_distinctive': p['T']['distinctive'], 'T_null_outcome': p['T_null']['outcome']}

    cs = candidate('mixed4c:3', 'selected', V_s, 3, labs_s, nms_s, rows_s, qs_s, x_s, joint=True)
    cls_h, D_h, T_h, peers_h, (RD_h, rD_h), (RT_h, rT_h), picks_h = by_hand('mixed4c:3', V_s, 3)
    assert list(cs) == list(RECORD_KEYS)
    assert (cs['group'], cs['unit'], cs['layer'], cs['n_D'], cs['n_T']) == (
        'selected', 'mixed4c:3', 'mixed4c', len(D_h), len(T_h))
    assert D_h and T_h and sorted(D_h + T_h) == list(range(n_s)) and sorted(cls_h) == list(range(6))
    assert cs['classes'] == cls_h and cs['peers'] == peers_h and len(peers_h) == 3 and 3 not in peers_h
    assert cs['n_rows'] == len(cs['rows']) == len(picks_h) > 0
    for r, p in zip(cs['rows'], picks_h):
        assert list(r) == list(ROW_KEYS + JOINT_KEYS), r
        assert {k: r[k] for k in ROW_KEYS} == plain_fields(p), (r, p)
        jD = joint_d(RD_h, rD_h, qs_s, p['q'], p['first'], p['second'])
        jT = joint_d(RT_h, rT_h, qs_s, p['q'], p['first'], p['second'])
        assert (r['jD_d'], r['jD_pct']) == ((None, None) if jD is None else (float(jD[3]), proploop.percentile(jD, 3)))
        assert (r['jT_d'], r['jT_pct']) == ((None, None) if jT is None else (float(jT[3]), proploop.percentile(jT, 3)))
        assert r['jT_outcome'] == ('too few' if jT is None else
                                   contrastloop.outcome(float(jT[3]), proploop.percentile(jT, 3)))
    assert all(r['q'] != 2 for r in cs['rows'])  # question 2 has too few photos per answer on D
    assert cs['n_D95'] == sum(r['D_pct'] >= 95 for r in cs['rows'])
    assert cs['n_held'] == sum(r['T_outcome'] == 'held' for r in cs['rows'])
    assert cs['n_distinctive'] == sum(r['T_distinctive'] for r in cs['rows'])
    assert cs['n_held_null'] == sum(r['T_null_outcome'] == 'held' for r in cs['rows'])
    assert cs['kurt_all'] == excess_kurtosis(x_s) and cs['kurt_pool'] == excess_kurtosis(V_s[:, 3])
    planted = [r for r in cs['rows'] if (r['q'], r['first'], r['second']) == (0, 0, 1)]
    assert len(planted) == 1 and planted[0]['D_pct'] >= 95 and planted[0]['T_outcome'] == 'held', planted
    assert planted[0]['jT_outcome'] == 'held' and cs['n_held'] >= 1, planted
    assert json.loads(json.dumps(cs)) == cs  # plain JSON types only

    # a direction: the layer's channels with its column appended, ranked among them; peers are channels
    w_s = rng.normal(size=m_s)
    col_s = (V_s.astype(np.float64) @ (w_s / np.linalg.norm(w_s))).astype(np.float32)
    Vd = np.concatenate([V_s, col_s[:, None]], axis=1)
    cd = candidate('mixed4c:dir07', 'direction', Vd, m_s, labs_s, nms_s, rows_s, qs_s, x_s)
    cls_d, Dd, Td, peers_d, _, _, picks_d = by_hand('mixed4c:dir07', Vd, m_s)
    assert list(cd) == list(RECORD_KEYS) and (cd['group'], cd['unit'], cd['layer']) == (
        'direction', 'mixed4c:dir07', 'mixed4c')
    assert (cd['classes'], cd['peers'], cd['n_D'], cd['n_T']) == (cls_d, peers_d, len(Dd), len(Td))
    assert len(peers_d) == 3 and all(0 <= x < m_s for x in peers_d)
    assert cd['rows'] == [plain_fields(p) for p in picks_d] and cd['n_rows'] == len(picks_d)
    assert cd['kurt_pool'] == excess_kurtosis(col_s) and json.loads(json.dumps(cd)) == cd

    # summary and report on hand records
    def rec(layer, group, nh, nd, nn, d95, k):
        return {'group': group, 'unit': f'{layer}:x', 'layer': layer, 'n_held': nh, 'n_distinctive': nd,
                'n_held_null': nn, 'n_D95': d95, 'kurt_all': k, 'rows': []}
    recs = [rec('mixed4c', 'channel', 0, 0, 1, 2, 1.0), rec('mixed4c', 'channel', 2, 1, 0, 3, 3.0),
            rec('mixed4c', 'channel', 1, 0, 0, 0, None), rec('mixed4d', 'channel', 0, 0, 0, 1, 5.0),
            rec('mixed4c', 'direction', 3, 2, 1, 4, 10.0), rec('mixed4d', 'direction', 0, 0, 0, 0, None)]
    sm = summary(recs)
    assert list(sm['by_layer']) == ['mixed4c', 'mixed4d'] and list(sm['by_group']) == ['channel', 'direction']
    assert list(sm['by_layer']['mixed4c']) == ['channel', 'direction']
    assert sm['by_layer']['mixed4c']['channel'] == {
        'n': 3, 'mean_held': 1.0, 'share_held': 2 / 3, 'mean_distinctive': 1 / 3, 'share_distinctive': 1 / 3,
        'mean_held_null': 1 / 3, 'share_held_null': 1 / 3, 'mean_D95': 5 / 3, 'median_kurt_all': 2.0}, sm
    assert sm['by_group']['channel']['n'] == 4 and sm['by_group']['channel']['median_kurt_all'] == 3.0
    assert sm['by_group']['channel']['share_held'] == 0.5 and sm['by_group']['channel']['mean_D95'] == 1.5
    assert sm['by_layer']['mixed4d']['direction'] == {
        'n': 1, 'mean_held': 0.0, 'share_held': 0.0, 'mean_distinctive': 0.0, 'share_distinctive': 0.0,
        'mean_held_null': 0.0, 'share_held_null': 0.0, 'mean_D95': 0.0, 'median_kurt_all': None}
    assert all(list(s) == list(SUMMARY_KEYS) for s in sm['by_group'].values())
    assert summary([]) == {'by_layer': {}, 'by_group': {}}
    sel = dict(rec('mixed4c', 'selected', 1, 0, 0, 2, 4.0), unit='mixed4c:447', rows=[
        {'q': 2, 'first': 0, 'second': 3, 'T_pct': 96.5, 'T_outcome': 'held', 'jT_outcome': 'not held'},
        {'q': 5, 'first': 1, 'second': 0, 'T_pct': 99.0, 'T_outcome': 'not held', 'jT_outcome': 'held'}])
    sel0 = dict(rec('mixed4d', 'selected', 0, 0, 0, 0, 4.0), unit='mixed4d:9')
    rt = report_text(recs + [sel, sel0], summary(recs + [sel, sel0])).splitlines()
    assert rt[0].split() == ['layer', 'group'] + list(SUMMARY_KEYS), rt[0]
    assert ['mixed4c', 'channel', '3', '1.000', '0.667', '0.333', '0.333', '0.333', '0.333', '1.667',
            '2.000'] in [x.split() for x in rt], rt
    assert ['all', 'direction', '2', '1.500', '0.500', '1.000', '0.500', '0.500', '0.500', '2.000',
            '10.000'] in [x.split() for x in rt], rt
    assert ['mixed4d', 'direction', '1', '0.000', '0.000', '0.000', '0.000', '0.000', '0.000', '0.000',
            '-'] in [x.split() for x in rt], rt
    assert rt[-2:] == ['mixed4c:447  n_held 1, n_distinctive 0: 2 0>3 96.5 joint not held',
                       'mixed4d:9  n_held 0, n_distinctive 0'], rt

    # direction values on a stand-in module (one layer, stopped after it), against project_max of its output
    import torch
    torch.manual_seed(0)

    class Boom(torch.nn.Module):
        def forward(self, x):
            raise AssertionError('ran past the stop')

    fm = torch.nn.Sequential(OrderedDict([('mixed3a', torch.nn.Conv2d(3, proploop.NCH['mixed3a'], 4, stride=4)),
                                          ('boom', Boom())])).eval()
    for q in fm.parameters():
        q.requires_grad_(False)
    xs = torch.randn(70, 3, 16, 16)

    def make_input(ids):
        return xs[torch.as_tensor(np.asarray(ids, dtype=np.int64))]
    Wd = draw('mixed3a', proploop.NCH['mixed3a'], [], 5)[1]
    dv = direction_values(fm, 'mixed3a', Wd, np.arange(70), 'cpu', make_input=make_input)
    with torch.no_grad():
        Fd = fm.mixed3a(xs).numpy()
    assert dv.shape == (70, 5) and dv.dtype == np.float32
    assert np.allclose(dv, project_max(Fd, Wd), rtol=1e-5, atol=1e-5)
    assert not fm.mixed3a._forward_hooks  # the hook is removed

    # the cache: loaded with the right shape and dtype (no model), else recomputed
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        good = np.random.default_rng(9).random((proploop.N_PHOTOS, 3)).astype(np.float32)
        np.save(td / 'dirvalues_mixed4c.npy', good)
        W3 = draw('mixed4c', 512, [], 3)[1]

        def no_model(dev):
            raise AssertionError('model loaded with the cache present')
        assert np.array_equal(cached_direction_values('mixed4c', W3, no_model, 'cpu', td), good)

        class Called(Exception):
            pass

        def gm(dev):
            raise Called
        for bad in (good[:, :2], good.astype(np.float64), good[:5]):
            np.save(td / 'dirvalues_mixed4c.npy', bad)
            try:
                cached_direction_values('mixed4c', W3, gm, 'cpu', td)
                raise AssertionError('a cache of the wrong shape or dtype was loaded')
            except Called:
                pass
    print('selftest ok')


# ---------- main ----------

def main():
    if '--selftest' in sys.argv[1:]:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=K_DRAW, help='random channels and random directions per layer')
    ap.add_argument('--out', default=OUT, help='a relative path is taken from the repo root')
    ap.add_argument('--device', default=None, help='default: cuda when available, else cpu')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.k < 1:
        ap.error('--k must be at least 1')
    out = r31_validate._path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = a.device
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # the pool, the answers, the sub-pool
    questions = pairtable.load_questions()
    nq = len(questions)
    names = json.loads(proploop.NAMES.read_text())
    with np.load(proploop.ACTS) as z:
        labels = np.asarray(z['classes'], dtype=np.int64)
    assert len(names) == len(labels) == proploop.N_PHOTOS, (len(names), len(labels))
    units = r31_validate.load_units([UNITS_FILE])
    layers = sorted({u['layer'] for u in units}, key=proploop.LAYERS.index)
    if not layers:
        raise SystemExit(f'{UNITS_FILE}: no units; stopping')
    missing = [L for L in layers if not (proploop.OUT / f'values_{L}.npy').exists()]
    if missing:
        raise SystemExit(f'no cached values of {missing} in {proploop.OUT}; stopping')
    src = r31_validate.merge(*[r31_validate.load_claude(r31_validate._path(p)) for p in ANSWERS])
    r31_validate.check_source(src, questions, len(names), 'answers')
    P = sub_pool(labels, load_classes(r31_validate._path(CLASSES_FILE)))
    labels_p = labels[P]
    names_p = [names[i] for i in P.tolist()]
    row_of = [list(src[i]) if i in src else [None] * nq for i in P.tolist()]
    print(f'sub-pool: {len(P)} photos of {len(set(labels_p.tolist()))} classes, '
          f'{sum(i in src for i in P.tolist())} with answers; layers {layers}; k {a.k}; device {device}', flush=True)

    held = {}

    def get_model(dev):
        if dev not in held:
            import torch
            torch.set_num_threads(4)
            mdl = bundlemax._bank().load_model(proploop.WEIGHTS)
            for q in mdl.parameters():
                q.requires_grad_(False)
            held[dev] = mdl.eval().to(dev)
        return held[dev]

    # one layer at a time: values, draw, direction values, every candidate
    records, text = [], ''
    for L in layers:
        print('==', L, flush=True)
        V = proploop.layer_values([L], None, np.arange(proploop.N_PHOTOS))[L]
        m = V.shape[1]
        sel = [u for u in units if u['layer'] == L]
        chans, W = draw(L, m, [u['chan'] for u in sel], a.k)
        DV = cached_direction_values(L, W, get_model, device, out)
        Vp, DVp = V[P], DV[P]
        common = (labels_p, names_p, row_of, questions)
        for u in sel:
            records.append(candidate(u['name'], 'selected', Vp, u['chan'], *common, V[:, u['chan']], joint=True))
        print(f'  selected {len(sel)}', flush=True)
        for ch in chans:
            records.append(candidate(f'{L}:{ch}', 'channel', Vp, ch, *common, V[:, ch]))
        print(f'  channels {len(chans)}', flush=True)
        for j in range(W.shape[1]):
            Vc = np.concatenate([Vp, DVp[:, j:j + 1]], axis=1)
            records.append(candidate(f'{L}:dir{j:02d}', 'direction', Vc, m, *common, DV[:, j]))
        print(f'  directions {W.shape[1]}', flush=True)
        sm = summary(records)
        proploop.save_json({'records': records, 'summary': sm}, out / 'basis.json')
        text = report_text(records, sm)
        (out / 'report.txt').write_text(text + '\n')
        del V, DV, Vp, DVp
        gc.collect()
    held.clear()
    print(text, flush=True)
    print('report ->', out / 'report.txt')


if __name__ == '__main__':
    main()
