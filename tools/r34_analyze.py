"""Which MLP units and residual directions carry one meaning across many surface forms (r34).

The sentences form a full form x meaning grid. Forms are split by a hash into a selection half S and a test half T;
units and directions are chosen on S and scored on T. Per layer: nearest-centroid balanced accuracy from the top-k
hidden units (acc_k), ROC AUC of the S mean-difference direction in the hidden space and in the residual stream,
the share of that hidden direction carried by its top 1% entries (and the same share for a random direction), and
the meaning / form eta^2 of the top-10 units. Also: the attention / MLP split of the final residual difference, the
best layer and its 8 most positive units, and acc_10 for the record fields stance, speech_act, voice when present.

usage:
  python tools/r34_analyze.py --record DIR [--target postponed] [--readout suffix|text] --out FILE
  python tools/r34_analyze.py --selftest

DIR holds sentences.json (row i of every array = sentence i) and record.npz with hidden_<readout> [n, L, H] and
pre_<readout>, mid_<readout>, post_<readout> [n, L, d] (residual before layer l, after its attention, after the layer).
numpy and scipy only; CPU.
"""
import argparse
import json
import math
import os
import zlib
from collections import Counter

import numpy as np
from scipy import stats

FACTORS = ('stance', 'speech_act', 'voice')
ARRAYS = ('hidden', 'pre', 'mid', 'post')


def half(form):
    """0 = selection half S, 1 = test half T"""
    return zlib.crc32(f'r34:{form}'.encode()) % 2


def f64(a):
    return np.asarray(a, dtype=np.float64)


def mean_diff(X, y):
    """mean over target rows minus mean over the other rows"""
    X, y = f64(X), np.asarray(y, bool)
    return X[y].mean(0) - X[~y].mean(0)


def unit_scores(X, y):
    """per column: (mean target - mean rest) / sd over all rows (ddof 1); 0 where the sd is 0"""
    X = f64(X)
    sd = X.std(0, ddof=1)
    return np.where(sd > 0, mean_diff(X, y) / np.where(sd > 0, sd, 1.0), 0.0)


def top_units(scores, k, sign=0):
    """indices of the k largest |scores| (sign 0) or largest scores (sign +1); ties by lower index"""
    assert sign in (0, 1)
    s = f64(scores)
    key = np.abs(s) if sign == 0 else s
    return np.argsort(-key, kind='stable')[:k]


def centroid_accuracy(X_S, y_S, X_T, y_T, units):
    """balanced accuracy on T of nearest-centroid target vs rest on the given columns, standardised on S"""
    units = np.asarray(units, dtype=int)
    y_S, y_T = np.asarray(y_S, bool), np.asarray(y_T, bool)
    A, B = f64(X_S)[:, units], f64(X_T)[:, units]
    mu, sd = A.mean(0), A.std(0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    A, B = (A - mu) / sd, (B - mu) / sd
    c_pos, c_neg = A[y_S].mean(0), A[~y_S].mean(0)
    pred = ((B - c_pos) ** 2).sum(1) < ((B - c_neg) ** 2).sum(1)  # equal distance -> rest
    return float(0.5 * (pred[y_T].mean() + (~pred[~y_T]).mean()))


def direction_auc(X_S, y_S, X_T, y_T):
    """ROC AUC on T of the projection onto d = mean_S(target) - mean_S(rest)"""
    y_T = np.asarray(y_T, bool)
    p = f64(X_T) @ mean_diff(X_S, y_S)
    pos, neg = p[y_T], p[~y_T]
    u = stats.mannwhitneyu(pos, neg, alternative='two-sided').statistic
    return float(u / (len(pos) * len(neg)))


def top_share(d, frac=0.01):
    """share of sum(d^2) carried by the ceil(frac * len(d)) entries of largest |d|"""
    d2 = f64(d) ** 2
    k = math.ceil(round(frac * len(d2), 9))  # round first so float error (e.g. 0.07 * 100) cannot add one entry
    return float(np.sort(d2)[::-1][:k].sum() / d2.sum())


def gaussian_share(m, frac=0.01, draws=20, seed=0):
    """mean top_share of `draws` standard normal vectors of length m (a uniformly random direction)"""
    rng = np.random.default_rng(seed)
    return float(np.mean([top_share(rng.standard_normal(m), frac) for _ in range(draws)]))


def eta2(x, meaning, form):
    """(eta2_meaning, eta2_form): share of the total sum of squares explained by meaning means and by form means"""
    x = f64(x)
    dev = x - x.mean()
    total = float(dev @ dev)
    if total == 0:
        return 0.0, 0.0

    def explained(g):
        g = np.asarray(g)
        return sum((g == v).sum() * (x[g == v].mean() - x.mean()) ** 2 for v in np.unique(g))
    return float(explained(meaning) / total), float(explained(form) / total)


def write_contributions(pre, mid, post, y, D):
    """[(attn, mlp)] per layer and the embedding term: target-minus-rest mean of each write, dotted with D / |D|"""
    y = np.asarray(y, bool)
    D = f64(D)
    u = D / np.linalg.norm(D)

    def proj(A):
        return float(mean_diff(A, y) @ u)
    L = pre.shape[1]
    terms = [(proj(f64(mid[:, l]) - f64(pre[:, l])), proj(f64(post[:, l]) - f64(mid[:, l]))) for l in range(L)]
    emb = proj(pre[:, 0])
    total = proj(post[:, L - 1])
    parts = emb + sum(a for a, _ in terms) + sum(m for _, m in terms)
    assert abs(parts - total) <= 1e-3 * abs(total), f'write identity fails: {parts} vs {total}'
    return terms, emb


def analyze(sent, rec, target, ks=(1, 10, 100)):
    """rec: {'hidden', 'pre', 'mid', 'post'} arrays of one readout, row i = sent[i]"""
    assert 10 in ks, 'best_layer and factors use acc_10'
    hidden, pre, mid, post = (rec[k] for k in ARRAYS)
    n, L, H = hidden.shape
    assert len(sent) == n and pre.shape == mid.shape == post.shape and pre.shape[:2] == (n, L)
    meaning = np.array([s['meaning'] for s in sent])
    form = np.array([s['form'] for s in sent])
    assert len(set(zip(form.tolist(), meaning.tolist()))) == n == len(set(form.tolist())) * len(set(meaning.tolist())), \
        'sentences are not a full form x meaning grid'
    y = meaning == target
    assert y.any() and not y.all(), f'target {target!r} absent or the only meaning'
    S = np.array([half(f) == 0 for f in form.tolist()])
    T = ~S
    assert S.any() and T.any(), 'one half of the forms is empty'

    fields = {}
    for name in FACTORS:
        if all(name in s for s in sent):
            vals = [s[name] for s in sent]
            cnt = Counter(vals)
            top = max(sorted(cnt, key=str), key=cnt.get)  # most common value; ties by sorted value
            fields[name] = (top, np.array([v == top for v in vals]))
    factors = {name: {'value': v, 'acc_10': []} for name, (v, _) in fields.items()}

    share_gauss = gaussian_share(H)
    layers = []
    for l in range(L):
        X, R = f64(hidden[:, l]), f64(post[:, l])
        scores = unit_scores(X[S], y[S])
        row = {'layer': l}
        for k in ks:
            row[f'acc_{k}'] = centroid_accuracy(X[S], y[S], X[T], y[T], top_units(scores, k))
        row['auc_dir_hidden'] = direction_auc(X[S], y[S], X[T], y[T])
        row['share_top1pct'] = top_share(mean_diff(X[S], y[S]))
        row['share_gauss'] = share_gauss
        row['auc_dir_resid'] = direction_auc(R[S], y[S], R[T], y[T])
        e = np.array([eta2(X[:, u], meaning, form) for u in top_units(scores, 10)])
        row['eta2_top10'] = {'meaning': float(e[:, 0].mean()), 'form': float(e[:, 1].mean())}
        layers.append(row)
        for name, (_, yf) in fields.items():
            sf = unit_scores(X[S], yf[S])
            factors[name]['acc_10'].append(centroid_accuracy(X[S], yf[S], X[T], yf[T], top_units(sf, 10)))

    D = mean_diff(post[S, L - 1], y[S])
    terms, emb = write_contributions(pre[T], mid[T], post[T], y[T], D)
    attn, mlp = [a for a, _ in terms], [m for _, m in terms]
    best = int(np.argmax([r['acc_10'] for r in layers]))  # first maximum = lower layer on ties
    scores = unit_scores(hidden[S, best], y[S])
    selected = [{'unit': int(u), 'score': float(scores[u])} for u in top_units(scores, 8, sign=1)]
    return {'target': target, 'ks': list(ks), 'n': n,
            'forms_S': sorted(set(form[S].tolist())), 'forms_T': sorted(set(form[T].tolist())),
            'layers': layers, 'writes': {'embed': emb, 'attn': attn, 'mlp': mlp},
            'attn_share': float(sum(attn) / (sum(attn) + sum(mlp))),
            'best_layer': best, 'selected': selected, 'factors': factors}


def table(res):
    cols = [f'acc_{k}' for k in res['ks']] + ['auc_dir_hidden', 'auc_dir_resid', 'share_top1pct', 'share_gauss']
    out = ['layer ' + ' '.join(f'{c:>14}' for c in cols)]
    for r in res['layers']:
        out.append(f"{r['layer']:5d} " + ' '.join(f'{r[c]:14.3f}' for c in cols))
    out.append(f"attn_share {res['attn_share']:.3f}   best_layer {res['best_layer']}")
    out.append('selected ' + ', '.join(f"{s['unit']} ({s['score']:+.3f})" for s in res['selected']))
    return '\n'.join(out)


def _close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, (a, b)


def _synthetic(L=3, H=200, d=6, per_half=10, emb=1.0, attn=(2.0, 0.0, 1.0), mlp=(-1.0, 3.0, 0.0), seed=0):
    """full grid of 2 * per_half forms x 3 meanings. hidden: layer 0 noise; layer 1 only unit 7 = 5 * target;
    layer 2 unit 11 = 3 * (voice active, form-only) and unit 13 = 1 for meanings postponed and m1.
    residual: target rows get emb / attn[l] / mlp[l] along e_0; every other write is form-only and orthogonal to e_0."""
    rng = np.random.default_rng(seed)
    forms = [f for f in range(1000) if half(f) == 0][:per_half] + [f for f in range(1000) if half(f) == 1][:per_half]
    voice = {f: 'passive' if i % 4 == 0 else 'active' for i, f in enumerate(forms)}
    pairs = [(f, m) for f in forms for m in ('postponed', 'm1', 'm2')]
    sent = [{'sid': i, 'form': f, 'meaning': m, 'text': f'{m} {f}', 'voice': voice[f]} for i, (f, m) in enumerate(pairs)]
    n = len(sent)
    y = np.array([s['meaning'] == 'postponed' for s in sent])
    hidden = np.zeros((n, L, H))
    hidden[:, 0] = rng.standard_normal((n, H))
    hidden[:, 1, 7] = 5.0 * y
    hidden[:, 2, 11] = 3.0 * np.array([s['voice'] == 'active' for s in sent])
    hidden[:, 2, 13] = np.array([s['meaning'] in ('postponed', 'm1') for s in sent], float)
    u = np.eye(d)[0]
    fz = {f: rng.standard_normal((1 + 2 * L, d)) * (1 - u) for f in forms}
    pre, mid, post = (np.zeros((n, L, d)) for _ in range(3))
    for i, s in enumerate(sent):
        z, t = fz[s['form']], float(y[i])
        x = z[0] + emb * t * u
        for l in range(L):
            pre[i, l] = x
            x = x + z[1 + 2 * l] + attn[l] * t * u
            mid[i, l] = x
            x = x + z[2 + 2 * l] + mlp[l] * t * u
            post[i, l] = x
    return sent, {'hidden': hidden, 'pre': pre, 'mid': mid, 'post': post}


def selftest():
    r3 = np.sqrt(3)
    X = np.array([[2, 0, 1, 4], [2, 0, 0, 4], [0, 1, 1, 4], [0, 1, 0, 4]], float)
    y = np.array([1, 1, 0, 0], bool)
    assert np.allclose(unit_scores(X, y), [r3, -r3, 0, 0])

    assert list(top_units([1, 3, 3, -5], 2)) == [3, 1]
    assert list(top_units([1, 3, 3, -5], 3, sign=1)) == [1, 2, 0]

    XS, yS = np.array([[0, 7], [0, 7], [2, 7], [2, 7]], float), np.array([0, 0, 1, 1], bool)
    XT, yT = np.array([[3, 9], [0.5, 1], [1.5, 7], [-1, 4]], float), np.array([1, 0, 0, 0], bool)
    _close(centroid_accuracy(XS, yS, XT, yT, [0]), 5 / 6)
    _close(centroid_accuracy(XS, yS, XT, yT, [0, 1]), 5 / 6)  # a column constant on S (sd -> 1) changes nothing
    XS2 = np.array([[10, 1], [10, 1], [0, 0], [0, 0]], float)
    _close(centroid_accuracy(XS2, [1, 1, 0, 0], np.array([[8, 0], [2, 1]], float), [0, 1], [0, 1]), 1.0)  # raw: 0.0

    XS = np.array([[2, 1], [2, 1], [0, 1], [0, 1]], float)
    yS = yT = np.array([1, 1, 0, 0], bool)
    _close(direction_auc(XS, yS, np.array([[3, 50], [1, -50], [2, 9], [0, 0]], float), yT), 0.75)
    _close(direction_auc(XS, yS, np.array([[2, 0], [1, 0], [2, 0], [0, 0]], float), yT), 0.625)

    _close(top_share([3, 4]), 0.64)
    v = np.zeros(200)
    v[5], v[7], v[9] = 3, -4, 1
    _close(top_share(v), 25 / 26)
    _close(top_share(np.ones(700)), 0.01)
    _close(top_share(np.ones(300)), 0.01)

    _close(gaussian_share(1), 1.0)
    assert gaussian_share(500, seed=3) == gaussian_share(500, seed=3)
    c = stats.norm.ppf(0.995)
    assert abs(gaussian_share(20000) - 2 * (c * stats.norm.pdf(c) + stats.norm.sf(c))) < 0.005  # ~0.0845

    e = eta2([1, 3, 5, 7], ['a', 'a', 'b', 'b'], [0, 1, 0, 1])
    _close(e[0], 0.8)
    _close(e[1], 0.2)
    assert eta2([2, 2, 2, 2], ['a', 'a', 'b', 'b'], [0, 1, 0, 1]) == (0.0, 0.0)

    sent, rec = _synthetic()
    yall = np.array([s['meaning'] == 'postponed' for s in sent])
    terms, emb = write_contributions(rec['pre'], rec['mid'], rec['post'], yall, [2, 0, 0, 0, 0, 0])
    _close(emb, 1.0)
    for (a, m), (ha, hm) in zip(terms, [(2, -1), (0, 3), (1, 0)]):
        _close(a, ha)
        _close(m, hm)
    bad = rec['pre'].copy()
    bad[yall, 1, 0] += 0.5
    fired = False
    try:
        write_contributions(bad, rec['mid'], rec['post'], yall, [1, 0, 0, 0, 0, 0])
    except AssertionError:
        fired = True
    assert fired, 'write identity check did not fire'

    res = analyze(sent, rec, 'postponed')
    l1 = res['layers'][1]
    for key in ('acc_1', 'acc_10', 'acc_100', 'auc_dir_hidden', 'share_top1pct'):
        _close(l1[key], 1.0)
    _close(l1['eta2_top10']['meaning'], 0.1)
    _close(l1['eta2_top10']['form'], 0.0)
    assert res['layers'][0]['acc_10'] < 1
    _close(res['layers'][2]['acc_10'], 0.75)
    g = gaussian_share(200)
    for r in res['layers']:
        _close(r['auc_dir_resid'], 1.0)
        assert r['share_gauss'] == g
    assert res['best_layer'] == 1
    assert [s['unit'] for s in res['selected']] == [7, 0, 1, 2, 3, 4, 5, 6]
    _close(res['selected'][0]['score'], 5 / np.sqrt(1500 / 9 / 29))
    assert all(s['score'] == 0 for s in res['selected'][1:])
    _close(res['writes']['embed'], 1.0)
    for got, want in zip(res['writes']['attn'] + res['writes']['mlp'], [2, 0, 1, -1, 3, 0]):
        _close(got, want)
    _close(res['attn_share'], 0.6)
    assert list(res['factors']) == ['voice'] and res['factors']['voice']['value'] == 'active'
    _close(res['factors']['voice']['acc_10'][1], 0.5)
    _close(res['factors']['voice']['acc_10'][2], 1.0)
    json.dumps(res)
    table(res)
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--record', help='directory with sentences.json and record.npz')
    ap.add_argument('--target', default='postponed')
    ap.add_argument('--readout', default='suffix', choices=['suffix', 'text'])
    ap.add_argument('--out', help='JSON output file')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    assert a.record and a.out, '--record and --out are required'
    with open(os.path.join(a.record, 'sentences.json')) as f:
        sent = json.load(f)
    npz = np.load(os.path.join(a.record, 'record.npz'))
    rec = {k: npz[f'{k}_{a.readout}'] for k in ARRAYS}
    res = {'record': a.record, 'readout': a.readout, **analyze(sent, rec, a.target)}
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(table(res))
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
