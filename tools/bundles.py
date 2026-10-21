"""Shared or separate input channels: input-layer attribution of one unit on two kinds of photo.

Units: mixed4e:N / mixed4d:N is that channel's post-ReLU 14x14 module output; sum:mixed4e:A+B is a planted unit,
map = map_A / s_A + map_B / s_B, s_c = channel c's largest spatial max over all photos. The input layer is the
previous module (mixed4d for mixed4e, mixed4c for mixed4d). Photo kind comes from its class (bank/img/acts.npz).
Top 150 photos per unit by spatial max of its map. v = map at its argmax location; A_i = sum over positions of
dv/dx * x for input channel i (x = input-layer output as a leaf), unit-L2 normalised.
ratio = cos(mean A over kind A, mean A over kind B) / mean(split-half cos within A, within B).
Decision on ratio: < 0.50 SEPARATE, > 0.80 SHARED, else INTERMEDIATE (NaN UNDEFINED).
Bundles: kmeans2 (seed 0, '++') for k = 2, 3, 4: cluster x kind table, NMI over A and B photos, top-6 input channels.
Input range follows tools/battery.py: pixels (0-255) - 117. Outputs go to runs/bundles/.

usage: python -I tools/bundles.py --units mixed4d:409,sum:mixed4e:809+804 --kinds "A=151-268/B=407,436;A=.../B=..."
       python -I tools/bundles.py --selftest
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2

J = Path(__file__).resolve().parent.parent
OUT = J / 'runs/bundles'
TOP = 150
BATCH = 64
GBATCH = 16
KS = (2, 3, 4)
ORDER = ('mixed4c', 'mixed4d', 'mixed4e')
PREV = {'mixed4d': 'mixed4c', 'mixed4e': 'mixed4d'}
NCH = {'mixed4c': 512, 'mixed4d': 528, 'mixed4e': 832}
KINDS = ('A', 'B', 'other')


def decide(r):
    """Neutral label on the ratio: < 0.50 SEPARATE, > 0.80 SHARED, else INTERMEDIATE."""
    if np.isnan(r):
        return 'UNDEFINED'
    return 'SEPARATE' if r < 0.50 else 'SHARED' if r > 0.80 else 'INTERMEDIATE'


def parse_unit(u):
    """'mixed4e:N' -> ('mixed4e', [N], False); 'sum:mixed4e:A+B' -> ('mixed4e', [A, B], True)."""
    p = u.strip().split(':')
    if p[0] == 'sum':
        layer, chans, planted = p[1], [int(c) for c in p[2].split('+')], True
        assert len(chans) == 2, u
    else:
        layer, chans, planted = p[0], [int(p[1])], False
    assert layer in PREV, u
    return layer, chans, planted


def parse_classes(v):
    """'151-268' -> {151, ..., 268} (inclusive); '407,436' -> {407, 436}."""
    s = set()
    for t in v.split(','):
        lo, _, hi = t.strip().partition('-')
        s.update(range(int(lo), int(hi or lo) + 1))
    return s


def parse_kinds(text):
    """'A=lo-hi/B=i,j;A=.../B=...' -> [{'A': set, 'B': set}, ...], one per unit."""
    out = []
    for part in text.split(';'):
        d = {}
        for kv in part.split('/'):
            k, v = kv.split('=')
            d[k.strip()] = parse_classes(v)
        assert set(d) == {'A', 'B'} and not d['A'] & d['B'], part
        out.append(d)
    return out


def kind_of(cls, kd):
    return np.array(['A' if c in kd['A'] else 'B' if c in kd['B'] else 'other' for c in np.asarray(cls).tolist()])


def scale(m):
    """s_c: largest spatial max over all photos of one channel's maps (n, 14, 14)."""
    return float(m.max(axis=(1, 2)).max())


def unit_map(ms, w):
    return sum(wc * m for m, wc in zip(ms, w))


def unit_l2(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.where(n > 0, n, 1)


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else float('nan')


def mean(A, idx):
    return A[idx].mean(axis=0) if len(idx) else np.full(A.shape[1], np.nan)


def halves(idx):
    """Photos of one kind, in rank order, alternately assigned to two halves."""
    return idx[0::2], idx[1::2]


def score(A, kn):
    ia, ib = np.flatnonzero(kn == 'A'), np.flatnonzero(kn == 'B')
    between = cos(mean(A, ia), mean(A, ib))
    within = [cos(*(mean(A, h) for h in halves(i))) for i in (ia, ib)]
    return dict(between=between, within_A=within[0], within_B=within[1], ratio=float(between / np.mean(within)),
                nA=len(ia), nB=len(ib), nO=len(kn) - len(ia) - len(ib))


def table(lab, kn, k):
    T = np.zeros((k, len(KINDS)), dtype=np.int64)
    for c, q in zip(lab, kn):
        T[c, KINDS.index(q)] += 1
    return T


def ent(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def nmi(T):
    """NMI of a contingency table, arithmetic-mean normalisation; NaN if undefined."""
    T = np.asarray(T, dtype=np.float64)
    if T.sum() == 0:
        return float('nan')
    P = T / T.sum()
    pr, pc = P.sum(axis=1), P.sum(axis=0)
    nz = P > 0
    mi = float((P[nz] * np.log(P[nz] / np.outer(pr, pc)[nz])).sum())
    h = (ent(pr) + ent(pc)) / 2
    return mi / h if h > 0 else float('nan')


def _bank():
    p = str(J / 'bank')
    if p not in sys.path:
        sys.path.insert(0, p)
    import img_acts
    return img_acts


def load():
    import torch
    torch.set_num_threads(4)
    m = _bank().load_model(J / 'raw/weights/inception5h.pth')
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def photo(i):
    with Image.open(J / 'bank/img/thumbs' / f'{int(i):05d}.jpg') as im:
        img = im.convert('RGB')
    assert img.size == (224, 224), (i, img.size)
    return img


def inputs(ids):
    import torch
    x = torch.from_numpy(np.stack([np.asarray(photo(i), dtype=np.float32) / 255.0 for i in ids])).permute(0, 3, 1, 2)
    return x * 255 - 117


def maps(model, n, need):
    """{(layer, c): (n, 14, 14) float32} post-ReLU module-output maps of the needed channels on photos 0..n-1."""
    import torch
    Stop = _bank().Stop
    last = max(need, key=ORDER.index)
    got = {}

    def keep(layer):
        def f(_m, _i, out):
            assert tuple(out.shape[1:]) == (NCH[layer], 14, 14), (layer, tuple(out.shape))
            got[layer] = out[:, need[layer]].numpy().astype(np.float32)
            if layer == last:
                raise Stop
        return f

    hs = [getattr(model, l).register_forward_hook(keep(l)) for l in need]
    res = {l: [] for l in need}
    try:
        for s in range(0, n, BATCH):
            got.clear()
            try:
                with torch.no_grad():
                    model(inputs(range(s, min(s + BATCH, n))))
            except Stop:
                pass
            for l in need:
                res[l].append(got[l])
            if s % (BATCH * 32) == 0:
                print('maps', s, '/', n, flush=True)
    finally:
        for h in hs:
            h.remove()
    out = {}
    for l, cs in need.items():
        a = np.concatenate(res[l])
        for j, c in enumerate(cs):
            out[l, c] = a[:, j]
    return out


def attrib(model, layer, chans, w, ids):
    """Raw A (len(ids), C_in) = sum over positions of dv/dx * x, and v (len(ids),); v = unit map at its argmax."""
    import torch
    Stop = _bank().Stop
    got = {}

    def leaf(_m, _i, out):
        got['x'] = out.detach().requires_grad_(True)
        return got['x']

    def keep(_m, _i, out):
        got['y'] = out
        raise Stop

    hs = [getattr(model, PREV[layer]).register_forward_hook(leaf), getattr(model, layer).register_forward_hook(keep)]
    res, vs = [], []
    try:
        for s in range(0, len(ids), GBATCH):
            got.clear()
            try:
                model(inputs(ids[s:s + GBATCH]))
            except Stop:
                pass
            x, y = got['x'], got['y']
            assert tuple(x.shape[1:]) == (NCH[PREV[layer]], 14, 14), tuple(x.shape)
            assert tuple(y.shape[1:]) == (NCH[layer], 14, 14), tuple(y.shape)
            m = sum(wc * y[:, c] for c, wc in zip(chans, w)).flatten(1)
            v = m.gather(1, m.argmax(dim=1, keepdim=True))[:, 0]
            g, = torch.autograd.grad(v.sum(), x)
            res.append((g * x).sum(dim=(2, 3)).detach().numpy().astype(np.float64))
            vs.append(v.detach().numpy())
    finally:
        for h in hs:
            h.remove()
    return np.concatenate(res), np.concatenate(vs)


def describe(u, layer, s, A, kn):
    sc = score(A, kn)
    L = [f'unit {u}  input={PREV[layer]}' + (f'  s={",".join(f"{x:.4g}" for x in s)}' if s else ''),
         f'  top {len(kn)}: A={sc["nA"]} B={sc["nB"]} other={sc["nO"]}',
         f'  cos_between={sc["between"]:.3f} within_A={sc["within_A"]:.3f} within_B={sc["within_B"]:.3f}'
         f' ratio={sc["ratio"]:.3f} -> {decide(sc["ratio"])}']
    for k in KS:
        _, lab = kmeans2(A, k, seed=0, minit='++')
        T = table(lab, kn, k)
        L.append(f'  k={k}  NMI(cluster, kind) over A,B photos = {nmi(T[:, :2]):.3f}')
        L.append('    cluster    A    B other  top-6 input channels: mean attribution')
        for j in range(k):
            sel = lab == j
            top = '-'
            if sel.any():
                mu = A[sel].mean(axis=0)
                top = ' '.join(f'{c}:{mu[c]:+.3f}' for c in np.argsort(-mu, kind='stable')[:6])
            L.append(f'    {j:7d} {T[j, 0]:4d} {T[j, 1]:4d} {T[j, 2]:5d}  {top}')
    return L


def run(units, kinds):
    OUT.mkdir(parents=True, exist_ok=True)
    n = len(json.loads((J / 'bank/img/names.json').read_text()))
    with np.load(J / 'bank/img/acts.npz') as z:
        classes = z['classes']
    assert len(classes) == n, (len(classes), n)
    us = [parse_unit(u) for u in units]
    need = {}
    for layer, chans, _ in us:
        need.setdefault(layer, set()).update(chans)
    need = {l: sorted(cs) for l, cs in need.items()}
    model = load()
    print('maps: running on', n, 'photos', flush=True)
    M = maps(model, n, need)
    lines = []
    for u, (layer, chans, planted), kd in zip(units, us, kinds):
        s = [scale(M[layer, c]) for c in chans] if planted else []
        w = [1 / x for x in s] if planted else [1.0]
        smax = unit_map([M[layer, c] for c in chans], w).max(axis=(1, 2))
        ids = np.argsort(-smax, kind='stable')[:TOP]
        print(u, ': attributions on', len(ids), 'photos', flush=True)
        A, v = attrib(model, layer, chans, w, ids)
        assert np.allclose(v, smax[ids], rtol=1e-3, atol=1e-4), 'attribution pass map differs from first pass'
        A = unit_l2(A)
        kn = kind_of(classes[ids], kd)
        np.savez(OUT / f'{u}.npz', ids=ids, A=A, kinds=kn)
        lines += describe(u, layer, s, A, kn) + ['']
    text = '\n'.join(lines)
    (OUT / 'report.txt').write_text(text)
    print(text)


def selftest():
    kd = parse_kinds(' A=1-3/B=7,9 ;A=5/B=6-7,10')
    assert kd == [{'A': {1, 2, 3}, 'B': {7, 9}}, {'A': {5}, 'B': {6, 7, 10}}], kd
    assert list(kind_of(np.array([2, 9, 4, 3]), kd[0])) == ['A', 'B', 'other', 'A']
    assert parse_unit('sum:mixed4e:809+804') == ('mixed4e', [809, 804], True)
    assert parse_unit('mixed4d:409') == ('mixed4d', [409], False)

    # sum unit: channel a peaks at 10 (photo 0), channel b at 0.1 (photo 1); normalised, both photos reach 1
    ma, mb = np.zeros((3, 14, 14)), np.zeros((3, 14, 14))
    ma[0, 3, 3], ma[2, 1, 1] = 10, 5
    mb[1, 5, 5], mb[2, 0, 0] = 0.1, 0.02
    s = [scale(ma), scale(mb)]
    assert s == [10, 0.1], s
    m = unit_map([ma, mb], [1 / x for x in s])
    assert np.allclose(m, ma / 10 + mb / 0.1)
    assert np.allclose(m.max(axis=(1, 2)), [1, 1, 0.5]), m.max(axis=(1, 2))

    h0, h1 = halves(np.flatnonzero(np.array(['A', 'B', 'A', 'other', 'A', 'A', 'B']) == 'A'))
    assert list(h0) == [0, 4] and list(h1) == [2, 5], (h0, h1)

    x = np.array([1.0, 2.0, 0.0])
    assert np.isclose(cos(x, 3 * x), 1) and np.isclose(cos(x, -x), -1)
    assert cos(np.array([1.0, 0.0]), np.array([0.0, 2.0])) == 0 and np.isnan(cos(x, 0 * x))

    rng = np.random.default_rng(0)
    n = 40
    a = np.zeros((n, 20))
    a[:, :10] = rng.random((n, 10)) + 0.1
    b = np.zeros((n, 20))
    b[:, 10:] = rng.random((n, 10)) + 0.1
    b1 = np.zeros((n, 20))
    b1[:, :10] = rng.random((n, 10)) + 0.1
    kn = np.array(['A'] * n + ['B'] * n + ['other'] * 3)
    oth = rng.random((3, 20))
    A = unit_l2(np.vstack([a, b, oth]))
    assert np.allclose(np.linalg.norm(A, axis=1), 1)
    r = score(A, kn)
    assert (r['nA'], r['nB'], r['nO']) == (n, n, 3), r
    assert abs(r['ratio']) < 0.05 and r['within_A'] > 0.9 and r['within_B'] > 0.9, r
    r = score(unit_l2(np.vstack([a, b1, oth])), kn)
    assert abs(r['ratio'] - 1) < 0.1, r
    assert decide(0.1) == 'SEPARATE' and decide(0.9) == 'SHARED' and decide(0.6) == 'INTERMEDIATE'
    assert decide(float('nan')) == 'UNDEFINED'

    assert np.isclose(nmi([[5, 0], [0, 7]]), 1) and np.isclose(nmi([[0, 4], [3, 0]]), 1)
    assert abs(nmi([[2, 4], [3, 6]])) < 1e-9
    T = table(np.array([0, 1, 1, 0]), np.array(['A', 'B', 'B', 'other']), 2)
    assert T.tolist() == [[1, 0, 1], [0, 2, 0]], T
    assert np.isclose(nmi(T[:, :2]), 1)
    print('selftest ok')


def arg(name):
    return sys.argv[sys.argv.index(name) + 1]


def main():
    if '--selftest' in sys.argv:
        return selftest()
    units = [u.strip() for u in arg('--units').split(',')]
    kinds = parse_kinds(arg('--kinds'))
    assert len(units) == len(kinds), (len(units), len(kinds))
    run(units, kinds)


if __name__ == '__main__':
    main()
