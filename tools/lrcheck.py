"""Left- vs right-facing within kind, for every channel of a layer.

For each labelled photo of runs/views/labels.json (labels made by a model that never saw any activation) and its
left-right mirror: the spatial max of every channel of each layer. Per layer, channel and kind (D, V), on the
originals: v = value / (the channel's max over all labelled originals of both kinds); d = (mean v over left - mean v
over right) / (sd of v over that kind's left and right photos together, ddof 1), 0 when the sd is 0. d_mirror: the
same on the mirrors, each photo keeping its original label. |d| percentile = 100 x share of the layer's channels with
|d| below the channel's. Spec: authors/spec_lrcheck.md. Outputs: runs/lrcheck/.

usage: python -I tools/lrcheck.py [--layers mixed4b,mixed4c,mixed4d,mixed4e] [--named ...] [--selftest]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
OUT = J / 'runs/lrcheck'
LABELS = J / 'runs/views/labels.json'
WEIGHTS = J / 'raw/weights/inception5h.pth'
BATCH = 64
KINDS = ('D', 'V')
LEFT = frozenset({'left', 'toward_left', 'away_left'})
RIGHT = frozenset({'right', 'toward_right', 'away_right'})
UNUSED = frozenset({'toward', 'away', 'unclear'})  # the other facing values in labels.json
ORDER = ('mixed3a', 'mixed3b', 'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b')
NCH = {'mixed3a': 256, 'mixed3b': 480, 'mixed4b': 512, 'mixed4c': 512, 'mixed4d': 528, 'mixed4e': 832}  # bundlemax
DEFAULT_LAYERS = 'mixed4b,mixed4c,mixed4d,mixed4e'
DEFAULT_NAMED = 'mixed4b:443,431,429,384,409,418,237,373,mixed4c:254,407,447,mixed4d:409,mixed4e:55'
LARGE, SMALL = 0.8, 0.2
COLS = ['unit'] + [f'{c}_{k}' for k in KINDS for c in ('d', 'absd_pct', 'd_mirror')]


# ---------- statistic ----------

def side_of(facing):
    """'left', 'right' or None (any other facing value, or not a string)"""
    if isinstance(facing, str):
        if facing in LEFT:
            return 'left'
        if facing in RIGHT:
            return 'right'
    return None


def d_stat(v, sides):
    """(mean v over 'left' - mean v over 'right') / sd (ddof 1) of v over the left and right entries together;
    0 when that sd is 0; None entries ignored; nan when there is no left or no right entry"""
    v = np.asarray(v, dtype=np.float64).ravel()
    sides = list(sides)
    if len(sides) != len(v):
        raise ValueError(f'{len(v)} values, {len(sides)} sides')
    bad = [s for s in sides if s not in ('left', 'right', None)]
    if bad:
        raise ValueError(f'sides must be left, right or None: {bad[:3]}')
    lv = v[np.array([i for i, s in enumerate(sides) if s == 'left'], dtype=np.int64)]
    rv = v[np.array([i for i, s in enumerate(sides) if s == 'right'], dtype=np.int64)]
    if not len(lv) or not len(rv):
        return float('nan')
    both = np.concatenate([lv, rv])
    if np.all(both == both[0]):
        return 0.0
    sd = float(both.std(ddof=1))
    if sd == 0:
        return 0.0
    return float((lv.mean() - rv.mean()) / sd)


def pct_abs(d_all, d):
    """100 x share of entries of d_all with |entry| < |d| (nan when d is nan or d_all is empty)"""
    a = np.abs(np.asarray(d_all, dtype=np.float64).ravel())
    if not a.size or np.isnan(d):
        return float('nan')
    return float(100.0 * np.mean(a < abs(float(d))))


def absd_summary(d):
    """median, 90th and 95th percentile of |d| over the non-nan entries"""
    a = np.abs(np.asarray(d, dtype=np.float64).ravel())
    a = a[~np.isnan(a)]
    if not a.size:
        return (float('nan'),) * 3
    return tuple(float(x) for x in np.percentile(a, [50, 90, 95]))


def layer_stats(X, Xm, kinds, sides):
    """X, Xm: (labels, channels) spatial maxes of the originals and the mirrors, rows in label order. Per kind:
    d, d_mirror and |d| percentile per channel; also the number of channels whose max over the originals is <= 0
    (for those the values are left unscaled, which leaves d unchanged since d does not depend on a positive scale)"""
    X, Xm = np.asarray(X, dtype=np.float64), np.asarray(Xm, dtype=np.float64)
    kinds = np.asarray(kinds)
    assert X.shape == Xm.shape and X.shape[0] == len(kinds) == len(sides), (X.shape, Xm.shape, len(kinds), len(sides))
    M = X.max(axis=0)
    s = np.where(M > 0, M, 1.0)
    V, Vm = X / s, Xm / s
    out = {}
    for k in KINDS:
        r = np.flatnonzero(kinds == k)
        sd = [sides[i] for i in r]
        d = np.array([d_stat(V[r, c], sd) for c in range(X.shape[1])])
        dm = np.array([d_stat(Vm[r, c], sd) for c in range(X.shape[1])])
        out[k] = {'d': d, 'd_mirror': dm, 'pct': np.array([pct_abs(d, x) for x in d])}
    return out, int((M <= 0).sum())


def unit_values(st, c):
    """d, |d| percentile, d_mirror per kind for channel c, in COLS order"""
    return [float(st[k][key][c]) for k in KINDS for key in ('d', 'pct', 'd_mirror')]


def decide(d):
    """Neutral label on |d|: >= 0.8 LARGE, < 0.2 SMALL, else MEDIUM; nan UNDEFINED."""
    if np.isnan(d):
        return 'UNDEFINED'
    return 'LARGE' if abs(d) >= LARGE else 'SMALL' if abs(d) < SMALL else 'MEDIUM'


# ---------- arguments, labels ----------

def parse_layers(s):
    out = []
    for t in s.split(','):
        t = t.strip()
        if not t:
            continue
        if t not in ORDER:
            raise ValueError(f'bad layer {t!r}')
        if t not in out:
            out.append(t)
    if not out:
        raise ValueError('no layers')
    return out


def parse_named(s):
    """'mixed4b:443,431,mixed4c:254' -> [('mixed4b', 443), ('mixed4b', 431), ('mixed4c', 254)]"""
    out, layer = [], None
    for t in s.split(','):
        t = t.strip()
        if not t:
            continue
        if ':' in t:
            layer, t = t.split(':', 1)
            if layer not in ORDER:
                raise ValueError(f'bad layer {layer!r}')
        if layer is None:
            raise ValueError(f'channel {t!r} before any layer')
        c = int(t)
        if c < 0 or (layer in NCH and c >= NCH[layer]):
            raise ValueError(f'bad channel {layer}:{c}')
        out.append((layer, c))
    return out


def read_rows(obj):
    """labels.json object -> (photo ids, kinds, facings, sides), one entry per label, in file order"""
    labs = obj['labels']
    ids = np.array([int(lab['photo']) for lab in labs], dtype=np.int64)
    kinds = np.array([lab['kind'] for lab in labs], dtype=object)
    bad = sorted(set(kinds.tolist()) - set(KINDS))
    if bad:
        raise ValueError(f'kind not D or V: {bad}')
    facings = [lab.get('facing') for lab in labs]
    return ids, kinds, facings, [side_of(f) for f in facings]


def deepest(layers):
    return max(layers, key=ORDER.index)


# ---------- network ----------

def _modules():
    for p in (str(J / 'tools'), str(J / 'bank')):
        if p not in sys.path:
            sys.path.insert(0, p)
    import bundlemax
    import img_acts
    return bundlemax, img_acts


def to_input(imgs):
    """as bundlemax.net_input, from PIL images: pixels (0-255) - 117, NCHW"""
    import torch
    x = torch.from_numpy(np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in imgs])).permute(0, 3, 1, 2)
    return x * 255 - 117


def measure(model, dev, ids, layers):
    """spatial max of every channel of each layer for photo ids and their mirrors; one forward pass per batch
    hooking all layers, stopped after the deepest -> ({layer: (n, C)}, {layer: (n, C)})"""
    import torch
    from PIL import ImageOps
    bm, img_acts = _modules()
    Stop = img_acts.Stop
    last = deepest(layers)
    got = {}

    def hook(name):
        def f(_m, _i, out):
            if name in NCH:
                assert out.shape[1] == NCH[name], (name, tuple(out.shape))
            got[name] = out.amax(dim=(2, 3)).float().cpu().numpy()
            if name == last:
                raise Stop
        return f

    res = {'orig': {l: [] for l in layers}, 'mirror': {l: [] for l in layers}}
    hs = [getattr(model, l).register_forward_hook(hook(l)) for l in layers]
    try:
        for s in range(0, len(ids), BATCH):
            chunk = ids[s:s + BATCH]
            imgs = [bm.photo(i) for i in chunk]
            if s == 0:
                assert torch.equal(to_input(imgs), bm.net_input(chunk)), 'input differs from bundlemax.net_input'
            for key, ims in (('orig', imgs), ('mirror', [ImageOps.mirror(im) for im in imgs])):
                got.clear()
                try:
                    with torch.no_grad():
                        model(to_input(ims).to(dev))
                except Stop:
                    pass
                assert set(got) == set(layers), (sorted(got), layers)
                for l in layers:
                    res[key][l].append(got[l])
            print(f'  {min(s + BATCH, len(ids))}/{len(ids)} photos', flush=True)
    finally:
        for h in hs:
            h.remove()
    o = {l: np.concatenate(res['orig'][l]).astype(np.float32) for l in layers}
    m = {l: np.concatenate(res['mirror'][l]).astype(np.float32) for l in layers}
    for l in layers:
        assert o[l].shape == m[l].shape and o[l].shape[0] == len(ids), (l, o[l].shape, m[l].shape)
    return o, m


def values(ids, layers, get_model):
    """cached in runs/lrcheck/values.npz: ids, <layer> (originals) and <layer>_mirror, rows in ids order"""
    f = OUT / 'values.npz'
    have = {}
    if f.exists():
        with np.load(f) as z:
            if 'ids' in z.files and np.array_equal(z['ids'], ids):
                have = {k: z[k] for k in z.files if k != 'ids'}
    missing = [l for l in layers if l not in have or l + '_mirror' not in have]
    if missing:
        print('measuring', ','.join(missing), flush=True)
        model, dev = get_model()
        o, m = measure(model, dev, ids, missing)
        for l in missing:
            have[l], have[l + '_mirror'] = o[l], m[l]
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, ids=ids, **have)
    else:
        print('values: cached', flush=True)
    return have


# ---------- report ----------

def fmt(x, spec='+.3f'):
    return 'nan' if np.isnan(x) else format(x, spec)


def line(name, vals):
    p = [fmt(x, '+.3f' if i % 3 != 1 else '.1f') for i, x in enumerate(vals)]
    return f'{name:<13}' + ''.join(f'{x:>11}' for x in p)


def report(ids, kinds, facings, sides, layers, vals, stats, nonpos, named):
    nu = len(np.unique(ids))
    L = [f'labels {len(ids)} ({nu} distinct photos) from runs/views/labels.json',
         'left = ' + ', '.join(sorted(LEFT)) + '; right = ' + ', '.join(sorted(RIGHT)) + '; other facings unused']
    for k in KINDS:
        r = np.flatnonzero(kinds == k)
        nl = sum(sides[i] == 'left' for i in r)
        nr = sum(sides[i] == 'right' for i in r)
        L.append(f'kind {k}: left {nl}, right {nr}, unused {len(r) - nl - nr}')
    odd = sorted({repr(f) for f in facings if not (isinstance(f, str) and f in LEFT | RIGHT | UNUSED)})
    if odd:
        L.append('unrecognised facing values (unused): ' + ', '.join(odd))

    L += ['', 'per layer, |d| over channels: median / 90th / 95th percentile']
    for l in layers:
        extra = f', {nonpos[l]} with max <= 0' if nonpos[l] else ''
        parts = [f'{k}: ' + ' / '.join(fmt(x, '.3f') for x in absd_summary(stats[l][k]['d'])) for k in KINDS]
        L.append(f'{l} ({vals[l].shape[1]} channels{extra})   ' + '   '.join(parts))

    hdr = ['d', '|d|pct', 'd_mirror']
    L += ['', 'named units', f'{"unit":<13}' + ''.join(f'{h + "_" + k:>11}' for k in KINDS for h in hdr)]
    rows = []
    for l, c in named:
        name = f'{l}:{c}'
        if l not in stats:
            L.append(f'{name:<13}  (layer not in --layers)')
            continue
        if c >= vals[l].shape[1]:
            raise ValueError(f'{name}: layer has {vals[l].shape[1]} channels')
        v = unit_values(stats[l], c)
        rows.append((name, v))
        L.append(line(name, v))

    L += ['', f'decision on |d| per kind: |d| >= {LARGE} LARGE, {SMALL} <= |d| < {LARGE} MEDIUM, |d| < {SMALL} SMALL, '
              'nan UNDEFINED']
    for name, v in rows:
        L.append(f'{name:<13}' + ''.join(f'  {k} {decide(v[3 * j])}' for j, k in enumerate(KINDS)))
    return '\n'.join(L) + '\n'


def table(layers, vals, stats):
    out = [','.join(COLS)]
    for l in layers:
        for c in range(vals[l].shape[1]):
            out.append(','.join([f'{l}:{c}'] + [f'{x:.6f}' for x in unit_values(stats[l], c)]))
    return '\n'.join(out) + '\n'


# ---------- selftest ----------

def selftest():
    from PIL import Image, ImageOps
    s3 = np.sqrt(3.0)

    # side_of on every facing value of labels.json, and others
    for f in ('left', 'toward_left', 'away_left'):
        assert side_of(f) == 'left', f
    for f in ('right', 'toward_right', 'away_right'):
        assert side_of(f) == 'right', f
    for f in ('toward', 'away', 'unclear', None, '', 'Left', 'RIGHT', ' left', 'leftish', 3, ['left']):
        assert side_of(f) is None, f
    assert LEFT | RIGHT | UNUSED == {'left', 'toward_left', 'away_left', 'right', 'toward_right', 'away_right',
                                     'toward', 'away', 'unclear'}

    # d_stat: sign
    assert abs(d_stat([3, 3, 1, 1], ['left', 'left', 'right', 'right']) - s3) < 1e-12
    assert abs(d_stat([3, 3, 1, 1], ['right', 'right', 'left', 'left']) + s3) < 1e-12
    assert abs(d_stat(np.array([1.0, 2.0, 4.0]), ('left', 'right', 'right')) + 2 / np.sqrt(7 / 3)) < 1e-12
    assert abs(d_stat([5, 1, 2, 3], ['left', 'right', 'right', 'right']) - 3 / np.sqrt(8.75 / 3)) < 1e-12
    # ignored entries
    assert abs(d_stat([3, 100, 3, 1, -50, 1], ['left', None, 'left', 'right', None, 'right']) - s3) < 1e-12
    rng = np.random.default_rng(0)
    v = rng.normal(size=40)
    sides = [('left', 'right', None)[i % 3] for i in range(40)]
    lv, rv = v[0::3], v[1::3]
    want = (lv.mean() - rv.mean()) / np.concatenate([lv, rv]).std(ddof=1)
    assert abs(d_stat(v, sides) - want) < 1e-12
    assert abs(d_stat(v * 7.5, sides) - want) < 1e-12, 'd must not depend on a positive scale'
    # zero sd
    assert d_stat([2, 2, 2, 2], ['left', 'right', 'left', 'right']) == 0.0
    assert d_stat([0.1] * 5 + [7.0], ['left', 'right', 'left', 'right', 'left', None]) == 0.0
    assert d_stat([0, 0], ['left', 'right']) == 0.0
    # no left or no right
    assert np.isnan(d_stat([1, 2], ['left', 'left'])) and np.isnan(d_stat([1, 2], [None, None]))
    assert np.isnan(d_stat([], []))
    for bad_v, bad_s in (([1, 2, 3], ['left', 'right']), ([1, 2], ['left', 'toward'])):
        try:
            d_stat(bad_v, bad_s)
        except ValueError:
            continue
        raise AssertionError(f'accepted {bad_s}')

    # pct_abs
    da = [-3.0, -1.0, 0.0, 2.0, 5.0]
    assert pct_abs(da, 2.0) == 40.0 and pct_abs(da, -2.0) == 40.0 and pct_abs(da, -3.0) == 60.0
    assert pct_abs(da, 0.0) == 0.0 and pct_abs(da, 5.0) == 80.0 and pct_abs(da, 10.0) == 100.0
    assert np.isnan(pct_abs(da, float('nan'))) and np.isnan(pct_abs([], 1.0))

    # percentiles of |d|
    d = np.arange(101.0)
    d[::2] *= -1
    assert absd_summary(d) == (50.0, 90.0, 95.0)
    assert all(np.isnan(x) for x in absd_summary([float('nan')]))

    # per-layer statistic: kinds, ignored labels, mirrors keep the original label, a channel with max 0;
    # channel 2 is channel 0 scaled by 4 (a power of 2, so its d is bit-identical to channel 0's)
    kinds = np.array(['D', 'D', 'D', 'D', 'V', 'V', 'V', 'V', 'D'], dtype=object)
    sides = ['left', 'left', 'right', 'right', 'left', 'right', 'left', 'right', None]
    c0 = np.array([4, 4, 2, 2, 1, 1, 1, 1, 9.0])
    c3 = np.array([1, 1, 1, 1, 3, 1, 3, 1, 0.0])
    X = np.stack([c0, np.zeros(9), 4 * c0, c3], axis=1)
    m0 = np.array([2, 2, 4, 4, 1, 1, 1, 1, 9.0])
    Xm = np.stack([m0, np.array([0, 0, 5, 5, 0, 0, 0, 0, 0.0]), 4 * m0, c3], axis=1)
    st, nonpos = layer_stats(X, Xm, kinds, sides)
    assert nonpos == 1
    for k, key, want in (('D', 'd', [s3, 0, s3, 0]), ('V', 'd', [0, 0, 0, s3]),
                         ('D', 'd_mirror', [-s3, -s3, -s3, 0]), ('V', 'd_mirror', [0, 0, 0, s3]),
                         ('D', 'pct', [50, 0, 50, 0]), ('V', 'pct', [0, 0, 0, 75])):
        assert np.allclose(st[k][key], want, atol=1e-9), (k, key, st[k][key])
    assert unit_values(st, 3) == [0.0, 0.0, 0.0, float(st['V']['d'][3]), 75.0, float(st['V']['d_mirror'][3])]
    assert len(COLS) == 7 and COLS[0] == 'unit'

    # mirror is a left-right flip
    a = rng.integers(0, 256, size=(4, 6, 3), dtype=np.uint8)
    assert np.array_equal(np.asarray(ImageOps.mirror(Image.fromarray(a))), a[:, ::-1])

    # labels, layers, named units
    obj = {'photos': [5, 9, 7], 'labels': [{'photo': 5, 'kind': 'D', 'facing': 'toward_left'},
                                           {'photo': 9, 'kind': 'V', 'facing': 'away'},
                                           {'photo': 7, 'kind': 'D', 'facing': 'away_right'}]}
    ids, ks, fs, sd = read_rows(obj)
    assert ids.tolist() == [5, 9, 7] and ks.tolist() == ['D', 'V', 'D'] and sd == ['left', None, 'right']
    assert fs == ['toward_left', 'away', 'away_right']
    try:
        read_rows({'labels': [{'photo': 1, 'kind': 'X', 'facing': 'left'}]})
        raise AssertionError('kind X accepted')
    except ValueError:
        pass
    assert parse_layers(DEFAULT_LAYERS) == ['mixed4b', 'mixed4c', 'mixed4d', 'mixed4e']
    assert parse_layers(' mixed4c, mixed4b,mixed4c ') == ['mixed4c', 'mixed4b']
    assert deepest(['mixed4e', 'mixed4b', 'mixed4d']) == 'mixed4e' and deepest(['mixed4c']) == 'mixed4c'
    nm = parse_named(DEFAULT_NAMED)
    assert nm == ([('mixed4b', c) for c in (443, 431, 429, 384, 409, 418, 237, 373)]
                  + [('mixed4c', c) for c in (254, 407, 447)] + [('mixed4d', 409), ('mixed4e', 55)]), nm
    for bad in ('mixed4z', ''):
        try:
            parse_layers(bad)
        except ValueError:
            continue
        raise AssertionError('accepted layers ' + repr(bad))
    for bad in ('443', 'mixed9z:1', 'mixed4b:512', 'mixed4b:x', 'mixed4b:-1', 'mixed4e:832'):
        try:
            parse_named(bad)
        except ValueError:
            continue
        raise AssertionError('accepted named ' + bad)

    # decision labels
    assert decide(0.8) == 'LARGE' and decide(-1.5) == 'LARGE' and decide(0.79) == 'MEDIUM'
    assert decide(-0.2) == 'MEDIUM' and decide(0.19) == 'SMALL' and decide(0.0) == 'SMALL'
    assert decide(float('nan')) == 'UNDEFINED'
    assert fmt(float('nan')) == 'nan' and fmt(0.5) == '+0.500'
    print('selftest ok')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', default=DEFAULT_LAYERS)
    ap.add_argument('--named', default=DEFAULT_NAMED)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    layers = parse_layers(a.layers)
    named = parse_named(a.named)
    ids, kinds, facings, sides = read_rows(json.loads(LABELS.read_text()))
    uniq = np.unique(ids)
    row = np.searchsorted(uniq, ids)
    held = []

    def get_model():
        if not held:
            import torch
            torch.set_num_threads(4)
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            m = _modules()[1].load_model(WEIGHTS)
            for p in m.parameters():
                p.requires_grad_(False)
            held.append((m.to(dev), dev))
            print('model on', dev, flush=True)
        return held[0]

    vals = values(uniq, layers, get_model)
    stats, nonpos = {}, {}
    for l in layers:
        stats[l], nonpos[l] = layer_stats(vals[l][row], vals[l + '_mirror'][row], kinds, sides)
    rep = report(ids, kinds, facings, sides, layers, vals, stats, nonpos, named)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.txt').write_text(rep)
    (OUT / 'table.csv').write_text(table(layers, vals, stats))
    print(rep, flush=True)
    print('report ->', OUT / 'report.txt')
    print('table  ->', OUT / 'table.csv')


if __name__ == '__main__':
    main()
