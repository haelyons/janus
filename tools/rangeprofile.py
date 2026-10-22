"""The invariance profile in three activation strata of each channel (spec: authors/spec_rangeprofile.md).

P0 (the random 1000 card-pool photos of tools/lookviews.py) and, per layer, the spatial max of every channel on them,
originals and under each transform of battery.TRANSFORMS: from the first profile_<layer>.npz among runs/rangeview,
runs/lookviews and runs/sparedview (checked as lookviews.profiles checks a cache), else made by rangeview.profiles in
runs/rangeprofile (on the CPU, as lookviews.profiles runs: it feeds CPU tensors and reads its hooks with .numpy()).
Per channel: its P0 photos with original value above 0, highest first (stable on ties), in strata s1 = ranks 1-20,
s2 = ranks 21-100, s3 = ranks 101-300; a stratum is missing when the channel has fewer such photos than its last rank.
Per layer, channel, transform and stratum: retention = median of t / o, change = median of |t - o| / M (M = the
channel's max of o over P0), each one's percentile among the layer's channels, and a neutral label on the change
percentile. Nothing is scored here. Outputs: runs/rangeprofile/<layer>.csv and runs/rangeprofile/report.txt.

usage: python -I tools/rangeprofile.py [--layers ...] [--units-file authors/round26_units.json] [--selftest]
"""
import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))  # -I does not add the script dir
import battery  # noqa: E402
import bundlemax  # noqa: E402
import lookviews  # noqa: E402
import rangeview  # noqa: E402

OUT = J / 'runs/rangeprofile'
SOURCES = (J / 'runs/rangeview', J / 'runs/lookviews', J / 'runs/sparedview')  # the first that has a layer's profile
UNITS_FILE = 'authors/round26_units.json'
LAYERS = ('mixed3a', 'mixed3b', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a')  # the --layers default
STRATA = ((1, 20), (21, 100), (101, 300))  # (first rank, last rank), both included; rank 1 = the highest original
STATS = ('ret', 'ret_pct', 'chg', 'chg_pct')
PCT_MORE, PCT_LESS = 95.0, 5.0
NAME_W = max(len(n) for n in battery.TRANSFORMS)

REPORT_HEAD = ("Invariance profile in strata of each channel's P0 photos with original value o above 0, highest "
               'first: s1 ranks 1-20, s2 ranks 21-100, s3 ranks 101-300 (n/a: the channel has fewer such photos than '
               "the last rank). ret = median of t / o (t transformed); chg = median of |t - o| / M (M = the channel's "
               "max of o over P0); pct = 100 x the share of the layer's channels with a finite value strictly below. "
               'Label on the chg pct: >= 95 CHANGES_MORE, <= 5 CHANGES_LESS, else TYPICAL; n/a MISSING.')


# ---------- strata and statistics ----------

def strata(o_col, bounds=STRATA):
    """one channel's photos with original value above 0, sorted by it, highest first (stable on ties); per stratum
    (first rank, last rank), 1-based and inclusive: the positions in o_col of its ranks, in rank order, or None when
    fewer photos are above 0 than its last rank"""
    o = np.asarray(o_col, dtype=np.float64)
    assert o.ndim == 1, o.shape
    pos = np.flatnonzero(o > 0)
    order = pos[np.argsort(-o[pos], kind='stable')]
    out = []
    for a, b in bounds:
        assert 1 <= a <= b, (a, b)
        out.append(order[a - 1:b] if len(order) >= b else None)
    return out


def stratum_stats(o_col, t_col, idx):
    """(retention, change) of one channel on one stratum: the medians over the photos idx of t / o and of |t - o| / M,
    with o = o_col, t = t_col and M = the max of o_col (the channel's max over P0); (NaN, NaN) for a missing stratum"""
    o, t = np.asarray(o_col, dtype=np.float64), np.asarray(t_col, dtype=np.float64)
    assert o.ndim == 1 and o.shape == t.shape, (o.shape, t.shape)
    if idx is None or not len(idx):
        return float('nan'), float('nan')
    idx = np.asarray(idx, dtype=np.int64)
    M = float(np.nanmax(o))
    return float(np.median(t[idx] / o[idx])), float(np.median(np.abs(t[idx] - o[idx]) / M))


def layer_stats(orig, trans, bounds=STRATA):
    """orig, trans: (photos, channels), the originals and one transform's values -> (ret, chg), each (channels,
    strata): stratum_stats of every channel on each of its strata"""
    orig, trans = np.asarray(orig, dtype=np.float64), np.asarray(trans, dtype=np.float64)
    assert orig.ndim == 2 and orig.shape == trans.shape, (orig.shape, trans.shape)
    ret = np.full((orig.shape[1], len(bounds)), np.nan)
    chg = np.full((orig.shape[1], len(bounds)), np.nan)
    for c in range(orig.shape[1]):
        for s, idx in enumerate(strata(orig[:, c], bounds)):
            ret[c, s], chg[c, s] = stratum_stats(orig[:, c], trans[:, c], idx)
    return ret, chg


def pct_below(x):
    """x: one value per channel of a layer -> per channel, 100 x the share of the layer's channels (all of x) with a
    finite value strictly below its own; NaN where its own is NaN"""
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 1, x.shape
    fin = np.sort(x[np.isfinite(x)])
    out = np.full(x.shape, np.nan)
    ok = ~np.isnan(x)
    out[ok] = 100.0 * np.searchsorted(fin, x[ok], side='left') / len(x)  # (100 k) / n: one rounding
    return out


def label(p):
    """neutral label on a change percentile: >= 95 CHANGES_MORE, <= 5 CHANGES_LESS, else TYPICAL; NaN MISSING"""
    p = float(p)
    if np.isnan(p):
        return 'MISSING'
    return 'CHANGES_MORE' if p >= PCT_MORE else 'CHANGES_LESS' if p <= PCT_LESS else 'TYPICAL'


def pct_table(a):
    """pct_below of each column (stratum) of a (channels, strata) table"""
    return np.stack([pct_below(a[:, s]) for s in range(a.shape[1])], axis=1)


def layer_table(d, names=None, bounds=STRATA):
    """{transform: {'ret', 'ret_pct', 'chg', 'chg_pct'}}, each (channels, strata), of one layer's profile d ('orig'
    and 't_<transform>'), transforms in battery.TRANSFORMS order"""
    names = list(battery.TRANSFORMS) if names is None else names
    out = {}
    for n in names:
        ret, chg = layer_stats(d['orig'], d[f't_{n}'], bounds)
        out[n] = {'ret': ret, 'ret_pct': pct_table(ret), 'chg': chg, 'chg_pct': pct_table(chg)}
    return out


# ---------- profiles ----------

def parse_layers(s):
    """'mixed4c,mixed3a' -> the layers in network order; each one of lookviews.LAYERS, listed once"""
    out = [x.strip() for x in s.split(',') if x.strip()]
    if not out:
        raise ValueError('no layers')
    bad = [x for x in out if x not in lookviews.LAYERS]
    if bad:
        raise ValueError(f'unknown layers {bad}')
    if len(set(out)) != len(out):
        raise ValueError('a layer is listed twice')
    return sorted(out, key=lookviews.LAYERS.index)


def find_source(layer, srcs=SOURCES):
    """the first dir of srcs that has profile_<layer>.npz, or None"""
    for s in srcs:
        if (Path(s) / f'profile_{layer}.npz').is_file():
            return Path(s)
    return None


def load_profiles(layers, P0, get_model, srcs=SOURCES, out=OUT):
    """{layer: lookviews.profiles' dict}: a layer from the first dir of srcs that has its profile_<layer>.npz, checked
    there by lookviews.profiles (layer, P0, transform list; stops on a mismatch; no model is used); the layers in none
    of them by rangeview.profiles(..., out=out), which uses or makes out/profile_<layer>.npz"""
    prof, todo = {}, []
    for L in layers:
        s = find_source(L, srcs)
        if s is None:
            todo.append(L)
            continue
        print(L, 'profile from', s, flush=True)
        prof[L] = rangeview.profiles([L], P0, None, out=s)[L]
    if todo:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        prof.update(rangeview.profiles(todo, P0, get_model, out=out))
    return {L: prof[L] for L in layers}


def check_profile(layer, d, n_photos):
    """every condition of a layer's profile is (photos, channels of the layer)"""
    want = (n_photos, lookviews.NCH[layer])
    for k in ['orig'] + [f't_{n}' for n in battery.TRANSFORMS]:
        assert k in d and tuple(d[k].shape) == want, (layer, k, tuple(d[k].shape) if k in d else None, want)


# ---------- outputs ----------

def fmt6(x):
    """6 decimals; '' for NaN"""
    x = float(x)
    return '' if np.isnan(x) else f'{x:.6f}'


def csv_columns(names=None, n_strata=len(STRATA)):
    """'unit', then per transform and stratum s: <t>_ret_s<s>, <t>_ret_pct_s<s>, <t>_chg_s<s>, <t>_chg_pct_s<s>"""
    names = list(battery.TRANSFORMS) if names is None else names
    return ['unit'] + [f'{n}_{k}_s{s}' for n in names for s in range(1, n_strata + 1) for k in STATS]


def csv_rows(layer, tab, names=None):
    """one row per channel, in csv_columns order"""
    names = list(battery.TRANSFORMS) if names is None else names
    n_ch, n_s = tab[names[0]]['ret'].shape
    return [[f'{layer}:{c}'] + [fmt6(tab[n][k][c, s]) for n in names for s in range(n_s) for k in STATS]
            for c in range(n_ch)]


def write_csv(path, layer, tab, names=None):
    names = list(battery.TRANSFORMS) if names is None else names
    rows = csv_rows(layer, tab, names)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(csv_columns(names, tab[names[0]]['ret'].shape[1]))
        w.writerows(rows)
    tmp.replace(path)


def num(x, nd):
    x = float(x)
    return 'n/a' if np.isnan(x) else f'{x:.{nd}f}'


def transform_line(name, ret, ret_pct, chg, chg_pct):
    """the transform, then per stratum: retention, its percentile, change, its percentile, the change label"""
    parts = [f's{s} ret {num(r, 3)} pct {num(rp, 1)} chg {num(c, 3)} pct {num(cp, 1)} {label(cp)}'
             for s, (r, rp, c, cp) in enumerate(zip(ret, ret_pct, chg, chg_pct), 1)]
    return f'{name:<{NAME_W}}  ' + ' | '.join(parts)


def unit_section(name, chan, tab, npos, n_photos, names=None, bounds=STRATA):
    """'== unit' with its count of P0 photos above 0 and the layer's count of channels with each stratum; then one
    transform_line per transform. npos: per channel of the layer, its P0 photos above 0"""
    names = list(battery.TRANSFORMS) if names is None else names
    npos = np.asarray(npos)
    have = '/'.join(str(int((npos >= b).sum())) for _, b in bounds)
    head = (f'== {name}  P0 photos above 0: {int(npos[chan])} of {n_photos}; layer channels with '
            + '/'.join(f's{s}' for s in range(1, len(bounds) + 1)) + f': {have} of {len(npos)}')
    return '\n'.join([head] + [transform_line(n, tab[n]['ret'][chan], tab[n]['ret_pct'][chan], tab[n]['chg'][chan],
                                              tab[n]['chg_pct'][chan]) for n in names])


# ---------- selftest ----------

def selftest():
    import tempfile
    nan = float('nan')
    assert STRATA == ((1, 20), (21, 100), (101, 300))
    assert LAYERS == ('mixed3a', 'mixed3b', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a')

    # strata: values <= 0 and NaN skipped, ties in index order, missing strata
    o = np.array([0, 3, 5, 0, 5, 1, 2, 4, -1, 3, nan])
    got = strata(o, ((1, 2), (3, 5), (6, 7)))
    assert [g.tolist() for g in got] == [[2, 4], [7, 1, 9], [6, 5]], got
    got = strata(o, ((1, 2), (3, 5), (6, 8)))
    assert [g.tolist() for g in got[:2]] == [[2, 4], [7, 1, 9]] and got[2] is None, got
    assert strata(o, ((1, 7),))[0].tolist() == [2, 4, 7, 1, 9, 6, 5] and strata(o, ((1, 8),)) == [None]
    assert strata(o) == [None, None, None] and strata(np.zeros(400)) == [None, None, None]
    assert strata(o.tolist(), ((1, 2),))[0].tolist() == [2, 4]
    assert strata(o.astype(np.float32), ((3, 5),))[0].tolist() == [7, 1, 9]
    for n_pos, want in [(300, [20, 80, 200]), (299, [20, 80, None]), (100, [20, 80, None]), (99, [20, None, None]),
                        (20, [20, None, None]), (19, [None, None, None])]:
        g = strata(np.concatenate([np.zeros(40), np.linspace(1.0, 2.0, n_pos)]))
        assert [None if x is None else len(x) for x in g] == want, (n_pos, g)
    g = strata(np.concatenate([np.zeros(40), np.linspace(1.0, 2.0, 300)]))
    assert g[0].tolist() == list(range(339, 319, -1)) and g[1].tolist() == list(range(319, 239, -1))
    assert g[2].tolist() == list(range(239, 39, -1))
    tie = np.ones(350)
    tie[::7] = 0  # 300 photos above 0, all tied
    kp = np.flatnonzero(tie > 0)
    assert [x.tolist() for x in strata(tie)] == [kp[:20].tolist(), kp[20:100].tolist(), kp[100:300].tolist()]

    # stratum statistics against hand values; M is the channel's max, not the stratum's
    o = np.array([0, 3, 5, 0, 5, 1, 2, 4, -1, 3], dtype=np.float64)
    t = np.array([9, 3, 4, 9, 1, 0, 1, 2, 9, 6], dtype=np.float64)
    s1, s2, s3 = strata(o, ((1, 2), (3, 5), (6, 7)))
    assert np.allclose(stratum_stats(o, t, s1), (0.5, 0.5))  # t / o 0.8, 0.2; |t - o| / 5 0.2, 0.8
    assert np.allclose(stratum_stats(o, t, s2), (1.0, 0.4))  # 0.5, 1, 2; 0.4, 0, 0.6 (M = 5, the stratum's max is 4)
    assert np.allclose(stratum_stats(o, t, s3), (0.25, 0.2))  # 0.5, 0; 0.2, 0.2
    assert np.allclose(stratum_stats(o, t, [7, 1, 9]), (1.0, 0.4))
    assert np.allclose(stratum_stats(o, t, np.array([4])), (0.2, 0.8))
    assert all(isinstance(x, float) for x in stratum_stats(o, t, s1))
    assert all(np.isnan(x) for x in stratum_stats(o, t, None))

    # layer statistics: shape, missing strata, values
    n_ph = 350
    orig, trans = np.zeros((n_ph, 4)), np.zeros((n_ph, 4))
    orig[:, 0] = np.arange(n_ph, 0, -1)  # 350 .. 1: every stratum
    trans[:, 0] = 0.5 * orig[:, 0]
    orig[:150, 1] = np.arange(150, 0, -1)  # 150 photos above 0: no s3
    trans[:, 1] = 2 * orig[:, 1]
    trans[150:, 1] = 5.0  # on photos with original 0: not used
    orig[:50, 2] = 1.0  # 50 photos above 0, tied: s1 = photos 0-19, no s2 or s3
    trans[:50, 2] = np.arange(50)
    trans[:, 3] = 7.0  # no photo above 0
    ret, chg = layer_stats(orig, trans)
    assert ret.shape == chg.shape == (4, 3), (ret.shape, chg.shape)
    miss = [[False] * 3, [False, False, True], [False, True, True], [True] * 3]
    assert np.isnan(ret).tolist() == miss and np.isnan(chg).tolist() == miss
    assert np.allclose(ret[0], 0.5) and np.allclose(chg[0], [0.5 * 340.5 / 350, 0.5 * 290.5 / 350, 0.5 * 150.5 / 350])
    assert np.allclose(ret[1, :2], 2.0) and np.allclose(chg[1, :2], [140.5 / 150, 90.5 / 150])
    assert np.isclose(ret[2, 0], 9.5) and np.isclose(chg[2, 0], 8.5)
    for c in range(4):
        for s, idx in enumerate(strata(orig[:, c])):
            assert np.allclose(stratum_stats(orig[:, c], trans[:, c], idx), (ret[c, s], chg[c, s]), equal_nan=True)
    r1, c1 = layer_stats(orig, trans, bounds=((1, 10),))
    assert r1.shape == c1.shape == (4, 1) and np.isclose(r1[2, 0], 4.5)
    assert layer_stats(orig.astype(np.float32), trans.astype(np.float32))[0].shape == (4, 3)
    rng = np.random.default_rng(0)
    o_r = rng.random((400, 6))
    o_r[o_r < 0.3] = 0
    o_r[:, 5] = 0
    o_r[:10, 5] = 1 + rng.random(10)  # 10 photos above 0: lookviews.retention uses them; s1 is missing here
    t_r = o_r * 2 * rng.random((400, 6)) + 0.1 * rng.random((400, 6))
    r_r, ref = layer_stats(o_r, t_r)[0][:, 0], lookviews.retention(o_r, t_r)
    assert np.allclose(r_r[:5], ref[:5], rtol=1e-12, atol=0) and np.isnan(r_r[5]) and np.isfinite(ref[5])

    # percentiles: strictly below, NaN channels in the share but never below
    p = pct_below(np.array([1.0, 2.0, nan, 3.0, 2.0]))
    assert np.allclose(p, [0, 20, nan, 60, 20], equal_nan=True), p
    assert np.isnan(pct_below(np.full(4, nan))).all() and pct_below(np.array([])).shape == (0,)
    p = pct_below(np.arange(20.0)[::-1])
    assert p[0] == 95.0 and p[18] == 5.0 and p[19] == 0.0, p
    assert np.isclose(pct_below(np.r_[np.arange(20.0), nan])[19], 100 * 19 / 21)
    assert np.allclose(pct_below([3.0, 1.0, 2.0]), [200 / 3, 0, 100 / 3])

    # labels at the boundaries
    assert label(95) == label(95.0) == label(100) == label(np.float64(95)) == label(np.float32(95)) == 'CHANGES_MORE'
    assert label(np.nextafter(95.0, 0)) == 'TYPICAL' and label(np.nextafter(5.0, 10)) == 'TYPICAL'
    assert label(5) == label(5.0) == label(0) == 'CHANGES_LESS' and label(50) == 'TYPICAL'
    assert label(nan) == label(np.float64('nan')) == 'MISSING'
    assert [label(x) for x in pct_below(np.arange(20.0))] == ['CHANGES_LESS'] * 2 + ['TYPICAL'] * 17 + ['CHANGES_MORE']

    # layer table, CSV columns and rows
    names2 = ['a', 'b']
    tab = layer_table({'orig': orig, 't_a': trans, 't_b': orig.copy()}, names2)
    assert list(tab) == names2 and all(sorted(tab[n]) == sorted(STATS) for n in names2)
    assert np.allclose(tab['a']['ret'], ret, equal_nan=True) and np.allclose(tab['a']['chg'], chg, equal_nan=True)
    for n in names2:
        for k in ('ret', 'chg'):
            assert tab[n][k + '_pct'].shape == (4, 3)
            for s in range(3):
                assert np.allclose(tab[n][k + '_pct'][:, s], pct_below(tab[n][k][:, s]), equal_nan=True)
    have = ~np.isnan(ret)
    assert np.allclose(tab['b']['ret'][have], 1) and np.allclose(tab['b']['chg'][have], 0)
    cols = csv_columns(names2)
    assert cols[:6] == ['unit', 'a_ret_s1', 'a_ret_pct_s1', 'a_chg_s1', 'a_chg_pct_s1', 'a_ret_s2'], cols
    assert cols[13:17] == ['b_ret_s1', 'b_ret_pct_s1', 'b_chg_s1', 'b_chg_pct_s1'] and len(cols) == 1 + 2 * 3 * 4
    full = csv_columns()
    assert len(full) == 1 + len(battery.TRANSFORMS) * 3 * 4 and full[1] == 'grayscale_ret_s1'
    assert full[-1] == 'mirror_chg_pct_s3' and full[12] == 'grayscale_chg_pct_s3' and full[13] == 'low_pass_ret_s1'
    assert fmt6(0.5) == '0.500000' and fmt6(1 / 3) == '0.333333'
    assert fmt6(nan) == '' and fmt6(np.float32(2)) == '2.000000'
    with tempfile.TemporaryDirectory() as td:
        pth = Path(td) / 'mixed4c.csv'
        write_csv(pth, 'mixed4c', tab, names2)
        with open(pth, newline='') as f:
            rows = list(csv.reader(f))
        assert rows[0] == cols and len(rows) == 5 and [r[0] for r in rows[1:]] == [f'mixed4c:{c}' for c in range(4)]
        assert rows[4][1:] == [''] * 24  # channel 3: no stratum
        r2 = dict(zip(cols, rows[3]))
        assert r2['a_ret_s1'] == '9.500000' and r2['a_chg_s1'] == '8.500000' and r2['a_ret_s2'] == ''
        assert r2['a_ret_pct_s1'] == '50.000000' and r2['b_chg_s1'] == '0.000000' and r2['b_chg_pct_s1'] == '0.000000'
        assert not list(Path(td).glob('*.tmp'))

    # report lines
    ln = transform_line('mirror', [0.5, nan, 1.25], [20.0, nan, 95.0], [0.125, nan, 0.0], [5.0, nan, 100.0])
    assert ln == ('mirror'.ljust(NAME_W) + '  s1 ret 0.500 pct 20.0 chg 0.125 pct 5.0 CHANGES_LESS'
                  ' | s2 ret n/a pct n/a chg n/a pct n/a MISSING'
                  ' | s3 ret 1.250 pct 95.0 chg 0.000 pct 100.0 CHANGES_MORE'), ln
    sec = unit_section('mixed4c:2', 2, tab, (orig > 0).sum(axis=0), n_ph, names2).splitlines()
    assert sec[0] == '== mixed4c:2  P0 photos above 0: 50 of 350; layer channels with s1/s2/s3: 3/2/1 of 4', sec[0]
    assert len(sec) == 3
    want1 = 'a'.ljust(NAME_W) + '  s1 ret 9.500 pct 50.0 chg 8.500 pct 50.0 TYPICAL | s2 ret n/a'
    assert sec[1].startswith(want1), sec
    assert sec[2].endswith('| s3 ret n/a pct n/a chg n/a pct n/a MISSING'), sec
    assert '>= 95 CHANGES_MORE, <= 5 CHANGES_LESS, else TYPICAL; n/a MISSING' in REPORT_HEAD

    # layers
    assert parse_layers('mixed4c, mixed3a') == ['mixed3a', 'mixed4c'] and parse_layers(','.join(LAYERS)) == list(LAYERS)
    for bad in ['', ' , ', 'mixed6a', 'mixed4c,mixed4c']:
        try:
            parse_layers(bad)
        except ValueError:
            continue
        raise AssertionError(f'accepted layers {bad!r}')

    # profile sources: the first dir that has the layer, checked there; else made in out
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        a_, b_, o_ = td / 'a', td / 'b', td / 'o'
        a_.mkdir()
        b_.mkdir()
        P0t = np.arange(0, 30, 3)

        def npz(path, layer, val, P0=P0t):
            dct = {'layer': np.array(layer), 'P0': P0, 'transforms': np.array(list(battery.TRANSFORMS)),
                   'orig': np.full((len(P0), lookviews.NCH[layer]), val, np.float32)}
            dct.update({f't_{n}': dct['orig'] for n in battery.TRANSFORMS})
            np.savez_compressed(path, **dct)

        class Asked(Exception):
            pass

        def gm():
            raise Asked
        keep = lookviews.OUT
        npz(b_ / 'profile_mixed4c.npz', 'mixed4c', 2.0)
        assert find_source('mixed4c', [a_, b_]) == b_ and find_source('mixed3a', [a_, b_]) is None
        got = load_profiles(['mixed4c'], P0t, gm, srcs=[a_, b_], out=o_)
        assert float(got['mixed4c']['orig'][0, 0]) == 2.0 and not o_.exists() and lookviews.OUT is keep
        check_profile('mixed4c', got['mixed4c'], len(P0t))
        npz(a_ / 'profile_mixed4c.npz', 'mixed4c', 1.0)
        assert float(load_profiles(['mixed4c'], P0t, gm, srcs=[a_, b_], out=o_)['mixed4c']['orig'][0, 0]) == 1.0
        npz(a_ / 'profile_mixed4c.npz', 'mixed4c', 1.0, P0=P0t + 1)
        try:
            load_profiles(['mixed4c'], P0t, gm, srcs=[a_, b_], out=o_)
            raise AssertionError('profile for other photos accepted')
        except SystemExit:
            pass
        assert lookviews.OUT is keep
        (a_ / 'profile_mixed4c.npz').unlink()
        try:
            load_profiles(['mixed4c', 'mixed3a'], P0t, gm, srcs=[a_, b_], out=o_)
            raise AssertionError('a layer in no source was not made with the model')
        except Asked:
            pass
        assert o_.is_dir() and lookviews.OUT is keep
        npz(o_ / 'profile_mixed3a.npz', 'mixed3a', 3.0)  # as rangeview.profiles would have left it
        got = load_profiles(['mixed4c', 'mixed3a'], P0t, gm, srcs=[a_, b_], out=o_)
        assert list(got) == ['mixed4c', 'mixed3a'] and lookviews.OUT is keep
        assert float(got['mixed4c']['orig'][0, 0]) == 2.0 and float(got['mixed3a']['orig'][0, 0]) == 3.0
        check_profile('mixed3a', got['mixed3a'], len(P0t))
        for bad in [{k: v for k, v in got['mixed3a'].items() if k != 't_mirror'},
                    dict(got['mixed3a'], orig=got['mixed3a']['orig'][:5])]:
            try:
                check_profile('mixed3a', bad, len(P0t))
                raise RuntimeError('bad profile accepted')
            except AssertionError:
                pass
    print('selftest ok')


# ---------- main ----------

def main():
    if '--selftest' in sys.argv[1:]:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', default=','.join(LAYERS))
    ap.add_argument('--units-file', default=UNITS_FILE, help='relative paths are taken from the repo root')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    layers = parse_layers(a.layers)
    uf = Path(a.units_file)
    units = lookviews.load_units(json.loads((uf if uf.is_absolute() else J / uf).read_text()))
    off = [u['name'] for u in units if u['layer'] not in layers]
    if off:
        raise SystemExit(f'units of layers not in --layers: {off}; stopping')
    names = json.loads(lookviews.openmethod.nearmiss2.NAMES.read_text())
    C, _ = bundlemax.pools(names)
    P0 = np.sort(np.random.default_rng(0).choice(C, lookviews.N_PROFILE, replace=False))  # as lookviews.main
    held = []

    def get_model():  # as lookviews.main: the CPU, which lookviews.profiles needs
        if not held:
            import torch
            torch.set_num_threads(4)
            m = bundlemax._bank().load_model(lookviews.openmethod.WEIGHTS)
            for q in m.parameters():
                q.requires_grad_(False)
            held.append(m)
        return held[0]

    OUT.mkdir(parents=True, exist_ok=True)
    prof = load_profiles(layers, P0, get_model)
    held.clear()
    gc.collect()

    tabs, npos = {}, {}
    for L in layers:
        d = prof.pop(L)
        check_profile(L, d, len(P0))
        npos[L] = (np.asarray(d['orig'], dtype=np.float64) > 0).sum(axis=0)
        tabs[L] = layer_table(d)
        present = np.stack([npos[L] >= b for _, b in STRATA], axis=1)
        for n in battery.TRANSFORMS:
            for k in ('ret', 'chg'):
                assert np.array_equal(np.isfinite(tabs[L][n][k]), present), (L, n, k, 'NaN outside missing strata')
        write_csv(OUT / f'{L}.csv', L, tabs[L])
        print(f'{L}: {len(npos[L])} channels, with s1/s2/s3: '
              + '/'.join(str(int(x)) for x in present.sum(axis=0)) + f' -> {OUT / f"{L}.csv"}', flush=True)
        del d
        gc.collect()

    secs = [unit_section(u['name'], u['chan'], tabs[u['layer']], npos[u['layer']], len(P0)) for u in units]
    text = REPORT_HEAD + '\n\n' + '\n\n'.join(secs) + '\n'
    (OUT / 'report.txt').write_text(text)
    print(text, flush=True)
    print('report ->', OUT / 'report.txt')


if __name__ == '__main__':
    main()
