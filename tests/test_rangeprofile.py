"""Black-box tests of tools/rangeprofile.py: strata, stratum_stats, layer_stats, pct_below, label.

Expected values are derived by hand from authors/spec_rangeprofile.md (Interface, Strata, Statistics).
Plain Python, no pytest, no GPU, model or Claude. Prints one line per failure, or exactly `tests ok`.

    python -I tests/test_rangeprofile.py
"""
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

FAILURES = []
NAN = float("nan")
DEFAULT_BOUNDS = ((1, 20), (21, 100), (101, 300))
MORE, LESS, TYPICAL, MISSING = "CHANGES_MORE", "CHANGES_LESS", "TYPICAL", "MISSING"


def fail(msg):
    FAILURES.append(msg)


def check(cond, msg):
    if not cond:
        fail(msg)


def below(x):
    """The largest float strictly below x."""
    return float(np.nextafter(x, -np.inf))


def above(x):
    """The smallest float strictly above x."""
    return float(np.nextafter(x, np.inf))


def err(e):
    return f"{type(e).__name__}: {e}"


def close(a, b, rtol=1e-9, atol=1e-12):
    """a equals b within tolerance; a NaN expectation is met only by NaN."""
    a = float(a)
    b = float(b)
    if np.isnan(b):
        return bool(np.isnan(a))
    return bool(np.isfinite(a)) and abs(a - b) <= atol + rtol * abs(b)


def show(seq, limit=12):
    vals = [float(v) for v in np.asarray(seq, dtype=np.float64).ravel()]
    txt = ["nan" if np.isnan(v) else f"{v:.10g}" for v in vals]
    if len(txt) > limit:
        txt = txt[:limit] + [f"... ({len(vals)} values)"]
    return "[" + ", ".join(txt) + "]"


def show_idx(vals, limit=8):
    vals = [int(v) for v in vals]
    if len(vals) <= limit:
        return str(vals)
    k = limit // 2
    head = ", ".join(str(v) for v in vals[:k])
    tail = ", ".join(str(v) for v in vals[-k:])
    return f"{len(vals)} indices [{head}, ..., {tail}]"


# ---------------------------------------------------------------------------------------------
# strata: a channel's photos with original value above 0, sorted highest first, stable on ties
# (equal values keep input order, so the lower index ranks first). Stratum (a, b) is ranks a..b,
# rank 1 the highest; None when the channel has fewer than b positive photos.
# Index arrays are compared in rank order.
# ---------------------------------------------------------------------------------------------
SMALL_O = [0.5, 0.0, 0.9, 0.5, -0.2, 0.7, 0.5, 0.0, 0.3, 0.9, 0.1]
# By hand: positives are 0, 2, 3, 5, 6, 8, 9, 10 (1 and 7 are zero, 4 is negative): 8 in all.
# Ranked: 2 (0.9), 9 (0.9), 5 (0.7), 0 (0.5), 3 (0.5), 6 (0.5), 8 (0.3), 10 (0.1).
SMALL_CASES = [
    # ties of 0.5 straddle the s2/s3 boundary: 0 and 3 in s2, 6 in s3
    (((1, 2), (3, 5), (6, 8)), [[2, 9], [5, 0, 3], [6, 8, 10]]),
    # 9 > 8 positives: s3 missing although the column has 11 photos
    (((1, 2), (3, 5), (6, 9)), [[2, 9], [5, 0, 3], None]),
    (((1, 3), (4, 5), (6, 9)), [[2, 9, 5], [0, 3], None]),
    # tie of 0.9 at the top: index 2 before 9
    (((1, 1), (2, 2), (3, 3)), [[2], [9], [5]]),
    # three-way tie of 0.5 split over three single-rank strata
    (((4, 4), (5, 5), (6, 6)), [[0], [3], [6]]),
    # ranks 9-10 would exist only if zeros were counted
    (((1, 2), (3, 4), (9, 10)), [[2, 9], [5, 0], None]),
    (((1, 8), (9, 10)), [[2, 9, 5, 0, 3, 6, 8, 10], None]),
]


def _idx_list(g):
    return [int(v) for v in np.asarray(g).ravel()]


def _show_any(g):
    try:
        return show_idx(_idx_list(g))
    except Exception:
        return repr(g)[:80]


def _first_diff(g, w):
    for i, (a, b) in enumerate(zip(g, w)):
        if a != b:
            return f"; first difference at position {i}: got {a}, expected {b}"
    return f"; lengths {len(g)} vs {len(w)}"


def _check_strata(tag, o, bounds, want):
    try:
        if bounds is None:
            got = rangeprofile.strata(np.array(o, copy=True))
        else:
            got = rangeprofile.strata(np.array(o, copy=True), bounds=bounds)
        got = list(got)
    except Exception as e:
        fail(f"{tag}: raised {err(e)}")
        return
    if len(got) != len(want):
        fail(f"{tag}: returned {len(got)} strata, expected {len(want)}")
        return
    for s, (g, w) in enumerate(zip(got, want), start=1):
        if w is None:
            if g is not None:
                fail(f"{tag}: stratum {s} should be missing (None), got {_show_any(g)}")
            continue
        if g is None:
            fail(f"{tag}: stratum {s} is None, expected {show_idx(w)}")
            continue
        try:
            g = _idx_list(g)
        except Exception as e:
            fail(f"{tag}: stratum {s} is not an index array: {err(e)}")
            continue
        if g != w:
            note = " (same indices, different order)" if sorted(g) == sorted(w) else ""
            fail(f"{tag}: stratum {s} got {show_idx(g)}, expected {show_idx(w)}{note}{_first_diff(g, w)}")


def test_strata_small():
    for bounds, want in SMALL_CASES:
        _check_strata(f"strata small bounds={bounds}", np.array(SMALL_O, dtype=np.float64), bounds, want)
    for bounds, want in SMALL_CASES[:2]:
        _check_strata(f"strata small float32 bounds={bounds}", np.array(SMALL_O, dtype=np.float32), bounds,
                      want)


def _distinct_case():
    """355 photos in shuffled order: 320 positive with distinct values, 30 zeros, 5 negatives.
    The photo at perm[r - 1] has rank r."""
    perm = np.random.default_rng(7).permutation(355)
    o = np.empty(355, dtype=np.float64)
    for r in range(1, 321):
        o[perm[r - 1]] = (330 - r) / 8.0
    o[perm[320:350]] = 0.0
    o[perm[350:]] = -0.5
    ranked = [int(i) for i in perm[:320]]
    return o, [ranked[0:20], ranked[20:100], ranked[100:300]]


def _all_tied_case():
    """330 photos, 30 zeros (i % 11 == 3), the other 300 all 1.0: ranks are the positives in index order,
    and exactly 300 positives make stratum 3 present."""
    o = np.array([0.0 if i % 11 == 3 else 1.0 for i in range(330)], dtype=np.float64)
    pos = [i for i in range(330) if i % 11 != 3]
    return o, [pos[0:20], pos[20:100], pos[100:300]]


def _tie_groups_case():
    """365 photos in shuffled order: 25 at 3.0, 100 at 2.0, 200 at 1.0, 40 zeros. Ties straddle ranks
    20/21 (inside the 3.0 group) and 100/101 (inside the 2.0 group); within a group, lower index first."""
    perm = np.random.default_rng(3).permutation(365)
    o = np.zeros(365, dtype=np.float64)
    g3, g2, g1 = perm[0:25], perm[25:125], perm[125:325]
    o[g3] = 3.0
    o[g2] = 2.0
    o[g1] = 1.0
    ranked = [int(i) for i in sorted(g3)] + [int(i) for i in sorted(g2)] + [int(i) for i in sorted(g1)]
    return o, [ranked[0:20], ranked[20:100], ranked[100:300]]


def test_strata_default_bounds():
    o, want = _distinct_case()
    _check_strata("strata distinct values (default bounds)", o, None, want)
    _check_strata("strata distinct values (bounds given)", o, DEFAULT_BOUNDS, want)
    o, want = _all_tied_case()
    _check_strata("strata all tied (default bounds)", o, None, want)
    o, want = _tie_groups_case()
    _check_strata("strata tie groups across boundaries (default bounds)", o, None, want)


COUNT_CASES = [0, 1, 19, 20, 21, 99, 100, 101, 150, 299, 300, 301]


def _count_case(n_pos):
    """25 zeros, then n_pos distinct positives in descending order, then 25 negatives (-1.0).
    Rank r sits at index 24 + r; a stratum is present only when n_pos >= its last rank."""
    o = np.concatenate([np.zeros(25), np.arange(n_pos, 0, -1, dtype=np.float64), np.full(25, -1.0)])
    want = []
    for a, b in DEFAULT_BOUNDS:
        want.append(list(range(24 + a, 24 + b + 1)) if n_pos >= b else None)
    return o, want


def test_strata_counts():
    for n in COUNT_CASES:
        o, want = _count_case(n)
        _check_strata(f"strata {n} positive photos of {len(o)} (default bounds)", o, None, want)


# ---------------------------------------------------------------------------------------------
# stratum_stats: retention = median over idx of t / o; change = median over idx of |t - o| / M,
# M = max of o over the whole o_col (not over idx). Even counts: mean of the two middle values.
# ---------------------------------------------------------------------------------------------
SS_O = [4.0, 0.0, 2.0, 8.0, 1.0, 16.0]  # M = 16 (photo 5)
SS_T = [2.0, 5.0, 3.0, 2.0, 1.0, 4.0]   # max of t (5, at a zero original) is not M
SS_CASES = [
    # (idx, retention, change)
    # t/o 0.5, 1.5, 1.0; |t-o|/16 0.125, 0.0625, 0 (stratum max 4 would give change 0.25)
    ([0, 2, 4], 1.0, 0.0625),
    ([4, 2, 0], 1.0, 0.0625),
    # even count: t/o 0.25, 0.5, 1.0, 1.5 -> 0.75; |t-o|/16 0, 0.0625, 0.125, 0.375 -> 0.09375
    ([0, 2, 3, 4], 0.75, 0.09375),
    # t/o 0.25, 0.25; |t-o|/16 0.375, 0.75 -> 0.5625
    ([3, 5], 0.25, 0.5625),
    ([5], 0.25, 0.75),
    ([2], 1.5, 0.0625),
]
SS2_O = [10.0, 5.0, 2.5, 0.0, 20.0]  # M = 20 (photo 4)
SS2_T = [0.0, 5.0, 5.0, 3.0, 10.0]
SS2_CASES = [
    # t/o 0, 1, 2 -> 1.0; |t-o|/20 0.5, 0, 0.125 -> 0.125 (stratum max 10 would give 0.25)
    ([0, 1, 2], 1.0, 0.125),
    # t/o 0, 0.5, 1, 2 -> 0.75; |t-o|/20 0, 0.125, 0.5, 0.5 -> 0.3125
    ([0, 1, 2, 4], 0.75, 0.3125),
    # t/o 0, 2 -> 1.0; |t-o|/20 0.5, 0.125 -> 0.3125
    ([0, 2], 1.0, 0.3125),
]


def _check_ss(tag, o, t, idx, want_r, want_c):
    try:
        got = rangeprofile.stratum_stats(np.array(o, dtype=np.float64), np.array(t, dtype=np.float64),
                                         np.array(idx, dtype=np.int64))
        r, c = got
        r = float(r)
        c = float(c)
    except Exception as e:
        fail(f"{tag}: failed: {err(e)}")
        return
    check(close(r, want_r), f"{tag}: retention {r!r}, expected {want_r!r}")
    check(close(c, want_c), f"{tag}: change {c!r}, expected {want_c!r}")


def test_stratum_stats():
    for idx, want_r, want_c in SS_CASES:
        _check_ss(f"stratum_stats o={SS_O} t={SS_T} idx={idx}", SS_O, SS_T, idx, want_r, want_c)
    for idx, want_r, want_c in SS2_CASES:
        _check_ss(f"stratum_stats o={SS2_O} t={SS2_T} idx={idx}", SS2_O, SS2_T, idx, want_r, want_c)


# ---------------------------------------------------------------------------------------------
# layer_stats: (ret, chg), each channels x strata; NaN for a missing stratum.
# ---------------------------------------------------------------------------------------------
# photos x channels, 8 x 3.
LS_ORIG = np.array([
    [3.0, 2.0, 0.0],
    [0.0, 2.0, 1.0],
    [6.0, 0.0, 0.0],
    [1.0, 2.0, 0.0],
    [2.0, 8.0, 0.0],
    [0.0, -1.0, 0.0],
    [4.0, 2.0, 0.0],
    [5.0, 0.0, 0.0],
])
LS_TRANS = np.array([
    [6.0, 1.0, 5.0],
    [9.0, 3.0, 5.0],
    [3.0, 7.0, 5.0],
    [2.0, 2.0, 5.0],
    [2.0, 4.0, 5.0],
    [9.0, 7.0, 5.0],
    [2.0, 0.0, 5.0],
    [5.0, 7.0, 5.0],
])
LS_BOUNDS = ((1, 2), (3, 4), (5, 6))
# By hand, with LS_BOUNDS:
# channel 0: positives ranked 2 (6), 7 (5), 6 (4), 0 (3), 4 (2), 3 (1); M = 6.
#   s1 {2, 7}: t/o 0.5, 1.0 -> 0.75; |t-o|/6 0.5, 0 -> 0.25
#   s2 {6, 0}: t/o 0.5, 2.0 -> 1.25; |t-o|/6 1/3, 1/2 -> 5/12
#   s3 {4, 3}: t/o 1.0, 2.0 -> 1.5;  |t-o|/6 0, 1/6 -> 1/12
# channel 1: positives ranked 4 (8), then the tied 2s in index order 0, 1, 3, 6; M = 8; 5 positives.
#   s1 {4, 0}: t/o 0.5, 0.5 -> 0.5;  |t-o|/8 0.5, 0.125 -> 0.3125
#   s2 {1, 3}: t/o 1.5, 1.0 -> 1.25; |t-o|/8 0.125, 0 -> 0.0625
#   s3: 5 < 6 positives -> NaN
# channel 2: 1 positive photo -> all strata NaN.
LS_RET = [[0.75, 1.25, 1.5], [0.5, 1.25, NAN], [NAN, NAN, NAN]]
LS_CHG = [[0.25, 5.0 / 12.0, 1.0 / 12.0], [0.3125, 0.0625, NAN], [NAN, NAN, NAN]]

# Default-bounds layer, 340 photos. Channel k has DEF_NPOS[k] positive photos at shuffled positions;
# the photo of rank r has original DEF_SCALE[k] * (400 - r), so M = DEF_SCALE[k] * 399, and
# transformed g * original, g = DEF_G[k][s] for ranks in stratum s, G_BEYOND past rank 300.
# Photos whose original is not above 0 (zeros, some -1.0) get transformed T_NONPOS.
# Hence retention = g exactly, and change = |g - 1| * median(400 - r over the stratum) / 399, the
# medians (even counts) being: ranks 1-20 -> (390 + 389) / 2, 21-100 -> (340 + 339) / 2,
# 101-300 -> (200 + 199) / 2.
DEF_P = 340
DEF_NPOS = (320, 150, 50)
DEF_SCALE = (1.0, 2.0, 0.5)
DEF_G = ((0.5, 0.75, 0.25), (1.5, 0.125, 3.0), (2.0, 0.0625, 0.375))
DEF_MED = (389.5, 339.5, 199.5)
G_BEYOND = 4.0
T_NONPOS = 7.0


def _default_layer(channels):
    orig = np.zeros((DEF_P, len(channels)), dtype=np.float64)
    trans = np.full((DEF_P, len(channels)), T_NONPOS, dtype=np.float64)
    for j, k in enumerate(channels):
        perm = np.random.default_rng(100 + k).permutation(DEF_P)
        n = DEF_NPOS[k]
        for r in range(1, n + 1):
            p = perm[r - 1]
            o = DEF_SCALE[k] * (400 - r)
            s = 0 if r <= 20 else 1 if r <= 100 else 2 if r <= 300 else None
            g = G_BEYOND if s is None else DEF_G[k][s]
            orig[p, j] = o
            trans[p, j] = g * o
        rest = perm[n:]
        orig[rest[::7], j] = -1.0
    return orig, trans


def _default_want(channels):
    ret, chg = [], []
    for k in channels:
        rr, cc = [], []
        for s, (_, b) in enumerate(DEFAULT_BOUNDS):
            if DEF_NPOS[k] >= b:
                g = DEF_G[k][s]
                rr.append(g)
                cc.append(abs(g - 1.0) * DEF_MED[s] / 399.0)
            else:
                rr.append(NAN)
                cc.append(NAN)
        ret.append(rr)
        chg.append(cc)
    return ret, chg


def _check_grid(tag, name, got, want, rtol):
    try:
        g = np.asarray(got, dtype=np.float64)
    except Exception as e:
        fail(f"{tag}: {name} is not a numeric array: {err(e)}")
        return
    w = np.asarray(want, dtype=np.float64)
    if g.shape != w.shape:
        fail(f"{tag}: {name} has shape {g.shape}, expected {w.shape} (channels x strata)")
        return
    for c in range(w.shape[0]):
        for s in range(w.shape[1]):
            if not close(g[c, s], w[c, s], rtol=rtol):
                fail(f"{tag}: {name}[channel {c}, stratum {s + 1}] = {g[c, s]!r}, expected {w[c, s]!r}")


def _check_layer(tag, orig, trans, bounds, want_ret, want_chg, rtol=1e-9):
    try:
        if bounds is None:
            got = rangeprofile.layer_stats(orig.copy(), trans.copy())
        else:
            got = rangeprofile.layer_stats(orig.copy(), trans.copy(), bounds=bounds)
        ret, chg = got
    except Exception as e:
        fail(f"{tag}: failed: {err(e)}")
        return
    _check_grid(tag, "ret", ret, want_ret, rtol)
    _check_grid(tag, "chg", chg, want_chg, rtol)


def test_layer_stats_small():
    _check_layer(f"layer_stats hand case bounds={LS_BOUNDS}", LS_ORIG, LS_TRANS, LS_BOUNDS, LS_RET, LS_CHG)
    b2 = LS_BOUNDS[:2]
    _check_layer(f"layer_stats hand case bounds={b2}", LS_ORIG, LS_TRANS, b2,
                 [row[:2] for row in LS_RET], [row[:2] for row in LS_CHG])
    _check_layer(f"layer_stats hand case float32 bounds={LS_BOUNDS}", LS_ORIG.astype(np.float32),
                 LS_TRANS.astype(np.float32), LS_BOUNDS, LS_RET, LS_CHG, rtol=1e-5)


def test_layer_stats_default_bounds():
    chans = (0, 1, 2)
    orig, trans = _default_layer(chans)
    want_ret, want_chg = _default_want(chans)
    _check_layer("layer_stats default bounds (320, 150, 50 positives)", orig, trans, None, want_ret, want_chg)


def test_layer_stats_zero_channel():
    orig0, trans0 = _default_layer((0,))
    orig = np.column_stack([orig0[:, 0], np.zeros(DEF_P)])
    trans = np.column_stack([trans0[:, 0], np.full(DEF_P, T_NONPOS)])
    want_ret, want_chg = _default_want((0,))
    want_ret = want_ret + [[NAN, NAN, NAN]]
    want_chg = want_chg + [[NAN, NAN, NAN]]
    _check_layer("layer_stats with an all-zero channel (default bounds)", orig, trans, None, want_ret, want_chg)


# ---------------------------------------------------------------------------------------------
# pct_below: 100 x the share of the layer's channels with a finite value strictly below its own;
# NaN in, NaN out. Without NaN the denominator is all channels (the channel itself included).
# With NaN present the spec leaves the denominator open (all channels, or the finite ones): either is
# accepted, the same one for every channel; the numerator (finite values strictly below) is exact.
# ---------------------------------------------------------------------------------------------
PERM20 = [float((7 * i) % 20) for i in range(20)]  # a shuffle of 0..19: value v has v values below it
PCT_CASES = [
    ([0.3, 0.1, 0.3, 0.7, 0.0, 0.1, 0.5, 0.3], [37.5, 12.5, 37.5, 87.5, 0.0, 12.5, 75.0, 37.5]),
    ([-1.0, -3.0, 2.0, -1.0], [25.0, 0.0, 75.0, 25.0]),
    ([2.0, 2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0]),
    ([1.0], [0.0]),
    ([1.0, below(1.0)], [50.0, 0.0]),
    ([0.5, below(0.5), 0.5], [100.0 / 3.0, 0.0, 100.0 / 3.0]),
    (PERM20, [5.0 * v for v in PERM20]),
]
# (x, count of finite values strictly below each entry, None where the entry is NaN)
PCT_NAN_CASES = [
    ([0.4, NAN, 0.2, 0.4, NAN, 0.9, 0.0, 0.2], [3, None, 1, 3, None, 5, 0, 1]),
    ([NAN, 3.0, 5.0], [None, 0, 1]),
    ([1.0, NAN, 1.0], [0, None, 0]),
    ([NAN, -2.0], [None, 0]),
    ([NAN, NAN, NAN], [None, None, None]),
]


def _call_pct(x):
    got = rangeprofile.pct_below(np.array(x, dtype=np.float64))
    return [float(v) for v in np.asarray(got, dtype=np.float64).ravel()]


def test_pct_below_exact():
    for x, want in PCT_CASES:
        tag = f"pct_below({show(x)})"
        try:
            got = _call_pct(x)
        except Exception as e:
            fail(f"{tag}: raised {err(e)}")
            continue
        if len(got) != len(want):
            fail(f"{tag}: returned {len(got)} values, expected {len(want)}")
            continue
        ok = all(close(g, w) for g, w in zip(got, want))
        check(ok, f"{tag} = {show(got)}, expected {show(want)}")


def test_pct_below_nan():
    for x, ks in PCT_NAN_CASES:
        tag = f"pct_below({show(x)})"
        n = len(x)
        nf = sum(1 for v in x if np.isfinite(v))
        want_all = [NAN if k is None else 100.0 * k / n for k in ks]
        want_fin = [NAN if k is None else 100.0 * k / nf for k in ks]
        try:
            got = _call_pct(x)
        except Exception as e:
            fail(f"{tag}: raised {err(e)}")
            continue
        if len(got) != n:
            fail(f"{tag}: returned {len(got)} values, expected {n}")
            continue
        ok_all = all(close(g, w) for g, w in zip(got, want_all))
        ok_fin = all(close(g, w) for g, w in zip(got, want_fin))
        check(ok_all or ok_fin,
              f"{tag} = {show(got)}, expected {show(want_all)} (share of all {n} channels) "
              f"or {show(want_fin)} (share of the {nf} finite channels)")


# ---------------------------------------------------------------------------------------------
# label: >= 95 CHANGES_MORE, <= 5 CHANGES_LESS, else TYPICAL; NaN MISSING.
# ---------------------------------------------------------------------------------------------
LABEL_CASES = [
    (100.0, MORE), (99.5, MORE), (above(95.0), MORE), (95.0, MORE),
    (below(95.0), TYPICAL), (94.999, TYPICAL), (50.0, TYPICAL), (5.000001, TYPICAL), (above(5.0), TYPICAL),
    (5.0, LESS), (below(5.0), LESS), (4.5, LESS), (0.0, LESS),
    (NAN, MISSING),
]
LABEL_F32 = [
    (np.float32(95.0), MORE),
    (np.nextafter(np.float32(95.0), np.float32(0.0)), TYPICAL),
    (np.float32(50.0), TYPICAL),
    (np.nextafter(np.float32(5.0), np.float32(100.0)), TYPICAL),
    (np.float32(5.0), LESS),
    (np.float32("nan"), MISSING),
]


def test_label():
    cases = [(x, w) for p, w in LABEL_CASES for x in (p, np.float64(p))] + LABEL_F32
    for x, want in cases:
        tag = f"label({type(x).__name__} {float(x)!r})"
        try:
            got = rangeprofile.label(x)
        except Exception as e:
            fail(f"{tag}: raised {err(e)}")
            continue
        check(isinstance(got, str) and got == want, f"{tag} = {got!r}, expected {want!r}")


TESTS = [
    test_strata_small,
    test_strata_default_bounds,
    test_strata_counts,
    test_stratum_stats,
    test_layer_stats_small,
    test_layer_stats_default_bounds,
    test_layer_stats_zero_channel,
    test_pct_below_exact,
    test_pct_below_nan,
    test_label,
]


def main():
    global rangeprofile
    try:
        import rangeprofile  # noqa: F401
    except Exception as e:
        print(f"FAIL import rangeprofile: {err(e)}".replace("\n", " "))
        sys.exit(1)
    # Missing strata and NaN inputs may make numpy warn (empty median, 0/0); keep stdout to the verdict.
    warnings.simplefilter("ignore", RuntimeWarning)
    for t in TESTS:
        try:
            t()
        except Exception as e:
            fail(f"{t.__name__}: raised {err(e)}")
    if FAILURES:
        for f in FAILURES:
            print(("FAIL " + f).replace("\n", " "))
        sys.exit(1)
    print("tests ok")


if __name__ == "__main__":
    main()
