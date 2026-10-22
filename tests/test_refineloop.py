"""Black-box tests of tools/refineloop.py, written from authors/spec_refineloop.md alone:
z_score, t_record, kind, trigger_f, trigger_r, panel_f, panel_r, parse_refined, conditional,
table_text, refine_content.

Every expected value is derived by hand from the spec (Earlier predictions on T; Triggers; Panels;
Refinement calls; Interface) on small synthetic arrays, hand records and hand JSON replies; the rules it
reuses from contrastloop (contrast, outcome, parse_predictions's entry rule) as tests/test_contrastloop.py
pins them. One exception: proploop.percentile is named by the spec but not defined there, so expected
percentiles are proploop.percentile of the hand-derived d vector (an oracle; see GAP at t_record).
Plain Python, no pytest, no GPU, model or Claude. Prints one line per failure, or exactly `tests ok`.

    python -I tests/test_refineloop.py

Where the spec leaves something open, only what it fixes is tested; each such place is marked GAP.
"""
import copy
import json
import math
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

FAILURES = []
TOL = 1e-6          # float tolerance, relative for values above 1
MISSING = object()
proploop = None     # percentile oracle, imported in main()


def fail(msg):
    FAILURES.append(msg)


def check(cond, msg):
    if not cond:
        fail(msg)


def err(e):
    return f"{type(e).__name__}: {e}"


def attempt(fn):
    """Run fn(). (True, exception) if it raised anything (SystemExit included), else (False, result)."""
    try:
        out = fn()
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        return True, e
    return False, out


def show(x):
    return f"{type(x).__name__} {x!r}"


def as_ids(x):
    """x as a list of Python ints (photo or channel ids), or None if it is not a sequence of integers."""
    try:
        out = [int(i) for i in x]
        if any(i != j for i, j in zip(x, out)):
            return None
    except Exception:
        return None
    return out


def as_int(x):
    if isinstance(x, (bool, np.bool_)):
        return None
    try:
        i = int(x)
        return i if i == x else None
    except Exception:
        return None


def as_float(x):
    if isinstance(x, (bool, np.bool_)):
        return None
    try:
        return float(x)
    except Exception:
        return None


def close(got, want, tol=TOL):
    g = as_float(got)
    return g is not None and math.isfinite(g) and abs(g - want) <= tol * max(1.0, abs(want))


def held_value(x):
    """True / False for a held flag, None if it is neither.
    GAP: the type of t_record's "held" is not fixed; a bool is the reading taken here (and the hand records
    below use bools), and contrastloop.outcome's strings "held" / "not held" are also accepted from t_record."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x == "held":
        return True
    if x == "not held":
        return False
    return None


def oracle_pct(d, j):
    return float(proploop.percentile(np.array(d, dtype=float), j))


# ---------------------------------------------------------------------------------------------
# z_score(d_u, d_p, rho, n_hi, n_lo) = (d_u - d_p) / sqrt(2 (1 - rho) (1/n_hi + 1/n_lo)); 0 when
# 1 - rho <= 1e-12 (rho 1, and rho a hair above 1, give 0, not a division by zero or a NaN).
# ---------------------------------------------------------------------------------------------
_R_2E12 = 1 - 2e-12                    # 1 - rho is about 2.0002e-12 in floats: just above the cut
_R_1E6 = 1 - 1e-6
Z_CASES = [
    # (name, (d_u, d_p, rho, n_hi, n_lo), expected)
    ("rho 0.5, n 4 and 4", (1.0, 0.0, 0.5, 4, 4), math.sqrt(2.0)),            # 1 / sqrt(2 x 0.5 x 0.5)
    ("d_u < d_p, unequal n", (0.5, 2.0, 0.0, 5, 10), -1.5 / math.sqrt(0.6)),  # 2 x 1 x (0.2 + 0.1)
    ("negative rho", (2.0, -1.0, -0.5, 10, 5), 3.0 / math.sqrt(0.9)),         # 2 x 1.5 x 0.3
    ("rho -1", (1.0, 0.0, -1.0, 5, 5), 1.0 / math.sqrt(1.6)),                 # 2 x 2 x 0.4
    ("equal d, rho 0", (0.3, 0.3, 0.0, 5, 5), 0.0),
    ("rho 1", (1.0, 0.2, 1.0, 5, 5), 0.0),
    ("rho 1, equal d", (0.7, 0.7, 1.0, 5, 5), 0.0),
    ("rho 1 (int), n 1 and 1", (2.0, -3.0, 1, 1, 1), 0.0),
    ("1 - rho = 5e-13 (<= 1e-12)", (1.0, 0.0, 1 - 5e-13, 5, 5), 0.0),
    ("rho one ulp above 1", (1.0, 0.0, float(np.nextafter(1.0, 2.0)), 5, 5), 0.0),
    ("1 - rho about 2e-12 (> 1e-12)", (1.0, 0.0, _R_2E12, 5, 5),
     1.0 / math.sqrt(2 * (1 - _R_2E12) * 0.4)),
    ("1 - rho = 1e-6", (1.0, 0.0, _R_1E6, 5, 5), 1.0 / math.sqrt(2 * (1 - _R_1E6) * 0.4)),
    ("numpy scalars", (np.float64(1.0), np.float64(0.0), np.float64(0.5), np.int64(4), np.int64(4)),
     math.sqrt(2.0)),
]


def test_z_score():
    for name, args, want in Z_CASES:
        tag = f"z_score {name} {args!r}"
        raised, got = attempt(lambda: refineloop.z_score(*args))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        check(close(got, want), f"{tag}: got {show(got)}, expected {want!r}")


# ---------------------------------------------------------------------------------------------
# t_record(R, ans, most, least, c, peers): contrastloop.contrast(R, ans, most, least) (min_n 5); None
# when not checkable; else n_hi, n_lo, d_u = d[c], pct_u = proploop.percentile(d, c), held as
# contrastloop.outcome (pct_u >= 95 and d_u > 0), and per peer p (in the given order): d[p],
# percentile(d, p), rho = Pearson of R[:, c] and R[:, p] over the compared photos only (0 when
# undefined), z as z_score.
# GAP: proploop.percentile is not defined in the spec; expected percentiles are proploop.percentile of
# the hand d vector (all d distinct where a percentile is checked, so float noise cannot move a rank).
# The order of the "peers" list is taken to be the order of the given peers.
# Fixture: 13 photos, 5 channels [B, C, U, D, X]; the unit is c = 2 (U); most = 1, least = 0.
# Compared photos: hi (1) 0, 3, 6, 8, 10; lo (0) 1, 4, 7, 9, 11. Photos 2, 12 (answer 2) and 5 (None)
# have values that would change every d and rho if they were used.
#   U: hi 2,4,6,8,10, lo 1,3,5,7,9 -> d_u = 1 / sqrt(82.5/9) (ten values 1..10, ddof 1).
#   B: 3 on every compared photo -> d 0, rho undefined -> 0, z = d_u / sqrt(2 x 1 x 0.4).
#   C: hi 0, lo 1 -> d = -1 / sqrt(2.5/9); rho = -2.5 / sqrt(82.5 x 2.5) = -1/sqrt(33).
#   D: hi 2,4,6,8,9, lo 1,3,5,7,10 (U with 9 and 10 swapped) -> d = 0.6 d_u; rho = 81.5 / 82.5.
#   X: -U -> d = -d_u (not a peer in the main case).
# ---------------------------------------------------------------------------------------------
TR_ANS = [1, 0, 2, 1, 0, None, 1, 0, 1, 0, 1, 0, 2]
TR_R = np.array([
    # B     C     U      D      X
    [3.0, 0.0, 2.0, 2.0, -2.0],         # 0  hi
    [3.0, 1.0, 1.0, 1.0, -1.0],         # 1  lo
    [7.0, 5.0, 100.0, -40.0, 3.0],      # 2  answer 2
    [3.0, 0.0, 4.0, 4.0, -4.0],         # 3  hi
    [3.0, 1.0, 3.0, 3.0, -3.0],         # 4  lo
    [-7.0, -5.0, -50.0, 60.0, 0.0],     # 5  unanswered
    [3.0, 0.0, 6.0, 6.0, -6.0],         # 6  hi
    [3.0, 1.0, 5.0, 5.0, -5.0],         # 7  lo
    [3.0, 0.0, 8.0, 8.0, -8.0],         # 8  hi
    [3.0, 1.0, 7.0, 7.0, -7.0],         # 9  lo
    [3.0, 0.0, 10.0, 9.0, -10.0],       # 10 hi
    [3.0, 1.0, 9.0, 10.0, -9.0],        # 11 lo
    [1.0, 2.0, -100.0, 30.0, 9.0],      # 12 answer 2
])
TR_DU = 1.0 / math.sqrt(82.5 / 9)
TR_DC = -1.0 / math.sqrt(2.5 / 9)
TR_D = [0.0, TR_DC, TR_DU, 0.6 * TR_DU, -TR_DU]
TR_RHO_C = -1.0 / math.sqrt(33.0)
TR_RHO_D = 81.5 / 82.5
# Case 2 adds channel 5 (A) = 2U + 1 on the compared photos (0 elsewhere): rho 1 -> z 0, d = d_u. A ties
# the unit's d, so the unit's and A's percentiles are not checked there; X (-U, rho -1) is.
TR_R2 = np.hstack([TR_R, np.array([[5.0], [3.0], [0.0], [9.0], [7.0], [0.0], [13.0], [11.0], [17.0],
                                   [15.0], [21.0], [19.0], [0.0]])])
TR_D2 = TR_D + [TR_DU]

TR_CASES = [
    # (name, R, ans, most, least, c, peers, expected or None)
    # expected: (n_hi, n_lo, d_u, d vector for the oracle, check pct_u, [(chan, d, check pct, rho, z)])
    ("hand case, peers out of channel order", TR_R, TR_ANS, 1, 0, 2, [3, 0, 1],
     (5, 5, TR_DU, TR_D, True, [
         (3, 0.6 * TR_DU, True, TR_RHO_D, 0.4 * TR_DU / math.sqrt(2 * (1 - TR_RHO_D) * 0.4)),
         (0, 0.0, True, 0.0, TR_DU / math.sqrt(2 * 1 * 0.4)),
         (1, TR_DC, True, TR_RHO_C, (TR_DU - TR_DC) / math.sqrt(2 * (1 - TR_RHO_C) * 0.4)),
     ])),
    ("rho 1 gives z 0; rho -1", TR_R2, TR_ANS, 1, 0, 2, [5, 4],
     (5, 5, TR_DU, TR_D2, False, [
         (5, TR_DU, False, 1.0, 0.0),
         (4, -TR_DU, True, -1.0, 2 * TR_DU / math.sqrt(2 * 2 * 0.4)),
     ])),
    ("most and least swapped", TR_R, TR_ANS, 0, 1, 2, [1],
     (5, 5, -TR_DU, [-x for x in TR_D], True, [
         (1, -TR_DC, True, TR_RHO_C, (-TR_DU + TR_DC) / math.sqrt(2 * (1 - TR_RHO_C) * 0.4)),
     ])),
    ("too few: 4 most-active photos", TR_R, [1, 0, 2, 1, 0, None, 1, 0, 1, 0, 2, 0, 2], 1, 0, 2, [3, 0, 1],
     None),
    ("too few: 4 least-active photos", TR_R, [1, 0, 2, 1, 0, None, 1, 0, 1, 0, 1, None, 2], 1, 0, 2,
     [3, 0, 1], None),
    ("too few: the least-active answer never given", TR_R, TR_ANS, 1, 3, 2, [3, 0, 1], None),
]


def test_t_record():
    pct_ok = proploop is not None and hasattr(proploop, "percentile")
    if not pct_ok:
        fail("t_record: proploop.percentile (the percentile oracle) is not importable; percentiles not checked")
    for name, R, ans, most, least, c, peers, want in TR_CASES:
        tag = f"t_record {name}"
        raised, got = attempt(lambda: refineloop.t_record(R.copy(), list(ans), most, least, c, list(peers)))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        if want is None:
            check(got is None, f"{tag}: got {show(got)}, expected None (not checkable)")
            continue
        if not isinstance(got, dict):
            fail(f"{tag}: got {show(got)}, expected a dict")
            continue
        n_hi, n_lo, d_u, dvec, check_pu, wpeers = want
        check(as_int(got.get("n_hi")) == n_hi and as_int(got.get("n_lo")) == n_lo,
              f"{tag}: n_hi, n_lo = {got.get('n_hi')!r}, {got.get('n_lo')!r}, expected {n_hi}, {n_lo}")
        check(close(got.get("d_u"), d_u), f"{tag}: d_u {got.get('d_u')!r}, expected {d_u!r}")
        pct_u = None
        if pct_ok:
            raised2, pct_u = attempt(lambda: oracle_pct(dvec, c))
            if raised2:
                fail(f"{tag}: proploop.percentile on the hand d raised {err(pct_u)}")
                pct_u = None
        if check_pu and pct_u is not None:
            check(close(got.get("pct_u"), pct_u),
                  f"{tag}: pct_u {got.get('pct_u')!r}, expected proploop.percentile(d, {c}) = {pct_u!r}")
            h = held_value(got.get("held"))
            want_h = pct_u >= 95 and d_u > 0
            check(h is want_h, f"{tag}: held {got.get('held')!r}, expected {want_h!r} (pct_u {pct_u}, d_u {d_u})")
        elif not check_pu:
            check(held_value(got.get("held")) is not None, f"{tag}: held {got.get('held')!r}, not a held flag")
        gp = got.get("peers")
        if not isinstance(gp, list) or len(gp) != len(wpeers):
            fail(f"{tag}: peers {show(gp)}, expected a list of {len(wpeers)} dicts")
            continue
        for i, (g, (ch, d, check_p, rho, z)) in enumerate(zip(gp, wpeers)):
            ptag = f"{tag}: peer {i + 1}"
            if not isinstance(g, dict):
                fail(f"{ptag}: {show(g)}, expected a dict")
                continue
            check(as_int(g.get("chan")) == ch, f"{ptag}: chan {g.get('chan')!r}, expected {ch}")
            check(close(g.get("d"), d), f"{ptag} (chan {ch}): d {g.get('d')!r}, expected {d!r}")
            check(close(g.get("rho"), rho), f"{ptag} (chan {ch}): rho {g.get('rho')!r}, expected {rho!r}")
            check(close(g.get("z"), z), f"{ptag} (chan {ch}): z {g.get('z')!r}, expected {z!r}")
            if check_p and pct_ok:
                raised2, p = attempt(lambda: oracle_pct(dvec, ch))
                if not raised2:
                    check(close(g.get("pct"), p),
                          f"{ptag} (chan {ch}): pct {g.get('pct')!r}, expected proploop.percentile(d, {ch}) = {p!r}")


# ---------------------------------------------------------------------------------------------
# Hand records shaped like t_record's output. Every peer has rho 0.5 and n_hi = n_lo = 8, so z_score
# gives z = (d_u - d_p) / sqrt(2 x 0.5 x (1/8 + 1/8)) = 2 (d_u - d_p): the records are consistent with
# z_score whether a function reads "z" or recomputes it. held as contrastloop.outcome.
# ---------------------------------------------------------------------------------------------
def mkrec(d_u, pct_u, peers):
    return {"n_hi": 8, "n_lo": 8, "d_u": d_u, "pct_u": pct_u, "held": bool(pct_u >= 95 and d_u > 0),
            "peers": [{"chan": ch, "d": d, "pct": p, "rho": 0.5, "z": 2 * (d_u - d)} for ch, d, p in peers]}


# ---------------------------------------------------------------------------------------------
# kind(rec): "shared" (held, some peer P_p >= 95; peer = highest P_p, ties by peer order); "own" (held
# otherwise); "differs" (not held, some |z_p| >= 2; peer = highest |z_p|, ties by peer order); "none";
# ("too few", None) for None. own and none have no peer (None).
# ---------------------------------------------------------------------------------------------
KIND_CASES = [
    # (name, rec, (kind, peer))
    ("shared: the peer with the highest P_p",
     mkrec(1.0, 97, [(7, 0.75, 80), (3, 0.5, 97), (9, 1.25, 96)]), ("shared", 3)),
    ("shared: P_p tie by peer order, not by channel",
     mkrec(1.0, 99, [(7, 0.5, 99), (3, 0.75, 50), (2, 0.25, 99)]), ("shared", 7)),
    ("shared: P_p exactly 95 (and P_u exactly 95)",
     mkrec(0.5, 95, [(7, 0.5, 94.99), (3, 0.25, 95), (9, 0.0, 10)]), ("shared", 3)),
    ("shared: P_p exactly 95.0 (float)",
     mkrec(0.5, 99.0, [(7, 0.5, 95.0), (3, 0.25, 12.0), (9, 0.0, 10.0)]), ("shared", 7)),
    ("own: no peer P_p >= 95, big |z| ignored when held",
     mkrec(2.0, 99, [(7, 0.0, 94.99), (3, 4.5, 90), (9, 1.5, 0)]), ("own", None)),
    ("differs: the peer with the highest |z_p| (negative z)",
     mkrec(0.25, 60, [(7, -0.5, 50), (3, 1.5, 20), (9, -0.75, 99)]), ("differs", 3)),
    ("differs: |z| tie by peer order, not by channel",
     mkrec(0.25, 60, [(9, -1.0, 40), (4, 1.5, 40), (1, 0.0, 40)]), ("differs", 9)),
    ("differs: z exactly -2",
     mkrec(0.25, 30, [(7, 0.0, 50), (3, 1.25, 50), (9, -0.6, 50)]), ("differs", 3)),
    ("differs: z exactly +2",
     mkrec(0.25, 30, [(7, -0.75, 50), (3, 0.0, 50), (9, 0.25, 50)]), ("differs", 7)),
    ("none: |z| 1.999 and peers above 95 while the unit did not hold",
     mkrec(0.25, 60, [(7, -0.7495, 99), (3, 1.2495, 99), (9, 0.25, 99)]), ("none", None)),
    ("none: P_u >= 95 but d_u < 0 is not held, so not shared",
     mkrec(-0.25, 97, [(7, -0.5, 99), (3, -0.25, 99), (9, 0.0, 98)]), ("none", None)),
    ("differs: P_u >= 95 but d_u < 0",
     mkrec(-0.25, 97, [(7, 1.0, 99), (3, -0.25, 99), (9, 0.0, 98)]), ("differs", 7)),
    ("none: d_u 0 is not held",
     mkrec(0.0, 99, [(7, 0.0, 99), (3, 0.25, 99), (9, -0.25, 99)]), ("none", None)),
    ("too few", None, ("too few", None)),
]


def test_kind():
    for name, rec, want in KIND_CASES:
        tag = f"kind {name}"
        raised, got = attempt(lambda: refineloop.kind(copy.deepcopy(rec)))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        if not (isinstance(got, (tuple, list)) and len(got) == 2):
            fail(f"{tag}: got {show(got)}, expected (kind, peer channel or None)")
            continue
        k, p = got
        ok = k == want[0] and ((p is None) if want[1] is None else as_int(p) == want[1])
        check(ok, f"{tag}: got {got!r}, expected {want!r}")


# ---------------------------------------------------------------------------------------------
# trigger_f(recs): the "shared" one with the highest P_u (ties lower index) with its peer; else the
# "differs" one with the highest |z| of its peer (ties lower index) with that peer; else None.
# Returns (index, peer channel, kind).
# ---------------------------------------------------------------------------------------------
SH_A = mkrec(1.0, 96, [(7, 0.5, 99), (3, 0.75, 60), (9, 0.25, 40)])      # shared, peer 7
SH_B = mkrec(1.0, 98, [(7, 0.5, 60), (3, 0.75, 97), (9, 0.25, 40)])      # shared, peer 3
SH_C = mkrec(0.5, 98, [(7, 0.25, 10), (3, 0.0, 20), (9, 0.5, 95)])       # shared, peer 9
OWN_HI = mkrec(3.0, 100, [(7, 0.0, 94), (3, 6.0, 90), (9, -1.5, 0)])     # own; |z| 6, 6, 9
DF_A = mkrec(0.25, 60, [(7, -0.8, 50), (3, 1.2, 50), (9, -0.7, 50)])     # differs, peer 7; z 2.1, -1.9, 1.9
DF_B = mkrec(0.25, 10, [(7, 0.0, 50), (3, -0.75, 50), (9, 1.55, 50)])    # differs, peer 9; z 0.5, 2, -2.6
DF_C = mkrec(0.25, 70, [(7, 1.55, 50), (3, 0.25, 50), (9, 0.25, 50)])    # differs, peer 7; z -2.6, 0, 0
NONE_A = mkrec(-0.25, 99, [(7, -0.5, 99), (3, 0.0, 99), (9, -0.25, 99)])  # none (d_u < 0)

TRIGGER_F_CASES = [
    # (name, recs, expected (index, peer, kind) or None)
    ("shared preferred over differs and own; the highest P_u", [OWN_HI, SH_A, DF_B, SH_B, None],
     (3, 3, "shared")),
    ("shared P_u tie by lower index", [None, SH_B, SH_A, SH_C], (1, 3, "shared")),
    ("shared at index 0", [SH_C, SH_A, SH_B], (0, 9, "shared")),
    ("a later shared over an earlier differs", [DF_B, SH_A], (1, 7, "shared")),
    ("differs: the highest |z| of its peer (not the sum of |z|)", [DF_A, OWN_HI, DF_B, NONE_A, None],
     (2, 9, "differs")),
    ("differs |z| tie by lower index", [DF_A, DF_C, DF_B], (1, 7, "differs")),
    ("differs |z| tie by lower index, not by P_u", [DF_B, DF_C], (0, 9, "differs")),
    ("an own with bigger |z| is not a differs", [DF_A, OWN_HI], (0, 7, "differs")),
    ("no shared or differs", [OWN_HI, NONE_A, None], None),
    ("all None", [None, None], None),
    ("no predictions", [], None),
]


def test_trigger_f():
    for name, recs, want in TRIGGER_F_CASES:
        tag = f"trigger_f {name}"
        raised, got = attempt(lambda: refineloop.trigger_f(copy.deepcopy(recs)))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        if want is None:
            check(got is None, f"{tag}: got {show(got)}, expected None")
            continue
        ok = (isinstance(got, (tuple, list)) and len(got) == 3 and as_int(got[0]) == want[0]
              and as_int(got[1]) == want[1] and got[2] == want[2])
        check(ok, f"{tag}: got {show(got)}, expected {want!r}")


# ---------------------------------------------------------------------------------------------
# trigger_r(recs): the index of the checkable record with the highest P_u (held or not), ties by lower
# index; None when none is checkable.
# ---------------------------------------------------------------------------------------------
def rr(pct_u, d_u):
    return mkrec(d_u, pct_u, [(7, 0.0, 50), (3, 0.0, 50), (9, 0.0, 50)])


TRIGGER_R_CASES = [
    ("the highest P_u, tie by lower index", [None, rr(50, 0.5), rr(80, 0.25), None, rr(80, 1.0)], 2),
    ("tie at index 0", [rr(100, 1.0), rr(100, 2.0)], 0),
    ("the highest P_u even when not held", [SH_A, NONE_A], 1),
    ("the highest P_u, not held because d_u is 0", [rr(10, -1.0), rr(99.5, 0.0), rr(99.4, 1.0)], 1),
    ("P_u 0 is still checkable", [None, rr(0, -1.0)], 1),
    ("all None", [None, None, None], None),
    ("no predictions", [], None),
]


def test_trigger_r():
    for name, recs, want in TRIGGER_R_CASES:
        tag = f"trigger_r {name}"
        raised, got = attempt(lambda: refineloop.trigger_r(copy.deepcopy(recs)))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        if want is None:
            check(got is None, f"{tag}: got {show(got)}, expected None")
        else:
            check(got is not None and as_int(got) == want, f"{tag}: got {show(got)}, expected {want}")


# ---------------------------------------------------------------------------------------------
# panel_f(R, ids, ans, a, c, k, n=3): G = the ids whose answer is a; e = R[:, c] - R[:, k] over G;
# "more" the n highest e, highest first; "less" the n lowest e, lowest first; ties by lower photo id (so
# a tie at the bottom cut picks the lower id too); when |G| < 6, floor(|G| / 2) each.
# panel_r(R, ids, ans, a, c, n=3): the same on R[:, c] ("most", "least").
# ids are given in photo index order (as the spec says), not contiguous, and rows are aligned with them.
# GAP: the small-group rule with n other than 3 (is the cut 6 or 2n?) and ties that span the most and
# the least picks (overlap) are not tested; n other than 3 is tested only on a group of 8.
# ---------------------------------------------------------------------------------------------
PN_IDS = [2, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
PN_ANS = [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, None, 2]
PN_R = np.array([
    # c=0   dist  k=2       e = col0 - col2
    [5.0, 50.0, 4.0],      # 2   a0  e 1
    [10.0, -3.0, 0.0],     # 5   a1  e 10
    [3.0, 40.0, 0.0],      # 7   a0  e 3
    [5.0, -20.0, 2.0],     # 11  a0  e 3
    [-10.0, 7.0, 0.0],     # 13  a1  e -10
    [6.0, 30.0, 8.0],      # 17  a0  e -2
    [-12.0, 1.0, -20.0],   # 19  a1  e 8
    [3.0, -1.0, 0.0],      # 23  a0  e 3
    [7.0, 0.0, 1.0],       # 29  a1  e 6
    [1.0, 60.0, 1.0],      # 31  a0  e 0
    [0.0, -60.0, 2.0],     # 37  a0  e -2
    [5.0, 11.0, 0.0],      # 41  a0  e 5
    [100.0, 0.0, -100.0],  # 43  unanswered
    [-100.0, 0.0, 100.0],  # 47  a2
])
# Group a=0 (8): e 41:5, 7:3, 11:3, 23:3, 2:1, 31:0, 17:-2, 37:-2.
#   column 0: 17:6, 2:5, 11:5, 41:5, 7:3, 23:3, 31:1, 37:0. column 2: 17:8, 2:4, 11:2, 37:2, 31:1, 7/23/41:0.
#   column 0 - column 1: 37:60, 11:25, 23:4, 41:-6, 17:-24, 7:-37, 2:-45, 31:-59.
# Group a=1 (4 -> 2 each): e 5:10, 19:8, 29:6, 13:-10; column 0: 5:10, 29:7, 13:-10, 19:-12.
# Group a=2: photo 47 only -> 0 each; a=3: no photos.
PANEL_F_CASES = [
    # (name, R, ids, ans, a, c, k, n or None, more, less)
    ("group of 8, ties at both cuts", PN_R, PN_IDS, PN_ANS, 0, 0, 2, None, [41, 7, 11], [17, 37, 31]),
    ("group of 8, n=3 by keyword", PN_R, PN_IDS, PN_ANS, 0, 0, 2, 3, [41, 7, 11], [17, 37, 31]),
    ("c and k swapped", PN_R, PN_IDS, PN_ANS, 0, 2, 0, None, [17, 37, 31], [41, 7, 11]),
    ("another peer column", PN_R, PN_IDS, PN_ANS, 0, 0, 1, None, [37, 11, 23], [31, 2, 7]),
    ("n=2", PN_R, PN_IDS, PN_ANS, 0, 0, 2, 2, [41, 7], [17, 37]),
    ("n=1", PN_R, PN_IDS, PN_ANS, 0, 0, 2, 1, [41], [17]),
    ("n=4", PN_R, PN_IDS, PN_ANS, 0, 0, 2, 4, [41, 7, 11, 23], [17, 37, 31, 2]),
    ("group of 4: 2 each", PN_R, PN_IDS, PN_ANS, 1, 0, 2, None, [5, 19], [13, 29]),
    ("group of 1: none", PN_R, PN_IDS, PN_ANS, 2, 0, 2, None, [], []),
    ("empty group", PN_R, PN_IDS, PN_ANS, 3, 0, 2, None, [], []),
]
PANEL_R_CASES = [
    # (name, R, ids, ans, a, c, n or None, most, least)
    ("group of 8, ties at both cuts", PN_R, PN_IDS, PN_ANS, 0, 0, None, [17, 2, 11], [37, 31, 7]),
    ("another column, three tied lowest", PN_R, PN_IDS, PN_ANS, 0, 2, None, [17, 2, 11], [7, 23, 41]),
    ("n=2", PN_R, PN_IDS, PN_ANS, 0, 0, 2, [17, 2], [37, 31]),
    ("n=4", PN_R, PN_IDS, PN_ANS, 0, 0, 4, [17, 2, 11, 41], [37, 31, 7, 23]),
    ("group of 4: 2 each", PN_R, PN_IDS, PN_ANS, 1, 0, None, [5, 29], [19, 13]),
    ("group of 1: none", PN_R, PN_IDS, PN_ANS, 2, 0, None, [], []),
    ("empty group", PN_R, PN_IDS, PN_ANS, 3, 0, None, [], []),
]

# Small groups: ids 10..70, columns [k, c] (c = 1, k = 0).
# c: 10:6, 20:1, 30:4, 40:3, 50:5, 60:2, 70:0; k: 10:0, 20:0, 30:5, 40:-4, 50:0, 60:0, 70:-9;
# e = c - k: 10:6, 20:1, 30:-1, 40:7, 50:5, 60:2, 70:9.
SM_IDS = [10, 20, 30, 40, 50, 60, 70]
SM_R = np.array([[0.0, 6.0], [0.0, 1.0], [5.0, 4.0], [-4.0, 3.0], [0.0, 5.0], [0.0, 2.0], [-9.0, 0.0]])
SM_CASES = [
    # (name, ans, a, (more, less), (most, least))
    ("7 photos: 3 each", [0] * 7, 0, ([70, 40, 10], [30, 20, 60]), ([10, 50, 30], [70, 20, 60])),
    ("6 photos: 3 each", [0, 0, 0, 0, 0, 0, 1], 0, ([40, 10, 50], [30, 20, 60]), ([10, 50, 30], [20, 60, 40])),
    ("5 photos: 2 each", [0, 0, 0, 0, 0, 1, 1], 0, ([40, 10], [30, 20]), ([10, 50], [20, 40])),
    ("4 photos: 2 each", [1, 0, 0, 1, 0, 0, 1], 0, ([50, 60], [30, 20]), ([50, 30], [20, 60])),
    ("4 photos (the other answer): 2 each", [0, 0, 0, 1, 1, 1, 1], 1, ([70, 40], [60, 50]),
     ([50, 40], [70, 60])),
    ("3 photos: 1 each", [0, 0, 0, 1, 1, 1, 1], 0, ([10], [30]), ([10], [20])),
    ("2 photos: 1 each", [0, 0, 0, 0, 0, 1, 1], 1, ([70], [60]), ([60], [70])),
    ("1 photo: none", [None, None, None, None, 0, None, 2], 0, ([], []), ([], [])),
    ("1 photo of another answer: none", [None, None, None, None, 0, None, 2], 2, ([], []), ([], [])),
]


def check_pair(tag, got, names, want):
    if not (isinstance(got, (tuple, list)) and len(got) == 2):
        fail(f"{tag}: got {show(got)}, expected a pair ({names[0]}, {names[1]})")
        return
    for part, g, w in zip(names, got, want):
        ok = isinstance(g, list) and as_ids(g) == w
        check(ok, f"{tag}: {part} is {show(g)}, expected {w!r}")


def test_panel_f():
    cases = list(PANEL_F_CASES) + [(f"small group, {name}", SM_R, SM_IDS, ans, a, 1, 0, None, f_want[0], f_want[1])
                                   for name, ans, a, f_want, _r in SM_CASES]
    for name, R, ids, ans, a, c, k, n, more, less in cases:
        tag = f"panel_f {name} (a={a}, c={c}, k={k}, n={n})"
        kw = {} if n is None else {"n": n}
        raised, got = attempt(lambda: refineloop.panel_f(R.copy(), list(ids), list(ans), a, c, k, **kw))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        check_pair(tag, got, ("more", "less"), (more, less))


def test_panel_r():
    cases = list(PANEL_R_CASES) + [(f"small group, {name}", SM_R, SM_IDS, ans, a, 1, None, r_want[0], r_want[1])
                                   for name, ans, a, _f, r_want in SM_CASES]
    for name, R, ids, ans, a, c, n, most, least in cases:
        tag = f"panel_r {name} (a={a}, c={c}, n={n})"
        kw = {} if n is None else {"n": n}
        raised, got = attempt(lambda: refineloop.panel_r(R.copy(), list(ids), list(ans), a, c, **kw))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        check_pair(tag, got, ("most", "least"), (most, least))


# ---------------------------------------------------------------------------------------------
# parse_refined(text, trigger_answers): the last JSON object with "predictions"; keep, in order, entries
# that pass contrastloop.parse_predictions's entry rule and whose "splits" is a string equal (stripped,
# casefolded) to exactly one of the two compared answers (stripped, casefolded); store as contrastloop
# (question and answers stripped, most / least answer indices) plus "splits" = that answer's index in
# trigger_answers; the first 2 kept; valid when at least 1 kept ("notes" not checked); returns
# {"notes": notes as given or None, "predictions": [...]} or None.
# GAP: parse_refined gets the trigger question's whole answer list, not which two of its answers are
# compared (most and least); so a split equal to a third answer of a longer list is not tested, and a
# split is tested as kept only with 2-answer lists (both answers compared). With longer lists only a
# split matching none of its answers is tested (dropped). Extra keys in the stored predictions, whether
# "notes" is copied or the same object, and bundlemax.last_obj's code-fence handling are not tested.
# ---------------------------------------------------------------------------------------------
NOTES = ["a dog on grass", "a cat on a sofa", "a lamp in a dark room"]
TA2 = ["outdoors", "indoors"]


def rentry(base, **kw):
    e = copy.deepcopy(base)
    for k, v in kw.items():
        if v is MISSING:
            e.pop(k, None)
        else:
            e[k] = v
    return e


def rreply(preds, notes=NOTES, **kw):
    obj = {}
    if notes is not MISSING:
        obj["notes"] = notes
    obj["predictions"] = preds
    return json.dumps(obj, ensure_ascii=False, **kw)


# Valid entries and what they parse to with TA2: (question, answers, most, least, splits).
R_WIN = {"splits": "indoors", "question": "Is there a window?", "answers": ["yes", "no", "not applicable"],
         "most_active": "yes", "least_active": "no"}
W_WIN = ("Is there a window?", ["yes", "no", "not applicable"], 0, 1, 1)
R_SKY = {"splits": "  OUTDOORS \n", "question": " What is the weather?\n",
         "answers": ["sunny", " cloudy", "Rainy ", "no sky"], "most_active": " rainy", "least_active": "SUNNY"}
W_SKY = ("What is the weather?", ["sunny", "cloudy", "Rainy", "no sky"], 2, 0, 0)
R_LAMP = {"splits": "Indoors", "question": "Is a lamp lit?", "answers": ["lit", "unlit", "no lamp"],
          "most_active": "no lamp", "least_active": "lit", "why": "an extra key is allowed"}
W_LAMP = ("Is a lamp lit?", ["lit", "unlit", "no lamp"], 2, 0, 1)
R_SHADE = {"splits": "outdoors", "question": "Is the subject in shade?", "answers": ["shade", "sun"],
           "most_active": "sun", "least_active": "shade"}
W_SHADE = ("Is the subject in shade?", ["shade", "sun"], 1, 0, 0)


def with_split(w, s):
    return w[:4] + (s,)


# Each must be dropped (valid splits; one entry rule broken where possible).
BAD_REFINED_ENTRIES = [
    ("entry a string", "Is there a window?"),
    ("entry null", None),
    ("entry a list", ["indoors", "Is there a window?", ["yes", "no"], "yes", "no"]),
    ("question missing", rentry(R_WIN, question=MISSING)),
    ("question only whitespace", rentry(R_WIN, question="  \n ")),
    ("question a number", rentry(R_WIN, question=3)),
    ("answers a string", rentry(R_WIN, answers="yes; no; not applicable")),
    ("1 answer", rentry(R_WIN, answers=["yes"])),
    ("6 answers", rentry(R_WIN, answers=["yes", "no", "not applicable", "partly", "unclear", "other"])),
    ("an empty answer", rentry(R_WIN, answers=["yes", "no", ""])),
    ("answers repeat after strip and casefold", rentry(R_WIN, answers=["yes", "no", " YES"])),
    ("most_active matches no answer", rentry(R_WIN, most_active="maybe")),
    ("least_active missing", rentry(R_WIN, least_active=MISSING)),
    ("most_active an index", rentry(R_WIN, most_active=0)),
    ("most equal to least after case and spaces", rentry(R_WIN, most_active=" No", least_active="no")),
]
# Each must be dropped with TA2 (the entry is otherwise valid).
BAD_SPLITS = [
    ("splits missing", MISSING),
    ("splits null", None),
    ("splits the index 1", 1),
    ("splits the index 0", 0),
    ("splits a list", ["indoors"]),
    ("splits an object", {"answer": "indoors"}),
    ("splits empty", ""),
    ("splits only spaces", "   "),
    ("splits with an inner space", "in doors"),
    ("splits a prefix", "indoor"),
    ("splits with extra words", "indoors only"),
    ("splits both answers", "outdoors; indoors"),
    ("splits not an answer of the trigger question", "not applicable"),
    ("splits an answer of the new question", "yes"),
    ("splits the new question", "Is there a window?"),
]

PARSE_CASES = [
    # (name, trigger_answers, text, expected notes, expected predictions or None for a None result)
    ("one valid; notes a list", TA2, rreply([R_WIN]), NOTES, [W_WIN]),
    ("splits case and spaces; fields stripped, most / least as indices", TA2, rreply([R_SKY, R_LAMP]),
     NOTES, [W_SKY, W_LAMP]),
    ("3 valid: the first 2 kept", TA2, rreply([R_WIN, R_SKY, R_LAMP]), NOTES, [W_WIN, W_SKY]),
    ("the first 2 kept, not the first 2 entries", TA2,
     rreply([rentry(R_WIN, splits="not applicable"), R_LAMP, BAD_REFINED_ENTRIES[7][1], R_SHADE, R_SKY]),
     NOTES, [W_LAMP, W_SHADE]),
    ("a split that is not a compared answer is dropped, order kept", TA2,
     rreply([rentry(R_LAMP, splits="yes"), R_SHADE, rentry(R_SKY, splits="sunny"), R_WIN]),
     NOTES, [W_SHADE, W_WIN]),
    ("notes missing: None, still valid", TA2, rreply([R_WIN], notes=MISSING), None, [W_WIN]),
    ("notes null", TA2, rreply([R_WIN], notes=None), None, [W_WIN]),
    ("notes a string, as given", TA2, rreply([R_WIN], notes="All indoor scenes."), "All indoor scenes.",
     [W_WIN]),
    ("notes an empty list, as given", TA2, rreply([R_WIN], notes=[]), [], [W_WIN]),
    ("notes a number (notes are not checked)", TA2, rreply([R_WIN], notes=7), 7, [W_WIN]),
    ("an explanation and no notes", TA2, json.dumps({"explanation": "Lamps.", "predictions": [R_WIN]}),
     None, [W_WIN]),
    ("casefold, not lower (ß)", ["Straße", "Gasse"],
     rreply([rentry(R_SHADE, splits="STRASSE"), rentry(R_WIN, splits=" gasse")]),
     NOTES, [with_split(W_SHADE, 0), with_split(W_WIN, 1)]),
    ("the trigger answers stripped and casefolded too", [" Red ", "BLUE"],
     rreply([rentry(R_WIN, splits="blue"), rentry(R_SHADE, splits="red ")]),
     NOTES, [with_split(W_WIN, 1), with_split(W_SHADE, 0)]),
    ("splits is the index in the trigger's list", ["indoors", "outdoors"], rreply([R_WIN, R_SHADE]),
     NOTES, [with_split(W_WIN, 0), with_split(W_SHADE, 1)]),
    ("a 3-answer trigger list: a split matching none of its answers", ["yes", "no", "not applicable"],
     rreply([rentry(R_WIN, splits="maybe")]), None, None),
    ("prose around pretty-printed JSON", TA2,
     "Here are my notes.\n\n" + rreply([R_WIN, R_SHADE], indent=2) + "\n\nI hope this helps.",
     NOTES, [W_WIN, W_SHADE]),
    ("two objects with predictions: the last one", TA2,
     "Draft:\n" + rreply([R_WIN], notes=["first"]) + "\n\nRevised:\n" + rreply([R_SHADE], notes=["last"]),
     ["last"], [W_SHADE]),
    ("a later object without predictions is not the one", TA2,
     rreply([R_WIN]) + '\n{"confidence": "low"}', NOTES, [W_WIN]),
    ("the last object with predictions keeps none", TA2,
     rreply([R_WIN]) + "\n" + rreply([rentry(R_SHADE, splits="nowhere")]), None, None),
    ("every entry dropped", TA2,
     rreply([b for _n, b in BAD_REFINED_ENTRIES[:6]] + [rentry(R_WIN, splits="nowhere")]), None, None),
    ("predictions empty", TA2, rreply([]), None, None),
    ("predictions an object", TA2, rreply(R_WIN), None, None),
    ("predictions a string", TA2, rreply("Is there a window?"), None, None),
    ("predictions null", TA2, rreply(None), None, None),
    ("no predictions key", TA2, json.dumps({"notes": NOTES}), None, None),
    ("empty text", TA2, "", None, None),
    ("no JSON", TA2, "The unit likes lamps.", None, None),
    ("broken JSON", TA2, '{"notes": [], "predictions": [', None, None),
]


def refined_view(got):
    preds = got.get("predictions")
    if not isinstance(preds, list):
        return preds
    out = []
    for p in preds:
        if not isinstance(p, dict):
            out.append(("not a dict", p))
            continue
        a = p.get("answers")
        out.append((p.get("question"), list(a) if isinstance(a, (list, tuple)) else a,
                    p.get("most"), p.get("least"), p.get("splits")))
    return out


def check_refined(tag, got, want_notes, want_preds):
    if want_preds is None:
        check(got is None, f"{tag}: got {show(got)}, expected None")
        return
    if not isinstance(got, dict):
        fail(f"{tag}: got {show(got)}, expected a dict")
        return
    if "notes" not in got:
        fail(f"{tag}: no \"notes\" key in {got!r}")
    else:
        n = got["notes"]
        ok = (n is None) if want_notes is None else (n == want_notes and type(n) is type(want_notes))
        check(ok, f"{tag}: notes {n!r}, expected {want_notes!r}")
    view = refined_view(got)
    check(view == list(want_preds),
          f"{tag}: predictions (question, answers, most, least, splits) {view!r}, expected {list(want_preds)!r}")


def test_parse_refined():
    for name, ta, text, want_notes, want_preds in PARSE_CASES:
        tag = f"parse_refined {name}"
        raised, got = attempt(lambda: refineloop.parse_refined(text, list(ta)))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        check_refined(tag, got, want_notes, want_preds)


def test_parse_refined_bad():
    bad = [(f"bad entry ({n})", b) for n, b in BAD_REFINED_ENTRIES]
    bad += [(f"bad split ({n})", rentry(R_WIN, splits=s)) for n, s in BAD_SPLITS]
    for name, b in bad:
        tag = f"parse_refined {name} before a valid one"
        text = rreply([b, R_SHADE])
        raised, got = attempt(lambda: refineloop.parse_refined(text, list(TA2)))
        if raised:
            fail(f"{tag}: raised {err(got)}")
        else:
            check_refined(tag, got, NOTES, [W_SHADE])
        tag = f"parse_refined {name} alone"
        text = rreply([b])
        raised, got = attempt(lambda: refineloop.parse_refined(text, list(TA2)))
        if raised:
            fail(f"{tag}: raised {err(got)}")
        else:
            check_refined(tag, got, None, None)


# ---------------------------------------------------------------------------------------------
# conditional(parent_ans, child_ans, a): the child answer where the parent answer is a, else None.
# ---------------------------------------------------------------------------------------------
C_PARENT = [0, 1, None, 0, 2, 0, 1]
C_CHILD = [2, 0, 1, None, 1, 0, 3]
COND_CASES = [
    ("a=0 (answer 0 kept, not taken for None)", C_PARENT, C_CHILD, 0, [2, None, None, None, None, 0, None]),
    ("a=1", C_PARENT, C_CHILD, 1, [None, 0, None, None, None, None, 3]),
    ("a=2", C_PARENT, C_CHILD, 2, [None, None, None, None, 1, None, None]),
    ("an answer never given", C_PARENT, C_CHILD, 4, [None] * 7),
    ("no photos", [], [], 0, []),
]


def test_conditional():
    for name, parent, child, a, want in COND_CASES:
        tag = f"conditional {name}"
        raised, got = attempt(lambda: refineloop.conditional(list(parent), list(child), a))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        ok = (isinstance(got, list) and len(got) == len(want)
              and all((g is None) if w is None else (g is not None and as_int(g) == w) for g, w in zip(got, want)))
        check(ok, f"{tag}: got {show(got)}, expected {want!r}")


# ---------------------------------------------------------------------------------------------
# table_text(rows, arm): one block per row, separated by blank lines; lines compared exactly.
# d with a sign and 2 decimals; percentiles whole (only whole values are used: proploop._whole on a
# non-whole value is not in the spec). The too-few form ("Too few photos to check." after the first
# three lines) is read as applying to both arms.
# GAP: a trailing newline is ignored; d that rounds to 0.00 (its sign) and records with fewer than 3
# peers are not tested.
# ---------------------------------------------------------------------------------------------
TT_ROWS_1 = [
    {"question": "Is the sky visible?", "answers": ["yes", "no", "not applicable"], "most": 0, "least": 1,
     "rec": mkrec(1.2345, 97, [(7, 0.9, 96), (3, 0.1049, 60.0), (9, -0.5, 3)]), "kind": "shared", "peer_no": 1},
    {"question": "What colour is the main object?",
     "answers": ["red", "blue", "green", "other", "no main object"], "most": 2, "least": 4,
     "rec": mkrec(0.5, 95, [(4, 0.3051, 94), (12, -0.4449, 12), (2, 2.0, 0)]), "kind": "own", "peer_no": None},
    {"question": "How many animals are there?", "answers": ["none", "one", "several"], "most": 2, "least": 0,
     "rec": None, "kind": "too few", "peer_no": None},
    {"question": "Was the photo taken indoors?", "answers": ["indoors", "outdoors"], "most": 1, "least": 0,
     "rec": mkrec(-0.4449, 3, [(7, 0.1, 50), (3, 12.3456, 100), (9, -1.8974, 0)]), "kind": "differs",
     "peer_no": 2},
    {"question": "Is there water in the photo?", "answers": ["yes", "no"], "most": 0, "least": 1,
     "rec": mkrec(0.25, np.float64(88.0), [(7, 0.2, 80), (3, 0.3, np.float64(90.0)), (9, 0.1, 70)]),
     "kind": "none", "peer_no": None},
]
TT_HEAD_1 = [
    "Prediction 1. Is the sky visible?\n"
    "Answers: yes; no; not applicable\n"
    "Predicted most active: yes. Predicted least active: no.\n",
    "Prediction 2. What colour is the main object?\n"
    "Answers: red; blue; green; other; no main object\n"
    "Predicted most active: green. Predicted least active: no main object.\n",
    "Prediction 3. How many animals are there?\n"
    "Answers: none; one; several\n"
    "Predicted most active: several. Predicted least active: none.\n",
    "Prediction 4. Was the photo taken indoors?\n"
    "Answers: indoors; outdoors\n"
    "Predicted most active: outdoors. Predicted least active: indoors.\n",
    "Prediction 5. Is there water in the photo?\n"
    "Answers: yes; no\n"
    "Predicted most active: yes. Predicted least active: no.\n",
]
TT_F_1 = [
    TT_HEAD_1[0]
    + "This unit: d +1.23, percentile 97. Similar unit 1: d +0.90, percentile 96. Similar unit 2: d +0.10,"
      " percentile 60. Similar unit 3: d -0.50, percentile 3.\n"
      "Held for this unit and for similar unit 1.",
    TT_HEAD_1[1]
    + "This unit: d +0.50, percentile 95. Similar unit 1: d +0.31, percentile 94. Similar unit 2: d -0.44,"
      " percentile 12. Similar unit 3: d +2.00, percentile 0.\n"
      "Held for this unit only.",
    TT_HEAD_1[2]
    + "Too few photos to check.",
    TT_HEAD_1[3]
    + "This unit: d -0.44, percentile 3. Similar unit 1: d +0.10, percentile 50. Similar unit 2: d +12.35,"
      " percentile 100. Similar unit 3: d -1.90, percentile 0.\n"
      "Did not hold for this unit; this unit and similar unit 2 clearly differ on it.",
    TT_HEAD_1[4]
    + "This unit: d +0.25, percentile 88. Similar unit 1: d +0.20, percentile 80. Similar unit 2: d +0.30,"
      " percentile 90. Similar unit 3: d +0.10, percentile 70.\n"
      "Did not hold for this unit, and no similar unit clearly differs from it.",
]
TT_R_1 = [
    TT_HEAD_1[0] + "This unit: d +1.23, percentile 97.\nHeld.",
    TT_HEAD_1[1] + "This unit: d +0.50, percentile 95.\nHeld.",
    TT_HEAD_1[2] + "Too few photos to check.",
    TT_HEAD_1[3] + "This unit: d -0.44, percentile 3.\nDid not hold.",
    TT_HEAD_1[4] + "This unit: d +0.25, percentile 88.\nDid not hold.",
]
TT_ROWS_2 = [
    {"question": "Is it night?", "answers": ["day", "night", "unclear"], "most": 1, "least": 0,
     "rec": None, "kind": "too few", "peer_no": None},
    {"question": "Is there a person?", "answers": ["person", "no person"], "most": 0, "least": 1,
     "rec": mkrec(0.75, 100, [(5, 0.5, 90), (6, 0.25, 94), (8, 1.0, 99)]), "kind": "shared", "peer_no": 3},
    {"question": "Which way does the road turn?", "answers": ["left", "right", "straight", "no road"],
     "most": 3, "least": 2,
     "rec": mkrec(-0.25, 97, [(5, -0.5, 99), (6, 0.5, 98), (8, 1.0, 96)]), "kind": "differs", "peer_no": 3},
]
TT_HEAD_2 = [
    "Prediction 1. Is it night?\n"
    "Answers: day; night; unclear\n"
    "Predicted most active: night. Predicted least active: day.\n",
    "Prediction 2. Is there a person?\n"
    "Answers: person; no person\n"
    "Predicted most active: person. Predicted least active: no person.\n",
    "Prediction 3. Which way does the road turn?\n"
    "Answers: left; right; straight; no road\n"
    "Predicted most active: no road. Predicted least active: straight.\n",
]
TT_F_2 = [
    TT_HEAD_2[0] + "Too few photos to check.",
    TT_HEAD_2[1]
    + "This unit: d +0.75, percentile 100. Similar unit 1: d +0.50, percentile 90. Similar unit 2: d +0.25,"
      " percentile 94. Similar unit 3: d +1.00, percentile 99.\n"
      "Held for this unit and for similar unit 3.",
    TT_HEAD_2[2]
    + "This unit: d -0.25, percentile 97. Similar unit 1: d -0.50, percentile 99. Similar unit 2: d +0.50,"
      " percentile 98. Similar unit 3: d +1.00, percentile 96.\n"
      "Did not hold for this unit; this unit and similar unit 3 clearly differ on it.",
]
TT_R_2 = [
    TT_HEAD_2[0] + "Too few photos to check.",
    TT_HEAD_2[1] + "This unit: d +0.75, percentile 100.\nHeld.",
    TT_HEAD_2[2] + "This unit: d -0.25, percentile 97.\nDid not hold.",
]
TT_CASES = [
    ("shared, own, too few, differs, none", TT_ROWS_1, "F", TT_F_1),
    ("shared, own, too few, differs, none", TT_ROWS_1, "R", TT_R_1),
    ("too few first; similar unit 3; percentile >= 95 with d < 0", TT_ROWS_2, "F", TT_F_2),
    ("too few first; percentile >= 95 with d < 0 did not hold", TT_ROWS_2, "R", TT_R_2),
    ("one row", TT_ROWS_1[3:4], "F", [TT_F_1[3].replace("Prediction 4.", "Prediction 1.")]),
]


def test_table_text():
    for name, rows, arm, blocks in TT_CASES:
        tag = f"table_text arm {arm} {name}"
        raised, got = attempt(lambda: refineloop.table_text(copy.deepcopy(rows), arm))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        if not isinstance(got, str):
            fail(f"{tag}: got {show(got)}, expected a str")
            continue
        paras = got.rstrip("\n").split("\n\n")
        if len(paras) != len(blocks):
            fail(f"{tag}: {len(paras)} blocks separated by blank lines, expected {len(blocks)}: {got!r}")
            continue
        for j, (g, w) in enumerate(zip(paras, blocks), 1):
            check(g == w, f"{tag}: block {j} is {g!r}, expected {w!r}")


# ---------------------------------------------------------------------------------------------
# refine_content(arm, m, table, j, k_no, groups, block): the content list of a refinement call, with a
# fake block. Texts compared exactly: wrapped spec lines joined with single spaces, paragraph breaks as
# blank lines; item 1 is one text block (its paragraphs, {table} as given, the last paragraph).
# GAP: the form of a text block is not given; {"type": "text", "text": ...} is assumed (as in
# test_contrastloop). The default block (block=None) is not tested. A group with empty "hi" and "lo"
# lists is read as keeping both headings (the spec states them unconditionally).
# ---------------------------------------------------------------------------------------------
RC_P1 = "Below is what is known about one unit (a single neuron) of an image classification network."
RC_P2_HEAD = [
    "Someone who looked at photos of the unit's activity wrote predictions about it, and each was checked on other",
    "photos. Someone who never saw the unit's activity answered the prediction's question about each photo. d is how",
    "much more active the unit is on photos given the predicted most-active answer than on photos given the predicted",
    "least-active answer, after removing the differences between photo classes, in standard deviations; the",
    "percentile places it among all {m} units of the unit's layer on the same photos and answers. A prediction held",
    "when its percentile is at least 95.",
]
RC_P2_TAIL = [  # arm F only (arm R drops the last three sentences, from "The same was computed")
    "The same was computed for the 3 units of the layer whose average activity",
    "over photo classes is most like this unit's (similar units 1 to 3). A prediction that held for this unit and also",
    "for a similar unit describes something they share, not what is particular to this unit. A prediction on which",
    "this unit and a similar unit clearly differ points to something particular to one of them.",
]
RC_P3_F = [
    "Prediction {j} is examined below, with similar unit {k}. For each of the two answers it compares, there are photos",
    "given that answer: first some on which this unit is much more active than similar unit {k}, then some on which it",
    "is much less active, each relative to its own average on the photo's class. The percentages are each unit's",
    "activity as a share of the most it reaches on any of 10000 photos.",
]
RC_P3_R = [
    "Prediction {j} is examined below. For each of the two answers it compares, there are photos given that answer:",
    "first those on which the unit is most active, then those on which it is least active, each relative to its",
    "average on the photo's class. The percentages are the unit's activity as a share of the most it reaches on any of",
    "10000 photos.",
]
RC_ASK_F = " ".join([
    'First, in "notes", describe each photo above in a few words. Then, within one of the two answers, find what',
    "separates the photos on which this unit is more active than similar unit {k} from those on which it is less",
    "active, and write it as up to 2 predictions. Each prediction splits one of the two answers: a new question to be",
    "asked about new photos given that answer, with 2 to 5 short answers that do not overlap, so that every photo fits",
    "exactly one of them, including one for photos to which it does not apply; the answer on whose photos you predict",
    "this unit will be most active; and the answer on whose photos you predict it will be least active. Each prediction",
    "will be checked on new photos given the answer it splits, within the same class, against all units of the layer,",
    'and should hold for this unit but not for similar unit {k}. Reply with only JSON: {"notes": ["..."],',
    '"predictions": [{"splits": "...", "question": "...", "answers": ["...", "..."], "most_active": "...",',
    '"least_active": "..."}]}, where "splits" is one of the two answers above, exactly as written.',
])
RC_ASK_SWAPS = [
    ("the photos on which this unit is more active than similar unit {k} from those on which it is less active",
     "the photos on which this unit is more active from those on which it is less active"),
    ("and should hold for this unit but not for similar unit {k}.", "and should hold for this unit."),
]
RC_ASK_R = RC_ASK_F
for _old, _new in RC_ASK_SWAPS:
    RC_ASK_R = RC_ASK_R.replace(_old, _new)


def fill(s, **kw):
    for k, v in kw.items():
        s = s.replace("{" + k + "}", str(v))
    return s


def text1_f(m, table, j, k):
    return "\n\n".join([RC_P1, fill(" ".join(RC_P2_HEAD + RC_P2_TAIL), m=m), table,
                        fill(" ".join(RC_P3_F), j=j, k=k)])


def text1_r(m, table, j):
    return "\n\n".join([RC_P1, fill(" ".join(RC_P2_HEAD), m=m), table, fill(" ".join(RC_P3_R), j=j)])


RC_TABLE_F = "\n\n".join(TT_F_2)
RC_TABLE_R = "\n\n".join(TT_R_2)
RC_GROUPS = [
    {"answer": "outdoors",
     "hi": [(101, "goldfinch", 100, 20), (57, "goldfinch", 88, 0), (3, "Granny Smith", 71, 70)],
     "lo": [(942, "goldfinch", 4, 90), (12, "tiger shark", 9, 33), (77, "goldfinch", 0, 55)]},
    {"answer": "indoors",
     "hi": [(220, "Granny Smith", 65, 3), (18, "golf ball", 30, 1)],
     "lo": [(5, "Granny Smith", 2, 0), (9001, "golf ball", 12, 64)]},
]
RC_WANT_F = [
    ("text", text1_f(512, RC_TABLE_F, 2, 3)),
    ("text", 'Answer 1: "outdoors"'),
    ("text", "This unit much more active than similar unit 3:"),
    ("text", "Photo 1.1 (goldfinch): this unit 100%, similar unit 3 20%"), ("image", 101),
    ("text", "Photo 1.2 (goldfinch): this unit 88%, similar unit 3 0%"), ("image", 57),
    ("text", "Photo 1.3 (Granny Smith): this unit 71%, similar unit 3 70%"), ("image", 3),
    ("text", "This unit much less active than similar unit 3:"),
    ("text", "Photo 1.4 (goldfinch): this unit 4%, similar unit 3 90%"), ("image", 942),
    ("text", "Photo 1.5 (tiger shark): this unit 9%, similar unit 3 33%"), ("image", 12),
    ("text", "Photo 1.6 (goldfinch): this unit 0%, similar unit 3 55%"), ("image", 77),
    ("text", 'Answer 2: "indoors"'),
    ("text", "This unit much more active than similar unit 3:"),
    ("text", "Photo 2.1 (Granny Smith): this unit 65%, similar unit 3 3%"), ("image", 220),
    ("text", "Photo 2.2 (golf ball): this unit 30%, similar unit 3 1%"), ("image", 18),
    ("text", "This unit much less active than similar unit 3:"),
    ("text", "Photo 2.3 (Granny Smith): this unit 2%, similar unit 3 0%"), ("image", 5),
    ("text", "Photo 2.4 (golf ball): this unit 12%, similar unit 3 64%"), ("image", 9001),
    ("text", fill(RC_ASK_F, k=3)),
]
RC_WANT_R = [
    ("text", text1_r(2048, RC_TABLE_R, 1)),
    ("text", 'Answer 1: "outdoors"'),
    ("text", "Most active:"),
    ("text", "Photo 1.1 (goldfinch): 100%"), ("image", 101),
    ("text", "Photo 1.2 (goldfinch): 88%"), ("image", 57),
    ("text", "Photo 1.3 (Granny Smith): 71%"), ("image", 3),
    ("text", "Least active:"),
    ("text", "Photo 1.4 (goldfinch): 4%"), ("image", 942),
    ("text", "Photo 1.5 (tiger shark): 9%"), ("image", 12),
    ("text", "Photo 1.6 (goldfinch): 0%"), ("image", 77),
    ("text", 'Answer 2: "indoors"'),
    ("text", "Most active:"),
    ("text", "Photo 2.1 (Granny Smith): 65%"), ("image", 220),
    ("text", "Photo 2.2 (golf ball): 30%"), ("image", 18),
    ("text", "Least active:"),
    ("text", "Photo 2.3 (Granny Smith): 2%"), ("image", 5),
    ("text", "Photo 2.4 (golf ball): 12%"), ("image", 9001),
    ("text", RC_ASK_R),
]
RC_GROUPS_SMALL = [
    {"answer": "a red car", "hi": [(8, "sports car", 77, 12)], "lo": [(9, "sports car", 3, 41)]},
    {"answer": "no car", "hi": [], "lo": []},
]
RC_WANT_SMALL = [
    ("text", text1_f(64, RC_TABLE_F, 4, 1)),
    ("text", 'Answer 1: "a red car"'),
    ("text", "This unit much more active than similar unit 1:"),
    ("text", "Photo 1.1 (sports car): this unit 77%, similar unit 1 12%"), ("image", 8),
    ("text", "This unit much less active than similar unit 1:"),
    ("text", "Photo 1.2 (sports car): this unit 3%, similar unit 1 41%"), ("image", 9),
    ("text", 'Answer 2: "no car"'),
    ("text", "This unit much more active than similar unit 1:"),
    ("text", "This unit much less active than similar unit 1:"),
    ("text", fill(RC_ASK_F, k=1)),
]
RC_CASES = [
    # (name, (arm, m, table, j, k_no, groups), expected, image count)
    ("arm F", ("F", 512, RC_TABLE_F, 2, 3, RC_GROUPS), RC_WANT_F, 10),
    ("arm R", ("R", 2048, RC_TABLE_R, 1, None, RC_GROUPS), RC_WANT_R, 10),
    ("arm F, 1 + 1 photos and an empty group", ("F", 64, RC_TABLE_F, 4, 1, RC_GROUPS_SMALL), RC_WANT_SMALL, 2),
]


def fake_block(i):
    return {"type": "image", "_test_photo": int(i)}


def classify(item):
    if isinstance(item, dict) and "_test_photo" in item:
        return ("image", item["_test_photo"])
    if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
        return ("text", item["text"])
    return ("other", item)


def test_refine_content_fixture():
    # The arm R ask text must differ from arm F's by exactly the two swaps the spec names.
    for old, _new in RC_ASK_SWAPS:
        check(RC_ASK_F.count(old) == 1, f"refine_content fixture: {old!r} not found once in the arm F text")
    check("{k}" not in RC_ASK_R and "similar unit" not in RC_ASK_R,
          f"refine_content fixture: arm R text still names a similar unit: {RC_ASK_R!r}")


def test_refine_content():
    for name, args, want, n_images in RC_CASES:
        tag = f"refine_content {name}"
        raised, got = attempt(lambda: refineloop.refine_content(*copy.deepcopy(args), block=fake_block))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        if not isinstance(got, list):
            fail(f"{tag}: got {show(got)}, expected a list of content blocks")
            continue
        seq = [classify(x) for x in got]
        n_img = sum(1 for kind_, _ in seq if kind_ == "image")
        check(n_img == n_images, f"{tag}: {n_img} image blocks, expected {n_images}")
        check(len(seq) == len(want), f"{tag}: {len(seq)} blocks, expected {len(want)}")
        for i, (g, w) in enumerate(zip(seq, want)):
            if g != w:
                fail(f"{tag}: block {i} is {g!r}, expected {w!r} (first difference)")
                break
        else:
            if len(seq) > len(want):
                fail(f"{tag}: extra blocks after the last expected one: {seq[len(want):]!r}")


TESTS = [
    test_z_score,
    test_t_record,
    test_kind,
    test_trigger_f,
    test_trigger_r,
    test_panel_f,
    test_panel_r,
    test_parse_refined,
    test_parse_refined_bad,
    test_conditional,
    test_table_text,
    test_refine_content_fixture,
    test_refine_content,
]


def main():
    global refineloop, proploop
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    try:
        import refineloop  # noqa: F401
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        print(f"FAIL import refineloop: {err(e)}".replace("\n", " "))
        sys.exit(1)
    try:
        import proploop as _p
        proploop = _p
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        proploop = None
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
