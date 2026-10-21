"""Black-box tests of tools/rangeview.py: bin_of, stratified_pick, top_labels, least_active.

Expected values are derived by hand from authors/spec_rangeview.md (Interface, View S, View N).
Plain Python, no pytest, no GPU, model or Claude. Prints one line per failure, or exactly `tests ok`.

    python -I tests/test_rangeview.py
"""
import os
import sys
import zlib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


def check(cond, msg):
    if not cond:
        fail(msg)


def below(x):
    """The largest float strictly below x."""
    return float(np.nextafter(x, -np.inf))


def err(e):
    return f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------------------------
# bin_of: [0.6, 1.0] -> 0, [0.4, 0.6) -> 1, [0.2, 0.4) -> 2, [0.05, 0.2) -> 3, below 0.05 -> None
# ---------------------------------------------------------------------------------------------
BIN_CASES = [
    (1.0, 0), (below(1.0), 0), (0.999, 0), (0.8, 0),
    (0.6, 0), (below(0.6), 1), (0.599999, 1), (0.5, 1),
    (0.4, 1), (below(0.4), 2), (0.399999, 2), (0.3, 2),
    (0.2, 2), (below(0.2), 3), (0.199999, 3), (0.1, 3),
    (0.05, 3), (below(0.05), None), (0.049999, None), (0.01, None), (0.0, None), (-0.5, None),
]


def test_bin_of():
    for r, want in BIN_CASES:
        for x in (r, np.float64(r)):
            try:
                got = rangeview.bin_of(x)
            except Exception as e:
                fail(f"bin_of({x!r}) raised {err(e)}")
                continue
            if want is None:
                ok = got is None
            else:
                ok = got is not None and not isinstance(got, (bool, np.bool_)) and got == want
            check(ok, f"bin_of({x!r}) = {got!r}, expected {want!r}")


# ---------------------------------------------------------------------------------------------
# stratified_pick
# ---------------------------------------------------------------------------------------------
SEEDS = [0, 1, 2024, zlib.crc32("rangeview-test-unit".encode())]

# (id, r), deliberately not in id order.
POOL = [
    (19, 0.8), (2, 0.4), (31, 0.99), (8, below(0.2)), (14, 0.55), (3, 1.0), (16, 0.0),
    (26, 0.45), (9, below(0.4)), (11, 0.95), (5, 0.3), (20, -0.1), (6, below(0.6)),
    (27, 0.65), (1, 0.2), (22, 0.5), (12, below(0.05)), (15, 0.61), (4, 0.05), (10, 0.5),
    (23, 0.7), (18, 0.42), (7, 0.6),
]
POOL_EXCLUDE = {31, 22, 5, 999}
# Bins by hand, ascending ids, after exclusion (31 from bin 0, 22 from bin 1, 5 from bin 2).
POOL_BINS_EXCL = [
    [3, 7, 11, 15, 19, 23, 27],
    [2, 6, 10, 14, 18, 26],
    [1, 9],
    [4, 8],
]
# Bins by hand with no exclusion.
POOL_BINS_ALL = [
    [3, 7, 11, 15, 19, 23, 27, 31],
    [2, 6, 10, 14, 18, 22, 26],
    [1, 5, 9],
    [4, 8],
]

# Short bins: bin 0 empty after exclusion, bin 1 one photo, bin 2 three, bin 3 eight.
SHORT_POOL = [
    (113, 0.11), (100, 1.0), (105, below(0.4)), (117, 0.05), (121, 0.0), (102, 0.4),
    (111, 0.19), (104, 0.25), (115, 0.1), (101, 0.9), (110, 0.15), (120, 0.04),
    (116, 0.06), (103, 0.2), (112, 0.12), (114, 0.18),
]
SHORT_EXCLUDE = {100, 101}
SHORT_BINS = [
    [],
    [102],
    [103, 104, 105],
    [110, 111, 112, 113, 114, 115, 116, 117],
]

# Nothing at or above 0.05 once exclusion is applied.
EMPTY_POOL = [(7, 1.0), (3, 0.04), (5, 0.0), (9, below(0.05)), (1, -0.2)]
EMPTY_EXCLUDE = {7}
EMPTY_BINS = [[], [], [], []]


def _arrays(pool):
    ids = np.array([i for i, _ in pool], dtype=np.int64)
    r = np.array([x for _, x in pool], dtype=np.float64)
    return ids, r


def _expected_pick(bins, seed, per_bin=5):
    """The spec's rule: one rng, bins in order, rng.choice on the bin's ids in ascending id order,
    min(per_bin, size) without replacement. An empty bin draws nothing."""
    rng = np.random.default_rng(seed)
    out = []
    for b in bins:
        k = min(per_bin, len(b))
        if k == 0:
            out.append([])
            continue
        draw = rng.choice(np.array(sorted(b), dtype=np.int64), size=k, replace=False)
        out.append([int(v) for v in draw])
    return out


def _call_pick(pool, exclude, seed, per_bin=None):
    ids, r = _arrays(pool)
    rng = np.random.default_rng(seed)
    if per_bin is None:
        got = rangeview.stratified_pick(ids, r, set(exclude), rng)
    else:
        got = rangeview.stratified_pick(ids, r, set(exclude), rng, per_bin=per_bin)
    return [[int(v) for v in b] for b in got]


def _check_pick(name, pool, exclude, bins, seed, per_bin=None):
    tag = f"stratified_pick {name} seed={seed}" + ("" if per_bin is None else f" per_bin={per_bin}")
    try:
        got = _call_pick(pool, exclude, seed, per_bin)
    except Exception as e:
        fail(f"{tag}: raised {err(e)}")
        return None
    if len(got) != 4:
        fail(f"{tag}: returned {len(got)} lists, expected 4")
        return got
    pb = 5 if per_bin is None else per_bin
    for j, (g, b) in enumerate(zip(got, bins)):
        want_n = min(pb, len(b))
        check(len(g) == want_n, f"{tag}: bin {j} has {len(g)} ids, expected {want_n}")
        check(len(set(g)) == len(g), f"{tag}: bin {j} repeats ids: {g}")
        excluded = sorted(set(g) & set(exclude))
        check(not excluded, f"{tag}: bin {j} contains excluded ids {excluded}")
        stray = sorted(set(g) - set(b) - set(exclude))
        check(not stray, f"{tag}: bin {j} contains ids outside its bin {stray} (bin is {b})")
    want = _expected_pick(bins, seed, pb)
    if got != want:
        same_sets = [sorted(g) for g in got] == [sorted(w) for w in want]
        note = " (same ids per bin, different order)" if same_sets else ""
        fail(f"{tag}: got {got}, expected rng.choice draws {want}{note}")
    return got


def test_pick_with_exclusion():
    for seed in SEEDS:
        _check_pick("pool+exclude", POOL, POOL_EXCLUDE, POOL_BINS_EXCL, seed)


def test_pick_without_exclusion():
    for seed in SEEDS:
        _check_pick("pool, no exclude", POOL, set(), POOL_BINS_ALL, seed)


def test_pick_short_bins():
    for seed in SEEDS:
        _check_pick("short bins", SHORT_POOL, SHORT_EXCLUDE, SHORT_BINS, seed)


def test_pick_per_bin():
    for seed in SEEDS:
        _check_pick("pool+exclude", POOL, POOL_EXCLUDE, POOL_BINS_EXCL, seed, per_bin=2)
        _check_pick("short bins", SHORT_POOL, SHORT_EXCLUDE, SHORT_BINS, seed, per_bin=2)
        _check_pick("pool, no exclude", POOL, set(), POOL_BINS_ALL, seed, per_bin=8)


def test_pick_empty():
    for seed in SEEDS[:2]:
        _check_pick("all below 0.05 or excluded", EMPTY_POOL, EMPTY_EXCLUDE, EMPTY_BINS, seed)


def test_pick_determinism():
    for seed in SEEDS:
        tag = f"stratified_pick determinism seed={seed}"
        try:
            a = _call_pick(POOL, POOL_EXCLUDE, seed)
            b = _call_pick(POOL, POOL_EXCLUDE, seed)
            # Same photos given in a different input order: draws use ascending id order, so equal.
            c = _call_pick(list(reversed(POOL)), POOL_EXCLUDE, seed)
            d = _call_pick(sorted(POOL), POOL_EXCLUDE, seed)
        except Exception as e:
            fail(f"{tag}: raised {err(e)}")
            continue
        check(a == b, f"{tag}: two calls with fresh rngs of the same seed differ: {a} vs {b}")
        check(a == c, f"{tag}: result depends on input order (given vs reversed): {a} vs {c}")
        check(a == d, f"{tag}: result depends on input order (given vs id-sorted): {a} vs {d}")


# ---------------------------------------------------------------------------------------------
# top_labels: most common first; ties by higher summed r, then lower label index; up to k.
# All r values are dyadic so every sum is exact in floating point.
# ---------------------------------------------------------------------------------------------
# Counts 7:6 (sum 3.0), 3:5 (2.5), 9:4 (4.0), 1:3 (0.375), 4:1 (0.75), 2:1 (0.5).
TOP_A = [
    (9, 1.0), (7, 0.5), (2, 0.5), (3, 0.5), (7, 0.5), (9, 1.0), (1, 0.125), (3, 0.5),
    (7, 0.5), (4, 0.75), (9, 1.0), (3, 0.5), (7, 0.5), (1, 0.125), (3, 0.5), (9, 1.0),
    (7, 0.5), (3, 0.5), (1, 0.125), (7, 0.5),
]
# Counts 40:4 (3.5), 30:4 (2.0), 12:4 (1.5), 5:4 (1.5), 8:2 (2.0), 99:2 (1.0).
TOP_B = [
    (12, 0.5), (99, 0.5), (40, 0.875), (5, 0.75), (30, 0.5), (12, 0.25), (8, 1.0),
    (5, 0.25), (40, 0.875), (30, 0.5), (12, 0.5), (5, 0.25), (99, 0.5), (40, 0.875),
    (30, 0.5), (12, 0.25), (8, 1.0), (5, 0.25), (30, 0.5), (40, 0.875),
]
# Three distinct labels: 2:10 (5.0), 1:5 (2.5), 0:5 (2.5).
TOP_C = [
    (2, 0.5), (1, 0.5), (1, 0.5), (2, 0.5), (0, 0.5), (2, 0.5), (1, 0.5), (2, 0.5),
    (0, 0.5), (2, 0.5), (1, 0.5), (0, 0.5), (2, 0.5), (2, 0.5), (1, 0.5), (0, 0.5),
    (2, 0.5), (0, 0.5), (2, 0.5), (2, 0.5),
]
TOP_D = [(17, 1.0 - j / 64.0) for j in range(20)]

TOP_CASES = [
    # count dominates sum (9 has the largest sum, 4 beats 2 on sum despite higher index)
    ("A", TOP_A, None, [7, 3, 9, 1, 4]),
    ("A", TOP_A, 5, [7, 3, 9, 1, 4]),
    ("A", TOP_A, 6, [7, 3, 9, 1, 4, 2]),
    ("A", TOP_A, 1, [7]),
    # sum tie-break (30 before 5 and 12), then lower index (5 before 12)
    ("B", TOP_B, None, [40, 30, 5, 12, 8]),
    ("B", TOP_B, 3, [40, 30, 5]),
    ("B", TOP_B, 4, [40, 30, 5, 12]),
    ("B", TOP_B, 6, [40, 30, 5, 12, 8, 99]),
    # fewer than k distinct labels; label 0 on a full tie with label 1
    ("C", TOP_C, None, [2, 0, 1]),
    ("C", TOP_C, 2, [2, 0]),
    ("D", TOP_D, None, [17]),
]


def test_top_labels():
    for name, data, k, want in TOP_CASES:
        tag = f"top_labels {name} k={'default' if k is None else k}"
        if len(data) != 20:
            fail(f"{tag}: test data has {len(data)} photos, not 20 (test bug)")
            continue
        labels = [int(a) for a, _ in data]
        rs = [float(b) for _, b in data]
        try:
            if k is None:
                got = rangeview.top_labels(labels, rs)
            else:
                got = rangeview.top_labels(labels, rs, k=k)
            got = [int(x) for x in got]
        except Exception as e:
            fail(f"{tag}: raised {err(e)}")
            continue
        check(got == want, f"{tag}: got {got}, expected {want}")


# ---------------------------------------------------------------------------------------------
# least_active: the n ids of that label with the lowest value, ties by lower id, excluding exclude.
# ---------------------------------------------------------------------------------------------
# (id, label, value), not in id order; other labels hold the lowest values overall.
LA_POOL = [
    (50, 3, 0.2), (10, 3, 0.5), (30, 3, 0.1), (60, 1, -5.0), (20, 3, 0.1), (90, 0, -1.0),
    (40, 3, 0.1), (15, 0, 2.0), (70, 1, -9.0), (80, 3, 0.3), (25, 0, -1.0),
]
LA_CASES = [
    # (label, exclude, n, expected)
    (3, set(), None, [20, 30]),
    (3, set(), 2, [20, 30]),
    (3, {20}, 2, [30, 40]),
    (3, {20, 30, 40}, 2, [50, 80]),
    (3, {30, 70, 25}, 2, [20, 40]),
    (3, set(), 3, [20, 30, 40]),
    (3, set(), 4, [20, 30, 40, 50]),
    (3, {20}, 1, [30]),
    (1, set(), 2, [70, 60]),
    (0, set(), 2, [25, 90]),
    (0, {25}, 2, [90, 15]),
    (0, {90, 60}, None, [25, 15]),
]


def test_least_active():
    ids = np.array([i for i, _, _ in LA_POOL], dtype=np.int64)
    labels = np.array([lab for _, lab, _ in LA_POOL], dtype=np.int64)
    values = np.array([v for _, _, v in LA_POOL], dtype=np.float64)
    for label, exclude, n, want in LA_CASES:
        tag = f"least_active label={label} exclude={sorted(exclude)} n={'default' if n is None else n}"
        try:
            if n is None:
                got = rangeview.least_active(ids.copy(), values.copy(), labels.copy(), label, set(exclude))
            else:
                got = rangeview.least_active(ids.copy(), values.copy(), labels.copy(), label, set(exclude),
                                             n=n)
            got = [int(x) for x in got]
        except Exception as e:
            fail(f"{tag}: raised {err(e)}")
            continue
        check(got == want, f"{tag}: got {got}, expected {want}")


TESTS = [
    test_bin_of,
    test_pick_with_exclusion,
    test_pick_without_exclusion,
    test_pick_short_bins,
    test_pick_per_bin,
    test_pick_empty,
    test_pick_determinism,
    test_top_labels,
    test_least_active,
]


def main():
    global rangeview
    try:
        import rangeview  # noqa: F401
    except Exception as e:
        print(f"FAIL import rangeview: {err(e)}".replace("\n", " "))
        sys.exit(1)
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
