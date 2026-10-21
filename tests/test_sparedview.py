"""Black-box tests of the Interface of tools/sparedview.py: pick_tstar, select_pairs, drop_of, strip_text.

Written from authors/spec_sparedview.md only (the implementation was not read). Every expected value below
was worked out by hand from the spec's rules. Plain Python, no pytest, no GPU, model or Claude.

Usage:
    python -I tests/test_sparedview.py

Prints one line per failure and exits 1, or prints exactly `tests ok`.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

sv = None  # the sparedview module, set in main()
FAILURES = []


def one_line(s):
    return " ".join(str(s).split())


def fail(test, msg):
    FAILURES.append(f"FAIL {test}: {one_line(msg)}")


def check(test, cond, msg):
    if not cond:
        fail(test, msg)


def ints(xs):
    return [int(x) for x in xs]


def row(name, u, p):
    return {"name": name, "u": u, "p": p}


def pick_is(got, name, p):
    """True when pick_tstar returned the pair (name, p)."""
    if got is None:
        return False
    try:
        g_name, g_p = got
    except (TypeError, ValueError):
        return False
    try:
        return g_name == name and abs(float(g_p) - p) < 1e-9
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------------------------------------------
# pick_tstar(rows): lowest percentile; ties by lower retention (u), then list (battery) order; None if lowest p > 20.
# ---------------------------------------------------------------------------------------------------------------

def test_pick_lowest_percentile():
    t = "pick_lowest_percentile"
    rows = [row("blur", 0.50, 40.0), row("gray", 0.30, 12.0), row("jpeg", 0.80, 55.0)]
    got = sv.pick_tstar(rows)
    check(t, pick_is(got, "gray", 12.0), f"got {got!r}, expected ('gray', 12.0)")
    # the lowest percentile wins even when another row has lower retention
    rows = [row("blur", 0.05, 15.0), row("gray", 0.90, 3.0)]
    got = sv.pick_tstar(rows)
    check(t, pick_is(got, "gray", 3.0), f"got {got!r}, expected ('gray', 3.0)")
    rows = [row("x", 0.90, 0.0)]
    got = sv.pick_tstar(rows)
    check(t, pick_is(got, "x", 0.0), f"got {got!r}, expected ('x', 0.0)")


def test_pick_threshold_inclusive():
    t = "pick_threshold_inclusive"
    rows = [row("blur", 0.50, 35.0), row("gray", 0.60, 20.0), row("jpeg", 0.70, 60.0)]
    got = sv.pick_tstar(rows)
    check(t, pick_is(got, "gray", 20.0), f"percentile exactly 20.0: got {got!r}, expected ('gray', 20.0)")
    rows = [row("a", 0.50, 20)]
    got = sv.pick_tstar(rows)
    check(t, pick_is(got, "a", 20.0), f"percentile exactly 20 (int): got {got!r}, expected ('a', 20)")
    rows = [row("a", 0.50, 30.0), row("b", 0.50, 19.9)]
    got = sv.pick_tstar(rows)
    check(t, pick_is(got, "b", 19.9), f"percentile 19.9: got {got!r}, expected ('b', 19.9)")


def test_pick_none_above_threshold():
    t = "pick_none_above_threshold"
    rows = [row("blur", 0.50, 20.5), row("gray", 0.10, 33.0)]
    got = sv.pick_tstar(rows)
    check(t, got is None, f"lowest percentile 20.5: got {got!r}, expected None")
    rows = [row("a", 0.10, 21.0), row("b", 0.20, 80.0), row("c", 0.30, 99.0)]
    got = sv.pick_tstar(rows)
    check(t, got is None, f"lowest percentile 21.0: got {got!r}, expected None")


def test_pick_tie_lower_retention():
    t = "pick_tie_lower_retention"
    # A and B tie on p = 10; B has lower u. C has the lowest u but a higher p, so it does not compete.
    rows = [row("A", 0.60, 10.0), row("B", 0.40, 10.0), row("C", 0.20, 30.0)]
    got = sv.pick_tstar(rows)
    check(t, pick_is(got, "B", 10.0), f"got {got!r}, expected ('B', 10.0)")
    rows = [row("B", 0.40, 10.0), row("A", 0.60, 10.0)]
    got = sv.pick_tstar(rows)
    check(t, pick_is(got, "B", 10.0), f"lower-u row first: got {got!r}, expected ('B', 10.0)")
    # tie exactly at the threshold, broken by retention
    rows = [row("D", 0.50, 25.0), row("E", 0.30, 20.0), row("F", 0.10, 20.0)]
    got = sv.pick_tstar(rows)
    check(t, pick_is(got, "F", 20.0), f"tie at 20: got {got!r}, expected ('F', 20.0)")


def test_pick_tie_list_order():
    t = "pick_tie_list_order"
    rows = [row("A", 0.40, 10.0), row("B", 0.40, 10.0)]
    got = sv.pick_tstar(rows)
    check(t, pick_is(got, "A", 10.0), f"got {got!r}, expected ('A', 10.0)")
    # list order, not alphabetical order
    rows = [row("B", 0.40, 10.0), row("A", 0.40, 10.0)]
    got = sv.pick_tstar(rows)
    check(t, pick_is(got, "B", 10.0), f"got {got!r}, expected ('B', 10.0)")
    rows = [row("D", 0.50, 25.0), row("E", 0.30, 20.0), row("F", 0.30, 20.0)]
    got = sv.pick_tstar(rows)
    check(t, pick_is(got, "E", 20.0), f"full tie at 20: got {got!r}, expected ('E', 20.0)")


# ---------------------------------------------------------------------------------------------------------------
# drop_of(a, b) = 1 - b / a, elementwise
# ---------------------------------------------------------------------------------------------------------------

def test_drop_of():
    t = "drop_of"
    a = np.array([10.0, 4.0, 2.0, 8.0, 3.0])
    b = np.array([5.0, 4.0, 3.0, 0.0, 1.0])
    exp = np.array([0.5, 0.0, -0.5, 1.0, 2.0 / 3.0])
    got = np.asarray(sv.drop_of(a, b), dtype=float)
    check(t, got.shape == exp.shape, f"shape {got.shape}, expected {exp.shape}")
    if got.shape == exp.shape:
        check(t, np.allclose(got, exp, rtol=1e-9, atol=1e-12), f"got {got.tolist()}, expected {exp.tolist()}")


# ---------------------------------------------------------------------------------------------------------------
# strip_text(k, q) = 'activation k, loses q%' with k as given and q the drop rounded to a whole percentage
# ---------------------------------------------------------------------------------------------------------------

def test_strip_text():
    t = "strip_text"
    cases = [
        ((7, 0.634), "activation 7, loses 63%"),   # the spec's example
        ((7, 0.636), "activation 7, loses 64%"),   # rounds, does not truncate
        ((3, 0.29), "activation 3, loses 29%"),    # 0.29 * 100 = 28.999...; rounds to 29
        ((4, 0.07), "activation 4, loses 7%"),
        ((5, 0.999), "activation 5, loses 100%"),
        ((2, 0.004), "activation 2, loses 0%"),
        ((0, 0.0), "activation 0, loses 0%"),
        ((10, 1.0), "activation 10, loses 100%"),
        ((6, 0.5), "activation 6, loses 50%"),
        ((7, np.float64(0.634)), "activation 7, loses 63%"),  # a drop taken from a numpy array
    ]
    for args, exp in cases:
        got = sv.strip_text(*args)
        check(t, got == exp, f"strip_text{args!r} = {got!r}, expected {exp!r}")


# ---------------------------------------------------------------------------------------------------------------
# select_pairs(a, drop, n=10)
#   affected = the n with the largest drop (ties: higher a), in that order;
#   candidates = the third of X with the smallest drop, excluding affected;
#   for each affected in order, the unused candidate with the smallest |log a - log a_affected|.
# All hand arrays have len(X) divisible by 3 and no drop ties at the third's boundary.
# ---------------------------------------------------------------------------------------------------------------

def test_select_third_and_greedy_order():
    t = "select_third_and_greedy_order"
    # X = 12, third = 4 -> candidates {3, 4, 5, 6}. Affected (n = 2): 1 (0.90), 2 (0.80).
    # Affected 1 (a 10):   3 (a 11) |log| 0.0953 beats 4 (a 9) 0.1054 -> 3.
    # Affected 2 (a 11.5): 3 used; 4 (a 9) 0.2451 beats 5 (a 50) 1.4697, 6 (a 1) 2.4423 -> 4.
    # (A globally optimal assignment would give 1 -> 4, 2 -> 3. Photo 0 (a 10) lies outside the third.)
    a = np.array([10.0, 10.0, 11.5, 11.0, 9.0, 50.0, 1.0, 10.2, 11.4, 20.0, 30.0, 2.0])
    drop = np.array([0.30, 0.90, 0.80, 0.02, 0.03, 0.04, 0.05, 0.20, 0.25, 0.40, 0.50, 0.60])
    aff, spr = sv.select_pairs(a, drop, n=2)
    aff, spr = ints(aff), ints(spr)
    check(t, len(aff) == 2 and len(spr) == 2, f"lengths {len(aff)}, {len(spr)}, expected 2, 2")
    check(t, set(aff) == {1, 2}, f"affected set {sorted(set(aff))}, expected [1, 2]")
    check(t, aff == [1, 2], f"affected {aff}, expected [1, 2] (largest drop first)")
    check(t, set(spr) <= {3, 4, 5, 6}, f"spared {spr} not all inside the smallest-drop third {{3, 4, 5, 6}}")
    check(t, spr == [3, 4], f"spared {spr}, expected [3, 4]")


def test_select_affected_ties_higher_a():
    t = "select_affected_ties_higher_a"
    # X = 12, n = 3. Largest drop: 2 (0.95); then 0, 1, 3 tie at 0.70 -> higher a: 1 (8.0), 3 (6.0); 0 (5.0) out.
    # Third = {8, 9, 10, 11}.
    # Affected 2 (a 3.0): 9 (a 3.5) 0.154 -> 9.
    # Affected 1 (a 8.0): 10 (a 7.5) 0.0645 beats 8 (a 6.5) 0.2076, 11 (a 5.5) 0.3747 -> 10.
    # Affected 3 (a 6.0): 8 (a 6.5) 0.0800 beats 11 (a 5.5) 0.0870 -> 8.
    a = np.array([5.0, 8.0, 3.0, 6.0, 7.0, 4.0, 9.0, 2.0, 6.5, 3.5, 7.5, 5.5])
    drop = np.array([0.70, 0.70, 0.95, 0.70, 0.60, 0.50, 0.40, 0.30, 0.01, 0.02, 0.03, 0.04])
    aff, spr = sv.select_pairs(a, drop, n=3)
    aff, spr = ints(aff), ints(spr)
    check(t, len(aff) == 3 and len(spr) == 3, f"lengths {len(aff)}, {len(spr)}, expected 3, 3")
    check(t, set(aff) == {1, 2, 3}, f"affected set {sorted(set(aff))}, expected [1, 2, 3]")
    check(t, aff == [2, 1, 3], f"affected {aff}, expected [2, 1, 3]")
    check(t, spr == [9, 10, 8], f"spared {spr}, expected [9, 10, 8]")


def test_select_log_metric():
    t = "select_log_metric"
    # X = 9, n = 1. Affected 0 (a 10). Smallest-drop candidates: 1 (a 5), 2 (a 16), 3 (a 100).
    # |log| difference: 2 -> 0.470 beats 1 -> 0.693 (a linear difference would pick 1: 5 < 6).
    # Photos 4..8 (a 10) lie outside the third.
    a = np.array([10.0, 5.0, 16.0, 100.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    drop = np.array([0.90, 0.01, 0.02, 0.03, 0.30, 0.40, 0.50, 0.60, 0.70])
    aff, spr = sv.select_pairs(a, drop, n=1)
    aff, spr = ints(aff), ints(spr)
    check(t, aff == [0], f"affected {aff}, expected [0]")
    check(t, spr == [2], f"spared {spr}, expected [2]")


def test_select_no_reuse():
    t = "select_no_reuse"
    # X = 12, n = 2. Affected 0, 1 (both a 10). Third = {2 (a 10), 3 (a 20), 4 (a 40), 5 (a 1000)}.
    # Affected 0 -> 2 (exact). Affected 1: 2 is used -> 3 (log 2) beats 4 (log 4), 5 (log 100).
    a = np.array([10.0, 10.0, 10.0, 20.0, 40.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    drop = np.array([0.90, 0.80, 0.01, 0.02, 0.03, 0.04, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
    aff, spr = sv.select_pairs(a, drop, n=2)
    aff, spr = ints(aff), ints(spr)
    check(t, aff == [0, 1], f"affected {aff}, expected [0, 1]")
    check(t, len(set(spr)) == len(spr), f"spared {spr} reuses a photo")
    check(t, spr == [2, 3], f"spared {spr}, expected [2, 3]")


def test_select_excludes_affected():
    t = "select_excludes_affected"
    # X = 18, n = 3. Photo 0 has drop 0.8; the other 17 tie at 0.1. Affected: 0, then the tied photos with the
    # highest a: 1 (a 100), 17 (a 90). Which 6 tied photos form the smallest-drop third is not fixed by the spec,
    # so only properties are checked: whatever the third, affected photos 1 and 17 must not be offered as spared.
    a = np.array([5.0, 100.0] + [1.0 + 0.5 * i for i in range(2, 17)] + [90.0])
    drop = np.array([0.8] + [0.1] * 17)
    aff, spr = sv.select_pairs(a, drop, n=3)
    aff, spr = ints(aff), ints(spr)
    check(t, aff == [0, 1, 17], f"affected {aff}, expected [0, 1, 17]")
    check(t, len(spr) == 3, f"spared {spr}, expected 3 photos")
    check(t, set(spr).isdisjoint(set(aff)), f"spared {spr} overlaps affected {aff}")
    check(t, len(set(spr)) == len(spr), f"spared {spr} reuses a photo")
    check(t, all(0 <= i < 18 for i in spr), f"spared {spr} has an index outside X")


def test_select_default_n():
    t = "select_default_n"
    # X = 30, a[i] = 1 + i, drop[i] = (7 i mod 30) / 30 (all distinct). Default n = 10, third = 10.
    # Affected (drop 29/30 down to 20/30): 17, 4, 21, 8, 25, 12, 29, 16, 3, 20.
    # Third (drop 0..9/30): {0, 1, 5, 9, 13, 14, 18, 22, 26, 27}, which the 10 spared must use exactly once each.
    # Greedy in affected order (affected a -> spared a): 18->19, 5->6, 22->23, 9->10, 26->27, 13->14, 30->28,
    # 17->15, 4->2, 21->1.
    a = np.array([1.0 + i for i in range(30)])
    drop = np.array([((7 * i) % 30) / 30.0 for i in range(30)])
    aff, spr = sv.select_pairs(a, drop)
    aff, spr = ints(aff), ints(spr)
    exp_aff = [17, 4, 21, 8, 25, 12, 29, 16, 3, 20]
    third = {0, 1, 5, 9, 13, 14, 18, 22, 26, 27}
    exp_spr = [18, 5, 22, 9, 26, 13, 27, 14, 1, 0]
    check(t, len(aff) == 10 and len(spr) == 10, f"lengths {len(aff)}, {len(spr)}, expected 10, 10")
    check(t, set(aff) == set(exp_aff), f"affected set {sorted(set(aff))}, expected {sorted(exp_aff)}")
    check(t, aff == exp_aff, f"affected {aff}, expected {exp_aff}")
    check(t, set(spr) == third, f"spared set {sorted(set(spr))}, expected {sorted(third)}")
    check(t, spr == exp_spr, f"spared {spr}, expected {exp_spr}")


TESTS = [
    test_pick_lowest_percentile,
    test_pick_threshold_inclusive,
    test_pick_none_above_threshold,
    test_pick_tie_lower_retention,
    test_pick_tie_list_order,
    test_drop_of,
    test_strip_text,
    test_select_third_and_greedy_order,
    test_select_affected_ties_higher_a,
    test_select_log_metric,
    test_select_no_reuse,
    test_select_excludes_affected,
    test_select_default_n,
]


def main():
    global sv
    try:
        import sparedview
    except Exception as e:  # noqa: BLE001 - any import failure is a test failure
        print(f"FAIL import sparedview: {one_line(type(e).__name__ + ': ' + str(e))}")
        sys.exit(1)
    sv = sparedview
    for test in TESTS:
        try:
            test()
        except Exception as e:  # noqa: BLE001 - an exception is reported as one failure line
            fail(test.__name__, f"raised {type(e).__name__}: {e}")
    for line in FAILURES:
        print(line)
    if FAILURES:
        sys.exit(1)
    print("tests ok")


if __name__ == "__main__":
    main()
