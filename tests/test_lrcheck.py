"""Black-box tests of tools/lrcheck.py, written from authors/spec_lrcheck.md only (Interface section).

Covers side_of, d_stat and pct_abs. Expected values are worked out by hand from the spec rules:
  left  = facing in {"left", "toward_left", "away_left"}
  right = facing in {"right", "toward_right", "away_right"}; anything else -> None
  d     = (mean v over left - mean v over right) / sd(v over left and right entries together, ddof 1);
          None entries ignored; d = 0 when that sd is 0
  pct_abs(d_all, d) = 100 x share of entries of d_all with |entry| < |d|  (strict)

No model, GPU or network. Plain Python, no pytest.
Run: python -I tests/test_lrcheck.py   -> prints `tests ok` (exit 0) or one line per failure (exit 1).
"""
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tools'))

FAILS = []


def fail(name, detail):
    FAILS.append(f'FAIL {name}: {detail}')


def as_float(x):
    try:
        return float(x)
    except Exception:
        return None


def close(got, want, tol=1e-9):
    g = as_float(got)
    if g is None or not math.isfinite(g):
        return False
    return abs(g - want) <= tol * max(1.0, abs(want))


def check_close(name, got, want, tol=1e-9):
    if not close(got, want, tol):
        fail(name, f'got {got!r}, want {want!r}')


L, R, N = 'left', 'right', None


# ---------------------------------------------------------------- side_of

def test_side_of_left_values():
    for f in ('left', 'toward_left', 'away_left'):
        got = lrcheck.side_of(f)
        if not (isinstance(got, str) and got == 'left'):
            fail('side_of_left', f'side_of({f!r}) = {got!r}, want \'left\'')


def test_side_of_right_values():
    for f in ('right', 'toward_right', 'away_right'):
        got = lrcheck.side_of(f)
        if not (isinstance(got, str) and got == 'right'):
            fail('side_of_right', f'side_of({f!r}) = {got!r}, want \'right\'')


def test_side_of_other_values():
    # Spec: "others unused" -> None. Any value outside the six listed strings, including a JSON null.
    others = ('front', 'back', 'toward', 'away', 'toward_camera', 'away_camera', 'unclear', 'unknown',
              'none', 'sideways', '', None)
    for f in others:
        try:
            got = lrcheck.side_of(f)
        except Exception as e:
            fail('side_of_other', f'side_of({f!r}) raised {type(e).__name__}: {e}')
            continue
        if got is not None:
            fail('side_of_other', f'side_of({f!r}) = {got!r}, want None')


# ---------------------------------------------------------------- d_stat

SQRT_2_4 = math.sqrt(2.4)  # 2 / sqrt(5/3)


def test_d_basic_sign_and_magnitude():
    # left {1,2} mean 1.5, right {3,4} mean 3.5, diff -2.
    # sd over {1,2,3,4}, ddof 1: sum sq dev = 5, var = 5/3 -> d = -2/sqrt(5/3) = -sqrt(2.4) ~ -1.549193.
    # (A within-group pooled sd would give -2/sqrt(0.5) ~ -2.828; ddof 0 would give -2/sqrt(1.25) ~ -1.789.)
    v = np.array([1.0, 2.0, 3.0, 4.0])
    check_close('d_basic', lrcheck.d_stat(v, [L, L, R, R]), -SQRT_2_4)


def test_d_sign_flip():
    # Swapping the labels negates d.
    v = np.array([1.0, 2.0, 3.0, 4.0])
    check_close('d_sign_flip', lrcheck.d_stat(v, [R, R, L, L]), SQRT_2_4)


def test_d_order_independent():
    # Same values and labels as the basic case, interleaved.
    v = np.array([3.0, 1.0, 4.0, 2.0])
    check_close('d_order', lrcheck.d_stat(v, [R, L, R, L]), -SQRT_2_4)


def test_d_none_ignored():
    # The basic case with large outliers at None positions; they must not enter means or sd.
    v = np.array([1.0, 100.0, 2.0, 3.0, -50.0, 4.0])
    check_close('d_none_ignored', lrcheck.d_stat(v, [L, N, L, R, N, R]), -SQRT_2_4)


def test_d_none_ignored_positive():
    # left {0.9,0.7} mean 0.8, right {0.2,0.1} mean 0.15, diff 0.65.
    # sd over {0.9,0.7,0.2,0.1}: mean 0.475, sum sq dev 0.4475, var 0.4475/3 -> d = 0.65/sqrt(0.4475/3) ~ 1.68297.
    v = np.array([0.9, 0.7, 0.2, 0.1, 0.5])
    check_close('d_none_positive', lrcheck.d_stat(v, [L, L, R, R, N]), 0.65 / math.sqrt(0.4475 / 3))


def test_d_unequal_groups_ddof1():
    # left {0,0} mean 0, right {3} mean 3, diff -3.
    # sd over {0,0,3}: mean 1, sum sq dev 6, ddof 1 var 3 -> d = -3/sqrt(3) = -sqrt(3) ~ -1.732051.
    # (ddof 0 would give -3/sqrt(2) ~ -2.121.)
    v = np.array([0.0, 0.0, 3.0])
    check_close('d_unequal_ddof1', lrcheck.d_stat(v, [L, L, R]), -math.sqrt(3.0))


def test_d_two_entries():
    # left {1}, right {3}: diff -2; sd over {1,3} ddof 1 = sqrt(2) -> d = -sqrt(2).
    v = np.array([1.0, 3.0])
    check_close('d_two_entries', lrcheck.d_stat(v, [L, R]), -math.sqrt(2.0))


def test_d_scale_shift_invariant():
    # 10 * [1,2,3,4] + 7: d unchanged.
    v = np.array([17.0, 27.0, 37.0, 47.0])
    check_close('d_scale_shift', lrcheck.d_stat(v, [L, L, R, R]), -SQRT_2_4)


def test_d_float32_input():
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    check_close('d_float32', lrcheck.d_stat(v, [L, L, R, R]), -SQRT_2_4, tol=1e-5)


def test_d_zero_sd():
    # All left/right values equal -> sd 0 -> d = 0 (finite, not nan).
    cases = [
        ('d_zero_sd_4', np.array([0.5, 0.5, 0.5, 0.5]), [L, L, R, R]),
        ('d_zero_sd_2', np.array([2.0, 2.0]), [L, R]),
        ('d_zero_sd_zeros', np.array([0.0, 0.0, 0.0]), [L, R, R]),
        # Nones carry different values; sd over left+right only is still 0.
        ('d_zero_sd_with_none', np.array([5.0, 9.0, 5.0, 5.0, 0.0]), [L, N, R, R, N]),
    ]
    for name, v, sides in cases:
        try:
            got = lrcheck.d_stat(v, sides)
        except Exception as e:
            fail(name, f'raised {type(e).__name__}: {e}')
            continue
        check_close(name, got, 0.0)


def test_d_with_side_of_labels():
    # Sides produced by side_of from raw facing labels; 'front' -> None and is ignored.
    facings = ['toward_left', 'front', 'away_left', 'right', 'unclear', 'toward_right']
    sides = [lrcheck.side_of(f) for f in facings]
    v = np.array([1.0, 40.0, 2.0, 3.0, -7.0, 4.0])
    check_close('d_with_side_of', lrcheck.d_stat(v, sides), -SQRT_2_4)


# ---------------------------------------------------------------- pct_abs

def test_pct_abs_strict():
    d_all = np.array([0.5, -1.0, 2.0, -3.0])  # |.| = 0.5, 1, 2, 3
    cases = [
        (2.0, 50.0),    # 0.5, 1 below; the tie at 2 is not counted (<= would give 75)
        (-2.0, 50.0),   # |d| is used
        (0.0, 0.0),
        (0.5, 0.0),     # tie with 0.5 not counted
        (0.75, 25.0),
        (1.0, 25.0),    # |-1.0| == 1 is not strictly below
        (-3.0, 75.0),
        (3.5, 100.0),
        (10.0, 100.0),
    ]
    for d, want in cases:
        check_close(f'pct_abs(d={d})', lrcheck.pct_abs(d_all, d), want)


def test_pct_abs_ties():
    d_all = np.array([1.0, -1.0, 1.0, -1.0])
    check_close('pct_ties_pos', lrcheck.pct_abs(d_all, 1.0), 0.0)
    check_close('pct_ties_neg', lrcheck.pct_abs(d_all, -1.0), 0.0)
    check_close('pct_ties_above', lrcheck.pct_abs(d_all, 1.0000001), 100.0)


def test_pct_abs_thirds():
    d_all = np.array([1.0, 2.0, 3.0])
    check_close('pct_thirds_2.5', lrcheck.pct_abs(d_all, 2.5), 200.0 / 3.0)
    check_close('pct_thirds_2', lrcheck.pct_abs(d_all, 2.0), 100.0 / 3.0)


def test_pct_abs_percent_scale():
    # Result is on a 0-100 scale, not a 0-1 share.
    check_close('pct_scale', lrcheck.pct_abs(np.array([0.1, 0.2]), 0.15), 50.0)


def test_pct_abs_own_entry():
    # Each channel's own percentile within its layer: self is never strictly below itself.
    d_all = np.array([-0.2, 1.5, 0.3, -0.9, 0.0])  # |.| sorted: 0.0, 0.2, 0.3, 0.9, 1.5
    want = [20.0, 80.0, 40.0, 60.0, 0.0]
    for i, w in enumerate(want):
        check_close(f'pct_own[{i}]', lrcheck.pct_abs(d_all, d_all[i]), w)


TESTS = [
    test_side_of_left_values,
    test_side_of_right_values,
    test_side_of_other_values,
    test_d_basic_sign_and_magnitude,
    test_d_sign_flip,
    test_d_order_independent,
    test_d_none_ignored,
    test_d_none_ignored_positive,
    test_d_unequal_groups_ddof1,
    test_d_two_entries,
    test_d_scale_shift_invariant,
    test_d_float32_input,
    test_d_zero_sd,
    test_d_with_side_of_labels,
    test_pct_abs_strict,
    test_pct_abs_ties,
    test_pct_abs_thirds,
    test_pct_abs_percent_scale,
    test_pct_abs_own_entry,
]


if __name__ == '__main__':
    try:
        import lrcheck
    except Exception as e:
        print(f'FAIL import lrcheck: {type(e).__name__}: {e}')
        sys.exit(1)
    for t in TESTS:
        try:
            t()
        except Exception as e:
            fail(t.__name__, f'raised {type(e).__name__}: {e}')
    if FAILS:
        for line in FAILS:
            print(line)
        sys.exit(1)
    print('tests ok')
