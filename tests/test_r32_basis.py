"""Black-box tests for tools/r32_basis.py (draw, project_max, joint_d, excess_kurtosis, candidate, summary, selftest).

Written from authors/spec_r32_basis.interface.md only; the implementation and the full spec were not read. Expected
values are computed with the functions the interface names (pairtable.unit_photos, table_rows, pick, NULL_D, NULL_T,
MIN_N; proploop.centre, permute_within, percentile; contrastloop.peers, contrast, outcome), following its steps, and
with the interface's own formulas for draw, project_max, joint_d, excess_kurtosis and summary written out here.

No real pool, model, GPU or network: synthetic arrays and hand-made answer rows only.
- draw: the rng steps of the interface; several layers, sizes and exclusions.
- project_max: an explicit loop on small random F (and all-negative projections, one position, float32 values).
- joint_d: 80 photos, 3 questions. Answer x ("yes" to question 0) comes with answer y ("yes" to question 1) except on 8
  photos; column 0 depends only on y, column 1 only on x, column 6 is exactly y's indicator; columns 3 and 4 are
  constant; question 2's third answer has 3 photos (no column at min_n 5). Every question is unanswered on some photos
  (3 photos on all), so the intercept and min_n change the fit (guarded by test_fixture_joint_case).
- candidate: a sub-pool of 30 classes x 8 photos, 21 channels; the unit (channel 3) follows its class profile plus
  "red" on question 1; question 0 is confounded with "red"; question 3's third answer is given to 6 photos of D and 2
  of T (rows that are "too few" on T, and a joint_d of None on T); 5 photos have no answers; a direction candidate is
  the 21 channels with a 22nd column appended.
- summary: hand-made records over 2 layers x 3 groups with distinct numbers per cell. The structure of the summary is
  not fixed by the interface, so the tests look for any dict carrying the nine listed keys, named by layer and group
  (path keys or its own string fields) or by group and no layer.
"""

import copy
import json
import statistics
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import contrastloop  # noqa: E402
import pairtable  # noqa: E402
import proploop  # noqa: E402
import r32_basis  # noqa: E402


# ---------------------------------------------------------------------------
# comparison (as tests/test_r31_tables.py, with a tolerance argument)
# ---------------------------------------------------------------------------

MISSING = "<missing>"


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def _cmp(got, want, path, rel, abs_):
    if isinstance(want, dict):
        assert isinstance(got, dict), f"{path}: {got!r} is not a dict"
        assert set(got) == set(want), f"{path}: keys {sorted(map(str, got))} != {sorted(map(str, want))}"
        for k in want:
            _cmp(got[k], want[k], f"{path}[{k!r}]", rel, abs_)
    elif isinstance(want, list):
        assert isinstance(got, list), f"{path}: {got!r} is not a list"
        assert len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            _cmp(g, w, f"{path}[{i}]", rel, abs_)
    elif want is None:
        assert got is None, f"{path}: {got!r} != None"
    elif isinstance(want, bool):
        assert isinstance(got, bool) and got == want, f"{path}: {got!r} != {want!r}"
    elif isinstance(want, float):
        assert isinstance(got, (int, float)) and not isinstance(got, bool), f"{path}: {got!r} != {want!r}"
        assert got == pytest.approx(want, rel=rel, abs=abs_), f"{path}: {got!r} != {want!r}"
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


def _same(got, want, path, rel=1e-9, abs_=1e-12):
    _cmp(_plain(got), _plain(want), path, rel, abs_)


def _fields(r, keys):
    return {k: r.get(k, MISSING) for k in keys}


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------

DRAWS = [
    ("mixed4c", 512, [3, 17, 200, 511], 40),
    ("mixed3a", 30, [0, 5, 29], 10),
    ("mixed4a", 10, [2, 4], 8),  # every channel left is drawn
    ("mixed5b", 12, [], 12),  # nothing excluded, every channel drawn
]


def _draw_ref(layer, m, exclude, k):
    """the interface's steps: one rng seeded by crc32('r32:' + layer); the channels, then W from the same rng"""
    rng = np.random.default_rng(zlib.crc32(("r32:" + layer).encode()))
    left = [c for c in range(m) if c not in set(exclude)]
    chans = sorted(int(x) for x in rng.choice(left, size=k, replace=False))
    W = rng.standard_normal((m, k))
    return chans, W / np.linalg.norm(W, axis=0)


def test_fixture_draw_W_comes_after_the_choice_from_one_rng():
    layer, m, exclude, k = DRAWS[0]
    chans, W = _draw_ref(layer, m, exclude, k)
    fresh = np.random.default_rng(zlib.crc32(("r32:" + layer).encode())).standard_normal((m, k))
    assert not np.allclose(W, fresh / np.linalg.norm(fresh, axis=0))
    unsalted = np.random.default_rng(zlib.crc32(layer.encode()))
    left = [c for c in range(m) if c not in exclude]
    assert sorted(int(x) for x in unsalted.choice(left, size=k, replace=False)) != chans


@pytest.mark.parametrize("layer,m,exclude,k", DRAWS)
def test_draw_matches_the_interface_steps(layer, m, exclude, k):
    chans, W = r32_basis.draw(layer, m, list(exclude), k)
    want_chans, want_W = _draw_ref(layer, m, exclude, k)
    assert [int(x) for x in chans] == want_chans
    W = np.asarray(W)
    assert W.shape == (m, k)
    np.testing.assert_allclose(W, want_W, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("layer,m,exclude,k", DRAWS)
def test_draw_channels_are_sorted_distinct_ints_outside_exclude(layer, m, exclude, k):
    chans, _ = r32_basis.draw(layer, m, list(exclude), k)
    chans = list(chans)
    assert len(chans) == k
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in chans), [type(x) for x in chans]
    assert chans == sorted(chans) and len(set(chans)) == k
    assert all(0 <= x < m and x not in exclude for x in chans)


@pytest.mark.parametrize("layer,m,exclude,k", DRAWS)
def test_draw_columns_have_unit_norm(layer, m, exclude, k):
    _, W = r32_basis.draw(layer, m, list(exclude), k)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(W, dtype=np.float64), axis=0), np.ones(k), rtol=0, atol=1e-6)


def test_draw_is_deterministic_and_seeded_by_the_layer():
    c1, W1 = r32_basis.draw("mixed4c", 40, [1, 2], 6)
    c2, W2 = r32_basis.draw("mixed4c", 40, [1, 2], 6)
    assert list(c1) == list(c2)
    np.testing.assert_array_equal(np.asarray(W1), np.asarray(W2))
    c3, W3 = r32_basis.draw("mixed4d", 40, [1, 2], 6)
    assert not np.allclose(np.asarray(W1), np.asarray(W3))
    assert [int(x) for x in c3] == _draw_ref("mixed4d", 40, [1, 2], 6)[0]


# ---------------------------------------------------------------------------
# project_max
# ---------------------------------------------------------------------------

def _project_loop(F, W):
    F = np.asarray(F, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n, m, h, w = F.shape
    k = W.shape[1]
    out = np.empty((n, k))
    for i in range(n):
        for j in range(k):
            s = np.zeros((h, w))
            for c in range(m):
                s += W[c, j] * F[i, c]
            out[i, j] = s.max()
    return out


def _rand_FW(seed, n=5, m=7, h=4, w=3, k=4):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, m, h, w)), rng.standard_normal((m, k))


def test_fixture_project_max_cases_tell_max_of_sum_from_other_orders():
    F, W = _rand_FW(1)
    want = _project_loop(F, W)
    assert not np.allclose(want, F.max(axis=(2, 3)) @ W, atol=1e-3)  # projected channel maxima
    P = np.einsum("nchw,ck->nkhw", F, W)
    assert not np.allclose(want, np.abs(P).max(axis=(2, 3)), atol=1e-3)  # maximum of the absolute value


def test_project_max_against_a_loop():
    F, W = _rand_FW(1)
    got = np.asarray(r32_basis.project_max(F, W))
    assert got.shape == (5, 4)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, _project_loop(F, W), rtol=1e-5, atol=1e-5)


def test_project_max_float32_values_and_a_drawn_W():
    rng = np.random.default_rng(2)
    F = (rng.random((6, 9, 5, 5)) * 4.0).astype(np.float32)
    _, W = _draw_ref("mixed4b", 9, [0, 4], 3)
    got = np.asarray(r32_basis.project_max(F, W))
    assert got.shape == (6, 3)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, _project_loop(F, W), rtol=1e-5, atol=1e-5)


def test_project_max_all_negative_projections():
    rng = np.random.default_rng(3)
    F = -(np.abs(rng.standard_normal((4, 6, 3, 5))) + 0.5)
    W = np.abs(rng.standard_normal((6, 2))) + 0.1
    want = _project_loop(F, W)
    assert np.all(want < 0)
    got = np.asarray(r32_basis.project_max(F, W))
    assert got.dtype == np.float32
    assert np.all(got < 0)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_project_max_one_position():
    rng = np.random.default_rng(4)
    F = rng.standard_normal((3, 5, 1, 1))
    W = rng.standard_normal((5, 1))
    got = np.asarray(r32_basis.project_max(F, W))
    assert got.shape == (3, 1)
    np.testing.assert_allclose(got, F[:, :, 0, 0] @ W, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# joint_d
# ---------------------------------------------------------------------------

JQ = [
    {"question": "Is the photo taken outdoors?", "answers": ["yes", "no"]},
    {"question": "Is the main object red?", "answers": ["yes", "no"]},
    {"question": "How many objects are there?", "answers": ["one", "two", "three"]},
]
JN, JM = 80, 7
J_BLANK = (11, 47, 70)  # no answer to any question
J_ANS2 = (5, 17, 42)  # answer 2 of question 2: 3 photos
J_NONE2 = (3, 9, 28, 33, 50, 61, 77)  # question 2 not answered
JQUERIES = [(0, 0, 1), (0, 1, 0), (1, 0, 1), (1, 1, 0), (2, 0, 1), (2, 1, 0)]


def _jq():
    return copy.deepcopy(JQ)


def _confounded():
    """(R, ans_rows): see the module docstring"""
    rng = np.random.default_rng(3203)
    q1 = rng.permutation(np.repeat([0, 1], JN // 2))
    q2 = rng.integers(0, 2, size=JN)
    q2[list(J_ANS2)] = 2
    taken = set(J_BLANK) | set(J_ANS2) | set(J_NONE2)

    def first(cond, k):
        out = [i for i in range(JN) if cond(i) and i not in taken][:k]
        taken.update(out)
        return out

    none0 = first(lambda i: q1[i] == 0, 5)  # question 0 not answered (all "yes" to question 1)
    none1 = first(lambda i: q1[i] == 1, 5)  # question 1 not answered
    flip = first(lambda i: q1[i] == 0, 4) + first(lambda i: q1[i] == 1, 4)  # x without y, y without x
    q0 = q1.copy()
    q0[flip] = 1 - q0[flip]
    rows = []
    for i in range(JN):
        if i in J_BLANK:
            rows.append([None, None, None])
        else:
            rows.append([None if i in none0 else int(q0[i]),
                         None if i in none1 else int(q1[i]),
                         None if i in J_NONE2 else int(q2[i])])
    x = np.array([r[0] == 0 for r in rows], dtype=np.float64)
    y = np.array([r[1] == 0 for r in rows], dtype=np.float64)
    t = np.array([r[2] == 1 for r in rows], dtype=np.float64)
    off = np.array([r[0] is None for r in rows], dtype=np.float64)
    e = rng.standard_normal((JN, 4))
    R = np.zeros((JN, JM))
    R[:, 0] = y + 0.02 * e[:, 0]  # depends only on question 1
    R[:, 1] = x + 0.02 * e[:, 1]  # depends only on question 0
    R[:, 2] = e[:, 2] + 2.0 * off  # noise, offset where question 0 has no answer
    R[:, 3] = 0.0  # sd 0
    R[:, 4] = 1.5  # sd 0
    R[:, 5] = 0.5 * t + 0.3 * e[:, 3]  # depends on question 2
    R[:, 6] = y  # exactly y's indicator
    return R, rows


def _joint_ref(R, ans_rows, questions, q, a, b, min_n=pairtable.MIN_N, intercept=True):
    """the interface's joint_d written out: design, lstsq, difference of two coefficients over each column's sd"""
    R = np.asarray(R, dtype=np.float64)
    n = len(ans_rows)
    cols, where = ([np.ones(n)] if intercept else []), {}
    for qq, Q in enumerate(questions):
        codes = np.array([-1 if r[qq] is None else int(r[qq]) for r in ans_rows], dtype=np.int64)
        for v in range(len(Q["answers"])):
            on = codes == v
            if np.count_nonzero(on) >= min_n:
                where[(qq, v)] = len(cols)
                cols.append(on.astype(np.float64))
    if (q, a) not in where or (q, b) not in where:
        return None
    X = np.column_stack(cols)
    beta = np.linalg.lstsq(X, R, rcond=None)[0]
    diff = beta[where[(q, a)]] - beta[where[(q, b)]]
    sd = R.std(axis=0, ddof=1)
    return np.where(sd > 0, diff / np.where(sd > 0, sd, 1.0), 0.0)


def _vec(x):
    assert x is not None
    v = np.asarray(x, dtype=np.float64)
    assert v.shape == (JM,), v.shape
    return v


def test_fixture_joint_case():
    R, rows = _confounded()
    plain = contrastloop.contrast(R, [r[0] for r in rows], 0, 1)
    assert plain is not None and plain[2][0] > 1.2  # question 0's plain contrast on column 0 is large
    j0 = _joint_ref(R, rows, JQ, 0, 0, 1)
    j1 = _joint_ref(R, rows, JQ, 1, 0, 1)
    assert abs(j0[0]) < 0.2 and j1[0] > 1.5
    assert j0[1] > 1.5 and abs(j1[1]) < 0.2
    assert abs(j0[6]) < 1e-9
    assert _joint_ref(R, rows, JQ, 2, 2, 0) is None
    assert _joint_ref(R, rows, JQ, 2, 2, 0, min_n=3) is not None
    # min_n and the intercept change the numbers on this case
    assert not np.allclose(_joint_ref(R, rows, JQ, 0, 0, 1, min_n=3), j0, rtol=1e-6, atol=1e-9)
    assert not np.allclose(_joint_ref(R, rows, JQ, 1, 0, 1, intercept=False), j1, rtol=1e-6, atol=1e-9)
    # sd over the n photos, not over the photos given a or b
    on = np.array([r[0] in (0, 1) for r in rows])
    assert not np.isclose(R[:, 2].std(ddof=1), R[on, 2].std(ddof=1))


@pytest.mark.parametrize("q,a,b", JQUERIES)
def test_joint_d_matches_the_design_least_squares(q, a, b):
    R, rows = _confounded()
    got = _vec(r32_basis.joint_d(R, rows, _jq(), q, a, b))
    np.testing.assert_allclose(got, _joint_ref(R, rows, JQ, q, a, b), rtol=1e-6, atol=1e-9)


def test_joint_d_confounded_plain_large_joint_near_zero():
    R, rows = _confounded()
    plain = contrastloop.contrast(R, [r[0] for r in rows], 0, 1)[2]
    j0 = _vec(r32_basis.joint_d(R, rows, _jq(), 0, 0, 1))
    j1 = _vec(r32_basis.joint_d(R, rows, _jq(), 1, 0, 1))
    assert plain[0] > 1.2
    assert abs(j0[0]) < 0.2  # x's joint contrast is near 0 on the column driven by y
    assert j1[0] > 1.5  # y's stays
    assert j0[1] > 1.5 and abs(j1[1]) < 0.2  # the other way round on the column driven by x
    assert abs(j0[6]) < 1e-8  # exactly y: x's joint contrast is 0
    assert j1[6] == pytest.approx(1.0 / R[:, 6].std(ddof=1), rel=1e-8)


@pytest.mark.parametrize("q,a,b", [(2, 2, 0), (2, 0, 2), (2, 1, 2), (2, 2, 1)])
def test_joint_d_none_when_an_answer_has_no_column(q, a, b):
    R, rows = _confounded()
    assert r32_basis.joint_d(R, rows, _jq(), q, a, b) is None


def test_joint_d_none_when_the_question_is_never_answered():
    R, rows = _confounded()
    rows = [[r[0], r[1], None] for r in rows]
    assert r32_basis.joint_d(R, rows, _jq(), 2, 0, 1) is None
    got = _vec(r32_basis.joint_d(R, rows, _jq(), 0, 0, 1))
    np.testing.assert_allclose(got, _joint_ref(R, rows, JQ, 0, 0, 1), rtol=1e-6, atol=1e-9)


def test_joint_d_min_n():
    R, rows = _confounded()
    got = _vec(r32_basis.joint_d(R, rows, _jq(), 2, 2, 0, min_n=3))  # exactly 3 photos: a column
    np.testing.assert_allclose(got, _joint_ref(R, rows, JQ, 2, 2, 0, min_n=3), rtol=1e-6, atol=1e-9)
    assert r32_basis.joint_d(R, rows, _jq(), 2, 2, 0, min_n=4) is None
    got = _vec(r32_basis.joint_d(R, rows, _jq(), 0, 0, 1, min_n=3))
    np.testing.assert_allclose(got, _joint_ref(R, rows, JQ, 0, 0, 1, min_n=3), rtol=1e-6, atol=1e-9)


def test_joint_d_default_min_n_is_pairtable_min_n():
    R, rows = _confounded()
    got = _vec(r32_basis.joint_d(R, rows, _jq(), 0, 0, 1))
    np.testing.assert_allclose(got, _joint_ref(R, rows, JQ, 0, 0, 1, min_n=pairtable.MIN_N), rtol=1e-6, atol=1e-9)
    got = _vec(r32_basis.joint_d(R, rows, _jq(), 0, 0, 1, min_n=pairtable.MIN_N))
    np.testing.assert_allclose(got, _joint_ref(R, rows, JQ, 0, 0, 1), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("q", [0, 1, 2])
def test_joint_d_swapping_the_answers_negates(q):
    R, rows = _confounded()
    ab = _vec(r32_basis.joint_d(R, rows, _jq(), q, 0, 1))
    ba = _vec(r32_basis.joint_d(R, rows, _jq(), q, 1, 0))
    np.testing.assert_allclose(ab, -ba, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("q,a,b", JQUERIES)
def test_joint_d_zero_where_the_sd_is_zero(q, a, b):
    R, rows = _confounded()
    got = _vec(r32_basis.joint_d(R, rows, _jq(), q, a, b))
    assert np.all(np.isfinite(got))
    assert abs(got[3]) <= 1e-12 and abs(got[4]) <= 1e-12


# ---------------------------------------------------------------------------
# excess_kurtosis
# ---------------------------------------------------------------------------

def _kurt_ref(x):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    if var == 0:
        return None
    return float(((x - mu) ** 4).mean() / var ** 2 - 3.0)


@pytest.mark.parametrize("x,want", [
    ([1.0, 2.0, 3.0, 4.0], -1.36),  # var 1.25 (ddof 0), fourth moment 2.5625
    ([0.0, 0.0, 0.0, 0.0, 10.0], 0.25),  # var 16, fourth moment 832
    ([-1.0, 1.0], -2.0),
    ([2.0, -2.0, 0.0, 0.0], -1.0),  # var 2, fourth moment 8
])
def test_excess_kurtosis_known_vectors(x, want):
    got = r32_basis.excess_kurtosis(np.asarray(x))
    assert got is not None
    assert float(got) == pytest.approx(want, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("x", [[3.0] * 6, [0.0] * 4, [7.5]])
def test_excess_kurtosis_none_when_the_variance_is_zero(x):
    assert r32_basis.excess_kurtosis(np.asarray(x)) is None


def test_excess_kurtosis_random_vectors():
    rng = np.random.default_rng(5)
    x = rng.standard_t(5, size=500) * 3.0 + 10.0
    got = r32_basis.excess_kurtosis(x)
    assert float(got) == pytest.approx(_kurt_ref(x), rel=1e-9, abs=1e-12)
    x32 = x.astype(np.float32)
    got = r32_basis.excess_kurtosis(x32)
    assert float(got) == pytest.approx(_kurt_ref(x32), rel=1e-4, abs=1e-6)


# ---------------------------------------------------------------------------
# candidate: synthetic sub-pool
# ---------------------------------------------------------------------------

N_CLS, PER = 30, 8
M = 21  # channels: 20 below a column puts it at percentile 95.24 (22 columns with a direction: 95.45)
C = 3  # the selected unit
C2 = 7  # a "channel" candidate: its own class profile, follows question 0
PEERS = (5, 9, 14)  # follow C's class profile
TOP_PROF = 70.0  # C's class profile is 10, 20, ..., 300: its 24 classes are those at 70 or above
LABS = [100 + 7 * k for k in range(N_CLS)]
N_EXTRA = 80  # photos of x_all outside the sub-pool
CQ = [
    {"question": "Is the photo taken outdoors?", "answers": ["yes", "no"]},
    {"question": "What colour is the main object?", "answers": ["red", "green", "blue"]},
    {"question": "Is there any text?", "answers": ["yes", "no"]},
    {"question": "How many objects are there?", "answers": ["one", "two", "many"]},
]
KINDS = ("selected", "channel", "direction")
NAME_OF = {"selected": "mixed4c:3", "channel": "mixed4c:7", "direction": "mixed4c:dir05"}
REC_KEYS = ("group", "unit", "layer", "n_D", "n_T", "classes", "peers", "kurt_all", "kurt_pool", "rows", "n_rows",
            "n_D95", "n_held", "n_distinctive", "n_held_null")
HEADER = ("group", "unit", "layer", "n_D", "n_T", "classes", "peers")
COUNTS = ("n_rows", "n_D95", "n_held", "n_distinctive", "n_held_null")
ROW_KEYS = ("row", "q", "first", "second", "D_d", "D_pct", "T_d", "T_pct", "T_outcome", "T_distinctive",
            "T_null_outcome")
JOINT_KEYS = ("jD_d", "jD_pct", "jT_d", "jT_pct", "jT_outcome")


def _pool():
    rng = np.random.default_rng(3202)
    prof = 10.0 * (rng.permutation(N_CLS) + 1)
    prof2 = 10.0 * (rng.permutation(N_CLS) + 1)
    cls = rng.permutation(np.repeat(np.arange(N_CLS), PER))
    n = len(cls)
    names = [f"r32pool/{i:05d}.jpeg" for i in range(n)]
    labels = np.asarray([LABS[k] for k in cls], dtype=np.int64)
    parity = [zlib.crc32(s.encode()) % 2 for s in names]
    top = [i for i in range(n) if prof[cls[i]] >= TOP_PROF]
    blank = set(top[:3]) | set(top[-2:])

    q1 = rng.integers(0, 3, size=n)
    q0 = np.where(q1 == 0, 0, 1)
    q0 = np.where(rng.random(n) < 0.2, 1 - q0, q0)  # "outdoors" mostly comes with "red"
    q2 = rng.integers(0, 3, size=n)  # 2 -> None
    q3 = np.arange(n) % 2
    many_D = [i for i in top if i not in blank and parity[i] == 0][:6]
    many_T = [i for i in top if i not in blank and parity[i] == 1][:2]
    q3[many_D + many_T] = 2
    row_of = {i: ([None] * 4 if i in blank else
                  [int(q0[i]), int(q1[i]), None if int(q2[i]) == 2 else int(q2[i]), int(q3[i])])
              for i in range(n)}

    def on(q, v):
        return np.array([row_of[i][q] == v for i in range(n)], dtype=np.float64)

    V = rng.random((n, M)) * 300.0
    V[:, C] = prof[cls] + 3.0 * on(1, 0) + 0.3 * rng.standard_normal(n)
    for p in PEERS:
        V[:, p] = prof[cls] + 2.0 * rng.standard_normal(n)
    V[:, C2] = prof2[cls] + 3.0 * on(0, 0) + 1.5 * rng.standard_normal(n)
    xdir = 0.5 * prof2[cls] + 3.0 * on(2, 0) + 0.3 * rng.standard_normal(n)
    extra = 150.0 + 200.0 * rng.standard_normal(N_EXTRA) ** 3
    pool = {"V": V.astype(np.float32), "labels": labels, "names": names, "row_of": row_of,
            "xdir": xdir.astype(np.float32), "extra": extra.astype(np.float32)}
    info = {"prof": prof, "top": top, "blank": sorted(blank), "many_D": many_D, "many_T": many_T}
    return pool, info


def _inputs(kind, name=None):
    pool, _ = _pool()
    V = pool["V"]
    if kind == "direction":
        Vc, c = np.column_stack([V, pool["xdir"]]).astype(np.float32), M
    else:
        Vc, c = V, (C if kind == "selected" else C2)
    x_all = np.concatenate([Vc[:, c], pool["extra"]]).astype(np.float32)
    return {"name": name or NAME_OF[kind], "group": kind, "Vc": Vc, "c": c, "labels": pool["labels"],
            "names": pool["names"], "row_of": pool["row_of"], "questions": copy.deepcopy(CQ), "x_all": x_all}


def _run(kind, joint=None, name=None):
    inp = _inputs(kind, name)
    args = (inp["name"], inp["group"], inp["Vc"], inp["c"], inp["labels"], inp["names"], inp["row_of"],
            inp["questions"], inp["x_all"])
    if joint is None:
        return r32_basis.candidate(*args)
    return r32_basis.candidate(*args, joint=joint)


def _reference(inp, joint):
    """the candidate's record by hand: pairtable.run_unit's steps without readings, every real row picked"""
    name, c = inp["name"], inp["c"]
    Vc, labels, names = np.asarray(inp["Vc"]), np.asarray(inp["labels"]), inp["names"]
    row_of, questions = inp["row_of"], inp["questions"]
    classes, _, D, T = pairtable.unit_photos({"chan": c}, Vc, labels, names)
    peers = [p for p, _ in contrastloop.peers(Vc, labels, c)]
    seed = zlib.crc32(name.encode())
    D_ids, T_ids = np.asarray(D, dtype=np.int64), np.asarray(T, dtype=np.int64)
    rows_D = [row_of[i] for i in D]
    rows_T = [row_of[i] for i in T]
    null_D = proploop.permute_within(rows_D, labels[D_ids], classes, np.random.default_rng(seed + pairtable.NULL_D))
    null_T = proploop.permute_within(rows_T, labels[T_ids], classes, np.random.default_rng(seed + pairtable.NULL_T))
    R_D = proploop.centre(Vc[D_ids], labels[D_ids])
    R_T = proploop.centre(Vc[T_ids], labels[T_ids])
    table = pairtable.table_rows(R_D, rows_D, questions, c, peers)
    picks = [pairtable.pick(r, questions, R_T, rows_T, null_T, c, peers) for r in table]
    rows = []
    for r, p in zip(table, picks):
        rec = {"row": r["row"], "q": r["q"], "first": r["first"], "second": r["second"], "D_d": r["d"],
               "D_pct": r["pct"], "T_d": p["T"]["d"], "T_pct": p["T"]["pct"], "T_outcome": p["T"]["outcome"],
               "T_distinctive": p["T"]["distinctive"], "T_null_outcome": p["T_null"]["outcome"]}
        if joint:
            for side, R, rr in (("D", R_D, rows_D), ("T", R_T, rows_T)):
                jd = _joint_ref(R, rr, questions, r["q"], r["first"], r["second"])
                rec[f"j{side}_d"] = None if jd is None else float(jd[c])
                rec[f"j{side}_pct"] = None if jd is None else proploop.percentile(jd, c)
            rec["jT_outcome"] = ("too few" if rec["jT_d"] is None
                                 else contrastloop.outcome(rec["jT_d"], rec["jT_pct"]))
        rows.append(rec)
    want = {"group": inp["group"], "unit": name, "layer": name.split(":")[0], "n_D": len(D), "n_T": len(T),
            "classes": classes, "peers": peers, "kurt_all": _kurt_ref(inp["x_all"]),
            "kurt_pool": _kurt_ref(Vc[:, c]), "rows": rows, "n_rows": len(rows),
            "n_D95": sum(r["D_pct"] >= 95.0 for r in rows),
            "n_held": sum(r["T_outcome"] == "held" for r in rows),
            "n_distinctive": sum(bool(r["T_distinctive"]) for r in rows),
            "n_held_null": sum(r["T_null_outcome"] == "held" for r in rows)}
    aux = {"D": D, "T": T, "null_D": null_D, "null_T": null_T}
    return want, aux


# ---------------------------------------------------------------------------
# candidate: fixture guards (reference functions only)
# ---------------------------------------------------------------------------

def test_fixture_candidate_pool():
    _, info = _pool()
    want, aux = _reference(_inputs("selected"), True)
    assert len(want["classes"]) == pairtable.K_CLASSES == 24
    assert sorted(want["classes"]) == sorted(LABS[k] for k in range(N_CLS) if info["prof"][k] >= TOP_PROF)
    assert set(want["peers"]) == set(PEERS)
    assert sorted(aux["D"] + aux["T"]) == info["top"]
    assert set(info["many_D"]) <= set(aux["D"]) and set(info["many_T"]) <= set(aux["T"])
    rows = want["rows"]
    assert len(rows) == 8  # Q0 1 pair, Q1 3, Q2 1, Q3 3
    outs = [r["T_outcome"] for r in rows]
    assert "held" in outs and "too few" in outs
    assert any(r["jT_d"] is None for r in rows) and all(r["jD_d"] is not None for r in rows)
    assert max(abs(r["D_d"] - r["jD_d"]) for r in rows) > 0.3  # joint and plain differ on this pool
    assert abs(want["kurt_all"] - want["kurt_pool"]) > 0.5
    for kind in ("channel", "direction"):
        w, _ = _reference(_inputs(kind), False)
        assert w["rows"]


# ---------------------------------------------------------------------------
# candidate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("joint", (False, True))
@pytest.mark.parametrize("kind", KINDS)
def test_candidate_returns_every_listed_key(kind, joint):
    rec = _run(kind, joint)
    assert isinstance(rec, dict)
    missing = [k for k in REC_KEYS if k not in rec]
    assert not missing, missing
    rows = rec["rows"]
    assert isinstance(rows, list) and rows
    keys = ROW_KEYS + (JOINT_KEYS if joint else ())
    for j, r in enumerate(rows):
        miss = [k for k in keys if k not in r]
        assert not miss, (j, miss)


@pytest.mark.parametrize("kind", KINDS)
def test_candidate_without_joint_rows_have_no_joint_fields(kind):
    for rec in (_run(kind), _run(kind, False)):
        for j, r in enumerate(rec["rows"]):
            extra = [k for k in JOINT_KEYS if k in r]
            assert not extra, (j, extra)


@pytest.mark.parametrize("joint", (False, True))
@pytest.mark.parametrize("kind", KINDS)
def test_candidate_header_fields(kind, joint):
    want, _ = _reference(_inputs(kind), joint)
    rec = _plain(_run(kind, joint))
    for k in HEADER:
        _same(rec[k], want[k], k)


@pytest.mark.parametrize("joint", (False, True))
@pytest.mark.parametrize("kind", KINDS)
def test_candidate_rows_match_the_hand_steps(kind, joint):
    want, _ = _reference(_inputs(kind), joint)
    rec = _plain(_run(kind, joint))
    _same([_fields(r, ROW_KEYS) for r in rec["rows"]], [_fields(r, ROW_KEYS) for r in want["rows"]], "rows")


@pytest.mark.parametrize("joint", (False, True))
@pytest.mark.parametrize("kind", KINDS)
def test_candidate_counts(kind, joint):
    want, _ = _reference(_inputs(kind), joint)
    rec = _plain(_run(kind, joint))
    for k in COUNTS:
        assert rec[k] == want[k], (k, rec[k], want[k])
    rows = rec["rows"]
    assert rec["n_rows"] == len(rows)
    assert rec["n_D95"] == sum(r["D_pct"] >= 95.0 for r in rows)
    assert rec["n_held"] == sum(r["T_outcome"] == "held" for r in rows)
    assert rec["n_distinctive"] == sum(bool(r["T_distinctive"]) for r in rows)
    assert rec["n_held_null"] == sum(r["T_null_outcome"] == "held" for r in rows)


@pytest.mark.parametrize("kind", KINDS)
def test_candidate_kurtosis(kind):
    inp = _inputs(kind)
    rec = _plain(_run(kind, False))
    assert rec["kurt_all"] == pytest.approx(_kurt_ref(inp["x_all"]), rel=1e-4, abs=1e-6)
    assert rec["kurt_pool"] == pytest.approx(_kurt_ref(inp["Vc"][:, inp["c"]]), rel=1e-4, abs=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_candidate_joint_fields(kind):
    want, _ = _reference(_inputs(kind), True)
    rec = _plain(_run(kind, True))
    _same([_fields(r, JOINT_KEYS) for r in rec["rows"]], [_fields(r, JOINT_KEYS) for r in want["rows"]],
          "rows (joint)", rel=1e-6, abs_=1e-9)


def test_candidate_joint_too_few_on_T():
    want, _ = _reference(_inputs("selected"), True)
    rec = _plain(_run("selected", True))
    nones = [j for j, r in enumerate(want["rows"]) if r["jT_d"] is None]
    assert nones
    for j in nones:
        r = rec["rows"][j]
        assert r["jT_outcome"] == "too few", (j, r["jT_outcome"])
        assert r["jT_d"] is None and r["jT_pct"] is None, (j, r["jT_d"], r["jT_pct"])


@pytest.mark.parametrize("kind", KINDS)
def test_candidate_joint_leaves_the_other_fields_unchanged(kind):
    a = _plain(_run(kind, False))
    b = _plain(_run(kind, True))
    for k in REC_KEYS:
        if k != "rows":
            _same(b[k], a[k], k)
    _same([_fields(r, ROW_KEYS) for r in b["rows"]], [_fields(r, ROW_KEYS) for r in a["rows"]], "rows")


def test_candidate_real_fields_do_not_depend_on_the_name():
    names = ("mixed4c:3", "mixed4c:13")
    a, b = (_plain(_run("selected", True, name=nm)) for nm in names)
    assert (a["unit"], b["unit"]) == names
    for k in ("layer", "n_D", "n_T", "classes", "peers", "kurt_all", "kurt_pool", "n_rows", "n_D95", "n_held",
              "n_distinctive"):
        _same(b[k], a[k], k)
    keys = tuple(k for k in ROW_KEYS + JOINT_KEYS if k != "T_null_outcome")
    _same([_fields(r, keys) for r in b["rows"]], [_fields(r, keys) for r in a["rows"]], "rows")
    for nm, rec in zip(names, (a, b)):  # each name's null answers come from its own seed
        want, _ = _reference(_inputs("selected", nm), False)
        assert [r["T_null_outcome"] for r in rec["rows"]] == [r["T_null_outcome"] for r in want["rows"]]
        assert rec["n_held_null"] == want["n_held_null"]


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

SUMMARY_KEYS = ("n", "mean_held", "share_held", "mean_distinctive", "share_distinctive", "mean_held_null",
                "share_held_null", "mean_D95", "median_kurt_all")
S_LAYERS = ("mixed4c", "mixed5a")
S_GROUPS = ("selected", "channel", "direction")
# per (layer, group): one (n_held, n_distinctive, n_held_null, n_D95, kurt_all) per record
S_CELLS = {
    ("mixed4c", "selected"): [(2, 1, 0, 3, 1.5), (0, 0, 1, 1, 4.0)],
    ("mixed4c", "channel"): [(0, 0, 0, 0, 0.5), (1, 0, 0, 2, -0.3), (0, 0, 0, 1, 2.2)],
    ("mixed4c", "direction"): [(0, 0, 0, 0, 0.1), (0, 0, 1, 0, 0.2), (0, 0, 0, 0, -0.4), (1, 1, 0, 1, 0.3)],
    ("mixed5a", "selected"): [(3, 2, 1, 4, 6.0)],
    ("mixed5a", "channel"): [(0, 0, 0, 1, 1.0), (2, 1, 0, 2, 3.0), (1, 1, 0, 1, 0.0), (0, 0, 2, 0, -1.0)],
    ("mixed5a", "direction"): [(0, 0, 0, 0, 0.0), (0, 0, 0, 0, 0.6), (1, 0, 1, 1, 0.9)],
}


def _summary_record(layer, group, j, t):
    n_held, n_dist, n_hn, n_d95, kurt = t
    n_rows = max(n_held, n_hn, n_d95) + 1
    rows = [{"row": r + 1, "q": r % 3, "first": 0, "second": 1, "D_d": 0.8, "D_pct": 97.0 if r < n_d95 else 60.0,
             "T_d": 0.7 if r < n_held else 0.1, "T_pct": 96.0 if r < n_held else 55.0,
             "T_outcome": "held" if r < n_held else "not held", "T_distinctive": r < n_dist,
             "T_null_outcome": "held" if r < n_hn else "not held"} for r in range(n_rows)]
    unit = f"{layer}:dir{j:02d}" if group == "direction" else f"{layer}:{10 + j}"
    return {"group": group, "unit": unit, "layer": layer, "n_D": 90 + j, "n_T": 95 - j, "classes": [100, 107, 114],
            "peers": [1, 2, 3], "kurt_all": kurt, "kurt_pool": kurt - 0.25, "rows": rows, "n_rows": n_rows,
            "n_D95": n_d95, "n_held": n_held, "n_distinctive": n_dist, "n_held_null": n_hn}


def _summary_records():
    recs, j = [], 0
    for (layer, group), ts in S_CELLS.items():
        for t in ts:
            recs.append(_summary_record(layer, group, j, t))
            j += 1
    order = np.random.default_rng(7).permutation(len(recs))
    return [recs[int(i)] for i in order]


def _stats(recs):
    n = len(recs)

    def mean(k):
        return sum(r[k] for r in recs) / n

    def share(k):
        return sum(r[k] >= 1 for r in recs) / n

    return {"n": n, "mean_held": mean("n_held"), "share_held": share("n_held"),
            "mean_distinctive": mean("n_distinctive"), "share_distinctive": share("n_distinctive"),
            "mean_held_null": mean("n_held_null"), "share_held_null": share("n_held_null"),
            "mean_D95": mean("n_D95"), "median_kurt_all": statistics.median(r["kurt_all"] for r in recs)}


def _entries(x, ctx=()):
    """every dict in x with all SUMMARY_KEYS, with the strings that name it: the keys on its path (a tuple key's
    parts) and its own string values"""
    out = []
    if isinstance(x, dict):
        if all(k in x for k in SUMMARY_KEYS):
            out.append((x, ctx + tuple(v for v in x.values() if isinstance(v, str))))
        for k, v in x.items():
            out += _entries(v, ctx + (tuple(map(str, k)) if isinstance(k, tuple) else (str(k),)))
    elif isinstance(x, (list, tuple)):
        for v in x:
            out += _entries(v, ctx)
    return out


def _mentions(ctx, s):
    return any(s in x for x in ctx)


def _num(x):
    if x is None or isinstance(x, (bool, np.bool_, str)):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _agrees(d, want):
    for k in SUMMARY_KEYS:
        g = _num(d[k])
        if g is None or g != pytest.approx(want[k], rel=1e-9, abs=1e-12):
            return False
    return True


def _np_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


def test_fixture_summary_hand_values_and_distinct_entries():
    recs = _summary_records()
    cell = _stats([r for r in recs if (r["layer"], r["group"]) == ("mixed4c", "selected")])
    assert cell == pytest.approx({"n": 2, "mean_held": 1.0, "share_held": 0.5, "mean_distinctive": 0.5,
                                  "share_distinctive": 0.5, "mean_held_null": 0.5, "share_held_null": 0.5,
                                  "mean_D95": 2.0, "median_kurt_all": 2.75})
    group = _stats([r for r in recs if r["group"] == "channel"])
    assert group == pytest.approx({"n": 7, "mean_held": 4 / 7, "share_held": 3 / 7, "mean_distinctive": 2 / 7,
                                   "share_distinctive": 2 / 7, "mean_held_null": 2 / 7, "share_held_null": 1 / 7,
                                   "mean_D95": 1.0, "median_kurt_all": 0.5})
    wants = ([_stats([r for r in recs if (r["layer"], r["group"]) == c]) for c in S_CELLS]
             + [_stats([r for r in recs if r["group"] == g]) for g in S_GROUPS])
    keys = [tuple(round(w[k], 9) for k in SUMMARY_KEYS) for w in wants]
    assert len(set(keys)) == len(keys)


@pytest.mark.parametrize("cell", list(S_CELLS))
def test_summary_per_layer_and_group(cell):
    layer, group = cell
    recs = _summary_records()
    want = _stats([r for r in recs if (r["layer"], r["group"]) == cell])
    ent = _entries(_plain(r32_basis.summary(copy.deepcopy(recs))))
    assert ent, f"no dict with every one of {SUMMARY_KEYS}"
    named = [(d, ctx) for d, ctx in ent if _mentions(ctx, layer) and _mentions(ctx, group)]
    assert named, f"no entry named by {layer} and {group}: {[ctx for _, ctx in ent]}"
    assert any(_agrees(d, want) for d, _ in named), \
        f"{layer} {group}: {[{k: d[k] for k in SUMMARY_KEYS} for d, _ in named]} != {want}"


@pytest.mark.parametrize("group", S_GROUPS)
def test_summary_per_group_over_all_layers(group):
    recs = _summary_records()
    want = _stats([r for r in recs if r["group"] == group])
    ent = _entries(_plain(r32_basis.summary(copy.deepcopy(recs))))
    assert ent, f"no dict with every one of {SUMMARY_KEYS}"
    named = [(d, ctx) for d, ctx in ent if _mentions(ctx, group) and not any(_mentions(ctx, L) for L in S_LAYERS)]
    assert named, f"no entry named by {group} and no layer: {[ctx for _, ctx in ent]}"
    assert any(_agrees(d, want) for d, _ in named), \
        f"{group}: {[{k: d[k] for k in SUMMARY_KEYS} for d, _ in named]} != {want}"


def test_summary_is_json_serialisable():
    s = r32_basis.summary(_summary_records())
    json.dumps({"summary": s}, default=_np_default)


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def test_selftest_prints_ok(capsys):
    r32_basis.selftest()
    assert "selftest ok" in capsys.readouterr().out


def test_main_selftest_flag(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["r32_basis.py", "--selftest"])
    try:
        r32_basis.main()
    except SystemExit as e:
        assert e.code in (None, 0), e.code
    assert "selftest ok" in capsys.readouterr().out
