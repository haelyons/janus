"""Black-box tests for tools/r31_validate.py.

Written from the "Answer sources", "Split" and "Interface" sections of
authors/spec_r31_validate.md only, without reading the implementation.
Expected values are worked out by hand, with zlib for half(), or with direct
calls to the functions the spec names (pairtable.unit_photos, table_rows,
t_check; proploop.centre; contrastloop.peers) on a small synthetic pool.
No photo pool, GPU, model or network is used; answer files are temporary JSON
files.

Where the spec leaves something open, only what it fixes is tested; each such
place is marked GAP.
"""

import copy
import json
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
import r31_validate as rv  # noqa: E402


def _as_lists(src):
    """a source with its rows as plain lists, for comparison"""
    return {k: list(v) for k, v in src.items()}


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return path


# ---------------------------------------------------------------------------
# load_claude: a pairtable answers.json (batch records with photos, answers,
# failed); every photo of every batch whose failed is false, with its row.
# ---------------------------------------------------------------------------

def _claude_record(key, photos, answers, failed):
    # the shape proploop.cached_calls writes, with proploop._answer_fields
    return {"key": key, "photos": photos, "answers": answers, "failed": failed,
            "model": "test-model", "system_prompt": "", "content_sha256": "0" * 64,
            "texts": ["reply"], "text": "reply", "valid": not failed}


CLAUDE_FILE = {
    "1": _claude_record("1", [10, 11, 12], [[0, 1, None], [1, 0, 2], [None, None, None]], False),
    "2": _claude_record("2", [20, 21], [[None, None, None], [None, None, None]], True),
    "3": _claude_record("3", [30, 4], [[1, 1, 1], [0, None, 2]], False),
    # failed, yet with answers: failed alone decides
    "4": _claude_record("4", [40], [[1, 0, 1]], True),
}
CLAUDE_EXPECTED = {10: [0, 1, None], 11: [1, 0, 2], 12: [None, None, None], 30: [1, 1, 1],
                   4: [0, None, 2]}


def test_load_claude_keeps_every_photo_of_batches_not_failed(tmp_path):
    got = rv.load_claude(_write(tmp_path / "answers.json", CLAUDE_FILE))
    assert _as_lists(got) == CLAUDE_EXPECTED


def test_load_claude_skips_failed_batches(tmp_path):
    got = rv.load_claude(_write(tmp_path / "answers.json", CLAUDE_FILE))
    for i in (20, 21, 40):
        assert i not in got


def test_load_claude_ids_are_ints(tmp_path):
    got = rv.load_claude(_write(tmp_path / "answers.json", CLAUDE_FILE))
    assert all(type(k) is int for k in got)
    assert sorted(got) == [4, 10, 11, 12, 30]


def test_load_claude_all_failed_gives_empty_source(tmp_path):
    only_failed = {"2": CLAUDE_FILE["2"], "4": CLAUDE_FILE["4"]}
    assert dict(rv.load_claude(_write(tmp_path / "answers.json", only_failed))) == {}


# ---------------------------------------------------------------------------
# load_vlm: {id (str): {"answers", "valid", ...}}; every record with valid
# true, id converted to int.
# ---------------------------------------------------------------------------

VLM_FILE = {
    "5": {"answers": [0, None, 1], "valid": True, "text": "ok"},
    "7": {"answers": None, "valid": False, "text": "garbled"},
    "12": {"answers": [1, 2, 0], "valid": True, "text": "ok"},
    "30": {"answers": [0, 0, 0], "valid": False, "text": "bad"},
    "100": {"answers": [None, None, None], "valid": True, "text": "all ?"},
}
VLM_EXPECTED = {5: [0, None, 1], 12: [1, 2, 0], 100: [None, None, None]}


def test_load_vlm_keeps_only_valid_records(tmp_path):
    got = rv.load_vlm(_write(tmp_path / "vlm.json", VLM_FILE))
    assert _as_lists(got) == VLM_EXPECTED
    assert 7 not in got and 30 not in got


def test_load_vlm_ids_are_ints(tmp_path):
    got = rv.load_vlm(_write(tmp_path / "vlm.json", VLM_FILE))
    assert all(type(k) is int for k in got)
    assert sorted(got) == [5, 12, 100]
    assert "5" not in got


def test_load_vlm_none_valid_gives_empty_source(tmp_path):
    bad = {"7": VLM_FILE["7"], "30": VLM_FILE["30"]}
    assert dict(rv.load_vlm(_write(tmp_path / "vlm.json", bad))) == {}


# ---------------------------------------------------------------------------
# merge: one source; for a photo in several, the last source given wins.
# GAP: whether merge may change its inputs is not fixed; every call gets fresh
# copies.
# ---------------------------------------------------------------------------

M1 = {1: [0, 0], 2: [1, 1]}
M2 = {2: [0, 1], 3: [1, 0]}
M3 = {1: [1, None]}


def _m(*srcs):
    return [copy.deepcopy(s) for s in srcs]


def test_merge_last_source_wins():
    assert _as_lists(rv.merge(*_m(M1, M2, M3))) == {1: [1, None], 2: [0, 1], 3: [1, 0]}


def test_merge_order_reversed():
    assert _as_lists(rv.merge(*_m(M3, M2, M1))) == {1: [0, 0], 2: [1, 1], 3: [1, 0]}


def test_merge_two_sources():
    assert _as_lists(rv.merge(*_m(M1, M2))) == {1: [0, 0], 2: [0, 1], 3: [1, 0]}
    assert _as_lists(rv.merge(*_m(M2, M1))) == {1: [0, 0], 2: [1, 1], 3: [1, 0]}


def test_merge_single_source():
    assert _as_lists(rv.merge(*_m(M2))) == M2


def test_merge_loaded_files_last_wins(tmp_path):
    cpath = _write(tmp_path / "answers.json", CLAUDE_FILE)
    vpath = _write(tmp_path / "vlm.json", VLM_FILE)
    cv = _as_lists(rv.merge(rv.load_claude(cpath), rv.load_vlm(vpath)))
    vc = _as_lists(rv.merge(rv.load_vlm(vpath), rv.load_claude(cpath)))
    assert sorted(cv) == sorted(vc) == [4, 5, 10, 11, 12, 30, 100]
    # photo 12 is in both: the later source's row
    assert cv[12] == [1, 2, 0]
    assert vc[12] == [None, None, None]
    # photo 30's vlm record is not valid, so the claude row stays either way
    assert cv[30] == vc[30] == [1, 1, 1]
    for i in (4, 10, 11):
        assert cv[i] == vc[i] == CLAUDE_EXPECTED[i]
    for i in (5, 100):
        assert cv[i] == vc[i] == VLM_EXPECTED[i]


# ---------------------------------------------------------------------------
# half(name) = zlib.crc32(("r31:" + name).encode()) % 2
# ---------------------------------------------------------------------------

HALF_NAMES = (["img/0.jpeg", "img/1.jpeg", "n01440764/ILSVRC2012_val_00000293.JPEG", "", "r31:",
               "café.jpg"]
              + [f"bank/img/{i:05d}.jpg" for i in range(40)])


def _half(name):
    return zlib.crc32(("r31:" + name).encode()) % 2


@pytest.mark.parametrize("name", HALF_NAMES[:6])
def test_half_hand_names(name):
    assert rv.half(name) == _half(name)


def test_half_many_names():
    assert [rv.half(n) for n in HALF_NAMES] == [_half(n) for n in HALF_NAMES]


def test_half_values_0_and_1_both_occur():
    got = [rv.half(n) for n in HALF_NAMES]
    assert set(got) == {0, 1}


def test_half_uses_the_prefix():
    # guard on the names: some have a half different from crc32(name) % 2 without the prefix
    differ = [n for n in HALF_NAMES if _half(n) != zlib.crc32(n.encode()) % 2]
    assert differ
    for n in differ:
        assert rv.half(n) == _half(n)


# ---------------------------------------------------------------------------
# question_agreement(a, b, ids, nq)
#
# Photos 0-3 and 7 are in both sources and in ids; 4 only in b, 5 only in a,
# 99 in neither (all in ids); 6 in both but not in ids.
#   photo   a               b
#   0       [0, 1, None]    [0, 0, 1]
#   1       [1, 1, 0]       [1, None, 0]
#   2       [None, 0, 0]    [1, 0, None]
#   3       [2, None, 1]    [0, 1, 1]
#   7       [None, None, 2] [None, 1, 2]
# q0: answered in both by 0, 1, 3 (n 3), equal on 0, 1 (agree 2); None in a: 2, 7; in b: 7.
# q1: answered in both by 0, 2 (n 2), equal on 2 (agree 1); None in a: 3, 7; in b: 1.
# q2: answered in both by 1, 3, 7 (n 3), all equal (agree 3); None in a: 0; in b: 2.
# ---------------------------------------------------------------------------

AGREE_KEYS = ("n", "agree", "rate", "none_a", "none_b")

QA_A = {0: [0, 1, None], 1: [1, 1, 0], 2: [None, 0, 0], 3: [2, None, 1], 5: [0, 0, 0],
        6: [0, 0, 0], 7: [None, None, 2]}
QA_B = {0: [0, 0, 1], 1: [1, None, 0], 2: [1, 0, None], 3: [0, 1, 1], 4: [0, 0, 0],
        6: [1, 1, 1], 7: [None, 1, 2]}
QA_IDS = [0, 1, 2, 3, 4, 5, 7, 99]


def _qa(a, b, ids, nq):
    return rv.question_agreement(copy.deepcopy(a), copy.deepcopy(b), list(ids), nq)


def _agree_check(got, expected):
    got = list(got)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert (g["n"], g["agree"], g["none_a"], g["none_b"]) == (e["n"], e["agree"], e["none_a"],
                                                                  e["none_b"])
        if e["rate"] is None:
            assert g["rate"] is None
        else:
            assert float(g["rate"]) == pytest.approx(e["rate"])


def _agree(n, agree, rate, none_a, none_b):
    return {"n": n, "agree": agree, "rate": rate, "none_a": none_a, "none_b": none_b}


def test_question_agreement_hand_case():
    _agree_check(_qa(QA_A, QA_B, QA_IDS, 3), [_agree(3, 2, 2 / 3, 2, 1),
                                              _agree(2, 1, 0.5, 2, 1),
                                              _agree(3, 3, 1.0, 1, 1)])


def test_question_agreement_returns_nq_dicts_with_keys():
    got = list(_qa(QA_A, QA_B, QA_IDS, 3))
    assert len(got) == 3
    for g in got:
        assert set(AGREE_KEYS) <= set(g)


def test_question_agreement_photo_in_both_counts_once_in_ids():
    # photo 6 (a [0, 0, 0], b [1, 1, 1]) added to ids: one more answered per question, none equal
    _agree_check(_qa(QA_A, QA_B, QA_IDS + [6], 3), [_agree(4, 2, 0.5, 2, 1),
                                                    _agree(3, 1, 1 / 3, 2, 1),
                                                    _agree(4, 3, 0.75, 1, 1)])


def test_question_agreement_photos_in_one_source_only_ignored():
    # dropping the photos in only one source (4, 5) and in neither (99) from ids changes nothing
    full = _qa(QA_A, QA_B, QA_IDS, 3)
    both = _qa(QA_A, QA_B, [0, 1, 2, 3, 7], 3)
    assert [{k: g[k] for k in AGREE_KEYS} for g in full] == [{k: g[k] for k in AGREE_KEYS} for g in both]


def test_question_agreement_rate_none_when_n_is_0():
    a = {0: [None, 1], 1: [None, 0]}
    b = {0: [1, 1], 1: [None, 1]}
    _agree_check(_qa(a, b, [0, 1], 2), [_agree(0, 0, None, 2, 1),
                                        _agree(2, 1, 0.5, 0, 0)])


def test_question_agreement_no_ids_in_both():
    _agree_check(_qa(QA_A, QA_B, [4, 5, 99], 3), [_agree(0, 0, None, 0, 0)] * 3)


def test_question_agreement_empty_ids():
    _agree_check(_qa(QA_A, QA_B, [], 2), [_agree(0, 0, None, 0, 0)] * 2)


# ---------------------------------------------------------------------------
# match_rows(rows_a, rows_b)
# ---------------------------------------------------------------------------

def _row(no, q, first, second, d, pct, n_first=6, n_second=7):
    return {"row": no, "q": q, "first": first, "second": second, "n_first": n_first,
            "n_second": n_second, "d": d, "pct": pct, "peer_pct": [10.0, 20.0, 30.0],
            "gap": pct - 20.0}


ROWS_A = [
    _row(1, 0, 0, 1, 0.5, 96.0),
    _row(2, 0, 2, 0, 0.3, 70.0),
    _row(3, 1, 1, 0, 1.2, 99.0),
    _row(4, 2, 0, 1, 0.1, 40.0),        # only in a
]
ROWS_B = [
    _row(1, 0, 1, 2, 0.2, 30.0),        # q0 (1, 2): only in b
    _row(2, 1, 0, 1, 0.7, 98.0, 9, 4),  # q1, written the other way round from a's row 3
    _row(3, 1, 2, 0, 0.6, 80.0),        # the answers of a's row 2, another question
    _row(4, 0, 1, 0, 0.9, 97.0, 8, 5),  # q0, the other way round from a's row 1
    _row(5, 0, 2, 0, 0.4, 60.0),        # q0, same order as a's row 2
    _row(6, 2, 0, 2, 0.8, 90.0),        # q2, another pair than a's row 4
    _row(7, 3, 0, 1, 0.5, 50.0),        # q3: only in b
]
# (a's row, b's row, d of b's row in the pair)
MATCH_EXPECTED = [(1, 4, -0.9), (2, 5, 0.4), (3, 2, -0.7)]


def _match(rows_a, rows_b):
    got = list(rv.match_rows(copy.deepcopy(rows_a), copy.deepcopy(rows_b)))
    for p in got:
        assert len(p) == 2
    return [(p[0], p[1]) for p in got]


def test_match_rows_pairs_in_row_a_order_and_drops_unmatched():
    got = _match(ROWS_A, ROWS_B)
    assert [(ra["row"], rb["row"]) for ra, rb in got] == [(a, b) for a, b, _ in MATCH_EXPECTED]


def test_match_rows_d_negated_only_when_flipped():
    got = _match(ROWS_A, ROWS_B)
    assert [float(rb["d"]) for _, rb in got] == pytest.approx([d for _, _, d in MATCH_EXPECTED])


def test_match_rows_row_a_as_given():
    got = _match(ROWS_A, ROWS_B)
    assert [ra for ra, _ in got] == [ROWS_A[a - 1] for a, _, _ in MATCH_EXPECTED]


def test_match_rows_row_b_other_fields_as_given():
    got = _match(ROWS_A, ROWS_B)
    for (_, rb), (_, b, _) in zip(got, MATCH_EXPECTED):
        src = ROWS_B[b - 1]
        assert {k: v for k, v in rb.items() if k != "d"} == {k: v for k, v in src.items() if k != "d"}


def test_match_rows_negates_a_copy_not_the_input():
    a, b = copy.deepcopy(ROWS_A), copy.deepcopy(ROWS_B)
    rv.match_rows(a, b)
    assert a == ROWS_A and b == ROWS_B


def test_match_rows_empty_sides():
    assert _match([], ROWS_B) == []
    assert _match(ROWS_A, []) == []
    assert _match(ROWS_A, [ROWS_B[0], ROWS_B[2], ROWS_B[5], ROWS_B[6]]) == []


def test_match_rows_on_table_rows_with_an_inverted_question():
    # Q0 answered the same in both; Q1 (two answers) inverted in b: the same photo groups with
    # swapped labels, so b's Q1 row is written the other way round with the same d.
    R = np.random.default_rng(7).normal(size=(40, 6))
    q0 = [i % 3 for i in range(40)]                         # 14, 13, 13 photos
    q1 = [1 if (i * 7) % 10 < 5 else 0 for i in range(40)]  # 20, 20 photos
    qs = [{"question": "Q0?", "answers": ["a", "b", "c"]}, {"question": "Q1?", "answers": ["a", "b"]}]
    ta = pairtable.table_rows(R, [[x, y] for x, y in zip(q0, q1)], qs, 2, [0, 4])
    tb = pairtable.table_rows(R, [[x, 1 - y] for x, y in zip(q0, q1)], qs, 2, [0, 4])
    assert [r["q"] for r in ta] == [0, 0, 0, 1] and [r["q"] for r in tb] == [0, 0, 0, 1]
    got = _match(ta, tb)
    assert [ra for ra, _ in got] == ta
    for (ra, rb), rb_src in zip(got[:3], tb[:3]):
        assert rb == rb_src
        assert float(rb["d"]) == pytest.approx(float(ra["d"]))
    ra, rb = got[3]
    assert rb["first"] == ra["second"] and rb["second"] == ra["first"]
    assert float(rb["d"]) == pytest.approx(-float(ra["d"]))
    assert float(rb["pct"]) == pytest.approx(float(ra["pct"]))


# ---------------------------------------------------------------------------
# unit_compare on a small synthetic pool: 600 photos of 60 classes, 30
# channels; channel 3 carries a binary photo attribute (Q0). Source a answers
# the attributes (Q1 None on some photos, Q2 None on others); source b agrees
# on most, differs on some Q0 and Q1 answers, has Q0 None on some photos, and
# inverts every Q2 answer (None on the same photos as a). Some photos are
# absent from a, others from b. No row is all None.
# ---------------------------------------------------------------------------

QS = [{"question": "Is it red?", "answers": ["yes", "no"]},
      {"question": "Which side?", "answers": ["left", "middle", "right"]},
      {"question": "Is there text?", "answers": ["yes", "no"]}]


@pytest.fixture(scope="module")
def pool():
    rng = np.random.default_rng(31)
    n, m = 600, 30
    labels = rng.permutation(np.repeat(np.arange(60), 10)).astype(np.int64)
    names = [f"img/{i}.jpeg" for i in range(n)]
    attr0 = rng.integers(0, 2, n)
    attr1 = rng.integers(0, 3, n)
    attr2 = rng.integers(0, 2, n)
    V = rng.random((n, m))
    V[:, 3] += 1.5 * attr0
    V = V.astype(np.float32)
    a, b = {}, {}
    for i in range(n):
        x0, x1, x2 = int(attr0[i]), int(attr1[i]), int(attr2[i])
        ra = [x0, None if i % 17 == 0 else x1, None if i % 13 == 0 else x2]
        rb = [None if i % 19 == 0 else (1 - x0 if i % 5 == 0 else x0),
              (x1 + 1) % 3 if i % 4 == 0 else x1,
              None if i % 13 == 0 else 1 - x2]
        if i % 11 != 0:
            a[i] = ra
        if i % 7 != 3:
            b[i] = rb
    return {"V": V, "labels": labels, "names": names, "a": a, "b": b}


def _direct(u, V, labels, names, a, b, questions):
    """the spec's unit_compare, by direct calls on the photos answered in both"""
    c = u["chan"]
    _, _, D, T = pairtable.unit_photos(u, V, labels, names)
    peers = [p for p, _ in contrastloop.peers(V, labels, c)]
    Dk = [i for i in D if i in a and i in b]
    Tk = [i for i in T if i in a and i in b]
    Di, Ti = np.asarray(Dk, dtype=np.int64), np.asarray(Tk, dtype=np.int64)
    R_D = proploop.centre(V[Di], labels[Di])
    R_T = proploop.centre(V[Ti], labels[Ti])
    ta = pairtable.table_rows(R_D, [a[i] for i in Dk], questions, c, peers)
    tb = pairtable.table_rows(R_D, [b[i] for i in Dk], questions, c, peers)
    pairs = []
    for ra in ta:
        for rb in tb:
            if rb["q"] == ra["q"] and {rb["first"], rb["second"]} == {ra["first"], ra["second"]}:
                d_b = -rb["d"] if rb["first"] == ra["second"] else rb["d"]
                pairs.append([ra["d"], d_b, ra["pct"], rb["pct"]])
                break
    outcomes = []
    for ra in ta:
        q, f, s = ra["q"], ra["first"], ra["second"]
        oa = pairtable.t_check(R_T, [a[i][q] for i in Tk], f, s, c, peers)["outcome"]
        ob = pairtable.t_check(R_T, [b[i][q] for i in Tk], f, s, c, peers)["outcome"]
        outcomes.append([oa, ob])
    return {"D": D, "T": T, "Dk": Dk, "Tk": Tk, "ta": ta, "tb": tb, "pairs": pairs,
            "t_outcomes": outcomes}


def _run(pool, unit, a=None, b=None):
    u = proploop.parse_unit(unit)
    a = pool["a"] if a is None else a
    b = pool["b"] if b is None else b
    got = rv.unit_compare(copy.deepcopy(u), pool["V"], pool["labels"], pool["names"], copy.deepcopy(a),
                          copy.deepcopy(b), copy.deepcopy(QS))
    want = _direct(u, pool["V"], pool["labels"], pool["names"], a, b, QS)
    return u, got, want


def _check_pairs(got_pairs, want_pairs):
    got_pairs = [list(p) for p in got_pairs]
    assert len(got_pairs) == len(want_pairs)
    for g, w in zip(got_pairs, want_pairs):
        assert len(g) == 4
        assert [float(x) for x in g] == pytest.approx(w)


UNITS = ["mixed4c:3", "mixed4c:7"]


@pytest.mark.parametrize("unit", UNITS)
def test_unit_compare_keys_and_unit(pool, unit):
    u, got, _ = _run(pool, unit)
    assert {"unit", "nD", "nT", "pairs", "t_outcomes"} <= set(got)
    # GAP: the spec names the field "unit" without its form; the unit's name or its record
    assert got["unit"] == u["name"] or got["unit"] == u


@pytest.mark.parametrize("unit", UNITS)
def test_unit_compare_photo_counts_kept_to_answered_in_both(pool, unit):
    _, got, want = _run(pool, unit)
    # guard: photos of D and T are absent from a or b, so the keep step matters
    assert 0 < len(want["Dk"]) < len(want["D"]) and 0 < len(want["Tk"]) < len(want["T"])
    assert got["nD"] == len(want["Dk"])
    assert got["nT"] == len(want["Tk"])


@pytest.mark.parametrize("unit", UNITS)
def test_unit_compare_pairs_match_direct_tables(pool, unit):
    _, got, want = _run(pool, unit)
    assert want["pairs"]  # guard
    _check_pairs(got["pairs"], want["pairs"])


@pytest.mark.parametrize("unit", UNITS)
def test_unit_compare_t_outcomes_match_direct_checks(pool, unit):
    _, got, want = _run(pool, unit)
    assert len(want["t_outcomes"]) == len(want["ta"]) > 0  # one per row of a's table
    assert [list(x) for x in got["t_outcomes"]] == want["t_outcomes"]


@pytest.mark.parametrize("unit", UNITS)
def test_unit_compare_inverted_question_is_a_flipped_pair(pool, unit):
    # Q2 is inverted in b with None on the same photos: b's row compares the same photo groups
    # written the other way round, so the pair has d_b = -d_a and pct_b = pct_a; on T the two
    # sources' d have opposite signs, so the row cannot be held with both
    _, got, want = _run(pool, unit)
    assert want["ta"][-1]["q"] == 2  # guard: Q2's single pair is a's last row
    pairs = [list(p) for p in got["pairs"]]
    d_a, d_b, pct_a, pct_b = (float(x) for x in pairs[-1])
    assert d_a == pytest.approx(float(want["ta"][-1]["d"]))
    assert d_b == pytest.approx(-d_a)
    assert pct_b == pytest.approx(pct_a)
    oa, ob = list(got["t_outcomes"])[-1]
    assert not (oa == "held" and ob == "held")


@pytest.mark.parametrize("unit", UNITS)
def test_unit_compare_same_source_on_both_sides(pool, unit):
    _, got, want = _run(pool, unit, a=pool["a"], b=pool["a"])
    assert got["nD"] == len([i for i in want["D"] if i in pool["a"]])
    assert got["nT"] == len([i for i in want["T"] if i in pool["a"]])
    _check_pairs(got["pairs"], want["pairs"])
    assert len(want["pairs"]) == len(want["ta"])
    for d_a, d_b, pct_a, pct_b in ([float(x) for x in p] for p in got["pairs"]):
        assert d_b == pytest.approx(d_a) and pct_b == pytest.approx(pct_a)
    outs = [list(x) for x in got["t_outcomes"]]
    assert outs == want["t_outcomes"]
    assert all(oa == ob for oa, ob in outs)


# ---------------------------------------------------------------------------
# summarise(units)
#
# GAP: the spec does not name summarise's output fields. The tests flatten the
# result (dicts and lists, recursively) and look for each expected number
# among its values; the inputs are chosen so that the expected numbers differ
# from each other and from the counts of units, pairs and rows. A swap of two
# fields' names is not detected. Shares are taken as fractions (0 to 1).
# ---------------------------------------------------------------------------

def _flat(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _flat(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _flat(v)
    else:
        yield x


def _numbers(result):
    return [float(v) for v in _flat(result)
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))]


def _has(result, value):
    return any(v == pytest.approx(value, rel=1e-9, abs=1e-12) for v in _numbers(result))


def _nones(result):
    return sum(v is None for v in _flat(result))


def _no_nan(result):
    return not any(np.isnan(v) for v in _numbers(result))


def _unit(name, pairs, outcomes, nD=50, nT=48):
    return {"unit": name, "nD": nD, "nT": nT, "pairs": pairs, "t_outcomes": outcomes}


def _summarise(units):
    return rv.summarise(copy.deepcopy(units))


SUM_UNITS = [
    _unit("mixed4c:3",
          [[0.5, 0.4, 96.0, 97.0], [1.2, 1.0, 99.0, 90.0], [0.1, -0.2, 40.0, 10.0]],
          [["held", "held"], ["held", "not held"], ["not held", "not held"], ["too few", "not held"]]),
    _unit("mixed4c:7",
          [[0.8, 0.9, 95.0, 94.0], [0.3, 0.6, 20.0, 30.0]],
          [["not held", "held"], ["too few", "too few"], ["not held", "held"], ["not held", "not held"]],
          nD=40, nT=41),
]
# pooled over both units: 5 pairs, 8 rows
SUM_D_A = [0.5, 1.2, 0.1, 0.8, 0.3]
SUM_D_B = [0.4, 1.0, -0.2, 0.9, 0.6]
SUM_PCT_A = [96.0, 99.0, 40.0, 95.0, 20.0]
SUM_PCT_B = [97.0, 90.0, 10.0, 94.0, 30.0]
SUM_PCT_AGREE = 3 / 5     # pairs 1, 3, 5 on the same side of 95 (95 counts as >= 95)
SUM_EQUAL = 4 / 8         # rows 1, 3, 6, 8
SUM_HELD_A = 2 / 8        # rows 1, 2
SUM_HELD_B = 3 / 8        # rows 1, 5, 7


def _r(x, y):
    return float(np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0, 1])


def test_summarise_expected_values_are_distinct():
    # guard on the test's inputs (no call to the module)
    vals = [_r(SUM_D_A, SUM_D_B), _r(SUM_PCT_A, SUM_PCT_B), SUM_PCT_AGREE, SUM_EQUAL, SUM_HELD_A,
            SUM_HELD_B]
    assert len({round(v, 6) for v in vals}) == 6
    assert not {round(v, 6) for v in vals} & {0.0, 1.0, 2.0, 5.0, 8.0}


def test_summarise_d_correlation_pooled_over_units():
    assert _has(_summarise(SUM_UNITS), _r(SUM_D_A, SUM_D_B))


def test_summarise_pct_correlation_pooled_over_units():
    assert _has(_summarise(SUM_UNITS), _r(SUM_PCT_A, SUM_PCT_B))


def test_summarise_pct_agreement_share():
    assert _has(_summarise(SUM_UNITS), SUM_PCT_AGREE)


def test_summarise_equal_outcome_share():
    assert _has(_summarise(SUM_UNITS), SUM_EQUAL)


def test_summarise_held_shares_a_and_b():
    got = _summarise(SUM_UNITS)
    assert _has(got, SUM_HELD_A)
    assert _has(got, SUM_HELD_B)


def test_summarise_no_nan_when_all_defined():
    assert _no_nan(_summarise(SUM_UNITS))


def test_summarise_three_pairs_across_units_is_enough():
    # 1 pair in one unit, 2 in the other: fewer than 3 per unit, 3 pooled
    units = [_unit("mixed4c:3", [[0.2, 0.1, 30.0, 20.0]], [["not held", "not held"]]),
             _unit("mixed4c:7", [[0.4, 0.5, 60.0, 70.0], [0.9, 0.7, 97.0, 92.0]],
                   [["not held", "held"], ["held", "not held"]])]
    got = _summarise(units)
    assert _has(got, _r([0.2, 0.4, 0.9], [0.1, 0.5, 0.7]))
    assert _has(got, _r([30.0, 60.0, 97.0], [20.0, 70.0, 92.0]))
    assert _no_nan(got)


def test_summarise_fewer_than_3_pairs_both_correlations_none():
    units = [_unit("mixed4c:3", [[0.1, 0.2, 96.0, 99.0], [0.7, 0.3, 50.0, 96.0]],
                   [["held", "held"], ["not held", "too few"], ["too few", "too few"]])]
    got = _summarise(units)
    assert _nones(got) >= 2
    assert _no_nan(got)
    assert _has(got, 0.5)       # pct agreement: pair 1 both >= 95, pair 2 not
    assert _has(got, 2 / 3)     # equal outcomes: rows 1, 3
    assert _has(got, 1 / 3)     # held with a (row 1), and with b (row 1)


def test_summarise_constant_d_a_gives_d_correlation_none():
    pairs = [[0.5, 0.1, 10.0, 20.0], [0.5, 0.4, 50.0, 40.0], [0.5, 0.9, 99.0, 97.0]]
    got = _summarise([_unit("mixed4c:3", pairs, [["held", "held"]])])
    assert _nones(got) >= 1
    assert _no_nan(got)
    assert _has(got, _r([10.0, 50.0, 99.0], [20.0, 40.0, 97.0]))


def test_summarise_constant_d_b_gives_d_correlation_none():
    pairs = [[0.1, -0.3, 10.0, 20.0], [0.4, -0.3, 50.0, 40.0], [0.9, -0.3, 99.0, 60.0]]
    got = _summarise([_unit("mixed4c:3", pairs, [["held", "held"]])])
    assert _nones(got) >= 1
    assert _no_nan(got)
    assert _has(got, _r([10.0, 50.0, 99.0], [20.0, 40.0, 60.0]))


def test_summarise_constant_pct_a_gives_pct_correlation_none():
    pairs = [[0.1, 0.3, 96.0, 20.0], [0.5, 0.2, 96.0, 60.0], [0.9, 0.8, 96.0, 97.0]]
    outcomes = [["held", "not held"], ["held", "held"], ["not held", "not held"]]
    got = _summarise([_unit("mixed4c:3", pairs, outcomes)])
    assert _nones(got) >= 1
    assert _no_nan(got)
    assert _has(got, _r([0.1, 0.5, 0.9], [0.3, 0.2, 0.8]))
    assert _has(got, 1 / 3)     # pct agreement: only pair 3 on the same side
    assert _has(got, 2 / 3)     # equal outcomes rows 2, 3; held with a rows 1, 2


# ---------------------------------------------------------------------------
# selftest: model-free, without the pool
# ---------------------------------------------------------------------------

def test_selftest_prints_ok(capsys):
    rv.selftest()
    assert "selftest ok" in capsys.readouterr().out
