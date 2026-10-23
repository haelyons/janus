"""Black-box tests for tools/pairtable.py.

Written from the Interface section of authors/spec_pairtable.md only. Every
expected value is worked out by hand from the definitions the spec uses
(contrastloop.contrast, contrastloop.outcome, proploop.percentile,
proploop._whole); none is computed with the module's own helpers.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import pairtable  # noqa: E402


ROW_KEYS = {"row", "q", "first", "second", "n_first", "n_second", "d", "pct",
            "peer_pct", "gap"}
CHECK_KEYS = {"n_first", "n_second", "d", "pct", "peer_pct", "outcome",
              "distinctive"}


def _approx_list(seq, expected):
    assert seq is not None
    assert [float(x) for x in seq] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------

def _tuples(ps):
    return [tuple(p) for p in ps]


def test_pairs_k0_k1_empty():
    assert _tuples(pairtable.pairs(0)) == []
    assert _tuples(pairtable.pairs(1)) == []


def test_pairs_k2():
    assert _tuples(pairtable.pairs(2)) == [(0, 1)]


def test_pairs_k3():
    assert _tuples(pairtable.pairs(3)) == [(0, 1), (0, 2), (1, 2)]


def test_pairs_k4_order_by_a_then_b():
    assert _tuples(pairtable.pairs(4)) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def test_pairs_returns_list():
    assert isinstance(pairtable.pairs(4), list)
    assert len(pairtable.pairs(8)) == 28


# ---------------------------------------------------------------------------
# table_rows: hand case
#
# 13 photos, 5 channels, c = 0, peers = [3, 1, 4] (deliberately not sorted).
# Photo blocks: A = photos 0-4, B = 5-6, C = 7-10, X = photo 11, E = photo 12;
# every photo of a block has the same values:
#   channel:   0    1    2    3    4
#   A          4    0    2    5    1
#   B          1    1    0    1    3
#   C          0    0    1    0   -1
#   X         50  -50   50  -50   50   (never in a compared pair)
#   E          0    5   -3    0    2
#
# Q0 (3 answers): A -> 0, B and C -> 1, X -> 2, E -> None.
#   (0, 1): 5 vs 6 photos -> a row.  (0, 2), (1, 2): answer 2 has 1 photo -> skipped.
# Q1 (2 answers): A and B -> 1, C and E -> 0, X -> None.
#   (0, 1): answer 0 = C+E (5 photos) vs answer 1 = A+B (7 photos).
#
# Within one pair all channels share the group sizes, so d orders the channels as
# r = (mean_hi - mean_lo) / sqrt(within-group sum of squares) does.
# Q0 (0,1): r = ch0 3.18, ch1 -0.29, ch2 1.15, ch3 4.04, ch4 0.14
#   -> pct ch1 0, ch4 20, ch2 40, ch0 60, ch3 80; d[0] > 0, no swap.
#   d[0]: values 4 x5, 1 x2, 0 x4 -> mean 2, SS 38, sd sqrt(38/10); d = (4 - 1/3) / sd.
# Q1 (0,1): r = ch0 -0.88, ch1 0.15, ch2 -0.29, ch3 -0.81, ch4 -0.55; d[0] < 0, so the
#   row is (1, 0) with counts (7, 5) and d negated: ch1 -0.15 < ch2 0.29 < ch4 0.55
#   < ch3 0.81 < ch0 0.88 -> pct ch1 0, ch2 20, ch4 40, ch3 60, ch0 80.
#   d[0]: hi all 0, lo 4 x5 and 1 x2 (mean 22/7); all 12 values: SS 125/3,
#   sd sqrt(125/33); d = -(22/7) / sd, negated to +(22/7) / sd.
# ---------------------------------------------------------------------------

_A = [4.0, 0.0, 2.0, 5.0, 1.0]
_B = [1.0, 1.0, 0.0, 1.0, 3.0]
_C = [0.0, 0.0, 1.0, 0.0, -1.0]
_X = [50.0, -50.0, 50.0, -50.0, 50.0]
_E = [0.0, 5.0, -3.0, 0.0, 2.0]

HAND_QUESTIONS = [
    {"question": "What colour is the main object?", "answers": ["red", "blue", "green"]},
    {"question": "How big is the main object?", "answers": ["small", "large"]},
]
HAND_C = 0
HAND_PEERS = [3, 1, 4]


def _hand_R():
    return np.array([_A] * 5 + [_B] * 2 + [_C] * 4 + [_X] + [_E], dtype=float)


def _hand_ans():
    return ([[0, 1] for _ in range(5)] + [[1, 1] for _ in range(2)]
            + [[1, 0] for _ in range(4)] + [[2, None]] + [[None, 0]])


def _hand_rows(R=None, **kw):
    R = _hand_R() if R is None else R
    return pairtable.table_rows(R, _hand_ans(), HAND_QUESTIONS, HAND_C, list(HAND_PEERS), **kw)


def _check_row1(r):
    assert r["row"] == 1
    assert r["q"] == 0
    assert (r["first"], r["second"]) == (0, 1)
    assert (r["n_first"], r["n_second"]) == (5, 6)
    assert float(r["d"]) == pytest.approx((4 - 1 / 3) / np.sqrt(38 / 10))
    assert float(r["pct"]) == pytest.approx(60.0)
    _approx_list(r["peer_pct"], [80.0, 0.0, 20.0])
    assert float(r["gap"]) == pytest.approx(60.0 - (80.0 + 0.0 + 20.0) / 3)


def _check_row2(r, row=2):
    assert r["row"] == row
    assert r["q"] == 1
    assert (r["first"], r["second"]) == (1, 0)
    assert (r["n_first"], r["n_second"]) == (7, 5)
    assert float(r["d"]) == pytest.approx((22 / 7) / np.sqrt(125 / 33))
    assert float(r["pct"]) == pytest.approx(80.0)
    _approx_list(r["peer_pct"], [60.0, 0.0, 40.0])
    assert float(r["gap"]) == pytest.approx(80.0 - (60.0 + 0.0 + 40.0) / 3)


def test_table_rows_hand_case_numbering_and_keys():
    rows = _hand_rows()
    assert isinstance(rows, list)
    assert len(rows) == 2
    assert [r["row"] for r in rows] == [1, 2]
    for r in rows:
        assert set(r) == ROW_KEYS


def test_table_rows_positive_d_kept_in_pair_order():
    _check_row1(_hand_rows()[0])


def test_table_rows_negative_d_swapped_with_counts_and_negated_d():
    _check_row2(_hand_rows()[1])


def test_table_rows_skips_pairs_with_too_few_photos():
    rows = _hand_rows()
    assert [(r["q"], sorted((r["first"], r["second"]))) for r in rows] == [(0, [0, 1]), (1, [0, 1])]
    assert all(2 not in (r["first"], r["second"]) for r in rows)


def test_table_rows_none_answers_are_neither_answer():
    # Counts exclude the None photos (photo 12 in Q0, photo 11 in Q1): 5/6 and 7/5,
    # checked in _check_row1/_check_row2. Photo 11 is None in Q1 and only in skipped
    # pairs of Q0, so changing its values changes nothing.
    R = _hand_R()
    R[11] = [-7.0, 9.0, -11.0, 13.0, -15.0]
    rows = _hand_rows(R)
    assert len(rows) == 2
    _check_row1(rows[0])
    _check_row2(rows[1])


def test_table_rows_min_n():
    assert _hand_rows(min_n=6) == []
    rows = _hand_rows(min_n=1)
    assert [r["row"] for r in rows] == [1, 2, 3, 4]
    assert [(r["q"], min(r["first"], r["second"]), max(r["first"], r["second"])) for r in rows] == [
        (0, 0, 1), (0, 0, 2), (0, 1, 2), (1, 0, 1)]
    _check_row2(rows[3], row=4)


def test_table_rows_all_none_answers_gives_no_rows():
    ans = [[None, None] for _ in range(13)]
    assert pairtable.table_rows(_hand_R(), ans, HAND_QUESTIONS, HAND_C, list(HAND_PEERS)) == []


# ---------------------------------------------------------------------------
# table_rows: order by question, then pair
# ---------------------------------------------------------------------------

ORDER_QUESTIONS = [
    {"question": "Q0?", "answers": ["a", "b", "c"]},
    {"question": "Q1?", "answers": ["a", "b"]},
    {"question": "Q2?", "answers": ["only"]},
    {"question": "Q3?", "answers": ["a", "b"]},
    {"question": "Q4?", "answers": ["a", "b"]},
]
# answer -> number of photos given it, per question
ORDER_COUNTS = [{0: 5, 1: 5, 2: 5}, {0: 8, 1: 7}, {0: 15}, {0: 12, 1: 3}, {0: 8, 1: 7}]


def _order_ans():
    out = []
    for i in range(15):
        out.append([i // 5,                 # Q0: 0-4 -> 0, 5-9 -> 1, 10-14 -> 2
                    0 if i < 8 else 1,      # Q1: 8 vs 7
                    0,                      # Q2: one answer, no pairs
                    0 if i < 12 else 1,     # Q3: 12 vs 3 -> skipped
                    1 if i < 7 else 0])     # Q4: answer 1 on 7, answer 0 on 8
    return out


def test_table_rows_order_question_then_pair():
    R = np.random.default_rng(3).normal(size=(15, 4))
    rows = pairtable.table_rows(R, _order_ans(), ORDER_QUESTIONS, 1, [0, 2, 3])
    assert [r["row"] for r in rows] == [1, 2, 3, 4, 5]
    assert [(r["q"], min(r["first"], r["second"]), max(r["first"], r["second"])) for r in rows] == [
        (0, 0, 1), (0, 0, 2), (0, 1, 2), (1, 0, 1), (4, 0, 1)]
    for r in rows:
        assert r["n_first"] == ORDER_COUNTS[r["q"]][r["first"]]
        assert r["n_second"] == ORDER_COUNTS[r["q"]][r["second"]]
        assert float(r["d"]) >= 0
        assert len(r["peer_pct"]) == 3
        assert float(r["gap"]) == pytest.approx(
            float(r["pct"]) - sum(float(p) for p in r["peer_pct"]) / 3)


# ---------------------------------------------------------------------------
# table_text
# ---------------------------------------------------------------------------

HEAD_LINES = [
    "Below is what is known about one unit (a single neuron) of an image classification network. Someone who never saw",
    "the unit's activity answered the questions below about each of {n} photos of the {K} photo classes on which the",
    "unit is most active on average ({classes}). For each question, every two of its answers that were each given to at",
    "least 5 photos are compared. d is how much more active the unit is on the photos given the first answer than on",
    "the photos given the second, after removing the differences between photo classes, in standard deviations; each",
    "pair is written with the answer on which the unit is more active first. The percentile places d among all {m}",
    "units of the unit's layer on the same photos and answers (95 or above is unusual for this layer). The same",
    "percentile is given for the 3 units of the layer whose average activity over photo classes is most like this",
    "unit's (similar units 1 to 3): a row on which a similar unit is also at 95 or above describes something they",
    "share, not what is particular to this unit.",
]
TASK_LINES = [
    "The rows will be checked on new photos of the same classes, none of them among those above, answered in the same",
    "way: a row holds when this unit's d, with the same answer first, is again at the 95th percentile of its layer or",
    "above. Say in a few sentences what this unit responds to, and choose up to 4 rows that you expect to hold for this",
    "unit on the new photos and that describe what is particular to this unit rather than what it shares with the",
    'similar units. Reply with only JSON: {"explanation": "...", "rows": [n, ...]}',
]

TEXT_N, TEXT_K, TEXT_M = 37, 24, 512
TEXT_CLASSES = ["tabby cat", "goldfish", "pickup truck"]
TEXT_QUESTIONS = [
    {"question": "Is the main object red?", "answers": ["yes", "no", "partly"]},
    {"question": "How many animals are there?", "answers": ["none", "one", "several"]},
]
TEXT_ROWS = [
    {"row": 1, "q": 0, "first": 0, "second": 1, "n_first": 5, "n_second": 6, "d": 0.456,
     "pct": 40.0, "peer_pct": [0.0, 95.3125, 60.0], "gap": -11.77083},
    {"row": 2, "q": 0, "first": 2, "second": 1, "n_first": 9, "n_second": 10, "d": 2.0,
     "pct": 99.8046875, "peer_pct": [95.0, 94.7, 3.2], "gap": 35.50469},
    {"row": 3, "q": 1, "first": 2, "second": 0, "n_first": 7, "n_second": 12, "d": 1.234,
     "pct": 97.65625, "peer_pct": [12.6, 50.0, 99.4], "gap": 43.65625},
]
TEXT_LINES = [
    'Row 1. Is the main object red? "yes" (5 photos) more than "no" (6 photos): '
    'd +0.46, percentile 40; similar units 0, 95, 60.',
    'Row 2. Is the main object red? "partly" (9 photos) more than "no" (10 photos): '
    'd +2.00, percentile 100; similar units 95, 95, 3.',
    'Row 3. How many animals are there? "several" (7 photos) more than "none" (12 photos): '
    'd +1.23, percentile 98; similar units 13, 50, 99.',
]


def _head():
    return " ".join(HEAD_LINES).format(n=TEXT_N, K=TEXT_K, classes="tabby cat, goldfish, pickup truck",
                                       m=TEXT_M)


def _task():
    return " ".join(TASK_LINES)


def _text(rows=None):
    rows = TEXT_ROWS if rows is None else rows
    return pairtable.table_text(TEXT_N, TEXT_K, list(TEXT_CLASSES), TEXT_M, TEXT_QUESTIONS, rows)


def test_table_text_row_lines_exact_and_in_order():
    lines = _text().split("\n")
    idx = []
    for expected in TEXT_LINES:
        assert expected in lines
        idx.append(lines.index(expected))
    assert idx == sorted(idx)
    assert idx == list(range(idx[0], idx[0] + 3))


def test_table_text_whole_numbers_round_half_to_even():
    row = {"row": 1, "q": 1, "first": 1, "second": 0, "n_first": 8, "n_second": 5, "d": 0.5,
           "pct": 12.5, "peer_pct": [0.5, 87.5, 2.5], "gap": 0.0}
    expected = ('Row 1. How many animals are there? "one" (8 photos) more than "none" (5 photos): '
                'd +0.50, percentile 12; similar units 0, 88, 2.')
    assert expected in _text([row]).split("\n")


def test_table_text_head_paragraph_appears():
    text = _text()
    assert _head() in text
    assert _head() in text.split("\n")


def test_table_text_task_paragraph_appears():
    text = _text()
    assert _task() in text
    assert _task() in text.split("\n")


def test_table_text_blank_lines_between_parts():
    text = _text()
    assert _head() + "\n\n" + TEXT_LINES[0] in text
    assert "\n".join(TEXT_LINES) in text
    assert TEXT_LINES[-1] + "\n\n" + _task() in text
    assert text.strip() == _head() + "\n\n" + "\n".join(TEXT_LINES) + "\n\n" + _task()


def test_table_text_one_row():
    lines = _text([TEXT_ROWS[2]]).split("\n")
    assert TEXT_LINES[2] in lines
    assert not any(line.startswith("Row 1.") or line.startswith("Row 2.") for line in lines)


# ---------------------------------------------------------------------------
# parse_rows
# ---------------------------------------------------------------------------

def test_parse_rows_valid_case_strips_explanation():
    text = 'Here is my answer.\n{"explanation": "  Red objects, mostly cars.  ", "rows": [3, 1]}'
    assert pairtable.parse_rows(text, 3) == {"explanation": "Red objects, mostly cars.", "rows": [3, 1]}


def test_parse_rows_valid_inside_code_fence():
    text = '```json\n{"explanation": "Water.", "rows": [2]}\n```'
    assert pairtable.parse_rows(text, 5) == {"explanation": "Water.", "rows": [2]}


def test_parse_rows_drops_bools():
    text = '{"explanation": "x", "rows": [true, 2, false, 1]}'
    assert pairtable.parse_rows(text, 5) == {"explanation": "x", "rows": [2, 1]}


def test_parse_rows_only_bools_is_none():
    assert pairtable.parse_rows('{"explanation": "x", "rows": [true, false]}', 5) is None


def test_parse_rows_drops_duplicates_keeping_first():
    text = '{"explanation": "x", "rows": [2, 2, 1, 2, 1, 3]}'
    assert pairtable.parse_rows(text, 5) == {"explanation": "x", "rows": [2, 1, 3]}


def test_parse_rows_drops_out_of_range():
    text = '{"explanation": "x", "rows": [0, 4, -1, 3, 100, 1]}'
    assert pairtable.parse_rows(text, 3) == {"explanation": "x", "rows": [3, 1]}


def test_parse_rows_drops_non_ints():
    text = '{"explanation": "x", "rows": [1.0, "2", 2.5, null, 3]}'
    assert pairtable.parse_rows(text, 5) == {"explanation": "x", "rows": [3]}


def test_parse_rows_cap_at_4():
    text = '{"explanation": "x", "rows": [5, 4, 3, 2, 1]}'
    assert pairtable.parse_rows(text, 5) == {"explanation": "x", "rows": [5, 4, 3, 2]}


def test_parse_rows_cap_applies_to_kept_rows():
    text = '{"explanation": "x", "rows": [1, 1, 0, 2, true, 9, 3, 2.0, 4, 5]}'
    assert pairtable.parse_rows(text, 5) == {"explanation": "x", "rows": [1, 2, 3, 4]}


@pytest.mark.parametrize("text", [
    '{"rows": [1]}',
    '{"explanation": "", "rows": [1]}',
    '{"explanation": "   \\n  ", "rows": [1]}',
    '{"explanation": null, "rows": [1]}',
    '{"explanation": 3, "rows": [1]}',
    '{"explanation": ["a"], "rows": [1]}',
])
def test_parse_rows_missing_or_empty_explanation_is_none(text):
    assert pairtable.parse_rows(text, 5) is None


@pytest.mark.parametrize("text", [
    '{"explanation": "x", "rows": []}',
    '{"explanation": "x", "rows": [0, 6, true, 1.0, "1"]}',
])
def test_parse_rows_no_kept_row_is_none(text):
    assert pairtable.parse_rows(text, 5) is None


@pytest.mark.parametrize("text", [
    '{"explanation": "x", "rows": 2}',
    '{"explanation": "x", "rows": "1, 2"}',
    '{"explanation": "x", "rows": null}',
])
def test_parse_rows_rows_not_a_list_is_none(text):
    assert pairtable.parse_rows(text, 5) is None


@pytest.mark.parametrize("text", [
    "no json here at all",
    '{"explanation": "x", "picks": [1]}',
    "",
])
def test_parse_rows_no_object_with_rows_is_none(text):
    assert pairtable.parse_rows(text, 5) is None


def test_parse_rows_last_object_with_rows_wins():
    text = ('{"explanation": "first", "rows": [1]}\nOn reflection:\n'
            '{"explanation": "second", "rows": [2, 3]}')
    assert pairtable.parse_rows(text, 5) == {"explanation": "second", "rows": [2, 3]}


def test_parse_rows_later_object_without_rows_ignored():
    text = '{"explanation": "kept", "rows": [2]}\n{"note": "nothing here"}'
    assert pairtable.parse_rows(text, 5) == {"explanation": "kept", "rows": [2]}


def test_parse_rows_last_object_with_rows_invalid_gives_none():
    text = '{"explanation": "good", "rows": [1]}\n{"explanation": "bad", "rows": [0]}'
    assert pairtable.parse_rows(text, 5) is None
    text = '{"explanation": "good", "rows": [1]}\n{"rows": [2]}'
    assert pairtable.parse_rows(text, 5) is None


# ---------------------------------------------------------------------------
# rule_rows
# ---------------------------------------------------------------------------

def _rec(row, pct, gap):
    return {"row": row, "q": 0, "first": 0, "second": 1, "n_first": 5, "n_second": 5,
            "d": 1.0, "pct": pct, "peer_pct": [0.0, 0.0, 0.0], "gap": gap}


def test_rule_rows_95_cut():
    rows = [_rec(1, 94.9, 50.0), _rec(2, 95.0, 1.0), _rec(3, 99.8, 10.0),
            _rec(4, 50.0, 90.0), _rec(5, 96.0, 5.0)]
    assert list(pairtable.rule_rows(rows)) == [3, 5, 2]


def test_rule_rows_gap_order_highest_first():
    rows = [_rec(1, 97.0, -5.0), _rec(2, 99.0, 30.0), _rec(3, 95.5, 12.0)]
    assert list(pairtable.rule_rows(rows)) == [2, 3, 1]


def test_rule_rows_ties_by_lower_row_number():
    rows = [_rec(6, 96.0, 20.0), _rec(2, 97.0, 20.0), _rec(4, 95.0, 30.0), _rec(9, 99.0, 20.0)]
    assert list(pairtable.rule_rows(rows)) == [4, 2, 6, 9]
    rows_in_order = sorted(rows, key=lambda r: r["row"])
    assert list(pairtable.rule_rows(rows_in_order)) == [4, 2, 6, 9]


def test_rule_rows_cap():
    rows = [_rec(1, 96.0, 3.0), _rec(2, 97.0, 8.0), _rec(3, 95.0, 1.0),
            _rec(4, 99.0, 6.0), _rec(5, 98.0, 9.0), _rec(6, 96.5, 2.0)]
    assert list(pairtable.rule_rows(rows)) == [5, 2, 4, 1]
    assert list(pairtable.rule_rows(rows, k=2)) == [5, 2]
    assert list(pairtable.rule_rows(rows, k=10)) == [5, 2, 4, 1, 6, 3]


def test_rule_rows_none_eligible():
    assert list(pairtable.rule_rows([_rec(1, 94.0, 10.0), _rec(2, 0.0, 50.0)])) == []
    assert list(pairtable.rule_rows([])) == []


# ---------------------------------------------------------------------------
# t_check
#
# 10 photos: answer 0 on photos 0-4, answer 1 on photos 5-9. Channel j has the
# values noise + delta_j on photos 0-4 and noise on photos 5-9, noise = -2..2 in
# each group. Then mean difference = delta, total SS = 2.5 delta^2 + 20 (ddof 1
# over 10 values), d_j = 3 delta_j / sqrt(2.5 delta_j^2 + 20), rising in delta_j.
# UP: delta_j = j - 10 for 20 channels, so channel j has j channels strictly
# below it and pct = 5 j.
# DOWN: delta_j = j - 30, every d < 0 and channel 19 still the largest (pct 95).
# ---------------------------------------------------------------------------

T_ANS = [0] * 5 + [1] * 5
UP = [j - 10 for j in range(20)]
DOWN = [j - 30 for j in range(20)]


def _shift_matrix(deltas):
    noise = np.array([-2.0, -1.0, 0.0, 1.0, 2.0] * 2)
    hi = np.array([1.0] * 5 + [0.0] * 5)
    return np.column_stack([noise + dl * hi for dl in deltas])


def test_t_check_held_and_distinctive():
    rec = pairtable.t_check(_shift_matrix(UP), list(T_ANS), 0, 1, 19, [0, 1, 18])
    assert set(rec) == CHECK_KEYS
    assert (rec["n_first"], rec["n_second"]) == (5, 5)
    assert float(rec["d"]) == pytest.approx(27 / np.sqrt(222.5))
    assert float(rec["pct"]) == pytest.approx(95.0)
    _approx_list(rec["peer_pct"], [0.0, 5.0, 90.0])
    assert rec["outcome"] == "held"
    assert rec["distinctive"] == True  # noqa: E712


def test_t_check_held_not_distinctive_when_min_peer_is_50():
    rec = pairtable.t_check(_shift_matrix(UP), list(T_ANS), 0, 1, 19, [17, 10, 18])
    assert float(rec["pct"]) == pytest.approx(95.0)
    _approx_list(rec["peer_pct"], [85.0, 50.0, 90.0])
    assert rec["outcome"] == "held"
    assert rec["distinctive"] == False  # noqa: E712


def test_t_check_held_not_distinctive_when_peers_high():
    rec = pairtable.t_check(_shift_matrix(UP), list(T_ANS), 0, 1, 19, [17, 18, 16])
    _approx_list(rec["peer_pct"], [85.0, 90.0, 80.0])
    assert rec["outcome"] == "held"
    assert rec["distinctive"] == False  # noqa: E712


def test_t_check_not_held_below_95():
    rec = pairtable.t_check(_shift_matrix(UP), list(T_ANS), 0, 1, 18, [0, 1, 2])
    assert (rec["n_first"], rec["n_second"]) == (5, 5)
    assert float(rec["d"]) == pytest.approx(24 / np.sqrt(180.0))
    assert float(rec["pct"]) == pytest.approx(90.0)
    _approx_list(rec["peer_pct"], [0.0, 5.0, 10.0])
    assert rec["outcome"] == "not held"
    assert rec["distinctive"] == False  # noqa: E712


def test_t_check_not_held_when_d_not_positive():
    rec = pairtable.t_check(_shift_matrix(DOWN), list(T_ANS), 0, 1, 19, [0, 1, 2])
    assert float(rec["d"]) == pytest.approx(-33 / np.sqrt(322.5))
    assert float(rec["pct"]) == pytest.approx(95.0)
    _approx_list(rec["peer_pct"], [0.0, 5.0, 10.0])
    assert rec["outcome"] == "not held"
    assert rec["distinctive"] == False  # noqa: E712


def test_t_check_keeps_given_answer_order():
    # first = 1, second = 0: no swap, d is negative and the channel order reverses.
    rec = pairtable.t_check(_shift_matrix(UP), list(T_ANS), 1, 0, 19, [0, 1, 18])
    assert (rec["n_first"], rec["n_second"]) == (5, 5)
    assert float(rec["d"]) == pytest.approx(-27 / np.sqrt(222.5))
    assert float(rec["pct"]) == pytest.approx(0.0)
    _approx_list(rec["peer_pct"], [95.0, 90.0, 5.0])
    assert rec["outcome"] == "not held"
    assert rec["distinctive"] == False  # noqa: E712


def test_t_check_ignores_none_answers():
    R = np.vstack([np.full((1, 20), 1000.0), np.full((1, 20), -1000.0), _shift_matrix(UP)])
    ans = [None, None] + list(T_ANS)
    rec = pairtable.t_check(R, ans, 0, 1, 19, [0, 1, 18])
    assert (rec["n_first"], rec["n_second"]) == (5, 5)
    assert float(rec["d"]) == pytest.approx(27 / np.sqrt(222.5))
    assert float(rec["pct"]) == pytest.approx(95.0)
    _approx_list(rec["peer_pct"], [0.0, 5.0, 90.0])
    assert rec["outcome"] == "held"


def _too_few(n_first, n_second):
    return {"n_first": n_first, "n_second": n_second, "d": None, "pct": None,
            "peer_pct": None, "outcome": "too few", "distinctive": False}


def test_t_check_too_few_first():
    ans = [0] * 4 + [None] + [1] * 5
    rec = pairtable.t_check(_shift_matrix(UP), ans, 0, 1, 19, [0, 1, 18])
    assert rec == _too_few(4, 5)


def test_t_check_too_few_second():
    ans = [0] * 5 + [1] * 4 + [2]
    rec = pairtable.t_check(_shift_matrix(UP), ans, 0, 1, 19, [0, 1, 18])
    assert rec == _too_few(5, 4)


def test_t_check_too_few_answer_never_given():
    rec = pairtable.t_check(_shift_matrix(UP), list(T_ANS), 0, 2, 19, [0, 1, 18])
    assert rec == _too_few(5, 0)
