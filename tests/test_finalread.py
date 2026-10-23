"""Black-box tests of the interface of tools/finalread.py (authors/spec_finalread.md).

Only the interface section is tested: check_line(pick, chk), parse_final(text, n_rows)
and content_text(st, s, arm). Inputs are hand-made and expected values hand-computed.
For content_text, the table head and row lines (parts 1 and 2) are built with
proploop.fill, pairtable.TABLE_HEAD and pairtable.row_line, as the spec says the
content uses them; parts 3 to 5 are written out from the spec's verbatim text
(wrapped paragraphs joined with single spaces).

proploop._whole(x) = int(round(float(x))), so percentiles round half to even.
"""
import copy
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import finalread  # noqa: E402
import pairtable  # noqa: E402
import proploop  # noqa: E402


# ---------------------------------------------------------------------------
# check_line
# ---------------------------------------------------------------------------

def _pick(row=4, first_answer="in water", second_answer="indoors"):
    # The pick carries its own T / T_null; check_line must use the chk it is given.
    return {
        "row": row, "q": 1, "first": 2, "second": 1,
        "question": "Where is the animal?",
        "first_answer": first_answer, "second_answer": second_answer,
        "D": {"n_first": 8, "n_second": 18, "d": 0.27, "pct": 61.0,
              "peer_pct": [50.0, 52.4, 49.9], "outcome": "not held", "distinctive": False},
        "T": {"n_first": 2, "n_second": 3, "d": None, "pct": None,
              "peer_pct": None, "outcome": "too few", "distinctive": False},
        "T_null": {"n_first": 33, "n_second": 44, "d": 3.33, "pct": 12.0,
                   "peer_pct": [1.0, 2.0, 3.0], "outcome": "not held", "distinctive": False},
    }


def test_check_line_held():
    chk = {"n_first": 14, "n_second": 9, "d": 0.8765, "pct": 96.6,
           "peer_pct": [12.4, 50.6, 99.49], "outcome": "held", "distinctive": True}
    assert finalread.check_line(_pick(), chk) == (
        'Row 4: "in water" (14 photos) more than "indoors" (9 photos): '
        'd +0.88, percentile 97; similar units 12, 51, 99: held.')


def test_check_line_not_held():
    chk = {"n_first": 6, "n_second": 21, "d": -1.2, "pct": 3.2,
           "peer_pct": [40.0, 7.8, 64.3], "outcome": "not held", "distinctive": False}
    assert finalread.check_line(_pick(), chk) == (
        'Row 4: "in water" (6 photos) more than "indoors" (21 photos): '
        'd -1.20, percentile 3; similar units 40, 8, 64: did not hold.')


def test_check_line_too_few():
    chk = {"n_first": 4, "n_second": 20, "d": None, "pct": None,
           "peer_pct": None, "outcome": "too few", "distinctive": False}
    assert finalread.check_line(_pick(), chk) == (
        'Row 4: "in water" (4 photos) more than "indoors" (20 photos): '
        'too few to check (each needs at least 5).')


def test_check_line_too_few_other_side_and_other_pick():
    pick = _pick(row=12, first_answer="white", second_answer="brown")
    chk = {"n_first": 12, "n_second": 2, "d": None, "pct": None,
           "peer_pct": None, "outcome": "too few", "distinctive": False}
    assert finalread.check_line(pick, chk) == (
        'Row 12: "white" (12 photos) more than "brown" (2 photos): '
        'too few to check (each needs at least 5).')


def test_check_line_is_one_line():
    chk = {"n_first": 14, "n_second": 9, "d": 0.8765, "pct": 96.6,
           "peer_pct": [12.4, 50.6, 99.49], "outcome": "held", "distinctive": True}
    line = finalread.check_line(_pick(), chk)
    assert isinstance(line, str)
    assert "\n" not in line


def test_check_line_outcome_alone_decides_held():
    # "held" when the check's outcome is "held" (distinctive does not enter).
    chk = {"n_first": 10, "n_second": 11, "d": 0.62, "pct": 95.2,
           "peer_pct": [96.1, 97.3, 98.8], "outcome": "held", "distinctive": False}
    assert finalread.check_line(_pick(), chk) == (
        'Row 4: "in water" (10 photos) more than "indoors" (11 photos): '
        'd +0.62, percentile 95; similar units 96, 97, 99: held.')


def test_check_line_outcome_alone_decides_not_held():
    chk = {"n_first": 10, "n_second": 10, "d": 0.5, "pct": 97.0,
           "peer_pct": [60.0, 70.0, 80.0], "outcome": "not held", "distinctive": True}
    assert finalread.check_line(_pick(), chk) == (
        'Row 4: "in water" (10 photos) more than "indoors" (10 photos): '
        'd +0.50, percentile 97; similar units 60, 70, 80: did not hold.')


@pytest.mark.parametrize("pct, shown, outcome, verdict", [
    (96.6, "97", "held", "held"),
    (95.4, "95", "held", "held"),
    (95.0, "95", "held", "held"),
    (99.7, "100", "held", "held"),
    (97, "97", "held", "held"),
    (94.4, "94", "not held", "did not hold"),
    (0.3, "0", "not held", "did not hold"),
])
def test_check_line_pct_whole(pct, shown, outcome, verdict):
    chk = {"n_first": 14, "n_second": 9, "d": 0.8765, "pct": pct,
           "peer_pct": [12.4, 50.6, 99.49], "outcome": outcome, "distinctive": True}
    assert finalread.check_line(_pick(), chk) == (
        'Row 4: "in water" (14 photos) more than "indoors" (9 photos): '
        'd +0.88, percentile ' + shown + '; similar units 12, 51, 99: ' + verdict + '.')


@pytest.mark.parametrize("peer, shown", [
    ([12.4, 50.6, 99.49], "12, 51, 99"),
    ([0.2, 100.0, 49.51], "0, 100, 50"),
    ([3, 4, 5], "3, 4, 5"),
    ([88.8, 45.0, 66.6], "89, 45, 67"),
])
def test_check_line_peer_pct_whole(peer, shown):
    chk = {"n_first": 14, "n_second": 9, "d": 0.8765, "pct": 96.6,
           "peer_pct": peer, "outcome": "held", "distinctive": True}
    assert finalread.check_line(_pick(), chk) == (
        'Row 4: "in water" (14 photos) more than "indoors" (9 photos): '
        'd +0.88, percentile 97; similar units ' + shown + ': held.')


@pytest.mark.parametrize("pct, pct_shown, peer, peer_shown", [
    # int(round(float(x))): Python rounds halves to the even integer.
    (96.5, "96", [0.5, 1.5, 2.5], "0, 2, 2"),
    (97.5, "98", [99.5, 50.5, 3.5], "100, 50, 4"),
])
def test_check_line_whole_rounds_half_to_even(pct, pct_shown, peer, peer_shown):
    chk = {"n_first": 14, "n_second": 9, "d": 0.8765, "pct": pct,
           "peer_pct": peer, "outcome": "held", "distinctive": True}
    assert finalread.check_line(_pick(), chk) == (
        'Row 4: "in water" (14 photos) more than "indoors" (9 photos): '
        'd +0.88, percentile ' + pct_shown + '; similar units ' + peer_shown + ': held.')


@pytest.mark.parametrize("d, shown", [
    (0.8765, "+0.88"),
    (2.0, "+2.00"),
    (0.05, "+0.05"),
    (12.3456, "+12.35"),
    (-0.4, "-0.40"),
    (-1.236, "-1.24"),
    (-0.071, "-0.07"),
])
def test_check_line_d_sign_and_two_decimals(d, shown):
    chk = {"n_first": 14, "n_second": 9, "d": d, "pct": 40.2,
           "peer_pct": [12.4, 50.6, 99.49], "outcome": "not held", "distinctive": False}
    assert finalread.check_line(_pick(), chk) == (
        'Row 4: "in water" (14 photos) more than "indoors" (9 photos): '
        'd ' + shown + ', percentile 40; similar units 12, 51, 99: did not hold.')


# ---------------------------------------------------------------------------
# parse_final
# ---------------------------------------------------------------------------

def test_parse_final_basic():
    r = finalread.parse_final('{"explanation": "Fires for white fur.", "rows": [3, 1]}', 5)
    assert r == {"explanation": "Fires for white fur.", "rows": [3, 1]}


def test_parse_final_with_surrounding_prose():
    text = ('Here is my reading.\n'
            '{"explanation": "Fires for white fur.", "rows": [3]}\n'
            'That is all.')
    assert finalread.parse_final(text, 5) == {"explanation": "Fires for white fur.", "rows": [3]}


def test_parse_final_empty_list_is_valid():
    r = finalread.parse_final('{"explanation": "Nothing here holds up.", "rows": []}', 5)
    assert r == {"explanation": "Nothing here holds up.", "rows": []}


def test_parse_final_keeps_order():
    r = finalread.parse_final('{"explanation": "Mixed.", "rows": [4, 2, 5]}', 5)
    assert r == {"explanation": "Mixed.", "rows": [4, 2, 5]}


def test_parse_final_no_cap():
    rows = list(range(40, 0, -1))
    text = '{"explanation": "Everything.", "rows": [%s]}' % ", ".join(str(x) for x in rows)
    r = finalread.parse_final(text, 40)
    assert r == {"explanation": "Everything.", "rows": rows}


@pytest.mark.parametrize("rows_json, n_rows, expected", [
    ("[true, 2, false, 1]", 3, [2, 1]),
    ("[true]", 3, []),
    ("[false, true]", 3, []),
    ("[true, 1]", 3, [1]),
    ("[1, true, 3]", 3, [1, 3]),
])
def test_parse_final_drops_bools(rows_json, n_rows, expected):
    text = '{"explanation": "Fires.", "rows": %s}' % rows_json
    assert finalread.parse_final(text, n_rows) == {"explanation": "Fires.", "rows": expected}


def test_parse_final_duplicates_first_occurrence_only():
    r = finalread.parse_final('{"explanation": "Fires.", "rows": [3, 1, 3, 2, 1]}', 5)
    assert r == {"explanation": "Fires.", "rows": [3, 1, 2]}


def test_parse_final_out_of_range_dropped():
    r = finalread.parse_final('{"explanation": "Fires.", "rows": [0, 1, 5, 6, -2, 100]}', 5)
    assert r == {"explanation": "Fires.", "rows": [1, 5]}


def test_parse_final_all_out_of_range_still_valid():
    r = finalread.parse_final('{"explanation": "Fires.", "rows": [0, 6, 7]}', 5)
    assert r == {"explanation": "Fires.", "rows": []}


@pytest.mark.parametrize("item", ['"2"', '"three"', '2.5', '1.5', 'null', '[2]'])
def test_parse_final_non_ints_dropped(item):
    text = '{"explanation": "Fires.", "rows": [1, %s, 3]}' % item
    assert finalread.parse_final(text, 5) == {"explanation": "Fires.", "rows": [1, 3]}


def test_parse_final_mixed():
    text = ('{"explanation": "Mixed.", '
            '"rows": [true, 2, "3", 2, 0, 7, 1, false, 1.5, 5, null]}')
    assert finalread.parse_final(text, 5) == {"explanation": "Mixed.", "rows": [2, 1, 5]}


def test_parse_final_missing_explanation():
    assert finalread.parse_final('{"rows": [1, 2]}', 5) is None


@pytest.mark.parametrize("expl", ['""', '"   "', '"\\n\\t "'])
def test_parse_final_empty_explanation(expl):
    text = '{"explanation": %s, "rows": [1, 2]}' % expl
    assert finalread.parse_final(text, 5) is None


@pytest.mark.parametrize("expl", ['7', 'null', '["Fires."]', 'true'])
def test_parse_final_explanation_not_a_string(expl):
    text = '{"explanation": %s, "rows": [1, 2]}' % expl
    assert finalread.parse_final(text, 5) is None


def test_parse_final_padded_explanation_is_valid():
    r = finalread.parse_final('{"explanation": "  Fires for dogs.  ", "rows": [2]}', 5)
    assert r is not None
    assert r["rows"] == [2]
    assert r["explanation"].strip() == "Fires for dogs."


@pytest.mark.parametrize("rows_json", ['2', '"1, 2"', 'null', '{"1": true}', 'true'])
def test_parse_final_rows_not_a_list(rows_json):
    text = '{"explanation": "Fires.", "rows": %s}' % rows_json
    assert finalread.parse_final(text, 5) is None


def test_parse_final_no_rows_key():
    assert finalread.parse_final('{"explanation": "Fires."}', 5) is None


@pytest.mark.parametrize("text", ["", "No JSON at all.", "{not json", '{"explanation": "Fires.", "rows": [1'])
def test_parse_final_no_object(text):
    assert finalread.parse_final(text, 5) is None


def test_parse_final_last_object_with_rows_wins():
    text = ('Draft: {"explanation": "First try.", "rows": [1]}\n'
            'Final answer: {"explanation": "Second try.", "rows": [2, 3]}')
    assert finalread.parse_final(text, 5) == {"explanation": "Second try.", "rows": [2, 3]}


def test_parse_final_later_object_without_rows_ignored():
    text = ('{"explanation": "Only answer.", "rows": [4]}\n'
            'P.S. {"note": "no rows here"}')
    assert finalread.parse_final(text, 5) == {"explanation": "Only answer.", "rows": [4]}


# ---------------------------------------------------------------------------
# content_text
# ---------------------------------------------------------------------------

QUESTIONS = [
    {"question": "What color is the animal's fur?", "answers": ["brown", "black", "white"]},
    {"question": "Where is the animal?", "answers": ["on grass", "indoors", "in water"]},
]


def _table_rows():
    return [
        {"row": 1, "q": 0, "first": 0, "second": 1, "n_first": 30, "n_second": 25,
         "d": 0.91, "pct": 97.2, "peer_pct": [40.1, 55.7, 12.3], "gap": 41.5},
        {"row": 2, "q": 1, "first": 1, "second": 0, "n_first": 18, "n_second": 22,
         "d": -0.35, "pct": 20.4, "peer_pct": [60.2, 33.3, 71.9], "gap": -51.5},
        {"row": 3, "q": 0, "first": 2, "second": 0, "n_first": 12, "n_second": 30,
         "d": 1.42, "pct": 99.1, "peer_pct": [88.8, 45.0, 66.6], "gap": 10.3},
        {"row": 4, "q": 1, "first": 2, "second": 1, "n_first": 8, "n_second": 18,
         "d": 0.27, "pct": 61.0, "peer_pct": [50.0, 52.4, 49.9], "gap": 8.6},
    ]


def _pick_row3():
    return {
        "row": 3, "q": 0, "first": 2, "second": 0,
        "question": "What color is the animal's fur?",
        "first_answer": "white", "second_answer": "brown",
        "D": {"n_first": 12, "n_second": 30, "d": 1.42, "pct": 99.1,
              "peer_pct": [88.8, 45.0, 66.6], "outcome": "held", "distinctive": True},
        "T": {"n_first": 7, "n_second": 15, "d": 1.106, "pct": 98.6,
              "peer_pct": [70.4, 31.7, 59.49], "outcome": "held", "distinctive": True},
        "T_null": {"n_first": 7, "n_second": 15, "d": -0.218, "pct": 30.2,
                   "peer_pct": [44.4, 61.6, 50.2], "outcome": "not held", "distinctive": False},
    }


def _pick_row1():
    return {
        "row": 1, "q": 0, "first": 0, "second": 1,
        "question": "What color is the animal's fur?",
        "first_answer": "brown", "second_answer": "black",
        "D": {"n_first": 30, "n_second": 25, "d": 0.91, "pct": 97.2,
              "peer_pct": [40.1, 55.7, 12.3], "outcome": "held", "distinctive": True},
        "T": {"n_first": 3, "n_second": 11, "d": None, "pct": None,
              "peer_pct": None, "outcome": "too few", "distinctive": False},
        "T_null": {"n_first": 9, "n_second": 6, "d": 0.534, "pct": 95.8,
                   "peer_pct": [20.1, 15.3, 88.7], "outcome": "held", "distinctive": True},
    }


def _pick_row2():
    return {
        "row": 2, "q": 1, "first": 1, "second": 0,
        "question": "Where is the animal?",
        "first_answer": "indoors", "second_answer": "on grass",
        "D": {"n_first": 18, "n_second": 22, "d": -0.35, "pct": 20.4,
              "peer_pct": [60.2, 33.3, 71.9], "outcome": "not held", "distinctive": False},
        "T": {"n_first": 10, "n_second": 13, "d": -0.6, "pct": 4.4,
              "peer_pct": [35.6, 72.2, 9.9], "outcome": "not held", "distinctive": False},
        "T_null": {"n_first": 4, "n_second": 13, "d": None, "pct": None,
                   "peer_pct": None, "outcome": "too few", "distinctive": False},
    }


def make_st():
    return {
        "m": 40,
        "class_names": ["dog", "cat", "horse"],
        "D": list(range(101, 108)),   # 7 photos in the table
        "T": list(range(201, 210)),   # 9 new photos for the checks
        "questions": copy.deepcopy(QUESTIONS),
        "tables": {"real": _table_rows()},
        "readings": {
            "real_s1": {"valid": True,
                        "explanation": "Fires for white or pale fur, mostly on grass.",
                        "rows": [3, 1]},
            "real_s2": {"valid": True,
                        "explanation": "Weaker for animals indoors than on grass.",
                        "rows": [2]},
            "real_s3": {"valid": False, "explanation": None, "rows": None},
        },
        "picks": {
            "real_s1": [_pick_row3(), _pick_row1()],
            "real_s2": [_pick_row2()],
        },
    }


# Check lines, hand-computed from the picks above.
R3_REAL = ('Row 3: "white" (7 photos) more than "brown" (15 photos): '
           'd +1.11, percentile 99; similar units 70, 32, 59: held.')
R3_NULL = ('Row 3: "white" (7 photos) more than "brown" (15 photos): '
           'd -0.22, percentile 30; similar units 44, 62, 50: did not hold.')
R1_REAL = ('Row 1: "brown" (3 photos) more than "black" (11 photos): '
           'too few to check (each needs at least 5).')
R1_NULL = ('Row 1: "brown" (9 photos) more than "black" (6 photos): '
           'd +0.53, percentile 96; similar units 20, 15, 89: held.')
R2_REAL = ('Row 2: "indoors" (10 photos) more than "on grass" (13 photos): '
           'd -0.60, percentile 4; similar units 36, 72, 10: did not hold.')
R2_NULL = ('Row 2: "indoors" (4 photos) more than "on grass" (13 photos): '
           'too few to check (each needs at least 5).')

PART3_S1 = ('Someone who read this table wrote: "Fires for white or pale fur, mostly on grass." '
            'They picked rows 3, 1 as the ones expected to hold for this unit on new photos '
            'and to describe what is particular to it.')
PART3_S2 = ('Someone who read this table wrote: "Weaker for animals indoors than on grass." '
            'They picked rows 2 as the ones expected to hold for this unit on new photos '
            'and to describe what is particular to it.')


def part4_intro(n):
    return ("The picked rows were then checked on %d new photos of the same classes, "
            "none of them among those above, answered in the same way. A row held when "
            "this unit's d, with the same answer first, is at the 95th percentile of its "
            "layer or above." % n)


PART5 = ('Taking all of this into account, say in a few sentences what this unit responds to, '
         'and list the rows of the table, by number, that you are confident describe this unit. '
         'Reply with only JSON: {"explanation": "...", "rows": [n, ...]}')


def expected_head(st):
    return proploop.fill(pairtable.TABLE_HEAD, n=len(st["D"]), K=24,
                         classes=", ".join(st["class_names"]), m=st["m"])


def expected_rows(st):
    return "\n".join(pairtable.row_line(row, st["questions"]) for row in st["tables"]["real"])


def expected_content(st, part3, n, lines):
    part4 = part4_intro(n) + "\n" + "\n".join(lines)
    return "\n\n".join([expected_head(st), expected_rows(st), part3, part4, PART5])


CASES = {
    (1, "real"): (PART3_S1, [R3_REAL, R1_REAL]),
    (1, "null"): (PART3_S1, [R3_NULL, R1_NULL]),
    (2, "real"): (PART3_S2, [R2_REAL]),
    (2, "null"): (PART3_S2, [R2_NULL]),
}


@pytest.mark.parametrize("s, arm", sorted(CASES))
def test_content_text_full(s, arm):
    st = make_st()
    part3, lines = CASES[(s, arm)]
    assert finalread.content_text(st, s, arm) == expected_content(st, part3, 9, lines)


@pytest.mark.parametrize("arm", ["real", "null"])
def test_content_text_part_order(arm):
    st = make_st()
    text = finalread.content_text(st, 1, arm)
    lines = CASES[(1, arm)][1]
    parts = [expected_head(st), expected_rows(st), PART3_S1,
             part4_intro(9) + "\n" + "\n".join(lines), PART5]
    assert text.startswith(parts[0])
    pos = 0
    for i, p in enumerate(parts):
        at = text.find(p, pos)
        assert at >= 0, "part %d missing or out of order" % (i + 1)
        pos = at + len(p)


@pytest.mark.parametrize("arm", ["real", "null"])
def test_content_text_blank_line_joins(arm):
    st = make_st()
    text = finalread.content_text(st, 1, arm)
    lines = CASES[(1, arm)][1]
    assert text.startswith(expected_head(st) + "\n\n" + expected_rows(st) + "\n\n"
                           + PART3_S1 + "\n\n" + part4_intro(9) + "\n")
    assert text.endswith("\n".join(lines) + "\n\n" + PART5)


def test_content_text_real_arm_uses_T():
    text = finalread.content_text(make_st(), 1, "real")
    assert part4_intro(9) + "\n" + R3_REAL + "\n" + R1_REAL + "\n\n" in text
    assert R3_NULL not in text
    assert R1_NULL not in text


def test_content_text_null_arm_uses_T_null():
    text = finalread.content_text(make_st(), 1, "null")
    assert part4_intro(9) + "\n" + R3_NULL + "\n" + R1_NULL + "\n\n" in text
    assert R3_REAL not in text
    assert R1_REAL not in text


@pytest.mark.parametrize("arm", ["real", "null"])
def test_content_text_s_selects_reading_and_picks(arm):
    text = finalread.content_text(make_st(), 2, arm)
    assert PART3_S2 in text
    assert PART3_S1 not in text
    for line in (R3_REAL, R3_NULL, R1_REAL, R1_NULL):
        assert line not in text


@pytest.mark.parametrize("arm", ["real", "null"])
def test_content_text_n_is_len_T(arm):
    st = make_st()
    st["T"] = [1, 2, 3, 4, 5]
    part3, lines = CASES[(1, arm)]
    text = finalread.content_text(st, 1, arm)
    assert "\n\n" + part4_intro(5) + "\n" in text
    assert text == expected_content(st, part3, 5, lines)


@pytest.mark.parametrize("arm", ["real", "null"])
def test_content_text_head_uses_len_D_and_class_names(arm):
    st = make_st()
    st["D"] = list(range(11))
    st["class_names"] = ["zebra", "lion"]
    st["m"] = 25
    part3, lines = CASES[(1, arm)]
    assert finalread.content_text(st, 1, arm) == expected_content(st, part3, 9, lines)


@pytest.mark.parametrize("arm", ["real", "null"])
def test_content_text_ignores_null_table_readings_and_picks(arm):
    # Both arms use the real table, readings["real_s{s}"] and picks["real_s{s}"].
    st = make_st()
    part3, lines = CASES[(1, arm)]
    expected = expected_content(st, part3, 9, lines)
    st["tables"]["null"] = [
        {"row": 1, "q": 1, "first": 0, "second": 2, "n_first": 5, "n_second": 5,
         "d": 0.11, "pct": 55.5, "peer_pct": [1.0, 2.0, 3.0], "gap": 52.5},
    ]
    st["readings"]["null_s1"] = {"valid": True, "explanation": "Decoy null reading.", "rows": [1]}
    st["picks"]["null_s1"] = [{
        "row": 1, "q": 1, "first": 0, "second": 2,
        "question": "Where is the animal?",
        "first_answer": "on grass", "second_answer": "in water",
        "D": {"n_first": 5, "n_second": 5, "d": 0.11, "pct": 55.5,
              "peer_pct": [1.0, 2.0, 3.0], "outcome": "not held", "distinctive": False},
        "T": {"n_first": 50, "n_second": 60, "d": 0.77, "pct": 99.9,
              "peer_pct": [1.0, 2.0, 3.0], "outcome": "held", "distinctive": True},
        "T_null": {"n_first": 70, "n_second": 80, "d": 0.66, "pct": 98.8,
                   "peer_pct": [4.0, 5.0, 6.0], "outcome": "held", "distinctive": True},
    }]
    text = finalread.content_text(st, 1, arm)
    assert "Decoy null reading." not in text
    assert '"on grass" (50 photos)' not in text
    assert '"on grass" (70 photos)' not in text
    assert text == expected
