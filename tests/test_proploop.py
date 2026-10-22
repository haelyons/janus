"""Black-box tests of tools/proploop.py: top_classes, parse_questions, questions_block, parse_answers, centre,
question_stats, percentile, permute_within, table_text.

Expected values are derived by hand from authors/spec_proploop.md (Interface, Photo set, Questions, Answers,
Statistics, Table text). Plain Python, no pytest, no GPU, model or Claude. Prints one line per failure, or exactly
`tests ok`.

    python -I tests/test_proploop.py
"""
import json
import math
import os
import sys
import zlib
from collections import Counter

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


def check(cond, msg):
    if not cond:
        fail(msg)


def err(e):
    return f"{type(e).__name__}: {e}"


def is_int(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def first_diff(got, want):
    """The first differing line of two texts, for a one-line failure message."""
    gl, wl = got.split("\n"), want.split("\n")
    for i in range(max(len(gl), len(wl))):
        g = gl[i] if i < len(gl) else "<no line>"
        w = wl[i] if i < len(wl) else "<no line>"
        if g != w:
            return f"line {i + 1}: got {g!r}, expected {w!r}"
    return "texts differ"


# ---------------------------------------------------------------------------------------------
# top_classes: the k labels with the highest class mean, ties by lower label.
# All values are dyadic so every class mean is exact in floating point.
# ---------------------------------------------------------------------------------------------
TC_A = {
    0: [0.0, 0.5],           # mean 0.25
    1: [0.5, 1.0],           # 0.75
    2: [0.5, 0.5],           # 0.5
    3: [0.75, 0.75],         # 0.75
    4: [0.125, 0.125],       # 0.125
    5: [1.0, 1.0],           # 1.0
    6: [0.25, 0.75],         # 0.5
    7: [0.375, 0.375],       # 0.375
    8: [0.0, 0.125],         # 0.0625
    9: [0.75, 1.0],          # 0.875
    10: [0.0, 1.0, 0.5],     # 0.5 (largest sum of the 0.5 classes)
    11: [0.5, 0.75],         # 0.625
    12: [0.0625, 0.0],       # 0.03125
    13: [0.25, 0.5],         # 0.375
    14: [1.0, 0.5, 0.75],    # 0.75 (largest sum of all classes)
}
# By hand: 1.0: 5 | 0.875: 9 | 0.75: 1, 3, 14 | 0.625: 11 | 0.5: 2, 6, 10 | 0.375: 7, 13 | 0.25: 0
#          | 0.125: 4 | 0.0625: 8 | 0.03125: 12
TC_A_RANK = [5, 9, 1, 3, 14, 11, 2, 6, 10, 7, 13, 0, 4, 8, 12]

# Labels spread over 0..999 (most absent); means 3: 2.0, 17: 4.0, 64: 0.5, 250: 4.0, 999: 1.0.
TC_B = {
    3: [2.0, 2.0],
    999: [1.0, 1.0, 1.0],
    250: [4.0, 4.0],
    17: [3.0, 5.0],
    64: [0.5],
}
TC_B_RANK = [17, 250, 3, 999, 64]

TC_CASES = [
    # (name, groups, k, expected)
    ("A", TC_A, None, [5, 9, 1, 3, 14, 11, 2, 6, 10, 7, 13, 0]),
    ("A", TC_A, 12, [5, 9, 1, 3, 14, 11, 2, 6, 10, 7, 13, 0]),
    ("A", TC_A, 15, TC_A_RANK),
    ("A", TC_A, 10, [5, 9, 1, 3, 14, 11, 2, 6, 10, 7]),   # cut inside the 0.375 tie: 7 in, 13 out
    ("A", TC_A, 4, [5, 9, 1, 3]),                         # cut inside the 0.75 tie: 1, 3 in, 14 out
    ("A", TC_A, 1, [5]),
    ("B", TC_B, 3, [17, 250, 3]),
    ("B", TC_B, 5, TC_B_RANK),
]


def _flatten(groups, seed):
    """Photos of all labels in a fixed shuffled order (class means do not depend on photo order)."""
    items = [(lab, v) for lab, vals in groups.items() for v in vals]
    order = np.random.default_rng(seed).permutation(len(items))
    items = [items[int(i)] for i in order]
    labels = np.array([lab for lab, _ in items], dtype=np.int64)
    vals = np.array([v for _, v in items], dtype=np.float64)
    return vals, labels


def test_top_classes():
    for name, groups, k, want in TC_CASES:
        for seed in (7, 8):
            for dtype in (np.float64, np.float32):
                tag = (f"top_classes {name} k={'default' if k is None else k} order={seed} "
                       f"dtype={np.dtype(dtype).name}")
                vals, labels = _flatten(groups, seed)
                vals = vals.astype(dtype)
                try:
                    if k is None:
                        got = proploop.top_classes(vals, labels)
                    else:
                        got = proploop.top_classes(vals, labels, k=k)
                    got = list(got)
                except Exception as e:
                    fail(f"{tag}: raised {err(e)}")
                    continue
                bad = [x for x in got if not is_int(x)]
                if bad:
                    fail(f"{tag}: non-int labels {bad!r}")
                    continue
                got = [int(x) for x in got]
                check(got == want, f"{tag}: got {got}, expected {want}")


# ---------------------------------------------------------------------------------------------
# parse_questions: last object with "questions"; keep entries that are objects with a non-empty string
# "question" and 2-5 string answers, each non-empty and distinct after strip; stored stripped; first 8;
# None when fewer than 3 are kept or no object is found.
# ---------------------------------------------------------------------------------------------
def Q(question, answers):
    return {"question": question, "answers": answers}


def qreply(entries, pre="Here are the questions.\n\n", post="\n"):
    return pre + json.dumps({"questions": entries}) + post


PQ_MIXED = [
    Q("  Is there a dog in the photo? ", [" yes ", "no", "no animal in the photo "]),  # kept, stored stripped
    "Is there a cat in the photo?",                                                   # not an object
    ["Is there a cat in the photo?", ["yes", "no"]],                                  # not an object
    Q("", ["yes", "no"]),                                                             # empty question
    Q(5, ["yes", "no"]),                                                              # question not a string
    Q(None, ["yes", "no"]),                                                           # question not a string
    {"answers": ["yes", "no"]},                                                       # no question
    Q("How big is it?", ["big"]),                                                     # 1 answer
    Q("How big is it?", []),                                                          # 0 answers
    Q("How big is it?", ["a", "b", "c", "d", "e", "f"]),                              # 6 answers
    Q("What colour is it?", "red, blue"),                                             # answers not a list
    Q("What colour is it?", None),                                                    # answers not a list
    {"question": "What colour is it?"},                                               # no answers
    Q("What colour is it?", ["red", 3]),                                              # an answer not a string
    Q("What colour is it?", ["red", "   "]),                                          # empty after strip
    Q("What colour is it?", ["red", "blue", " red "]),                                # duplicate after strip
    Q("Is it outdoors?", ["yes", "no"]),                                              # kept: 2 answers
    Q("How much of the frame does it fill?", ["all", "most", "half", "little", "none"]),  # kept: 5 answers
    Q("Is the dog wet?", ["Yes", "yes", "no"]),                                       # kept: distinct after strip
]
PQ_MIXED_KEPT = [
    Q("Is there a dog in the photo?", ["yes", "no", "no animal in the photo"]),
    Q("Is it outdoors?", ["yes", "no"]),
    Q("How much of the frame does it fill?", ["all", "most", "half", "little", "none"]),
    Q("Is the dog wet?", ["Yes", "yes", "no"]),
]

PQ_TEN = [Q(f"Is feature {i} visible?", [f"yes {i}", f"no {i}"]) for i in range(1, 11)]
PQ_CAP = PQ_TEN[:2] + [Q("Broken?", ["only one"])] + PQ_TEN[2:]   # 10 valid, an invalid third entry

PQ_FENCED = ("Sure. These cover presence and variation.\n```json\n"
             + json.dumps({"questions": PQ_MIXED}, indent=2) + "\n```\nLet me know if you need more.")

PQ_CASES = [
    # (name, reply text, expected)
    ("mixed entries, fenced with prose", PQ_FENCED, PQ_MIXED_KEPT),
    ("mixed entries, bare JSON", json.dumps({"questions": PQ_MIXED}), PQ_MIXED_KEPT),
    ("10 valid + 1 invalid, cap 8", qreply(PQ_CAP), PQ_TEN[:8]),
    ("exactly 8 valid", qreply(PQ_TEN[:8]), PQ_TEN[:8]),
    ("9 valid", qreply(PQ_TEN[:9]), PQ_TEN[:8]),
    ("8 valid then invalid", qreply(PQ_TEN[:8] + [Q("x", ["a"]), Q("", ["a", "b"])]), PQ_TEN[:8]),
    ("exactly 3 valid", qreply(PQ_TEN[:3]), PQ_TEN[:3]),
    ("3 valid among invalid",
     qreply([Q("", ["a", "b"])] + PQ_TEN[:1] + [Q("Broken?", ["one"])] + PQ_TEN[1:3]), PQ_TEN[:3]),
    ("2 valid", qreply(PQ_TEN[:2]), None),
    ("2 valid + invalid",
     qreply(PQ_TEN[:2] + [Q("Broken?", ["one"]), Q("What colour?", ["red", " red"]), "text"]), None),
    ("empty list", qreply([]), None),
    ("questions not a list", '{"questions": "none"}', None),
    ("no object", "I cannot write questions for this unit.", None),
    ("empty text", "", None),
    ("truncated JSON", '{"questions": [{"question": "Is there a dog?", "answers": ["yes", "no"]}, ', None),
    ("other key only", json.dumps({"items": PQ_TEN[:4]}), None),
    ("last object wins",
     qreply(PQ_TEN[:3]) + "\nOn reflection, these are better:\n" + qreply(PQ_TEN[5:9], pre=""), PQ_TEN[5:9]),
]


def _norm_questions(got):
    out = []
    for d in got:
        if not isinstance(d, dict):
            out.append(d)
            continue
        out.append({k: (list(v) if k == "answers" and isinstance(v, (list, tuple)) else v) for k, v in d.items()})
    return out


def test_parse_questions():
    for name, text, want in PQ_CASES:
        tag = f"parse_questions {name}"
        try:
            got = proploop.parse_questions(text)
        except Exception as e:
            fail(f"{tag}: raised {err(e)}")
            continue
        if want is None:
            check(got is None, f"{tag}: got {got!r}, expected None")
            continue
        if got is None:
            fail(f"{tag}: got None, expected {len(want)} questions")
            continue
        got = _norm_questions(list(got))
        if len(got) != len(want):
            fail(f"{tag}: got {len(got)} questions, expected {len(want)}: {got!r}")
            continue
        for j, (g, w) in enumerate(zip(got, want)):
            if g != w:
                fail(f"{tag}: question {j + 1} is {g!r}, expected {w!r}")
                break


# ---------------------------------------------------------------------------------------------
# questions_block: one line per question, "Q{j}. {question} " then "A) {answer}" ... joined by single spaces.
# ---------------------------------------------------------------------------------------------
QB_CASES = [
    ("three questions",
     [Q("Is there a dog?", ["yes", "no"]),
      Q("What colour is its coat?", ["black", "brown", "white", "mixed", "no dog"]),
      Q("Where is it?", ["indoors", "outdoors", "cannot see"])],
     "Q1. Is there a dog? A) yes B) no\n"
     "Q2. What colour is its coat? A) black B) brown C) white D) mixed E) no dog\n"
     "Q3. Where is it? A) indoors B) outdoors C) cannot see"),
    ("one question",
     [Q("Is the sky visible?", ["yes", "no"])],
     "Q1. Is the sky visible? A) yes B) no"),
    ("eight questions",
     PQ_TEN[:8],
     "Q1. Is feature 1 visible? A) yes 1 B) no 1\n"
     "Q2. Is feature 2 visible? A) yes 2 B) no 2\n"
     "Q3. Is feature 3 visible? A) yes 3 B) no 3\n"
     "Q4. Is feature 4 visible? A) yes 4 B) no 4\n"
     "Q5. Is feature 5 visible? A) yes 5 B) no 5\n"
     "Q6. Is feature 6 visible? A) yes 6 B) no 6\n"
     "Q7. Is feature 7 visible? A) yes 7 B) no 7\n"
     "Q8. Is feature 8 visible? A) yes 8 B) no 8"),
]


def test_questions_block():
    for name, questions, want in QB_CASES:
        tag = f"questions_block {name}"
        try:
            got = proploop.questions_block([dict(q, answers=list(q["answers"])) for q in questions])
        except Exception as e:
            fail(f"{tag}: raised {err(e)}")
            continue
        if not isinstance(got, str):
            fail(f"{tag}: returned {type(got).__name__}, expected str")
            continue
        check(got == want, f"{tag}: {first_diff(got, want)}")


# ---------------------------------------------------------------------------------------------
# parse_answers: last object with "images"; ids exactly 1..b each once; one entry per question, a string that
# stripped and upper-cased is "?" or one of that question's letters; letter index or None for "?".
# ---------------------------------------------------------------------------------------------
PA_QUESTIONS = [
    Q("Is there a dog?", ["yes", "no"]),                                             # letters A-B
    Q("What colour is its coat?", ["black", "brown", "white", "mixed", "no dog"]),   # letters A-E
    Q("Where is it?", ["indoors", "outdoors", "cannot see"]),                       # letters A-C
]


def img(i, answers):
    return {"id": i, "answers": answers}


def areply(images, pre="", post=""):
    return pre + json.dumps({"images": images}) + post


PA_GOOD = [img(2, ["b", " E ", "?"]), img(1, ["A", "c", "B"]), img(3, [" ? ", "a", "C"])]
PA_GOOD_WANT = [[0, 2, 1], [1, 4, None], [None, 0, 2]]

LET = "ABCDE"
PA_TWELVE = [img(i, [LET[i % 2], LET[i % 5], "?" if i % 4 == 0 else LET[i % 3].lower()])
             for i in range(12, 0, -1)]
PA_TWELVE_WANT = [[i % 2, i % 5, None if i % 4 == 0 else i % 3] for i in range(1, 13)]

PA_VALID = [
    # (name, text, b, expected)
    ("ids out of order, lower-case, padded, '?'", areply(PA_GOOD), 3, PA_GOOD_WANT),
    ("fenced with prose", areply(PA_GOOD, pre="Here are my answers:\n```json\n", post="\n```\nDone."), 3,
     PA_GOOD_WANT),
    ("one image", areply([img(1, ["B", "?", "c"])]), 1, [[1, None, 2]]),
    ("all '?'", areply([img(1, ["?", "?", "?"]), img(2, ["?", " ? ", "?"])]), 2,
     [[None, None, None], [None, None, None]]),
    ("twelve images, reversed", areply(PA_TWELVE), 12, PA_TWELVE_WANT),
    ("last object wins",
     areply([img(1, ["B", "B", "B"]), img(2, ["B", "B", "B"]), img(3, ["B", "B", "B"])])
     + "\nCorrection:\n" + areply(PA_GOOD), 3, PA_GOOD_WANT),
]

PA_INVALID = [
    # (name, text, b)
    ("missing id 2", areply([img(1, ["A", "A", "A"]), img(3, ["A", "A", "A"])]), 3),
    ("entry without id", areply([img(1, ["A", "A", "A"]), {"answers": ["A", "A", "A"]},
                                 img(3, ["A", "A", "A"])]), 3),
    ("extra id 4", areply([img(i, ["A", "A", "A"]) for i in (1, 2, 3, 4)]), 3),
    ("id 2 twice, 3 missing", areply([img(i, ["A", "A", "A"]) for i in (1, 2, 2)]), 3),
    ("id 3 twice", areply([img(i, ["A", "A", "A"]) for i in (1, 2, 3, 3)]), 3),
    ("ids from 0", areply([img(i, ["A", "A", "A"]) for i in (0, 1, 2)]), 3),
    ("b larger than reply", areply(PA_GOOD), 4),
    ("b smaller than reply", areply(PA_GOOD), 2),
    ("too few answers", areply([img(1, ["A", "A", "A"]), img(2, ["A", "B"]), img(3, ["A", "A", "A"])]), 3),
    ("too many answers", areply([img(1, ["A", "A", "A"]), img(2, ["A", "B", "C", "A"]),
                                 img(3, ["A", "A", "A"])]), 3),
    ("letter C for a 2-answer question", areply([img(1, ["A", "A", "A"]), img(2, ["C", "A", "A"]),
                                                 img(3, ["A", "A", "A"])]), 3),
    ("letter F for a 5-answer question", areply([img(1, ["A", "A", "A"]), img(2, ["A", "F", "A"]),
                                                 img(3, ["A", "A", "A"])]), 3),
    ("letter D for a 3-answer question", areply([img(1, ["A", "A", "A"]), img(2, ["A", "A", "D"]),
                                                 img(3, ["A", "A", "A"])]), 3),
    ("letter with bracket", areply([img(1, ["A", "A", "A"]), img(2, ["A)", "A", "A"]),
                                    img(3, ["A", "A", "A"])]), 3),
    ("empty entry", areply([img(1, ["A", "A", "A"]), img(2, ["A", "", "A"]), img(3, ["A", "A", "A"])]), 3),
    ("two letters", areply([img(1, ["A", "A", "A"]), img(2, ["A", "A", "AB"]), img(3, ["A", "A", "A"])]), 3),
    ("integer entry", areply([img(1, ["A", "A", "A"]), img(2, [0, "A", "A"]), img(3, ["A", "A", "A"])]), 3),
    ("null entry", areply([img(1, ["A", "A", "A"]), img(2, ["A", None, "A"]), img(3, ["A", "A", "A"])]), 3),
    ("answers a string", areply([img(1, ["A", "A", "A"]), img(2, "AAA"), img(3, ["A", "A", "A"])]), 3),
    ("images not a list", json.dumps({"images": {"1": ["A", "A", "A"], "2": ["A", "A", "A"],
                                                  "3": ["A", "A", "A"]}}), 3),
    ("no images key", json.dumps({"answers": [img(i, ["A", "A", "A"]) for i in (1, 2, 3)]}), 3),
    ("no object", "I cannot see the images.", 3),
    ("empty text", "", 3),
]


def test_parse_answers():
    for name, text, b, want in PA_VALID:
        tag = f"parse_answers {name} b={b}"
        try:
            got = proploop.parse_answers(text, b, [dict(q) for q in PA_QUESTIONS])
        except Exception as e:
            fail(f"{tag}: raised {err(e)}")
            continue
        if got is None:
            fail(f"{tag}: got None, expected {want}")
            continue
        try:
            rows = [list(r) for r in got]
        except Exception as e:
            fail(f"{tag}: result is not a list of lists: {err(e)}")
            continue
        bad = [x for r in rows for x in r if not (x is None or is_int(x))]
        if bad:
            fail(f"{tag}: entries neither int nor None: {bad!r}")
            continue
        rows = [[None if x is None else int(x) for x in r] for r in rows]
        check(rows == want, f"{tag}: got {rows}, expected {want}")
    for name, text, b in PA_INVALID:
        tag = f"parse_answers invalid {name} b={b}"
        try:
            got = proploop.parse_answers(text, b, [dict(q) for q in PA_QUESTIONS])
        except Exception as e:
            fail(f"{tag}: raised {err(e)}")
            continue
        check(got is None, f"{tag}: got {got!r}, expected None")


# ---------------------------------------------------------------------------------------------
# centre: v_j = X[:, j] / max X[:, j] (all 0 when the max is 0), minus the mean of v_j within each label.
# ---------------------------------------------------------------------------------------------
CE_X = np.array([
    # col0 col1 col2 col3
    [2.0, 0.0, 3.0, 0.0],
    [4.0, 0.0, 1.0, 0.0],
    [6.0, 0.0, 2.0, 0.0],
    [8.0, 0.0, 3.0, 0.0],
    [2.0, 0.0, 3.0, 5.0],
    [8.0, 0.0, 1.0, 0.0],
])
CE_LABELS = np.array([5, 3, 5, 3, 3, 7], dtype=np.int64)
# col0: v = [1/4, 1/2, 3/4, 1, 1/4, 1]; label 5 (rows 0, 2) mean 1/2; label 3 (rows 1, 3, 4) mean 7/12;
#       label 7 (row 5) mean 1.
# col1: max 0 -> v all 0 -> R all 0.
# col2: v = [1, 1/3, 2/3, 1, 1, 1/3]; label 5 mean 5/6; label 3 mean 7/9; label 7 mean 1/3.
# col3: v = [0, 0, 0, 0, 1, 0]; label 5 mean 0; label 3 mean 1/3; label 7 mean 0.
CE_WANT = np.array([
    [-1 / 4, 0.0, 1 / 6, 0.0],
    [-1 / 12, 0.0, -4 / 9, -1 / 3],
    [1 / 4, 0.0, -1 / 6, 0.0],
    [5 / 12, 0.0, 2 / 9, -1 / 3],
    [-1 / 3, 0.0, 2 / 9, 2 / 3],
    [0.0, 0.0, 0.0, 0.0],
])


def test_centre():
    tag = "centre hand case"
    try:
        got = proploop.centre(CE_X.copy(), CE_LABELS.copy())
        got = np.asarray(got, dtype=np.float64)
    except Exception as e:
        fail(f"{tag}: raised {err(e)}")
        return
    if got.shape != CE_WANT.shape:
        fail(f"{tag}: shape {got.shape}, expected {CE_WANT.shape}")
        return
    check(not np.isnan(got[:, 1]).any(), f"{tag}: zero column gives NaN: {got[:, 1].tolist()}")
    for j in range(CE_WANT.shape[1]):
        ok = np.allclose(got[:, j], CE_WANT[:, j], rtol=0, atol=1e-9)
        check(ok, f"{tag}: column {j} is {np.round(got[:, j], 6).tolist()}, "
                  f"expected {np.round(CE_WANT[:, j], 6).tolist()}")


# ---------------------------------------------------------------------------------------------
# question_stats
# ---------------------------------------------------------------------------------------------
S = math.sqrt

QS_CASES = [
    # (name, groups [(answer, rows of R)], k, compared, eta2, d)
    #
    # k = 4. Answer 0: 5 photos, answer 1: 6, answer 2: 3 (too few), answer 3: none, None: 2.
    # Kept photos: answers 0 and 1 (11). Non-kept rows hold extreme values that must not count.
    # col0: a0 = 1..5 (mean 3), a1 = 0,0,1,1,2,2 (mean 1); kept mean 21/11.
    #       SSB = 5 (12/11)^2 + 6 (10/11)^2 = 120/11; SST = 65 - 441/11 = 274/11; eta2 = 60/137.
    #       sd = sqrt((274/11) / 10) = sqrt(137/55); d0 = (3 - 1) / sd; d1 = (1 - 3) / sd.
    # col1: 0.5 on every kept photo: denominator 0 -> eta2 0; sd 0 -> d 0.
    # col2: a0 all 2, a1 all 0: SSB = SST = 120/11 -> eta2 1; sd = sqrt(12/11); d0 = 2 / sd; d1 = -2 / sd.
    # col3: all 0 -> eta2 0, d 0.
    ("two compared, small answer, None answers, zero sd", [
        (0, [[1, .5, 2, 0], [2, .5, 2, 0], [3, .5, 2, 0], [4, .5, 2, 0], [5, .5, 2, 0]]),
        (1, [[0, .5, 0, 0], [0, .5, 0, 0], [1, .5, 0, 0], [1, .5, 0, 0], [2, .5, 0, 0], [2, .5, 0, 0]]),
        (2, [[100, 7, 5, 0], [-100, -3, -5, 0], [50, 9, 9, 0]]),
        (None, [[40, 11, 30, 0], [-60, -11, -2, 0]]),
    ], 4, [0, 1],
        [60 / 137, 0.0, 1.0, 0.0],
        {0: [2 / S(137 / 55), 0.0, 2 / S(12 / 11), 0.0],
         1: [-2 / S(137 / 55), 0.0, -2 / S(12 / 11), 0.0]}),
    #
    # k = 3, three compared answers of 5 photos each.
    # col0: a0 all 1, a1 all 0, a2 all -1: SSW 0 -> eta2 1; SST 10, sd = sqrt(10/14) = sqrt(5/7);
    #       d0 = (1 - (-0.5)) / sd; d1 = (0 - 0) / sd; d2 = (-1 - 0.5) / sd.
    # col1: a0 = 2,0,1,1,1 (mean 1), a1 = 3,1,2,2,2 (mean 2), a2 = 0,0,0,0,5 (mean 1); kept mean 4/3.
    #       SSB = 5 (1/9 + 4/9 + 1/9) = 10/3; SSW = 2 + 2 + 20 = 24; SST = 82/3; eta2 = 5/41.
    #       sd = sqrt((82/3) / 14) = sqrt(41/21); d0 = (1 - 1.5) / sd; d1 = (2 - 1) / sd; d2 = (1 - 1.5) / sd.
    ("three compared", [
        (0, [[1, 2], [1, 0], [1, 1], [1, 1], [1, 1]]),
        (1, [[0, 3], [0, 1], [0, 2], [0, 2], [0, 2]]),
        (2, [[-1, 0], [-1, 0], [-1, 0], [-1, 0], [-1, 5]]),
    ], 3, [0, 1, 2],
        [1.0, 5 / 41],
        {0: [1.5 / S(5 / 7), -0.5 / S(41 / 21)],
         1: [0.0, 1.0 / S(41 / 21)],
         2: [-1.5 / S(5 / 7), -0.5 / S(41 / 21)]}),
    #
    # k = 3, counts 5, 4, 5: answer 1 (4 photos) is not compared and its photos are not kept.
    # a0 = 1..5 (mean 3), a2 all 1; kept mean 2; SSB 10, SST 20 -> eta2 0.5; sd = sqrt(20/9);
    # d0 = (3 - 1) / sd = 3 / sqrt(5); d2 = -3 / sqrt(5).
    ("counts 5, 4, 5", [
        (0, [[1], [2], [3], [4], [5]]),
        (1, [[100], [100], [100], [100]]),
        (2, [[1], [1], [1], [1], [1]]),
    ], 3, [0, 2],
        [0.5],
        {0: [3 / S(5)], 2: [-3 / S(5)]}),
    #
    # Fewer than 2 compared answers -> None.
    ("counts 5, 4 and 6 None", [
        (0, [[float(i), 1.0] for i in range(5)]),
        (1, [[float(i), 2.0] for i in range(4)]),
        (None, [[float(i), 3.0] for i in range(6)]),
    ], 2, None, None, None),
    ("all None", [
        (None, [[float(i), 1.0] for i in range(10)]),
    ], 3, None, None, None),
    ("counts 4, 4, 4", [
        (0, [[float(i), 1.0] for i in range(4)]),
        (1, [[float(i), 2.0] for i in range(4)]),
        (2, [[float(i), 3.0] for i in range(4)]),
    ], 3, None, None, None),
    ("one answer of 12 and 3 None", [
        (1, [[float(i), 1.0] for i in range(12)]),
        (None, [[float(i), 2.0] for i in range(3)]),
    ], 3, None, None, None),
]


def _photos(groups, seed):
    """R and answers for the photos of all groups, in a fixed shuffled order (statistics do not depend on it)."""
    items = [(a, row) for a, rows in groups for row in rows]
    order = np.random.default_rng(seed).permutation(len(items))
    items = [items[int(i)] for i in order]
    R = np.array([row for _, row in items], dtype=np.float64)
    ans = [a for a, _ in items]
    return R, ans


def _close(got, want):
    g = np.asarray(got, dtype=np.float64)
    w = np.asarray(want, dtype=np.float64)
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=0, atol=1e-9))


def test_question_stats():
    for name, groups, k, want_comp, want_eta2, want_d in QS_CASES:
        for seed in (3, 4):
            tag = f"question_stats {name} order={seed}"
            R, ans = _photos(groups, seed)
            m = R.shape[1]
            try:
                got = proploop.question_stats(R.copy(), list(ans), k)
            except Exception as e:
                fail(f"{tag}: raised {err(e)}")
                continue
            if want_comp is None:
                check(got is None, f"{tag}: got {got!r}, expected None")
                continue
            if not isinstance(got, dict):
                fail(f"{tag}: got {got!r}, expected a dict")
                continue
            missing = [key for key in ("compared", "eta2", "d") if key not in got]
            if missing:
                fail(f"{tag}: result lacks keys {missing}")
                continue
            comp = sorted(int(a) for a in got["compared"])
            check(comp == want_comp, f"{tag}: compared {comp}, expected {want_comp}")
            eta2 = np.asarray(got["eta2"], dtype=np.float64)
            check(_close(eta2, want_eta2),
                  f"{tag}: eta2 {np.round(eta2, 6).tolist()}, expected {np.round(want_eta2, 6).tolist()} "
                  f"(m={m})")
            d = got["d"]
            keys = sorted(int(a) for a in d)
            check(keys == sorted(want_d), f"{tag}: d has answers {keys}, expected {sorted(want_d)}")
            for a, wd in want_d.items():
                if a not in d:
                    fail(f"{tag}: d lacks answer {a}")
                    continue
                gd = np.asarray(d[a], dtype=np.float64)
                check(_close(gd, wd),
                      f"{tag}: d[{a}] {np.round(gd, 6).tolist()}, expected {np.round(wd, 6).tolist()}")


# ---------------------------------------------------------------------------------------------
# percentile: 100 x share of values strictly below values[j].
# ---------------------------------------------------------------------------------------------
PC_CASES = [
    # (values, j, expected)
    ([0.3, 0.1, 0.3, 0.5, 0.2], 0, 40.0),
    ([0.3, 0.1, 0.3, 0.5, 0.2], 1, 0.0),
    ([0.3, 0.1, 0.3, 0.5, 0.2], 2, 40.0),
    ([0.3, 0.1, 0.3, 0.5, 0.2], 3, 80.0),
    ([0.3, 0.1, 0.3, 0.5, 0.2], 4, 20.0),
    ([0.5, 0.5, 0.2, 0.9, 0.5, 0.0, 0.2], 0, 300 / 7),
    ([0.5, 0.5, 0.2, 0.9, 0.5, 0.0, 0.2], 4, 300 / 7),
    ([0.5, 0.5, 0.2, 0.9, 0.5, 0.0, 0.2], 2, 100 / 7),
    ([0.5, 0.5, 0.2, 0.9, 0.5, 0.0, 0.2], 6, 100 / 7),
    ([0.5, 0.5, 0.2, 0.9, 0.5, 0.0, 0.2], 3, 600 / 7),
    ([0.5, 0.5, 0.2, 0.9, 0.5, 0.0, 0.2], 5, 0.0),
    ([0.0, 0.0, 0.0, 0.0], 2, 0.0),
    ([-0.4, 0.1, -1.2, 0.0], 0, 25.0),
    ([-0.4, 0.1, -1.2, 0.0], 1, 75.0),
    ([-0.4, 0.1, -1.2, 0.0], 2, 0.0),
    ([-0.4, 0.1, -1.2, 0.0], 3, 50.0),
    ([0.7], 0, 0.0),
    ([i / 200 for i in range(200)], 199, 99.5),
    ([i / 200 for i in range(200)], 0, 0.0),
]


def test_percentile():
    for vals, j, want in PC_CASES:
        tag = f"percentile j={j} of {len(vals)} values" + (f" {vals}" if len(vals) <= 8 else "")
        try:
            got = float(proploop.percentile(np.array(vals, dtype=np.float64), j))
        except Exception as e:
            fail(f"{tag}: raised {err(e)}")
            continue
        check(abs(got - want) < 1e-9, f"{tag}: got {got!r}, expected {want!r}")


# ---------------------------------------------------------------------------------------------
# permute_within: for each class in class order, the class's photos (in set order) receive the rows of
# those same photos in the order rng.permutation(count).
# ---------------------------------------------------------------------------------------------
PW_CASES = [
    # (name, set labels, class order)
    ("four classes", [2, 9, 7, 7, 2, 9, 2, 7, 7, 9, 2, 7, 4], [7, 2, 9, 4]),
    ("two classes", [1, 3, 3, 1, 3, 3, 1, 3, 3, 1], [3, 1]),
]
PW_SEEDS = [0, 1, 2024, zlib.crc32("mixed4a:97".encode())]


def _pw_rows(n):
    # Each row is unique (its last entry is the photo position), so every move is traceable.
    return [[i % 3, None if i % 4 == 0 else (5 * i) % 4, i] for i in range(n)]


def _expected_permute(rows, labels, order, seed):
    rng = np.random.default_rng(seed)
    out = list(rows)
    for c in order:
        idx = [i for i, lab in enumerate(labels) if lab == c]
        perm = rng.permutation(len(idx))
        for t, i in enumerate(idx):
            out[i] = rows[idx[int(perm[t])]]
    return out


def _call_permute(rows, labels, order, seed):
    got = proploop.permute_within([list(r) for r in rows], np.array(labels, dtype=np.int64), list(order),
                                  np.random.default_rng(seed))
    return [list(r) for r in got]


def test_permute_within():
    for name, labels, order in PW_CASES:
        n = len(labels)
        rows = _pw_rows(n)
        for seed in PW_SEEDS:
            tag = f"permute_within {name} seed={seed}"
            try:
                a = _call_permute(rows, labels, order, seed)
                b = _call_permute(rows, labels, order, seed)
            except Exception as e:
                fail(f"{tag}: raised {err(e)}")
                continue
            if len(a) != n:
                fail(f"{tag}: returned {len(a)} rows, expected {n}")
                continue
            check(a == b, f"{tag}: two calls with fresh rngs of the same seed differ: {a} vs {b}")
            for c in order:
                idx = [i for i, lab in enumerate(labels) if lab == c]
                own = Counter(tuple(rows[i]) for i in idx)
                now = Counter(tuple(a[i]) for i in idx)
                check(own == now, f"{tag}: class {c} holds rows {sorted(r[-1] for r in now)} "
                                  f"(by origin), expected its own {sorted(r[-1] for r in own)}")
            want = _expected_permute(rows, labels, order, seed)
            check(a == want, f"{tag}: origins {[r[-1] for r in a]}, expected {[r[-1] for r in want]}")


# ---------------------------------------------------------------------------------------------
# table_text: built directly from the spec's verbatim text (wrapped lines of the paragraph joined by spaces).
# ---------------------------------------------------------------------------------------------
def header_lines(n, k, classes, m):
    return [
        "# Checks on ordinary photos",
        "",
        f"Someone who never saw the unit's activity answered the questions below about each of {n} photos: all "
        f"photos of the {k} photo classes on which the unit is most active on average ({classes}). For each "
        "answer: the number of photos given it, and d, how much more (positive) or less (negative) active the "
        "unit is on them than on the other photos, after removing the differences between classes, in standard "
        "deviations. Each question also has a separation score, the share of the unit's within-class variation "
        "that its answers account for. Each number comes with this unit's percentile among all "
        f"{m} units of its layer, computed on the same photos and answers (above 95 or below 5 is unusual for "
        "this layer).",
    ]


# Main case: 12 classes; a non-compared answer, a "could not tell" line, a question with no unclear photos,
# percentiles on exact halves (round half to even) and d rounding/sign cases.
TT_MAIN_CLASSES = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead", "electric ray",
                   "stingray", "cock", "hen", "ostrich", "brambling", "goldfinch"]
TT_MAIN_QUESTIONS = [
    Q("Is there a fish in the photo?", ["yes, one", "yes, several", "no"]),
    Q("How large is the main animal in the frame?", ["fills most of the frame", "about half", "small",
                                                      "no animal"]),
    Q("What is the background mostly?", ["water", "grass or leaves"]),
]
TT_MAIN_QS = [
    {"counts": [40, 3, 70], "unclear": 5, "compared": [0, 2],
     "d": {0: 0.5349, 2: -0.31}, "d_pct": {0: 97.5, 2: 2.5}, "eta2_pct": 42.5},
    {"counts": [30, 50, 33, 5], "unclear": 0, "compared": [0, 1, 2, 3],
     "d": {0: 1.996, 1: 0.0, 2: -1.2, 3: 0.004},
     "d_pct": {0: 100.0, 1: 50.0, 2: 0.0, 3: 66.66666666666667}, "eta2_pct": 99.5},
    {"counts": [100, 18], "unclear": 0, "compared": [0, 1],
     "d": {0: -0.0504, 1: 0.2817}, "d_pct": {0: 33.333333333333336, 1: 12.5}, "eta2_pct": 0.5},
]
TT_MAIN_WANT = "\n".join(header_lines(
    118, 12, "tench, goldfish, great white shark, tiger shark, hammerhead, electric ray, stingray, cock, hen, "
             "ostrich, brambling, goldfinch", 256) + [
    "",
    "Q1. Is there a fish in the photo?",
    "separation: percentile 42",
    "- yes, one: 40 photos, d +0.53, percentile 98",
    "- yes, several: 3 photos (too few to compare)",
    "- no: 70 photos, d -0.31, percentile 2",
    "- could not tell: 5 photos",
    "",
    "Q2. How large is the main animal in the frame?",
    "separation: percentile 100",
    "- fills most of the frame: 30 photos, d +2.00, percentile 100",
    "- about half: 50 photos, d +0.00, percentile 50",
    "- small: 33 photos, d -1.20, percentile 0",
    "- no animal: 5 photos, d +0.00, percentile 67",
    "",
    "Q3. What is the background mostly?",
    "separation: percentile 0",
    "- water: 100 photos, d -0.05, percentile 33",
    "- grass or leaves: 18 photos, d +0.28, percentile 12",
])

# Second case: 3 classes; question 2 has no statistics (fewer than 2 compared answers). Its entry carries the
# counts the answer lines need, with no statistics (compared [0], empty d, eta2_pct None).
TT_NS_CLASSES = ["tench", "goldfish", "great white shark"]
TT_NS_QUESTIONS = [
    Q("Is there a person in the photo?", ["yes", "no"]),
    Q("Is the photo taken underwater?", ["yes", "no", "does not apply"]),
    Q("What colour is the water?", ["blue", "green", "brown", "no water"]),
]
TT_NS_Q1 = {"counts": [12, 28], "unclear": 0, "compared": [0, 1],
            "d": {0: -0.75, 1: 0.32}, "d_pct": {0: 4.4, 1: 95.6}, "eta2_pct": 3.0}
TT_NS_Q2 = {"counts": [6, 4, 2], "unclear": 28, "compared": [0], "d": {}, "d_pct": {}, "eta2_pct": None}
TT_NS_Q3 = {"counts": [9, 7, 3, 19], "unclear": 2, "compared": [0, 1, 3],
            "d": {0: 0.1, 1: -0.6666, 3: 0.25}, "d_pct": {0: 60.0, 1: 7.5, 3: 51.0}, "eta2_pct": 88.5}
TT_NS_HEAD = header_lines(40, 3, "tench, goldfish, great white shark", 508) + [
    "",
    "Q1. Is there a person in the photo?",
    "separation: percentile 3",
    "- yes: 12 photos, d -0.75, percentile 4",
    "- no: 28 photos, d +0.32, percentile 96",
    "",
    "Q2. Is the photo taken underwater?",
    "separation: too few photos per answer to compare",
]
TT_NS_Q2_ANSWERS = [
    "- yes: 6 photos (too few to compare)",
    "- no: 4 photos (too few to compare)",
    "- does not apply: 2 photos (too few to compare)",
    "- could not tell: 28 photos",
]
TT_NS_TAIL = [
    "",
    "Q3. What colour is the water?",
    "separation: percentile 88",
    "- blue: 9 photos, d +0.10, percentile 60",
    "- green: 7 photos, d -0.67, percentile 8",
    "- brown: 3 photos (too few to compare)",
    "- no water: 19 photos, d +0.25, percentile 51",
    "- could not tell: 2 photos",
]
TT_NS_WANT = "\n".join(TT_NS_HEAD + TT_NS_Q2_ANSWERS + TT_NS_TAIL)


def _copy_qs(qs):
    out = []
    for e in qs:
        if e is None:
            out.append(None)
            continue
        out.append({"counts": list(e["counts"]), "unclear": e["unclear"], "compared": list(e["compared"]),
                    "d": dict(e["d"]), "d_pct": dict(e["d_pct"]), "eta2_pct": e["eta2_pct"]})
    return out


def _call_table(classes, n, m, questions, qs):
    got = proploop.table_text(list(classes), n, m, [dict(q, answers=list(q["answers"])) for q in questions],
                              _copy_qs(qs))
    if not isinstance(got, str):
        raise TypeError(f"returned {type(got).__name__}, expected str")
    # A single trailing newline is tolerated; nothing else may differ.
    return got[:-1] if got.endswith("\n") else got


def test_table_text_main():
    tag = "table_text main case"
    try:
        got = _call_table(TT_MAIN_CLASSES, 118, 256, TT_MAIN_QUESTIONS, TT_MAIN_QS)
    except Exception as e:
        fail(f"{tag}: raised {err(e)}")
        return
    check(got == TT_MAIN_WANT, f"{tag}: {first_diff(got, TT_MAIN_WANT)}")


def test_table_text_no_stats():
    tag = "table_text question without statistics (entry with counts, < 2 compared, eta2_pct None)"
    try:
        got = _call_table(TT_NS_CLASSES, 40, 508, TT_NS_QUESTIONS, [TT_NS_Q1, TT_NS_Q2, TT_NS_Q3])
    except Exception as e:
        fail(f"{tag}: raised {err(e)}")
        return
    check(got == TT_NS_WANT, f"{tag}: {first_diff(got, TT_NS_WANT)}")


def test_table_text_none_entry():
    # The Interface's None entry carries no counts, so only the lines that do not need them are checked:
    # everything up to the question's separation line, and the blank line plus everything from the next
    # question on.
    tag = "table_text question without statistics (None entry)"
    try:
        got = _call_table(TT_NS_CLASSES, 40, 508, TT_NS_QUESTIONS, [TT_NS_Q1, None, TT_NS_Q3])
    except Exception as e:
        fail(f"{tag}: raised {err(e)}")
        return
    head = "\n".join(TT_NS_HEAD)
    tail = "\n" + "\n".join(TT_NS_TAIL)
    check(got.startswith(head), f"{tag}: start differs, {first_diff(got[:len(head)], head)}")
    check(got.endswith(tail), f"{tag}: end differs, {first_diff(got[-len(tail):], tail)}")


TESTS = [
    test_top_classes,
    test_parse_questions,
    test_questions_block,
    test_parse_answers,
    test_centre,
    test_question_stats,
    test_percentile,
    test_permute_within,
    test_table_text_main,
    test_table_text_no_stats,
    test_table_text_none_entry,
]


def main():
    global proploop
    try:
        import proploop  # noqa: F401
    except Exception as e:
        print(f"FAIL import proploop: {err(e)}".replace("\n", " "))
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
