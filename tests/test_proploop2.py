"""Black-box tests of tools/proploop2.py: merge_rows, all_questions, rows_in_set_order.

Expected values are derived by hand from authors/spec_proploop2.md (Interface; Questions and answers).
Plain Python, no pytest, no GPU, model or Claude. Prints one line per failure, or exactly `tests ok`.

    python -I tests/test_proploop2.py
"""
import copy
import os
import sys

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


def attempt(fn):
    """Run fn(). (True, exception) if it raised anything (SystemExit included), else (False, result)."""
    try:
        out = fn()
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        return True, e
    return False, out


def is_rows(x):
    """A list of rows, each row a list (the spec: answer rows are lists)."""
    return isinstance(x, list) and all(isinstance(r, list) for r in x)


def show(x):
    return f"{type(x).__name__} {x!r}"


# ---------------------------------------------------------------------------------------------
# merge_rows(rows_a, rows_b): row i of the result = rows_a[i] + rows_b[i]; error on a length mismatch.
# Each case builds fresh inputs; expected values are literals.
# ---------------------------------------------------------------------------------------------
MERGE_CASES = [
    ("three rows, ints and None",
     lambda: ([[1, 0], [0, 1], [1, 1]],
              [[None, 1, 0], [1, 1, 1], [0, None, None]]),
     [[1, 0, None, 1, 0], [0, 1, 1, 1, 1], [1, 1, 0, None, None]]),
    ("one row, strings",
     lambda: ([["yes", "no"]], [["no", "yes", "unclear"]]),
     [["yes", "no", "no", "yes", "unclear"]]),
    ("8 own + 2 standing per row",
     lambda: ([[1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 1, 1, 1]],
              [[None, 1], [1, 0]]),
     [[1, 0, 1, 0, 1, 0, 1, 0, None, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]]),
    ("rows of None on the right (failed standing batch)",
     lambda: ([[1, 1], [0, 1]], [[None, None, None], [None, None, None]]),
     [[1, 1, None, None, None], [0, 1, None, None, None]]),
    ("empty rows on the left",
     lambda: ([[], []], [[1], [2]]),
     [[1], [2]]),
    ("empty rows on the right",
     lambda: ([[3], [4]], [[], []]),
     [[3], [4]]),
    ("no rows",
     lambda: ([], []),
     []),
]

MERGE_MISMATCH = [
    ("2 vs 3 rows", lambda: ([[1], [0]], [[1], [0], [1]])),
    ("3 vs 2 rows", lambda: ([[1], [0], [1]], [[1], [0]])),
    ("0 vs 1 rows", lambda: ([], [[1]])),
    ("1 vs 0 rows", lambda: ([[1]], [])),
    ("4 vs 1 rows", lambda: ([[1, 0], [0, 0], [1, 1], [0, 1]], [[None]])),
]


def test_merge_rows():
    for name, build, want in MERGE_CASES:
        tag = f"merge_rows {name}"
        a, b = build()
        raised, got = attempt(lambda: proploop2.merge_rows(a, b))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        check(is_rows(got) and got == want, f"{tag}: got {show(got)}, expected {want!r}")


def test_merge_rows_mismatch():
    for name, build in MERGE_MISMATCH:
        tag = f"merge_rows length mismatch {name}"
        a, b = build()
        raised, got = attempt(lambda: proploop2.merge_rows(a, b))
        check(raised, f"{tag}: returned {show(got)}, expected an error")


# ---------------------------------------------------------------------------------------------
# all_questions(own, standing): own followed by standing, as a new list; inputs unchanged.
# ---------------------------------------------------------------------------------------------
AQ_CASES = [
    ("8 own + 3 standing, strings",
     lambda: (["own 1", "own 2", "own 3", "own 4", "own 5", "own 6", "own 7", "own 8"],
              ["standing 1", "standing 2", "standing 3"]),
     ["own 1", "own 2", "own 3", "own 4", "own 5", "own 6", "own 7", "own 8",
      "standing 1", "standing 2", "standing 3"]),
    ("dict questions",
     lambda: ([{"q": "Is it red?"}, {"q": "Is it round?"}], [{"q": "Is there a face?"}]),
     [{"q": "Is it red?"}, {"q": "Is it round?"}, {"q": "Is there a face?"}]),
    ("order kept within each list",
     lambda: (["z", "a", "m"], ["y", "b"]),
     ["z", "a", "m", "y", "b"]),
    ("same text in both is kept twice",
     lambda: (["a", "b"], ["b", "c"]),
     ["a", "b", "b", "c"]),
    ("no standing",
     lambda: (["a", "b"], []),
     ["a", "b"]),
    ("no own",
     lambda: ([], ["s"]),
     ["s"]),
    ("both empty",
     lambda: ([], []),
     []),
]


def test_all_questions():
    for name, build, want in AQ_CASES:
        tag = f"all_questions {name}"
        own, standing = build()
        own_before = copy.deepcopy(own)
        standing_before = copy.deepcopy(standing)
        raised, got = attempt(lambda: proploop2.all_questions(own, standing))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        check(isinstance(got, list) and got == want, f"{tag}: got {show(got)}, expected {want!r}")
        check(own == own_before, f"{tag}: own was changed to {own!r} (was {own_before!r})")
        check(standing == standing_before,
              f"{tag}: standing was changed to {standing!r} (was {standing_before!r})")
        check(got is not own, f"{tag}: returned the own list itself, not a new list")
        check(got is not standing, f"{tag}: returned the standing list itself, not a new list")
        if isinstance(got, list) and got is not own and got is not standing:
            got.append("__appended_by_test__")
            check(own == own_before, f"{tag}: appending to the result changed own to {own!r}")
            check(standing == standing_before,
                  f"{tag}: appending to the result changed standing to {standing!r}")


# ---------------------------------------------------------------------------------------------
# rows_in_set_order(batches, set_ids, nq): one row per photo of set_ids, in set order.
# batches: list of (photo index list, rows or None). A failed batch (None) gives each of its photos
# a row of nq Nones. Error if a photo of the set is missing from the batches, or repeated in them.
# ---------------------------------------------------------------------------------------------
ORDER_CASES = [
    # (name, build -> (batches, set_ids), nq, expected)
    ("two batches, neither in set order",
     lambda: ([([10, 50], [["r10a", "r10b"], ["r50a", "r50b"]]),
               ([30, 40, 20], [["r30a", "r30b"], ["r40a", "r40b"], ["r20a", "r20b"]])],
              [40, 10, 30, 20, 50]),
     2,
     [["r40a", "r40b"], ["r10a", "r10b"], ["r30a", "r30b"], ["r20a", "r20b"], ["r50a", "r50b"]]),
    ("one batch already in set order",
     lambda: ([([5, 6, 7], [[1], [0], [1]])], [5, 6, 7]),
     1,
     [[1], [0], [1]]),
    ("one batch, set reversed",
     lambda: ([([5, 6, 7], [[1], [0], [None]])], [7, 6, 5]),
     1,
     [[None], [0], [1]]),
    ("failed batch in the middle",
     lambda: ([([3, 7], [[1, 0, 1], [0, 0, 1]]),
               ([11, 2], None),
               ([9, 5], [[1, 1, 0], [0, 1, 1]])],
              [7, 3, 11, 5, 2, 9]),
     3,
     [[0, 0, 1], [1, 0, 1], [None, None, None], [0, 1, 1], [None, None, None], [1, 1, 0]]),
    ("failed batches first and last",
     lambda: ([([0, 1], None), ([2, 3], [[1, 0], [0, 1]]), ([4], None)], [4, 3, 2, 1, 0]),
     2,
     [[None, None], [0, 1], [1, 0], [None, None], [None, None]]),
    ("every batch failed, nq=5",
     lambda: ([([2, 1], None)], [1, 2]),
     5,
     [[None, None, None, None, None], [None, None, None, None, None]]),
    ("every batch failed, nq=1",
     lambda: ([([4], None), ([3], None)], [3, 4]),
     1,
     [[None], [None]]),
    ("empty set, no batches",
     lambda: ([], []),
     3,
     []),
]

ORDER_ERRORS = [
    # (name, build -> (batches, set_ids), nq)
    ("photo 3 missing",
     lambda: ([([1, 2], [[0], [1]]), ([4], [[1]])], [1, 2, 3, 4]), 1),
    ("photo 3 missing, another batch failed",
     lambda: ([([1, 2], None), ([4], [[1]])], [1, 2, 3, 4]), 1),
    ("first photo of the set missing",
     lambda: ([([2, 3], [[0], [1]])], [1, 2, 3]), 1),
    ("no batches for a set of one",
     lambda: ([], [9]), 1),
    ("photo 2 in two batches",
     lambda: ([([1, 2], [[0], [1]]), ([2, 3], [[1], [0]])], [1, 2, 3]), 1),
    ("photo 1 twice in one batch, same rows",
     lambda: ([([1, 1, 2], [[0], [0], [1]])], [1, 2]), 1),
    ("photo 2 in a good and a failed batch",
     lambda: ([([1, 2], [[0], [1]]), ([2, 3], None)], [1, 2, 3]), 1),
    ("photo 3 in two failed batches",
     lambda: ([([1, 3], None), ([2, 3], None)], [1, 2, 3]), 2),
]


def test_rows_in_set_order():
    for name, build, nq, want in ORDER_CASES:
        tag = f"rows_in_set_order {name} nq={nq}"
        batches, set_ids = build()
        raised, got = attempt(lambda: proploop2.rows_in_set_order(batches, set_ids, nq))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        check(is_rows(got) and got == want, f"{tag}: got {show(got)}, expected {want!r}")


def test_rows_in_set_order_nq_keyword():
    tag = "rows_in_set_order failed batch, nq passed by keyword"
    batches = [([3, 7], [[1, 0, 1], [0, 0, 1]]), ([11, 2], None), ([9, 5], [[1, 1, 0], [0, 1, 1]])]
    set_ids = [7, 3, 11, 5, 2, 9]
    want = [[0, 0, 1], [1, 0, 1], [None, None, None], [0, 1, 1], [None, None, None], [1, 1, 0]]
    raised, got = attempt(lambda: proploop2.rows_in_set_order(batches, set_ids, nq=3))
    if raised:
        fail(f"{tag}: raised {err(got)}")
        return
    check(is_rows(got) and got == want, f"{tag}: got {show(got)}, expected {want!r}")


def test_rows_in_set_order_errors():
    for name, build, nq in ORDER_ERRORS:
        tag = f"rows_in_set_order {name}"
        batches, set_ids = build()
        raised, got = attempt(lambda: proploop2.rows_in_set_order(batches, set_ids, nq))
        check(raised, f"{tag}: returned {show(got)}, expected an error")


# ---------------------------------------------------------------------------------------------
# The three together: round-25 rows and standing rows put in set order, then merged per photo.
# ---------------------------------------------------------------------------------------------
def test_merge_of_set_ordered_rows():
    tag = "merge_rows(rows_in_set_order(round-25), rows_in_set_order(standing, one batch failed))"
    set_ids = [30, 10, 20]
    r25 = [([10, 20], [[1, 0], [0, 0]]), ([30], [[1, 1]])]
    st = [([10, 20], None), ([30], [["s30"]])]
    want = [[1, 1, "s30"], [1, 0, None], [0, 0, None]]
    raised, got = attempt(lambda: proploop2.merge_rows(proploop2.rows_in_set_order(r25, set_ids, 2),
                                                       proploop2.rows_in_set_order(st, set_ids, 1)))
    if raised:
        fail(f"{tag}: raised {err(got)}")
        return
    check(is_rows(got) and got == want, f"{tag}: got {show(got)}, expected {want!r}")
    raised, qs = attempt(lambda: proploop2.all_questions(["Q1", "Q2"], ["S1"]))
    if raised:
        fail(f"{tag}: all_questions raised {err(qs)}")
        return
    if isinstance(qs, list) and is_rows(got):
        bad = [r for r in got if len(r) != len(qs)]
        check(not bad, f"{tag}: rows {bad!r} do not have one entry per question of {qs!r}")


TESTS = [
    test_merge_rows,
    test_merge_rows_mismatch,
    test_all_questions,
    test_rows_in_set_order,
    test_rows_in_set_order_nq_keyword,
    test_rows_in_set_order_errors,
    test_merge_of_set_ordered_rows,
]


def main():
    global proploop2
    try:
        import proploop2  # noqa: F401
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        print(f"FAIL import proploop2: {err(e)}".replace("\n", " "))
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
