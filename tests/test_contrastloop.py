"""Black-box tests of tools/contrastloop.py, written from authors/spec_contrastloop.md alone:
scaled, split, range_panel, peers, peer_photos, parse_predictions, contrast, outcome, results_text,
looking_content.

Every expected value is derived by hand from the spec (Photos; Looking; Checks; Results text;
Interface) on small synthetic arrays, hand names (their crc32 parity computed here with zlib) and hand
JSON replies. Plain Python, no pytest, no GPU, model or Claude. Prints one line per failure, or exactly
`tests ok`.

    python -I tests/test_contrastloop.py

Where the spec leaves something open, only what it fixes is tested; each such place is marked GAP.
"""
import json
import math
import os
import sys
import warnings
import zlib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

FAILURES = []
TOL = 1e-6  # float tolerance (loose enough for a float32 implementation; tight enough to tell ddof 0 from 1)


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


def as_float(x):
    try:
        return float(x)
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------
# scaled(V): each column divided by its max over the rows; a column whose max is 0 is all 0
# (also when it has negative entries: "a column with max 0 is all 0").
# GAP: a column whose max is below 0 (the rule as written would divide by a negative max) is not tested.
# ---------------------------------------------------------------------------------------------
SCALED_CASES = [
    ("an all-zero column, a max-0 column with a negative entry, a mixed-sign column",
     [[1.0, 0.0, 2.0, -1.0, 0.0],
      [2.0, 0.0, 4.0, 2.0, -3.0],
      [4.0, 0.0, 1.0, 1.0, 0.0]],
     [[0.25, 0.0, 0.5, -0.5, 0.0],
      [0.5, 0.0, 1.0, 1.0, 0.0],
      [1.0, 0.0, 0.25, 0.5, 0.0]]),
    ("one row", [[3.0, 0.0, 5.0]], [[1.0, 0.0, 1.0]]),
    ("non-dyadic values",
     [[0.3, 0.2], [0.6, 0.8], [0.15, 0.4]],
     [[0.5, 0.25], [1.0, 1.0], [0.25, 0.5]]),
]


def test_scaled():
    for name, v, want in SCALED_CASES:
        tag = f"scaled {name}"
        V = np.array(v, dtype=float)
        raised, got = attempt(lambda: contrastloop.scaled(V.copy()))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        try:
            g = np.asarray(got, dtype=float)
        except Exception as e:
            fail(f"{tag}: got {show(got)}, not an array of numbers: {err(e)}")
            continue
        w = np.array(want, dtype=float)
        if g.shape != w.shape:
            fail(f"{tag}: shape {g.shape}, expected {w.shape}")
        elif not np.all(np.isfinite(g)):
            fail(f"{tag}: non-finite values {g.tolist()!r}, expected {want!r} (a column with max 0 is all 0)")
        else:
            check(np.allclose(g, w, rtol=0, atol=TOL), f"{tag}: got {g.tolist()!r}, expected {want!r}")


# ---------------------------------------------------------------------------------------------
# split(set_ids, names): D = photos of set_ids with crc32(name.encode()) even, T = odd; each a list
# in set (index) order. Parity of the hand names is computed here with zlib.
# ---------------------------------------------------------------------------------------------
SPLIT_NAMES = (["ILSVRC2012_val_%08d.JPEG" % i for i in range(1, 17)]
               + ["n01440764_10026.JPEG", "n02102040_7256.JPEG", "café_0001.jpg",
                  "Ünïcode photo 2.png", "a", "abc", "123456789", "x y z.jpg"])
SPLIT_SET = [0, 1, 3, 4, 6, 7, 9, 10, 12, 15, 16, 18, 19, 20, 21, 23]  # not 2, 5, 8, 11, 13, 14, 17, 22


def crc_parity(name):
    return zlib.crc32(name.encode("utf-8")) % 2


def split_want(set_ids):
    d = [i for i in set_ids if crc_parity(SPLIT_NAMES[i]) == 0]
    t = [i for i in set_ids if crc_parity(SPLIT_NAMES[i]) == 1]
    return d, t


SPLIT_CASES = [
    ("16 of 24 hand names", SPLIT_SET),
    ("all 24 names", list(range(len(SPLIT_NAMES)))),
    ("one photo (non-ASCII name)", [18]),
    ("first and last photo", [0, 23]),
    ("empty set", []),
]


def test_split():
    d_all, t_all = split_want(SPLIT_SET)
    check(d_all and t_all,
          f"split fixture: the hand names give D={d_all!r}, T={t_all!r}; both should be non-empty")
    for name, set_ids in SPLIT_CASES:
        tag = f"split {name}"
        want = split_want(set_ids)
        raised, got = attempt(lambda: contrastloop.split(list(set_ids), list(SPLIT_NAMES)))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        if not (isinstance(got, (tuple, list)) and len(got) == 2):
            fail(f"{tag}: got {show(got)}, expected a pair (D, T)")
            continue
        for part, g, w in (("D", got[0], want[0]), ("T", got[1], want[1])):
            check(isinstance(g, list) and as_ids(g) == w, f"{tag}: {part} is {show(g)}, expected {w!r}")


# ---------------------------------------------------------------------------------------------
# range_panel(vals, labels, D, classes, k=4, cap=6): per class of the first k, its D photos sorted by
# vals highest first, ties by lower photo index; when more than 6, the 3 highest then the 3 lowest.
# The 3 lowest stay in the same highest-first order (sorted[:3] + sorted[-3:]): the list is "sorted ...
# highest first" and Part 1 shows each class "from most to least active".
# GAP: cap values other than 6 (how many highest / lowest for another or an odd cap) and fewer classes
# than k are not tested.
# ---------------------------------------------------------------------------------------------
RP_N = 54
RP_OTHER_LABEL = 8          # a label that is never one of the classes
RP_CLASSES = [7, 3, 9, 5, 1]
RP_PHOTOS = {               # photo id: (label, value, in D)
    # class 7: 10 D photos; 0/2/6/10 tie across the top cut, 12/18 tie across the bottom cut;
    # 1 and 3 are of the class but not in D
    0: (7, 9.0, True), 2: (7, 9.0, True), 4: (7, 1.0, True), 6: (7, 9.0, True), 8: (7, 3.0, True),
    10: (7, 9.0, True), 12: (7, 2.0, True), 14: (7, 3.0, True), 16: (7, 0.5, True), 18: (7, 2.0, True),
    1: (7, 100.0, False), 3: (7, -5.0, False),
    # class 3: exactly 6 D photos (no cut), 22/24 tied; 21 not in D
    20: (3, 1.0, True), 22: (3, 4.0, True), 24: (3, 4.0, True), 26: (3, -2.0, True), 28: (3, 0.0, True),
    30: (3, 3.5, True), 21: (3, 50.0, False),
    # class 9: 7 D photos (one dropped from the middle)
    32: (9, 0.1, True), 34: (9, 0.7, True), 36: (9, 0.3, True), 38: (9, 0.5, True), 40: (9, 0.2, True),
    42: (9, 0.6, True), 44: (9, 0.4, True),
    # class 5: 2 tied D photos; 47 not in D
    46: (5, 2.0, True), 48: (5, 2.0, True), 47: (5, 9.0, False),
    # class 1 (the 5th class): 1 D photo
    50: (1, 10.0, True),
    # a D photo of a label that is not a class
    52: (RP_OTHER_LABEL, 1000.0, True),
}
RP_LABELS = np.full(RP_N, RP_OTHER_LABEL, dtype=int)
RP_VALS = np.zeros(RP_N, dtype=float)
for _i, (_lab, _val, _ind) in RP_PHOTOS.items():
    RP_LABELS[_i] = _lab
    RP_VALS[_i] = _val
RP_D = sorted(i for i, (_l, _v, ind) in RP_PHOTOS.items() if ind)

# class 7 sorted: 0, 2, 6, 10 (9.0), 8, 14 (3.0), 12, 18 (2.0), 4 (1.0), 16 (0.5) -> 3 highest + 3 lowest
RP_WANT = {
    7: [0, 2, 6, 18, 4, 16],
    3: [22, 24, 30, 20, 28, 26],
    9: [34, 42, 38, 36, 40, 32],
    5: [46, 48],
    1: [50],
    6: [],                  # label 6: no photos at all
}
RP_CASES = [
    ("default k=4 cap=6: order, ties, the cap", RP_CLASSES, {},
     [RP_WANT[7], RP_WANT[3], RP_WANT[9], RP_WANT[5]]),
    ("cap=6 by keyword", RP_CLASSES, {"cap": 6},
     [RP_WANT[7], RP_WANT[3], RP_WANT[9], RP_WANT[5]]),
    ("k=2", RP_CLASSES, {"k": 2}, [RP_WANT[7], RP_WANT[3]]),
    ("k=5", RP_CLASSES, {"k": 5}, [RP_WANT[7], RP_WANT[3], RP_WANT[9], RP_WANT[5], RP_WANT[1]]),
    ("a class with no D photos gives an empty list", [6, 7, 3, 9], {},
     [RP_WANT[6], RP_WANT[7], RP_WANT[3], RP_WANT[9]]),
]


def test_range_panel():
    for name, classes, kw, want in RP_CASES:
        tag = f"range_panel {name}"
        raised, got = attempt(lambda: contrastloop.range_panel(RP_VALS.copy(), RP_LABELS.copy(), list(RP_D),
                                                               list(classes), **kw))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        ok = (isinstance(got, list) and all(isinstance(x, list) for x in got)
              and [as_ids(x) for x in got] == want)
        check(ok, f"{tag}: got {show(got)}, expected {want!r}")


# ---------------------------------------------------------------------------------------------
# peers(V, labels, c, k=3): the k channels j != c whose class-mean vectors (mean of V[:, j] per label,
# over all labels present) correlate best (Pearson) with c's; ties by lower j; an undefined correlation
# (zero variance) counts as lowest. Returns [(channel, correlation)].
# Labels are not contiguous (0, 4, 9 / 1, 2, 3 in mixed photo order) and class sizes differ; tied
# channels have identical class-mean vectors (so the tie is exact) but different photo values; raw
# per-photo correlations would not give these values. All class means are exact small numbers.
# GAP: the correlation reported for an undefined case (NaN, None, ...) and the order among several
# undefined channels are not tested; only that an undefined one ranks last.
# ---------------------------------------------------------------------------------------------
# Fixture A, 6 photos x 7 channels. Labels: photo 0 -> 4; photos 1, 3 -> 0; photos 2, 4, 5 -> 9.
# Class means (labels 0, 4, 9): ch0 [2,4,6] r=1; ch1 [3,2,1] r=-1; ch2 = c [1,2,3]; ch3 [1,3,2] r=0.5;
# ch4 [5,5,5] undefined; ch5 [2,4,6] r=1; ch6 [1,3,2] r=0.5.
PEER_LABELS_A = np.array([4, 0, 9, 0, 9, 9])
PEER_V_A = np.array([
    [4, 2, 2, 3, 5, 4, 3],
    [2, 3, 0, 1, 4, 0, 2],
    [6, 1, 3, 2, 5, 9, 0],
    [2, 3, 2, 1, 6, 4, 0],
    [6, 1, 1, 2, 4, 3, 4],
    [6, 1, 5, 2, 6, 6, 2],
], dtype=float)
# Fixture B, 4 photos x 6 channels. Labels: photo 0 -> 3; photos 1, 3 -> 1; photo 2 -> 2.
# Class means (labels 1, 2, 3): ch0 [7,7,7] undefined; ch1 = c [1,2,3]; ch2 [3,2,1] r=-1;
# ch3 [3,1,2] r=-0.5; ch4 [3,1,2] r=-0.5; ch5 all 0 undefined.
# Every defined correlation is negative, so an undefined one treated as 0 (or as high) would be picked.
PEER_LABELS_B = np.array([3, 1, 2, 1])
PEER_V_B = np.array([
    [7, 3, 1, 2, 2, 0],
    [6, 1, 3, 3, 2, 0],
    [7, 2, 2, 1, 1, 0],
    [8, 1, 3, 3, 4, 0],
], dtype=float)

PEERS_CASES = [
    # (name, V, labels, c, kwargs, [(channel, correlation or None = not checked)])
    ("default k=3: ties by lower channel, c excluded", PEER_V_A, PEER_LABELS_A, 2, {},
     [(0, 1.0), (5, 1.0), (3, 0.5)]),
    ("k=4", PEER_V_A, PEER_LABELS_A, 2, {"k": 4},
     [(0, 1.0), (5, 1.0), (3, 0.5), (6, 0.5)]),
    ("k=5", PEER_V_A, PEER_LABELS_A, 2, {"k": 5},
     [(0, 1.0), (5, 1.0), (3, 0.5), (6, 0.5), (1, -1.0)]),
    ("k=6: the undefined correlation ranks last", PEER_V_A, PEER_LABELS_A, 2, {"k": 6},
     [(0, 1.0), (5, 1.0), (3, 0.5), (6, 0.5), (1, -1.0), (4, None)]),
    ("undefined correlations rank below negative ones", PEER_V_B, PEER_LABELS_B, 1, {},
     [(3, -0.5), (4, -0.5), (2, -1.0)]),
]


def test_peers():
    for name, V, labels, c, kw, want in PEERS_CASES:
        tag = f"peers {name}"
        raised, got = attempt(lambda: contrastloop.peers(V.copy(), labels.copy(), c, **kw))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        if not isinstance(got, list):
            fail(f"{tag}: got {show(got)}, expected a list of (channel, correlation)")
            continue
        try:
            pairs = all(len(p) == 2 for p in got)
            chans = [int(p[0]) for p in got]
            corrs = [p[1] for p in got]
        except Exception as e:
            fail(f"{tag}: got {show(got)}, expected a list of (channel, correlation): {err(e)}")
            continue
        if not pairs:
            fail(f"{tag}: got {show(got)}, expected pairs (channel, correlation)")
            continue
        want_ch = [w[0] for w in want]
        if chans != want_ch:
            fail(f"{tag}: channels {chans!r}, expected {want_ch!r} (got {got!r})")
            continue
        for (ch, wr), gr in zip(want, corrs):
            if wr is None:
                continue
            g = as_float(gr)
            check(g is not None and abs(g - wr) <= TOL,
                  f"{tag}: correlation of channel {ch} is {gr!r}, expected {wr!r}")


# ---------------------------------------------------------------------------------------------
# peer_photos(S, labels, D, c, p, n=2): Z = S[D] (columns c, p) minus their mean over the D photos of
# the same label; diff = Z_c - Z_p; more = the n D photos with the highest diff, less = the n with the
# lowest, ties by lower photo index.
# Values are eighths, so every mean and diff is exact. 14 photos, 3 channels (column 1 a distractor).
# Classes over D: A (label 5): 0, 2, 4, 6; B (label 8): 1, 3, 9, 11; C (label 2): 12, 13.
# Photos 5, 8 (label 5) and 7, 10 (label 8) are not in D and have extreme values.
# Centred diffs (eighths): +3: 4, 9, 12; +1: 1, 2; -1: 6, 11; -3: 0, 3, 13.
# Uncentred, centred over all photos of a label, or centred over all of D, the picks differ
# (raw diffs: A 1, 5, 7, 3; B -3, -7, -1, -5); a non-D photo picked would be 7/10 (more) or 5/8 (less).
# GAP: the order within more and within less is not stated (only which photos), so they are compared
# as sets, except for n=1; overlap of more and less when D has fewer than 2n photos is not tested.
# ---------------------------------------------------------------------------------------------
PP_LABELS = np.array([5, 8, 5, 8, 5, 5, 5, 8, 5, 8, 8, 8, 2, 2])
PP_D = [0, 1, 2, 3, 4, 6, 9, 11, 12, 13]
PP_S8 = [            # (column 0, column 1 distractor, column 2) in eighths
    (4, 8, 3),       # 0  A  D   centred diff c=0,p=2: -3
    (2, 0, 5),       # 1  B  D   +1
    (6, 0, 1),       # 2  A  D   +1
    (0, 8, 7),       # 3  B  D   -3
    (7, 4, 0),       # 4  A  D   +3
    (0, 4, 8),       # 5  A  not D
    (5, 8, 2),       # 6  A  D   -1
    (8, 0, 0),       # 7  B  not D
    (0, 0, 8),       # 8  A  not D
    (3, 0, 4),       # 9  B  D   +3
    (8, 8, 0),       # 10 B  not D
    (1, 8, 6),       # 11 B  D   -1
    (5, 4, 2),       # 12 C  D   +3
    (2, 4, 5),       # 13 C  D   -3
]
PP_S = np.array(PP_S8, dtype=float) / 8.0
PP_CASES = [
    # (name, c, p, n or None for the default, more, less); a set = any order, a list = exactly
    ("c=0 p=2 default n", 0, 2, None, {4, 9}, {0, 3}),
    ("c=0 p=2 n=2", 0, 2, 2, {4, 9}, {0, 3}),
    ("c=0 p=2 n=1", 0, 2, 1, [4], [0]),
    ("c=0 p=2 n=3", 0, 2, 3, {4, 9, 12}, {0, 3, 13}),
    ("c=0 p=2 n=4 (ties at +1 and -1)", 0, 2, 4, {1, 4, 9, 12}, {0, 3, 6, 13}),
    ("c=2 p=0 (roles swapped: diff negated)", 2, 0, None, {0, 3}, {4, 9}),
]


def test_peer_photos():
    for name, c, p, n, want_more, want_less in PP_CASES:
        tag = f"peer_photos {name}"
        kw = {} if n is None else {"n": n}
        raised, got = attempt(lambda: contrastloop.peer_photos(PP_S.copy(), PP_LABELS.copy(), list(PP_D),
                                                               c, p, **kw))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        if not (isinstance(got, (tuple, list)) and len(got) == 2):
            fail(f"{tag}: got {show(got)}, expected a pair (more, less)")
            continue
        for part, g, w in (("more", got[0], want_more), ("less", got[1], want_less)):
            ids = as_ids(g) if isinstance(g, list) else None
            if isinstance(w, list):
                ok = ids == w
                desc = f"{w!r}"
            else:
                ok = ids is not None and len(ids) == len(w) and set(ids) == w
                desc = f"{sorted(w)!r} in any order"
            check(ok, f"{tag}: {part} is {show(g)}, expected the list {desc}")


# ---------------------------------------------------------------------------------------------
# parse_predictions(text): the last JSON object with key "predictions"; valid explanation; kept
# entries in order, at most the first 4 kept; most / least as answer indices; None when invalid.
# GAP: whether the stored explanation is stripped (only that it strips to the text is tested), extra
# keys in the returned dicts, and code-fence handling of bundlemax.last_obj are not tested.
# ---------------------------------------------------------------------------------------------
EXPL = "Open sky above a horizon, most of all at dusk."
MISSING = object()


def reply(obj, **kw):
    return json.dumps(obj, ensure_ascii=False, **kw)


def preds_reply(preds, expl=EXPL):
    return reply({"explanation": expl, "predictions": preds})


def entry(**kw):
    """A valid entry with fields overridden (MISSING removes the field)."""
    e = {"question": "Is it outdoors?", "answers": ["yes", "no", "not applicable"],
         "most_active": "yes", "least_active": "no"}
    for k, v in kw.items():
        if v is MISSING:
            e.pop(k, None)
        else:
            e[k] = v
    return e


# Valid entries and what they parse to: (question, answers, most, least).
P_SKY = {"question": "Is the sky visible?", "answers": ["yes", "no", "not applicable"],
         "most_active": "yes", "least_active": "no"}
W_SKY = ("Is the sky visible?", ["yes", "no", "not applicable"], 0, 1)
P_COLOUR = {"question": "  What colour is the main object?\n",
            "answers": [" Red ", "Blue", "green", "Other ", "No main object"],   # 5 answers
            "most_active": " red", "least_active": "BLUE  "}
W_COLOUR = ("What colour is the main object?", ["Red", "Blue", "green", "Other", "No main object"], 0, 1)
P_COUNT = {"question": "How many animals are there?", "answers": ["none", "one", "several"],
           "most_active": "Several", "least_active": " NONE "}
W_COUNT = ("How many animals are there?", ["none", "one", "several"], 2, 0)
P_PLACE = {"question": "Was the photo taken indoors or outdoors?", "answers": ["Indoors", "Outdoors"],
           "most_active": "outdoors", "least_active": "indoors", "why": "an extra key is allowed"}
W_PLACE = ("Was the photo taken indoors or outdoors?", ["Indoors", "Outdoors"], 1, 0)
P_TEXT = {"question": "Is there writing in the photo?", "answers": ["yes", "no"],
          "most_active": "no", "least_active": "yes"}
W_TEXT = ("Is there writing in the photo?", ["yes", "no"], 1, 0)
P_SIGN = {"question": "What does the street sign say?", "answers": ["Straße", "Gasse", "no sign"],
          "most_active": "STRASSE", "least_active": "gasse"}   # casefold: "straße" -> "strasse"
W_SIGN = ("What does the street sign say?", ["Straße", "Gasse", "no sign"], 0, 1)

# Each must be dropped. Where possible only one rule is broken.
BAD_ENTRIES = [
    ("entry a string", "Is it outdoors?"),
    ("entry a list", ["Is it outdoors?", ["yes", "no"], "yes", "no"]),
    ("entry a number", 7),
    ("entry null", None),
    ("entry an empty object", {}),
    ("question missing", entry(question=MISSING)),
    ("question empty", entry(question="")),
    ("question only whitespace", entry(question=" \n\t ")),
    ("question a number", entry(question=5)),
    ("question a list", entry(question=["Is it outdoors?"])),
    ("answers missing", entry(answers=MISSING)),
    ("answers a string", entry(answers="yes; no; not applicable")),
    ("answers an object", entry(answers={"yes": 1, "no": 0})),
    ("answers empty", entry(answers=[])),
    ("1 answer", entry(answers=["yes"])),
    ("6 answers", entry(answers=["yes", "no", "not applicable", "partly", "unclear", "other"])),
    ("an empty answer", entry(answers=["yes", "no", ""])),
    ("an answer of only spaces", entry(answers=["yes", "no", "   "])),
    ("an answer a number", entry(answers=["yes", "no", 3])),
    ("an answer null", entry(answers=["yes", "no", None])),
    ("answers repeat after strip", entry(answers=["yes", "no", "not applicable", "not applicable "])),
    ("answers repeat after casefold", entry(answers=["yes", "no", "Not applicable", "NOT APPLICABLE"])),
    ("answers repeat after casefold (ß)", entry(answers=["yes", "no", "Straße", "STRASSE"])),
    ("most_active missing", entry(most_active=MISSING)),
    ("least_active missing", entry(least_active=MISSING)),
    ("most_active matches no answer", entry(most_active="maybe")),
    ("least_active matches no answer", entry(least_active="never")),
    ("most_active only a prefix of an answer", entry(most_active="ye")),
    ("most_active with inner spaces changed", entry(most_active="not  applicable")),
    ("most_active an index, not a string", entry(most_active=0)),
    ("least_active an index, not a string", entry(least_active=1)),
    ("most_active null", entry(most_active=None)),
    ("most_active a list", entry(most_active=["yes"])),
    ("most equal to least", entry(most_active="no", least_active="no")),
    ("most equal to least after case and spaces", entry(most_active=" No", least_active="NO ")),
]

_PRETTY = json.dumps({"explanation": EXPL, "predictions": [P_SKY, P_COUNT]}, indent=2, ensure_ascii=False)

PARSE_CASES = [
    # (name, text, expected explanation, expected predictions or None for a None result)
    ("plain JSON, two valid", preds_reply([P_SKY, P_TEXT]), EXPL, [W_SKY, W_TEXT]),
    ("case and spaces in question, answers, most/least; 5 and 2 answers",
     preds_reply([P_COLOUR, P_COUNT, P_PLACE]), EXPL, [W_COLOUR, W_COUNT, W_PLACE]),
    ("casefold, not lower (ß)", preds_reply([P_SIGN]), EXPL, [W_SIGN]),
    ("bad entries dropped, order kept",
     preds_reply([P_SKY, BAD_ENTRIES[6][1], P_COUNT, BAD_ENTRIES[15][1], None, P_COLOUR]),
     EXPL, [W_SKY, W_COUNT, W_COLOUR]),
    ("6 valid: the first 4 kept",
     preds_reply([P_SKY, P_COLOUR, P_COUNT, P_PLACE, P_TEXT, P_SIGN]),
     EXPL, [W_SKY, W_COLOUR, W_COUNT, W_PLACE]),
    ("the first 4 kept, not the first 4 entries",
     preds_reply([P_SKY, BAD_ENTRIES[6][1], P_COLOUR, BAD_ENTRIES[15][1], P_COUNT, P_PLACE, P_TEXT]),
     EXPL, [W_SKY, W_COLOUR, W_COUNT, W_PLACE]),
    ("exactly 4", preds_reply([P_SIGN, P_TEXT, P_PLACE, P_COUNT]), EXPL, [W_SIGN, W_TEXT, W_PLACE, W_COUNT]),
    ("prose around pretty-printed JSON", "Here is what I see.\n\n" + _PRETTY + "\n\nI hope this helps.",
     EXPL, [W_SKY, W_COUNT]),
    ("two objects with predictions: the last one",
     "Draft:\n" + reply({"explanation": "First draft.", "predictions": [P_SKY]})
     + "\n\nOn reflection:\n" + reply({"explanation": "Final answer.", "predictions": [P_COUNT]}),
     "Final answer.", [W_COUNT]),
    ("a later object without predictions is not the one",
     preds_reply([P_SKY]) + '\n{"confidence": "medium"}', EXPL, [W_SKY]),
    # The spec parses only the last object with "predictions"; it names no fall-back to an earlier one.
    ("the last object with predictions is invalid",
     preds_reply([P_SKY]) + "\n" + reply({"explanation": "  ", "predictions": [P_COUNT]}), None, None),
    ("no explanation", reply({"predictions": [P_SKY]}), None, None),
    ("explanation empty", preds_reply([P_SKY], expl=""), None, None),
    ("explanation only whitespace", preds_reply([P_SKY], expl="  \n\t "), None, None),
    ("explanation null", preds_reply([P_SKY], expl=None), None, None),
    ("explanation a number", preds_reply([P_SKY], expl=42), None, None),
    ("explanation a list", preds_reply([P_SKY], expl=[EXPL]), None, None),
    ("predictions empty", preds_reply([]), None, None),
    ("predictions an object", preds_reply(P_SKY), None, None),
    ("predictions a string", preds_reply("Is the sky visible?"), None, None),
    ("predictions null", preds_reply(None), None, None),
    ("every entry bad", preds_reply([b for _n, b in BAD_ENTRIES[:8]]), None, None),
    ("no predictions key", reply({"explanation": EXPL}), None, None),
    ("empty text", "", None, None),
    ("no JSON", "The unit likes skies.", None, None),
    ("broken JSON", '{"explanation": "x", "predictions": [', None, None),
]


def parsed_view(got):
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
                    p.get("most"), p.get("least")))
    return out


def check_parsed(tag, got, want_expl, want_preds):
    if want_preds is None:
        check(got is None, f"{tag}: got {show(got)}, expected None")
        return
    if not isinstance(got, dict):
        fail(f"{tag}: got {show(got)}, expected a dict")
        return
    check(got.get("explanation") == want_expl,
          f"{tag}: explanation {got.get('explanation')!r}, expected {want_expl!r}")
    view = parsed_view(got)
    check(view == list(want_preds),
          f"{tag}: predictions (question, answers, most, least) {view!r}, expected {list(want_preds)!r}")


def test_parse_predictions():
    for name, text, want_expl, want_preds in PARSE_CASES:
        tag = f"parse_predictions {name}"
        raised, got = attempt(lambda: contrastloop.parse_predictions(text))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        check_parsed(tag, got, want_expl, want_preds)


def test_parse_predictions_bad_entries():
    for name, bad in BAD_ENTRIES:
        tag = f"parse_predictions bad entry ({name}) before a valid one"
        text = preds_reply([bad, P_TEXT])
        raised, got = attempt(lambda: contrastloop.parse_predictions(text))
        if raised:
            fail(f"{tag}: raised {err(got)}")
        else:
            check_parsed(tag, got, EXPL, [W_TEXT])
        tag = f"parse_predictions bad entry ({name}) alone"
        text = preds_reply([bad])
        raised, got = attempt(lambda: contrastloop.parse_predictions(text))
        if raised:
            fail(f"{tag}: raised {err(got)}")
        else:
            check_parsed(tag, got, None, None)


def test_parse_predictions_explanation_spaces():
    # Valid: non-empty after strip. GAP: whether it is stored stripped; only that it strips to the text.
    tag = "parse_predictions explanation with surrounding spaces"
    text = preds_reply([P_SKY], expl="  " + EXPL + " \n")
    raised, got = attempt(lambda: contrastloop.parse_predictions(text))
    if raised:
        fail(f"{tag}: raised {err(got)}")
        return
    if not isinstance(got, dict):
        fail(f"{tag}: got {show(got)}, expected a dict")
        return
    e = got.get("explanation")
    check(isinstance(e, str) and e.strip() == EXPL, f"{tag}: explanation {e!r}, expected {EXPL!r} (stripped)")
    view = parsed_view(got)
    check(view == [W_SKY], f"{tag}: predictions {view!r}, expected {[W_SKY]!r}")


# ---------------------------------------------------------------------------------------------
# contrast(R, ans, hi, lo, min_n=5): n_hi, n_lo = photos answered hi / lo; checkable when both >= min_n;
# d_j = (mean R_j over hi - mean over lo) / sd of R_j over the hi and lo photos (ddof 1), 0 when that sd
# is 0; None when not checkable.
# GAP: the type of ans is not fixed; the tests pass a list of answer indices (like hi and lo), and in one
# case None for an unanswered photo (proploop's failed answers).
# ---------------------------------------------------------------------------------------------
CT_ANS = [1, 0, 2, 1, 0, 1, 0, 2, 1, 0, 1, 0]   # hi=1: 0, 3, 5, 8, 10; lo=0: 1, 4, 6, 9, 11; 2: 2, 7
CT_R = np.array([
    [2.0, 3.0, 0.0],          # 0  hi
    [1.0, 3.0, 1.0],          # 1  lo
    [100.0, 50.0, 7.0],       # 2  other answer
    [4.0, 3.0, 0.0],          # 3  hi
    [3.0, 3.0, 1.0],          # 4  lo
    [6.0, 3.0, 0.0],          # 5  hi
    [5.0, 3.0, 1.0],          # 6  lo
    [-100.0, -50.0, -7.0],    # 7  other answer
    [8.0, 3.0, 0.0],          # 8  hi
    [7.0, 3.0, 1.0],          # 9  lo
    [10.0, 3.0, 0.0],         # 10 hi
    [9.0, 3.0, 1.0],          # 11 lo
])
# Column 0: hi 2,4,6,8,10 (mean 6), lo 1,3,5,7,9 (mean 5); all ten are 1..10, squared deviations from
# 5.5 sum to 82.5, ddof 1 -> var 82.5 / 9. (A pooled within-group sd would give 1/sqrt(10); ddof 0 differs.)
CT_D0 = 1.0 / math.sqrt(82.5 / 9)
# Column 1: 3.0 on every hi and lo photo -> sd 0 -> d 0 (the other-answer photos differ, and are ignored).
# Column 2: hi all 0, lo all 1; five 0s and five 1s: squared deviations sum 2.5, var 2.5 / 9 (ddof 0: -2).
CT_D2 = -1.0 / math.sqrt(2.5 / 9)
# Photo 10 answered 2 instead: hi 2,4,6,8 (mean 5) = lo mean 5 -> d0 0; column 2: four 0s, five 1s, mean
# 5/9, squared deviations 4 (5/9)^2 + 5 (4/9)^2 = 20/9, ddof 1 over 9 photos -> var (20/9) / 8.
CT_ANS_HI4 = [1, 0, 2, 1, 0, 1, 0, 2, 1, 0, 2, 0]
CT_D2_HI4 = -1.0 / math.sqrt((20 / 9) / 8)
CT_ANS_LO4 = [1, 0, 2, 1, 0, 1, 0, 2, 1, 0, 1, 2]           # photo 11 answered 2: 4 least-active photos
CT_ANS_NONE = [1, 0, None, 1, 0, 1, 0, None, 1, 0, 1, 0]   # photos 2 and 7 unanswered

CONTRAST_CASES = [
    # (name, ans, hi, lo, kwargs, expected (n_hi, n_lo, d) or None)
    ("5 and 5 photos (exactly min_n), a third answer ignored, zero-sd column",
     CT_ANS, 1, 0, {}, (5, 5, [CT_D0, 0.0, CT_D2])),
    ("hi and lo swapped", CT_ANS, 0, 1, {}, (5, 5, [-CT_D0, 0.0, -CT_D2])),
    ("unanswered photos (None) ignored", CT_ANS_NONE, 1, 0, {}, (5, 5, [CT_D0, 0.0, CT_D2])),
    ("4 most-active photos with min_n=4", CT_ANS_HI4, 1, 0, {"min_n": 4}, (4, 5, [0.0, 0.0, CT_D2_HI4])),
    ("too few: 4 most-active photos", CT_ANS_HI4, 1, 0, {}, None),
    ("too few: 4 least-active photos", CT_ANS_LO4, 1, 0, {}, None),
    ("too few: the most-active answer never given", CT_ANS, 3, 0, {}, None),
    ("too few: 5 and 5 with min_n=6", CT_ANS, 1, 0, {"min_n": 6}, None),
]


def test_contrast():
    for name, ans, hi, lo, kw, want in CONTRAST_CASES:
        tag = f"contrast {name}"
        raised, got = attempt(lambda: contrastloop.contrast(CT_R.copy(), list(ans), hi, lo, **kw))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        if want is None:
            check(got is None, f"{tag}: got {show(got)}, expected None (not checkable)")
            continue
        if not (isinstance(got, (tuple, list)) and len(got) == 3):
            fail(f"{tag}: got {show(got)}, expected (n_hi, n_lo, d)")
            continue
        n_hi, n_lo, d = got
        check(n_hi == want[0] and n_lo == want[1],
              f"{tag}: n_hi, n_lo = {n_hi!r}, {n_lo!r}, expected {want[0]}, {want[1]}")
        try:
            d = np.asarray(d, dtype=float)
        except Exception as e:
            fail(f"{tag}: d is {show(d)}, not an array of numbers: {err(e)}")
            continue
        if d.shape != (len(want[2]),):
            fail(f"{tag}: d has shape {d.shape}, expected ({len(want[2])},), one value per column of R")
            continue
        check(np.all(np.isfinite(d)) and np.allclose(d, want[2], rtol=0, atol=TOL),
              f"{tag}: d {d.tolist()!r}, expected {want[2]!r}")


# ---------------------------------------------------------------------------------------------
# outcome(d_unit, pct): "held" when pct >= 95 and d_unit > 0; else "not held".
# ---------------------------------------------------------------------------------------------
OUTCOME_CASES = [
    (1.0, 95, "held"),
    (0.01, 95.0, "held"),
    (1e-12, 95, "held"),
    (0.5, 100, "held"),
    (2.0, 99.5, "held"),
    (0.5, 94.99, "not held"),
    (0.5, 94, "not held"),
    (0.5, 0, "not held"),
    (0.0, 95, "not held"),
    (0.0, 100, "not held"),
    (-0.0, 99, "not held"),
    (-1e-12, 99, "not held"),
    (-0.3, 99, "not held"),
    (-0.3, 10, "not held"),
    (np.float64(0.2), np.float64(95.0), "held"),
    (np.float64(0.0), np.float64(95.0), "not held"),
]


def test_outcome():
    for d, pct, want in OUTCOME_CASES:
        tag = f"outcome(d={d!r}, pct={pct!r})"
        raised, got = attempt(lambda: contrastloop.outcome(d, pct))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        check(got == want, f"{tag}: got {show(got)}, expected {want!r}")


# ---------------------------------------------------------------------------------------------
# results_text(n, m, K, preds, res): the header paragraph, then, separated by blank lines, one block per
# prediction; the block lines are compared exactly.
# GAP: whether the header paragraph is one line (its wrapped lines joined with spaces) is stated only for
# the Looking content, so the header is compared with whitespace collapsed; a trailing newline is ignored;
# held / did not hold when the percentile is >= 95 but d <= 0 (the header says percentile only, outcome
# also needs d > 0) is not tested; proploop._whole on a non-whole percentile is not tested (only whole
# values are used); the sign of a d that rounds to 0.00 is not tested.
# ---------------------------------------------------------------------------------------------
RT_HEADER_LINES = [  # the spec's wrapped lines, joined with spaces
    "The predictions were then checked on new photos: {n} photos of the {K} photo classes on which the unit is most",
    "active on average, none of them among the photos looked at. Someone who never saw the unit's activity answered",
    "each question about each photo. For each prediction: the number of photos given each of the two answers it",
    "compares, and d, how much more active the unit is on the photos given the predicted most-active answer than on",
    "those given the predicted least-active answer, after removing the differences between classes, in standard",
    "deviations, with this unit's percentile among all {m} units of its layer on the same photos and answers. A",
    "prediction held when that percentile is at least 95.",
]
RT_HEADER = " ".join(RT_HEADER_LINES)

RT_CASES = [
    ("held, did not hold, too few, did not hold with d < 0", 1234, 512, 24,
     [{"question": "Is the sky visible?", "answers": ["yes", "no", "not applicable"], "most": 0, "least": 1},
      {"question": "What colour is the main object?",
       "answers": ["red", "blue", "green", "other", "no main object"], "most": 2, "least": 4},
      {"question": "How many animals are there?", "answers": ["none", "one", "several"], "most": 2, "least": 0},
      {"question": "Was the photo taken indoors?", "answers": ["indoors", "outdoors"], "most": 1, "least": 0}],
     [{"n_hi": 40, "n_lo": 37, "d": 1.2345, "pct": 97},
      {"n_hi": 25, "n_lo": 30, "d": 0.3051, "pct": 62.0},
      {"n_hi": 4, "n_lo": 12, "d": None, "pct": None},
      {"n_hi": 18, "n_lo": 9, "d": -0.4449, "pct": 3}],
     ["Prediction 1. Is the sky visible?\n"
      "Answers: yes; no; not applicable\n"
      "Predicted most active: yes. Predicted least active: no.\n"
      "Result: 40 and 37 photos, d +1.23, percentile 97: held.",
      "Prediction 2. What colour is the main object?\n"
      "Answers: red; blue; green; other; no main object\n"
      "Predicted most active: green. Predicted least active: no main object.\n"
      "Result: 25 and 30 photos, d +0.31, percentile 62: did not hold.",
      "Prediction 3. How many animals are there?\n"
      "Answers: none; one; several\n"
      "Predicted most active: several. Predicted least active: none.\n"
      "Result: 4 and 12 photos: too few to check (each needs at least 5).",
      "Prediction 4. Was the photo taken indoors?\n"
      "Answers: indoors; outdoors\n"
      "Predicted most active: outdoors. Predicted least active: indoors.\n"
      "Result: 18 and 9 photos, d -0.44, percentile 3: did not hold."]),
    ("held at percentile exactly 95, too few on the least-active side", 600, 2048, 12,
     [{"question": "Is there water in the photo?", "answers": ["yes", "no"], "most": 0, "least": 1},
      {"question": "Is the animal facing the camera?", "answers": ["facing", "not facing", "no animal"],
       "most": 0, "least": 2}],
     [{"n_hi": 12, "n_lo": 5, "d": 0.5, "pct": 95},
      {"n_hi": 12, "n_lo": 4, "d": None, "pct": None}],
     ["Prediction 1. Is there water in the photo?\n"
      "Answers: yes; no\n"
      "Predicted most active: yes. Predicted least active: no.\n"
      "Result: 12 and 5 photos, d +0.50, percentile 95: held.",
      "Prediction 2. Is the animal facing the camera?\n"
      "Answers: facing; not facing; no animal\n"
      "Predicted most active: facing. Predicted least active: no animal.\n"
      "Result: 12 and 4 photos: too few to check (each needs at least 5)."]),
]


def test_results_text():
    for name, n, m, K, preds, res, blocks in RT_CASES:
        tag = f"results_text {name}"
        raised, got = attempt(lambda: contrastloop.results_text(n, m, K, preds, res))
        if raised:
            fail(f"{tag}: raised {err(got)}")
            continue
        if not isinstance(got, str):
            fail(f"{tag}: got {show(got)}, expected a str")
            continue
        paras = got.rstrip("\n").split("\n\n")
        if len(paras) != 1 + len(blocks):
            fail(f"{tag}: {len(paras)} paragraphs separated by blank lines, expected {1 + len(blocks)}"
                 f" (header + one per prediction): {got!r}")
            continue
        header = RT_HEADER.format(n=n, m=m, K=K)
        check(" ".join(paras[0].split()) == header, f"{tag}: header {paras[0]!r}, expected {header!r}")
        for j, (g, w) in enumerate(zip(paras[1:], blocks), 1):
            check(g == w, f"{tag}: prediction block {j} is {g!r}, expected {w!r}")


# ---------------------------------------------------------------------------------------------
# looking_content(panel_a, panel_b, block): the content list of the Looking section, with a fake block.
# Texts compared exactly: wrapped spec lines joined with spaces, a paragraph break kept as a blank line.
# GAP: the form of a text block is not given by the spec; {"type": "text", "text": ...} (Claude API) is
# assumed. The default block (run_subjects.img_block of the thumb) is not tested; neither are the literal
# "4" / "3" in Parts 1 and 2 when the panels have other lengths (the fixture has 4 classes, 3 peers).
# ---------------------------------------------------------------------------------------------
LOOK_TEXT1 = (" ".join([
    "Below are photos with the activity of one unit (a single neuron) of an image classification network on each, as",
    "a percentage of the most it reaches on any of 10000 photos.",
]) + "\n\n" + " ".join([
    "Part 1. For each of the 4 photo classes on which the unit is most active on average, photos of that class, from",
    "most to least active.",
]))
LOOK_TEXT3 = " ".join([
    "Part 2. Similar units: the 3 other units of the same layer whose average activity over photo classes is most",
    "like this unit's. For each, photos on which this unit is much more active than the similar unit, and photos on",
    "which it is much less active, each compared with their own averages on the photo's class.",
])
LOOK_TEXT5 = " ".join([
    "Whatever separates the more active from the less active photos of one class cannot be the class itself, and",
    "whatever separates this unit from a similar unit is what is particular to it. From these photos, write hypotheses",
    "about what this unit's activity depends on, each as a prediction to be checked on new photos of the same classes:",
    "someone who never sees the unit's activity will answer a question about each new photo, and the unit's activity",
    "will be compared between photos given different answers, within the same class. For each prediction give the",
    "question, which must be answerable by looking at a small photo; 2 to 5 short answers that do not overlap, so that",
    "every photo fits exactly one of them, including one for photos to which the question does not apply; the answer",
    "on whose photos you predict the unit is most active; and the answer on whose photos you predict it is least",
    "active. Write up to 4 predictions: the ones you expect to separate the unit's activity most clearly, including",
    "from the similar units. Also write a short explanation of what the unit responds to. Reply with only JSON:",
    '{"explanation": "...", "predictions": [{"question": "...", "answers": ["...", "..."], "most_active": "...",',
    '"least_active": "..."}]}',
])

LOOK_PANEL_A = [
    ("goldfinch", [(101, 100), (57, 88), (3, 71), (942, 40), (12, 9), (77, 0)]),
    ("Granny Smith", [(220, 65), (18, 30), (5, 2)]),
    ("golf ball", [(9001, 12)]),
    ("tiger shark", [(44, 50), (45, 50)]),
]
LOOK_PANEL_B = [
    [(101, "goldfinch", 100, 20), (18, "Granny Smith", 30, 1), (942, "goldfinch", 40, 90),
     (77, "goldfinch", 0, 55)],
    [(3, "goldfinch", 71, 70), (5, "Granny Smith", 2, 0), (44, "tiger shark", 50, 99),
     (9001, "golf ball", 12, 64)],
    [(57, "goldfinch", 88, 0), (220, "Granny Smith", 65, 3), (12, "goldfinch", 9, 33),
     (45, "tiger shark", 50, 50)],
]
LOOK_WANT = [
    ("text", LOOK_TEXT1),
    ("text", "Class 1: goldfinch"),
    ("text", "Photo 1.1: 100%"), ("image", 101),
    ("text", "Photo 1.2: 88%"), ("image", 57),
    ("text", "Photo 1.3: 71%"), ("image", 3),
    ("text", "Photo 1.4: 40%"), ("image", 942),
    ("text", "Photo 1.5: 9%"), ("image", 12),
    ("text", "Photo 1.6: 0%"), ("image", 77),
    ("text", "Class 2: Granny Smith"),
    ("text", "Photo 2.1: 65%"), ("image", 220),
    ("text", "Photo 2.2: 30%"), ("image", 18),
    ("text", "Photo 2.3: 2%"), ("image", 5),
    ("text", "Class 3: golf ball"),
    ("text", "Photo 3.1: 12%"), ("image", 9001),
    ("text", "Class 4: tiger shark"),
    ("text", "Photo 4.1: 50%"), ("image", 44),
    ("text", "Photo 4.2: 50%"), ("image", 45),
    ("text", LOOK_TEXT3),
    ("text", "Similar unit 1"),
    ("text", "Photo S1.1 (goldfinch): this unit 100%, similar unit 1 20%"), ("image", 101),
    ("text", "Photo S1.2 (Granny Smith): this unit 30%, similar unit 1 1%"), ("image", 18),
    ("text", "Photo S1.3 (goldfinch): this unit 40%, similar unit 1 90%"), ("image", 942),
    ("text", "Photo S1.4 (goldfinch): this unit 0%, similar unit 1 55%"), ("image", 77),
    ("text", "Similar unit 2"),
    ("text", "Photo S2.1 (goldfinch): this unit 71%, similar unit 2 70%"), ("image", 3),
    ("text", "Photo S2.2 (Granny Smith): this unit 2%, similar unit 2 0%"), ("image", 5),
    ("text", "Photo S2.3 (tiger shark): this unit 50%, similar unit 2 99%"), ("image", 44),
    ("text", "Photo S2.4 (golf ball): this unit 12%, similar unit 2 64%"), ("image", 9001),
    ("text", "Similar unit 3"),
    ("text", "Photo S3.1 (goldfinch): this unit 88%, similar unit 3 0%"), ("image", 57),
    ("text", "Photo S3.2 (Granny Smith): this unit 65%, similar unit 3 3%"), ("image", 220),
    ("text", "Photo S3.3 (goldfinch): this unit 9%, similar unit 3 33%"), ("image", 12),
    ("text", "Photo S3.4 (tiger shark): this unit 50%, similar unit 3 50%"), ("image", 45),
    ("text", LOOK_TEXT5),
]
LOOK_IMAGES = 24  # 12 photos in panel_a + 3 x 4 in panel_b


def fake_block(i):
    return {"type": "image", "_test_photo": int(i)}


def classify(item):
    if isinstance(item, dict) and "_test_photo" in item:
        return ("image", item["_test_photo"])
    if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
        return ("text", item["text"])
    return ("other", item)


def test_looking_content():
    tag = "looking_content"
    raised, got = attempt(lambda: contrastloop.looking_content(LOOK_PANEL_A, LOOK_PANEL_B, block=fake_block))
    if raised:
        fail(f"{tag}: raised {err(got)}")
        return
    if not isinstance(got, list):
        fail(f"{tag}: got {show(got)}, expected a list of content blocks")
        return
    seq = [classify(x) for x in got]
    n_img = sum(1 for kind, _ in seq if kind == "image")
    check(n_img == LOOK_IMAGES, f"{tag}: {n_img} image blocks, expected {LOOK_IMAGES}")
    check(len(seq) == len(LOOK_WANT), f"{tag}: {len(seq)} blocks, expected {len(LOOK_WANT)}")
    for i, (g, w) in enumerate(zip(seq, LOOK_WANT)):
        if g != w:
            fail(f"{tag}: block {i} is {g!r}, expected {w!r} (first difference)")
            break
    else:
        if len(seq) > len(LOOK_WANT):
            fail(f"{tag}: extra blocks after the last expected one: {seq[len(LOOK_WANT):]!r}")


TESTS = [
    test_scaled,
    test_split,
    test_range_panel,
    test_peers,
    test_peer_photos,
    test_parse_predictions,
    test_parse_predictions_bad_entries,
    test_parse_predictions_explanation_spaces,
    test_contrast,
    test_outcome,
    test_results_text,
    test_looking_content,
]


def main():
    global contrastloop
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    try:
        import contrastloop  # noqa: F401
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        print(f"FAIL import contrastloop: {err(e)}".replace("\n", " "))
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
