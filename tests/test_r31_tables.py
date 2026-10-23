"""Black-box tests for tools/r31_tables.py (unit_new and selftest).

Written from the "## Interface" and "## Pools" sections of authors/spec_r31_tables.md only. The implementation was
not read. Expected values are computed with the functions the spec names (pairtable.unit_photos, table_rows,
rule_rows, pick, t_check, tally, NULL_D, NULL_T; proploop.centre, permute_within, parse_unit; contrastloop.split,
peers), following pairtable.run_unit's steps after its photos and peers.

No real pool, GPU or network: a synthetic reference pool (30 classes x 3 photos, 21 channels) and a synthetic new
pool (30 classes x 8 photos, 21 channels), with hand-made answer rows (some all None, some partly None). The
fixture is built so that taking the classes or peers from the new pool, or splitting the new photos by the
reference names, gives a different result (guarded by the test_fixture_* tests).
"""

import copy
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
import r31_tables  # noqa: E402


KEYS = ("unit", "layer", "chan", "m", "classes", "class_names", "peers", "D", "T", "tables", "picks", "tally",
        "n_unanswered_D", "n_unanswered_T")
COMPARED = tuple(k for k in KEYS if k != "class_names")  # class_names: own test (torchvision categories)
ARMS = ("rule_real", "rule_null", "every")
UNITS = ("mixed4c:2", "mixed5b:2")  # same channel and data, different names: different null seeds
SCENARIOS = ("base", "class_absent", "renamed")

C = 2  # the units' channel
M = 21  # channels: 20 below the unit puts it at percentile 95.24
N_CLS = 30  # more than pairtable.K_CLASSES (24), so the unit's classes are a strict subset
LABS = [100 + 7 * k for k in range(N_CLS)]  # class labels 100..303 (valid GoogLeNet category indices)
REF_PER, NEW_PER = 3, 8  # photos per class in each pool
REF_PEERS = (5, 9, 14)  # follow the unit's class profile in the reference pool only
NEW_PEERS = (3, 7, 11)  # follow the unit's class profile in the new pool only
ABSENT_LABEL = 999  # "class_absent": the new photos of the unit's top reference class are given this label

QUESTIONS = [
    {"question": "Is the main object red?", "answers": ["yes", "no"]},
    {"question": "Where is the main object?", "answers": ["left", "middle", "right"]},
    {"question": "Is there any text?", "answers": ["yes", "no"]},
]


def _questions():
    return copy.deepcopy(QUESTIONS)


# ---------------------------------------------------------------------------
# synthetic pools
#
# Reference pool: channel C = prof[class] + small noise, prof a permutation of 1..30, so the unit's 24 classes are
# those with prof >= 7. Channels 5, 9, 14 follow prof (the reference peers); the rest are uniform noise.
# New pool: channel C = (31 - prof)[class] + 3 if Q0 is answer 0, + noise; channels 3, 7, 11 follow 31 - prof (the
# new pool's own peers would be these); the rest are uniform noise. So the new pool's own top classes and peers
# differ from the reference pool's, and the unit's d on Q0 (yes vs no) is far above every other channel's.
# Answers: Q0 random; Q1 "right" given to exactly 6 answered photos of D and 2 of T (so its rows exist on D and are
# "too few" on T); Q2 random with None on about a third of photos (rows partly None). All None: the first 3 photos
# of D, the last 4 of T and 2 photos outside the unit's classes.
# ---------------------------------------------------------------------------

def _build(scenario="base"):
    rng = np.random.default_rng(101)
    prof = rng.permutation(N_CLS) + 1.0

    # reference pool
    r_idx = rng.permutation(np.repeat(np.arange(N_CLS), REF_PER))
    n_ref = len(r_idx)
    V_ref = rng.random((n_ref, M))
    V_ref[:, C] = prof[r_idx] + 0.1 * rng.standard_normal(n_ref)
    for j, s in zip(REF_PEERS, (1.0, 2.0, 3.0)):
        V_ref[:, j] = prof[r_idx] + s * rng.standard_normal(n_ref)
    ref = {"V": V_ref.astype(np.float32),
           "labels": np.asarray([LABS[k] for k in r_idx], dtype=np.int64),
           "names": [f"ref/{i:05d}.jpeg" for i in range(n_ref)]}

    # new pool
    n_idx = rng.permutation(np.repeat(np.arange(N_CLS), NEW_PER))
    n = len(n_idx)
    names = [f"newpool/{i:05d}.png" for i in range(n)]
    sel = [i for i in range(n) if prof[n_idx[i]] >= 7]
    outside = [i for i in range(n) if prof[n_idx[i]] < 7]
    D = [i for i in sel if zlib.crc32(names[i].encode()) % 2 == 0]
    T = [i for i in sel if zlib.crc32(names[i].encode()) % 2 == 1]
    blank = set(D[:3]) | set(T[-4:]) | set(outside[:2])
    q0 = rng.integers(0, 2, size=n)
    q1 = rng.integers(0, 3, size=n)
    q2 = rng.integers(0, 3, size=n)  # 2 -> None (could not tell)
    for half, n_right in ((D, 6), (T, 2)):
        answered = [i for i in half if i not in blank]
        for t, i in enumerate(answered):
            q1[i] = 2 if t < n_right else t % 2
    row_of = {i: ([None, None, None] if i in blank
                  else [int(q0[i]), int(q1[i]), None if int(q2[i]) == 2 else int(q2[i])])
              for i in range(n)}
    on0 = np.array([row_of[i][0] == 0 for i in range(n)], dtype=np.float64)
    newprof = (N_CLS + 1) - prof
    V_new = rng.random((n, M))
    V_new[:, C] = newprof[n_idx] + 3.0 * on0 + 0.5 * rng.standard_normal(n)
    for j in NEW_PEERS:
        V_new[:, j] = newprof[n_idx] + rng.standard_normal(n)
    labels = np.asarray([LABS[k] for k in n_idx], dtype=np.int64)
    if scenario == "class_absent":
        labels = np.where(labels == LABS[int(np.argmax(prof))], ABSENT_LABEL, labels).astype(np.int64)
    elif scenario == "renamed":
        names = [f"elsewhere/photo_{i}.jpg" for i in range(n)]
    else:
        assert scenario == "base", scenario
    new = {"V": V_new.astype(np.float32), "labels": labels, "names": names}
    info = {"prof": prof, "sel": sel, "outside": outside, "D": D, "T": T, "blank": sorted(blank)}
    return ref, new, row_of, info


# ---------------------------------------------------------------------------
# the same steps by hand with the pairtable functions
# ---------------------------------------------------------------------------

def _reference(u, ref, new, row_of, questions):
    name, layer, c = u["name"], u["layer"], u["chan"]
    classes = pairtable.unit_photos(u, ref["V"], ref["labels"], ref["names"])[0]
    pr = contrastloop.peers(np.asarray(ref["V"]), np.asarray(ref["labels"]), c)
    peers = [p for p, _ in pr]

    V, labels = np.asarray(new["V"]), np.asarray(new["labels"])
    ids = [i for i in range(len(labels)) if int(labels[i]) in classes]
    D, T = contrastloop.split(ids, new["names"])
    seed = zlib.crc32(name.encode())

    D_ids, T_ids = np.asarray(D, dtype=np.int64), np.asarray(T, dtype=np.int64)
    rows_D = [row_of[i] for i in D]
    rows_T = [row_of[i] for i in T]
    null_D = proploop.permute_within(rows_D, labels[D_ids], classes,
                                     np.random.default_rng(seed + pairtable.NULL_D))
    null_T = proploop.permute_within(rows_T, labels[T_ids], classes,
                                     np.random.default_rng(seed + pairtable.NULL_T))
    R_D = proploop.centre(V[D_ids], labels[D_ids])
    R_T = proploop.centre(V[T_ids], labels[T_ids])

    tables = {"real": pairtable.table_rows(R_D, rows_D, questions, c, peers),
              "null": pairtable.table_rows(R_D, null_D, questions, c, peers)}
    by_no = {t: {r["row"]: r for r in tables[t]} for t in tables}
    chosen = {"rule_real": ("real", pairtable.rule_rows(tables["real"])),
              "rule_null": ("null", pairtable.rule_rows(tables["null"])),
              "every": ("real", [r["row"] for r in tables["real"]])}
    picks = {arm: [pairtable.pick(by_no[t][no], questions, R_T, rows_T, null_T, c, peers) for no in nos]
             for arm, (t, nos) in chosen.items()}

    want = {"unit": name, "layer": layer, "chan": c, "m": int(V.shape[1]), "classes": classes,
            "peers": [{"chan": p, "unit": f"{layer}:{p}", "corr": r} for p, r in pr],
            "D": D, "T": T, "tables": tables, "picks": picks, "tally": pairtable.tally(picks),
            "n_unanswered_D": sum(all(x is None for x in row_of[i]) for i in D),
            "n_unanswered_T": sum(all(x is None for x in row_of[i]) for i in T)}
    aux = {"peers": peers, "R_D": R_D, "R_T": R_T, "rows_D": rows_D, "rows_T": rows_T, "null_D": null_D,
           "null_T": null_T, "seed": seed, "labels_D": labels[D_ids], "new_classes":
           pairtable.unit_photos(u, new["V"], new["labels"], new["names"])[0]}
    return want, aux


def _want(unit=UNITS[0], scenario="base"):
    u = proploop.parse_unit(unit)
    ref, new, row_of, info = _build(scenario)
    want, aux = _reference(u, ref, new, row_of, _questions())
    return want, aux, row_of, info


def _run(unit=UNITS[0], scenario="base", alter=None):
    u = proploop.parse_unit(unit)
    ref, new, row_of, info = _build(scenario)
    if alter is not None:
        alter(ref, new, row_of, info)
    return r31_tables.unit_new(u, ref, new, row_of, _questions())


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

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


def _cmp(got, want, path):
    if isinstance(want, dict):
        assert isinstance(got, dict), f"{path}: {got!r} is not a dict"
        assert set(got) == set(want), f"{path}: keys {sorted(map(str, got))} != {sorted(map(str, want))}"
        for k in want:
            _cmp(got[k], want[k], f"{path}[{k!r}]")
    elif isinstance(want, list):
        assert isinstance(got, list), f"{path}: {got!r} is not a list"
        assert len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            _cmp(g, w, f"{path}[{i}]")
    elif want is None:
        assert got is None, f"{path}: {got!r} != None"
    elif isinstance(want, bool):
        assert isinstance(got, bool) and got == want, f"{path}: {got!r} != {want!r}"
    elif isinstance(want, float):
        assert isinstance(got, (int, float)) and not isinstance(got, bool), f"{path}: {got!r} != {want!r}"
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12), f"{path}: {got!r} != {want!r}"
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


def _same(got, want, path):
    _cmp(_plain(got), _plain(want), path)


# ---------------------------------------------------------------------------
# fixture guards (reference functions only): the cases below can tell right from wrong
# ---------------------------------------------------------------------------

def test_fixture_reference_and_new_pools_disagree():
    ref, new, row_of, info = _build()
    u = proploop.parse_unit(UNITS[0])
    rc = pairtable.unit_photos(u, ref["V"], ref["labels"], ref["names"])[0]
    nc = pairtable.unit_photos(u, new["V"], new["labels"], new["names"])[0]
    assert len(rc) == pairtable.K_CLASSES == 24
    assert sorted(rc) == sorted(LABS[k] for k in range(N_CLS) if info["prof"][k] >= 7)
    assert set(rc) != set(nc)
    assert {p for p, _ in contrastloop.peers(ref["V"], ref["labels"], C)} == set(REF_PEERS)
    assert {p for p, _ in contrastloop.peers(new["V"], new["labels"], C)} == set(NEW_PEERS)
    assert info["outside"] and max(info["sel"]) >= len(ref["names"])
    ids = [i for i in range(len(new["labels"])) if int(new["labels"][i]) in rc]
    assert ids == info["sel"]
    assert contrastloop.split(ids, new["names"]) == (info["D"], info["T"])


def test_fixture_outcomes_cover_rule_picks_held_too_few_and_blank_rows():
    want, aux, row_of, info = _want()
    assert len(want["tables"]["real"]) == 5 and len(want["tables"]["null"]) == 5
    assert want["picks"]["rule_real"]
    every = want["tally"]["every"]["real"]
    assert every["held"] >= 1 and every["too few"] == 2
    assert want["n_unanswered_D"] == 3 and want["n_unanswered_T"] == 4
    partly = [i for i in want["D"] + want["T"]
              if any(x is None for x in row_of[i]) and not all(x is None for x in row_of[i])]
    assert partly


def test_fixture_null_tables_depend_on_seed_and_name():
    want_a, aux_a, _, _ = _want(UNITS[0])
    want_b, _, _, _ = _want(UNITS[1])
    d_null = [r["d"] for r in want_a["tables"]["null"]]
    assert d_null != [r["d"] for r in want_b["tables"]["null"]]
    _same(want_b["tables"]["real"], want_a["tables"]["real"], "real tables, two names")
    # other offsets would give other null tables and other T_null checks
    for off in (pairtable.NULL_T, pairtable.NULL_D + 1):
        other = proploop.permute_within(aux_a["rows_D"], aux_a["labels_D"], want_a["classes"],
                                        np.random.default_rng(aux_a["seed"] + off))
        rows = pairtable.table_rows(aux_a["R_D"], other, _questions(), C, aux_a["peers"])
        assert [r["d"] for r in rows] != d_null
    assert pairtable.NULL_D != pairtable.NULL_T
    want_c, _, _, _ = _want(UNITS[0])
    t_null = [p["T_null"]["d"] for p in want_a["picks"]["every"]]
    assert t_null == [p["T_null"]["d"] for p in want_c["picks"]["every"]]
    assert t_null != [p["T_null"]["d"] for p in want_b["picks"]["every"]]


# ---------------------------------------------------------------------------
# unit_new: returned keys and header fields
# ---------------------------------------------------------------------------

def test_returns_a_dict_with_every_listed_key():
    st = _run()
    assert isinstance(st, dict)
    missing = [k for k in KEYS if k not in st]
    assert not missing, missing


@pytest.mark.parametrize("unit", UNITS)
def test_unit_layer_chan_m(unit):
    st = _run(unit)
    u = proploop.parse_unit(unit)
    assert st["unit"] == unit
    assert st["layer"] == u["layer"]
    assert int(st["chan"]) == C
    assert int(st["m"]) == M


# ---------------------------------------------------------------------------
# classes and peers from the reference pool
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_classes_from_reference_pool(scenario):
    want, aux, _, _ = _want(scenario=scenario)
    st = _run(scenario=scenario)
    _same(st["classes"], want["classes"], "classes")
    assert set(_plain(st["classes"])) != set(aux["new_classes"])


@pytest.mark.parametrize("unit", UNITS)
def test_peers_from_reference_pool_as_run_unit_records(unit):
    want, _, _, _ = _want(unit)
    st = _run(unit)
    _same(st["peers"], want["peers"], "peers")
    layer = proploop.parse_unit(unit)["layer"]
    assert {int(p["chan"]) for p in st["peers"]} == set(REF_PEERS)
    assert all(p["unit"] == f"{layer}:{int(p['chan'])}" for p in st["peers"])


def test_class_names_from_googlenet_categories():
    from torchvision.models import GoogLeNet_Weights
    cats = list(GoogLeNet_Weights.IMAGENET1K_V1.meta["categories"])
    want, _, _, _ = _want()
    st = _run()
    assert list(st["class_names"]) == [cats[x] for x in want["classes"]]


# ---------------------------------------------------------------------------
# the new photos: by the reference classes, in id order, split on the new names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_new_photos_and_split(scenario):
    want, _, _, _ = _want(scenario=scenario)
    st = _run(scenario=scenario)
    _same(st["D"], want["D"], "D")
    _same(st["T"], want["T"], "T")


def test_new_photos_are_every_photo_of_the_classes_in_id_order():
    ref, new, _, info = _build()
    st = _run()
    D, T = _plain(st["D"]), _plain(st["T"])
    classes = set(_plain(st["classes"]))
    assert D == sorted(D) and T == sorted(T)
    assert not set(D) & set(T)
    assert sorted(D + T) == [i for i, x in enumerate(new["labels"].tolist()) if x in classes]
    assert sorted(D + T) == info["sel"]
    assert not set(D + T) & set(info["outside"])
    assert all(zlib.crc32(new["names"][i].encode()) % 2 == 0 for i in D)
    assert all(zlib.crc32(new["names"][i].encode()) % 2 == 1 for i in T)
    assert (D, T) == (info["D"], info["T"])


def test_photos_outside_the_classes_do_not_matter():
    st0 = _run()

    def alter(ref, new, row_of, info):
        rng = np.random.default_rng(303)
        for i in info["outside"]:
            new["V"][i] = (rng.random(M) * 1000.0 - 500.0).astype(np.float32)
            row_of[i] = [1, 2, 0]

    st1 = _run(alter=alter)
    for k in KEYS:
        _same(st1[k], st0[k], k)


# ---------------------------------------------------------------------------
# tables, arms, checks on T, tally
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("unit", UNITS)
def test_real_and_null_tables(unit, scenario):
    want, _, _, _ = _want(unit, scenario)
    st = _run(unit, scenario)
    assert set(st["tables"]) == {"real", "null"}
    _same(st["tables"]["real"], want["tables"]["real"], "tables['real']")
    _same(st["tables"]["null"], want["tables"]["null"], "tables['null']")


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("unit", UNITS)
def test_picks_of_the_three_arms(unit, scenario):
    want, _, _, _ = _want(unit, scenario)
    st = _run(unit, scenario)
    assert set(st["picks"]) == set(ARMS)
    for arm in ARMS:
        _same(st["picks"][arm], want["picks"][arm], f"picks[{arm!r}]")


def test_arm_rows_come_from_their_tables():
    st = _plain(_run())
    tables = st["tables"]
    by_no = {t: {r["row"]: r for r in tables[t]} for t in ("real", "null")}
    for arm, t in (("rule_real", "real"), ("rule_null", "null")):
        nos = pairtable.rule_rows(tables[t])
        assert [p["row"] for p in st["picks"][arm]] == nos
        _same([p["D"] for p in st["picks"][arm]], [by_no[t][n] for n in nos], f"{arm} D records")
    _same([p["D"] for p in st["picks"]["every"]], tables["real"], "every D records")
    assert [p["row"] for p in st["picks"]["every"]] == [r["row"] for r in tables["real"]]
    assert st["picks"]["rule_real"]


def test_checks_on_T_with_real_and_null_rows():
    want, aux, _, _ = _want()
    st = _run()
    qs = _questions()
    for arm in ARMS:
        ps = st["picks"][arm]
        assert len(ps) == len(want["picks"][arm])
        for j, p in enumerate(ps):
            q, a, b = int(p["q"]), int(p["first"]), int(p["second"])
            assert p["question"] == qs[q]["question"]
            assert (p["first_answer"], p["second_answer"]) == (qs[q]["answers"][a], qs[q]["answers"][b])
            t_real = pairtable.t_check(aux["R_T"], [r[q] for r in aux["rows_T"]], a, b, C, aux["peers"])
            t_null = pairtable.t_check(aux["R_T"], [r[q] for r in aux["null_T"]], a, b, C, aux["peers"])
            _same(p["T"], t_real, f"picks[{arm!r}][{j}]['T']")
            _same(p["T_null"], t_null, f"picks[{arm!r}][{j}]['T_null']")


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("unit", UNITS)
def test_tally_over_the_three_arms(unit, scenario):
    want, _, _, _ = _want(unit, scenario)
    st = _run(unit, scenario)
    assert set(st["tally"]) == set(ARMS)
    _same(st["tally"], want["tally"], "tally")
    _same(st["tally"], pairtable.tally(_plain(st["picks"])), "tally of the returned picks")


def test_null_depends_on_unit_name_real_does_not():
    st_a, st_b = _plain(_run(UNITS[0])), _plain(_run(UNITS[1]))
    _same(st_a["tables"]["real"], st_b["tables"]["real"], "real tables, two names")
    _same([p["T"] for p in st_a["picks"]["every"]], [p["T"] for p in st_b["picks"]["every"]], "every T, two names")
    assert [r["d"] for r in st_a["tables"]["null"]] != [r["d"] for r in st_b["tables"]["null"]]


# ---------------------------------------------------------------------------
# unanswered photos
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_n_unanswered_counts_all_none_rows(scenario):
    want, _, row_of, _ = _want(scenario=scenario)
    st = _run(scenario=scenario)
    assert int(st["n_unanswered_D"]) == want["n_unanswered_D"]
    assert int(st["n_unanswered_T"]) == want["n_unanswered_T"]
    D, T = _plain(st["D"]), _plain(st["T"])
    assert int(st["n_unanswered_D"]) == sum(all(x is None for x in row_of[i]) for i in D)
    assert int(st["n_unanswered_T"]) == sum(all(x is None for x in row_of[i]) for i in T)


def test_n_unanswered_base_case_values():
    # base: the first 3 photos of D and the last 4 of T are all None; rows only partly None are not counted
    st = _run()
    assert int(st["n_unanswered_D"]) == 3
    assert int(st["n_unanswered_T"]) == 4


# ---------------------------------------------------------------------------
# the whole stats against the hand computation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("unit", UNITS)
def test_stats_match_the_pairtable_steps(unit, scenario):
    want, _, _, _ = _want(unit, scenario)
    st = _run(unit, scenario)
    for k in COMPARED:
        _same(st[k], want[k], k)


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def test_selftest_prints_ok(capsys):
    r31_tables.selftest()
    assert "selftest ok" in capsys.readouterr().out


def test_main_selftest_flag(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["r31_tables.py", "--selftest"])
    try:
        r31_tables.main()
    except SystemExit as e:
        assert e.code in (None, 0), e.code
    assert "selftest ok" in capsys.readouterr().out
