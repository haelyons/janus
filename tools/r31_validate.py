"""Agreement of two answer sources on the standing questions, and of the pairwise tables they give (spec:
authors/spec_r31_validate.md).

The same photos can be answered by two sources (two runs of one answerer, or two answerers). An answer source is
{photo id: answer row}, a row one answer index or None per question of authors/standing.reply.json: from a
tools/pairtable.py answers.json (every batch not failed) or a tools/vlm_answer.py output (every record valid), several
merged in the order given, the last winning. Photos count by half (crc32('r31:' + name) % 2: 0 the calibration half,
1 the confirmation half; or all). Per question: of the photos in both sources with an answer in both, how many have
the same answer. Per unit (its looked-at photos D and checked photos T as tools/pairtable.py, kept to the photos in
both sources): pairtable's real table on D from each source's answers, rows matched by question and answer pair (the
second source's d negated when it writes the two answers in the other order), and every row of the first table checked
on T with each source's answers. Over all units: the Pearson correlation of the matched rows' d and of their
percentiles, the share of matched rows with both percentiles on the same side of 95, the share of rows with the same T
outcome, and the share held with each source. No model, no Claude calls, no GPU: layer values come from the caches of
tools/proploop.py.

Decision printed in the report, on each agreement number (a question's agreement rate; the correlation of d; the
correlation of pct; the share of matched rows on the same side of 95; the share of rows with the same T outcome):
>= 0.8 -> AT_LEAST_0.8, < 0.8 -> BELOW_0.8; NO_STATISTICS when the number is None.

usage: python -I tools/r31_validate.py --a KIND:PATH [KIND:PATH ...] --b KIND:PATH [KIND:PATH ...]
           --units FILE [FILE ...] [--half 0|1|all] --out PATH [--selftest]
       KIND is claude or vlm; relative paths are taken from the repo root
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))  # -I does not add the script dir
import contrastloop  # noqa: E402
import pairtable  # noqa: E402
import proploop  # noqa: E402

SALT = 'r31:'  # half(name) = crc32(SALT + name) % 2
HALVES = ('0', '1', 'all')
PCT_CUT = pairtable.PCT_RULE  # 95: the side of a matched row's two percentiles
MIN_PAIRS = 3  # values a correlation needs
CUT = 0.8  # the decision cut on an agreement number (a printed label; no number depends on it)


# ---------- answer sources ----------

def load_claude(path):
    """{photo id: answer row} from a pairtable answers.json (batch records with 'photos', 'answers', 'failed'): every
    photo of every batch whose failed is false"""
    out = {}
    for rec in json.loads(Path(path).read_text()).values():
        if rec['failed'] is not False:
            continue
        assert len(rec['photos']) == len(rec['answers']), (str(path), rec.get('key'))
        out.update((int(i), list(r)) for i, r in zip(rec['photos'], rec['answers']))
    return out


def load_vlm(path):
    """{photo id: answer row} from a tools/vlm_answer.py output ({id (str): {'answers', 'valid', ...}}): every record
    with valid true, id converted to int"""
    return {int(k): list(r['answers']) for k, r in json.loads(Path(path).read_text()).items() if r['valid'] is True}


LOADERS = {'claude': load_claude, 'vlm': load_vlm}


def merge(*sources):
    """one source; for a photo in several, the last source given wins"""
    out = {}
    for s in sources:
        out.update(s)
    return out


def source_spec(s):
    """'kind:path' (kind claude or vlm; the path may itself contain ':') -> (kind, path)"""
    kind, sep, p = str(s).partition(':')
    if not sep or kind not in LOADERS or not p:
        raise ValueError(f'bad source {s!r}: expected claude:PATH or vlm:PATH')
    return kind, p


def _path(p):
    """p as a Path, a relative one taken from the repo root"""
    p = Path(p)
    return p if p.is_absolute() else J / p


def load_sources(specs):
    """merge of the sources named by specs ('kind:path' each), in the order given"""
    return merge(*[LOADERS[k](_path(p)) for k, p in map(source_spec, specs)])


def check_source(src, questions, n_photos, tag):
    """stop when a photo id is not one of the pool's or a row is not, per question, None or the index of one of its
    answers"""
    ks = [len(q['answers']) for q in questions]
    for i, r in src.items():
        ok = 0 <= i < n_photos and isinstance(r, list) and len(r) == len(ks) and all(
            x is None or (isinstance(x, int) and not isinstance(x, bool) and 0 <= x < k) for x, k in zip(r, ks))
        if not ok:
            raise SystemExit(f'{tag}: photo {i} has row {r!r}: not one answer index or None per question; stopping')


def load_units(files):
    """proploop.parse_unit of every unit string of the JSON files, in order, each unit once (its first occurrence)"""
    units, seen = [], set()
    for f in files:
        for s in json.loads(_path(f).read_text()):
            u = proploop.parse_unit(s)
            if u['name'] not in seen:
                seen.add(u['name'])
                units.append(u)
    return units


# ---------- split ----------

def half(name):
    """crc32('r31:' + name) % 2: 0 the calibration half, 1 the confirmation half"""
    return zlib.crc32((SALT + name).encode()) % 2


# ---------- agreement on the questions ----------

def question_agreement(a, b, ids, nq):
    """per question q < nq, over the photos of ids in both a and b (each once): n (a non-None answer to q in both),
    agree (of those, equal answers), rate (agree / n, None when n is 0), none_a and none_b (None for q in a, in b)"""
    both = sorted({int(i) for i in ids} & set(a) & set(b))
    out = []
    for q in range(nq):
        n = agree = none_a = none_b = 0
        for i in both:
            x, y = a[i][q], b[i][q]
            none_a += x is None
            none_b += y is None
            if x is not None and y is not None:
                n += 1
                agree += x == y
        out.append({'n': n, 'agree': agree, 'rate': agree / n if n else None, 'none_a': none_a, 'none_b': none_b})
    return out


# ---------- agreement of the tables ----------

def _key(r):
    """a table row's question and its two answers, in either order"""
    return r['q'], min(r['first'], r['second']), max(r['first'], r['second'])


def match_rows(rows_a, rows_b):
    """[(row_a, row_b)]: per row of rows_a (in order) the row of rows_b (both pairtable.table_rows results) with the
    same question and the same two answers, in either order; row_b a copy with d negated when its first is row_a's
    second. Rows in only one are left out"""
    by = {_key(r): r for r in rows_b}
    out = []
    for ra in rows_a:
        rb = by.get(_key(ra))
        if rb is None:
            continue
        if rb['first'] == ra['second']:
            rb = dict(rb, d=-rb['d'])
        out.append((ra, rb))
    return out


def _centred(V, labels, ids):
    """proploop.centre of the rows ids of V (V and labels over all photos); 0 x channels when ids is empty"""
    V = np.asarray(V)
    if not ids:
        return np.zeros((0, V.shape[1]))
    x = np.asarray(ids, dtype=np.int64)
    return proploop.centre(V[x], np.asarray(labels)[x])


def unit_compare(u, V, labels, names, a, b, questions):
    """one unit (proploop.parse_unit record; V its layer's values, all photos x channels; labels over all photos): D
    and T of pairtable.unit_photos kept to the photos in both a and b (absent photos dropped); peers
    contrastloop.peers(V, labels, c); pairtable's real table on R_D = proploop.centre(V[D], labels[D]) from a's rows
    and from b's rows, matched by match_rows; every row of a's table checked by pairtable.t_check on R_T (likewise)
    with a's and with b's answers to its question (same first, second). -> {'unit', 'nD', 'nT', 'pairs': [[d_a, d_b,
    pct_a, pct_b]], 't_outcomes': [[outcome_a, outcome_b]]}"""
    c = u['chan']
    V, labels = np.asarray(V), np.asarray(labels)
    _, _, D, T = pairtable.unit_photos(u, V, labels, names)
    peers = [p for p, _ in contrastloop.peers(V, labels, c)]
    D = [i for i in D if i in a and i in b]
    T = [i for i in T if i in a and i in b]
    R_D, R_T = _centred(V, labels, D), _centred(V, labels, T)
    table_a = pairtable.table_rows(R_D, [a[i] for i in D], questions, c, peers)
    table_b = pairtable.table_rows(R_D, [b[i] for i in D], questions, c, peers)
    pairs = match_rows(table_a, table_b)
    t_outcomes = [[pairtable.t_check(R_T, [src[i][r['q']] for i in T], r['first'], r['second'], c, peers)['outcome']
                   for src in (a, b)] for r in table_a]
    return {'unit': u['name'], 'nD': len(D), 'nT': len(T),
            'pairs': [[ra['d'], rb['d'], ra['pct'], rb['pct']] for ra, rb in pairs], 't_outcomes': t_outcomes}


def pearson(x, y):
    """the Pearson correlation of x and y; None when fewer than MIN_PAIRS values or either side is constant"""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size < MIN_PAIRS or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    zx, zy = x - x.mean(), y - y.mean()
    return float((zx * zy).sum() / np.sqrt((zx ** 2).sum() * (zy ** 2).sum()))


def _share(k, n):
    return k / n if n else None


def summarise(units):
    """over unit_compare results: the Pearson correlation of d_a with d_b over all pairs of all units, and of pct_a
    with pct_b; the share of pairs with both pct >= 95 or both below 95; the share of rows with equal T outcome, with
    outcome 'held' with a, and with outcome 'held' with b (shares None when there is nothing to count)"""
    P = [p for u in units for p in u['pairs']]
    O = [o for u in units for o in u['t_outcomes']]
    return {'n_units': len(units), 'n_pairs': len(P), 'n_rows': len(O),
            'r_d': pearson([p[0] for p in P], [p[1] for p in P]),
            'r_pct': pearson([p[2] for p in P], [p[3] for p in P]),
            'same_side_95': _share(sum((p[2] >= PCT_CUT) == (p[3] >= PCT_CUT) for p in P), len(P)),
            'same_outcome': _share(sum(oa == ob for oa, ob in O), len(O)),
            'held_a': _share(sum(oa == 'held' for oa, _ in O), len(O)),
            'held_b': _share(sum(ob == 'held' for _, ob in O), len(O))}


def decide(x, cut=CUT):
    """Neutral label on an agreement number: >= cut AT_LEAST_<cut>, < cut BELOW_<cut>; None NO_STATISTICS"""
    if x is None:
        return 'NO_STATISTICS'
    return f'AT_LEAST_{cut:g}' if x >= cut else f'BELOW_{cut:g}'


# ---------- selftest ----------

def selftest():
    import math
    import tempfile

    assert PCT_CUT == 95.0 and MIN_PAIRS == 3 and CUT == 0.8 and HALVES == ('0', '1', 'all')

    # sources: the two loaders, merge (the last wins), kind:path
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cj, vj = td / 'answers.json', td / 'vlm.json'
        cj.write_text(json.dumps({
            '1': {'key': '1', 'photos': [5, 7], 'answers': [[0, None], [1, 1]], 'failed': False},
            '2': {'key': '2', 'photos': [9, 3], 'answers': [[None, None], [None, None]], 'failed': True},
            '3': {'key': '3', 'photos': [2], 'answers': [[2, 0]], 'failed': False}}))
        vj.write_text(json.dumps({
            '5': {'answers': [1, 0], 'valid': True},
            '8': {'answers': [None, None], 'valid': False},
            '9': {'answers': [0, None], 'valid': True}}))
        ca, va = load_claude(cj), load_vlm(vj)
        assert ca == {5: [0, None], 7: [1, 1], 2: [2, 0]} and va == {5: [1, 0], 9: [0, None]}, (ca, va)
        assert merge(ca, va) == {5: [1, 0], 7: [1, 1], 2: [2, 0], 9: [0, None]}
        assert merge(va, ca) == {5: [0, None], 7: [1, 1], 2: [2, 0], 9: [0, None]}
        assert merge() == {} and merge(ca) == ca and merge(ca) is not ca
        assert load_sources([f'claude:{cj}', f'vlm:{vj}']) == merge(ca, va)
        assert load_sources([f'vlm:{vj}', f'claude:{cj}']) == merge(va, ca)
    assert source_spec('claude:runs/pairtable/answers.json') == ('claude', 'runs/pairtable/answers.json')
    assert source_spec('vlm:/x/mixed4c:3/a.json') == ('vlm', '/x/mixed4c:3/a.json')
    for bad in ['runs/a.json', 'gpt:a.json', 'claude:', ':a.json', 'Claude:a.json']:
        try:
            source_spec(bad)
        except ValueError:
            continue
        raise AssertionError('accepted ' + bad)
    qs2 = [{'question': 'A?', 'answers': ['x', 'y', 'z']}, {'question': 'B?', 'answers': ['x', 'y']}]
    check_source({5: [0, None], 2: [2, 1], 9: [None, None]}, qs2, 10, 'a')
    for bad in [{10: [0, 0]}, {-1: [0, 0]}, {1: [0]}, {1: [0, 0, 0]}, {1: [3, 0]}, {1: [0, 2]}, {1: [0, -1]},
                {1: [True, 0]}, {1: ['A', 0]}, {1: [0.0, 0]}, {1: (0, 0)}]:
        try:
            check_source(bad, qs2, 10, 'a')
        except SystemExit:
            continue
        raise AssertionError(f'accepted {bad}')

    # half: crc32 of the salted name
    nm = [f'img/{i}.jpeg' for i in range(400)]
    hs = [half(x) for x in nm]
    assert set(hs) == {0, 1} and all(h == zlib.crc32(('r31:' + x).encode()) % 2 for h, x in zip(hs, nm))

    # question_agreement on a hand case: photos 1-3 in both, 4 only in b, 5 only in a, 6 in neither
    qa_a = {1: [0, 1, None], 2: [1, 1, 0], 3: [None, 0, 2], 5: [2, 2, 2]}
    qa_b = {1: [0, 0, None], 2: [1, None, 0], 3: [1, 0, 2], 4: [0, 0, 0]}
    got = question_agreement(qa_a, qa_b, [1, 2, 3, 4, 5, 6], 3)
    assert got == [{'n': 2, 'agree': 2, 'rate': 1.0, 'none_a': 1, 'none_b': 0},
                   {'n': 2, 'agree': 1, 'rate': 0.5, 'none_a': 0, 'none_b': 1},
                   {'n': 2, 'agree': 2, 'rate': 1.0, 'none_a': 1, 'none_b': 1}], got
    assert all(type(x[k]) is int for x in got for k in ('n', 'agree', 'none_a', 'none_b'))
    assert question_agreement(qa_a, qa_b, [2, 2, 5], 3) == [  # photo 2 once; photo 5 not in b
        {'n': 1, 'agree': 1, 'rate': 1.0, 'none_a': 0, 'none_b': 0},
        {'n': 0, 'agree': 0, 'rate': None, 'none_a': 0, 'none_b': 1},
        {'n': 1, 'agree': 1, 'rate': 1.0, 'none_a': 0, 'none_b': 0}]
    assert question_agreement(qa_a, qa_b, np.array([1, 3]), 1) == [
        {'n': 1, 'agree': 1, 'rate': 1.0, 'none_a': 1, 'none_b': 0}]
    empty = {'n': 0, 'agree': 0, 'rate': None, 'none_a': 0, 'none_b': 0}
    assert question_agreement(qa_a, qa_b, [], 2) == [empty, empty]
    assert question_agreement(qa_a, {}, [1, 2, 3], 1) == [empty]

    # match_rows on a hand case: same order, other order (d negated in a copy), in only one, other question
    def row(no, q, f, s, d, pct):
        return {'row': no, 'q': q, 'first': f, 'second': s, 'n_first': 5, 'n_second': 6, 'd': d, 'pct': pct,
                'peer_pct': [10.0, 20.0, 30.0], 'gap': pct - 20.0}
    ta = [row(1, 0, 0, 1, 0.5, 90.0), row(2, 0, 2, 0, 0.3, 50.0), row(3, 1, 1, 0, 0.2, 97.0),
          row(4, 2, 0, 1, 1.0, 99.0)]
    tb = [row(1, 0, 1, 0, 0.4, 80.0), row(2, 0, 1, 2, 0.6, 70.0), row(3, 1, 1, 0, 0.1, 96.0),
          row(4, 0, 2, 0, 0.7, 60.0), row(5, 3, 0, 1, 0.9, 99.5)]
    tb_before = json.loads(json.dumps(tb))
    mr = match_rows(ta, tb)
    assert mr == [(ta[0], dict(tb[0], d=-0.4)), (ta[1], tb[3]), (ta[2], tb[2])], mr
    assert tb == tb_before and mr[0][1] is not tb[0] and mr[0][1]['pct'] == 80.0 and mr[1][1]['d'] == 0.7
    assert match_rows(tb, ta) == [(tb[0], dict(ta[0], d=-0.5)), (tb[2], ta[2]), (tb[3], ta[1])]
    assert match_rows(ta, ta) == [(r, r) for r in ta]
    assert match_rows(ta, []) == [] and match_rows([], tb) == []

    # pearson against a plain-Python computation; too few values, a constant side
    rng = np.random.default_rng(5)
    xs, ys = rng.normal(size=20).tolist(), rng.normal(size=20).tolist()
    mx, my = sum(xs) / 20, sum(ys) / 20
    want = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)))
    assert abs(pearson(xs, ys) - want) < 1e-12
    assert abs(pearson([1, 2, 3], [3, 2, 1]) + 1) < 1e-12 and abs(pearson([1, 2, 3], [2, 4, 6]) - 1) < 1e-12
    assert pearson([1, 2], [1, 2]) is None and pearson([], []) is None
    assert pearson([1, 2, 3], [5, 5, 5]) is None and pearson([4, 4, 4], [1, 2, 3]) is None

    # summarise on a hand case: 3 pairs over 3 units (one empty), 4 rows
    us = [{'unit': 'mixed4c:1', 'nD': 10, 'nT': 9, 'pairs': [[1.0, 2.0, 96.0, 97.0], [2.0, 4.0, 50.0, 96.0]],
           't_outcomes': [['held', 'held'], ['not held', 'too few']]},
          {'unit': 'mixed4c:2', 'nD': 0, 'nT': 0, 'pairs': [], 't_outcomes': []},
          {'unit': 'mixed4c:3', 'nD': 5, 'nT': 5, 'pairs': [[3.0, 6.0, 10.0, 20.0]],
           't_outcomes': [['held', 'not held'], ['too few', 'too few']]}]
    sm = summarise(us)
    # pct: x 96, 50, 10 (mean 52), y 97, 96, 20 (mean 71): 3236 / sqrt(3704 * 3902)
    assert abs(sm['r_d'] - 1.0) < 1e-12 and abs(sm['r_pct'] - 3236 / math.sqrt(3704 * 3902)) < 1e-12, sm
    assert {k: v for k, v in sm.items() if k not in ('r_d', 'r_pct')} == {
        'n_units': 3, 'n_pairs': 3, 'n_rows': 4, 'same_side_95': 2 / 3, 'same_outcome': 0.5, 'held_a': 0.5,
        'held_b': 0.25}, sm
    below = float(np.nextafter(95.0, 0.0))
    s2 = summarise([{'unit': 'u', 'nD': 1, 'nT': 1, 'pairs': [[0.1, 0.2, 95.0, below], [0.3, 0.1, 95.0, 100.0]],
                     't_outcomes': []}])
    assert s2 == {'n_units': 1, 'n_pairs': 2, 'n_rows': 0, 'r_d': None, 'r_pct': None, 'same_side_95': 0.5,
                  'same_outcome': None, 'held_a': None, 'held_b': None}, s2
    s3 = summarise([{'unit': 'u', 'nD': 1, 'nT': 1,
                     'pairs': [[1.0, 0.5, 96.0, 10.0], [1.0, 0.7, 20.0, 30.0], [1.0, 0.9, 40.0, 80.0]],
                     't_outcomes': [['too few', 'too few']]}])
    assert s3['r_d'] is None and s3['r_pct'] is not None and s3['same_side_95'] == 2 / 3
    assert s3['same_outcome'] == 1.0 and s3['held_a'] == 0.0 and s3['held_b'] == 0.0
    assert summarise([]) == {'n_units': 0, 'n_pairs': 0, 'n_rows': 0, 'r_d': None, 'r_pct': None,
                             'same_side_95': None, 'same_outcome': None, 'held_a': None, 'held_b': None}
    assert summarise(json.loads(json.dumps(us))) == sm

    # decision labels
    assert decide(0.8) == 'AT_LEAST_0.8' and decide(float(np.nextafter(0.8, 0.0))) == 'BELOW_0.8'
    assert decide(1.0) == 'AT_LEAST_0.8' and decide(-1.0) == 'BELOW_0.8' and decide(None) == 'NO_STATISTICS'

    # unit_compare on synthetic values (no model, no pool): the same answers, one question's answers swapped,
    # photos missing from b, no photo in b, no T photo in b
    rng = np.random.default_rng(4)
    Vs = rng.random((600, 30)).astype(np.float32)
    labs = rng.permutation(np.repeat(np.arange(60), 10))
    nms = [f'img/{i}.jpeg' for i in range(600)]
    qs_t = [{'question': 'Is it red?', 'answers': ['yes', 'no']},
            {'question': 'Which side?', 'answers': ['left', 'middle', 'right']}]
    u = proploop.parse_unit('mixed4c:3')
    _, sid, D, T = pairtable.unit_photos(u, Vs, labs, nms)
    peers_t = [p for p, _ in contrastloop.peers(Vs, labs, 3)]
    ra = {i: [i % 2, i % 3] for i in sid.tolist()}

    def table(src, ids):
        x = np.asarray(ids, dtype=np.int64)
        return pairtable.table_rows(proploop.centre(Vs[x], labs[x]), [src[i] for i in ids], qs_t, 3, peers_t)

    def outs(src, rows, ids):
        x = np.asarray(ids, dtype=np.int64)
        R = proploop.centre(Vs[x], labs[x])
        return [pairtable.t_check(R, [src[i][r['q']] for i in ids], r['first'], r['second'], 3, peers_t)['outcome']
                for r in rows]

    tab = table(ra, D)
    assert len(tab) == 4, tab  # Q1 (0, 1); Q2 (0, 1), (0, 2), (1, 2)
    oa = outs(ra, tab, T)
    same = unit_compare(u, Vs, labs, nms, ra, dict(ra), qs_t)
    assert same == {'unit': 'mixed4c:3', 'nD': len(D), 'nT': len(T),
                    'pairs': [[r['d'], r['d'], r['pct'], r['pct']] for r in tab],
                    't_outcomes': [[o, o] for o in oa]}, same
    ssame = summarise([same])
    assert abs(ssame['r_d'] - 1.0) < 1e-12 and ssame['same_outcome'] == 1.0 and ssame['same_side_95'] == 1.0

    rb = {i: [1 - r[0], r[1]] for i, r in ra.items()}  # Q1's two answers swapped: b's row written the other way
    sw = unit_compare(u, Vs, labs, nms, ra, rb, qs_t)
    assert (sw['nD'], sw['nT']) == (len(D), len(T))
    assert sw['pairs'] == [[r['d'], -r['d'] if r['q'] == 0 else r['d'], r['pct'], r['pct']] for r in tab], sw
    assert sw['t_outcomes'] == [[o, x] for o, x in zip(oa, outs(rb, tab, T))]

    drop = set(D[::3]) | set(T[::4])
    rc = {i: r for i, r in ra.items() if i not in drop}
    D2, T2 = [i for i in D if i not in drop], [i for i in T if i not in drop]
    tab2 = table(ra, D2)
    assert len(tab2) == 4 and 0 < len(D2) < len(D) and 0 < len(T2) < len(T)
    mis = unit_compare(u, Vs, labs, nms, ra, rc, qs_t)
    assert mis == {'unit': 'mixed4c:3', 'nD': len(D2), 'nT': len(T2),
                   'pairs': [[r['d'], r['d'], r['pct'], r['pct']] for r in tab2],
                   't_outcomes': [[o, o] for o in outs(ra, tab2, T2)]}, mis
    assert unit_compare(u, Vs, labs, nms, rc, ra, qs_t) == mis

    assert unit_compare(u, Vs, labs, nms, ra, {}, qs_t) == {'unit': 'mixed4c:3', 'nD': 0, 'nT': 0, 'pairs': [],
                                                            't_outcomes': []}
    only_D = unit_compare(u, Vs, labs, nms, ra, {i: ra[i] for i in D}, qs_t)
    assert only_D['nT'] == 0 and only_D['pairs'] == same['pairs']
    assert only_D['t_outcomes'] == [['too few', 'too few']] * 4
    print('selftest ok')


# ---------- main ----------

def _f(x):
    return '-' if x is None else f'{x:.3f}'


def main():
    if '--selftest' in sys.argv[1:]:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', nargs='+', required=True, metavar='KIND:PATH',
                    help='kind claude (a pairtable answers.json) or vlm (a tools/vlm_answer.py output); several are '
                         'merged in the order given, the last winning; relative paths are taken from the repo root')
    ap.add_argument('--b', nargs='+', required=True, metavar='KIND:PATH', help='as --a')
    ap.add_argument('--units', nargs='+', required=True,
                    help='JSON files of unit strings; relative paths are taken from the repo root')
    ap.add_argument('--half', choices=HALVES, default='all', help='which photos count: half 0, half 1 or all')
    ap.add_argument('--out', required=True, help='the report JSON; a relative path is taken from the repo root')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    try:
        for s in args.a + args.b:
            source_spec(s)
    except ValueError as err:
        ap.error(str(err))

    # the pool
    questions = pairtable.load_questions()
    names = json.loads(proploop.NAMES.read_text())
    with np.load(proploop.ACTS) as z:
        labels = np.asarray(z['classes'], dtype=np.int64)
    assert len(names) == len(labels) == proploop.N_PHOTOS, (len(names), len(labels))
    units = load_units(args.units)
    layers = sorted({u['layer'] for u in units}, key=proploop.LAYERS.index)
    missing = [L for L in layers if not (proploop.OUT / f'values_{L}.npy').exists()]
    if missing:
        raise SystemExit(f'no cached values of {missing} in {proploop.OUT} (this tool runs no model); stopping')
    vals = proploop.layer_values(layers, None, np.arange(proploop.N_PHOTOS))

    # the two sources, kept to the half
    A, B = load_sources(args.a), load_sources(args.b)
    check_source(A, questions, len(names), 'a')
    check_source(B, questions, len(names), 'b')
    keep = {i for i in range(len(names)) if args.half == 'all' or half(names[i]) == int(args.half)}
    a = {i: r for i, r in A.items() if i in keep}
    b = {i: r for i, r in B.items() if i in keep}
    both = sorted(set(a) & set(b))
    qa = question_agreement(a, b, both, len(questions))
    L = ['a: ' + ' '.join(args.a), 'b: ' + ' '.join(args.b),
         f'half {args.half}: photos a {len(a)}, b {len(b)}, both {len(both)}']
    for j, (q, x) in enumerate(zip(questions, qa), 1):
        L.append(f'Q{j}. {q["question"]}  n {x["n"]}, agree {x["agree"]}, rate {_f(x["rate"])} -> '
                 f'{decide(x["rate"])}; None in a {x["none_a"]}, in b {x["none_b"]}')
    print('\n'.join(L), flush=True)

    # the tables, per unit, on the photos of the half in both sources
    res = []
    for u in units:
        r = unit_compare(u, vals[u['layer']], labels, names, a, b, questions)
        res.append(r)
        L.append(f'== {r["unit"]}  D {r["nD"]}, T {r["nT"]}; rows of a\'s table {len(r["t_outcomes"])}, matched in '
                 f'b\'s {len(r["pairs"])}')
        print(L[-1], flush=True)
    sm = summarise(res)
    dec = {'question_rate': [decide(x['rate']) for x in qa], 'r_d': decide(sm['r_d']), 'r_pct': decide(sm['r_pct']),
           'same_side_95': decide(sm['same_side_95']), 'same_outcome': decide(sm['same_outcome'])}
    tail = [f'tables: {sm["n_pairs"]} matched rows over {sm["n_units"]} units; r(d) {_f(sm["r_d"])} -> {dec["r_d"]}; '
            f'r(pct) {_f(sm["r_pct"])} -> {dec["r_pct"]}; both pct >= 95 or both below {_f(sm["same_side_95"])} -> '
            f'{dec["same_side_95"]}',
            f'T checks: {sm["n_rows"]} rows; same outcome {_f(sm["same_outcome"])} -> {dec["same_outcome"]}; held '
            f'with a {_f(sm["held_a"])}, held with b {_f(sm["held_b"])}']
    L += tail
    print('\n'.join(tail), flush=True)
    out = _path(args.out)
    proploop.save_json({'a': args.a, 'b': args.b, 'half': args.half, 'units_files': args.units,
                        'photos': {'a': len(a), 'b': len(b), 'both': len(both)},
                        'questions': [q['question'] for q in questions], 'question_agreement': qa, 'units': res,
                        'summary': sm, 'decision': dec, 'report': L}, out)
    print('report ->', out)


if __name__ == '__main__':
    main()
