"""A fixed question set answered blind, every answer pair compared, rows read by an agent (spec:
authors/spec_pairtable.md).

Per unit (one channel of mixed3a-mixed5b): the photos of the 24 classes with the highest class mean of the unit's
spatial max are split by crc32(photo name) % 2 into looked-at photos D (0) and checked photos T (1), as
tools/contrastloop.py. An agent that never sees any activity answers the 8 questions of authors/standing.reply.json,
the same for every unit, about every photo of every unit's set, 12 per call, in one pass shared by all units. On D,
every two answers of each question that each have at least 5 photos are compared: d (as contrastloop.contrast, on the
values scaled and centred within class) for every channel of the layer, written with the answer on which the unit is
more active first, with the unit's percentile among the layer's channels, its 3 similar units' percentiles and gap (the
unit's percentile minus their mean). An agent reads that table (no photos), says what the unit responds to and picks up
to 4 rows, 3 samples; the same on the table from answers permuted within class (null). A fixed rule also picks rows
from each table (percentile >= 95, highest gap first, up to 4). The picked rows, and every row of the real table, are
checked on T, with the real answers and with answers permuted within class. Outputs: runs/pairtable/.

Decision printed in the report, per checked row: the unit's d on T, with the same answer first, at a percentile >= 95
among its layer's channels and above 0 -> held, else not held; too few when either answer has fewer than 5 photos of
T; distinctive when held and the lowest of its 3 similar units' percentiles is below 50.

usage: python -I tools/pairtable.py [--units-file authors/round29_units.json] [--selftest]
"""
import argparse
import gc
import json
import sys
import zlib
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))  # -I does not add the script dir
import bundlemax  # noqa: E402
import contrastloop  # noqa: E402
import proploop  # noqa: E402
import refineloop  # noqa: E402

OUT = J / 'runs/pairtable'
UNITS_FILE = 'authors/round29_units.json'
STANDING = J / 'authors/standing.reply.json'  # its list "questions", used as given
N_QUESTIONS = 8
BATCH_KEY = 'standing'  # proploop.answer_batches('standing', P): one pass shared by all units
K_CLASSES = contrastloop.K_CLASSES  # 24
MIN_N = contrastloop.MIN_N  # photos each compared answer needs
MAX_ROWS = 4  # rows kept from a reading, and picked by the rule
PCT_RULE = contrastloop.PCT_HELD  # the rule's cut on a row's percentile
PCT_DISTINCT = refineloop.PCT_DISTINCT  # distinctive: held, and the lowest similar-unit percentile below this
SAMPLES = contrastloop.SAMPLES
TABLES = proploop.TABLES
READINGS = [f'{t}_s{s}' for t in TABLES for s in SAMPLES]
ARMS = READINGS + [f'rule_{t}' for t in TABLES] + ['every']
OUTCOMES = contrastloop.OUTCOMES
NULL_D, NULL_T = 3, 0  # added to crc32(unit) to seed the within-class permutations of the answers on D and on T
WORKERS = 1  # calls in flight: main sets proploop.WORKERS to this before any call
SNIPPET = 200  # characters of a reading's explanation in the report
_text = contrastloop._text

TABLE_HEAD = (
    'Below is what is known about one unit (a single neuron) of an image classification network. Someone who never '
    "saw the unit's activity answered the questions below about each of {n} photos of the {K} photo classes on which "
    'the unit is most active on average ({classes}). For each question, every two of its answers that were each given '
    'to at least 5 photos are compared. d is how much more active the unit is on the photos given the first answer '
    'than on the photos given the second, after removing the differences between photo classes, in standard '
    'deviations; each pair is written with the answer on which the unit is more active first. The percentile places d '
    "among all {m} units of the unit's layer on the same photos and answers (95 or above is unusual for this layer). "
    'The same percentile is given for the 3 units of the layer whose average activity over photo classes is most like '
    "this unit's (similar units 1 to 3): a row on which a similar unit is also at 95 or above describes something "
    'they share, not what is particular to this unit.')
TABLE_TASK = (
    'The rows will be checked on new photos of the same classes, none of them among those above, answered in the '
    "same way: a row holds when this unit's d, with the same answer first, is again at the 95th percentile of its "
    'layer or above. Say in a few sentences what this unit responds to, and choose up to 4 rows that you expect to '
    'hold for this unit on the new photos and that describe what is particular to this unit rather than what it '
    'shares with the similar units. Reply with only JSON: {"explanation": "...", "rows": [n, ...]}')


# ---------- questions and photos ----------

def load_questions(path=STANDING):
    """the list "questions" of the standing reply, in order and as given (8 entries, each {'question', 'answers'})"""
    qs = json.loads(Path(path).read_text())['questions']
    assert len(qs) == N_QUESTIONS, len(qs)
    for q in qs:
        assert isinstance(q['question'], str) and proploop.MIN_ANS <= len(q['answers']) <= proploop.MAX_ANS, q
    return qs


def unit_photos(u, V, labels, names):
    """(classes, set ids, D, T) of a unit: proploop.photo_set on its channel of V (all photos x channels) with k=24,
    then contrastloop.split"""
    classes, set_ids = proploop.photo_set(np.asarray(V)[:, u['chan']], labels, k=K_CLASSES)
    D, T = contrastloop.split(set_ids.tolist(), names)
    return classes, set_ids, D, T


# ---------- rows ----------

def pairs(k):
    """every (a, b) with 0 <= a < b < k, by a, then b"""
    return [(a, b) for a in range(k) for b in range(a + 1, k)]


def table_rows(R, ans_rows, questions, c, peers, min_n=MIN_N):
    """the row records of a photo set: R its centred values (n x m), ans_rows one answer row per photo (int or None
    per question). Per question in order and per pair (a, b) of its answers in pairs order: contrastloop.contrast,
    skipped when None; written with the answer on which column c is more active first (d negated and the counts
    swapped when d[c] < 0); pct and peer_pct proploop.percentile of that d at c and at each peer channel; gap = pct
    minus the mean of peer_pct (None when there are no peers). Rows numbered 1, 2, ..."""
    R = np.asarray(R, dtype=np.float64)
    assert R.ndim == 2 and len(ans_rows) == R.shape[0], (R.shape, len(ans_rows))
    out = []
    for q, Q in enumerate(questions):
        ans = [r[q] for r in ans_rows]
        for a, b in pairs(len(Q['answers'])):
            got = contrastloop.contrast(R, ans, a, b, min_n)
            if got is None:
                continue
            n_a, n_b, d = got
            first, second, n_first, n_second = a, b, n_a, n_b
            if d[c] < 0:
                first, second, n_first, n_second, d = b, a, n_b, n_a, -d
            pct = proploop.percentile(d, c)
            peer_pct = [proploop.percentile(d, int(p)) for p in peers]
            gap = pct - sum(peer_pct) / len(peer_pct) if peer_pct else None
            out.append({'row': len(out) + 1, 'q': q, 'first': first, 'second': second, 'n_first': n_first,
                        'n_second': n_second, 'd': float(d[c]), 'pct': pct, 'peer_pct': peer_pct, 'gap': gap})
    return out


def rule_rows(rows, k=MAX_ROWS):
    """the row numbers of the rows with pct >= 95, by gap highest first (a gap of None last), ties by lower row
    number; the first k"""
    on = [r for r in rows if r['pct'] >= PCT_RULE]
    on.sort(key=lambda r: (1, 0.0, r['row']) if r['gap'] is None else (0, -r['gap'], r['row']))
    return [r['row'] for r in on[:k]]


def t_check(R, ans, first, second, c, peers):
    """one row checked on R (n x m, centred) with ans (one answer per photo): n_first and n_second the photos given
    first and second; when contrastloop.contrast is None, d, pct and peer_pct None, outcome 'too few', distinctive
    false; else column c's d, its percentile and each peer channel's, outcome contrastloop.outcome, distinctive =
    held and the lowest peer percentile below 50"""
    codes = proploop._codes(ans)
    n_first, n_second = int(np.count_nonzero(codes == first)), int(np.count_nonzero(codes == second))
    got = contrastloop.contrast(R, ans, first, second)
    if got is None:
        return {'n_first': n_first, 'n_second': n_second, 'd': None, 'pct': None, 'peer_pct': None,
                'outcome': 'too few', 'distinctive': False}
    d = got[2]
    pct = proploop.percentile(d, c)
    peer_pct = [proploop.percentile(d, int(p)) for p in peers]
    oc = contrastloop.outcome(float(d[c]), pct)
    return {'n_first': n_first, 'n_second': n_second, 'd': float(d[c]), 'pct': pct, 'peer_pct': peer_pct,
            'outcome': oc, 'distinctive': oc == 'held' and bool(peer_pct) and min(peer_pct) < PCT_DISTINCT}


def pick(r, questions, R, rows, null_rows, c, peers):
    """a picked row: its record r (from its table), its question and answers, and its checks on R with the real and
    the null answer rows"""
    q, a, b = r['q'], r['first'], r['second']
    Q = questions[q]
    return {'row': r['row'], 'q': q, 'first': a, 'second': b, 'question': Q['question'],
            'first_answer': Q['answers'][a], 'second_answer': Q['answers'][b], 'D': r,
            'T': t_check(R, [x[q] for x in rows], a, b, c, peers),
            'T_null': t_check(R, [x[q] for x in null_rows], a, b, c, peers)}


def tally(picks):
    """{arm: {'real': {outcome: picks with that T outcome}, 'null': {outcome: picks with that T_null outcome}}}"""
    return {arm: {t: {o: sum(p[k]['outcome'] == o for p in ps) for o in OUTCOMES}
                  for t, k in zip(TABLES, ('T', 'T_null'))} for arm, ps in picks.items()}


# ---------- table text and replies ----------

def row_line(row, questions):
    """one line of the table text"""
    q = questions[row['q']]
    a, w = q['answers'], proploop._whole
    return (f'Row {w(row["row"])}. {q["question"]} "{a[row["first"]]}" ({w(row["n_first"])} photos) more than '
            f'"{a[row["second"]]}" ({w(row["n_second"])} photos): d {float(row["d"]):+.2f}, percentile '
            f'{w(row["pct"])}; similar units ' + ', '.join(str(w(p)) for p in row['peer_pct']) + '.')


def table_text(n, K, class_names, m, questions, rows):
    """the table text: the head with its fields filled, a blank line, one line per row, a blank line, the task"""
    w = proploop._whole
    head = proploop.fill(TABLE_HEAD, n=w(n), K=w(K), classes=', '.join(class_names), m=w(m))
    return '\n\n'.join([head, '\n'.join(row_line(r, questions) for r in rows), TABLE_TASK])


def parse_rows(text, n_rows):
    """{'explanation' (stripped), 'rows'} of the last JSON object with 'rows': the explanation a string non-empty after
    strip; rows a list, of which the entries that are ints (not bool) from 1 to n_rows are kept in order, each at its
    first occurrence, the first MAX_ROWS of them. None when the explanation is not valid, rows is not a list, no row
    is kept or no object is found"""
    o = bundlemax.last_obj(text, ('rows',))
    if o is None:
        return None
    e, rs = o.get('explanation'), o['rows']
    if not isinstance(e, str) or not e.strip() or not isinstance(rs, list):
        return None
    kept = []
    for x in rs:
        if isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= n_rows and x not in kept:
            kept.append(x)
    kept = kept[:MAX_ROWS]
    return {'explanation': e.strip(), 'rows': kept} if kept else None


# ---------- calls ----------

def _rows_parser(n_rows):
    def parse(text):
        return parse_rows(text, n_rows)
    return parse


def _reading_fields(p):
    """the cached fields of a reading: its explanation and rows, both None when it is not valid"""
    return {'explanation': None if p is None else p['explanation'], 'rows': None if p is None else p['rows']}


def answer_pass(P, questions, call=None, out=OUT, block=None):
    """the answers to the questions for every photo of P (sorted photo ids): batches proploop.answer_batches(
    'standing', P), content refineloop.d_content, through proploop.cached_calls at out/answers.json (checked first by
    contrastloop.check_answer_cache); a photo of a failed batch has a row of None. Stops, naming them, when batches are
    not cached after the pass (call errors). -> ({photo id: answer row}, batches, failed batch keys)"""
    nq = len(questions)
    batches = proploop.answer_batches(BATCH_KEY, P)
    keys = [str(b) for b in range(1, len(batches) + 1)]
    apath = out / 'answers.json'
    contrastloop.check_answer_cache(apath, batches)
    jobs = [(key, refineloop.d_content(batch, questions, block), proploop._answer_parser(len(batch), questions),
             proploop._answer_fields(batch, nq)) for key, batch in zip(keys, batches)]
    ac = proploop.cached_calls(BATCH_KEY, apath, jobs, call)
    del jobs
    missing = [k for k in keys if k not in ac]
    if missing:
        raise SystemExit(f'{apath}: answer batches {missing} not cached (call errors); a rerun redoes only those; '
                         'stopping')
    row_of = {}
    for key, batch in zip(keys, batches):
        if ac[key]['photos'] != batch or len(ac[key]['answers']) != len(batch):
            raise SystemExit(f'{apath}: batch {key} is for other photos; stopping')
        row_of.update(zip(batch, ac[key]['answers']))
    return row_of, batches, [k for k in keys if ac[k]['failed']]


# ---------- one unit, report ----------

def _row_text(r, questions):
    a = questions[r['q']]['answers']
    return (f'Row {r["row"]}. Q{r["q"] + 1} "{a[r["first"]]}" ({r["n_first"]}) more than "{a[r["second"]]}" '
            f'({r["n_second"]}): d {r["d"]:+.2f}, pct {r["pct"]:.1f}; similar units '
            + ', '.join(f'{p:.1f}' for p in r['peer_pct'])
            + '; gap ' + ('-' if r['gap'] is None else f'{r["gap"]:+.1f}'))


def _check_text(x):
    if x['d'] is None:
        return f'{x["n_first"]} and {x["n_second"]} photos -> {x["outcome"]}'
    return f'{x["n_first"]} and {x["n_second"]} photos, d {x["d"]:+.2f}, pct {x["pct"]:.1f} -> {x["outcome"]}'


def _pick_text(p):
    t = p['T']
    return (f'Row {p["row"]}. Q{p["q"] + 1} "{p["first_answer"]}" more than "{p["second_answer"]}": D pct '
            f'{p["D"]["pct"]:.1f}; T {_check_text(t)}, distinctive {"yes" if t["distinctive"] else "no"}; '
            f'T_null {p["T_null"]["outcome"]}')


def section(name, layer, m, nD, nT, classes, cats, peer_recs, questions, tables, readings, why, picks, tal):
    L = [f'== {name}  layer {layer} ({m} channels), {nD} photos looked at (D), {nT} photos checked (T); table rows: '
         + ', '.join(f'{t} {len(tables[t])}' for t in TABLES),
         'classes: ' + ', '.join(f'{cats[x]} ({x})' for x in classes),
         'peers: ' + ', '.join(f'{p["unit"]} (r {contrastloop._corr(p["corr"])})' for p in peer_recs),
         'real table:']
    L += ['   ' + _row_text(r, questions) for r in tables['real']]
    for key in READINGS:
        rec = readings[key]
        L.append(f'{key}  rows {rec["rows"]}  ' + ' '.join(rec['explanation'][:SNIPPET].split()) if rec['valid']
                 else f'{key}  no reading ({why[key]})')
    for arm in ARMS:
        L.append(f'{arm}: {len(picks[arm])} rows')
        L += ['   ' + _pick_text(p) for p in picks[arm]]
    for arm in ARMS:
        L.append(f'tally {arm}: ' + '; '.join(f'{t} ' + ', '.join(f'{o} {tal[arm][t][o]}' for o in OUTCOMES)
                                              for t in TABLES) + f' (of {len(picks[arm])} rows)')
    return '\n'.join(L)


def run_unit(u, V, labels, names, cats, row_of, questions, call=None, out=OUT):
    """one unit (u from proploop.parse_unit; V: its layer's values, photos x channels; row_of: the answer row of every
    photo, from answer_pass): photos and peers, the real and null tables on D, readings, picks, checks on T,
    stats.json. -> (report section, stats)"""
    name, layer, c = u['name'], u['layer'], u['chan']
    udir = out / name
    V, labels = np.asarray(V), np.asarray(labels)
    m = V.shape[1]
    seed = zlib.crc32(name.encode())

    # photos and peers
    classes, _, D, T = unit_photos(u, V, labels, names)
    cnames = [cats[x] for x in classes]
    pr = contrastloop.peers(V, labels, c)
    peers = [p for p, _ in pr]
    peer_recs = [{'chan': p, 'unit': f'{layer}:{p}', 'corr': r} for p, r in pr]

    # tables on D, real and null
    D_ids = np.asarray(D, dtype=np.int64)
    rows_D = [row_of[i] for i in D]
    null_D = proploop.permute_within(rows_D, labels[D_ids], classes, np.random.default_rng(seed + NULL_D))
    R_D = proploop.centre(V[D_ids], labels[D_ids])
    tables = {t: table_rows(R_D, rr, questions, c, peers) for t, rr in zip(TABLES, (rows_D, null_D))}

    # readings: three fresh samples of each table that has rows
    jobs = [(f'{t}_s{s}', [_text(table_text(len(D), K_CLASSES, cnames, m, questions, tables[t]))],
             _rows_parser(len(tables[t])), _reading_fields) for t in TABLES if tables[t] for s in SAMPLES]
    made = {j[0] for j in jobs}
    rd = proploop.cached_calls(name, udir / 'readings.json', jobs, call) if jobs else {}
    del jobs
    readings, why = {}, {}
    for key in READINGS:
        rec = rd.get(key) if key in made else None
        ok = rec is not None and bool(rec['valid'])
        readings[key] = {'valid': ok, 'explanation': rec['explanation'] if ok else None,
                         'rows': rec['rows'] if ok else None}
        why[key] = (None if ok else 'table has no rows' if key not in made else
                    'call failed, not cached; rerun' if rec is None else 'reply not valid after a retry')

    # picks: row numbers per arm, with the table they are from
    chosen = {key: (key.split('_')[0], readings[key]['rows'] or []) for key in READINGS}
    chosen.update({f'rule_{t}': (t, rule_rows(tables[t])) for t in TABLES})
    chosen['every'] = ('real', [r['row'] for r in tables['real']])
    by_no = {t: {r['row']: r for r in tables[t]} for t in TABLES}

    # checks on T, real and null
    T_ids = np.asarray(T, dtype=np.int64)
    rows_T = [row_of[i] for i in T]
    null_T = proploop.permute_within(rows_T, labels[T_ids], classes, np.random.default_rng(seed + NULL_T))
    R_T = proploop.centre(V[T_ids], labels[T_ids])
    picks = {arm: [pick(by_no[chosen[arm][0]][no], questions, R_T, rows_T, null_T, c, peers)
                   for no in chosen[arm][1]] for arm in ARMS}
    tal = tally(picks)

    stats = {'unit': name, 'layer': layer, 'chan': c, 'm': m, 'classes': classes, 'class_names': cnames,
             'peers': peer_recs, 'D': D, 'T': T, 'questions': questions, 'tables': tables, 'readings': readings,
             'picks': picks, 'tally': tal}
    proploop.save_json(stats, udir / 'stats.json')
    return section(name, layer, m, len(D), len(T), classes, cats, peer_recs, questions, tables, readings, why, picks,
                   tal), stats


# ---------- selftest ----------

def selftest():
    import math
    import statistics
    import tempfile

    def para(*pieces):
        return ' '.join(pieces)

    # prompt texts, verbatim from the spec: wrapped lines joined by single spaces
    assert TABLE_HEAD == para(
        'Below is what is known about one unit (a single neuron) of an image classification network. Someone who never saw',
        "the unit's activity answered the questions below about each of {n} photos of the {K} photo classes on which the",
        'unit is most active on average ({classes}). For each question, every two of its answers that were each given to at',
        'least 5 photos are compared. d is how much more active the unit is on the photos given the first answer than on',
        'the photos given the second, after removing the differences between photo classes, in standard deviations; each',
        'pair is written with the answer on which the unit is more active first. The percentile places d among all {m}',
        "units of the unit's layer on the same photos and answers (95 or above is unusual for this layer). The same",
        'percentile is given for the 3 units of the layer whose average activity over photo classes is most like this',
        "unit's (similar units 1 to 3): a row on which a similar unit is also at 95 or above describes something they",
        'share, not what is particular to this unit.')
    assert TABLE_TASK == para(
        'The rows will be checked on new photos of the same classes, none of them among those above, answered in the same',
        "way: a row holds when this unit's d, with the same answer first, is again at the 95th percentile of its layer or",
        'above. Say in a few sentences what this unit responds to, and choose up to 4 rows that you expect to hold for this',
        'unit on the new photos and that describe what is particular to this unit rather than what it shares with the',
        'similar units. Reply with only JSON: {"explanation": "...", "rows": [n, ...]}')
    assert READINGS == ['real_s1', 'real_s2', 'real_s3', 'null_s1', 'null_s2', 'null_s3']
    assert ARMS == READINGS + ['rule_real', 'rule_null', 'every'] and K_CLASSES == 24 and MIN_N == 5
    assert MAX_ROWS == 4 and PCT_RULE == 95.0 and PCT_DISTINCT == 50.0 and WORKERS == 1

    # pairs
    assert pairs(1) == [] and pairs(2) == [(0, 1)] and pairs(0) == []
    assert pairs(4) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # table rows on a hand case, m = 5, unit column 0, peers 3, 1, 2. Photos 0-4 (A), 5-9 (B), 10 (X), 11-14 (C),
    # 15-16 (N). Q0: A answer 0, B answer 1, C answer 2 (4 photos: its pairs are skipped), X and N None. Q1: B answer 0,
    # A and X answer 1 (the unit is less active on answer 0: written as (1, 0), counts swapped). Q2: all None.
    cols = [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 50, 50, 50, 50, -50, -50],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0.25] * 17,
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]]
    Rh = np.array(cols, dtype=np.float64).T
    q0 = [0] * 5 + [1] * 5 + [None] + [2] * 4 + [None] * 2
    q1 = [1] * 5 + [0] * 5 + [1] + [None] * 6
    ans_h = [list(x) for x in zip(q0, q1, [None] * 17)]
    qs_h = [{'question': 'Q0?', 'answers': ['a', 'b', 'c']}, {'question': 'Q1?', 'answers': ['a', 'b']},
            {'question': 'Q2?', 'answers': ['a', 'b']}]
    th = table_rows(Rh, ans_h, qs_h, 0, [3, 1, 2])
    # Q0 d: col0 sqrt(2.4), col1 -sqrt(2.4), col2 0, col3 1.242, col4 sqrt(3.6); Q1 d after the swap: col0 1.596,
    # col1 -1.596, col2 -0, col3 1.070, col4 0.989
    keys = ('row', 'q', 'first', 'second', 'n_first', 'n_second', 'pct', 'peer_pct')
    assert [{k: r[k] for k in keys} for r in th] == [
        {'row': 1, 'q': 0, 'first': 0, 'second': 1, 'n_first': 5, 'n_second': 5, 'pct': 60.0,
         'peer_pct': [40.0, 0.0, 20.0]},
        {'row': 2, 'q': 1, 'first': 1, 'second': 0, 'n_first': 6, 'n_second': 5, 'pct': 80.0,
         'peer_pct': [60.0, 0.0, 20.0]}], th
    assert all(sorted(r) == sorted(keys + ('d', 'gap')) for r in th)
    assert abs(th[0]['d'] - math.sqrt(2.4)) < 1e-12 and th[0]['gap'] == 40.0
    assert abs(th[1]['d'] - (5 / 6) / math.sqrt(3 / 11)) < 1e-12 and abs(th[1]['gap'] - 160 / 3) < 1e-9
    assert th[1]['d'] == -float(contrastloop.contrast(Rh, q1, 0, 1)[2][0])
    Rn = Rh.copy()
    Rn[15:] = 1e6  # photos with None for every question
    assert table_rows(Rn, ans_h, qs_h, 0, [3, 1, 2]) == th
    t4 = table_rows(Rh, ans_h, qs_h, 0, [3, 1, 2], min_n=4)
    assert [(r['row'], r['q'], r['first'], r['second'], r['n_first'], r['n_second']) for r in t4] == [
        (1, 0, 0, 1, 5, 5), (2, 0, 2, 0, 4, 5), (3, 0, 2, 1, 4, 5), (4, 1, 1, 0, 6, 5)], t4
    assert t4[0] == th[0] and dict(t4[3], row=2) == th[1]
    assert [(r['peer_pct'], r['gap']) for r in table_rows(Rh, ans_h, qs_h, 0, [])] == [([], None), ([], None)]
    assert table_rows(Rh, [[None] * 3] * 17, qs_h, 0, [1]) == []

    # table rows against a plain-Python computation on random data
    def brute_rows(R, ans_rows, questions, c, peers):
        res = []
        for q, Q in enumerate(questions):
            col = [r[q] for r in ans_rows]
            k = len(Q['answers'])
            for a in range(k):
                for b in range(a + 1, k):
                    ia = [t for t, x in enumerate(col) if x == a]
                    ib = [t for t, x in enumerate(col) if x == b]
                    if len(ia) < 5 or len(ib) < 5:
                        continue
                    ds = []
                    for j in range(R.shape[1]):
                        xa, xb = [float(R[t, j]) for t in ia], [float(R[t, j]) for t in ib]
                        sd = statistics.stdev(xa + xb)
                        ds.append((statistics.fmean(xa) - statistics.fmean(xb)) / sd if sd else 0.0)
                    if ds[c] < 0:
                        a2, b2, na, nb, ds = b, a, len(ib), len(ia), [-x for x in ds]
                    else:
                        a2, b2, na, nb = a, b, len(ia), len(ib)
                    pc = [100.0 * sum(x < y for x in ds) / len(ds) for y in ds]
                    res.append((q, a2, b2, na, nb, ds[c], pc[c], [pc[p] for p in peers]))
        return res
    rng = np.random.default_rng(1)
    Rr = rng.normal(size=(40, 8))
    bases = [[0] * 20 + [1] * 15 + [None] * 5, [0] * 4 + [1] * 18 + [2] * 13 + [None] * 5,
             [0] * 12 + [1] * 10 + [2] * 3 + [3] * 6 + [None] * 9]
    cols_r = [[bs[t] for t in rng.permutation(40)] for bs in bases]
    ans_r = [list(x) for x in zip(*cols_r)]
    qs_r = [{'question': f'R{k}?', 'answers': ['w', 'x', 'y', 'z'][:k]} for k in (2, 3, 4)]
    tr = table_rows(Rr, ans_r, qs_r, 2, [5, 0, 7])
    br = brute_rows(Rr, ans_r, qs_r, 2, [5, 0, 7])
    assert [(r['q'], sorted((r['first'], r['second']))) for r in tr] == [
        (0, [0, 1]), (1, [1, 2]), (2, [0, 1]), (2, [0, 3]), (2, [1, 3])], tr
    assert len(br) == 5 and [r['row'] for r in tr] == [1, 2, 3, 4, 5]
    for r, (q, a, b, na, nb, d, pc, pp) in zip(tr, br):
        assert (r['q'], r['first'], r['second'], r['n_first'], r['n_second']) == (q, a, b, na, nb), (r, q, a, b)
        assert abs(r['d'] - d) < 1e-9 and r['d'] >= 0 and r['pct'] == pc and r['peer_pct'] == pp, (r, d, pc, pp)
        assert abs(r['gap'] - (pc - sum(pp) / 3)) < 1e-9, r

    # table text on a hand case: whole numbers round half to even, d with a sign and 2 decimals
    tq = [{'question': 'How bright is it?', 'answers': ['dark', 'medium', 'bright']},
          {'question': 'Where is it?', 'answers': ['left', 'right']}]
    trows = [{'row': 1, 'q': 0, 'first': 2, 'second': 0, 'n_first': 12, 'n_second': 7, 'd': 0.456, 'pct': 96.5,
              'peer_pct': [12.5, 99.6, 50.0], 'gap': 42.47},
             {'row': 2, 'q': 1, 'first': 0, 'second': 1, 'n_first': 20, 'n_second': 9, 'd': 0.004, 'pct': 49.6,
              'peer_pct': [0.4, 2.5, 3.5], 'gap': 47.47}]
    head_t = (TABLE_HEAD.replace('{n}', '118').replace('{K}', '24').replace('{classes}', 'tabby, tiger cat')
              .replace('{m}', '512'))
    want_t = '\n\n'.join([
        head_t,
        'Row 1. How bright is it? "bright" (12 photos) more than "dark" (7 photos): d +0.46, percentile 96; similar '
        'units 12, 100, 50.\n'
        'Row 2. Where is it? "left" (20 photos) more than "right" (9 photos): d +0.00, percentile 50; similar units '
        '0, 2, 4.',
        TABLE_TASK])
    got_t = table_text(118, 24, ['tabby', 'tiger cat'], 512, tq, trows)
    assert got_t == want_t, got_t
    assert table_text(118, 24, ['tabby', 'tiger cat'], 512, tq, json.loads(json.dumps(trows))) == want_t
    assert '{' not in head_t and 'about each of 118 photos of the 24 photo classes' in head_t
    assert '(tabby, tiger cat)' in head_t and 'among all 512 units' in head_t

    # parse_rows: bools, duplicates, out of range and non-ints dropped, the cap, no explanation, no kept row
    def rep(obj):
        return 'Here:\n```json\n' + json.dumps(obj) + '\n```'
    assert parse_rows(rep({'explanation': ' Edges. ', 'rows': [3, True, 3, 0, 7, '2', 2.0, 1, -1, None, [4], 5, 2,
                                                               6]}), 6) == {'explanation': 'Edges.', 'rows': [3, 1, 5, 2]}
    assert parse_rows(rep({'explanation': 'x', 'rows': [2, 2, 2, 2, 4]}), 6) == {'explanation': 'x', 'rows': [2, 4]}
    assert parse_rows(rep({'explanation': 'x', 'rows': [True, 1, False]}), 2) == {'explanation': 'x', 'rows': [1]}
    assert parse_rows(rep({'explanation': 'x', 'rows': [6, 5, 4, 3, 2, 1]}), 6)['rows'] == [6, 5, 4, 3]
    for bad in [{'rows': [1]}, {'explanation': '  ', 'rows': [1]}, {'explanation': 5, 'rows': [1]},
                {'explanation': None, 'rows': [1]}, {'explanation': 'x', 'rows': []},
                {'explanation': 'x', 'rows': [0, 7, True, '1', 1.0, None]}, {'explanation': 'x', 'rows': 1},
                {'explanation': 'x', 'rows': '1'}, {'explanation': 'x'}]:
        assert parse_rows(json.dumps(bad), 6) is None, bad
    good = {'explanation': 'x', 'rows': [1]}
    assert parse_rows(json.dumps(good) + ' then ' + json.dumps({'explanation': 'y', 'rows': [9]}), 6) is None
    assert parse_rows(json.dumps({'explanation': 'y', 'rows': [9]}) + ' or ' + json.dumps(good), 6) == good
    assert parse_rows('no json here', 6) is None and parse_rows(json.dumps(good), 0) is None

    # rule_rows: the 95 cut, gap highest first, ties by lower row number, the first k
    def rr(no, pct, gap):
        return {'row': no, 'q': 0, 'first': 0, 'second': 1, 'n_first': 5, 'n_second': 5, 'd': 1.0, 'pct': pct,
                'peer_pct': [0.0, 0.0, 0.0], 'gap': gap}
    below = float(np.nextafter(95.0, 0.0))
    rs = [rr(4, 96.0, 10.0), rr(2, below, 90.0), rr(6, 97.0, 30.0), rr(1, 95.0, 10.0), rr(5, 100.0, -5.0),
          rr(3, 99.0, 30.0), rr(7, 98.0, 50.0), rr(8, 0.0, 99.0)]
    assert rule_rows(rs) == [7, 3, 6, 1], rule_rows(rs)
    assert rule_rows(rs, k=10) == [7, 3, 6, 1, 4, 5] and rule_rows(rs, k=2) == [7, 3]
    assert rule_rows([]) == [] and rule_rows([rr(1, below, 50.0)]) == []
    assert rule_rows(json.loads(json.dumps(rs))) == [7, 3, 6, 1]
    assert rule_rows([rr(1, 99.0, None), rr(2, 96.0, -50.0)]) == [2, 1]

    # t_check: 21 columns, so that a percentile can reach 95 (20 of 21 below). Column 0 has constant answer groups,
    # the largest d possible for 5 and 5 photos; photos 10 (answer 2) and 11 (None) are left out
    rng = np.random.default_rng(2)
    Rt = rng.normal(size=(12, 21))
    Rt[:, 0] = [1] * 5 + [0] * 5 + [100, -100]
    ans_t = [0] * 5 + [1] * 5 + [2, None]
    dt = contrastloop.contrast(Rt, ans_t, 0, 1)[2]
    order = sorted(range(1, 21), key=lambda j: dt[j])  # order[r]: the column with r columns below it
    assert all(dt[order[r]] < dt[order[r + 1]] for r in range(19)) and dt[order[19]] < dt[0]
    held = t_check(Rt, ans_t, 0, 1, 0, [order[19], order[11], order[18]])
    assert held == {'n_first': 5, 'n_second': 5, 'd': float(dt[0]), 'pct': 100.0 * 20 / 21,
                    'peer_pct': [100.0 * 19 / 21, 100.0 * 11 / 21, 100.0 * 18 / 21], 'outcome': 'held',
                    'distinctive': False}, held
    assert abs(held['d'] - math.sqrt(3.6)) < 1e-12
    dis = t_check(Rt, ans_t, 0, 1, 0, [order[19], order[10], order[18]])
    assert dis['outcome'] == 'held' and dis['distinctive'] is True and dis['peer_pct'][1] == 100.0 * 10 / 21
    lo = t_check(Rt, ans_t, 1, 0, 0, [order[0], order[1], order[2]])
    assert lo['outcome'] == 'not held' and lo['pct'] == 0.0 and lo['d'] == -float(dt[0]) and not lo['distinctive']
    # d reversed: column 0 lowest, order[r] with 20 - r columns below it
    assert (lo['n_first'], lo['n_second'], lo['peer_pct']) == (5, 5, [100.0 * 20 / 21, 100.0 * 19 / 21,
                                                                     100.0 * 18 / 21])
    mid = t_check(Rt, ans_t, 0, 1, order[10], [order[0], order[1], order[2]])
    assert mid['outcome'] == 'not held' and mid['pct'] == 100.0 * 10 / 21 and not mid['distinctive']
    ans_f = list(ans_t)
    ans_f[0] = None
    assert t_check(Rt, ans_f, 0, 1, 0, [1, 2, 3]) == {'n_first': 4, 'n_second': 5, 'd': None, 'pct': None,
                                                       'peer_pct': None, 'outcome': 'too few', 'distinctive': False}
    few = t_check(Rt, ans_t, 2, 0, 0, [1, 2, 3])
    assert (few['n_first'], few['n_second'], few['outcome'], few['distinctive']) == (1, 5, 'too few', False)

    # one answer pass and one unit end to end, with a stand-in for the agents and synthetic values (no model, no
    # Claude, no photos): a batch that fails after a retry, a batch with an is_error reply (stop, then a rerun)
    proploop.WORKERS = WORKERS
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rng = np.random.default_rng(4)
        Vs = rng.random((600, 30)).astype(np.float32)
        labs = rng.permutation(np.repeat(np.arange(60), 10))
        nms = [f'img/{i}.jpeg' for i in range(600)]
        cats_t = [f'class {i}' for i in range(60)]
        qs_t = [{'question': 'Is it red?', 'answers': ['yes', 'no']},
                {'question': 'Which side?', 'answers': ['left', 'middle', 'right']},
                {'question': 'Is there text?', 'answers': ['yes', 'no']}]
        us = [proploop.parse_unit('mixed4c:3'), proploop.parse_unit('mixed4c:7')]
        P_t = sorted({i for u in us for i in unit_photos(u, Vs, labs, nms)[1].tolist()})
        bt_t = proploop.answer_batches(BATCH_KEY, P_t)
        state = {'n': 0, 'err': True}

        def call_u(sp, turns):
            assert sp == '' and len(turns) == 1
            content = turns[0]
            state['n'] += 1
            texts = [b['text'] for b in content if b['type'] == 'text']
            ids = [b['id'] for b in content if b['type'] == 'image']
            if not ids:
                assert len(content) == 1 and texts[0].startswith('Below is what is known about one unit')
                return [{'text': 'Rows: ' + json.dumps({'explanation': ' edges ', 'rows': [2, 1, 99]}),
                         'is_error': False}]
            assert texts[-1].startswith(f'Answer every question below for each of the {len(ids)} images above.')
            if ids[0] == bt_t[0][0]:
                return [{'text': 'sorry', 'is_error': False}]
            if ids[0] == bt_t[1][0] and state['err']:
                state['err'] = False
                return [{'text': 'overloaded', 'is_error': True}]
            ent = [{'id': k, 'answers': ['AB'[i % 2], 'ABC'[i % 3], '?']} for k, i in enumerate(ids, 1)]
            return [{'text': json.dumps({'images': ent}), 'is_error': False}]

        def blk(i):
            return {'type': 'image', 'id': int(i)}
        try:
            answer_pass(P_t, qs_t, call=call_u, out=td, block=blk)
            raise AssertionError('answers with a batch not cached accepted')
        except SystemExit:
            pass
        assert state['n'] == len(bt_t) + 1, state  # batch 1 twice, batch 2 once (is_error), the others once
        state['n'] = 0
        row_of_t, bt, failed_t = answer_pass(P_t, qs_t, call=call_u, out=td, block=blk)
        assert state['n'] == 1 and bt == bt_t and failed_t == ['1'] and sorted(row_of_t) == P_t, state
        assert all(row_of_t[i] == [None] * 3 for i in bt[0])
        assert all(row_of_t[i] == [i % 2, i % 3, None] for b in bt[1:] for i in b)

        state['n'] = 0
        u0 = us[0]
        sec, st = run_unit(u0, Vs, labs, nms, cats_t, row_of_t, qs_t, call=call_u, out=td)
        assert state['n'] == 6, state  # 3 samples of each table
        cls_t, sid_t, D_t, T_t = unit_photos(u0, Vs, labs, nms)
        assert st['classes'] == cls_t and st['D'] == D_t and st['T'] == T_t and st['m'] == 30 and st['chan'] == 3
        assert sorted(D_t + T_t) == sid_t.tolist() and len(cls_t) == 24
        peers_t = [p for p, _ in contrastloop.peers(Vs, labs, 3)]
        assert [p['chan'] for p in st['peers']] == peers_t and st['peers'][0]['unit'] == f'mixed4c:{peers_t[0]}'
        for t in TABLES:
            assert [(r['row'], r['q'], sorted((r['first'], r['second']))) for r in st['tables'][t]] == [
                (1, 0, [0, 1]), (2, 1, [0, 1]), (3, 1, [0, 2]), (4, 1, [1, 2])], st['tables'][t]
        for key in READINGS:
            assert st['readings'][key] == {'valid': True, 'explanation': 'edges', 'rows': [2, 1]}, key
            assert [p['row'] for p in st['picks'][key]] == [2, 1]
        assert [p['D'] for p in st['picks']['every']] == st['tables']['real']
        assert [p['D'] for p in st['picks']['null_s1']] == [st['tables']['null'][1], st['tables']['null'][0]]
        for t in TABLES:
            assert [p['row'] for p in st['picks'][f'rule_{t}']] == rule_rows(st['tables'][t])
        for arm in ARMS:
            for p in st['picks'][arm]:
                assert p['question'] == qs_t[p['q']]['question'] and (p['first'], p['second']) == (
                    p['D']['first'], p['D']['second']) and p['first_answer'] == qs_t[p['q']]['answers'][p['first']]
            assert all(sum(st['tally'][arm][t].values()) == len(st['picks'][arm]) for t in TABLES)
        T_ids = np.asarray(T_t, dtype=np.int64)
        R_T = proploop.centre(Vs[T_ids], labs[T_ids])
        p0 = st['picks']['every'][0]
        assert p0['T'] == t_check(R_T, [row_of_t[i][p0['q']] for i in T_t], p0['first'], p0['second'], 3, peers_t)
        assert json.loads((td / 'mixed4c:3' / 'stats.json').read_text()) == st
        lines = sec.splitlines()
        assert lines[0].startswith('== mixed4c:3  layer mixed4c (30 channels), ') and lines[3] == 'real table:', lines
        assert 'real_s1  rows [2, 1]  edges' in lines and 'null_s3  rows [2, 1]  edges' in lines
        state['n'] = 0
        assert run_unit(u0, Vs, labs, nms, cats_t, row_of_t, qs_t, call=call_u, out=td) == (sec, st)
        assert state['n'] == 0
    print('selftest ok')


# ---------- main ----------

def main():
    if '--selftest' in sys.argv[1:]:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--units-file', default=UNITS_FILE, help='relative paths are taken from the repo root')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    proploop.WORKERS = WORKERS  # one call at a time, set before any call
    uf = Path(a.units_file)
    units = [proploop.parse_unit(s) for s in json.loads((uf if uf.is_absolute() else J / uf).read_text())]
    questions = load_questions()
    names = json.loads(proploop.NAMES.read_text())
    with np.load(proploop.ACTS) as z:
        labels = np.asarray(z['classes'], dtype=np.int64)
    assert len(names) == len(labels) == proploop.N_PHOTOS, (len(names), len(labels))
    assert labels.min() >= 0 and labels.max() < 1000, (labels.min(), labels.max())
    from torchvision.models import GoogLeNet_Weights
    cats = list(GoogLeNet_Weights.IMAGENET1K_V1.meta['categories'])
    assert len(cats) == 1000, len(cats)
    held = {}

    def get_model(dev):
        if dev not in held:
            import torch
            torch.set_num_threads(4)
            m = bundlemax._bank().load_model(proploop.WEIGHTS)
            for q in m.parameters():
                q.requires_grad_(False)
            held[dev] = m.to(dev)
        return held[dev]

    OUT.mkdir(parents=True, exist_ok=True)
    layers = sorted({u['layer'] for u in units}, key=proploop.LAYERS.index)
    vals = proploop.layer_values(layers, get_model, np.arange(proploop.N_PHOTOS), out=proploop.OUT)
    held.clear()
    gc.collect()

    # answers: one pass over the union of all units' photo sets, before any unit's table
    P = sorted({i for u in units for i in unit_photos(u, vals[u['layer']], labels, names)[1].tolist()})
    row_of, batches, failed = answer_pass(P, questions)
    secs = [f'answers: {len(P)} photos, answer batches failed after a retry: {len(failed)} of {len(batches)}'
            + (f' (batches {failed})' if failed else '')]
    print(secs[0], flush=True)
    for u in units:
        print('==', u['name'], flush=True)
        sec, _ = run_unit(u, vals[u['layer']], labels, names, cats, row_of, questions)
        secs.append(sec)
        (OUT / 'report.txt').write_text('\n\n'.join(secs) + '\n')
        print(sec, flush=True)
        gc.collect()
    print('report ->', OUT / 'report.txt')


if __name__ == '__main__':
    main()
