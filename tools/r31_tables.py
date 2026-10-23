"""pairtable's tables and checks on a second photo pool, without readings (spec: authors/spec_r31_tables.md).

Per unit (one channel of mixed3a-mixed5b): its 24 classes (pairtable.unit_photos) and its 3 similar units
(contrastloop.peers) are taken from the repo's photos, as tools/pairtable.py takes them. A second pool (--pool: a
directory with names.json, labels.npy and values_<layer>.npy, pool ids x channels) gives the unit's new photos: every
photo of those classes, in id order, split by crc32(photo name) % 2 into D (0) and T (1). Their answers to the questions
of authors/standing.reply.json come from a tools/vlm_answer.py output (--answers); a photo with no record, or with a
record whose valid is false, has a row of None. From there as pairtable.run_unit, on the new pool: the real table and the
null table (answers permuted within class) on D; the arms rule_real and rule_null (the rule's rows of each table) and
every (every row of the real table); each picked row checked on T with the real and with the permuted answers; the
outcomes tallied per arm. No readings and no calls. Outputs: <out>/<unit>/stats.json and <out>/report.txt.

Decision printed in the report, per checked row (as pairtable): the unit's d on T, with the same answer first, at a
percentile >= 95 among its layer's channels and above 0 -> held, else not held; too few when either answer has fewer
than 5 photos of T; distinctive when held and the lowest of its 3 similar units' percentiles is below 50.

usage: python -I tools/r31_tables.py --units UNITS.json [MORE.json ...] --pool DIR --answers FILE --out DIR
       python -I tools/r31_tables.py --selftest
(relative paths are taken from the repo root)
"""
import argparse
import functools
import gc
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

TABLES = pairtable.TABLES
ARMS = [a for a in pairtable.ARMS if a not in pairtable.READINGS]  # the arms that need no call
OUTCOMES = pairtable.OUTCOMES


# ---------- inputs ----------

@functools.lru_cache(maxsize=None)
def categories():
    """torchvision's GoogLeNet categories (1000 names), as pairtable.main"""
    from torchvision.models import GoogLeNet_Weights
    cats = tuple(GoogLeNet_Weights.IMAGENET1K_V1.meta['categories'])
    assert len(cats) == 1000, len(cats)
    return cats


def _path(s):
    """a path argument; relative paths are taken from the repo root"""
    p = Path(s)
    return p if p.is_absolute() else J / p


def load_pool(pool):
    """(names, labels) of the new pool: names.json (a list of names) and labels.npy (integers 0-999), one per pool id;
    labels as int64. Stops when they do not match"""
    names = json.loads((pool / 'names.json').read_text())
    labels = np.load(pool / 'labels.npy')
    ok = (isinstance(names, list) and labels.ndim == 1 and np.issubdtype(labels.dtype, np.integer)
          and len(names) == len(labels) > 0)
    if not ok or labels.min() < 0 or labels.max() >= 1000:
        raise SystemExit(f'{pool}: names.json ({type(names).__name__}) and labels.npy (shape {labels.shape}, dtype '
                         f'{labels.dtype}) are not one name and one label 0-999 per pool id; stopping')
    return names, labels.astype(np.int64)


def pool_values(pool, layer, n):
    """the new pool's values_<layer>.npy: n x the layer's channels, float32; stops otherwise"""
    path = pool / f'values_{layer}.npy'
    if not path.exists():
        raise SystemExit(f'{path}: missing; stopping')
    v = np.load(path)
    if v.shape != (n, proploop.NCH[layer]) or v.dtype != np.float32:
        raise SystemExit(f'{path}: shape {v.shape}, dtype {v.dtype}; expected ({n}, {proploop.NCH[layer]}) float32; '
                         'stopping')
    return v


def ref_values(layer):
    """the repo pool's values of one layer (N_PHOTOS x channels): proploop.layer_values from its cache, no model"""
    path = proploop.OUT / f'values_{layer}.npy'
    if not path.exists():
        raise SystemExit(f'{path}: missing, and no model is loaded here; stopping')
    return proploop.layer_values([layer], None, np.arange(proploop.N_PHOTOS))[layer]


def answer_rows(answers, n, questions):
    """{pool id: answer row} for every id 0..n-1 of the new pool, from a tools/vlm_answer.py output {id (str):
    {'answers', 'valid', ...}}: the record's answers when valid, a row of None for every question when the record is
    missing or valid is false. Stops when a key is not a pool id, or a valid record's answers is not one answer index
    (an int from 0 to the question's answers - 1) or None per question"""
    nq = len(questions)
    if not isinstance(answers, dict):
        raise SystemExit('answers: not an object {id: record}; stopping')
    out = {i: [None] * nq for i in range(n)}
    for key, rec in answers.items():
        ok_key = isinstance(key, str) and key.isascii() and key.isdigit() and str(int(key)) == key and int(key) < n
        if not ok_key or not isinstance(rec, dict):
            raise SystemExit(f'answers: {key!r} is not a pool id (0 to {n - 1}) with a record; stopping')
        if not rec.get('valid'):
            continue
        row = rec.get('answers')
        ok = isinstance(row, list) and len(row) == nq and all(
            x is None or (isinstance(x, int) and not isinstance(x, bool) and 0 <= x < len(q['answers']))
            for x, q in zip(row, questions))
        if not ok:
            raise SystemExit(f'answers: photo {key}: {row!r} is not one answer index or None per question; stopping')
        out[int(key)] = list(row)
    return out


# ---------- one unit, report ----------

def unit_new(u, ref, new, row_of, questions):
    """one unit (u from proploop.parse_unit) on the new pool. ref, new: {'V' (the unit's layer values, photos x
    channels), 'labels', 'names'} of the reference pool and of the new pool; row_of: {new pool id: answer row}. Classes
    and peers from the reference pool, as pairtable.run_unit takes them; the unit's new photos: every new pool id whose
    label is in classes, in id order, split into D and T by contrastloop.split; then as run_unit on the new pool: the
    real and null tables on D, the arms rule_real, rule_null and every, each pick checked on T with the real and the
    null answer rows, the tally. -> stats"""
    name, layer, c = u['name'], u['layer'], u['chan']
    Vr, lr = np.asarray(ref['V']), np.asarray(ref['labels'])
    V, labels = np.asarray(new['V']), np.asarray(new['labels'])
    m = V.shape[1]
    assert Vr.shape[1] == m and V.shape[0] == len(labels) == len(new['names']), (Vr.shape, V.shape, len(labels))
    seed = zlib.crc32(name.encode())

    # classes and peers: from the reference pool
    classes = pairtable.unit_photos(u, Vr, lr, ref['names'])[0]
    cnames = [categories()[x] for x in classes]
    pr = contrastloop.peers(Vr, lr, c)
    peers = [p for p, _ in pr]
    peer_recs = [{'chan': p, 'unit': f'{layer}:{p}', 'corr': r} for p, r in pr]

    # the unit's new photos
    D, T = contrastloop.split(np.flatnonzero(np.isin(labels, classes)).tolist(), new['names'])

    # tables on D, real and null
    D_ids = np.asarray(D, dtype=np.int64)
    rows_D = [row_of[i] for i in D]
    null_D = proploop.permute_within(rows_D, labels[D_ids], classes, np.random.default_rng(seed + pairtable.NULL_D))
    R_D = proploop.centre(V[D_ids], labels[D_ids])
    tables = {t: pairtable.table_rows(R_D, rr, questions, c, peers) for t, rr in zip(TABLES, (rows_D, null_D))}

    # picks: row numbers per arm, with the table they are from
    chosen = {f'rule_{t}': (t, pairtable.rule_rows(tables[t])) for t in TABLES}
    chosen['every'] = ('real', [r['row'] for r in tables['real']])
    by_no = {t: {r['row']: r for r in tables[t]} for t in TABLES}

    # checks on T, real and null
    T_ids = np.asarray(T, dtype=np.int64)
    rows_T = [row_of[i] for i in T]
    null_T = proploop.permute_within(rows_T, labels[T_ids], classes, np.random.default_rng(seed + pairtable.NULL_T))
    R_T = proploop.centre(V[T_ids], labels[T_ids])
    picks = {arm: [pairtable.pick(by_no[chosen[arm][0]][no], questions, R_T, rows_T, null_T, c, peers)
                   for no in chosen[arm][1]] for arm in ARMS}
    tal = pairtable.tally(picks)

    def unanswered(ids):
        return sum(all(x is None for x in row_of[i]) for i in ids)
    return {'unit': name, 'layer': layer, 'chan': c, 'm': m, 'classes': classes, 'class_names': cnames,
            'peers': peer_recs, 'D': D, 'T': T, 'tables': tables, 'picks': picks, 'tally': tal,
            'n_unanswered_D': unanswered(D), 'n_unanswered_T': unanswered(T)}


def section(st, questions):
    """one unit's report lines, in pairtable.section's format: the head, classes, peers, the real table, the picks of
    each arm, the tallies (no reading lines)"""
    tables, picks, tal = st['tables'], st['picks'], st['tally']
    L = [f'== {st["unit"]}  layer {st["layer"]} ({st["m"]} channels), {len(st["D"])} photos looked at (D), '
         f'{len(st["T"])} photos checked (T); table rows: ' + ', '.join(f'{t} {len(tables[t])}' for t in TABLES),
         'classes: ' + ', '.join(f'{n} ({x})' for x, n in zip(st['classes'], st['class_names'])),
         'peers: ' + ', '.join(f'{p["unit"]} (r {contrastloop._corr(p["corr"])})' for p in st['peers']),
         'real table:']
    L += ['   ' + pairtable._row_text(r, questions) for r in tables['real']]
    for arm in ARMS:
        L.append(f'{arm}: {len(picks[arm])} rows')
        L += ['   ' + pairtable._pick_text(p) for p in picks[arm]]
    for arm in ARMS:
        L.append(f'tally {arm}: ' + '; '.join(f'{t} ' + ', '.join(f'{o} {tal[arm][t][o]}' for o in OUTCOMES)
                                              for t in TABLES) + f' (of {len(picks[arm])} rows)')
    return '\n'.join(L)


# ---------- selftest ----------

def selftest():
    import tempfile

    assert ARMS == ['rule_real', 'rule_null', 'every'] and TABLES == ('real', 'null')
    qs = [{'question': 'Q0?', 'answers': ['a', 'b']}, {'question': 'Q1?', 'answers': ['a', 'b', 'c']},
          {'question': 'Q2?', 'answers': ['a', 'b']}]

    # answer rows: a missing record and a record not valid give a row of None; keys and valid answers checked
    got = answer_rows({'0': {'answers': [1, 2, None], 'valid': True}, '2': {'answers': [0, 0, 1], 'valid': False},
                       '3': {'answers': [None] * 3, 'valid': True}, '4': {'valid': False}}, 5, qs)
    assert got == {0: [1, 2, None], 1: [None] * 3, 2: [None] * 3, 3: [None] * 3, 4: [None] * 3}, got
    for bad in [{'5': {'answers': [0, 0, 0], 'valid': True}}, {'01': {'answers': [0, 0, 0], 'valid': True}},
                {'-1': {'answers': [0, 0, 0], 'valid': True}}, {'x': {'answers': [0, 0, 0], 'valid': True}},
                {'1': {'answers': [0, 0], 'valid': True}}, {'1': {'answers': [0, 3, 0], 'valid': True}},
                {'1': {'answers': [0, 0, 2], 'valid': True}}, {'1': {'answers': [True, 0, 0], 'valid': True}},
                {'1': {'answers': ['A', 0, 0], 'valid': True}}, {'1': {'answers': [0, 1.0, 0], 'valid': True}},
                {'1': {'valid': True}}, {'1': [0, 0, 0]}, [[0, 0, 0]]]:
        try:
            answer_rows(bad, 5, qs)
        except SystemExit:
            continue
        raise AssertionError(f'accepted {bad}')

    # two synthetic pools, 21 channels so that a percentile can reach 95 (20 of 21 below). Reference pool: labels
    # 40-65, 3 photos each; the unit's channel 3 has class means in label order (40 and 41 lowest, so not among its 24
    # classes); channels 14, 5 and 9 follow it (its similar units there)
    rng = np.random.default_rng(5)
    m = 21
    lr = rng.permutation(np.repeat(np.arange(40, 66), 3))
    nr = [f'ref/{i}.jpeg' for i in range(len(lr))]
    Vr = rng.random((len(lr), m))
    Vr[:, 3] = (lr - 40) + 0.1 * rng.random(len(lr))
    Vr[:, 14] = 2 * Vr[:, 3] + 0.1 * rng.random(len(lr))
    Vr[:, 5] = Vr[:, 3] + 0.5 * rng.random(len(lr))
    Vr[:, 9] = Vr[:, 3] + 2 * rng.random(len(lr))
    Vr = Vr.astype(np.float32)
    # new pool: labels 40-65, 8 photos each in id order; channel 3 is 1 higher on even ids within every class; its
    # class means put 40 and 41 first (the new pool's own classes would differ) and channels 7, 11 and 16 follow them
    # (so would its own similar units)
    n = 208
    ids_all = np.arange(n)
    ln = 40 + ids_all // 8
    nn = [f'new/{i}.jpeg' for i in range(n)]
    base = np.where(ln < 42, 30.0, 0.1 * (ln - 40))
    Vn = rng.random((n, m))
    Vn[:, 3] = base + (ids_all % 2 == 0) + 0.01 * rng.random(n)
    for j in (7, 11, 16):
        Vn[:, j] = base + rng.random(n)
    Vn = Vn.astype(np.float32)
    # answers by hand: id % 13 == 5 no record, == 7 a record not valid, == 11 a valid row of None; the others Q0 id % 2
    # (answer 0 on even ids), Q1 (id // 2) % 3, Q2 answer 0 or None (answer 1 never given, so Q2 has no row)
    unans = (5, 7, 11)
    ans = {}
    for i in range(n):
        if i % 13 == 7:
            ans[str(i)] = {'answers': [0, 0, 0], 'valid': False}
        elif i % 13 == 11:
            ans[str(i)] = {'answers': [None, None, None], 'valid': True}
        elif i % 13 != 5:
            ans[str(i)] = {'answers': [i % 2, (i // 2) % 3, 0 if i % 3 == 0 else None], 'valid': True}
    row_of = answer_rows(ans, n, qs)
    assert all(row_of[i] == ([None] * 3 if i % 13 in unans else ans[str(i)]['answers']) for i in range(n))

    u = proploop.parse_unit('mixed4c:3')
    st = unit_new(u, {'V': Vr, 'labels': lr, 'names': nr}, {'V': Vn, 'labels': ln, 'names': nn}, row_of, qs)

    # the same steps by hand with the pairtable functions
    cls_h = pairtable.unit_photos(u, Vr, lr, nr)[0]
    pr_h = contrastloop.peers(Vr, lr, 3)
    peers_h = [p for p, _ in pr_h]
    D_h, T_h = contrastloop.split([i for i in range(n) if int(ln[i]) in cls_h], nn)
    seed_h = zlib.crc32(b'mixed4c:3')
    Di, Ti = np.asarray(D_h, dtype=np.int64), np.asarray(T_h, dtype=np.int64)
    rD, rT = [row_of[i] for i in D_h], [row_of[i] for i in T_h]
    zD = proploop.permute_within(rD, ln[Di], cls_h, np.random.default_rng(seed_h + pairtable.NULL_D))
    zT = proploop.permute_within(rT, ln[Ti], cls_h, np.random.default_rng(seed_h + pairtable.NULL_T))
    RD, RT = proploop.centre(Vn[Di], ln[Di]), proploop.centre(Vn[Ti], ln[Ti])
    tab_h = {'real': pairtable.table_rows(RD, rD, qs, 3, peers_h), 'null': pairtable.table_rows(RD, zD, qs, 3, peers_h)}

    def picks_h(t, nos):
        by = {r['row']: r for r in tab_h[t]}
        return [pairtable.pick(by[no], qs, RT, rT, zT, 3, peers_h) for no in nos]
    pk_h = {'rule_real': picks_h('real', pairtable.rule_rows(tab_h['real'])),
            'rule_null': picks_h('null', pairtable.rule_rows(tab_h['null'])),
            'every': picks_h('real', [r['row'] for r in tab_h['real']])}

    assert list(st) == ['unit', 'layer', 'chan', 'm', 'classes', 'class_names', 'peers', 'D', 'T', 'tables', 'picks',
                        'tally', 'n_unanswered_D', 'n_unanswered_T'], list(st)
    assert (st['unit'], st['layer'], st['chan'], st['m']) == ('mixed4c:3', 'mixed4c', 3, m)
    assert st['classes'] == cls_h == list(range(65, 41, -1)), st['classes']
    assert st['class_names'] == [categories()[x] for x in cls_h]
    assert st['peers'] == [{'chan': p, 'unit': f'mixed4c:{p}', 'corr': r} for p, r in pr_h]
    assert sorted(peers_h) == [5, 9, 14], pr_h
    assert st['D'] == D_h and st['T'] == T_h and sorted(D_h + T_h) == list(range(16, n))
    assert st['tables'] == tab_h and st['picks'] == pk_h and st['tally'] == pairtable.tally(pk_h)
    assert st['n_unanswered_D'] == sum(i % 13 in unans for i in D_h) > 0
    assert st['n_unanswered_T'] == sum(i % 13 in unans for i in T_h) > 0
    # the new pool's own classes and similar units differ from the reference pool's
    assert 40 in pairtable.unit_photos(u, Vn, ln, nn)[0]
    assert sorted(p for p, _ in contrastloop.peers(Vn, ln, 3)) == [7, 11, 16]

    # rows and checks by construction: Q0's pair is row 1, the unit highest of 21 on D and on T
    for t in TABLES:
        assert [(r['q'], sorted((r['first'], r['second']))) for r in st['tables'][t]] == [
            (0, [0, 1]), (1, [0, 1]), (1, [0, 2]), (1, [1, 2])], st['tables'][t]
    r0 = st['tables']['real'][0]
    ans_D = [i for i in D_h if i % 13 not in unans]
    assert (r0['first'], r0['second'], r0['pct']) == (0, 1, 100.0 * 20 / 21), r0
    assert (r0['n_first'], r0['n_second']) == (sum(i % 2 == 0 for i in ans_D), sum(i % 2 for i in ans_D)), r0
    assert [p['D'] for p in st['picks']['every']] == st['tables']['real'] and st['picks']['rule_real']
    assert [p['row'] for p in st['picks']['rule_real']] == pairtable.rule_rows(st['tables']['real'])
    assert [p['row'] for p in st['picks']['rule_null']] == pairtable.rule_rows(st['tables']['null'])
    assert all(p['D'] == st['tables']['null'][p['row'] - 1] for p in st['picks']['rule_null'])
    p0 = st['picks']['every'][0]
    ans_T = [i for i in T_h if i % 13 not in unans]
    assert (p0['T']['n_first'], p0['T']['n_second'], p0['T']['outcome'], p0['T']['pct']) == (
        sum(i % 2 == 0 for i in ans_T), sum(i % 2 for i in ans_T), 'held', 100.0 * 20 / 21), p0['T']
    for arm in ARMS:
        assert all(sum(st['tally'][arm][t].values()) == len(st['picks'][arm]) for t in TABLES), arm
    assert json.loads(json.dumps(st)) == st

    # the report lines: pairtable.section's, without its reading lines and the reading arms' lines
    none = {k: [] for k in pairtable.READINGS}
    full = pairtable.section(st['unit'], st['layer'], st['m'], len(st['D']), len(st['T']), st['classes'], categories(),
                             st['peers'], qs, st['tables'], {k: {'valid': False} for k in pairtable.READINGS},
                             {k: 'none' for k in pairtable.READINGS}, {**none, **st['picks']},
                             pairtable.tally({**none, **st['picks']}))
    want = [x for x in full.splitlines()
            if not any(x.startswith((f'{k}  ', f'{k}: ', f'tally {k}: ')) for k in pairtable.READINGS)]
    sec = section(st, qs)
    assert sec == '\n'.join(want), sec
    lines = sec.splitlines()
    assert lines[0].startswith(f'== mixed4c:3  layer mixed4c ({m} channels), {len(D_h)} photos looked at (D), ')
    assert lines[3] == 'real table:' and len(lines) == 4 + 4 + sum(1 + len(st['picks'][a]) for a in ARMS) + 3
    assert lines[-3].startswith('tally rule_real: real held ') and lines[-1].startswith('tally every: ')
    assert not any(k in sec for k in pairtable.READINGS)

    # the pool's files
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / 'names.json').write_text(json.dumps(['a.jpg', 'b.jpg', 'c.jpg']))
        np.save(td / 'labels.npy', np.array([7, 3, 7], dtype=np.int64))
        nf, lf = load_pool(td)
        assert nf == ['a.jpg', 'b.jpg', 'c.jpg'] and lf.tolist() == [7, 3, 7] and lf.dtype == np.int64
        np.save(td / 'values_mixed3a.npy', np.ones((3, 256), dtype=np.float32))
        assert pool_values(td, 'mixed3a', 3).shape == (3, 256)
        for bad_v in [np.ones((2, 256), np.float32), np.ones((3, 480), np.float32), np.ones((3, 256), np.float64)]:
            np.save(td / 'values_mixed3a.npy', bad_v)
            try:
                pool_values(td, 'mixed3a', 3)
            except SystemExit:
                continue
            raise AssertionError(f'values {bad_v.shape} {bad_v.dtype} accepted')
        for bad_l in [np.array([7, 3]), np.array([7, 3, 1000]), np.array([7, -1, 3]), np.array([7.0, 3.0, 7.0])]:
            np.save(td / 'labels.npy', bad_l)
            try:
                load_pool(td)
            except SystemExit:
                continue
            raise AssertionError(f'labels {bad_l} accepted')
    print('selftest ok')


# ---------- main ----------

def main():
    if '--selftest' in sys.argv[1:]:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--units', nargs='+', required=True, help='JSON files, each a list of unit strings')
    ap.add_argument('--pool', required=True, help='a directory with names.json, labels.npy and values_<layer>.npy')
    ap.add_argument('--answers', required=True, help="a tools/vlm_answer.py output for the pool's photos")
    ap.add_argument('--out', required=True, help='an output directory')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()  # relative paths are taken from the repo root
    strs = [s for f in a.units for s in json.loads(_path(f).read_text())]
    units = [proploop.parse_unit(s) for s in dict.fromkeys(strs)]  # each unit once, in first-seen order
    questions = pairtable.load_questions()
    ref_names = json.loads(proploop.NAMES.read_text())
    with np.load(proploop.ACTS) as z:
        ref_labels = np.asarray(z['classes'], dtype=np.int64)
    assert len(ref_names) == len(ref_labels) == proploop.N_PHOTOS, (len(ref_names), len(ref_labels))
    assert ref_labels.min() >= 0 and ref_labels.max() < 1000, (ref_labels.min(), ref_labels.max())
    pool, out = _path(a.pool), _path(a.out)
    new_names, new_labels = load_pool(pool)
    row_of = answer_rows(json.loads(_path(a.answers).read_text()), len(new_labels), questions)
    categories()
    out.mkdir(parents=True, exist_ok=True)

    # one layer's values at a time; the report keeps the units' order
    secs = {}
    for layer in sorted({u['layer'] for u in units}, key=proploop.LAYERS.index):
        ref = {'V': ref_values(layer), 'labels': ref_labels, 'names': ref_names}
        new = {'V': pool_values(pool, layer, len(new_labels)), 'labels': new_labels, 'names': new_names}
        for u in [v for v in units if v['layer'] == layer]:
            print('==', u['name'], flush=True)
            st = unit_new(u, ref, new, row_of, questions)
            proploop.save_json(st, out / u['name'] / 'stats.json')
            secs[u['name']] = section(st, questions)
            (out / 'report.txt').write_text('\n\n'.join(secs[v['name']] for v in units if v['name'] in secs) + '\n')
            print(secs[u['name']], flush=True)
            gc.collect()
        del ref, new
        gc.collect()
    print('report ->', out / 'report.txt')


if __name__ == '__main__':
    main()
