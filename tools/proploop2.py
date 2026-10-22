"""Round-25 question tables plus a fixed set of standing questions (spec: authors/spec_proploop2.md).

Per unit (one channel of mixed3a-mixed5b): the questions are proploop's kept questions for the unit
(runs/proploop/<unit>/questions.json, Q1-Q8), then the standing questions of authors/standing.reply.json (Q9-...),
which are the same for every unit. The answers to the first come from proploop's answers.json. The standing questions
are answered by new calls on the same photos, in the same batches of 12 (proploop.answer_batches on the same set),
with proploop's answers prompt holding only the standing questions. Each photo's answer row is its round-25 row
followed by its standing row. proploop's statistics (eta2 and d per question as percentiles among the layer's
channels, the two halves, the within-class null), its tables and its revisions of each explanation (real and null)
are then redone on all questions. Before any call the run stops if a round-25 cache is missing, a round-25 batch is
recorded as failed, or a batch's photos differ from proploop.answer_batches. Outputs: runs/proploop2/.

Decision printed in the report, per question (own and standing): the unit's separation (eta2) percentile among its
layer's channels, > 95 -> ABOVE_95, < 5 -> BELOW_5, else WITHIN_5_95; NO_STATISTICS when fewer than 2 answers have 5
photos (proploop.decide).

usage: python -I tools/proploop2.py [--units-file authors/round25_units.json] [--selftest]
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
import proploop  # noqa: E402
import run_subjects  # noqa: E402

OUT = J / 'runs/proploop2'
R25 = proploop.OUT  # round 25: proploop's caches, read only
STANDING = J / 'authors/standing.reply.json'
UNITS_FILE = proploop.UNITS_FILE
WORKERS = proploop.WORKERS  # 3: proploop.cached_calls runs at most this many calls at once
R25_FILES = ('questions.json', 'answers.json')
MARK = ' (standing)'


# ---------- inputs ----------

def _no_call(_sp, _turns):
    raise SystemExit('a call for round-25 questions was about to be made; stopping')


def _no_model(_dev):
    raise SystemExit('a round-25 value cache is missing; stopping')


def load_standing(path=STANDING):
    """the 'questions' of path: a list in proploop.parse_questions's format (what it keeps from a reply holding exactly
    that list); stops otherwise"""
    qs = json.loads(Path(path).read_text()).get('questions')
    if not isinstance(qs, list) or proploop.parse_questions(json.dumps({'questions': qs})) != qs:
        raise SystemExit(f'{path}: "questions" is not a list in proploop.parse_questions\'s format; stopping')
    return qs


def check_caches(units, r25=R25):
    """stops, naming them, when any of proploop's caches the units need is missing: <unit>/questions.json,
    <unit>/answers.json and values_<layer>.npy in r25"""
    layers = sorted({u['layer'] for u in units}, key=proploop.LAYERS.index)
    need = [r25 / u['name'] / fn for u in units for fn in R25_FILES] + [r25 / f'values_{L}.npy' for L in layers]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise SystemExit('missing round-25 caches: ' + ', '.join(missing) + '; stopping')


def load_values(layers, r25=R25):
    """proploop.layer_values from the caches values_<layer>.npy in r25 (shape and dtype checked there); stops when one
    is missing, so no model is ever loaded"""
    missing = [str(r25 / f'values_{L}.npy') for L in layers if not (r25 / f'values_{L}.npy').exists()]
    if missing:
        raise SystemExit('missing round-25 value caches: ' + ', '.join(missing) + '; stopping')
    return proploop.layer_values(layers, _no_model, np.arange(proploop.N_PHOTOS), out=r25)


def batch_keys(batches):
    """the cache keys of the answer batches, as proploop: '1', '2', ..."""
    return [str(b) for b in range(1, len(batches) + 1)]


def prepare_unit(u, V, labels, r25=R25, src=proploop.RANGEVIEW):
    """one unit's inputs, all checked before any call (u from proploop.parse_unit; V: its layer's values, photos x
    channels): the explanations, the photo set and R as proploop; the kept questions of r25/<unit>/questions.json
    (checked against the explanations by proploop.get_questions, which makes no call on a cache); the batches of
    r25/<unit>/answers.json. Stops when a file, an explanation, the questions or a batch is missing, a batch is not
    expected, a batch is recorded as failed, a batch's photos differ from proploop.answer_batches on the set, or a
    batch does not hold one row per photo with one answer per question"""
    name, c = u['name'], u['chan']
    rdir = r25 / name
    for fn in R25_FILES:
        if not (rdir / fn).exists():
            raise SystemExit(f'{rdir / fn}: missing; stopping')
    expls = proploop.source_explanations(name, src)
    if expls is None:
        raise SystemExit(f'{name}: no explanation for each of {", ".join(proploop.SOURCE_KEYS)} in {src}; stopping')
    own = proploop.get_questions(name, expls, rdir, call=_no_call)['questions']
    if not own:
        raise SystemExit(f'{rdir / "questions.json"}: no kept questions; stopping')
    V, labels = np.asarray(V), np.asarray(labels)
    classes, set_ids = proploop.photo_set(V[:, c], labels)
    set_labels = labels[set_ids]
    batches = proploop.answer_batches(name, set_ids)
    keys = batch_keys(batches)
    path = rdir / 'answers.json'
    ac = json.loads(path.read_text())
    missing, extra = [k for k in keys if k not in ac], sorted(k for k in ac if k not in keys)
    if missing or extra:
        raise SystemExit(f'{path}: batches missing {missing}, batches not expected {extra}; stopping')
    failed = [k for k in keys if ac[k].get('failed') is not False]
    if failed:
        raise SystemExit(f'{path}: batches {failed} recorded as failed; stopping')
    for k, batch in zip(keys, batches):
        if ac[k].get('photos') != batch:
            raise SystemExit(f'{path}: batch {k} photos differ from proploop.answer_batches on the set; stopping')
        rows = ac[k].get('answers')
        if (not isinstance(rows, list) or len(rows) != len(batch)
                or any(not isinstance(r, list) or len(r) != len(own) for r in rows)):
            raise SystemExit(f'{path}: batch {k} does not hold one answer per question for each photo; stopping')
    return {'unit': u, 'name': name, 'chan': c, 'layer': u['layer'], 'm': V.shape[1], 'n': len(set_ids),
            'classes': classes, 'set_ids': set_ids, 'set_labels': set_labels,
            'R': proploop.centre(V[set_ids], set_labels), 'expls': expls, 'own': own, 'batches': batches,
            'own_batches': [(ac[k]['photos'], ac[k]['answers']) for k in keys]}


# ---------- questions and rows ----------

def merge_rows(rows_a, rows_b):
    """each row of rows_a followed by the row of rows_b at the same place (new lists); ValueError when the two lists
    differ in length"""
    if len(rows_a) != len(rows_b):
        raise ValueError(f'{len(rows_a)} rows and {len(rows_b)} rows')
    return [list(a) + list(b) for a, b in zip(rows_a, rows_b)]


def all_questions(own, standing):
    """own followed by standing, as a new list"""
    return list(own) + list(standing)


def rows_in_set_order(batches, set_ids, nq):
    """one answer row (a new list) per photo of set_ids, in set order. batches: [(photo ids, parsed rows, one per
    photo, or None for a failed batch)]; a failed batch's photos get rows of nq Nones. ValueError when a photo of the
    set is in no batch, a photo is in the batches more than once or is repeated in set_ids, a batch photo is not in the
    set, or a batch has a different number of rows than photos"""
    row_of = {}
    for photos, ans in batches:
        photos = [int(i) for i in photos]
        if ans is None:
            ans = [[None] * nq for _ in photos]
        elif len(ans) != len(photos):
            raise ValueError(f'a batch of {len(photos)} photos has {len(ans)} answer rows')
        for i, row in zip(photos, ans):
            if i in row_of:
                raise ValueError(f'photo {i} is repeated in the batches')
            row_of[i] = list(row)
    ids = [int(i) for i in set_ids]
    if len(set(ids)) != len(ids):
        raise ValueError('a photo is repeated in set_ids')
    missing = [i for i in ids if i not in row_of]
    if missing:
        raise ValueError(f'photos {missing[:10]} of the set are in no batch')
    extra = sorted(set(row_of) - set(ids))
    if extra:
        raise ValueError(f'photos {extra[:10]} of the batches are not in the set')
    return [row_of[i] for i in ids]


def standing_jobs(batches, standing, block=None):
    """proploop.cached_calls jobs for the standing answers, keyed by batch number: per batch, proploop's answers
    prompt with only the standing questions, its parse and its cached fields"""
    nq = len(standing)
    return [(key, proploop.answers_content(batch, standing, block), proploop._answer_parser(len(batch), standing),
             proploop._answer_fields(batch, nq)) for key, batch in zip(batch_keys(batches), batches)]


# ---------- one unit, report ----------

def mark_standing(sec, questions, n_own):
    """sec (proploop.section on questions) with MARK after the number of each question from n_own + 1 on:
    'Q9. text' -> 'Q9 (standing). text'"""
    for j in range(n_own, len(questions)):
        q = questions[j]['question']
        old, new = f'\nQ{j + 1}. {q}\n', f'\nQ{j + 1}{MARK}. {q}\n'
        assert sec.count(old) == 1, (j + 1, sec.count(old))
        sec = sec.replace(old, new)
    return sec


def run_unit(p, standing, names, cats, call=None, out=OUT, block=None):
    """one unit (p from prepare_unit): the standing answers (new calls, cached in out/<unit>/answers.json), the merged
    rows, statistics on all questions (stats.json), the real and null tables, revisions (revisions.json), each as
    proploop.run_unit. -> (report section, the revisions cache, or None when the standing answers are not all cached
    yet)"""
    name, c, batches = p['name'], p['chan'], p['batches']
    udir = out / name
    own, keys = p['own'], batch_keys(batches)

    # standing answers
    ac = proploop.cached_calls(name, udir / 'answers.json', standing_jobs(batches, standing, block), call)
    missing = [k for k in keys if k not in ac]
    if missing:
        return proploop.failed_section(name, f'standing answers incomplete: batches {missing} not cached (call '
                                             'errors); rerun'), None
    for key, batch in zip(keys, batches):
        if ac[key]['photos'] != batch or len(ac[key]['answers']) != len(batch):
            raise SystemExit(f'{udir / "answers.json"}: batch {key} is for other photos; stopping')
    failed = [k for k in keys if ac[k]['failed']]

    # rows: round 25 then standing, per photo of the set
    set_ids = p['set_ids']
    rows = merge_rows(rows_in_set_order(p['own_batches'], set_ids, len(own)),
                      rows_in_set_order([(ac[k]['photos'], None if ac[k]['failed'] else ac[k]['answers'])
                                         for k in keys], set_ids, len(standing)))
    qs = all_questions(own, standing)
    nq = len(qs)
    assert all(len(r) == nq for r in rows), nq

    # statistics, real and null, as proploop
    R, set_labels, classes = p['R'], p['set_labels'], p['classes']
    null_rows = proploop.permute_within(rows, set_labels, classes, np.random.default_rng(zlib.crc32(name.encode())))
    half = np.array([zlib.crc32(names[i].encode()) % 2 for i in set_ids.tolist()], dtype=np.int64)
    st = {t: [proploop.question_record(R, c, [r[q] for r in rr], len(qs[q]['answers']), half) for q in range(nq)]
          for t, rr in zip(proploop.TABLES, (rows, null_rows))}
    cnames = [cats[x] for x in classes]
    tables = {t: proploop.table_text(cnames, p['n'], p['m'], qs, st[t]) for t in proploop.TABLES}
    proploop.save_json({'unit': name, 'layer': p['layer'], 'chan': c, 'm': p['m'], 'n': p['n'], 'classes': classes,
                        'class_names': cnames, 'set_ids': set_ids.tolist(), 'answer_batches': len(batches),
                        'failed_batches': failed,
                        'questions': [{'question': q['question'], 'answers': q['answers'], 'standing': j >= len(own),
                                       'real': st['real'][j], 'null': st['null'][j]} for j, q in enumerate(qs)],
                        'tables': tables}, udir / 'stats.json')

    # revisions
    rjobs = [(f'{t}_s{s}', proploop.revision_content(p['expls'][s - 1], tables[t]), bundlemax.parse_expl,
              proploop._revision_fields) for t in proploop.TABLES for s in proploop.SAMPLES]
    rev = proploop.cached_calls(name, udir / 'revisions.json', rjobs, call)
    sec = proploop.section(name, p['layer'], p['m'], p['n'], classes, cats, qs, st, len(failed), len(batches), rev)
    return mark_standing(sec, qs, len(own)), rev


# ---------- selftest ----------

def selftest():
    import tempfile
    import threading

    own_q = [{'question': 'Is there a dog?', 'answers': ['yes', 'no']},
             {'question': 'Where is it?', 'answers': ['indoors', 'outdoors', 'not applicable']}]
    std_q = [{'question': 'Which way is the main subject facing?', 'answers': ['left', 'right', 'neither']},
             {'question': 'How sharp is the main subject?', 'answers': ['sharp', 'blurry']},
             {'question': 'Is the photo tilted?', 'answers': ['level', 'tilted']}]
    assert WORKERS == proploop.WORKERS == 3 and OUT == J / 'runs/proploop2' and R25 == J / 'runs/proploop'
    assert proploop.REVISIONS == ['real_s1', 'real_s2', 'real_s3', 'null_s1', 'null_s2', 'null_s3']

    # all_questions: own then standing, a new list, inputs unchanged
    aq = all_questions(own_q, std_q)
    assert aq == own_q + std_q and aq is not own_q and aq is not std_q
    aq.append('x')
    assert len(own_q) == 2 and len(std_q) == 3
    assert all_questions([], std_q) == std_q and all_questions(own_q, []) == own_q
    assert all_questions(tuple(own_q), std_q) == own_q + std_q

    # merge_rows: concatenated rows, new lists, a length mismatch is an error
    ra, rb = [[0, 1], [None, 2]], [[2], [None]]
    mr = merge_rows(ra, rb)
    assert mr == [[0, 1, 2], [None, 2, None]] and ra == [[0, 1], [None, 2]] and rb == [[2], [None]]
    mr[0].append(7)
    assert ra[0] == [0, 1] and merge_rows([], []) == []
    for a_, b_ in [(ra, rb[:1]), (ra[:1], rb), ([], rb)]:
        try:
            merge_rows(a_, b_)
        except ValueError:
            continue
        raise AssertionError('rows of different lengths accepted')

    # rows_in_set_order: set order, a failed batch, a missing, repeated or extra photo
    sid = [5, 3, 9, 7, 1]
    bts = [([9, 5], [[1, 0], [0, 1]]), ([1, 3, 7], [[2, 2], [0, None], [1, 1]])]
    want_r = [[0, 1], [0, None], [1, 0], [1, 1], [2, 2]]
    assert rows_in_set_order(bts, sid, 2) == want_r
    assert rows_in_set_order(bts, np.array(sid, dtype=np.int64), 2) == want_r
    assert rows_in_set_order(list(reversed(bts)), sid, 2) == want_r
    assert rows_in_set_order([(np.array([9, 5]), bts[0][1]), bts[1]], sid, 2) == want_r
    got_r = rows_in_set_order(bts, sid, 2)
    got_r[0].append(9)
    assert bts[0][1][1] == [0, 1]  # rows are new lists
    fl = rows_in_set_order([bts[0], ([1, 3, 7], None)], sid, 2)
    assert fl == [[0, 1], [None, None], [1, 0], [None, None], [None, None]]
    assert fl[1] is not fl[3] and fl[3] is not fl[4]
    assert rows_in_set_order([([9, 5], None), ([1, 3, 7], None)], sid, 4) == [[None] * 4] * 5
    for bad_b, bad_s in [(bts, sid + [4]),  # a photo of the set in no batch
                         (bts[:1], sid),  # a whole batch missing
                         (bts + [([3], [[0, 0]])], sid),  # a photo in two batches
                         ([([9, 9, 5], [[1, 0], [1, 0], [0, 1]]), bts[1]], sid),  # twice in one batch
                         (bts, sid + [5]),  # repeated in the set
                         (bts, [5, 3, 9, 7]),  # a batch photo not in the set
                         ([([9, 5], [[1, 0]]), bts[1]], sid)]:  # fewer rows than photos
        try:
            rows_in_set_order(bad_b, bad_s, 2)
        except ValueError:
            continue
        raise AssertionError(f'accepted {bad_b} on {bad_s}')

    # the standing answers prompt holds only the standing questions; batches, parse and fields as proploop
    sid2 = np.array([3, 8, 9, 20, 21, 40] + list(range(50, 70)))
    bt2 = proploop.answer_batches('mixed4c:447', sid2)
    assert [len(b) for b in bt2] == [12, 12, 2]

    def blk(i):
        return {'type': 'image', 'id': int(i)}
    jobs = standing_jobs(bt2, std_q, block=blk)
    assert [j[0] for j in jobs] == ['1', '2', '3']
    for (key, content, parse, fields), batch in zip(jobs, bt2):
        texts = [x['text'] for x in content if x['type'] == 'text']
        assert content == proploop.answers_content(batch, std_q, block=blk)
        assert [x['id'] for x in content if x['type'] == 'image'] == batch
        assert texts[:-1] == [f'Image {k}:' for k in range(1, len(batch) + 1)]
        assert texts[-1] == proploop.fill(proploop.A_PROMPT, b=len(batch), questions=proploop.questions_block(std_q))
        assert all(q['question'] in texts[-1] for q in std_q)
        assert not any(q['question'] in t for q in own_q for t in texts)
        assert f'Q{len(std_q)}. ' in texts[-1] and f'Q{len(std_q) + 1}. ' not in texts[-1]
        ids_ = range(1, len(batch) + 1)
        one = json.dumps({'images': [{'id': k, 'answers': ['A'] * len(std_q)} for k in ids_]})
        both = json.dumps({'images': [{'id': k, 'answers': ['A'] * (len(own_q) + len(std_q))} for k in ids_]})
        assert parse(one) == [[0] * len(std_q) for _ in batch] and parse(both) is None
        assert fields(None) == {'photos': batch, 'answers': [[None] * len(std_q) for _ in batch], 'failed': True}

    # the report mark
    sec_t = '\n'.join(['== u', 'classes: a (1)', 'Q1. Is there a dog?', '   yes: 5 photos', 'Q2. Which way?',
                       '   left: 5 photos', 'separation percentile above 95: real 0', 'real_s1  one  x'])
    qm = [{'question': 'Is there a dog?'}, {'question': 'Which way?'}]
    assert mark_standing(sec_t, qm, 1) == sec_t.replace('\nQ2. Which way?\n', '\nQ2 (standing). Which way?\n')
    assert mark_standing(sec_t, qm, 2) == sec_t
    assert mark_standing(sec_t, qm, 0).count(' (standing). ') == 2

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # the standing questions file: kept only when in proploop.parse_questions's format
        sp_ = td / 'standing.json'
        proploop.save_json({'prompt': 'p', 'reply': 'r', 'questions': std_q}, sp_)
        assert load_standing(sp_) == std_q
        for bad in [std_q[:2], [dict(std_q[0], answers=['left'])] + std_q[1:],
                    [dict(std_q[0], question=' padded ')] + std_q[1:], 'x', None]:
            proploop.save_json({'questions': bad}, sp_)
            try:
                load_standing(sp_)
            except SystemExit:
                continue
            raise AssertionError(f'standing questions accepted: {bad}')

        # one unit end to end on synthetic round-25 caches, with a stand-in for the agents (no model, no Claude, no
        # photos): stops before any call, then the merged statistics
        rng = np.random.default_rng(4)
        Vs = rng.random((600, 20)).astype(np.float32)
        labs = rng.permutation(np.repeat(np.arange(50), 12))
        nms = [f'img/{i}.jpeg' for i in range(600)]
        cats_t = [f'class {i}' for i in range(50)]
        name_t, cu = 'mixed4c:3', 3
        uu = proploop.parse_unit(name_t)
        expls = ['expl one', 'expl two', 'expl three']
        proploop.save_json({k: {'explanation': e} for k, e in zip(proploop.SOURCE_KEYS, expls)},
                           td / 'src' / name_t / 'readings.json')
        rdir = td / 'r25' / name_t
        proploop.save_json({'unit': name_t, 'content_sha256': proploop.sha256(proploop.questions_content(expls)),
                            'questions': own_q, 'valid': True}, rdir / 'questions.json')
        cls_t, sid_t = proploop.photo_set(Vs[:, cu], labs)
        bt_t = proploop.answer_batches(name_t, sid_t)
        assert len(sid_t) == 144 and len(bt_t) == 12

        def own_row(i):
            return [i % 2, (i // 3) % 3]
        good = {str(b): {'key': str(b), 'photos': batch, 'answers': [own_row(i) for i in batch], 'failed': False}
                for b, batch in enumerate(bt_t, 1)}

        def prep(ans):
            proploop.save_json(ans, rdir / 'answers.json')
            return prepare_unit(uu, Vs, labs, r25=td / 'r25', src=td / 'src')
        for bad in [{k: v for k, v in good.items() if k != '3'},  # a batch missing
                    dict(good, **{'13': good['1']}),  # a batch not expected
                    dict(good, **{'2': dict(good['2'], failed=True)}),  # recorded as failed
                    dict(good, **{'1': dict(good['1'], photos=good['1']['photos'][::-1])}),  # other photos
                    dict(good, **{'4': dict(good['4'], answers=[[0]] * len(bt_t[3]))})]:  # too few answers per row
            try:
                prep(bad)
            except SystemExit:
                continue
            raise AssertionError('round-25 answers accepted: ' + ', '.join(sorted(bad, key=int)))
        pu = prep(good)
        assert pu['own'] == own_q and pu['batches'] == bt_t and np.array_equal(pu['set_ids'], sid_t)
        assert pu['classes'] == cls_t and (pu['m'], pu['n']) == (20, 144) and pu['expls'] == expls
        own_rows = [own_row(i) for i in sid_t.tolist()]
        assert rows_in_set_order(pu['own_batches'], pu['set_ids'], len(own_q)) == own_rows

        R_t = proploop.centre(Vs[sid_t], labs[sid_t])
        r_of = dict(zip(sid_t.tolist(), R_t[:, cu].tolist()))
        lock, state = threading.Lock(), {'n': 0, 'err': True}

        def call_u(sp, turns):
            assert sp == '' and len(turns) == 1
            content = turns[0]
            texts = [b['text'] for b in content if b['type'] == 'text']
            with lock:
                state['n'] += 1
            if texts[-1] == run_subjects.TASK:
                assert len(content) == 2 and texts[0].startswith('Here is an explanation of what one unit')
                return [{'text': json.dumps({'class': 'one', 'explanation': 'revised'}), 'is_error': False}]
            ids = [b['id'] for b in content if b['type'] == 'image']
            assert texts[-1] == proploop.fill(proploop.A_PROMPT, b=len(ids), questions=proploop.questions_block(std_q))
            if ids[0] == bt_t[1][0] and state['err']:
                return [{'text': 'overloaded', 'is_error': True}]
            if ids[0] == bt_t[0][0]:
                return [{'text': 'sorry', 'is_error': False}]
            ent = [{'id': k, 'answers': ['ABC'[i % 3], 'A' if r_of[i] > 0 else 'b', '?' if i % 7 == 0 else 'B']}
                   for k, i in enumerate(ids, 1)]
            return [{'text': json.dumps({'images': ent}), 'is_error': False}]
        kw = dict(call=call_u, out=td / 'out', block=blk)
        ud = td / 'out' / name_t

        # an is_error reply: not cached, no statistics or revisions yet
        sec0, rev0 = run_unit(pu, std_q, nms, cats_t, **kw)
        assert rev0 is None and sec0.startswith(f'== {name_t}\nfailed: standing answers incomplete')
        assert state['n'] == 13 and not (ud / 'stats.json').exists()  # batch 1 retried, batch 2 is_error

        state.update(n=0, err=False)
        sec, rev = run_unit(pu, std_q, nms, cats_t, **kw)
        assert state['n'] == 1 + 6, state  # batch 2, 6 revisions
        assert sorted(rev) == sorted(proploop.REVISIONS) and rev['null_s2']['explanation'] == 'revised'
        lines = sec.splitlines()
        assert lines[0] == (f'== {name_t}  layer mixed4c (20 channels), 144 photos, answer batches failed after a '
                            'retry: 1 of 12'), lines[0]
        assert [x for x in lines if x.startswith('Q')] == [
            'Q1. Is there a dog?', 'Q2. Where is it?', 'Q3 (standing). Which way is the main subject facing?',
            'Q4 (standing). How sharp is the main subject?', 'Q5 (standing). Is the photo tilted?']
        assert 'real_s1  one  revised' in lines and lines[-1] == 'null_s3  one  revised'

        st_t = json.loads((ud / 'stats.json').read_text())
        assert [q['question'] for q in st_t['questions']] == [q['question'] for q in own_q + std_q]
        assert [q['standing'] for q in st_t['questions']] == [False, False, True, True, True]
        assert st_t['failed_batches'] == ['1'] and st_t['answer_batches'] == 12 and st_t['set_ids'] == sid_t.tolist()
        assert st_t['classes'] == cls_t and st_t['n'] == 144 and st_t['m'] == 20
        # the round-25 questions' numbers are proploop's on their own rows (the null permutes whole rows)
        half_t = np.array([zlib.crc32(nms[i].encode()) % 2 for i in sid_t.tolist()], dtype=np.int64)
        null_own = proploop.permute_within(own_rows, labs[sid_t], cls_t,
                                           np.random.default_rng(zlib.crc32(name_t.encode())))
        for t, rr in zip(proploop.TABLES, (own_rows, null_own)):
            for q in range(len(own_q)):
                want_q = proploop.question_record(R_t, cu, [r[q] for r in rr], len(own_q[q]['answers']), half_t)
                assert st_t['questions'][q][t] == json.loads(json.dumps(want_q)), (t, q)
        assert st_t['questions'][0]['real']['eta2_pct'] is not None
        q3 = st_t['questions'][2]['real']
        assert q3['unclear'] == 12 and sum(q3['counts']) == 132  # batch 1 failed: its 12 photos have no answer
        for q in st_t['questions']:
            assert q['null']['counts'] == q['real']['counts'] and q['null']['unclear'] == q['real']['unclear']
        for t in proploop.TABLES:
            assert '(standing)' not in st_t['tables'][t]
            assert '\n\nQ3. Which way is the main subject facing?\n' in st_t['tables'][t]
            assert st_t['tables'][t].startswith('# Checks on ordinary photos\n\n')
        an = json.loads((ud / 'answers.json').read_text())
        assert sorted(an, key=int) == batch_keys(bt_t) and an['1']['failed'] and not an['2']['failed']
        assert all(an[k]['photos'] == b and all(len(r) == len(std_q) for r in an[k]['answers'])
                   for k, b in zip(batch_keys(bt_t), bt_t))
        assert json.loads((rdir / 'answers.json').read_text()) == good  # round 25 is only read

        # a rerun redoes nothing
        state['n'] = 0
        assert run_unit(pu, std_q, nms, cats_t, **kw)[0] == sec and state['n'] == 0

        # a missing round-25 file stops
        (rdir / 'questions.json').unlink()
        try:
            prep(good)
            raise AssertionError('missing questions.json accepted')
        except SystemExit:
            pass
        try:
            check_caches([uu], r25=td / 'r25')
            raise AssertionError('missing caches accepted')
        except SystemExit:
            pass
    print('selftest ok')


# ---------- main ----------

def main():
    if '--selftest' in sys.argv[1:]:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--units-file', default=UNITS_FILE, help='relative paths are taken from the repo root')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    uf = Path(a.units_file)
    units = [proploop.parse_unit(s) for s in json.loads((uf if uf.is_absolute() else J / uf).read_text())]
    standing = load_standing()
    names = json.loads(proploop.NAMES.read_text())
    with np.load(proploop.ACTS) as z:
        labels = np.asarray(z['classes'], dtype=np.int64)
    assert len(names) == len(labels) == proploop.N_PHOTOS, (len(names), len(labels))
    assert labels.min() >= 0 and labels.max() < 1000, (labels.min(), labels.max())
    from torchvision.models import GoogLeNet_Weights
    cats = list(GoogLeNet_Weights.IMAGENET1K_V1.meta['categories'])
    assert len(cats) == 1000, len(cats)

    # every round-25 input of every unit, checked before any call
    check_caches(units)
    layers = sorted({u['layer'] for u in units}, key=proploop.LAYERS.index)
    vals = load_values(layers)
    preps = [prepare_unit(u, vals[u['layer']], labels) for u in units]
    del vals
    gc.collect()
    print(f'{len(preps)} units: round-25 questions, answers and values present, no failed batch, batches as '
          f'proploop.answer_batches; {len(standing)} standing questions', flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    secs, allr = [], {}
    for p in preps:
        n = p['name']
        print('==', n, flush=True)
        sec, rev = run_unit(p, standing, names, cats)
        rev = rev or {}
        allr[n] = {rid: (rev[rid]['explanation'] if rid in rev else None) for rid in proploop.REVISIONS}
        secs.append(sec)
        proploop.save_json(allr, OUT / 'readings_all.json')
        (OUT / 'report.txt').write_text('\n\n'.join(secs) + '\n')
        print(sec, flush=True)
        gc.collect()
    print('report ->', OUT / 'report.txt')


if __name__ == '__main__':
    main()
