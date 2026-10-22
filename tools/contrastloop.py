"""Predictions from contrasts, checked blind on other photos (spec: authors/spec_contrastloop.md).

Per unit (one channel of mixed3a-mixed5b): the photos of the 24 classes with the highest class mean of the unit's
spatial max are split by crc32(photo name) % 2 into looked-at photos (0) and checked photos (1). Three agents see the
same looked-at photos: for each of the unit's top 4 classes, its photos from most to least active; for each of the 3
other channels of its layer whose class means correlate most with the unit's, the photos on which the two differ most
after removing each one's class mean. Each writes an explanation and up to 4 predictions: a question about a photo, its
answers, and the answers on whose photos the unit will be most and least active within a class. A second agent, which
never sees any activity, answers the questions for every checked photo, 12 per call (as tools/proploop.py). Per
prediction, d = (mean over the most-active answer's photos - mean over the least-active answer's) / their sd, on the
values scaled and centred within class, for every channel of the layer on the same photos and answers, with the unit's
percentile among them; also after permuting the answers within class (null). Each explanation is then read again by an
agent shown its predictions' real outcomes, and by one shown the null outcomes. Outputs: runs/contrastloop/.

Decision printed in the report, per prediction: the unit's d percentile among its layer's channels >= 95 and its d
> 0 -> held, else not held; too few when either compared answer has fewer than 5 photos.

usage: python -I tools/contrastloop.py [--units-file authors/round27_units.json] [--selftest]
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

OUT = J / 'runs/contrastloop'
UNITS_FILE = 'authors/round27_units.json'
K_CLASSES = 24
N_RANGE, RANGE_CAP = 4, 6  # range panel: classes shown, photos per class at most
N_PEERS, PER_SIDE = 3, 2  # peer panel: similar channels, photos on each side
SAMPLES = (1, 2, 3)
LOOKS = [f's{s}' for s in SAMPLES]
MAX_PRED = 4
MIN_ANS, MAX_ANS = proploop.MIN_ANS, proploop.MAX_ANS
MIN_N = 5  # photos each compared answer needs
PCT_HELD = 95.0
OUTCOMES = ('held', 'not held', 'too few')
TABLES = proploop.TABLES
READINGS = [f'{t}_{s}' for t in TABLES for s in LOOKS]
WORKERS = proploop.WORKERS  # calls in flight, through proploop.cached_calls

LOOK_HEAD = (
    'Below are photos with the activity of one unit (a single neuron) of an image classification network on each, as '
    'a percentage of the most it reaches on any of 10000 photos.\n\n'
    'Part 1. For each of the 4 photo classes on which the unit is most active on average, photos of that class, from '
    'most to least active.')
PEER_HEAD = (
    'Part 2. Similar units: the 3 other units of the same layer whose average activity over photo classes is most '
    "like this unit's. For each, photos on which this unit is much more active than the similar unit, and photos on "
    "which it is much less active, each compared with their own averages on the photo's class.")
LOOK_TASK = (
    'Whatever separates the more active from the less active photos of one class cannot be the class itself, and '
    'whatever separates this unit from a similar unit is what is particular to it. From these photos, write hypotheses '
    "about what this unit's activity depends on, each as a prediction to be checked on new photos of the same classes: "
    "someone who never sees the unit's activity will answer a question about each new photo, and the unit's activity "
    'will be compared between photos given different answers, within the same class. For each prediction give the '
    'question, which must be answerable by looking at a small photo; 2 to 5 short answers that do not overlap, so that '
    'every photo fits exactly one of them, including one for photos to which the question does not apply; the answer '
    'on whose photos you predict the unit is most active; and the answer on whose photos you predict it is least '
    "active. Write up to 4 predictions: the ones you expect to separate the unit's activity most clearly, including "
    'from the similar units. Also write a short explanation of what the unit responds to. Reply with only JSON: '
    '{"explanation": "...", "predictions": [{"question": "...", "answers": ["...", "..."], "most_active": "...", '
    '"least_active": "..."}]}')
RESULTS_HEAD = (
    'The predictions were then checked on new photos: {n} photos of the {K} photo classes on which the unit is most '
    "active on average, none of them among the photos looked at. Someone who never saw the unit's activity answered "
    'each question about each photo. For each prediction: the number of photos given each of the two answers it '
    'compares, and d, how much more active the unit is on the photos given the predicted most-active answer than on '
    'those given the predicted least-active answer, after removing the differences between classes, in standard '
    "deviations, with this unit's percentile among all {m} units of its layer on the same photos and answers. A "
    'prediction held when that percentile is at least 95.')
R_PROMPT = (
    'Here is an explanation of what one unit (a single neuron) of an image classification network responds to, '
    "written with the predictions below by someone who looked at photos of the unit's activity:\n\n"
    '{explanation}\n\n'
    '{results}\n\n'
    'Taking both into account, say what the unit responds to.')


# ---------- photos and panels ----------

def scaled(V):
    """S: each column of V divided by its max over the rows (all 0 when the max is 0), float64"""
    V = np.asarray(V, dtype=np.float64)
    assert V.ndim == 2, V.shape
    mx = V.max(axis=0)
    return np.where(mx != 0, V / np.where(mx != 0, mx, 1.0), 0.0)


def split(set_ids, names):
    """(D, T): the photos of set_ids (in index order) with crc32(name) % 2 == 0, and those with 1, each a list in
    order"""
    D, T = [], []
    for i in set_ids:
        i = int(i)
        (T if zlib.crc32(names[i].encode()) % 2 else D).append(i)
    return D, T


def range_panel(vals, labels, D, classes, k=N_RANGE, cap=RANGE_CAP):
    """per class of the first k of classes: its photos in D sorted by vals highest first (ties by lower photo id);
    when more than cap, the cap - cap // 2 highest, then the cap // 2 lowest (in that sorted order). vals and labels:
    over all photos, indexed by photo id"""
    v = np.asarray(vals, dtype=np.float64)
    lab = np.asarray(labels)
    assert v.ndim == 1 and v.shape == lab.shape, (v.shape, lab.shape)
    out = []
    for cl in list(classes)[:k]:
        ids = sorted((int(i) for i in D if lab[int(i)] == cl), key=lambda i: (-v[i], i))
        if len(ids) > cap:
            ids = ids[:cap - cap // 2] + ids[len(ids) - cap // 2:]
        out.append(ids)
    return out


def class_means(V, labels):
    """the mean of each column of V over the rows of each label present (labels in ascending order), labels x m"""
    V = np.asarray(V, dtype=np.float64)
    lab = np.asarray(labels)
    assert V.ndim == 2 and len(lab) == V.shape[0], (V.shape, lab.shape)
    return np.stack([V[lab == x].mean(axis=0) for x in np.unique(lab)])


def peers(V, labels, c, k=N_PEERS):
    """the k channels j != c whose class-mean vectors (class_means over all rows of V) have the highest Pearson
    correlation with c's, ties by lower j; a correlation that is undefined (either vector constant) counts as lowest
    and is given as None. -> [(channel, correlation)]"""
    M = class_means(V, labels)
    Z = M - M.mean(axis=0)
    ss = (Z ** 2).sum(axis=0)
    num = (Z * Z[:, c:c + 1]).sum(axis=0)
    flat = np.ptp(M, axis=0) == 0  # zero variance over the classes
    out = []
    for j in range(M.shape[1]):
        if j == c:
            continue
        out.append((j, None if flat[j] or flat[c] else float(num[j] / np.sqrt(ss[j] * ss[c]))))
    out.sort(key=lambda t: (1, 0.0, t[0]) if t[1] is None else (0, -t[1], t[0]))
    return out[:k]


def peer_photos(S, labels, D, c, p, n=PER_SIDE):
    """(more, less): Z = columns c and p of S on the photos of D, minus their mean over the photos of D with the same
    label; diff = Z_c - Z_p; more = the n photos with the highest diff, less = the n with the lowest, ties by lower
    photo id in both. S and labels: over all photos, indexed by photo id"""
    S = np.asarray(S, dtype=np.float64)
    ids = np.asarray(D, dtype=np.int64)
    lab = np.asarray(labels)[ids]
    X = S[ids][:, [c, p]]
    Z = np.empty_like(X)
    for x in np.unique(lab):
        on = lab == x
        Z[on] = X[on] - X[on].mean(axis=0)
    diff = Z[:, 0] - Z[:, 1]
    more = sorted(range(len(ids)), key=lambda t: (-diff[t], ids[t]))[:n]
    less = sorted(range(len(ids)), key=lambda t: (diff[t], ids[t]))[:n]
    return [int(ids[t]) for t in more], [int(ids[t]) for t in less]


# ---------- prompts and replies ----------

def _text(t):
    return {'type': 'text', 'text': t}


def looking_content(panel_a, panel_b, block=None):
    """the looking content: the head; per class j its name, then 'Photo j.k: pct%' and the photo (block(id), default
    run_subjects.img_block of its thumb) per photo; the peer head; per peer j 'Similar unit j', then per photo
    'Photo Sj.k (class name): this unit pct_c%, similar unit j pct_p%' and the photo; the task. panel_a: per class
    (class name, [(photo id, pct)]); panel_b: per peer [(photo id, class name, pct_c, pct_p)]"""
    block = block or (lambda i: run_subjects.img_block(proploop.thumb(i)))
    content = [_text(LOOK_HEAD)]
    for j, (cname, photos) in enumerate(panel_a, 1):
        content.append(_text(f'Class {j}: {cname}'))
        for k, (i, pct) in enumerate(photos, 1):
            content += [_text(f'Photo {j}.{k}: {proploop._whole(pct)}%'), block(i)]
    content.append(_text(PEER_HEAD))
    for j, photos in enumerate(panel_b, 1):
        content.append(_text(f'Similar unit {j}'))
        for k, (i, cname, pc, pp) in enumerate(photos, 1):
            content += [_text(f'Photo S{j}.{k} ({cname}): this unit {proploop._whole(pc)}%, similar unit {j} '
                              f'{proploop._whole(pp)}%'), block(i)]
    content.append(_text(LOOK_TASK))
    return content


def _answer_index(x, keys):
    """the index of the key equal to x stripped and casefolded; None when x is not a string or equals none"""
    if not isinstance(x, str):
        return None
    k = x.strip().casefold()
    return keys.index(k) if k in keys else None


def parse_predictions(text):
    """{'explanation', 'predictions': [{'question', 'answers', 'most', 'least'}]} of the last JSON object with
    'predictions': the explanation a string, non-empty after strip (kept as given); the kept predictions in order,
    objects with a question non-empty after strip, 2-5 answers, each a string non-empty after strip and distinct after
    strip and casefold, and most_active and least_active strings equal (stripped, casefolded) to two different answers;
    question and answers stored stripped, most and least as answer indices; the first MAX_PRED of them. None when the
    explanation is not valid, no prediction is kept or no object is found"""
    o = bundlemax.last_obj(text, ('predictions',))
    if o is None:
        return None
    e = o.get('explanation')
    if not isinstance(e, str) or not e.strip():
        return None
    kept = []
    for p in o['predictions'] if isinstance(o['predictions'], list) else []:
        if not isinstance(p, dict):
            continue
        q, a = p.get('question'), p.get('answers')
        if not isinstance(q, str) or not q.strip():
            continue
        if not isinstance(a, list) or not MIN_ANS <= len(a) <= MAX_ANS or not all(isinstance(x, str) for x in a):
            continue
        a = [x.strip() for x in a]
        keys = [x.casefold() for x in a]
        if not all(a) or len(set(keys)) != len(keys):
            continue
        hi, lo = _answer_index(p.get('most_active'), keys), _answer_index(p.get('least_active'), keys)
        if hi is None or lo is None or hi == lo:
            continue
        kept.append({'question': q.strip(), 'answers': a, 'most': hi, 'least': lo})
    kept = kept[:MAX_PRED]
    return {'explanation': e, 'predictions': kept} if kept else None


def reading_content(explanation, results):
    return [_text(proploop.fill(R_PROMPT, explanation=explanation, results=results)), _text(run_subjects.TASK)]


# ---------- checks ----------

def contrast(R, ans, hi, lo, min_n=MIN_N):
    """R n x m, ans a length-n list of int or None. n_hi, n_lo = the photos given answer hi, answer lo; None when
    either is below min_n, else (n_hi, n_lo, d), d per column = (mean over the hi photos - mean over the lo photos) /
    sd over both (ddof 1), 0 where that sd is 0"""
    R = np.asarray(R, dtype=np.float64)
    codes = proploop._codes(ans)
    assert R.ndim == 2 and len(codes) == R.shape[0], (R.shape, len(codes))
    on_hi, on_lo = codes == hi, codes == lo
    n_hi, n_lo = int(np.count_nonzero(on_hi)), int(np.count_nonzero(on_lo))
    if n_hi < min_n or n_lo < min_n:
        return None
    sd = R[on_hi | on_lo].std(axis=0, ddof=1)
    diff = R[on_hi].mean(axis=0) - R[on_lo].mean(axis=0)
    return n_hi, n_lo, np.where(sd > 0, diff / np.where(sd > 0, sd, 1.0), 0.0)


def outcome(d_unit, pct):
    """'held' when the percentile is at least 95 and the unit's d is above 0, else 'not held'"""
    return 'held' if pct >= PCT_HELD and d_unit > 0 else 'not held'


def prediction_check(R, c, ans, hi, lo):
    """the unit's (column c) record for one prediction: n_hi, n_lo, its d, its d percentile among the m columns and
    the outcome; d and pct None and the outcome 'too few' when not checkable"""
    codes = proploop._codes(ans)
    n_hi, n_lo = int(np.count_nonzero(codes == hi)), int(np.count_nonzero(codes == lo))
    got = contrast(R, ans, hi, lo)
    if got is None:
        return {'n_hi': n_hi, 'n_lo': n_lo, 'd': None, 'pct': None, 'outcome': 'too few'}
    d = got[2]
    p = proploop.percentile(d, c)
    return {'n_hi': n_hi, 'n_lo': n_lo, 'd': float(d[c]), 'pct': p, 'outcome': outcome(float(d[c]), p)}


def results_text(n, m, K, preds, res):
    """the results text: the head, then one block per prediction. preds: as parse_predictions; res: per prediction
    {'n_hi', 'n_lo', 'd', 'pct'}, d and pct None when it is not checkable"""
    assert len(preds) == len(res), (len(preds), len(res))
    blocks = [proploop.fill(RESULTS_HEAD, n=n, K=K, m=m)]
    for j, (p, r) in enumerate(zip(preds, res), 1):
        a = p['answers']
        L = [f'Prediction {j}. {p["question"]}', 'Answers: ' + '; '.join(a),
             f'Predicted most active: {a[p["most"]]}. Predicted least active: {a[p["least"]]}.']
        if r['d'] is None or r['pct'] is None:
            L.append(f'Result: {r["n_hi"]} and {r["n_lo"]} photos: too few to check (each needs at least 5).')
        else:
            word = 'held' if outcome(r['d'], r['pct']) == 'held' else 'did not hold'
            L.append(f'Result: {r["n_hi"]} and {r["n_lo"]} photos, d {float(r["d"]):+.2f}, percentile '
                     f'{proploop._whole(r["pct"])}: {word}.')
        blocks.append('\n'.join(L))
    return '\n\n'.join(blocks)


def tally(st):
    """{table: {outcome: number of predictions}}; st: {table: {sample: [records]}}"""
    return {t: {o: sum(r['outcome'] == o for recs in st[t].values() for r in recs) for o in OUTCOMES} for t in TABLES}


# ---------- calls and caches ----------

def _look_fields(p):
    """the cached fields of a looking sample: its explanation and predictions, both None when it is not valid"""
    return {'explanation': None if p is None else p['explanation'],
            'predictions': None if p is None else p['predictions']}


def check_answer_cache(path, batches):
    """stop when the answers cache at path has a batch whose photo list is not that batch's"""
    if not path.exists():
        return
    want = {str(b): batch for b, batch in enumerate(batches, 1)}
    for key, rec in json.loads(path.read_text()).items():
        if key not in want or rec.get('photos') != want[key]:
            raise SystemExit(f'{path}: cached batch {key} is for other photos; stopping')


# ---------- one unit, report ----------

def _corr(r):
    return '-' if r is None else f'{r:.3f}'


def result_line(t, r):
    if r['d'] is None:
        return f'      {t}: {r["n_hi"]} and {r["n_lo"]} photos -> {r["outcome"]}'
    return (f'      {t}: {r["n_hi"]} and {r["n_lo"]} photos, d {r["d"]:+.2f}, percentile {r["pct"]:.1f} -> '
            f'{r["outcome"]}')


def reading_line(rid, rec, valid):
    if not valid:
        return f'{rid}  no reading (no valid sample)'
    if rec is None:
        return f'{rid}  no reading (call failed, not cached)'
    if rec['explanation'] is None:
        return f'{rid}  no reading (reply not parsed after a retry)'
    return f'{rid}  {rec["class"]}  ' + ' '.join(rec['explanation'][:160].split())


def section(name, layer, m, n_looked, n, classes, cats, pr, lc, st, tal, n_fail, n_batches, rd):
    L = [f'== {name}  layer {layer} ({m} channels), {n_looked} photos looked at, {n} photos checked, answer batches '
         f'failed after a retry: {n_fail} of {n_batches}',
         'classes: ' + ', '.join(f'{cats[x]} ({x})' for x in classes),
         'peers: ' + ', '.join(f'{layer}:{p} (r {_corr(r)})' for p, r in pr)]
    for s in LOOKS:
        rec = lc[s]
        if not rec['valid']:
            L.append(f'{s}  no valid sample (reply not parsed after a retry)')
            continue
        L.append(f'{s}  ' + ' '.join(rec['explanation'][:160].split()))
        for j, p in enumerate(rec['predictions']):
            a = p['answers']
            L.append(f'   P{j + 1}. {p["question"]}  answers: ' + '; '.join(a)
                     + f'; most active: {a[p["most"]]}; least active: {a[p["least"]]}')
            L += [result_line(t, st[t][s][j]) for t in TABLES]
    L.append('outcomes: ' + '; '.join(f'{t} ' + ', '.join(f'{o} {tal[t][o]}' for o in OUTCOMES) for t in TABLES)
             + f' (of {sum(tal["real"].values())} predictions)')
    for t in TABLES:
        for s in LOOKS:
            rid = f'{t}_{s}'
            L.append(reading_line(rid, rd.get(rid), lc[s]['valid']))
    return '\n'.join(L)


def run_unit(u, V, labels, names, cats, call=None, out=OUT, block=None):
    """one unit (u from proploop.parse_unit; V: its layer's values, photos x channels): photos and panels
    (panels.json), looking, answers, checks real and null, readings, stats.json. -> (report section, the readings
    cache, or None when the unit failed or its calls are not all cached yet)"""
    name, layer, c = u['name'], u['layer'], u['chan']
    udir = out / name
    V, labels = np.asarray(V), np.asarray(labels)
    m = V.shape[1]
    S = scaled(V)

    def pct(i, j):
        return proploop._whole(100.0 * S[i, j])

    # photos and panels
    classes, set_ids = proploop.photo_set(V[:, c], labels, k=K_CLASSES)
    D, T = split(set_ids.tolist(), names)
    cnames = [cats[x] for x in classes]
    rp = range_panel(V[:, c], labels, D, classes)
    pr = peers(V, labels, c)
    pp = [peer_photos(S, labels, D, c, p) for p, _ in pr]
    panel_a = [(cats[cl], [(i, pct(i, c)) for i in ids]) for cl, ids in zip(classes, rp)]
    panel_b = [[(i, cats[int(labels[i])], pct(i, c), pct(i, p)) for i in more + less]
               for (p, _), (more, less) in zip(pr, pp)]
    peer_recs = [{'chan': p, 'unit': f'{layer}:{p}', 'corr': r} for p, r in pr]
    proploop.save_json({
        'unit': name, 'layer': layer, 'chan': c, 'classes': classes, 'class_names': cnames,
        'range': [{'class': cl, 'class_name': cn, 'photos': [i for i, _ in ph], 'pct': [x for _, x in ph]}
                  for cl, (cn, ph) in zip(classes, panel_a)],
        'peers': [{**pk, 'more': more, 'less': less, 'photos': [x[0] for x in ph], 'class_names': [x[1] for x in ph],
                   'pct_c': [x[2] for x in ph], 'pct_p': [x[3] for x in ph]}
                  for pk, (more, less), ph in zip(peer_recs, pp, panel_b)]}, udir / 'panels.json')

    # looking: three samples of the same content
    content = looking_content(panel_a, panel_b, block)
    lc = proploop.cached_calls(name, udir / 'looking.json',
                               [(s, content, parse_predictions, _look_fields) for s in LOOKS], call)
    del content
    missing = [s for s in LOOKS if s not in lc]
    if missing:
        return proploop.failed_section(name, f'looking incomplete: samples {missing} not cached (call errors); '
                                             'rerun'), None
    valid = [s for s in LOOKS if lc[s]['valid']]
    if not valid:
        return proploop.failed_section(name, 'no valid sample after one retry'), None

    # questions: the kept predictions of the valid samples, in sample order
    qs, where = [], {}
    for s in valid:
        where[s] = []
        for p in lc[s]['predictions']:
            where[s].append(len(qs))
            qs.append({'question': p['question'], 'answers': p['answers']})
    nq = len(qs)

    # answers on the checked photos
    batches = proploop.answer_batches(name, T)
    keys = [str(b) for b in range(1, len(batches) + 1)]
    apath = udir / 'answers.json'
    check_answer_cache(apath, batches)
    jobs = [(key, proploop.answers_content(batch, qs, block), proploop._answer_parser(len(batch), qs),
             proploop._answer_fields(batch, nq)) for key, batch in zip(keys, batches)]
    ac = proploop.cached_calls(name, apath, jobs, call)
    del jobs
    missing = [k for k in keys if k not in ac]
    if missing:
        return proploop.failed_section(name, f'answers incomplete: batches {missing} not cached (call errors); '
                                             'rerun'), None
    row_of = {}
    for key, batch in zip(keys, batches):
        if ac[key]['photos'] != batch or len(ac[key]['answers']) != len(batch):
            raise SystemExit(f'{apath}: batch {key} is for other photos; stopping')
        row_of.update(zip(batch, ac[key]['answers']))
    rows = [row_of[i] for i in T]
    failed = [k for k in keys if ac[k]['failed']]

    # checks, real and null
    T_ids = np.asarray(T, dtype=np.int64)
    T_labels = labels[T_ids]
    R = proploop.centre(V[T_ids], T_labels)
    null_rows = proploop.permute_within(rows, T_labels, classes, np.random.default_rng(zlib.crc32(name.encode())))
    st = {t: {s: [prediction_check(R, c, [r[q] for r in rr], p['most'], p['least'])
                  for q, p in zip(where[s], lc[s]['predictions'])] for s in valid}
          for t, rr in zip(TABLES, (rows, null_rows))}
    tal = tally(st)
    texts = {t: {s: results_text(len(T), m, K_CLASSES, lc[s]['predictions'], st[t][s]) for s in valid}
             for t in TABLES}

    # readings
    rjobs = [(f'{t}_{s}', reading_content(lc[s]['explanation'], texts[t][s]), bundlemax.parse_expl,
              proploop._revision_fields) for t in TABLES for s in valid]
    rd = proploop.cached_calls(name, udir / 'readings.json', rjobs, call)

    proploop.save_json({
        'unit': name, 'layer': layer, 'chan': c, 'm': m, 'n_looked': len(D), 'n': len(T), 'classes': classes,
        'class_names': cnames, 'peers': peer_recs, 'D': D, 'T': T, 'answer_batches': len(batches),
        'failed_batches': failed, 'questions': qs,
        'samples': {s: {'valid': lc[s]['valid'], 'explanation': lc[s]['explanation'],
                        'predictions': None if s not in where else
                        [{**p, 'question_index': q, 'real': st['real'][s][j], 'null': st['null'][s][j]}
                         for j, (q, p) in enumerate(zip(where[s], lc[s]['predictions']))]}
                    for s in LOOKS},
        'results': {f'{t}_{s}': texts[t][s] for t in TABLES for s in valid},
        'tally': tal,
        'readings': {rid: (None if rid not in rd else {'class': rd[rid]['class'],
                                                       'explanation': rd[rid]['explanation']})
                     for rid in READINGS}}, udir / 'stats.json')
    return section(name, layer, m, len(D), len(T), classes, cats, pr, lc, st, tal, len(failed), len(batches),
                   rd), rd


# ---------- selftest ----------

def selftest():
    import math
    import statistics

    def para(*pieces):
        return ' '.join(pieces)

    # prompt texts, verbatim from the spec: wrapped lines joined by single spaces, paragraph breaks kept
    assert LOOK_HEAD == '\n\n'.join([
        para('Below are photos with the activity of one unit (a single neuron) of an image classification network',
             'on each, as a percentage of the most it reaches on any of 10000 photos.'),
        para('Part 1. For each of the 4 photo classes on which the unit is most active on average, photos of that',
             'class, from most to least active.')])
    assert PEER_HEAD == para(
        'Part 2. Similar units: the 3 other units of the same layer whose average activity over photo classes is',
        "most like this unit's. For each, photos on which this unit is much more active than the similar unit, and",
        "photos on which it is much less active, each compared with their own averages on the photo's class.")
    assert LOOK_TASK == para(
        'Whatever separates the more active from the less active photos of one class cannot be the class itself,',
        'and whatever separates this unit from a similar unit is what is particular to it. From these photos, write',
        "hypotheses about what this unit's activity depends on, each as a prediction to be checked on new photos of",
        "the same classes: someone who never sees the unit's activity will answer a question about each new photo,",
        "and the unit's activity will be compared between photos given different answers, within the same class.",
        'For each prediction give the question, which must be answerable by looking at a small photo; 2 to 5 short',
        'answers that do not overlap, so that every photo fits exactly one of them, including one for photos to',
        'which the question does not apply; the answer on whose photos you predict the unit is most active; and the',
        'answer on whose photos you predict it is least active. Write up to 4 predictions: the ones you expect to',
        "separate the unit's activity most clearly, including from the similar units. Also write a short",
        'explanation of what the unit responds to. Reply with only JSON:',
        '{"explanation": "...", "predictions": [{"question": "...", "answers": ["...", "..."],',
        '"most_active": "...", "least_active": "..."}]}')
    assert RESULTS_HEAD == para(
        'The predictions were then checked on new photos: {n} photos of the {K} photo classes on which the unit is',
        "most active on average, none of them among the photos looked at. Someone who never saw the unit's activity",
        'answered each question about each photo. For each prediction: the number of photos given each of the two',
        'answers it compares, and d, how much more active the unit is on the photos given the predicted most-active',
        'answer than on those given the predicted least-active answer, after removing the differences between',
        "classes, in standard deviations, with this unit's percentile among all {m} units of its layer on the same",
        'photos and answers. A prediction held when that percentile is at least 95.')
    assert R_PROMPT == '\n\n'.join([
        para('Here is an explanation of what one unit (a single neuron) of an image classification network responds',
             "to, written with the predictions below by someone who looked at photos of the unit's activity:"),
        '{explanation}', '{results}', 'Taking both into account, say what the unit responds to.'])
    assert LOOKS == ['s1', 's2', 's3'] and WORKERS == 3 and K_CLASSES == 24
    assert READINGS == ['real_s1', 'real_s2', 'real_s3', 'null_s1', 'null_s2', 'null_s3']
    rc = reading_content('an {results} explanation', 'RESULTS')
    assert rc == [_text(R_PROMPT.replace('{results}', 'RESULTS').replace('{explanation}', 'an {results} explanation')),
                  _text(run_subjects.TASK)]

    # scaled: column max, a zero column stays 0
    Ss = scaled(np.array([[2, 0, 1], [4, 0, 2], [1, 0, 4]], dtype=np.float32))
    assert Ss.dtype == np.float64 and np.array_equal(Ss, [[0.5, 0, 0.25], [1, 0, 0.5], [0.25, 0, 1]]), Ss

    # split on hand names: crc32 '' 0, '123456789' cbf43926, 'abc' 352441c2 even; 'a' e8b7be43, the fox 414fa339 odd
    fox = 'The quick brown fox jumps over the lazy dog'
    assert zlib.crc32(b'') == 0 and zlib.crc32(b'123456789') == 0xCBF43926 and zlib.crc32(b'abc') == 0x352441C2
    assert zlib.crc32(b'a') == 0xE8B7BE43 and zlib.crc32(fox.encode()) == 0x414FA339
    nm_h = ['a', '123456789', 'zzz', 'abc', fox, '']
    assert split([0, 1, 3, 4, 5], nm_h) == ([1, 3, 5], [0, 4])
    assert split(np.array([0, 1, 3, 4, 5]), nm_h) == ([1, 3, 5], [0, 4]) and split([2], nm_h) in (([2], []), ([], [2]))
    nm_r = [f'img/{i}.jpeg' for i in range(300)]
    Dr, Tr = split(np.arange(0, 300, 2), nm_r)
    assert sorted(Dr + Tr) == list(range(0, 300, 2)) and Dr == sorted(Dr) and Tr == sorted(Tr) and Dr and Tr
    assert all(zlib.crc32(nm_r[i].encode()) % 2 == 0 for i in Dr) and all(zlib.crc32(nm_r[i].encode()) % 2 for i in Tr)
    assert all(type(i) is int for i in Dr + Tr)

    # range panel: highest first, ties by lower id, only photos of D, the cap (3 highest, then 3 lowest), k classes
    lab_r = np.array([7, 7, 7, 7, 7, 7, 7, 7, 3, 3, 3, 3, 9, 5, 5, 2, 2, 7, 3, 5])
    val_r = np.array([0.5, 0.9, 0.5, 0.1, 0.7, 0.3, 0.5, 0.2, 0.2, 0.4, 0.2, 0.0, 0.6, 0.6, 0.6, 0.1, 0.3, 1.0, 0.9,
                      0.8], dtype=np.float32)
    D_r = list(range(12)) + [13, 14, 15, 16]
    cls_r = [7, 3, 9, 5, 2]
    assert range_panel(val_r, lab_r, D_r, cls_r) == [[1, 4, 0, 5, 7, 3], [9, 8, 10, 11], [], [13, 14]]
    assert range_panel(val_r, lab_r, D_r, cls_r, k=5)[4] == [16, 15]
    assert range_panel(val_r, lab_r, D_r, cls_r, cap=8)[0] == [1, 4, 0, 2, 6, 5, 7, 3]  # 8 photos: not more than 8
    assert range_panel(val_r, lab_r, D_r, cls_r, cap=4)[:2] == [[1, 4, 7, 3], [9, 8, 10, 11]]
    assert range_panel(val_r, lab_r, D_r[::-1], cls_r) == range_panel(val_r, lab_r, D_r, cls_r)

    # peers: class means of 4 labels (2 photos each, +-0.5 around the mean), channel 3 constant over the classes,
    # channel 4 equal to channel 1
    lab_c = np.array([7, 2, 7, 9, 2, 4, 9, 4])
    cm = {2: [1, 2, 4, 5, 2, 1], 4: [2, 4, 3, 5, 4, 2], 7: [3, 6, 2, 5, 6, 4], 9: [4, 8, 1, 5, 8, 3]}
    seen, rows_c = set(), []
    for x in lab_c.tolist():
        rows_c.append([v + (-0.5 if x in seen else 0.5) for v in cm[x]])
        seen.add(x)
    Vc = np.array(rows_c, dtype=np.float32)
    assert np.array_equal(class_means(Vc, lab_c), [cm[x] for x in (2, 4, 7, 9)])
    assert peers(Vc, lab_c, 0) == [(1, 1.0), (4, 1.0), (5, 0.8)], peers(Vc, lab_c, 0)
    assert peers(Vc, lab_c, 0, k=5) == [(1, 1.0), (4, 1.0), (5, 0.8), (2, -1.0), (3, None)]
    assert peers(Vc, lab_c, 4) == [(0, 1.0), (1, 1.0), (5, 0.8)]
    assert peers(Vc, lab_c, 3) == [(0, None), (1, None), (2, None)]  # c's own correlations all undefined
    rng = np.random.default_rng(6)
    Vr = rng.random((60, 9))
    lr = rng.integers(0, 10, 60)
    cr = np.corrcoef(class_means(Vr, lr).T)[2]
    pr_r = peers(Vr, lr, 2, k=8)
    assert [j for j, _ in pr_r] == sorted([j for j in range(9) if j != 2], key=lambda j: -cr[j])
    assert all(abs(r - cr[j]) < 1e-12 for j, r in pr_r)

    # peer photos: centred on the D photos of the same label only (photos 6, 8, 9 are not in D), ties by lower id
    S_p = np.zeros((10, 3))
    S_p[:, 0] = [0.5, 0.25, 0.75, 1.0, 0.0, 0.5, 1.0, 0.25, 0.0, 1.0]
    S_p[:, 1] = np.random.default_rng(7).random(10)
    S_p[:, 2] = [0.5, 0.5, 0.5, 0.25, 0.75, 0.5, 0.0, 0.75, 1.0, 0.0]
    lab_p = np.array([1, 1, 1, 2, 2, 2, 1, 3, 2, 3])
    D_p = [0, 1, 2, 3, 4, 5, 7]
    # diff: 0: 0, 1: -0.25, 2: 0.25, 3: 0.75, 4: -0.75, 5: 0, 7: 0
    assert peer_photos(S_p, lab_p, D_p, 0, 2) == ([3, 2], [4, 1])
    assert peer_photos(S_p, lab_p, D_p, 0, 2, n=4) == ([3, 2, 0, 5], [4, 1, 0, 5])
    assert peer_photos(S_p, lab_p, D_p, 2, 0) == ([4, 1], [3, 2])

    # predictions: bad entries dropped, case and spaces in most/least, most equal to least, the cap, the explanation
    g1 = {'question': ' Is the animal facing the camera? ', 'answers': [' facing', 'away ', 'no animal'],
          'most_active': 'FACING', 'least_active': ' No Animal '}
    g2 = {'question': 'Is there grass?', 'answers': ['yes', 'no'], 'most_active': 'yes', 'least_active': 'no'}
    g3 = {'question': 'Which sign?', 'answers': ['Straße', 'other', 'none'], 'most_active': ' STRASSE',
          'least_active': 'None'}
    g4 = {'question': 'How many?', 'answers': ['0', '1', '2', '3', 'many'], 'most_active': 'Many',
          'least_active': '0'}
    w1 = {'question': 'Is the animal facing the camera?', 'answers': ['facing', 'away', 'no animal'], 'most': 0,
          'least': 2}
    w2 = {'question': 'Is there grass?', 'answers': ['yes', 'no'], 'most': 0, 'least': 1}
    w3 = {'question': 'Which sign?', 'answers': ['Straße', 'other', 'none'], 'most': 0, 'least': 2}
    w4 = {'question': 'How many?', 'answers': ['0', '1', '2', '3', 'many'], 'most': 4, 'least': 0}

    def without(dct, k):
        return {x: v for x, v in dct.items() if x != k}
    bad = ['a string', 7, None, [g2],
           dict(g2, question=''), dict(g2, question='   '), dict(g2, question=3), without(g2, 'question'),
           dict(g2, answers=['yes']), dict(g2, answers=['yes', 'no', '1', '2', '3', '4']),
           dict(g2, answers=['yes', 'no', ' NO ']), dict(g3, answers=['Straße', 'STRASSE', 'none']),
           dict(g2, answers=['yes', 'no', '  ']), dict(g2, answers=['yes', 'no', 3]), dict(g2, answers='yes no'),
           without(g2, 'answers'), dict(g2, most_active='maybe'), dict(g2, least_active='yes!'),
           dict(g2, most_active=0), dict(g2, least_active=None), without(g2, 'most_active'),
           without(g2, 'least_active'), dict(g2, most_active='No', least_active=' no'),
           dict(g2, least_active=' YES ')]
    body = {'explanation': 'Grass, and animals facing the camera.',
            'predictions': bad[:2] + [g1] + bad[2:12] + [g2] + bad[12:] + [g3]}
    reply = 'Here it is:\n```json\n' + json.dumps(body) + '\n```\nDone.'
    assert parse_predictions(reply) == {'explanation': 'Grass, and animals facing the camera.',
                                        'predictions': [w1, w2, w3]}, parse_predictions(reply)
    capped = [g4, g2, g1, g3, dict(g2, question='Fifth?'), dict(g2, question='Sixth?')]
    assert parse_predictions(json.dumps({'explanation': 'x', 'predictions': capped}))['predictions'] == [w4, w2, w1, w3]
    assert parse_predictions(json.dumps({'explanation': ' Grass. ', 'predictions': [g2]})) == {
        'explanation': ' Grass. ', 'predictions': [w2]}
    for b in [{'predictions': [g2]}, {'explanation': '  ', 'predictions': [g2]}, {'explanation': 5, 'predictions': [g2]},
              {'explanation': None, 'predictions': [g2]}, {'explanation': 'x', 'predictions': bad},
              {'explanation': 'x', 'predictions': g2}, {'explanation': 'x', 'predictions': []}, {'explanation': 'x'}]:
        assert parse_predictions(json.dumps(b)) is None, b
    good_b = {'explanation': 'x', 'predictions': [g2]}
    assert parse_predictions(json.dumps(good_b) + ' then ' + json.dumps({'predictions': [g2]})) is None
    assert parse_predictions(json.dumps({'predictions': [g2]}) + ' or ' + json.dumps(good_b)) == {
        'explanation': 'x', 'predictions': [w2]}
    assert parse_predictions('no json here') is None

    # contrast on hand values: hi photos at even rows, lo at odd rows, rows 10-11 (answer 2, None) left out
    ans_c = [0, 1] * 5 + [2, None]
    Rc = np.array([[1, 0] * 5 + [100, -100], [0.25] * 10 + [9, -9], [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 50, -50]],
                  dtype=np.float64).T
    got = contrast(Rc, ans_c, 0, 1)
    assert got[:2] == (5, 5) and got[2].shape == (3,)
    assert abs(got[2][0] - math.sqrt(3.6)) < 1e-12 and abs(got[2][2] + 1 / math.sqrt(2.5)) < 1e-12, got[2]
    assert got[2][1] == 0  # zero sd over the compared photos
    sw = contrast(Rc, ans_c, 1, 0)
    assert sw[:2] == (5, 5) and np.array_equal(sw[2], -got[2])
    ans_f = list(ans_c)
    ans_f[0] = None
    assert contrast(Rc, ans_f, 0, 1) is None and contrast(Rc, ans_c, 0, 2) is None  # 4 photos; 1 photo
    assert contrast(Rc, ans_f, 0, 1, min_n=4)[:2] == (4, 5)
    assert contrast(np.zeros((12, 2)), ans_c, 0, 1)[2].tolist() == [0.0, 0.0]

    def brute_d(col, ans, hi, lo):
        on = [x for x, a in zip(col, ans) if a == hi]
        off = [x for x, a in zip(col, ans) if a == lo]
        sd = statistics.stdev(on + off)
        return (statistics.fmean(on) - statistics.fmean(off)) / sd if sd else 0.0
    Rr = rng.normal(size=(40, 6))
    ar = [0] * 9 + [1] * 12 + [2] * 7 + [3] * 3 + [None] * 9
    ar = [ar[t] for t in rng.permutation(40)]
    for hi, lo in [(1, 0), (2, 1), (0, 2)]:
        gr = contrast(Rr, ar, hi, lo)
        assert gr[:2] == (ar.count(hi), ar.count(lo)), (hi, lo)
        for j in range(6):
            assert abs(gr[2][j] - brute_d(Rr[:, j].tolist(), ar, hi, lo)) < 1e-9, (hi, lo, j)
    assert contrast(Rr, ar, 3, 0) is None

    # outcome at the boundary, and the unit's record
    assert outcome(0.01, 95.0) == 'held' and outcome(0.01, float(np.nextafter(95.0, 0.0))) == 'not held'
    assert outcome(0.0, 100.0) == 'not held' and outcome(-0.3, 99.0) == 'not held' and outcome(2.0, 100.0) == 'held'
    ck = prediction_check(Rc, 0, ans_c, 0, 1)
    assert ck == {'n_hi': 5, 'n_lo': 5, 'd': float(got[2][0]), 'pct': 100.0 * 2 / 3, 'outcome': 'not held'}, ck
    assert prediction_check(Rc, 0, ans_f, 0, 1) == {'n_hi': 4, 'n_lo': 5, 'd': None, 'pct': None, 'outcome': 'too few'}

    # results text on a hand case: held, not held, too few, and not held with a high percentile but d below 0
    preds_t = [{'question': 'Is the cat lying down?', 'answers': ['lying', 'standing', 'no cat'], 'most': 0,
                'least': 1},
               {'question': 'Is it indoors?', 'answers': ['indoors', 'outdoors'], 'most': 1, 'least': 0},
               {'question': 'Is there text?', 'answers': ['yes', 'no', 'cannot tell'], 'most': 0, 'least': 2},
               {'question': 'Is it dark?', 'answers': ['dark', 'bright'], 'most': 0, 'least': 1}]
    res_t = [{'n_hi': 14, 'n_lo': 9, 'd': 0.8361, 'pct': 97.5}, {'n_hi': 20, 'n_lo': 7, 'd': -0.004, 'pct': 49.6},
             {'n_hi': 3, 'n_lo': 30, 'd': None, 'pct': None}, {'n_hi': 11, 'n_lo': 6, 'd': -0.25, 'pct': 96.0}]
    head_t = RESULTS_HEAD.replace('{n}', '118').replace('{K}', '24').replace('{m}', '512')
    want_t = '\n\n'.join([
        head_t,
        'Prediction 1. Is the cat lying down?\nAnswers: lying; standing; no cat\n'
        'Predicted most active: lying. Predicted least active: standing.\n'
        'Result: 14 and 9 photos, d +0.84, percentile 98: held.',
        'Prediction 2. Is it indoors?\nAnswers: indoors; outdoors\n'
        'Predicted most active: outdoors. Predicted least active: indoors.\n'
        'Result: 20 and 7 photos, d -0.00, percentile 50: did not hold.',
        'Prediction 3. Is there text?\nAnswers: yes; no; cannot tell\n'
        'Predicted most active: yes. Predicted least active: cannot tell.\n'
        'Result: 3 and 30 photos: too few to check (each needs at least 5).',
        'Prediction 4. Is it dark?\nAnswers: dark; bright\n'
        'Predicted most active: dark. Predicted least active: bright.\n'
        'Result: 11 and 6 photos, d -0.25, percentile 96: did not hold.'])
    got_t = results_text(118, 512, 24, preds_t, res_t)
    assert got_t == want_t, got_t
    assert results_text(118, 512, 24, json.loads(json.dumps(preds_t)), json.loads(json.dumps(res_t))) == want_t
    assert head_t.startswith('The predictions were then checked on new photos: 118 photos of the 24 photo classes')
    assert '{' not in head_t and 'among all 512 units' in head_t

    # looking content: block order, numbering and image count
    def blk(i):
        return {'type': 'image', 'id': int(i)}
    pa = [('tabby', [(5, 100), (9, 87)]), ('tiger cat', [(12, 40)])]
    pb = [[(3, 'tabby', 80, 10), (4, 'tiger cat', 50, 0)], [(6, 'tabby', 5, 60), (7, 'tabby', 0, 33)]]
    assert looking_content(pa, pb, block=blk) == [
        _text(LOOK_HEAD), _text('Class 1: tabby'), _text('Photo 1.1: 100%'), blk(5), _text('Photo 1.2: 87%'), blk(9),
        _text('Class 2: tiger cat'), _text('Photo 2.1: 40%'), blk(12), _text(PEER_HEAD), _text('Similar unit 1'),
        _text('Photo S1.1 (tabby): this unit 80%, similar unit 1 10%'), blk(3),
        _text('Photo S1.2 (tiger cat): this unit 50%, similar unit 1 0%'), blk(4), _text('Similar unit 2'),
        _text('Photo S2.1 (tabby): this unit 5%, similar unit 2 60%'), blk(6),
        _text('Photo S2.2 (tabby): this unit 0%, similar unit 2 33%'), blk(7), _text(LOOK_TASK)]
    full = looking_content([(f'class {j}', [(10 * j + k, 50) for k in range(6)]) for j in range(4)],
                           [[(100 + 4 * j + k, 'x', 1, 2) for k in range(4)] for j in range(3)], block=blk)
    imgs = [b['id'] for b in full if b['type'] == 'image']
    assert len(full) == 82 and imgs == [10 * j + k for j in range(4) for k in range(6)] + list(range(100, 112))
    assert all(b['type'] == 'text' for b in full if 'id' not in b)
    assert full[0]['text'] == LOOK_HEAD and full[-1]['text'] == LOOK_TASK and full[53]['text'] == PEER_HEAD
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

    secs, allr = [], {}
    for u in units:
        n = u['name']
        print('==', n, flush=True)
        sec, rd = run_unit(u, vals[u['layer']], labels, names, cats)
        rd = rd or {}
        allr[n] = {rid: (rd[rid]['explanation'] if rid in rd else None) for rid in READINGS}
        secs.append(sec)
        proploop.save_json(allr, OUT / 'readings_all.json')
        (OUT / 'report.txt').write_text('\n\n'.join(secs) + '\n')
        print(sec, flush=True)
        gc.collect()
    print('report ->', OUT / 'report.txt')


if __name__ == '__main__':
    main()
