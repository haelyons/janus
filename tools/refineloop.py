"""Conditional refinements of earlier predictions, checked blind on other photos (spec: authors/spec_refineloop.md).

Per unit (one channel of mixed3a-mixed5b), from tools/contrastloop.py's outputs in runs/contrastloop/<unit>/: every
earlier prediction is checked again on the checked photos T, for the unit and for its 3 similar units (d, percentile,
the correlation of the two units over the compared photos, and z for the difference of their d). Two arms each pick one
earlier prediction: F by comparing the unit with its similar units (one that held for a similar unit too, else one on
which the two clearly differ), R the one with the unit's highest percentile. An agent sees the earlier results and, for
each of that prediction's two compared answers, photos of T given that answer (F: where the unit and the similar unit
differ most; R: where the unit is most and least active), and writes up to 2 predictions that split one of those answers
with a new question; 3 samples per arm. A second agent, which never sees any activity, answers the earlier and the new
questions for every looked-at photo D, 12 per call (as tools/proploop.py). Per new prediction, d on the D photos given
the split answer, for every channel of the layer, with the unit's and its similar units' percentiles among them; also
on all D photos, on the D photos not shown in round 27, and after permuting the answers within class (null).
Outputs: runs/refineloop/.

Decision printed in the report, per new prediction: the unit's d percentile among its layer's channels >= 95 and its d
> 0 -> held, else not held; too few when either compared answer has fewer than 5 photos within the split answer;
distinctive when held and the lowest of its similar units' percentiles is below 50.

usage: python -I tools/refineloop.py [--units-file authors/round28_units.json] [--selftest]
"""
import argparse
import gc
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))  # -I does not add the script dir
import bundlemax  # noqa: E402
import contrastloop  # noqa: E402
import proploop  # noqa: E402
import run_subjects  # noqa: E402

OUT = J / 'runs/refineloop'
SRC = contrastloop.OUT  # earlier predictions, their answers on T and the round-27 panels
UNITS_FILE = 'authors/round28_units.json'
LOOKS = contrastloop.LOOKS  # earlier samples, in order
SAMPLES = contrastloop.SAMPLES
ARMS = ('F', 'R')
CALLS = [f'{a}_s{s}' for a in ARMS for s in SAMPLES]
N_PANEL = 3  # photos on each side of a group's panel
MAX_REF = 2  # new predictions kept per reply
PCT_SHARED = 95.0  # a similar unit's percentile at least this: the prediction holds for it too
Z_DIFF = 2.0  # |z| at least this: the unit and the similar unit clearly differ
RHO_EPS = 1e-12
PCT_DISTINCT = 50.0  # distinctive: held, and the lowest similar-unit percentile below this
OUTCOMES = contrastloop.OUTCOMES
TABLES = proploop.TABLES
WORKERS = proploop.WORKERS  # calls in flight, through proploop.cached_calls
_text = contrastloop._text

INTRO = 'Below is what is known about one unit (a single neuron) of an image classification network.'
KNOWN = (
    "Someone who looked at photos of the unit's activity wrote predictions about it, and each was checked on other "
    "photos. Someone who never saw the unit's activity answered the prediction's question about each photo. d is how "
    'much more active the unit is on photos given the predicted most-active answer than on photos given the predicted '
    'least-active answer, after removing the differences between photo classes, in standard deviations; the '
    "percentile places it among all {m} units of the unit's layer on the same photos and answers. A prediction held "
    'when its percentile is at least 95.')
KNOWN_PEERS = (
    'The same was computed for the 3 units of the layer whose average activity over photo classes is most like this '
    "unit's (similar units 1 to 3). A prediction that held for this unit and also for a similar unit describes "
    'something they share, not what is particular to this unit. A prediction on which this unit and a similar unit '
    'clearly differ points to something particular to one of them.')
F_LOOK = (
    'Prediction {j} is examined below, with similar unit {k}. For each of the two answers it compares, there are photos '
    'given that answer: first some on which this unit is much more active than similar unit {k}, then some on which it '
    "is much less active, each relative to its own average on the photo's class. The percentages are each unit's "
    'activity as a share of the most it reaches on any of 10000 photos.')
R_LOOK = (
    'Prediction {j} is examined below. For each of the two answers it compares, there are photos given that answer: '
    'first those on which the unit is most active, then those on which it is least active, each relative to its '
    "average on the photo's class. The percentages are the unit's activity as a share of the most it reaches on any of "
    '10000 photos.')
F_HEAD = '\n\n'.join([INTRO, KNOWN + ' ' + KNOWN_PEERS, '{table}', F_LOOK])
R_HEAD = '\n\n'.join([INTRO, KNOWN, '{table}', R_LOOK])
F_TASK = (
    'First, in "notes", describe each photo above in a few words. Then, within one of the two answers, find what '
    'separates the photos on which this unit is more active than similar unit {k} from those on which it is less '
    'active, and write it as up to 2 predictions. Each prediction splits one of the two answers: a new question to be '
    'asked about new photos given that answer, with 2 to 5 short answers that do not overlap, so that every photo fits '
    'exactly one of them, including one for photos to which it does not apply; the answer on whose photos you predict '
    'this unit will be most active; and the answer on whose photos you predict it will be least active. Each prediction '
    'will be checked on new photos given the answer it splits, within the same class, against all units of the layer, '
    'and should hold for this unit but not for similar unit {k}. Reply with only JSON: {"notes": ["..."], '
    '"predictions": [{"splits": "...", "question": "...", "answers": ["...", "..."], "most_active": "...", '
    '"least_active": "..."}]}, where "splits" is one of the two answers above, exactly as written.')
R_TASK = (
    'First, in "notes", describe each photo above in a few words. Then, within one of the two answers, find what '
    'separates the photos on which this unit is more active from those on which it is less active, and write it as up '
    'to 2 predictions. Each prediction splits one of the two answers: a new question to be asked about new photos given '
    'that answer, with 2 to 5 short answers that do not overlap, so that every photo fits exactly one of them, including '
    'one for photos to which it does not apply; the answer on whose photos you predict this unit will be most active; '
    'and the answer on whose photos you predict it will be least active. Each prediction will be checked on new photos '
    'given the answer it splits, within the same class, against all units of the layer, and should hold for this unit. '
    'Reply with only JSON: {"notes": ["..."], "predictions": [{"splits": "...", "question": "...", "answers": ["...", '
    '"..."], "most_active": "...", "least_active": "..."}]}, where "splits" is one of the two answers above, exactly as '
    'written.')
VERDICTS = {
    'shared': 'Held for this unit and for similar unit {k}.',
    'own': 'Held for this unit only.',
    'differs': 'Did not hold for this unit; this unit and similar unit {k} clearly differ on it.',
    'none': 'Did not hold for this unit, and no similar unit clearly differs from it.'}
TOO_FEW = 'Too few photos to check.'
A_PROMPT = (
    'Answer every question below for each of the {b} images above. For each question choose exactly one of its '
    'answers, by letter, or "?" if you cannot tell. Unless a question says otherwise, left and right mean the left and '
    'right of the photo as you look at it.\n\n'
    '{questions}\n\n'
    'Reply with only JSON: {"images": [{"id": n, "answers": ["A", ...]}]}, one entry per image, each with one answer '
    'per question, in question order.')


# ---------- earlier predictions on T ----------

def z_score(d_u, d_p, rho, n_hi, n_lo):
    """(d_u - d_p) / sqrt(2 (1 - rho) (1/n_hi + 1/n_lo)); 0 when 1 - rho <= 1e-12"""
    q = 1.0 - float(rho)
    if q <= RHO_EPS:
        return 0.0
    return float((float(d_u) - float(d_p)) / math.sqrt(2.0 * q * (1.0 / n_hi + 1.0 / n_lo)))


def pearson(x, y):
    """the Pearson correlation of x and y; 0 when undefined (either constant)"""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    xc, yc = x - x.mean(), y - y.mean()
    return float((xc * yc).sum() / math.sqrt(float((xc ** 2).sum() * (yc ** 2).sum())))


def t_record(R, ans, most, least, c, peers):
    """one earlier prediction on R (n x m): None when not checkable (contrastloop.contrast), else the unit's (column c)
    n_hi, n_lo, d_u, pct_u and held (contrastloop.outcome), and per peer channel its d, pct, rho (Pearson of columns c
    and p over the photos given most or least, 0 when undefined) and z (z_score)"""
    R = np.asarray(R, dtype=np.float64)
    got = contrastloop.contrast(R, ans, most, least)
    if got is None:
        return None
    n_hi, n_lo, d = got
    d_u, pct_u = float(d[c]), proploop.percentile(d, c)
    codes = proploop._codes(ans)
    on = (codes == most) | (codes == least)
    ps = []
    for p in peers:
        p = int(p)
        rho = pearson(R[on, c], R[on, p])
        ps.append({'chan': p, 'd': float(d[p]), 'pct': proploop.percentile(d, p), 'rho': rho,
                   'z': z_score(d_u, float(d[p]), rho, n_hi, n_lo)})
    return {'n_hi': n_hi, 'n_lo': n_lo, 'd_u': d_u, 'pct_u': pct_u,
            'held': contrastloop.outcome(d_u, pct_u) == 'held', 'peers': ps}


def kind(rec):
    """(kind, peer channel or None): 'shared' when held and some peer's pct >= 95 (the peer with the highest pct, ties
    by peer order); 'own' when held otherwise; 'differs' when not held and some peer's |z| >= 2 (the peer with the
    highest |z|, ties by peer order); else 'none'; ('too few', None) for None"""
    if rec is None:
        return 'too few', None
    ps = rec['peers']
    if rec['held']:
        on = [t for t, p in enumerate(ps) if p['pct'] >= PCT_SHARED]
        if on:
            t = min(on, key=lambda t: (-ps[t]['pct'], t))
            return 'shared', ps[t]['chan']
        return 'own', None
    on = [t for t, p in enumerate(ps) if abs(p['z']) >= Z_DIFF]
    if on:
        t = min(on, key=lambda t: (-abs(ps[t]['z']), t))
        return 'differs', ps[t]['chan']
    return 'none', None


def _peer_z(rec, chan):
    return next(p['z'] for p in rec['peers'] if p['chan'] == chan)


def trigger_f(recs):
    """(index into recs, peer channel, kind) of arm F's earlier prediction: the 'shared' one with the highest pct_u,
    else the 'differs' one with the highest |z| of its peer, ties by lower index; None when there is neither"""
    ks = [kind(r) for r in recs]
    on = [i for i, (k, _) in enumerate(ks) if k == 'shared']
    if on:
        i = min(on, key=lambda i: (-recs[i]['pct_u'], i))
        return i, ks[i][1], 'shared'
    on = [i for i, (k, _) in enumerate(ks) if k == 'differs']
    if on:
        i = min(on, key=lambda i: (-abs(_peer_z(recs[i], ks[i][1])), i))
        return i, ks[i][1], 'differs'
    return None


def trigger_r(recs):
    """the index of the checkable record with the highest pct_u, ties by lower index; None when none is checkable"""
    on = [i for i, r in enumerate(recs) if r is not None]
    return min(on, key=lambda i: (-recs[i]['pct_u'], i)) if on else None


# ---------- panels ----------

def _group(ids, ans, a):
    """positions of the photos given answer a, in photo index order"""
    codes = proploop._codes(ans)
    assert len(codes) == len(ids), (len(codes), len(ids))
    return sorted(np.flatnonzero(codes == a).tolist(), key=lambda t: int(ids[t]))


def _ends(ids, vals, n):
    """(the n ids with the highest vals, the n with the lowest, lowest first), ties by lower id in both; floor(len / 2)
    each instead when there are fewer than 2n"""
    k = n if len(ids) >= 2 * n else len(ids) // 2
    hi = sorted(range(len(ids)), key=lambda t: (-vals[t], ids[t]))[:k]
    lo = sorted(range(len(ids)), key=lambda t: (vals[t], ids[t]))[:k]
    return [ids[t] for t in hi], [ids[t] for t in lo]


def panel_f(R, ids, ans, a, c, k, n=N_PANEL):
    """(more, less) photo ids of the group given answer a: e = R[:, c] - R[:, k]; R rows and ans aligned with ids"""
    R = np.asarray(R, dtype=np.float64)
    assert R.ndim == 2 and R.shape[0] == len(ids), (R.shape, len(ids))
    G = _group(ids, ans, a)
    return _ends([int(ids[t]) for t in G], [float(R[t, c] - R[t, k]) for t in G], n)


def panel_r(R, ids, ans, a, c, n=N_PANEL):
    """(most, least) photo ids of the group given answer a, on R[:, c]; R rows and ans aligned with ids"""
    R = np.asarray(R, dtype=np.float64)
    assert R.ndim == 2 and R.shape[0] == len(ids), (R.shape, len(ids))
    G = _group(ids, ans, a)
    return _ends([int(ids[t]) for t in G], [float(R[t, c]) for t in G], n)


# ---------- prompts and replies ----------

def table_text(rows, arm):
    """{table}: one block per earlier prediction, blocks separated by blank lines. rows: {'question', 'answers', 'most',
    'least', 'rec' (t_record output or None), 'kind', 'peer_no' (1-3 or None)}; arm 'F' or 'R'"""
    assert arm in ARMS, arm
    blocks = []
    for j, row in enumerate(rows, 1):
        a, rec = row['answers'], row['rec']
        L = [f'Prediction {j}. {row["question"]}', 'Answers: ' + '; '.join(a),
             f'Predicted most active: {a[row["most"]]}. Predicted least active: {a[row["least"]]}.']
        if rec is None:
            L.append(TOO_FEW)
            blocks.append('\n'.join(L))
            continue
        unit = f'This unit: d {float(rec["d_u"]):+.2f}, percentile {proploop._whole(rec["pct_u"])}.'
        if arm == 'F':
            L.append(unit + ''.join(f' Similar unit {t}: d {float(p["d"]):+.2f}, percentile '
                                    f'{proploop._whole(p["pct"])}.' for t, p in enumerate(rec['peers'], 1)))
            L.append(proploop.fill(VERDICTS[row['kind']], k=row['peer_no']))
        else:
            L += [unit, 'Held.' if rec['held'] else 'Did not hold.']
        blocks.append('\n'.join(L))
    return '\n\n'.join(blocks)


def refine_content(arm, m, table, j, k_no, groups, block=None):
    """the refinement content: the head; per group g its answer, then its 'hi' photos and its 'lo' photos, each with a
    heading, 'Photo g.i (class name): ...' and the photo (block(id), default run_subjects.img_block of its thumb),
    numbering continuing within a group; the task. groups: 2 entries {'answer', 'hi': [(photo id, class name, pct_c,
    pct_k)], 'lo': [...]}; k_no the similar unit's number (pct_k and k_no unused for arm R)"""
    assert arm in ARMS and len(groups) == 2, (arm, len(groups))
    block = block or (lambda i: run_subjects.img_block(proploop.thumb(i)))
    f = arm == 'F'
    content = [_text(proploop.fill(F_HEAD if f else R_HEAD, m=m, table=table, j=j, k=k_no))]
    heads = ((f'This unit much more active than similar unit {k_no}:',
              f'This unit much less active than similar unit {k_no}:') if f else ('Most active:', 'Least active:'))
    for g, grp in enumerate(groups, 1):
        content.append(_text(f'Answer {g}: "{grp["answer"]}"'))
        i = 0
        for head, photos in zip(heads, (grp['hi'], grp['lo'])):
            content.append(_text(head))
            for pid, cname, pc, pk in photos:
                i += 1
                t = (f'Photo {g}.{i} ({cname}): this unit {proploop._whole(pc)}%, similar unit {k_no} '
                     f'{proploop._whole(pk)}%' if f else f'Photo {g}.{i} ({cname}): {proploop._whole(pc)}%')
                content += [_text(t), block(pid)]
    content.append(_text(proploop.fill(F_TASK if f else R_TASK, k=k_no)))
    return content


def _entry(p):
    """p stored as contrastloop stores a prediction ({'question', 'answers', 'most', 'least'}) when it passes
    contrastloop.parse_predictions's entry rule, else None; the rule is applied by that function to p alone (values
    that are not strings or lists of strings, which the rule rejects anyway, are passed as null so that nothing nested
    in them is parsed)"""
    if not isinstance(p, dict):
        return None

    def plain(v):
        return v if isinstance(v, str) or (isinstance(v, list) and all(isinstance(x, str) for x in v)) else None
    one = {k: plain(p[k]) for k in ('question', 'answers', 'most_active', 'least_active') if k in p}
    got = contrastloop.parse_predictions(json.dumps({'explanation': '-', 'predictions': [one]}))
    return None if got is None else got['predictions'][0]


def parse_refined(text, trigger_answers, compared=None):
    """{'notes': as given or None, 'predictions': [...]} of the last JSON object with 'predictions': the entries, in
    order, that pass contrastloop.parse_predictions's entry rule and whose 'splits' is a string that, stripped and
    casefolded, equals exactly one compared answer of trigger_answers (stripped, casefolded); stored as contrastloop
    stores a prediction plus 'splits', that answer's index in trigger_answers; the first MAX_REF of them. compared: the
    indices of the trigger's two compared answers, None for all of trigger_answers. None when none is kept or no
    object is found"""
    o = bundlemax.last_obj(text, ('predictions',))
    if o is None:
        return None
    keys = [str(x).strip().casefold() for x in trigger_answers]
    allowed = list(range(len(keys))) if compared is None else [int(a) for a in compared]
    kept = []
    for p in o['predictions'] if isinstance(o['predictions'], list) else []:
        e = _entry(p)
        s = p.get('splits') if isinstance(p, dict) else None
        if e is None or not isinstance(s, str):
            continue
        hit = [a for a in allowed if keys[a] == s.strip().casefold()]
        if len(hit) == 1:
            kept.append({**e, 'splits': hit[0]})
    kept = kept[:MAX_REF]
    return {'notes': o.get('notes'), 'predictions': kept} if kept else None


def d_content(batch, questions, block=None):
    """proploop.answers_content with the last text block from A_PROMPT"""
    content = proploop.answers_content(batch, questions, block)
    content[-1] = _text(proploop.fill(A_PROMPT, b=len(batch), questions=proploop.questions_block(questions)))
    return content


# ---------- checks on D ----------

def conditional(parent_ans, child_ans, a):
    """per photo: the child answer where the parent answer is a, else None"""
    assert len(parent_ans) == len(child_ans), (len(parent_ans), len(child_ans))
    return [ch if pa is not None and pa == a else None for pa, ch in zip(parent_ans, child_ans)]


def d_check(R, ans, hi, lo, c, peers):
    """contrastloop.prediction_check for column c, plus each peer channel's d and percentile (None when not
    checkable)"""
    rec = contrastloop.prediction_check(R, c, ans, hi, lo)
    got = contrastloop.contrast(R, ans, hi, lo)
    rec['peers'] = [{'chan': int(p), 'd': None if got is None else float(got[2][p]),
                     'pct': None if got is None else proploop.percentile(got[2], p)} for p in peers]
    return rec


def distinctive(rec):
    """held, and the lowest peer percentile below 50"""
    ps = [p['pct'] for p in rec['peers']]
    return rec['outcome'] == 'held' and bool(ps) and min(ps) < PCT_DISTINCT


# ---------- inputs ----------

def load_inputs(name, src=SRC):
    """contrastloop's stats.json, answers.json and panels.json of the unit; stops when any is missing or any answer
    batch on T failed"""
    udir = src / name
    got = []
    for fn in ('stats.json', 'answers.json', 'panels.json'):
        if not (udir / fn).is_file():
            raise SystemExit(f'{udir / fn}: missing; stopping')
        got.append(json.loads((udir / fn).read_text()))
    st, an, pa = got
    bad = sorted(k for k, r in an.items() if r.get('failed'))
    if bad:
        raise SystemExit(f'{udir / "answers.json"}: answer batches {bad} on T failed; stopping')
    return st, an, pa


def t_rows(an, T, path):
    """each T photo's answer row from contrastloop's answers batches; stops when a photo has none"""
    row_of = {}
    for rec in an.values():
        row_of.update(zip((int(i) for i in rec['photos']), rec['answers']))
    missing = [i for i in T if i not in row_of]
    if missing:
        raise SystemExit(f'{path}: no answers for {len(missing)} photos of T; stopping')
    return [row_of[i] for i in T]


def earlier(st):
    """every prediction of s1, s2, s3 in that order"""
    out = []
    for s in LOOKS:
        for p in st['samples'][s]['predictions'] or []:
            out.append({'sample': s, 'question': p['question'], 'answers': p['answers'], 'most': p['most'],
                        'least': p['least'], 'question_index': p['question_index']})
    return out


def shown_photos(pa):
    """the photos of round 27's panels: the range photos and the peer photos"""
    return ({int(i) for r in pa['range'] for i in r['photos']} | {int(i) for p in pa['peers'] for i in p['photos']})


# ---------- calls ----------

def _refine_parser(answers, compared):
    def parse(text):
        return parse_refined(text, answers, compared)
    return parse


def _refine_fields(p):
    """the cached fields of a refinement call: its notes and predictions, both None when it is not valid"""
    return {'notes': None if p is None else p['notes'], 'predictions': None if p is None else p['predictions']}


# ---------- one unit, report ----------

def _t_text(rec, k, kp):
    if rec is None:
        return 'T: too few'
    ps = ', '.join(f'{p["chan"]} d {p["d"]:+.2f} pct {p["pct"]:.1f} rho {p["rho"]:.3f} z {p["z"]:+.2f}'
                   for p in rec['peers'])
    return (f'T: {rec["n_hi"]} and {rec["n_lo"]} photos, d {rec["d_u"]:+.2f}, percentile {rec["pct_u"]:.1f} -> '
            f'{"held" if rec["held"] else "not held"}; peers {ps} -> {k}' + ('' if kp is None else f' (peer {kp})'))


def _check_text(r):
    if r['d'] is None:
        return f'{r["n_hi"]} and {r["n_lo"]} photos -> {r["outcome"]}'
    return (f'{r["n_hi"]} and {r["n_lo"]} photos, d {r["d"]:+.2f}, percentile {r["pct"]:.1f} -> {r["outcome"]}; '
            'peers ' + ', '.join(f'{p["chan"]} d {p["d"]:+.2f} pct {p["pct"]:.1f}' for p in r['peers']))


def section(name, layer, m, nT, nD, peer_recs, preds, recs, kinds, trig, rc, refs, rechecks, tal, n_fail, n_batches):
    L = [f'== {name}  layer {layer} ({m} channels), {nT} photos of T, {nD} photos of D, answer batches on D failed '
         f'after a retry: {n_fail} of {n_batches}',
         'peers: ' + ', '.join(f'{layer}:{p["chan"]} (r {contrastloop._corr(p["corr"])})' for p in peer_recs),
         'earlier predictions:']
    for j, (p, rec, (k, kp)) in enumerate(zip(preds, recs, kinds), 1):
        a = p['answers']
        L.append(f'   P{j} ({p["sample"]}). {p["question"]}  answers: ' + '; '.join(a)
                 + f'; most active: {a[p["most"]]}; least active: {a[p["least"]]}')
        L.append('      ' + _t_text(rec, k, kp))
    for arm in ARMS:
        t = trig[arm]
        if t is None:
            L.append(f'trigger {arm}: none')
            continue
        L.append(f'trigger {arm}: P{t["j"]}, {t["kind"]}'
                 + ('' if t['peer'] is None else f', peer {layer}:{t["peer"]} (similar unit {t["peer_no"]})'))
        hi, lo = ('more', 'less') if arm == 'F' else ('most', 'least')
        for g in t['panels']:
            L.append(f'   "{g["answer"]}" ({g["n"]} photos of T): {hi} {g["hi"]}, {lo} {g["lo"]}')
        L += [f'   {key}  no valid reply (not parsed after a retry)' for key in (f'{arm}_s{s}' for s in SAMPLES)
              if not rc[key]['valid']]
        if arm in rechecks:
            L += [f'   re-check on D, {tn}: ' + _check_text(rechecks[arm][tn]) for tn in TABLES]
    for r in refs:
        a = r['answers']
        L.append(f'{r["call"]} #{r["no"]}. splits "{trig[r["arm"]]["answers"][r["splits"]]}": {r["question"]}  '
                 'answers: ' + '; '.join(a) + f'; most active: {a[r["most"]]}; least active: {a[r["least"]]}')
        for tn in TABLES:
            x = r[tn]
            L.append(f'   {tn}: {_check_text(x)}; distinctive {"yes" if x["distinctive"] else "no"}')
            L.append(f'   {tn}, all D photos: {_check_text(x["all_D"])}')
            L.append(f'   {tn}, D photos not shown in round 27: {_check_text(x["not_shown"])}')
    for arm in ARMS:
        L.append(f'tally {arm}: ' + '; '.join(f'{tn} ' + ', '.join(f'{o} {tal[arm][tn][o]}' for o in OUTCOMES)
                                              for tn in TABLES)
                 + f' (of {sum(r["arm"] == arm for r in refs)} new predictions)')
    return '\n'.join(L)


def run_unit(u, V, labels, cats, call=None, out=OUT, src=SRC, block=None):
    """one unit (u from proploop.parse_unit; V: its layer's values, photos x channels): earlier predictions on T,
    triggers and panels, refinement calls, answers on D, checks real and null, stats.json. -> the report section"""
    name, layer, c = u['name'], u['layer'], u['chan']
    udir = out / name
    V, labels = np.asarray(V), np.asarray(labels)
    m = V.shape[1]
    st, an, pa = load_inputs(name, src)
    if (st['unit'], st['layer'], st['chan'], st['m']) != (name, layer, c, m):
        raise SystemExit(f'{src / name / "stats.json"}: for {st["unit"]} with {st["m"]} channels; stopping')
    S = contrastloop.scaled(V)

    def pct(i, j):
        return proploop._whole(100.0 * S[i, j])

    D, T, classes = [int(i) for i in st['D']], [int(i) for i in st['T']], st['classes']
    peer_recs = st['peers']
    peers = [int(p['chan']) for p in peer_recs]
    peer_no = {p: t + 1 for t, p in enumerate(peers)}

    # earlier predictions on T
    T_ids = np.asarray(T, dtype=np.int64)
    R_T = proploop.centre(V[T_ids], labels[T_ids])
    rows_T = t_rows(an, T, src / name / 'answers.json')
    preds = earlier(st)
    recs, kinds, rows = [], [], []
    for p in preds:
        rec = t_record(R_T, [r[p['question_index']] for r in rows_T], p['most'], p['least'], c, peers)
        k, kp = kind(rec)
        recs.append(rec)
        kinds.append((k, kp))
        rows.append({'question': p['question'], 'answers': p['answers'], 'most': p['most'], 'least': p['least'],
                     'rec': rec, 'kind': k, 'peer_no': None if kp is None else peer_no[kp]})

    # triggers, panels and the refinement contents
    tf, tr = trigger_f(recs), trigger_r(recs)
    trig = {'F': None if tf is None else {'index': tf[0], 'peer': tf[1], 'kind': tf[2]},
            'R': None if tr is None else {'index': tr, 'peer': None, 'kind': kinds[tr][0]}}
    jobs = []
    for arm in ARMS:
        t = trig[arm]
        if t is None:
            continue
        p = preds[t['index']]
        ans = [r[p['question_index']] for r in rows_T]
        k = t['peer']
        t.update(j=t['index'] + 1, sample=p['sample'], question=p['question'], answers=p['answers'], most=p['most'],
                 least=p['least'], question_index_T=p['question_index'], peer_no=None if k is None else peer_no[k])
        groups, t['panels'] = [], []
        for a in (p['most'], p['least']):
            hi, lo = panel_f(R_T, T, ans, a, c, k) if arm == 'F' else panel_r(R_T, T, ans, a, c)
            groups.append({'answer': p['answers'][a],
                           'hi': [(i, cats[int(labels[i])], pct(i, c), None if k is None else pct(i, k)) for i in hi],
                           'lo': [(i, cats[int(labels[i])], pct(i, c), None if k is None else pct(i, k)) for i in lo]})
            t['panels'].append({'answer_index': a, 'answer': p['answers'][a],
                                'n': int(np.count_nonzero(proploop._codes(ans) == a)), 'hi': hi, 'lo': lo,
                                'hi_pct': [x[2:] for x in groups[-1]['hi']], 'lo_pct': [x[2:] for x in groups[-1]['lo']]})
        content = refine_content(arm, m, table_text(rows, arm), t['j'], t['peer_no'], groups, block)
        parse = _refine_parser(p['answers'], (p['most'], p['least']))
        jobs += [(f'{arm}_s{s}', content, parse, _refine_fields) for s in SAMPLES]

    # refinement calls: three samples of the same content per arm
    rc = proploop.cached_calls(name, udir / 'refine.json', jobs, call) if jobs else {}
    missing = [j[0] for j in jobs if j[0] not in rc]
    del jobs
    if missing:
        return proploop.failed_section(name, f'refinement incomplete: calls {missing} not cached (call errors); '
                                             'rerun')

    # questions: the distinct trigger questions, then the kept new predictions in call order
    qs, tq, refs = [], {}, []
    for arm in ARMS:
        if trig[arm] is not None:
            q = {'question': trig[arm]['question'], 'answers': trig[arm]['answers']}
            if q not in qs:
                qs.append(q)
            tq[arm] = qs.index(q)
    for key in CALLS:
        if key not in rc or not rc[key]['valid']:
            continue
        for n, p in enumerate(rc[key]['predictions'], 1):
            refs.append({'call': key, 'arm': key.split('_')[0], 'no': n, **p, 'question_index': len(qs)})
            qs.append({'question': p['question'], 'answers': p['answers']})
    nq = len(qs)

    # answers on the looked-at photos
    rows_D, failed, n_batches = None, [], 0
    if nq:
        batches = proploop.answer_batches(name, D)
        n_batches = len(batches)
        keys = [str(b) for b in range(1, n_batches + 1)]
        apath = udir / 'answers.json'
        contrastloop.check_answer_cache(apath, batches)
        ajobs = [(key, d_content(batch, qs, block), proploop._answer_parser(len(batch), qs),
                  proploop._answer_fields(batch, nq)) for key, batch in zip(keys, batches)]
        ac = proploop.cached_calls(name, apath, ajobs, call)
        del ajobs
        missing = [k for k in keys if k not in ac]
        if missing:
            return proploop.failed_section(name, f'answers incomplete: batches {missing} not cached (call errors); '
                                                 'rerun')
        row_of = {}
        for key, batch in zip(keys, batches):
            if ac[key]['photos'] != batch or len(ac[key]['answers']) != len(batch):
                raise SystemExit(f'{apath}: batch {key} is for other photos; stopping')
            row_of.update(zip(batch, ac[key]['answers']))
        rows_D = [row_of[i] for i in D]
        failed = [k for k in keys if ac[k]['failed']]

    # checks on D, real and null
    shown = shown_photos(pa)
    unseen = [i not in shown for i in D]
    rechecks = {}
    if rows_D is not None:
        D_ids = np.asarray(D, dtype=np.int64)
        D_labels = labels[D_ids]
        R_D = proploop.centre(V[D_ids], D_labels)
        null_rows = proploop.permute_within(rows_D, D_labels, classes,
                                            np.random.default_rng(zlib.crc32(name.encode()) + 2))
        for tn, rr in zip(TABLES, (rows_D, null_rows)):
            for arm, q in tq.items():
                t = trig[arm]
                rechecks.setdefault(arm, {})[tn] = d_check(R_D, [r[q] for r in rr], t['most'], t['least'], c, peers)
            for ref in refs:
                parent = [r[tq[ref['arm']]] for r in rr]
                child = [r[ref['question_index']] for r in rr]
                cond = conditional(parent, child, ref['splits'])
                rec = d_check(R_D, cond, ref['most'], ref['least'], c, peers)
                rec['distinctive'] = distinctive(rec)
                rec['all_D'] = d_check(R_D, child, ref['most'], ref['least'], c, peers)
                rec['not_shown'] = d_check(R_D, [x if ok else None for x, ok in zip(cond, unseen)], ref['most'],
                                           ref['least'], c, peers)
                ref[tn] = rec
    tal = {arm: {tn: {o: sum(r['arm'] == arm and r[tn]['outcome'] == o for r in refs) for o in OUTCOMES}
                 for tn in TABLES} for arm in ARMS}

    proploop.save_json({
        'unit': name, 'layer': layer, 'chan': c, 'm': m, 'classes': classes, 'peers': peer_recs, 'D': D, 'T': T,
        'shown_round27': sorted(shown),
        'earlier': [{**p, 'j': j, 'T': rec, 'kind': k, 'peer': kp}
                    for j, (p, rec, (k, kp)) in enumerate(zip(preds, recs, kinds), 1)],
        'triggers': trig,
        'calls': {key: {'valid': rc[key]['valid'], 'notes': rc[key]['notes'], 'predictions': rc[key]['predictions']}
                  for key in CALLS if key in rc},
        'questions': qs, 'trigger_questions': tq, 'answer_batches': n_batches, 'failed_batches': failed,
        'refinements': refs, 'rechecks': rechecks, 'tally': tal}, udir / 'stats.json')
    return section(name, layer, m, len(T), len(D), peer_recs, preds, recs, kinds, trig, rc, refs, rechecks, tal,
                   len(failed), n_batches)


# ---------- selftest ----------

def selftest():
    def para(*pieces):
        return ' '.join(pieces)

    # prompt texts, verbatim from the spec: wrapped lines joined by single spaces, paragraph breaks kept
    p1 = 'Below is what is known about one unit (a single neuron) of an image classification network.'
    p2 = para(
        "Someone who looked at photos of the unit's activity wrote predictions about it, and each was checked on other",
        "photos. Someone who never saw the unit's activity answered the prediction's question about each photo. d is how",
        'much more active the unit is on photos given the predicted most-active answer than on photos given the predicted',
        'least-active answer, after removing the differences between photo classes, in standard deviations; the',
        "percentile places it among all {m} units of the unit's layer on the same photos and answers. A prediction held",
        'when its percentile is at least 95. The same was computed for the 3 units of the layer whose average activity',
        "over photo classes is most like this unit's (similar units 1 to 3). A prediction that held for this unit and also",
        'for a similar unit describes something they share, not what is particular to this unit. A prediction on which',
        'this unit and a similar unit clearly differ points to something particular to one of them.')
    f4 = para(
        'Prediction {j} is examined below, with similar unit {k}. For each of the two answers it compares, there are photos',
        'given that answer: first some on which this unit is much more active than similar unit {k}, then some on which it',
        "is much less active, each relative to its own average on the photo's class. The percentages are each unit's",
        'activity as a share of the most it reaches on any of 10000 photos.')
    r4 = para(
        'Prediction {j} is examined below. For each of the two answers it compares, there are photos given that answer:',
        'first those on which the unit is most active, then those on which it is least active, each relative to its',
        "average on the photo's class. The percentages are the unit's activity as a share of the most it reaches on any of",
        '10000 photos.')
    assert F_HEAD == '\n\n'.join([p1, p2, '{table}', f4])
    assert R_HEAD == '\n\n'.join([p1, p2[:p2.index(' The same was computed')], '{table}', r4])
    assert p2.endswith('A prediction held when its percentile is at least 95. ' + KNOWN_PEERS)
    ft = para(
        'First, in "notes", describe each photo above in a few words. Then, within one of the two answers, find what',
        'separates the photos on which this unit is more active than similar unit {k} from those on which it is less',
        'active, and write it as up to 2 predictions. Each prediction splits one of the two answers: a new question to be',
        'asked about new photos given that answer, with 2 to 5 short answers that do not overlap, so that every photo fits',
        'exactly one of them, including one for photos to which it does not apply; the answer on whose photos you predict',
        'this unit will be most active; and the answer on whose photos you predict it will be least active. Each prediction',
        'will be checked on new photos given the answer it splits, within the same class, against all units of the layer,',
        'and should hold for this unit but not for similar unit {k}. Reply with only JSON: {"notes": ["..."],',
        '"predictions": [{"splits": "...", "question": "...", "answers": ["...", "..."], "most_active": "...",',
        '"least_active": "..."}]}, where "splits" is one of the two answers above, exactly as written.')
    assert F_TASK == ft
    a_f = 'the photos on which this unit is more active than similar unit {k} from those on which it is less active'
    b_f = 'and should hold for this unit but not for similar unit {k}.'
    assert ft.count(a_f) == 1 and ft.count(b_f) == 1
    assert R_TASK == ft.replace(a_f, 'the photos on which this unit is more active from those on which it is less '
                                     'active').replace(b_f, 'and should hold for this unit.')
    assert A_PROMPT == '\n\n'.join([
        para('Answer every question below for each of the {b} images above. For each question choose exactly one of its',
             'answers, by letter, or "?" if you cannot tell. Unless a question says otherwise, left and right mean the left and',
             'right of the photo as you look at it.'),
        '{questions}',
        para('Reply with only JSON: {"images": [{"id": n, "answers": ["A", ...]}]}, one entry per image, each with one answer',
             'per question, in question order.')])
    assert VERDICTS['differs'] == para('Did not hold for this unit; this unit and similar unit',
                                       '{k} clearly differ on it.')
    assert CALLS == ['F_s1', 'F_s2', 'F_s3', 'R_s1', 'R_s2', 'R_s3'] and LOOKS == ['s1', 's2', 's3'] and WORKERS == 3

    # z_score: the rule, rho 1 and above, negative rho
    assert abs(z_score(1.0, 0.5, 0.5, 10, 10) - 0.5 / math.sqrt(0.2)) < 1e-12
    assert abs(z_score(0.3, -0.1, 0.0, 5, 20) - 0.4 / math.sqrt(0.5)) < 1e-12
    assert abs(z_score(0.0, 1.0, -1.0, 4, 4) + 1 / math.sqrt(2)) < 1e-12
    assert z_score(1.0, 0.0, 1.0, 10, 10) == 0.0 and z_score(3.0, -3.0, 1.5, 5, 5) == 0.0
    assert z_score(1.0, 0.0, 1.0 - 1e-13, 10, 10) == 0.0

    # t_record on a hand case: answer 0 on rows 0-4, 1 on rows 5-9, rows 10-11 (answer 2, None) left out; column 1
    # equals the unit on the compared photos, column 2 is its mirror, column 3 uncorrelated, column 4 constant there
    ans_t = [0] * 5 + [1] * 5 + [2, None]
    Rt = np.zeros((12, 5))
    Rt[:, 0] = [2] * 5 + [0] * 5 + [100, -100]
    Rt[:, 1] = [2] * 5 + [0] * 5 + [-100, 100]
    Rt[:, 2] = [0] * 5 + [2] * 5 + [5, 5]
    Rt[:, 3] = [1, -1, 1, -1, 0] * 2 + [7, -7]
    Rt[:, 4] = [3] * 10 + [1, 2]
    du = 6 / math.sqrt(10)
    rt = t_record(Rt, ans_t, 0, 1, 0, [1, 2, 3, 4])
    assert rt['n_hi'] == 5 and rt['n_lo'] == 5 and abs(rt['d_u'] - du) < 1e-12 and rt['pct_u'] == 60.0, rt
    assert rt['held'] is False and [p['chan'] for p in rt['peers']] == [1, 2, 3, 4]
    assert [p['pct'] for p in rt['peers']] == [60.0, 0.0, 20.0, 20.0], rt['peers']
    assert [p['rho'] for p in rt['peers']] == [1.0, -1.0, 0.0, 0.0], rt['peers']
    for p, wd, wz in zip(rt['peers'], [du, -du, 0.0, 0.0], [0.0, 3.0, du / math.sqrt(0.8), du / math.sqrt(0.8)]):
        assert abs(p['d'] - wd) < 1e-12 and abs(p['z'] - wz) < 1e-9, (p, wd, wz)
    ans_f = list(ans_t)
    ans_f[0] = None
    assert t_record(Rt, ans_f, 0, 1, 0, [1, 2, 3, 4]) is None  # 4 photos given answer 0
    assert t_record(Rt, ans_t, 0, 2, 0, [1]) is None
    dc = d_check(Rt, ans_t, 0, 1, 0, [1, 2])
    assert dc['n_hi'] == 5 and dc['n_lo'] == 5 and abs(dc['d'] - du) < 1e-12 and dc['pct'] == 60.0
    assert dc['outcome'] == 'not held' and [(p['chan'], p['pct']) for p in dc['peers']] == [(1, 60.0), (2, 0.0)]
    assert abs(dc['peers'][1]['d'] + du) < 1e-12
    assert d_check(Rt, ans_f, 0, 1, 0, [1, 2]) == {
        'n_hi': 4, 'n_lo': 5, 'd': None, 'pct': None, 'outcome': 'too few',
        'peers': [{'chan': 1, 'd': None, 'pct': None}, {'chan': 2, 'd': None, 'pct': None}]}
    assert distinctive({'outcome': 'held', 'peers': [{'pct': 99.0}, {'pct': 49.9}]})
    assert not distinctive({'outcome': 'held', 'peers': [{'pct': 99.0}, {'pct': 50.0}]})
    assert not distinctive({'outcome': 'not held', 'peers': [{'pct': 0.0}]})

    # kind: each kind, the boundaries, ties by peer order
    def hrec(held, pcts, zs, pct_u=50.0):
        return {'n_hi': 6, 'n_lo': 6, 'd_u': 0.5, 'pct_u': pct_u, 'held': held,
                'peers': [{'chan': ch, 'd': 0.0, 'pct': float(p), 'rho': 0.0, 'z': float(z)}
                          for ch, p, z in zip((7, 3, 9), pcts, zs)]}
    assert kind(None) == ('too few', None)
    assert kind(hrec(True, [96, 99, 99], [0, 0, 0])) == ('shared', 3)
    assert kind(hrec(True, [95, 10, 94.9], [5, 5, 5])) == ('shared', 7)
    assert kind(hrec(True, [94.9, 50, 0], [5, -5, 5])) == ('own', None)
    assert kind(hrec(False, [99, 99, 99], [-2.5, 2.5, 1])) == ('differs', 7)
    assert kind(hrec(False, [0, 0, 0], [1.0, -2.0, -3.0])) == ('differs', 9)
    assert kind(hrec(False, [0, 0, 0], [0, -2.0, 1.9])) == ('differs', 3)
    assert kind(hrec(False, [99, 99, 99], [1.99, -1.99, 0])) == ('none', None)

    # triggers: shared before differs, ties by lower index, none
    sh96 = hrec(True, [99, 0, 0], [0, 0, 0], pct_u=96.0)
    sh98 = hrec(True, [0, 0, 97], [0, 0, 0], pct_u=98.0)
    sh98b = hrec(True, [0, 96, 0], [0, 0, 0], pct_u=98.0)
    own99 = hrec(True, [0, 0, 0], [9, 9, 9], pct_u=99.0)
    dif3 = hrec(False, [0, 0, 0], [0, 0, -3.0], pct_u=80.0)
    dif3b = hrec(False, [0, 0, 0], [3.0, 0, 0], pct_u=10.0)
    dif25 = hrec(False, [0, 0, 0], [0, 2.5, 0], pct_u=90.0)
    non = hrec(False, [99, 99, 99], [0, 0, 0], pct_u=94.0)
    assert trigger_f([None, dif3, sh96, sh98, own99]) == (3, 9, 'shared')
    assert trigger_f([dif3, sh98b, sh98, None]) == (1, 3, 'shared')
    assert trigger_f([None, own99, dif25, dif3, dif3b, non]) == (3, 9, 'differs')
    assert trigger_f([dif25, own99]) == (0, 3, 'differs')
    assert trigger_f([None, own99, non]) is None and trigger_f([]) is None and trigger_f([None]) is None
    assert trigger_r([None, dif3, sh98, own99, sh96]) == 3 and trigger_r([None, sh98b, sh98, dif3]) == 1
    assert trigger_r([None, None]) is None and trigger_r([]) is None

    # panels: order, ties by lower id, photos of the group only, small groups
    ids_p = [1, 3, 5, 7, 8, 10, 12, 20, 25]
    ans_p = [0, 0, 1, 0, 0, 0, 0, 0, None]
    ans_2 = [0, 0, 1, 0, 1, 0, 0, 1, None]
    Rp = np.zeros((9, 3))
    Rp[:, 0] = [0.75, 0.0, 9.0, 0.5, 0.25, 0.0, 1.0, -0.5, -9.0]
    Rp[:, 1] = [0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.25, 0.0, 0.0]
    Rp[:, 2] = np.random.default_rng(5).normal(size=9)
    # e = column 0 - column 1 on group 0: 1: 0.5, 3: -0.25, 7: 0.5, 8: 0, 10: -0.25, 12: 0.75, 20: -0.5
    assert panel_f(Rp, ids_p, ans_p, 0, 0, 1) == ([12, 1, 7], [20, 3, 10])
    assert panel_f(Rp, ids_p, ans_p, 0, 1, 0) == ([20, 3, 10], [12, 1, 7])
    assert panel_f(Rp, ids_p, ans_p, 0, 0, 1, n=2) == ([12, 1], [20, 3])
    assert panel_f(Rp, ids_p, ans_p, 0, 0, 1, n=4) == ([12, 1, 7], [20, 3, 10])  # 7 photos: 3 each
    assert panel_f(Rp, ids_p, ans_p, 1, 0, 1) == ([], []) and panel_f(Rp, ids_p, ans_p, 2, 0, 1) == ([], [])
    assert panel_f(Rp, ids_p, ans_2, 0, 0, 1) == ([12, 1], [3, 10])  # 5 photos: 2 each
    q = [8, 2, 0, 5, 1, 7, 3, 6, 4]
    got = panel_f(Rp[q], [ids_p[t] for t in q], [ans_p[t] for t in q], 0, 0, 1)
    assert got == ([12, 1, 7], [20, 3, 10]) and all(type(i) is int for i in got[0] + got[1])
    assert panel_r(Rp, ids_p, ans_p, 0, 0) == ([12, 1, 7], [20, 3, 10])
    assert panel_r(Rp, ids_p, ans_p, 0, 1) == ([1, 3, 8], [7, 20, 1])
    assert panel_r(Rp, ids_p, ans_2, 0, 0) == ([12, 1], [3, 10]) and panel_r(Rp, ids_p, ans_p, 1, 0) == ([], [])
    assert panel_r(Rp[q], [ids_p[t] for t in q], [ans_p[t] for t in q], 0, 1) == ([1, 3, 8], [7, 20, 1])

    # parse_refined: case and spaces in splits, a split that is not a compared answer, the cap, bad entries
    def without(dct, k):
        return {x: v for x, v in dct.items() if x != k}
    ta = ['Raised or turned sideways', 'Facing the camera', 'Head not visible', 'No animal']
    g1 = {'splits': '  raised OR turned SIDEWAYS ', 'question': ' Is the tail visible? ',
          'answers': [' yes', 'no ', 'no animal'], 'most_active': 'YES', 'least_active': 'No'}
    g2 = {'splits': 'Head not visible', 'question': 'Is it indoors?', 'answers': ['indoors', 'outdoors'],
          'most_active': 'outdoors', 'least_active': 'indoors'}
    g3 = dict(g2, question='A third one?')
    w1 = {'question': 'Is the tail visible?', 'answers': ['yes', 'no', 'no animal'], 'most': 0, 'least': 1,
          'splits': 0}
    w2 = {'question': 'Is it indoors?', 'answers': ['indoors', 'outdoors'], 'most': 1, 'least': 0, 'splits': 2}
    bad = ['a string', 7, None, [g2], dict(g2, splits='Facing the camera'), dict(g2, splits='head'),
           dict(g2, splits=2), dict(g2, splits=None), without(g2, 'splits'), dict(g2, question=''),
           dict(g2, question={'q': 'x'}), dict(g2, answers=['indoors']), dict(g2, answers=['in', 'IN ']),
           dict(g2, most_active='maybe'), dict(g2, least_active=' OUTDOORS'), without(g2, 'least_active')]
    body = {'notes': ['a dog running', 'a dog sitting'], 'predictions': bad[:4] + [g1] + bad[4:] + [g2, g3]}
    reply = 'Notes and predictions:\n```json\n' + json.dumps(body) + '\n```'
    assert parse_refined(reply, ta, (0, 2)) == {'notes': ['a dog running', 'a dog sitting'],
                                                'predictions': [w1, w2]}, parse_refined(reply, ta, (0, 2))
    assert parse_refined(json.dumps({'predictions': [dict(g2, splits=' facing THE camera')]}), ta) == {
        'notes': None, 'predictions': [dict(w2, splits=1)]}
    assert parse_refined(json.dumps({'notes': 'text', 'predictions': [g1]}),
                         ['Raised or turned sideways', 'Head not visible']) == {'notes': 'text', 'predictions': [w1]}
    for b in [{'notes': ['x'], 'predictions': bad}, {'predictions': g2}, {'predictions': []}, {'notes': ['x']}]:
        assert parse_refined(json.dumps(b), ta, (0, 2)) is None, b
    assert parse_refined('no json here', ta, (0, 2)) is None
    assert parse_refined(json.dumps({'predictions': [g2]}) + ' or ' + json.dumps({'predictions': [bad[4]]}), ta,
                         (0, 2)) is None
    assert parse_refined(json.dumps({'predictions': [bad[4]]}) + ' or ' + json.dumps({'predictions': [g2]}), ta,
                         (0, 2))['predictions'] == [w2]
    assert _entry(dict(g2, question={'explanation': 'x', 'predictions': [g2]})) is None  # nothing nested is parsed
    assert _entry(g2) == without(w2, 'splits') and _entry('x') is None

    # conditional
    assert conditional([0, 1, None, 0, 2], [3, 1, 2, None, 0], 0) == [3, None, None, None, None]
    assert conditional([1, 1], [0, 2], 1) == [0, 2] and conditional([], [], 0) == []

    # table text: every verdict, both arms, a too-few row
    def pk(ds, ps):
        return [{'chan': ch, 'd': d, 'pct': p, 'rho': 0.0, 'z': 0.0} for ch, d, p in zip((7, 3, 9), ds, ps)]
    rec1 = {'n_hi': 9, 'n_lo': 7, 'd_u': 0.8361, 'pct_u': 97.5, 'held': True,
            'peers': pk([0.1, -0.004, 0.9], [60.2, 49.6, 98.4])}
    rec2 = {'n_hi': 9, 'n_lo': 7, 'd_u': 1.2, 'pct_u': 99.0, 'held': True,
            'peers': pk([0.5, 0.25, -0.5], [80.0, 70.0, 10.0])}
    rec3 = {'n_hi': 9, 'n_lo': 7, 'd_u': -0.3, 'pct_u': 12.0, 'held': False,
            'peers': pk([1.5, 0.0, 0.25], [99.0, 50.0, 70.0])}
    rec4 = {'n_hi': 9, 'n_lo': 7, 'd_u': 0.5, 'pct_u': 90.0, 'held': False,
            'peers': pk([0.5, 0.25, -0.5], [80.0, 70.0, 10.0])}
    rows_t = [
        {'question': 'Is the dog running?', 'answers': ['running', 'still', 'no dog'], 'most': 0, 'least': 1,
         'rec': rec1, 'kind': 'shared', 'peer_no': 3},
        {'question': 'Is it outdoors?', 'answers': ['indoors', 'outdoors'], 'most': 1, 'least': 0, 'rec': rec2,
         'kind': 'own', 'peer_no': None},
        {'question': 'Is the head raised?', 'answers': ['raised', 'level', 'cannot see'], 'most': 0, 'least': 2,
         'rec': rec3, 'kind': 'differs', 'peer_no': 1},
        {'question': 'Is there snow?', 'answers': ['yes', 'no'], 'most': 0, 'least': 1, 'rec': rec4, 'kind': 'none',
         'peer_no': None},
        {'question': 'Is there a person?', 'answers': ['yes', 'no'], 'most': 1, 'least': 0, 'rec': None,
         'kind': 'too few', 'peer_no': None}]
    tops = ['Prediction 1. Is the dog running?\nAnswers: running; still; no dog\n'
            'Predicted most active: running. Predicted least active: still.\n',
            'Prediction 2. Is it outdoors?\nAnswers: indoors; outdoors\n'
            'Predicted most active: outdoors. Predicted least active: indoors.\n',
            'Prediction 3. Is the head raised?\nAnswers: raised; level; cannot see\n'
            'Predicted most active: raised. Predicted least active: cannot see.\n',
            'Prediction 4. Is there snow?\nAnswers: yes; no\nPredicted most active: yes. Predicted least active: no.\n']
    last = ('Prediction 5. Is there a person?\nAnswers: yes; no\nPredicted most active: no. Predicted least active: '
            'yes.\nToo few photos to check.')
    want_f = '\n\n'.join([
        tops[0] + 'This unit: d +0.84, percentile 98. Similar unit 1: d +0.10, percentile 60. Similar unit 2: d -0.00, '
        'percentile 50. Similar unit 3: d +0.90, percentile 98.\nHeld for this unit and for similar unit 3.',
        tops[1] + 'This unit: d +1.20, percentile 99. Similar unit 1: d +0.50, percentile 80. Similar unit 2: d +0.25, '
        'percentile 70. Similar unit 3: d -0.50, percentile 10.\nHeld for this unit only.',
        tops[2] + 'This unit: d -0.30, percentile 12. Similar unit 1: d +1.50, percentile 99. Similar unit 2: d +0.00, '
        'percentile 50. Similar unit 3: d +0.25, percentile 70.\nDid not hold for this unit; this unit and similar '
        'unit 1 clearly differ on it.',
        tops[3] + 'This unit: d +0.50, percentile 90. Similar unit 1: d +0.50, percentile 80. Similar unit 2: d +0.25, '
        'percentile 70. Similar unit 3: d -0.50, percentile 10.\nDid not hold for this unit, and no similar unit '
        'clearly differs from it.',
        last])
    want_r = '\n\n'.join([
        tops[0] + 'This unit: d +0.84, percentile 98.\nHeld.', tops[1] + 'This unit: d +1.20, percentile 99.\nHeld.',
        tops[2] + 'This unit: d -0.30, percentile 12.\nDid not hold.',
        tops[3] + 'This unit: d +0.50, percentile 90.\nDid not hold.', last])
    assert table_text(rows_t, 'F') == want_f, table_text(rows_t, 'F')
    assert table_text(rows_t, 'R') == want_r, table_text(rows_t, 'R')
    assert table_text(json.loads(json.dumps(rows_t)), 'F') == want_f and table_text([], 'R') == ''

    # refinement content: block order, texts and image count, both arms
    def blk(i):
        return {'type': 'image', 'id': int(i)}
    grp = [{'answer': 'raised', 'hi': [(5, 'tabby', 100, 10), (9, 'tabby', 87, 20)], 'lo': [(12, 'tiger cat', 40, 90)]},
           {'answer': 'facing', 'hi': [(3, 'tabby', 80, 10)], 'lo': [(4, 'tiger cat', 0, 50), (6, 'tabby', 5, 60)]}]
    head_f = F_HEAD.replace('{m}', '512').replace('{j}', '2').replace('{k}', '3').replace('{table}', 'TABLE {k} {j}')
    more, less = 'This unit much more active than similar unit 3:', 'This unit much less active than similar unit 3:'
    assert refine_content('F', 512, 'TABLE {k} {j}', 2, 3, grp, block=blk) == [
        _text(head_f), _text('Answer 1: "raised"'), _text(more),
        _text('Photo 1.1 (tabby): this unit 100%, similar unit 3 10%'), blk(5),
        _text('Photo 1.2 (tabby): this unit 87%, similar unit 3 20%'), blk(9), _text(less),
        _text('Photo 1.3 (tiger cat): this unit 40%, similar unit 3 90%'), blk(12),
        _text('Answer 2: "facing"'), _text(more), _text('Photo 2.1 (tabby): this unit 80%, similar unit 3 10%'), blk(3),
        _text(less), _text('Photo 2.2 (tiger cat): this unit 0%, similar unit 3 50%'), blk(4),
        _text('Photo 2.3 (tabby): this unit 5%, similar unit 3 60%'), blk(6), _text(F_TASK.replace('{k}', '3'))]
    head_r = R_HEAD.replace('{m}', '512').replace('{j}', '2').replace('{table}', 'TABLE')
    assert '{' not in head_r and head_r.endswith('10000 photos.') and '{' not in R_TASK.split('JSON')[0]
    assert refine_content('R', 512, 'TABLE', 2, None, grp, block=blk) == [
        _text(head_r), _text('Answer 1: "raised"'), _text('Most active:'), _text('Photo 1.1 (tabby): 100%'), blk(5),
        _text('Photo 1.2 (tabby): 87%'), blk(9), _text('Least active:'), _text('Photo 1.3 (tiger cat): 40%'), blk(12),
        _text('Answer 2: "facing"'), _text('Most active:'), _text('Photo 2.1 (tabby): 80%'), blk(3),
        _text('Least active:'), _text('Photo 2.2 (tiger cat): 0%'), blk(4), _text('Photo 2.3 (tabby): 5%'), blk(6),
        _text(R_TASK)]
    full = [{'answer': f'a{g}', 'hi': [(10 * g + t, 'x', 50, 40) for t in range(3)],
             'lo': [(10 * g + 5 + t, 'x', 1, 2) for t in range(3)]} for g in range(2)]
    for arm in ARMS:
        cf = refine_content(arm, 480, 'T', 1, 1, full, block=blk)
        assert len(cf) == 32 and [b['id'] for b in cf if b['type'] == 'image'] == [0, 1, 2, 5, 6, 7, 10, 11, 12, 15,
                                                                                    16, 17]
        assert all(b['type'] == 'text' for b in cf if 'id' not in b)
        assert cf[-1]['text'] == (F_TASK.replace('{k}', '1') if arm == 'F' else R_TASK)

    # the answers content on D, and the photos shown in round 27
    qa = [{'question': 'Is it on the left?', 'answers': ['yes', 'no']}]
    dcn = d_content([5, 7], qa, block=blk)
    assert dcn[:-1] == proploop.answers_content([5, 7], qa, block=blk)[:-1] and len(dcn) == 5
    assert dcn[-1] == _text(A_PROMPT.replace('{b}', '2').replace('{questions}', proploop.questions_block(qa)))
    assert shown_photos({'range': [{'photos': [1, 2]}, {'photos': [3]}], 'peers': [{'photos': [2, 9]}]}) == {1, 2, 3, 9}
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

    secs = []
    for u in units:
        print('==', u['name'], flush=True)
        sec = run_unit(u, vals[u['layer']], labels, cats)
        secs.append(sec)
        (OUT / 'report.txt').write_text('\n\n'.join(secs) + '\n')
        print(sec, flush=True)
        gc.collect()
    print('report ->', OUT / 'report.txt')


if __name__ == '__main__':
    main()
