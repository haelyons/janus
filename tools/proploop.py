"""Readings become photo questions, answered blind, checked within class (spec: authors/spec_proploop.md).

Per unit (one channel of mixed3a-mixed5b): one call turns the unit's three CP explanations from
runs/rangeview/<unit>/readings.json into up to 8 questions about a photo, each with 2-5 answers. A second agent, which
never sees the explanations or any activation, answers them for every photo of the 12 classes with the highest class
mean of the unit, 12 photos per call. Per question the answers are compared with the unit's spatial max (scaled by its
max over those photos, centred within class): eta2, the share of the within-class variation the answers account for,
and d per answer, each as the unit's percentile among all channels of its layer on the same photos and answers; also
eta2 in two halves of the photos, and all of it after permuting the answers within class (null). Each explanation is
then revised by an agent shown the real table, and by one shown the null table. Outputs: runs/proploop/.

Decision printed in the report, per question: the unit's separation (eta2) percentile among its layer's channels,
> 95 -> ABOVE_95, < 5 -> BELOW_5, else WITHIN_5_95; NO_STATISTICS when fewer than 2 answers have 5 photos.

usage: python -I tools/proploop.py [--units-file authors/round25_units.json] [--selftest]
"""
import argparse
import gc
import hashlib
import json
import re
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))  # -I does not add the script dir
import bundlemax  # noqa: E402
import claude_call  # noqa: E402
import run_subjects  # noqa: E402

OUT = J / 'runs/proploop'
RANGEVIEW = J / 'runs/rangeview'
UNITS_FILE = 'authors/round25_units.json'
WEIGHTS = J / 'raw/weights/inception5h.pth'
THUMBS = J / 'bank/img/thumbs'
NAMES = J / 'bank/img/names.json'
ACTS = J / 'bank/img/acts.npz'
N_PHOTOS = 10000
LAYERS = ('mixed3a', 'mixed3b', 'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b')
NCH = {'mixed3a': 256, 'mixed3b': 480, 'mixed4a': 508, 'mixed4b': 512, 'mixed4c': 512, 'mixed4d': 528,
       'mixed4e': 832, 'mixed5a': 832, 'mixed5b': 1024}
BATCH = 64
WORKERS = 3
K_CLASSES = 12
SOURCE_KEYS = ('CP_s1', 'CP_s2', 'CP_s3')
SAMPLES = (1, 2, 3)
MAX_Q, MIN_Q = 8, 3
MIN_ANS, MAX_ANS = 2, 5
LETTERS = 'ABCDE'
PER_CALL = 12
MIN_N, MIN_HALF = 5, 2  # photos an answer needs to be compared: over the set, within a half
TABLES = ('real', 'null')
REVISIONS = [f'{t}_s{s}' for t in TABLES for s in SAMPLES]
PCT_LOW, PCT_HIGH = 5.0, 95.0

Q_PROMPT = (
    'Below are explanations, written independently, of what one unit (a single neuron) of an image classification '
    'network responds to.\n\n'
    '{explanations}\n\n'
    "The explanations will be checked on ordinary photos: someone who never sees the unit's activity will answer "
    'questions about each photo, and the answers will be compared with how strongly the unit responds to it. Write '
    'up to 8 such questions. Cover both whether a photo contains what the explanations describe and the ways in which '
    'what they describe can differ from one photo that contains it to another. Each question must be answerable by '
    'looking at a small photo, and needs 2 to 5 short answers that do not overlap, so that every photo fits exactly '
    'one of them; include an answer for photos to which the question does not apply. Reply with only JSON: '
    '{"questions": [{"question": "...", "answers": ["...", "..."]}]}')
A_PROMPT = (
    'Answer every question below for each of the {b} images above. For each question choose exactly one of its '
    'answers, by letter, or "?" if you cannot tell.\n\n'
    '{questions}\n\n'
    'Reply with only JSON: {"images": [{"id": n, "answers": ["A", ...]}]}, one entry per image, each with one answer '
    'per question, in question order.')
TABLE_HEAD = (
    '# Checks on ordinary photos\n\n'
    "Someone who never saw the unit's activity answered the questions below about each of {n} photos: all photos of "
    'the {K} photo classes on which the unit is most active on average ({classes}). For each answer: the number of '
    'photos given it, and d, how much more (positive) or less (negative) active the unit is on them than on the other '
    'photos, after removing the differences between classes, in standard deviations. Each question also has a '
    "separation score, the share of the unit's within-class variation that its answers account for. Each number comes "
    "with this unit's percentile among all {m} units of its layer, computed on the same photos and answers (above 95 or "
    'below 5 is unusual for this layer).')
R_PROMPT = (
    'Here is an explanation of what one unit (a single neuron) of an image classification network responds to, '
    'written by someone who looked at the photos on which it is most active:\n\n'
    '{explanation}\n\n'
    'Later, these checks were run:\n\n'
    '{table}\n\n'
    'Taking both into account, say what the unit responds to.')


# ---------- units, photos, values ----------

def parse_unit(s):
    """'mixedNx:c', a single channel of mixed3a-mixed5b -> {'name', 'layer', 'chan'}"""
    p = str(s).split(':')
    if len(p) != 2 or p[0] not in NCH or not re.fullmatch(r'[0-9]+', p[1]) or int(p[1]) >= NCH[p[0]]:
        raise ValueError(f'bad unit {s!r}')
    return {'name': str(s), 'layer': p[0], 'chan': int(p[1])}


def thumb(i):
    return THUMBS / f'{int(i):05d}.jpg'


def no_tf32():
    """full float32 on CUDA (no TF32 in convolutions or matmuls), as rangeview.no_tf32"""
    import torch
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def forward_values(model, layers, ids, device, make_input=None, tag='values'):
    """{layer: the spatial max of every channel of the module of that name, len(ids) x channels, float32}: one forward
    pass per batch of BATCH photos (input make_input(ids), default bundlemax.net_input, on device) with every layer
    hooked, stopped after the deepest (img_acts.Stop); outputs moved to the CPU"""
    import torch
    make_input = make_input or bundlemax.net_input
    Stop = bundlemax._bank().Stop
    layers = sorted(set(layers), key=LAYERS.index)
    deepest = layers[-1]
    mods = dict(model.named_modules())
    got = {}

    def hook_for(L):
        def hook(_m, _i, out):
            assert out.shape[1] == NCH[L], (L, tuple(out.shape))
            got[L] = out.amax(dim=(2, 3)).detach().cpu().numpy().astype(np.float32)
            if L == deepest:
                raise Stop
        return hook

    res = {L: [] for L in layers}
    hs = [mods[L].register_forward_hook(hook_for(L)) for L in layers]
    try:
        for b, s in enumerate(range(0, len(ids), BATCH)):
            x = make_input(ids[s:s + BATCH]).to(device)
            got.clear()
            try:
                with torch.no_grad():
                    model(x)
            except Stop:
                pass
            assert sorted(got) == sorted(layers), (sorted(got), layers)
            for L in layers:
                res[L].append(got[L])
            if b % 20 == 19:
                print(f'  {tag} {min(s + BATCH, len(ids))}/{len(ids)}', flush=True)
    finally:
        for h in hs:
            h.remove()
    return {L: np.concatenate(res[L]) for L in layers}


def layer_values(layers, get_model, ids, device=None, out=OUT, make_input=None):
    """{layer: values_<layer>.npy in out (len(ids) x channels, float32)}; the layers without a cache come from one
    forward_values pass over ids on device (default CUDA when available, else the CPU), the only use of get_model"""
    paths = {L: out / f'values_{L}.npy' for L in layers}
    missing = [L for L in layers if not paths[L].exists()]
    if missing:
        import torch
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        no_tf32()
        print('values of', missing, 'on', device, flush=True)
        new = forward_values(get_model(device), missing, ids, device, make_input=make_input)
        out.mkdir(parents=True, exist_ok=True)
        for L in missing:
            tmp = out / f'values_{L}.tmp.npy'
            np.save(tmp, np.ascontiguousarray(new[L], dtype=np.float32))
            tmp.replace(paths[L])
    vals = {}
    for L in layers:
        v = np.load(paths[L])
        if v.shape != (len(ids), NCH[L]) or v.dtype != np.float32:
            raise SystemExit(f'{paths[L]}: shape {v.shape}, dtype {v.dtype}; expected ({len(ids)}, {NCH[L]}) '
                             'float32; stopping')
        vals[L] = v
    return vals


def source_explanations(name, src=RANGEVIEW):
    """the explanations of CP_s1, CP_s2, CP_s3 in src/<unit>/readings.json, in that order; None when the file or any
    of them is missing or empty"""
    path = src / name / 'readings.json'
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    out = []
    for k in SOURCE_KEYS:
        r = rec.get(k)
        e = r.get('explanation') if isinstance(r, dict) else None
        if not isinstance(e, str) or not e.strip():
            return None
        out.append(e)
    return out


# ---------- photo set and statistics ----------

def top_classes(unit_vals, labels, k=K_CLASSES):
    """the k labels (ints) with the highest mean of unit_vals over their photos, ties by lower label; both arrays over
    all photos"""
    v = np.asarray(unit_vals, dtype=np.float64)
    lab = np.asarray(labels)
    assert v.ndim == 1 and v.shape == lab.shape, (v.shape, lab.shape)
    means = {int(x): float(v[lab == x].mean()) for x in np.unique(lab)}
    return sorted(means, key=lambda x: (-means[x], x))[:k]


def photo_set(unit_vals, labels, k=K_CLASSES):
    """(the set's classes in order, the set: every photo of those labels, in photo index order)"""
    classes = top_classes(unit_vals, labels, k)
    return classes, np.flatnonzero(np.isin(np.asarray(labels), classes)).astype(np.int64)


def centre(X, labels):
    """R: each column of X (n x m) divided by its max (all 0 if the max is 0), minus its mean over the rows with the
    same label"""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    assert X.ndim == 2 and len(labels) == X.shape[0], (X.shape, labels.shape)
    mx = X.max(axis=0)
    v = np.where(mx != 0, X / np.where(mx != 0, mx, 1.0), 0.0)
    R = np.empty_like(v)
    for lab in np.unique(labels):
        on = labels == lab
        R[on] = v[on] - v[on].mean(axis=0)
    return R


def _codes(ans):
    """answers as an int array, -1 for None"""
    return np.array([-1 if a is None else int(a) for a in ans], dtype=np.int64)


def answer_counts(ans, k):
    """([photos per answer 0..k-1], photos with None)"""
    codes = _codes(ans)
    return [int(np.count_nonzero(codes == a)) for a in range(k)], int(sum(a is None for a in ans))


def _eta2(Rk, ck, used):
    """eta2 of every column of Rk over its rows (each with an answer in used): sum over a of n_a (mean_a - mean)^2 /
    sum of (R - mean)^2; 0 where the denominator is 0"""
    mu = Rk.mean(axis=0)
    num = np.zeros(Rk.shape[1])
    for a in used:
        on = ck == a
        num += np.count_nonzero(on) * (Rk[on].mean(axis=0) - mu) ** 2
    den = ((Rk - mu) ** 2).sum(axis=0)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def question_stats(R, ans, k):
    """R n x m, ans a length-n list of int or None, k answers. Compared answers: those with at least MIN_N photos; kept
    photos: those with a compared answer. None when fewer than 2 are compared, else {'compared': [answer indices],
    'eta2': array of m, 'd': {answer index: array of m}}, d = (mean over the answer - mean over the other kept photos)
    / sd over the kept photos (ddof 1), 0 where that sd is 0"""
    R = np.asarray(R, dtype=np.float64)
    codes = _codes(ans)
    assert R.ndim == 2 and len(codes) == R.shape[0], (R.shape, len(codes))
    compared = [a for a in range(k) if np.count_nonzero(codes == a) >= MIN_N]
    if len(compared) < 2:
        return None
    keep = np.isin(codes, compared)
    Rk, ck = R[keep], codes[keep]
    sd = Rk.std(axis=0, ddof=1)
    d = {}
    for a in compared:
        on = ck == a
        diff = Rk[on].mean(axis=0) - Rk[~on].mean(axis=0)
        d[a] = np.where(sd > 0, diff / np.where(sd > 0, sd, 1.0), 0.0)
    return {'compared': compared, 'eta2': _eta2(Rk, ck, compared), 'd': d}


def percentile(values, j):
    """100 x the share of values strictly below values[j]"""
    v = np.asarray(values, dtype=np.float64).ravel()
    return float(100.0 * np.count_nonzero(v < v[j]) / v.size)


def half_eta2_pct(R, ans, compared, in_half, c):
    """column c's eta2 percentile over the photos in_half whose answer is one of the compared answers with at least
    MIN_HALF photos in the half; None when fewer than 2 such answers"""
    codes = _codes(ans)
    h = np.asarray(in_half, dtype=bool)
    used = [a for a in compared if np.count_nonzero(h & (codes == a)) >= MIN_HALF]
    if len(used) < 2:
        return None
    keep = h & np.isin(codes, used)
    return percentile(_eta2(np.asarray(R, dtype=np.float64)[keep], codes[keep], used), c)


def question_record(R, c, ans, k, half):
    """the unit's (column c) numbers for one question: counts, could-not-tell count, compared answers, its eta2 and d
    with their percentiles among the m columns, and its eta2 percentile in each half (half: 0 or 1 per photo); the
    statistics None or empty when the question has none"""
    counts, unclear = answer_counts(ans, k)
    rec = {'counts': counts, 'unclear': unclear, 'compared': [], 'eta2': None, 'eta2_pct': None, 'd': {},
           'd_pct': {}, 'halves': [None, None]}
    st = question_stats(R, ans, k)
    if st is None:
        return rec
    cmp_ = st['compared']
    half = np.asarray(half)
    rec.update(compared=cmp_, eta2=float(st['eta2'][c]), eta2_pct=percentile(st['eta2'], c),
               d={a: float(st['d'][a][c]) for a in cmp_}, d_pct={a: percentile(st['d'][a], c) for a in cmp_},
               halves=[half_eta2_pct(R, ans, cmp_, half == h, c) for h in (0, 1)])
    return rec


def permute_within(rows, set_labels, class_order, rng):
    """for each class in class_order, the photos of that class (in set order) receive the rows of those same photos
    in the order rng.permutation(their count); rows: one answer row per photo of the set"""
    labs = np.asarray(set_labels)
    assert len(rows) == len(labs), (len(rows), len(labs))
    out = list(rows)
    for cl in class_order:
        idx = np.flatnonzero(labs == cl)
        p = rng.permutation(len(idx))
        for t, s in zip(idx.tolist(), idx[p].tolist()):
            out[t] = rows[s]
    return out


def decide(p):
    """Neutral label on a percentile among the layer's channels: > 95 ABOVE_95, < 5 BELOW_5, else WITHIN_5_95;
    None (no statistics) NO_STATISTICS"""
    if p is None:
        return 'NO_STATISTICS'
    return 'ABOVE_95' if p > PCT_HIGH else 'BELOW_5' if p < PCT_LOW else 'WITHIN_5_95'


# ---------- prompts and replies ----------

def fill(template, **fields):
    """template with each {name} of fields replaced by str(value), in one pass (values are not searched again)"""
    pat = re.compile(r'\{(' + '|'.join(re.escape(k) for k in fields) + r')\}')
    return pat.sub(lambda mo: str(fields[mo.group(1)]), template)


def explanations_text(expls):
    return '\n\n'.join(f'Explanation {s}: {t}' for s, t in enumerate(expls, 1))


def questions_content(expls):
    return [{'type': 'text', 'text': fill(Q_PROMPT, explanations=explanations_text(expls))}]


def parse_questions(text):
    """the kept questions [{'question', 'answers'}] of the last JSON object with 'questions': entries that are objects
    with a non-empty question and 2-5 answers, each non-empty and distinct after strip, stored stripped; the first
    MAX_Q of them. None when fewer than MIN_Q are kept or no object is found"""
    o = bundlemax.last_obj(text, ('questions',))
    if o is None or not isinstance(o['questions'], list):
        return None
    kept = []
    for e in o['questions']:
        if not isinstance(e, dict):
            continue
        q, a = e.get('question'), e.get('answers')
        if not isinstance(q, str) or not q.strip():
            continue
        if not isinstance(a, list) or not MIN_ANS <= len(a) <= MAX_ANS or not all(isinstance(x, str) for x in a):
            continue
        a = [x.strip() for x in a]
        if not all(a) or len(set(a)) != len(a):
            continue
        kept.append({'question': q.strip(), 'answers': a})
    kept = kept[:MAX_Q]
    return kept if len(kept) >= MIN_Q else None


def questions_block(questions):
    """one line per question: 'Q{j}. {question} A) {answer} B) {answer} ...'"""
    return '\n'.join(f'Q{j}. {q["question"]} ' + ' '.join(f'{LETTERS[i]}) {a}' for i, a in enumerate(q['answers']))
                     for j, q in enumerate(questions, 1))


def answer_batches(unit, set_ids, per=PER_CALL):
    """the set's photo ids in the order default_rng(crc32(unit) + 1).permutation(n), cut into batches of per"""
    ids = np.asarray(set_ids, dtype=np.int64)
    order = ids[np.random.default_rng(zlib.crc32(unit.encode()) + 1).permutation(len(ids))]
    return [order[s:s + per].tolist() for s in range(0, len(order), per)]


def answers_content(batch, questions, block=None):
    """'Image k:' and the photo (run_subjects.img_block, or block(id)) per photo of the batch, then the prompt"""
    block = block or (lambda i: run_subjects.img_block(thumb(i)))
    content = []
    for k, i in enumerate(batch, 1):
        content += [{'type': 'text', 'text': f'Image {k}:'}, block(i)]
    content.append({'type': 'text', 'text': fill(A_PROMPT, b=len(batch), questions=questions_block(questions))})
    return content


def parse_answers(text, b, questions):
    """b lists (image 1 first), one entry per question: the answer index (0 for A) or None for '?'; None when the last
    JSON object with 'images' is not a list with ids exactly 1..b, each once, each with one valid letter or '?' per
    question (stripped, upper-cased)"""
    o = bundlemax.last_obj(text, ('images',))
    if o is None or not isinstance(o['images'], list) or len(o['images']) != b:
        return None
    out = [None] * b
    for e in o['images']:
        if not isinstance(e, dict):
            return None
        i, ans = e.get('id'), e.get('answers')
        if not isinstance(i, int) or isinstance(i, bool) or not 1 <= i <= b or out[i - 1] is not None:
            return None
        if not isinstance(ans, list) or len(ans) != len(questions):
            return None
        row = []
        for x, q in zip(ans, questions):
            if not isinstance(x, str):
                return None
            s = x.strip().upper()
            if s == '?':
                row.append(None)
            elif len(s) == 1 and s in LETTERS[:len(q['answers'])]:
                row.append(LETTERS.index(s))
            else:
                return None
        out[i - 1] = row
    return out


def revision_content(explanation, table):
    return [{'type': 'text', 'text': fill(R_PROMPT, explanation=explanation, table=table)},
            {'type': 'text', 'text': run_subjects.TASK}]


# ---------- table ----------

def _whole(x):
    """a whole number, round half to even (Python round)"""
    return int(round(float(x)))


def _at(dct, a):
    """dct[a], or dct[str(a)] for keys read back from JSON"""
    return dct[a] if a in dct else dct[str(a)]


def table_text(classes, n, m, questions, qs):
    """the table text: the head, then one block per question. qs: per question None (no statistics) or {'counts',
    'unclear', 'compared', 'd', 'd_pct', 'eta2_pct'}; a dict with eta2_pct None is also a question without
    statistics, shown with its counts"""
    assert len(questions) == len(qs), (len(questions), len(qs))
    blocks = [fill(TABLE_HEAD, n=n, K=len(classes), classes=', '.join(classes), m=m)]
    for j, (q, s) in enumerate(zip(questions, qs), 1):
        stats = s is not None and s.get('eta2_pct') is not None
        L = [f'Q{j}. {q["question"]}',
             f'separation: percentile {_whole(s["eta2_pct"])}' if stats else
             'separation: too few photos per answer to compare']
        comp = set(s['compared']) if stats else set()
        for a, text in enumerate(q['answers']):
            if s is None:  # no counts given
                L.append(f'- {text}: (too few to compare)')
            elif a in comp:
                L.append(f'- {text}: {_whole(s["counts"][a])} photos, d {float(_at(s["d"], a)):+.2f}, '
                         f'percentile {_whole(_at(s["d_pct"], a))}')
            else:
                L.append(f'- {text}: {_whole(s["counts"][a])} photos (too few to compare)')
        if s is not None and s.get('unclear'):
            L.append(f'- could not tell: {_whole(s["unclear"])} photos')
        blocks.append('\n'.join(L))
    return '\n\n'.join(blocks)


# ---------- calls and caches ----------

def sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(path)


def attempt(content, parse, call):
    """a fresh single-turn call(system prompt '', [content]); one more in a fresh conversation when parse(reply) is
    None. A reply with is_error raises, so nothing of it is cached. -> (parsed or None, [reply texts])"""
    texts = []
    for _ in range(2):
        r = call('', [content])[0]
        if r.get('is_error'):
            raise RuntimeError(f'error reply, not cached: {str(r.get("text"))[:80]}')
        texts.append(r['text'])
        p = parse(r['text'])
        if p is not None:
            return p, texts
    return None, texts


def get_questions(name, expls, udir, call=None):
    """the questions record, from udir/questions.json or one attempt() (cached, valid or not); an is_error reply
    raises and is not cached; a cache made from other content stops the run"""
    path = udir / 'questions.json'
    content = questions_content(expls)
    sha = sha256(content)
    if path.exists():
        rec = json.loads(path.read_text())
        if rec['content_sha256'] != sha:
            raise SystemExit(f'{path}: cached questions were made from different content; stopping')
        return rec
    print(name, 'questions', flush=True)
    qs, texts = attempt(content, parse_questions, call or claude_call.converse)
    rec = {'unit': name, 'model': claude_call.MODEL, 'system_prompt': '', 'content_sha256': sha, 'texts': texts,
           'text': texts[-1], 'questions': qs, 'valid': qs is not None}
    save_json(rec, path)
    return rec


def cached_calls(name, path, jobs, call=None):
    """jobs: [(key, content, parse, fields)]. Each key not in the JSON cache at path: attempt() in up to WORKERS
    threads, cached as {'key', **fields(parsed), 'model', 'system_prompt', 'content_sha256', 'texts', 'text',
    'valid'}; a call that raised (is_error) leaves its key out, so a rerun redoes it. A cached key made from other
    content stops the run. -> the cache"""
    call = call or claude_call.converse
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = []
    for key, content, parse, fields in jobs:
        sha = sha256(content)
        if key not in cache:
            todo.append((key, content, parse, fields, sha))
        elif cache[key]['content_sha256'] != sha:
            raise SystemExit(f'{path}: cached {key} was made from different content; stopping')
    if todo:
        print(name, path.name, [t[0] for t in todo], flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(attempt, content, parse, call): (key, fields, sha) for key, content, parse, fields, sha in todo}
        for f in as_completed(futs):
            key, fields, sha = futs[f]
            try:
                parsed, texts = f.result()
            except Exception as err:  # not cached, so a rerun tries again
                print(f'{name} {path.name} {key} failed: {err}', flush=True)
                continue
            cache[key] = {'key': key, **fields(parsed), 'model': claude_call.MODEL, 'system_prompt': '',
                          'content_sha256': sha, 'texts': texts, 'text': texts[-1], 'valid': parsed is not None}
            save_json(cache, path)
    return cache


def _answer_parser(b, questions):
    def parse(text):
        return parse_answers(text, b, questions)
    return parse


def _answer_fields(batch, nq):
    """the cached fields of an answers batch: its photos and answers, all None when it failed"""
    def fields(p):
        return {'photos': [int(i) for i in batch], 'answers': p if p is not None else [[None] * nq for _ in batch],
                'failed': p is None}
    return fields


def _revision_fields(o):
    return {'class': None if o is None else o.get('class'), 'explanation': None if o is None else o['explanation'],
            'parsed': o}


# ---------- one unit, report ----------

def _pct(p):
    return '-' if p is None else f'{p:.1f}'


def failed_section(name, reason):
    return f'== {name}\nfailed: {reason}'


def section(name, layer, m, n, classes, cats, qs, st, n_fail, n_batches, rev):
    L = [f'== {name}  layer {layer} ({m} channels), {n} photos, answer batches failed after a retry: '
         f'{n_fail} of {n_batches}',
         'classes: ' + ', '.join(f'{cats[x]} ({x})' for x in classes)]
    for j, q in enumerate(qs):
        r, z = st['real'][j], st['null'][j]
        L.append(f'Q{j + 1}. {q["question"]}')
        for a, text in enumerate(q['answers']):
            if a in r['compared']:
                L.append(f'   {text}: {r["counts"][a]} photos, d {r["d"][a]:+.2f}, percentile {r["d_pct"][a]:.1f}')
            else:
                L.append(f'   {text}: {r["counts"][a]} photos (too few to compare)')
        if r['unclear']:
            L.append(f'   could not tell: {r["unclear"]} photos')
        if r['eta2_pct'] is None:
            L.append(f'   separation: too few photos per answer to compare -> {decide(None)}')
        else:
            L.append(f'   separation: eta2 {r["eta2"]:.4f}, percentile {r["eta2_pct"]:.1f} -> {decide(r["eta2_pct"])}'
                     f'; halves {_pct(r["halves"][0])}, {_pct(r["halves"][1])}; null percentile '
                     f'{_pct(z["eta2_pct"])} -> {decide(z["eta2_pct"])}')
    tally = {t: [decide(x['eta2_pct']) for x in st[t]] for t in TABLES}
    L.append('separation percentile above 95: ' + ', '.join(f'{t} {tally[t].count("ABOVE_95")}' for t in TABLES)
             + '; below 5: ' + ', '.join(f'{t} {tally[t].count("BELOW_5")}' for t in TABLES)
             + f' (of {len(qs)} questions)')
    for rid in REVISIONS:
        rec = rev.get(rid)
        if rec is None:
            L.append(f'{rid}  no revision (call failed, not cached)')
        elif rec['explanation'] is None:
            L.append(f'{rid}  no revision (reply not parsed after a retry)')
        else:
            L.append(f'{rid}  {rec["class"]}  ' + ' '.join(rec['explanation'][:160].split()))
    return '\n'.join(L)


def run_unit(u, V, labels, names, cats, call=None, out=OUT, src=RANGEVIEW, block=None):
    """one unit (u from parse_unit; V: its layer's values, photos x channels): explanations, photo set, questions,
    answers, statistics (stats.json), the real and null tables, revisions. -> (report section, the revisions cache,
    or None when the unit failed or its answers are not all cached yet)"""
    name, c = u['name'], u['chan']
    udir = out / name
    expls = source_explanations(name, src)
    if expls is None:
        return failed_section(name, 'no explanation for each of ' + ', '.join(SOURCE_KEYS)), None
    V, labels = np.asarray(V), np.asarray(labels)
    m = V.shape[1]
    classes, set_ids = photo_set(V[:, c], labels)
    set_labels = labels[set_ids]
    R = centre(V[set_ids], set_labels)
    n = len(set_ids)

    # questions
    try:
        qrec = get_questions(name, expls, udir, call)
    except Exception as err:  # an is_error reply: not cached, so a rerun tries again
        return failed_section(name, f'questions call failed, not cached: {err}'), None
    qs = qrec['questions']
    if qs is None:
        return failed_section(name, 'no valid questions after one retry'), None
    nq = len(qs)

    # answers
    batches = answer_batches(name, set_ids)
    keys = [str(b) for b in range(1, len(batches) + 1)]
    jobs = [(key, answers_content(batch, qs, block), _answer_parser(len(batch), qs), _answer_fields(batch, nq))
            for key, batch in zip(keys, batches)]
    ac = cached_calls(name, udir / 'answers.json', jobs, call)
    missing = [k for k in keys if k not in ac]
    if missing:
        return failed_section(name, f'answers incomplete: batches {missing} not cached (call errors); rerun'), None
    row_of = {}
    for key, batch in zip(keys, batches):
        if ac[key]['photos'] != batch or len(ac[key]['answers']) != len(batch):
            raise SystemExit(f'{udir / "answers.json"}: batch {key} is for other photos; stopping')
        row_of.update(zip(batch, ac[key]['answers']))
    rows = [row_of[i] for i in set_ids.tolist()]
    failed = [k for k in keys if ac[k]['failed']]

    # statistics, real and null
    null_rows = permute_within(rows, set_labels, classes, np.random.default_rng(zlib.crc32(name.encode())))
    half = np.array([zlib.crc32(names[i].encode()) % 2 for i in set_ids.tolist()], dtype=np.int64)
    st = {t: [question_record(R, c, [r[q] for r in rr], len(qs[q]['answers']), half) for q in range(nq)]
          for t, rr in zip(TABLES, (rows, null_rows))}
    cnames = [cats[x] for x in classes]
    tables = {t: table_text(cnames, n, m, qs, st[t]) for t in TABLES}
    save_json({'unit': name, 'layer': u['layer'], 'chan': c, 'm': m, 'n': n, 'classes': classes,
               'class_names': cnames, 'set_ids': set_ids.tolist(), 'answer_batches': len(batches),
               'failed_batches': failed,
               'questions': [{'question': q['question'], 'answers': q['answers'], 'real': st['real'][j],
                              'null': st['null'][j]} for j, q in enumerate(qs)],
               'tables': tables}, udir / 'stats.json')

    # revisions
    rjobs = [(f'{t}_s{s}', revision_content(expls[s - 1], tables[t]), bundlemax.parse_expl, _revision_fields)
             for t in TABLES for s in SAMPLES]
    rev = cached_calls(name, udir / 'revisions.json', rjobs, call)
    return section(name, u['layer'], m, n, classes, cats, qs, st, len(failed), len(batches), rev), rev


# ---------- selftest ----------

def selftest():
    import math
    import statistics
    import tempfile
    import threading
    from collections import OrderedDict

    def para(*pieces):
        return ' '.join(pieces)

    # prompt texts, verbatim from the spec: wrapped lines joined by single spaces, paragraph breaks kept
    assert Q_PROMPT == '\n\n'.join([
        para('Below are explanations, written independently, of what one unit (a single neuron) of an image',
             'classification network responds to.'),
        '{explanations}',
        para("The explanations will be checked on ordinary photos: someone who never sees the unit's activity",
             'will answer questions about each photo, and the answers will be compared with how strongly the unit',
             'responds to it. Write up to 8 such questions. Cover both whether a photo contains what the explanations',
             'describe and the ways in which what they describe can differ from one photo that contains it to',
             'another. Each question must be answerable by looking at a small photo, and needs 2 to 5 short answers',
             'that do not overlap, so that every photo fits exactly one of them; include an answer for photos to',
             'which the question does not apply. Reply with only JSON:',
             '{"questions": [{"question": "...", "answers": ["...", "..."]}]}')])
    assert A_PROMPT == '\n\n'.join([
        para('Answer every question below for each of the {b} images above. For each question choose exactly',
             'one of its answers, by letter, or "?" if you cannot tell.'),
        '{questions}',
        para('Reply with only JSON: {"images": [{"id": n, "answers": ["A", ...]}]}, one entry per image, each with',
             'one answer per question, in question order.')])
    assert TABLE_HEAD == '\n\n'.join([
        '# Checks on ordinary photos',
        para("Someone who never saw the unit's activity answered the questions below about each of {n} photos:",
             'all photos of the {K} photo classes on which the unit is most active on average ({classes}). For each',
             'answer: the number of photos given it, and d, how much more (positive) or less (negative) active the',
             'unit is on them than on the other photos, after removing the differences between classes, in standard',
             "deviations. Each question also has a separation score, the share of the unit's within-class variation",
             "that its answers account for. Each number comes with this unit's percentile among all {m} units of its",
             'layer, computed on the same photos and answers (above 95 or below 5 is unusual for this layer).')])
    assert R_PROMPT == '\n\n'.join([
        para('Here is an explanation of what one unit (a single neuron) of an image classification network',
             'responds to, written by someone who looked at the photos on which it is most active:'),
        '{explanation}', 'Later, these checks were run:', '{table}',
        'Taking both into account, say what the unit responds to.'])
    assert REVISIONS == ['real_s1', 'real_s2', 'real_s3', 'null_s1', 'null_s2', 'null_s3'] and WORKERS == 3
    assert fill('{a} {b} {c}', a='{b}', b='x') == '{b} x {c}'  # one pass; fields not given stay

    # units
    assert parse_unit('mixed4c:447') == {'name': 'mixed4c:447', 'layer': 'mixed4c', 'chan': 447}
    assert parse_unit('mixed3a:0')['chan'] == 0 and parse_unit('mixed5b:1023')['layer'] == 'mixed5b'
    for bad in ['mixed4c:512', 'mixed5b:1024', 'mixed4c:-1', 'mixed4c', 'mixed4c:x', 'mixed2a:1', 'sum:mixed4c:1+2',
                'mixed4c:1+2', 'mixed4c: 3']:
        try:
            parse_unit(bad)
        except ValueError:
            continue
        raise AssertionError('accepted ' + bad)

    # top classes: highest class mean, ties by lower label, fewer when fewer labels
    tv = [3, 1, 2, 2, 5, 0, 2, 2, 7]
    tl = [7, 7, 4, 4, 9, 9, 1, 1, 0]
    assert top_classes(tv, tl) == [0, 9, 1, 4, 7], top_classes(tv, tl)  # 7, 2.5, then 2.0 for 1, 4 and 7
    assert top_classes(np.array(tv), np.array(tl), k=3) == [0, 9, 1]
    assert top_classes([1.0, 1.0, 1.0], [5, 3, 8]) == [3, 5, 8] and top_classes([0.0, -1.0], [2, 1], k=1) == [2]
    cls_, ids_ = photo_set(np.array(tv, dtype=np.float32), np.array(tl), k=2)
    assert cls_ == [0, 9] and ids_.tolist() == [4, 5, 8]

    # questions: bad entries dropped, duplicates after strip, the cap, too few
    good = [{'question': ' Is there a dog? ', 'answers': [' yes', 'no ', 'not applicable']},
            {'question': 'Where is it?', 'answers': ['indoors', 'outdoors', 'cannot see']},
            {'question': 'Is it bright?', 'answers': ['bright', 'dark']}]
    bad = ['a string', {'question': '', 'answers': ['a', 'b']}, {'question': '   ', 'answers': ['a', 'b']},
           {'question': 5, 'answers': ['a', 'b']}, {'answers': ['a', 'b']}, {'question': 'One?', 'answers': ['x']},
           {'question': 'Six?', 'answers': ['1', '2', '3', '4', '5', '6']},
           {'question': 'Dup?', 'answers': ['big', ' big']}, {'question': 'Blank?', 'answers': ['a', '  ']},
           {'question': 'Num?', 'answers': ['a', 3]}, {'question': 'Str?', 'answers': 'ab'}, {'question': 'None?'}]
    want = [{'question': 'Is there a dog?', 'answers': ['yes', 'no', 'not applicable']}, good[1], good[2]]
    reply = ('Sure.\n```json\n' + json.dumps({'questions': [bad[0], good[0], bad[1], bad[2], good[1]] + bad[3:]
                                                           + [good[2]]}) + '\n```')
    assert parse_questions(reply) == want, parse_questions(reply)
    assert parse_questions(json.dumps({'questions': good[:2] + bad})) is None
    assert parse_questions('no json') is None and parse_questions('{"questions": "x"}') is None
    assert parse_questions(json.dumps({'questions': good}) + ' or ' + json.dumps({'questions': good[:2]})) is None
    assert parse_questions(json.dumps({'questions': good[:2]}) + ' or ' + json.dumps({'questions': good})) == want
    many = [{'question': f'Question {i}?', 'answers': ['1', '2', '3', '4', '5'] if i == 0 else ['yes', 'no']}
            for i in range(10)]
    assert parse_questions(json.dumps({'questions': many})) == many[:8]

    # the questions block and the answers parse
    qa = [{'question': 'Is there a dog?', 'answers': ['yes', 'no']},
          {'question': 'Where is it?', 'answers': ['indoors', 'outdoors', 'not applicable']}]
    assert questions_block(qa) == ('Q1. Is there a dog? A) yes B) no\n'
                                   'Q2. Where is it? A) indoors B) outdoors C) not applicable')

    def imgs(*entries):
        return 'Answers:\n' + json.dumps({'images': [{'id': i, 'answers': a} for i, a in entries]})
    assert parse_answers(imgs((2, ['b', '?']), (1, ['A', ' c '])), 2, qa) == [[0, 2], [1, None]]
    assert parse_answers(imgs((1, [' ? ', '?'])), 1, qa) == [[None, None]]
    for bad_r in [imgs((1, ['A', 'A'])), imgs((1, ['A', 'A']), (1, ['B', 'B'])),
                  imgs((1, ['A', 'A']), (3, ['B', 'B'])), imgs((1, ['A', 'A']), (2, ['B'])),
                  imgs((1, ['A', 'A']), (2, ['B', 'B', 'A'])), imgs((1, ['C', 'A']), (2, ['B', 'B'])),
                  imgs((1, ['A', 'D']), (2, ['B', 'B'])), imgs((1, ['A', '']), (2, ['B', 'B'])),
                  imgs((1, ['A', 'AB']), (2, ['B', 'B'])), imgs((1, ['A', 1]), (2, ['B', 'B'])),
                  imgs((1, ['A', None]), (2, ['B', 'B'])), imgs(('1', ['A', 'A']), (2, ['B', 'B'])),
                  imgs((True, ['A', 'A']), (2, ['B', 'B'])), imgs((1, ['A', 'A']), (2, ['B', 'B']), (3, ['A', 'A'])),
                  imgs((1, ['A', 'A']), (2, ['B', 'B'])) + ' then ' + imgs((1, ['A', 'A'])),
                  '{"images": {"id": 1}}', '{"images": [1, 2]}', '{"images": [{"id": 1}, {"id": 2}]}', 'no json']:
        assert parse_answers(bad_r, 2, qa) is None, bad_r

    # contents and batches
    qc = questions_content(['one thing', 'two', 'three'])
    assert qc == [{'type': 'text', 'text': Q_PROMPT.replace(
        '{explanations}', 'Explanation 1: one thing\n\nExplanation 2: two\n\nExplanation 3: three')}]
    ac = answers_content([5, 7], qa, block=lambda i: {'type': 'image', 'id': i})
    assert ac == [{'type': 'text', 'text': 'Image 1:'}, {'type': 'image', 'id': 5},
                  {'type': 'text', 'text': 'Image 2:'}, {'type': 'image', 'id': 7},
                  {'type': 'text', 'text': A_PROMPT.replace('{b}', '2').replace('{questions}', questions_block(qa))}]
    assert ac[-1]['text'].startswith('Answer every question below for each of the 2 images above.')
    rc = revision_content('an {table} explanation', 'TABLE')
    assert rc == [{'type': 'text', 'text': R_PROMPT.replace('{table}', 'TABLE').replace(
        '{explanation}', 'an {table} explanation')}, {'type': 'text', 'text': run_subjects.TASK}]
    sid = np.array([3, 8, 9, 20, 21, 40] + list(range(50, 70)))
    bt = answer_batches('mixed4c:447', sid)
    assert [len(b) for b in bt] == [12, 12, 2]
    assert sum(bt, []) == sid[np.random.default_rng(zlib.crc32(b'mixed4c:447') + 1).permutation(26)].tolist()

    # centre: scaled by the column max (zero column stays 0), centred within label
    Xc = np.array([[2, 0, 1], [1, 0, 4], [4, 0, 3], [3, 0, 2]], dtype=np.float32)
    Rc = centre(Xc, [5, 2, 5, 2])
    assert np.array_equal(Rc, [[-0.25, 0, -0.25], [-0.25, 0, 0.25], [0.25, 0, 0.25], [0.25, 0, -0.25]]), Rc
    assert Rc.dtype == np.float64

    # question statistics on a hand case: answer 2 (2 photos) and None dropped; zero column; zero sd on kept photos
    ans_h = [0, 1, 2, 0, None, 1, 0, 1, 2, 0, 1, 0, 1]
    col0 = [1, 0, 100, 2, 50, 0, 3, 1, -100, 2, -1, 2, 0]
    col2 = [0.5, 0.5, 7, 0.5, 9, 0.5, 0.5, 0.5, -7, 0.5, 0.5, 0.5, 0.5]
    col3 = [1, 0, -3, 2, 0, 0, 3, 1, 8, 2, -1, 2, 0]
    Rh = np.array([col0, [0] * 13, col2, col3], dtype=np.float64).T
    sh = question_stats(Rh, ans_h, 3)
    assert sh['compared'] == [0, 1] and sorted(sh['d']) == [0, 1] and sh['eta2'].shape == (4,)
    r14 = 6 / math.sqrt(14)
    assert np.allclose(sh['eta2'], [5 / 7, 0, 0, 5 / 7], rtol=0, atol=1e-12), sh['eta2']
    assert np.allclose(sh['d'][0], [r14, 0, 0, r14], rtol=0, atol=1e-12), sh['d'][0]
    assert np.allclose(sh['d'][1], [-r14, 0, 0, -r14], rtol=0, atol=1e-12), sh['d'][1]
    assert sh['eta2'][1] == 0 and sh['eta2'][2] == 0 and sh['d'][0][2] == 0 and sh['d'][1][1] == 0
    assert question_stats(Rh, [0] * 5 + [1] * 4 + [None] * 4, 3) is None  # answer 1 has 4 photos
    assert question_stats(Rh, [None] * 13, 2) is None and question_stats(Rh, [0] * 13, 2) is None
    s5 = question_stats(Rh, ans_h, 5)
    assert s5['compared'] == [0, 1] and np.array_equal(s5['eta2'], sh['eta2'])
    assert answer_counts(ans_h, 3) == ([5, 5, 2], 1) and answer_counts(ans_h, 4) == ([5, 5, 2, 0], 1)

    # against a plain-Python computation on random data
    def brute_eta2(col, ans, used):
        xs = [(x, a) for x, a in zip(col, ans) if a in used]
        mu = statistics.fmean(x for x, _ in xs)
        ssb = math.fsum(sum(1 for _, b in xs if b == a) * (statistics.fmean([x for x, b in xs if b == a]) - mu) ** 2
                        for a in used)
        sst = math.fsum((x - mu) ** 2 for x, _ in xs)
        return ssb / sst if sst else 0.0

    def brute_d(col, ans, used, a):
        on = [x for x, b in zip(col, ans) if b == a]
        off = [x for x, b in zip(col, ans) if b in used and b != a]
        sd = statistics.stdev(on + off)
        return (statistics.fmean(on) - statistics.fmean(off)) / sd if sd else 0.0
    rng = np.random.default_rng(1)
    Rr = rng.normal(size=(40, 6))
    ar = [0] * 10 + [1] * 12 + [2] * 6 + [3] * 3 + [None] * 9
    ar = [ar[t] for t in rng.permutation(40)]
    sr = question_stats(Rr, ar, 4)
    assert sr['compared'] == [0, 1, 2] and sorted(sr['d']) == [0, 1, 2]
    for j in range(6):
        col = Rr[:, j].tolist()
        assert abs(sr['eta2'][j] - brute_eta2(col, ar, [0, 1, 2])) < 1e-9, j
        for a in (0, 1, 2):
            assert abs(sr['d'][a][j] - brute_d(col, ar, [0, 1, 2], a)) < 1e-9, (j, a)

    # percentile: share strictly below
    assert percentile([3, 1, 2, 3, 5], 0) == 40.0 and percentile([3, 1, 2, 3, 5], 3) == 40.0
    assert percentile([3, 1, 2, 3, 5], 1) == 0.0 and percentile(np.array([3, 1, 2, 3, 5]), 4) == 80.0
    assert percentile(sh['eta2'], 0) == 50.0 and percentile(sh['eta2'], 1) == 0.0

    # halves and the per-question record
    ans_v = [0] * 5 + [1] * 5 + [2] * 5 + [None] * 5
    half_v = np.array([0, 0, 0, 1, 1] + [0, 1, 1, 1, 1] + [0, 0, 1, 1, 1] + [0, 1, 0, 1, 0])
    half_w = np.array([0] * 5 + [1] * 5 + [1, 1, 1, 1, 0] + [0] * 5)
    Rv = np.random.default_rng(2).normal(size=(20, 5))

    def brute_half(hv, h, used, c):
        sub = [t for t in range(20) if hv[t] == h]
        return percentile([brute_eta2([Rv[t, j] for t in sub], [ans_v[t] for t in sub], used) for j in range(5)], c)
    assert half_eta2_pct(Rv, ans_v, [0, 1, 2], half_v == 0, 1) == brute_half(half_v, 0, [0, 2], 1)  # 1 photo of 1
    assert half_eta2_pct(Rv, ans_v, [0, 1, 2], half_v == 1, 1) == brute_half(half_v, 1, [0, 1, 2], 1)
    assert half_eta2_pct(Rv, ans_v, [0, 1, 2], half_w == 0, 1) is None  # only answer 0 has 2 photos there
    assert half_eta2_pct(Rv, ans_v, [0, 1, 2], half_w == 1, 3) == brute_half(half_w, 1, [1, 2], 3)
    qr = question_record(Rv, 1, ans_v, 3, half_v)
    sv = question_stats(Rv, ans_v, 3)
    assert qr['counts'] == [5, 5, 5] and qr['unclear'] == 5 and qr['compared'] == [0, 1, 2]
    assert qr['eta2'] == float(sv['eta2'][1]) and qr['eta2_pct'] == percentile(sv['eta2'], 1)
    assert qr['d'] == {a: float(sv['d'][a][1]) for a in (0, 1, 2)}
    assert qr['d_pct'] == {a: percentile(sv['d'][a], 1) for a in (0, 1, 2)}
    assert qr['halves'] == [half_eta2_pct(Rv, ans_v, [0, 1, 2], half_v == h, 1) for h in (0, 1)]
    assert question_record(Rv, 1, [0] * 5 + [1] * 3 + [None] * 12, 3, half_v) == {
        'counts': [5, 3, 0], 'unclear': 12, 'compared': [], 'eta2': None, 'eta2_pct': None, 'd': {}, 'd_pct': {},
        'halves': [None, None]}
    assert decide(95.0) == 'WITHIN_5_95' and decide(95.1) == 'ABOVE_95' and decide(5.0) == 'WITHIN_5_95'
    assert decide(4.9) == 'BELOW_5' and decide(None) == 'NO_STATISTICS'

    # null: within-class permutation of whole rows, by the rule, deterministic
    rows = [[p, p % 3] for p in range(9)]
    slab = np.array([4, 7, 4, 4, 7, 9, 7, 4, 7])
    order = [7, 4, 9]
    pr = permute_within(rows, slab, order, np.random.default_rng(3))
    g = np.random.default_rng(3)
    want_p = list(rows)
    for cl in order:
        idx = np.flatnonzero(slab == cl)
        p = g.permutation(len(idx))
        for t, s in zip(idx.tolist(), idx[p].tolist()):
            want_p[t] = rows[s]
    assert pr == want_p and pr == permute_within(rows, slab, order, np.random.default_rng(3))
    for cl in order:
        idx = np.flatnonzero(slab == cl).tolist()
        assert sorted(pr[t] for t in idx) == sorted(rows[t] for t in idx), cl
    assert pr[5] == rows[5] and rows == [[p, p % 3] for p in range(9)]
    assert any(permute_within(rows, slab, order, np.random.default_rng(s)) != rows for s in range(5))

    # table text on a hand case
    tq = [{'question': 'Is there a cat?', 'answers': ['yes', 'no', 'not applicable']},
          {'question': 'Where is it?', 'answers': ['indoors', 'outdoors']},
          {'question': 'How big is it?', 'answers': ['small', 'large']},
          {'question': 'Is it lying down?', 'answers': ['yes', 'no']}]
    tqs = [{'counts': [12, 6, 2], 'unclear': 0, 'compared': [0, 1], 'd': {0: 0.456, 1: -0.9149},
            'd_pct': {0: 96.5, 1: 2.5}, 'eta2_pct': 97.5},
           None,
           {'counts': [3, 14], 'unclear': 3, 'compared': [], 'd': {}, 'd_pct': {}, 'eta2_pct': None},
           {'counts': [9, 8], 'unclear': 3, 'compared': [0, 1], 'd': {0: 0.004, 1: -0.004},
            'd_pct': {0: 50.0, 1: 49.6}, 'eta2_pct': 0.4}]
    head = (TABLE_HEAD.replace('{n}', '20').replace('{K}', '2').replace('{classes}', 'tabby, tiger cat')
            .replace('{m}', '480'))
    want_t = '\n\n'.join([
        head,
        'Q1. Is there a cat?\nseparation: percentile 98\n- yes: 12 photos, d +0.46, percentile 96\n'
        '- no: 6 photos, d -0.91, percentile 2\n- not applicable: 2 photos (too few to compare)',
        'Q2. Where is it?\nseparation: too few photos per answer to compare\n- indoors: (too few to compare)\n'
        '- outdoors: (too few to compare)',
        'Q3. How big is it?\nseparation: too few photos per answer to compare\n'
        '- small: 3 photos (too few to compare)\n- large: 14 photos (too few to compare)\n- could not tell: 3 photos',
        'Q4. Is it lying down?\nseparation: percentile 0\n- yes: 9 photos, d +0.00, percentile 50\n'
        '- no: 8 photos, d -0.00, percentile 50\n- could not tell: 3 photos'])
    got_t = table_text(['tabby', 'tiger cat'], 20, 480, tq, tqs)
    assert got_t == want_t, got_t
    tqs_json = json.loads(json.dumps(tqs))  # keys as strings, as read back from stats.json
    assert table_text(['tabby', 'tiger cat'], 20, 480, tq, tqs_json) == want_t
    assert head.startswith('# Checks on ordinary photos\n\nSomeone who never saw') and '{' not in head

    # cached calls: an is_error reply not cached, one retry on an invalid reply, changed content stops
    lock, calls = threading.Lock(), []

    def fake(err):
        state = {'err': set(err), 'seen': {}}

        def call(sp, turns):
            assert sp == '' and len(turns) == 1
            t = turns[0][0]['text']
            with lock:
                calls.append(t)
                state['seen'][t] = state['seen'].get(t, 0) + 1
                first = state['seen'][t] == 1
                if t in state['err']:
                    state['err'].discard(t)
                    return [{'text': 'overloaded', 'is_error': True}]
            if t == 'bad' or (t == 'retry' and first):
                return [{'text': 'no json here', 'is_error': False}]
            return [{'text': json.dumps({'ok': t}), 'is_error': False}]
        return call

    def parse_ok(text):
        o = bundlemax.last_obj(text, ('ok',))
        return None if o is None else o['ok']

    def jobs_for(texts):
        return [(t, [{'type': 'text', 'text': t}], parse_ok, lambda p: {'got': p}) for t in texts]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'c.json'
        cc = cached_calls('u', path, jobs_for(['good', 'retry', 'bad', 'err']), fake({'err'}))
        assert sorted(cc) == ['bad', 'good', 'retry'], sorted(cc)
        assert sorted(calls) == ['bad', 'bad', 'err', 'good', 'retry', 'retry'], calls
        assert cc['good']['got'] == 'good' and cc['good']['valid'] and len(cc['good']['texts']) == 1
        assert cc['retry']['got'] == 'retry' and cc['retry']['texts'][0] == 'no json here'
        assert len(cc['retry']['texts']) == 2 and cc['bad']['got'] is None and not cc['bad']['valid']
        assert json.loads(path.read_text()) == cc
        calls.clear()
        cc = cached_calls('u', path, jobs_for(['good', 'retry', 'bad', 'err']), fake(()))
        assert calls == ['err'] and sorted(cc) == ['bad', 'err', 'good', 'retry']
        calls.clear()
        try:
            cached_calls('u', path, [('good', [{'type': 'text', 'text': 'other'}], parse_ok, lambda p: {})], fake(()))
            raise AssertionError('changed content accepted')
        except SystemExit:
            pass
        assert calls == []

    # one unit end to end, with a stand-in for the agents and synthetic values (no model, no Claude, no photos)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rng = np.random.default_rng(4)
        Vs = rng.random((600, 20)).astype(np.float32)
        labs = rng.permutation(np.repeat(np.arange(50), 12))
        nms = [f'img/{i}.jpeg' for i in range(600)]
        cats_t = [f'class {i}' for i in range(50)]
        uu, cu = parse_unit('mixed4c:3'), 3
        save_json({k: {'explanation': e} for k, e in zip(SOURCE_KEYS, ('expl one', 'expl two', 'expl three'))},
                  td / 'src' / 'mixed4c:3' / 'readings.json')
        cls_t, sid_t = photo_set(Vs[:, cu], labs)
        assert len(cls_t) == 12 and len(sid_t) == 144 and cls_t == top_classes(Vs[:, cu], labs)
        R_t = centre(Vs[sid_t], labs[sid_t])
        r_of = dict(zip(sid_t.tolist(), R_t[:, cu].tolist()))
        bad_first = answer_batches('mixed4c:3', sid_t)[0][0]
        qreply = {'questions': [{'question': 'Is it bright?', 'answers': ['yes', 'no']}, {'question': 'bad'},
                                {'question': 'Which colour?', 'answers': ['red', 'green', 'blue']},
                                {'question': 'Is there text?', 'answers': ['yes', 'no']}]}
        state = {'n': 0, 'err': True}

        def call_u(sp, turns):
            assert sp == '' and len(turns) == 1
            content = turns[0]
            texts = [b['text'] for b in content if b['type'] == 'text']
            with lock:
                state['n'] += 1
            if texts[-1] == run_subjects.TASK:
                assert len(content) == 2 and texts[0].startswith('Here is an explanation of what one unit')
                assert '\n\nLater, these checks were run:\n\n# Checks on ordinary photos\n\n' in texts[0]
                if 'expl two' in texts[0] and state['err']:
                    return [{'text': 'overloaded', 'is_error': True}]
                return [{'text': json.dumps({'class': 'one', 'explanation': 'revised'}), 'is_error': False}]
            if len(content) == 1:
                assert texts[0].startswith('Below are explanations') and '\n\nExplanation 2: expl two\n\n' in texts[0]
                return [{'text': 'Here they are: ' + json.dumps(qreply), 'is_error': False}]
            ids = [b['id'] for b in content if b['type'] == 'image']
            assert texts[-1].startswith(f'Answer every question below for each of the {len(ids)} images above.')
            if ids[0] == bad_first:
                return [{'text': 'sorry', 'is_error': False}]
            ent = [{'id': k, 'answers': ['A' if r_of[i] > 0 else 'b', 'ABC'[i % 3], 'A' if i % 10 == 0 else '?']}
                   for k, i in enumerate(ids, 1)]
            return [{'text': json.dumps({'images': ent}), 'is_error': False}]

        def blk(i):
            return {'type': 'image', 'id': int(i)}
        kw = dict(call=call_u, out=td / 'out', src=td / 'src', block=blk)
        sec, rev = run_unit(uu, Vs, labs, nms, cats_t, **kw)
        assert state['n'] == 1 + 13 + 6, state  # questions, 12 batches + 1 retry, 6 revisions (2 is_error)
        assert sorted(rev) == ['null_s1', 'null_s3', 'real_s1', 'real_s3'] and rev['real_s1']['explanation'] == 'revised'
        lines = sec.splitlines()
        assert lines[0] == ('== mixed4c:3  layer mixed4c (20 channels), 144 photos, answer batches failed after a '
                            'retry: 1 of 12'), lines[0]
        assert 'real_s2  no revision (call failed, not cached)' in lines and 'real_s1  one  revised' in lines
        ud = td / 'out' / 'mixed4c:3'
        st_t = json.loads((ud / 'stats.json').read_text())
        assert st_t['classes'] == cls_t and st_t['set_ids'] == sid_t.tolist() and st_t['failed_batches'] == ['1']
        assert [q['question'] for q in st_t['questions']] == ['Is it bright?', 'Which colour?', 'Is there text?']
        q1, q3 = st_t['questions'][0], st_t['questions'][2]
        assert q1['real']['unclear'] == 12 and sum(q1['real']['counts']) == 132 and q1['real']['compared'] == [0, 1]
        assert q1['null']['counts'] == q1['real']['counts'] and q1['null']['unclear'] == 12
        assert q1['real']['eta2_pct'] == 95.0 and q1['real']['d_pct']['0'] == 95.0 and q1['real']['d']['0'] > 0
        assert q3['real']['eta2_pct'] is None and q3['real']['counts'][1] == 0 and q3['null']['eta2_pct'] is None
        assert 'Q3. Is there text?\nseparation: too few photos per answer to compare\n' in st_t['tables']['real']
        an = json.loads((ud / 'answers.json').read_text())
        assert sorted(an, key=int) == [str(b) for b in range(1, 13)] and an['1']['failed'] and len(an['1']['texts']) == 2
        assert all(x is None for row in an['1']['answers'] for x in row) and not an['2']['failed']
        state.update(n=0, err=False)
        sec2, rev2 = run_unit(uu, Vs, labs, nms, cats_t, **kw)
        assert state['n'] == 2 and sorted(rev2) == sorted(REVISIONS)
        state['n'] = 0
        assert run_unit(uu, Vs, labs, nms, cats_t, **kw)[0] == sec2 and state['n'] == 0
        sf, rf = run_unit(parse_unit('mixed4c:4'), Vs, labs, nms, cats_t, **kw)
        assert rf is None and sf.startswith('== mixed4c:4\nfailed: ') and state['n'] == 0

    # values: hooks on several layers, one pass per batch, stop after the deepest; on a small stand-in module
    import torch
    torch.manual_seed(0)

    class Boom(torch.nn.Module):
        def forward(self, x):
            raise AssertionError('ran past the stop')

    fm = torch.nn.Sequential(OrderedDict([('mixed3a', torch.nn.Conv2d(3, NCH['mixed3a'], 4, stride=4)),
                                          ('mixed3b', torch.nn.Conv2d(NCH['mixed3a'], NCH['mixed3b'], 1)),
                                          ('boom', Boom())])).eval()
    for q in fm.parameters():
        q.requires_grad_(False)
    xs = torch.randn(70, 3, 16, 16)

    def make_input(ids):
        return xs[torch.as_tensor(np.asarray(ids, dtype=np.int64))]
    got = forward_values(fm, ['mixed3b', 'mixed3a'], np.arange(70), 'cpu', make_input=make_input)
    with torch.no_grad():
        a3 = fm.mixed3a(xs)
        b3 = fm.mixed3b(a3)
    assert sorted(got) == ['mixed3a', 'mixed3b'] and got['mixed3a'].dtype == np.float32
    assert got['mixed3a'].shape == (70, 256) and got['mixed3b'].shape == (70, 480)
    assert np.allclose(got['mixed3a'], a3.amax(dim=(2, 3)).numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(got['mixed3b'], b3.amax(dim=(2, 3)).numpy(), rtol=1e-5, atol=1e-5)

    def no_run(_m, _i):
        raise AssertionError('ran past the deepest hooked layer')
    h = fm.mixed3b.register_forward_pre_hook(no_run)
    try:
        only = forward_values(fm, ['mixed3a'], np.arange(5), 'cpu', make_input=make_input)
    finally:
        h.remove()
    assert list(only) == ['mixed3a'] and np.allclose(only['mixed3a'], got['mixed3a'][:5], rtol=1e-5, atol=1e-5)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        loads = []

        def gm(dev):
            loads.append(dev)
            return fm

        def no_model(dev):
            raise AssertionError('model loaded with every cache present')
        v1 = layer_values(['mixed3b', 'mixed3a'], gm, np.arange(70), device='cpu', out=td, make_input=make_input)
        assert loads == ['cpu'] and sorted(p.name for p in td.iterdir()) == ['values_mixed3a.npy', 'values_mixed3b.npy']
        assert all(np.allclose(v1[L], got[L], rtol=1e-5, atol=1e-5) for L in got)
        v2 = layer_values(['mixed3a', 'mixed3b'], no_model, np.arange(70), device='cpu', out=td)
        assert all(np.array_equal(v1[L], v2[L]) for L in v1)
        (td / 'values_mixed3b.npy').unlink()
        loads.clear()
        v3 = layer_values(['mixed3a', 'mixed3b'], gm, np.arange(70), device='cpu', out=td, make_input=make_input)
        assert loads == ['cpu'] and np.allclose(v3['mixed3b'], v1['mixed3b'], rtol=1e-5, atol=1e-5)
        np.save(td / 'values_mixed3a.npy', np.zeros((5, 256), np.float32))
        try:
            layer_values(['mixed3a'], no_model, np.arange(70), device='cpu', out=td)
            raise AssertionError('cache of the wrong shape accepted')
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
    units = [parse_unit(s) for s in json.loads((uf if uf.is_absolute() else J / uf).read_text())]
    names = json.loads(NAMES.read_text())
    with np.load(ACTS) as z:
        labels = np.asarray(z['classes'], dtype=np.int64)
    assert len(names) == len(labels) == N_PHOTOS, (len(names), len(labels))
    assert labels.min() >= 0 and labels.max() < 1000, (labels.min(), labels.max())
    from torchvision.models import GoogLeNet_Weights
    cats = list(GoogLeNet_Weights.IMAGENET1K_V1.meta['categories'])
    assert len(cats) == 1000, len(cats)
    held = {}

    def get_model(dev):
        if dev not in held:
            import torch
            torch.set_num_threads(4)
            m = bundlemax._bank().load_model(WEIGHTS)
            for q in m.parameters():
                q.requires_grad_(False)
            held[dev] = m.to(dev)
        return held[dev]

    OUT.mkdir(parents=True, exist_ok=True)
    layers = sorted({u['layer'] for u in units}, key=LAYERS.index)
    vals = layer_values(layers, get_model, np.arange(N_PHOTOS))
    held.clear()
    gc.collect()

    secs, allr = [], {}
    for u in units:
        n = u['name']
        print('==', n, flush=True)
        sec, rev = run_unit(u, vals[u['layer']], labels, names, cats)
        rev = rev or {}
        allr[n] = {rid: (rev[rid]['explanation'] if rid in rev else None) for rid in REVISIONS}
        secs.append(sec)
        save_json(allr, OUT / 'readings_all.json')
        (OUT / 'report.txt').write_text('\n\n'.join(secs) + '\n')
        print(sec, flush=True)
        gc.collect()
    print('report ->', OUT / 'report.txt')


if __name__ == '__main__':
    main()
