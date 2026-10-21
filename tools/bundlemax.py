"""Input bundles of a unit, a blind maximum per bundle, and a judge on whether the bundle explanations are related.

Per unit (a channel of mixedNx, or a planted sum of two channels): attribution of the unit to the input layer (the
previous module) on its top-150 card-pool photos; the 24 input channels with the largest mean attribution are split
into bundles by k-means on their profiles. Condition b zeroes the 24 channels not in b; "whole" ablates nothing.
Each condition's top 20 of C go to Claude as a card for an explanation; a second call that sees only the explanation
picks 6 of 48 held-pool photos; best = max picked value / M_b, hit = share of picks >= 0.3 M_b. A third call rates
whether the bundles' explanations describe one thing. Spec: authors/spec_bundlemax.md. Outputs: runs/bundlemax/.

usage: python -I tools/bundlemax.py --units mixed4c:447,mixed4d:409 [--selftest]
"""
import argparse
import functools
import gc
import hashlib
import json
import random
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.cluster.vq import kmeans2

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
import claude_call  # noqa: E402
import run_subjects  # noqa: E402

OUT = J / 'runs/bundlemax'
BATCH = 64
WORKERS = 4
PREV = {'mixed3b': 'mixed3a', 'mixed4c': 'mixed4b', 'mixed4d': 'mixed4c', 'mixed4e': 'mixed4d'}
NCH = {'mixed3a': 256, 'mixed3b': 480, 'mixed4b': 512, 'mixed4c': 512, 'mixed4d': 528, 'mixed4e': 832}
N_ATTR, K, N_CLUST = 150, 24, 3
N_TOP, N_OTHER, N_HELD = 1500, 500, 1000
N_CARD, N_STRAT = 20, 16
HIT, LOW = 0.3, 0.05
SAMPLES, CALLS, JUDGE_CALLS = (1, 2), (1, 2), (1, 2, 3)

CARD = ('# Unit {name}\n\nRecorded from one unit (a single neuron) in an image classification network, run over several '
        'thousand natural photos.\nThe attached grid shows the 20 photos on which the unit is most active, strongest '
        'first. Under each is its activation, scaled 0-10 relative to the strongest.\n')
PICK = ('A unit (one neuron) of an image classification network has been explained as follows:\n\n{explanation}\n\n'
        'The attached sheet shows 48 numbered photos. You have not seen how the unit responds to any of them. Using '
        'only the explanation, choose the 6 photos on which the unit should fire most strongly. Reply with only JSON: '
        '{"fire": [six numbers]}')
JUDGE = ('These are explanations of different input pathways into one unit of a neural network. Each was written from '
         'the inputs that drive the unit through that pathway alone.\n\n{list}\n\nDo they describe one underlying thing '
         '(different parts, aspects or natural offshoots of the same thing), or unrelated things that happen to share '
         'the unit? Reply with only JSON: {"related": n, "shared": "..."}, where n is 1 (clearly unrelated) to 5 '
         '(clearly one thing) and shared names what they have in common, or is empty.')


# ---------- units, photos, network ----------

def parse_unit(s):
    """'mixedNx:c' or 'sum:mixedNx:a+b' -> dict(name, kind, layer, prev, chans)"""
    p = s.split(':')
    kind = 'single'
    if p[0] == 'sum':
        kind, p = 'sum', p[1:]
    if len(p) != 2 or p[0] not in PREV:
        raise ValueError(f'bad unit {s!r}')
    chans = [int(c) for c in p[1].split('+')]
    if (len(chans) != (2 if kind == 'sum' else 1) or len(set(chans)) != len(chans)
            or not all(0 <= c < NCH[p[0]] for c in chans)):
        raise ValueError(f'bad unit {s!r}')
    return {'name': s, 'kind': kind, 'layer': p[0], 'prev': PREV[p[0]], 'chans': chans}


def pools(names):
    """card pool (crc32(name) % 10 < 7) and held pool (the rest), as photo ids"""
    r = np.array([zlib.crc32(n.encode()) % 10 for n in names])
    return np.flatnonzero(r < 7).astype(np.int64), np.flatnonzero(r >= 7).astype(np.int64)


def _bank():
    p = str(J / 'bank')
    if p not in sys.path:
        sys.path.insert(0, p)
    import img_acts
    return img_acts


def photo(i):
    with Image.open(J / 'bank/img/thumbs' / f'{int(i):05d}.jpg') as im:
        img = im.convert('RGB')
    assert img.size == (224, 224), (i, img.size)
    return img


def net_input(ids):
    """as tools/battery.py: pixels (0-255) - 117, NCHW"""
    import torch
    x = torch.from_numpy(np.stack([np.asarray(photo(i), dtype=np.float32) / 255.0 for i in ids])).permute(0, 3, 1, 2)
    return x * 255 - 117


def umap(out, u, s):
    """the unit's map from its layer's output: sum over its channels of map_c / s_c (s = [1] for a single channel)"""
    m = out[:, u['chans'][0]] / float(s[0])
    for c, sc in zip(u['chans'][1:], s[1:]):
        m = m + out[:, c] / float(sc)
    return m


def zero_set(top, bundle):
    """input channels zeroed for a bundle: the top-K channels not in it"""
    b = set(int(c) for c in bundle)
    return np.array([int(c) for c in top if int(c) not in b], dtype=np.int64)


def apply_zero(out, z):
    """copy of out with channels z set to 0 at all positions (torch tensor or numpy array)"""
    out = out.clone() if hasattr(out, 'clone') else out.copy()
    out[:, z] = 0
    return out


def conditions(top, bundles):
    """condition names and their zero sets: 'whole' (None = no ablation), then one per bundle"""
    return ['whole'] + [b['name'] for b in bundles], [None] + [zero_set(top, b['channels']) for b in bundles]


def forward(model, u, ids, reduce, zeros=(None,), tag=''):
    """reduce(unit-layer output) for photo ids, once per zero set of input-layer channels; list of arrays"""
    import torch
    Stop = _bank().Stop
    got, cur = {}, {}

    def ablate(_m, _i, out):
        assert out.shape[1] == NCH[u['prev']], tuple(out.shape)
        return None if cur['z'] is None else apply_zero(out, cur['z'])

    def stop(_m, _i, out):
        assert out.shape[1] == NCH[u['layer']], tuple(out.shape)
        got['r'] = reduce(out)
        raise Stop

    zs = [None if z is None else torch.as_tensor(z, dtype=torch.long) for z in zeros]
    res = [[] for _ in zs]
    hs = [getattr(model, u['prev']).register_forward_hook(ablate),
          getattr(model, u['layer']).register_forward_hook(stop)]
    try:
        for b, s in enumerate(range(0, len(ids), BATCH)):
            x = net_input(ids[s:s + BATCH])
            for j, z in enumerate(zs):
                cur['z'] = z
                got.clear()
                try:
                    with torch.no_grad():
                        model(x)
                except Stop:
                    pass
                res[j].append(got['r'])
            if b % 20 == 19:
                print(f'  {tag} {min(s + BATCH, len(ids))}/{len(ids)}', flush=True)
    finally:
        for h in hs:
            h.remove()
    return [np.concatenate(r) for r in res]


def attribution(model, u, ids, s):
    """A[n, i] = sum over positions of dv/dx * x for input channel i, v = the unit's map at its argmax, x = the
    input-layer output as a leaf; each row unit-L2 normalised"""
    import torch
    Stop = _bank().Stop
    got = {}

    def leaf(_m, _i, out):
        assert out.shape[1] == NCH[u['prev']], tuple(out.shape)
        got['x'] = out.detach().requires_grad_(True)
        return got['x']

    def stop(_m, _i, out):
        got['m'] = umap(out, u, s)
        raise Stop

    res = []
    hs = [getattr(model, u['prev']).register_forward_hook(leaf),
          getattr(model, u['layer']).register_forward_hook(stop)]
    try:
        for st in range(0, len(ids), BATCH):
            x = net_input(ids[st:st + BATCH])
            got.clear()
            with torch.enable_grad():
                try:
                    model(x)
                except Stop:
                    pass
                m = got['m'].flatten(1)
                v = m.gather(1, m.argmax(dim=1, keepdim=True)).sum()
                g, = torch.autograd.grad(v, got['x'])
            res.append((g * got['x']).sum(dim=(2, 3)).detach().numpy().astype(np.float64))
    finally:
        for h in hs:
            h.remove()
    A = np.concatenate(res)
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.where(n > 0, n, 1)


def cluster(A_top):
    """k-means labels of the channels: profile = a channel's A over the photos, centred and unit-L2 normalised"""
    p = A_top.T - A_top.T.mean(axis=1, keepdims=True)
    n = np.linalg.norm(p, axis=1, keepdims=True)
    p = p / np.where(n > 0, n, 1)
    _, labels = kmeans2(p, N_CLUST, seed=0, minit='++')
    return labels


def make_bundles(top, mA, labels):
    """non-empty clusters in label order -> [{name, channels, share of the summed mean A of the top K}]"""
    out = []
    for lab in sorted(set(int(x) for x in labels)):
        m = np.asarray(labels) == lab
        out.append({'name': f'b{len(out) + 1}', 'channels': [int(c) for c in np.asarray(top)[m]],
                    'share': float(mA[m].sum() / mA.sum())})
    return out


def measure(model, u, card, held):
    """steps 1-3: attribution, bundles, the unit's spatial max under each condition on C and H"""
    if u['kind'] == 'sum':
        cm = forward(model, u, card, lambda o: o[:, u['chans']].amax(dim=(2, 3)).numpy(), tag='scales')[0]
        s = cm.max(axis=0).astype(np.float64)
        assert (s > 0).all(), ('channel max <= 0 over the card pool', s)
        del cm
    else:
        s = np.ones(1)

    def um(o):
        return umap(o, u, s).amax(dim=(1, 2)).numpy()

    cv = forward(model, u, card, um, tag='card pool')[0]
    order = np.argsort(-cv, kind='stable')

    attr_ids = card[order[:N_ATTR]]
    A = attribution(model, u, attr_ids, s)
    mA = A.mean(axis=0)
    top = np.argsort(-mA, kind='stable')[:K]
    labels = cluster(A[:, top])
    bundles = make_bundles(top, mA[top], labels)
    conds, zeros = conditions(top, bundles)

    rest = np.sort(card[order[N_TOP:]])
    C = np.concatenate([card[order[:N_TOP]], np.sort(np.random.default_rng(0).choice(rest, N_OTHER, replace=False))])
    H = np.sort(np.random.default_rng(1).choice(held, N_HELD, replace=False))
    v = forward(model, u, np.concatenate([C, H]), um, zeros, tag='conditions')
    vC = np.stack([x[:len(C)] for x in v]).astype(np.float32)
    vH = np.stack([x[len(C):] for x in v]).astype(np.float32)

    info = {'unit': u['name'], 'kind': u['kind'], 'layer': u['layer'], 'input_layer': u['prev'], 'chans': u['chans'],
            'scales': s.tolist(), 'attr_ids': attr_ids.tolist(), 'top': top.tolist(), 'meanA_top': mA[top].tolist(),
            'labels': [int(x) for x in labels], 'bundles': bundles, 'conditions': conds,
            'zeroed': {c: (None if z is None else z.tolist()) for c, z in zip(conds, zeros)}}
    arr = dict(card=card, card_vals=cv, attr_ids=attr_ids, A=A.astype(np.float32), C=C, H=H, vC=vC, vH=vH,
               conditions=np.array(conds))
    return info, arr


# ---------- cards, sheets, pools, scoring ----------

def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def act_k(v, M):
    return int(round(10 * float(v) / float(M)))


def card_name(unit, cond):
    r = random.Random(unit + cond)
    return 'U-' + ''.join(str(r.randrange(10)) for _ in range(3))


def make_card(path, ids, vals, M):
    """5-column grid of whole photos (224 px), a white strip under each: '#n   activation k'"""
    cols, tile, lab = 5, 224, 24
    rows = -(-len(ids) // cols)
    im = Image.new('RGB', (cols * tile, rows * (tile + lab)), 'white')
    dr, f = ImageDraw.Draw(im), _font(16)
    for n, (i, v) in enumerate(zip(ids, vals)):
        x0, y0 = (n % cols) * tile, (n // cols) * (tile + lab)
        im.paste(photo(i), (x0, y0))
        dr.text((x0 + 4, y0 + tile + 3), f'#{n + 1}   activation {act_k(v, M)}', fill='black', font=f)
    im.save(path, quality=90)


def make_sheet(path, ids):
    """8 columns x 6 rows of 112 px tiles, numbered from 1 in a white strip under each"""
    cols, tile, lab = 8, 112, 18
    rows = -(-len(ids) // cols)
    im = Image.new('RGB', (cols * tile, rows * (tile + lab)), 'white')
    dr, f = ImageDraw.Draw(im), _font(13)
    for n, i in enumerate(ids):
        x0, y0 = (n % cols) * tile, (n // cols) * (tile + lab)
        im.paste(photo(i).resize((tile, tile), Image.LANCZOS), (x0, y0))
        dr.text((x0 + 3, y0 + tile + 2), str(n + 1), fill='black', font=f)
    im.save(path, quality=90)


def strata(r, rng, n=N_STRAT):
    """positions into r: n with r >= 0.3, n with 0.05 <= r < 0.3, n with r < 0.05; a short stratum's deficit is filled
    from the next lower one"""
    groups = [np.flatnonzero(r >= HIT), np.flatnonzero((r >= LOW) & (r < HIT)), np.flatnonzero(r < LOW)]
    out, need = [], 0
    for g in groups:
        need += n
        k = min(need, len(g))
        if k:
            out.append(rng.choice(g, k, replace=False))
        need -= k
    if need:
        raise ValueError(f'pool short by {need} photos')
    return np.concatenate(out)


def score(picks, ratio):
    """best = max of the picks' value / M_b; hit = share of picks with value / M_b >= 0.3 (picks numbered from 1)"""
    v = np.asarray(ratio, dtype=np.float64)[np.asarray(picks) - 1]
    return float(v.max()), float(np.mean(v >= HIT))


def summarise(recs, conds, bnames):
    """per pool condition p: own = explanations of p scored on p's pool; cross = explanations of the other bundles
    of the unit (never 'whole') scored on p's pool; means over valid calls"""
    def m(rs, key):
        return float(np.mean([r[key] for r in rs])) if rs else float('nan')
    out = {}
    for p in conds:
        own = [r for r in recs if r['valid'] and r['pool'] == p and r['expl'] == p]
        cross = [r for r in recs if r['valid'] and r['pool'] == p and r['expl'] != p and r['expl'] in bnames]
        out[p] = {'own_best': m(own, 'best'), 'own_hit': m(own, 'hit'), 'own_n': len(own),
                  'cross_best': m(cross, 'best'), 'cross_hit': m(cross, 'hit'), 'cross_n': len(cross)}
    return out


def decide_gap(g):
    """Neutral label on own hit minus cross hit: > 0.25 OWN_HIGHER, < -0.25 CROSS_HIGHER, else SIMILAR."""
    if np.isnan(g):
        return 'UNDEFINED'
    return 'OWN_HIGHER' if g > 0.25 else 'CROSS_HIGHER' if g < -0.25 else 'SIMILAR'


def decide_judge(m):
    """Neutral label on the mean judge score (1-5): >= 3.5 HIGH, <= 2.5 LOW, else MIDDLE."""
    if np.isnan(m):
        return 'UNDEFINED'
    return 'HIGH' if m >= 3.5 else 'LOW' if m <= 2.5 else 'MIDDLE'


# ---------- replies ----------

def last_obj(text, keys):
    """the last JSON object in text that has all keys (bare, fenced, or among prose), else None"""
    dec, found = json.JSONDecoder(), None
    for i, ch in enumerate(text):
        if ch == '{':
            try:
                obj, _ = dec.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(obj, dict) and all(k in obj for k in keys):
                found = obj
    return found


def parse_expl(text):
    o = last_obj(text, ('class', 'explanation'))
    if o is None or not isinstance(o['explanation'], str) or not o['explanation'].strip():
        return None
    return o


def parse_fire(text):
    """the last object with 'fire'; valid = 6 distinct integers 1-48, else None"""
    o = last_obj(text, ('fire',))
    if o is None or not isinstance(o['fire'], list) or len(o['fire']) != 6:
        return None
    f = o['fire']
    if not all(isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 48 for n in f) or len(set(f)) != 6:
        return None
    return f


def parse_judge(text):
    """(related, shared): related a number 1-5 or None; shared a string"""
    o = last_obj(text, ('related',))
    if o is None:
        return None, ''
    n = o['related']
    ok = isinstance(n, (int, float)) and not isinstance(n, bool) and 1 <= n <= 5
    sh = o.get('shared', '')
    return (n if ok else None), (sh if isinstance(sh, str) else json.dumps(sh))


def ask_cached(udir, label, content):
    """one fresh conversation, empty system prompt, one user turn; reply cached by label and content hash"""
    sha = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    f = udir / 'replies' / f'{label}_{sha[:12]}.json'
    if f.exists():
        return json.loads(f.read_text())['text']
    r = claude_call.converse('', [content])[0]
    if r['is_error']:
        raise RuntimeError(f'error reply, not saved: {r["text"][:80]}')
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix('.tmp')
    tmp.write_text(json.dumps({'label': label, 'model': claude_call.MODEL, 'content_sha256': sha, **r},
                              ensure_ascii=False))
    tmp.replace(f)
    return r['text']


def pick(ask, label, content):
    """one blind-maximum call; one fresh retry if the reply is not valid"""
    texts = []
    for a in range(2):
        texts.append(ask(f'{label}_a{a}', content))
        p = parse_fire(texts[-1])
        if p is not None:
            return {'picks': p, 'attempts': a + 1, 'texts': texts}
    return {'picks': None, 'attempts': 2, 'texts': texts}


def pmap(fn, jobs):
    """fn over jobs in threads; job -> result, or the exception it raised"""
    out = {}
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fn, j): j for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            try:
                out[j] = f.result()
            except Exception as e:
                print('FAIL', j, repr(e)[:200], flush=True)
                out[j] = e
            if n % 10 == 0 or n == len(jobs):
                print(f'  {n}/{len(jobs)} calls', flush=True)
    return out


# ---------- one unit ----------

def run_unit(u, card, held, get_model):
    udir = OUT / u['name']
    udir.mkdir(parents=True, exist_ok=True)
    bf, vf = udir / 'bundles.json', udir / 'values.npz'
    if bf.exists() and vf.exists():
        info = json.loads(bf.read_text())
        assert info['unit'] == u['name'], (info['unit'], u['name'])
        print(u['name'], 'bundles and values: cached', flush=True)
    else:
        info, arr = measure(get_model(), u, card, held)
        np.savez_compressed(vf, **arr)
        bf.write_text(json.dumps(info, indent=1))
        del arr
        gc.collect()
    with np.load(vf) as z:
        C, H, vC, vH = z['C'], z['H'], z['vC'], z['vH']
    conds = info['conditions']
    bnames = [b['name'] for b in info['bundles']]
    assert vC.shape == (len(conds), len(C)) and vH.shape == (len(conds), len(H)), (vC.shape, vH.shape)
    M = vC.max(axis=1).astype(np.float64)
    assert (M > 0).all(), ('M_b <= 0', dict(zip(conds, M.tolist())))
    ask = functools.partial(ask_cached, udir)

    # 4. cards and explanations
    cont, expl = {}, {}
    for j, c in enumerate(conds):
        top = np.argsort(-vC[j], kind='stable')[:N_CARD]
        path = udir / f'card_{c}.jpg'
        make_card(path, C[top], vC[j, top], M[j])
        name = card_name(u['name'], c)
        cont[c] = [{'type': 'text', 'text': CARD.replace('{name}', name)}, run_subjects.img_block(path),
                   {'type': 'text', 'text': run_subjects.TASK}]
        expl[c] = {'name': name, 'card': path.name, 'M': float(M[j]), 'ids': C[top].tolist(),
                   'values': vC[j, top].tolist(), 'samples': []}
    print(u['name'], 'explanations', flush=True)
    got = pmap(lambda jb: ask(f'expl_{jb[0]}_s{jb[1]}', cont[jb[0]]), [(c, k) for c in conds for k in SAMPLES])
    for c in conds:
        for k in SAMPLES:
            t = got[(c, k)]
            if isinstance(t, Exception):
                expl[c]['samples'].append({'sample': k, 'class': None, 'explanation': None, 'error': repr(t)})
                continue
            o = parse_expl(t)
            expl[c]['samples'].append({'sample': k, 'class': None if o is None else o.get('class'),
                                       'explanation': None if o is None else o['explanation'], 'parsed': o,
                                       'text': t})
    (udir / 'explanations.json').write_text(json.dumps(expl, indent=1, ensure_ascii=False))
    del cont, got

    # 5. blind maximum
    pool, blocks = {}, {}
    for j, c in enumerate(conds):
        r = vH[j] / M[j]
        sel = strata(r, np.random.default_rng(2))
        sel = sel[np.random.default_rng(3).permutation(len(sel))]
        path = udir / f'pool_{c}.jpg'
        make_sheet(path, H[sel])
        blocks[c] = run_subjects.img_block(path)
        pool[c] = {'sheet': path.name, 'M': float(M[j]), 'ids': H[sel].tolist(), 'values': vH[j, sel].tolist(),
                   'ratio': r[sel].tolist(), 'chance_hit': float(np.mean(r[sel] >= HIT))}
    texts = {(c, s['sample']): s['explanation'] for c in conds for s in expl[c]['samples'] if s['explanation']}
    jobs = [(e, k, p, n) for (e, k) in texts for p in conds for n in CALLS]
    print(u['name'], 'blind maximum', flush=True)
    got = pmap(lambda jb: pick(ask, f'pick_{jb[0]}_s{jb[1]}_on_{jb[2]}_c{jb[3]}',
                               [{'type': 'text', 'text': PICK.replace('{explanation}', str(texts[jb[:2]]))},
                                blocks[jb[2]]]), jobs)
    recs = []
    for jb in jobs:
        e, k, p, n = jb
        rec = {'expl': e, 'sample': k, 'pool': p, 'call': n}
        g = got[jb]
        if isinstance(g, Exception):
            rec.update(valid=False, error=repr(g))
        else:
            rec.update(valid=g['picks'] is not None, picks=g['picks'], attempts=g['attempts'], replies=g['texts'])
            if rec['valid']:
                rec['best'], rec['hit'] = score(g['picks'], pool[p]['ratio'])
        recs.append(rec)
    pk = {'pools': pool, 'summary': summarise(recs, conds, bnames), 'calls': recs}
    (udir / 'picks.json').write_text(json.dumps(pk, indent=1, ensure_ascii=False))
    del blocks, got

    # 6. relatedness judge
    ex1 = [next((s['explanation'] for s in expl[b]['samples'] if s['sample'] == 1), None) for b in bnames]
    if len(bnames) < 2:
        judge = {'skipped': f'{len(bnames)} bundle(s)'}
    elif not all(ex1):
        judge = {'skipped': 'a bundle has no sample-1 explanation'}
    else:
        text = JUDGE.replace('{list}', '\n'.join(f'Pathway {i}: {t}' for i, t in enumerate(ex1, 1)))
        print(u['name'], 'judge', flush=True)
        got = pmap(lambda n: ask(f'judge_c{n}', [{'type': 'text', 'text': text}]), list(JUDGE_CALLS))
        calls = []
        for n in JUDGE_CALLS:
            if isinstance(got[n], Exception):
                calls.append({'call': n, 'related': None, 'shared': '', 'error': repr(got[n])})
                continue
            rel, sh = parse_judge(got[n])
            calls.append({'call': n, 'related': rel, 'shared': sh, 'text': got[n]})
        rs = [c['related'] for c in calls if c['related'] is not None]
        judge = {'pathways': dict(zip(bnames, ex1)), 'prompt': text, 'calls': calls,
                 'mean_related': float(np.mean(rs)) if rs else float('nan')}
    (udir / 'judge.json').write_text(json.dumps(judge, indent=1, ensure_ascii=False))
    return section(u, info, expl, pk, judge)


def section(u, info, expl, pk, judge):
    L = [f'== {u["name"]}  layer {u["layer"]}, input {u["prev"]}'
         + (f', scales {[round(x, 4) for x in info["scales"]]}' if u['kind'] == 'sum' else '')]
    for b in info['bundles']:
        L.append(f'bundle {b["name"]}: share {b["share"]:.3f}, {len(b["channels"])} channels {b["channels"]}')
    for c in info['conditions']:
        s, po = pk['summary'][c], pk['pools'][c]
        L.append(f'-- {c}  ({expl[c]["name"]}, M={po["M"]:.4g})')
        for e in expl[c]['samples']:
            t = e['explanation'] or '(no explanation: ' + e.get('error', 'reply not parsed') + ')'
            L.append(f'   s{e["sample"]} [{e["class"]}] ' + ' '.join(str(t)[:200].split()))
        gap = s['own_hit'] - s['cross_hit']
        L.append(f'   own best {s["own_best"]:.3f} hit {s["own_hit"]:.3f} (n={s["own_n"]}) | cross best '
                 f'{s["cross_best"]:.3f} hit {s["cross_hit"]:.3f} (n={s["cross_n"]}) | chance hit '
                 f'{po["chance_hit"]:.3f} | own-cross hit {gap:+.3f} -> {decide_gap(gap)}')
    if 'skipped' in judge:
        L.append(f'judge: skipped ({judge["skipped"]})')
    else:
        m = judge['mean_related']
        L.append(f'judge related {[c["related"] for c in judge["calls"]]} mean {m:.2f} -> {decide_judge(m)}')
        for c in judge['calls']:
            L.append(f'   shared {c["call"]}: {c["shared"]}')
    return '\n'.join(L)


# ---------- selftest ----------

def selftest():
    # unit parsing
    u = parse_unit('mixed4c:447')
    assert u == {'name': 'mixed4c:447', 'kind': 'single', 'layer': 'mixed4c', 'prev': 'mixed4b', 'chans': [447]}, u
    s = parse_unit('sum:mixed4d:3+9')
    assert (s['kind'], s['layer'], s['prev'], s['chans']) == ('sum', 'mixed4d', 'mixed4c', [3, 9]), s
    assert parse_unit('mixed3b:0')['prev'] == 'mixed3a' and parse_unit('mixed4e:831')['prev'] == 'mixed4d'
    for bad in ['mixed4a:1', 'mixed4c:512', 'mixed4c:-1', 'mixed4c', 'mixed4c:x', 'mixed4c:1+2', 'sum:mixed4c:3',
                'sum:mixed4c:3+3', 'sum:mixed4c:1+2+3', 'sum:mixed3a:1+2']:
        try:
            parse_unit(bad)
        except ValueError:
            continue
        raise AssertionError('accepted ' + bad)

    # pools
    names = [f'img/{i}.jpeg' for i in range(1000)]
    card, held = pools(names)
    assert len(card) + len(held) == 1000 and not set(card.tolist()) & set(held.tolist())
    assert all(zlib.crc32(names[i].encode()) % 10 < 7 for i in card)
    assert all(zlib.crc32(names[i].encode()) % 10 >= 7 for i in held)

    # bundles and ablation masks
    rng = np.random.default_rng(5)
    top = rng.choice(512, K, replace=False)
    labels = np.array([0] * 8 + [2] * 10 + [0] * 6)  # cluster 1 empty
    mA = rng.random(K) + 0.1
    bs = make_bundles(top, mA, labels)
    assert [b['name'] for b in bs] == ['b1', 'b2'], bs
    assert bs[0]['channels'] == [int(c) for c in top[labels == 0]] and bs[1]['channels'] == top[8:18].tolist()
    assert abs(bs[0]['share'] - mA[labels == 0].sum() / mA.sum()) < 1e-12
    assert abs(sum(b['share'] for b in bs) - 1) < 1e-12
    conds, zeros = conditions(top, bs)
    assert conds == ['whole', 'b1', 'b2'] and zeros[0] is None
    a = rng.random((2, 512, 3, 3)) + 1
    for b, z in zip(bs, zeros[1:]):
        on = np.setdiff1d(np.arange(512), z)
        assert len(z) == K - len(b['channels']) and not set(z.tolist()) & set(b['channels'])
        assert set(z.tolist()) | set(b['channels']) == set(top.tolist())
        y = apply_zero(a, z)
        assert (y[:, z] == 0).all() and np.array_equal(y[:, on], a[:, on]) and (a > 0).all()
        assert set(b['channels']) <= set(on.tolist())
    assert zero_set(top, top).size == 0

    # strata fill
    r = np.concatenate([np.full(5, 0.5), np.full(20, 0.1), np.full(100, 0.01)])
    sel = strata(r, np.random.default_rng(2))
    assert len(sel) == 48 and len(set(sel.tolist())) == 48
    assert ((r[sel] >= HIT).sum(), ((r[sel] >= LOW) & (r[sel] < HIT)).sum(), (r[sel] < LOW).sum()) == (5, 20, 23)
    r = np.concatenate([np.full(30, 0.6), np.full(4, 0.2), np.full(60, 0.0)])
    sel = strata(r, np.random.default_rng(2))
    assert ((r[sel] >= HIT).sum(), ((r[sel] >= LOW) & (r[sel] < HIT)).sum(), (r[sel] < LOW).sum()) == (16, 4, 28)
    r = np.random.default_rng(7).random(500) ** 3
    sel = strata(r, np.random.default_rng(2))
    assert ((r[sel] >= HIT).sum(), ((r[sel] >= LOW) & (r[sel] < HIT)).sum(), (r[sel] < LOW).sum()) == (16, 16, 16)
    assert np.array_equal(sel, strata(r, np.random.default_rng(2))), 'strata not reproducible'
    assert len(strata(np.array([0.3] * 16 + [0.05] * 16 + [0.0499] * 16), np.random.default_rng(2))) == 48
    try:
        strata(np.full(30, 0.01), np.random.default_rng(2))
        raise AssertionError('short pool accepted')
    except ValueError:
        pass

    # reply parsing
    assert parse_fire('{"fire": [1, 2, 3, 4, 5, 6]}') == [1, 2, 3, 4, 5, 6]
    assert parse_fire('```json\n{"fire": [48, 7, 3, 12, 9, 1]}\n```') == [48, 7, 3, 12, 9, 1]
    assert parse_fire('Thinking {"x": 1} then {"fire": [6, 5, 4, 3, 2, 1]} done.') == [6, 5, 4, 3, 2, 1]
    assert parse_fire('{"fire": [1, 2, 3, 4, 5, 6]} no: {"fire": [1, 1, 2, 3, 4, 5]}') is None
    for bad in ['{"fire": [1, 2, 3, 4, 5]}', '{"fire": [0, 2, 3, 4, 5, 6]}', '{"fire": [1, 2, 3, 4, 5, 49]}',
                '{"fire": [1, 2, 3, 4, 5, 6.0]}', '{"fire": [true, 2, 3, 4, 5, 6]}', '{"fire": ["1", 2, 3, 4, 5, 6]}',
                '{"fire": [1, 2, 3, 4, 5, 6, 7]}', '{"fire": 3}', 'no json here', '{"fire": [1, 2, 3, 4, 5, 6]']:
        assert parse_fire(bad) is None, bad
    e = parse_expl('Sure.\n```json\n{"class": "one", "explanation": "dog faces", "confidence": 0.7}\n```')
    assert e['explanation'] == 'dog faces' and e['class'] == 'one'
    assert parse_expl('{"explanation": "dogs"}') is None and parse_expl('{"class": "one", "explanation": ""}') is None
    assert parse_expl('{"class": "one", "explanation": "a"} {"class": "unrelated", "explanation": "b"}')[
        'explanation'] == 'b'
    assert parse_judge('{"related": 4, "shared": "fur"}') == (4, 'fur')
    assert parse_judge('x {"related": 2, "shared": ""} y') == (2, '')
    assert parse_judge('{"related": 7, "shared": "a"}') == (None, 'a')
    assert parse_judge('{"related": "4"}') == (None, '') and parse_judge('nothing') == (None, '')

    # retry
    calls = []

    def fake(replies):
        it = iter(replies)

        def ask(label, content):
            calls.append(label)
            return next(it)
        return ask
    g = pick(fake(['{"fire": [1, 1, 2, 3, 4, 5]}', '{"fire": [9, 8, 7, 6, 5, 4]}']), 'x', None)
    assert g['picks'] == [9, 8, 7, 6, 5, 4] and g['attempts'] == 2 and calls == ['x_a0', 'x_a1']
    calls.clear()
    g = pick(fake(['{"fire": [1, 2, 3, 4, 5, 6]}']), 'y', None)
    assert g['picks'] == [1, 2, 3, 4, 5, 6] and g['attempts'] == 1 and calls == ['y_a0']
    calls.clear()
    g = pick(fake(['nope', '{"fire": []}', 'unused']), 'z', None)
    assert g['picks'] is None and g['attempts'] == 2 and calls == ['z_a0', 'z_a1']

    # scoring and own/cross bookkeeping
    ratio = np.linspace(0, 1, 48)
    assert score([48, 1, 2, 3, 4, 5], ratio) == (1.0, 1 / 6)
    b, h = score([20, 21, 22, 30, 31, 40], ratio)
    assert abs(b - 39 / 47) < 1e-12 and abs(h - np.mean(ratio[[19, 20, 21, 29, 30, 39]] >= HIT)) < 1e-12
    conds, bn = ['whole', 'b1', 'b2', 'b3'], ['b1', 'b2', 'b3']
    recs = []
    for ie, e in enumerate(conds):
        for ip, p in enumerate(conds):
            for k in SAMPLES:
                for n in CALLS:
                    v = 10 * ie + ip + k / 10 + n / 100
                    recs.append({'expl': e, 'sample': k, 'pool': p, 'call': n, 'valid': True, 'best': v, 'hit': v / 100})
    recs.append({'expl': 'b1', 'sample': 1, 'pool': 'b1', 'call': 3, 'valid': False})
    recs.append({'expl': 'b2', 'sample': 1, 'pool': 'b1', 'call': 3, 'valid': False, 'error': 'x'})
    S = summarise(recs, conds, bn)

    def mean_of(es, p):
        ip = conds.index(p)
        return float(np.mean([10 * conds.index(e) + ip + k / 10 + n / 100 for e in es for k in SAMPLES for n in CALLS]))
    for p in conds:
        own_e = [p]
        cross_e = [x for x in bn if x != p]
        assert S[p]['own_n'] == 4 and S[p]['cross_n'] == 4 * len(cross_e), (p, S[p])
        assert abs(S[p]['own_best'] - mean_of(own_e, p)) < 1e-9 and abs(S[p]['own_hit'] - mean_of(own_e, p) / 100) < 1e-9
        assert abs(S[p]['cross_best'] - mean_of(cross_e, p)) < 1e-9
        assert abs(S[p]['cross_hit'] - mean_of(cross_e, p) / 100) < 1e-9
    assert S['whole']['cross_n'] == 12 and S['b2']['cross_n'] == 8
    S = summarise([r for r in recs if r['expl'] == 'whole'], conds, bn)
    assert S['b1']['cross_n'] == 0 and np.isnan(S['b1']['cross_hit']) and S['b1']['own_n'] == 0

    # card text, names, labels
    assert card_name('mixed4c:447', 'b1') == card_name('mixed4c:447', 'b1')
    nm = card_name('sum:mixed4c:1+2', 'whole')
    assert len(nm) == 5 and nm[:2] == 'U-' and nm[2:].isdigit()
    assert CARD.replace('{name}', nm).startswith(f'# Unit {nm}\n\nRecorded') and CARD.endswith('the strongest.\n')
    assert act_k(7.3, 7.3) == 10 and act_k(0, 7.3) == 0 and act_k(2.0, 10.0) == 2 and act_k(3.6, 10.0) == 4
    assert decide_gap(0.3) == 'OWN_HIGHER' and decide_gap(-0.3) == 'CROSS_HIGHER' and decide_gap(0.1) == 'SIMILAR'
    assert decide_judge(4.0) == 'HIGH' and decide_judge(2.0) == 'LOW' and decide_judge(3.0) == 'MIDDLE'
    assert decide_gap(float('nan')) == 'UNDEFINED' and decide_judge(float('nan')) == 'UNDEFINED'
    print('selftest ok')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--units', required=True)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    units = [parse_unit(s.strip()) for s in a.units.split(',') if s.strip()]
    names = json.loads((J / 'bank/img/names.json').read_text())
    card, held = pools(names)
    held_model = []

    def get_model():
        if not held_model:
            import torch
            torch.set_num_threads(4)
            m = _bank().load_model(J / 'raw/weights/inception5h.pth')
            for p in m.parameters():
                p.requires_grad_(False)
            held_model.append(m)
        return held_model[0]

    OUT.mkdir(parents=True, exist_ok=True)
    secs = []
    for u in units:
        print('==', u['name'], flush=True)
        secs.append(run_unit(u, card, held, get_model))
        (OUT / 'report.txt').write_text('\n\n'.join(secs) + '\n')
        gc.collect()
        print(secs[-1], flush=True)
    print('report ->', OUT / 'report.txt')


if __name__ == '__main__':
    main()
