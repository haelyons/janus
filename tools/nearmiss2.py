"""Blind maximum and near misses on a pool built from each explanation's own classes (spec: authors/spec_nearmiss2.md).

Per unit, one blind call per explanation names the ImageNet classes the unit should fire on and near-miss classes;
the pool is 48 held photos drawn at random, round robin, from those classes, fixed before any activation is read.
Then, as tools/nearmiss.py, an agent that sees only one explanation picks 6 fire and 6 near-miss photos from the pool,
scored by ratio = the unit's "whole" value (tools/bundlemax.py) / M. Class AUC: P(ratio of a random held photo of an
explanation's fire classes > ratio of one of its near-miss classes), ties 1/2. Outputs: runs/nearmiss2/.

usage: python -I tools/nearmiss2.py --units mixed4c:447,mixed4e:55 [--selftest]
"""
import argparse
import base64
import gc
import json
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))  # -I does not add the script dir
import bundlemax  # noqa: E402
import claude_call  # noqa: E402
import nearmiss  # noqa: E402

EXPL = J / 'authors/explanations/nearmiss2.json'
NAMES = J / 'bank/img/names.json'
ACTS = J / 'bank/img/acts.npz'
BM = J / 'runs/bundlemax'
OUT = J / 'runs/nearmiss2'
N_POOL, MIN_KEPT, SAMPLES, WORKERS = 48, 4, 3, 4
SIDES = ('fire', 'near_miss')

LISTS = ('Below are the 1000 ImageNet-1k class names, separated by semicolons. A unit (one neuron) in an image '
         'classification network has been explained as follows:\n\n{text}\n\n'
         'Using only the explanation, judging each class by what its typical photo shows, list:\n'
         '- "fire": 8 to 15 classes on whose typical photo the unit should fire strongly;\n'
         '- "near_miss": 8 to 15 classes whose typical photo looks related to what the explanation describes or shares '
         'obvious features with it, but on which, according to the explanation, the unit should barely fire.\n'
         'Copy names exactly as listed and put no class in both lists. Reply with only JSON: '
         '{"fire": [...], "near_miss": [...]}\n\n'
         'Class names:\n{names}')


# ---------- class lists ----------

def lists_prompt(text, cats):
    """{names} filled first, so an explanation containing '{names}' is not expanded"""
    return LISTS.replace('{names}', '; '.join(cats)).replace('{text}', text)


def filter_lists(obj, cats):
    """keep exact class names (each once, in reply order); a name in both lists is dropped from both;
    valid = at least MIN_KEPT kept in each list"""
    known = set(cats)
    kept = {k: [] for k in SIDES}
    dropped = {'not_a_class': [], 'in_both': [], 'repeated': []}
    if obj is None:
        return {**kept, 'dropped': dropped, 'parsed': False, 'valid': False}
    for k in SIDES:
        v = obj[k]
        if not isinstance(v, list):
            dropped['not_a_class'].append(v)
            continue
        for x in v:
            if not isinstance(x, str) or x not in known:
                dropped['not_a_class'].append(x)
            elif x in kept[k]:
                dropped['repeated'].append(x)
            else:
                kept[k].append(x)
    both = set(kept['fire']) & set(kept['near_miss'])
    dropped['in_both'] = [x for x in kept['fire'] if x in both]
    for k in SIDES:
        kept[k] = [x for x in kept[k] if x not in both]
    return {**kept, 'dropped': dropped, 'parsed': True, 'valid': all(len(kept[k]) >= MIN_KEPT for k in SIDES)}


def get_lists(text, cats, call):
    """one fresh single-turn call; if not valid, one retry in a fresh conversation. An error reply raises (not cached,
    so a rerun tries again)."""
    content = [{'type': 'text', 'text': lists_prompt(text, cats)}]
    attempts = []
    for _ in range(2):
        r = call(content)
        if r.get('is_error'):
            raise RuntimeError(f'error reply, not cached: {str(r.get("text"))[:80]}')
        f = filter_lists(nearmiss.parse_reply(r['text']), cats)  # last object with both keys
        attempts.append({'reply': r['text'], **f})
        if f['valid']:
            break
    last = attempts[-1]
    return {'text': text, 'valid': last['valid'], 'fire': last['fire'], 'near_miss': last['near_miss'],
            'dropped': last['dropped'], 'attempts': attempts}


def class_lists(expl, cats, path, call=None):
    """class lists per explanation id, cached in path; cached ids are skipped"""
    call = call or (lambda content: claude_call.converse('', [content])[0])
    cache = json.loads(path.read_text()) if path.exists() else {}
    for e, x in expl.items():
        if e in cache and cache[e]['text'] != x['text']:
            raise SystemExit(f'{path}: cached lists of {e} were made from a different text')
    jobs = [e for e in expl if e not in cache]
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(get_lists, expl[e]['text'], cats, call): e for e in jobs}
        for f in as_completed(futs):
            e = futs[f]
            try:
                cache[e] = f.result()
            except Exception as err:  # not cached, so a rerun tries again
                print(f'{e} class lists failed: {err}', flush=True)
                continue
            save_json(cache, path)
    return cache


# ---------- groups and pool ----------

def label_set(class_names, cats):
    """labels k whose class name cats[k] is one of the names (a name listed twice in cats gives both labels)"""
    s = set(class_names)
    return {k for k, c in enumerate(cats) if c in s}


def photos_of(labs, H, labels):
    """held photos (ids, ascending) whose label is in labs"""
    return H[np.isin(labels[H], np.array(sorted(labs), dtype=np.int64))]


def make_groups(expl, lists, cats, H, labels):
    """two groups per valid explanation, in file order: its fire classes, then its near-miss classes"""
    out = []
    for e in expl:
        if lists[e]['valid']:
            for side in SIDES:
                labs = label_set(lists[e][side], cats)
                out.append({'id': f'{e}:{side}', 'classes': lists[e][side], 'labels': sorted(labs),
                            'photos': photos_of(labs, H, labels)})
    return out


def draw_pool(groups, seed, n=N_POOL):
    """groups = [(group id, photo ids)] in order. Cycle through them, each turn drawing one photo uniformly
    (rng.integers) from the group's photos not yet in the pool, skipping groups with none left, until n photos;
    ValueError if every group runs out first. -> photo ids in draw order, the group that drew each"""
    rng = np.random.default_rng(seed)
    ids, by, taken = [], [], set()
    while len(ids) < n:
        drew = False
        for gid, g in groups:
            if len(ids) == n:
                break
            left = [int(i) for i in g if int(i) not in taken]
            if not left:
                continue
            i = left[int(rng.integers(len(left)))]
            ids.append(i)
            by.append(gid)
            taken.add(i)
            drew = True
        if not drew:
            raise ValueError(f'all groups ran out at {len(ids)} of {n} photos')
    return ids, by


def number_pool(ids, by):
    """shuffle with default_rng(1).permutation as tools/nearmiss.py; numbers 1..n"""
    drawn = dict(zip(ids, by))
    perm = np.random.default_rng(1).permutation(np.array(ids, dtype=np.int64))
    return [{'number': k + 1, 'index': int(i), 'drawn_by': drawn[int(i)]} for k, i in enumerate(perm)]


def pool_records(numbered, groups, labels, cats, ratio):
    """number, photo index, label name, ratio, the group that drew it, and every group whose classes contain it"""
    out = []
    for p in numbered:
        lab = int(labels[p['index']])
        out.append({'number': p['number'], 'index': p['index'], 'label': cats[lab], 'ratio': float(ratio[p['index']]),
                    'drawn_by': p['drawn_by'], 'in_groups': [g['id'] for g in groups if lab in g['labels']]})
    return out


# ---------- the unit's values ----------

def check_vals(H, vals, Hb, vHb, conds):
    """vals (aligned with H) must match the 'whole' row of bundlemax's vH on its held ids Hb (np.allclose, rtol 1e-3)"""
    pos = np.searchsorted(H, Hb)
    if (pos >= len(H)).any() or not np.array_equal(H[pos], Hb):
        raise ValueError('values.npz has held ids that are not in H')
    a = np.asarray(vals, dtype=np.float64)[pos]
    b = np.asarray(vHb[list(conds).index('whole')], dtype=np.float64)
    if not np.allclose(a, b, rtol=1e-3):
        raise ValueError(f'held values differ from values.npz "whole": max abs diff {np.abs(a - b).max():.4g}')


def held_values(u, H, udir, get_model):
    """the unit's "whole" value (spatial max of bundlemax.umap, no channels zeroed) on every held photo, cached in
    held_vals.npy after the check against runs/bundlemax/<unit>/values.npz; M = max of its card_vals"""
    bdir = BM / u['name']
    info = json.loads((bdir / 'bundles.json').read_text())
    assert info['unit'] == u['name'], (info['unit'], u['name'])
    s = np.asarray(info['scales'], dtype=np.float64)
    assert len(s) == len(u['chans']), (s, u['chans'])
    with np.load(bdir / 'values.npz') as z:
        M = float(np.max(z['card_vals']))
        Hb, vHb, conds = z['H'], z['vH'], [str(c) for c in z['conditions']]
    assert M > 0, ('M <= 0', M)
    f = udir / 'held_vals.npy'
    cached = f.exists()
    if cached:
        vals = np.load(f)
        print(u['name'], 'held values: cached', flush=True)
    else:
        def um(o):
            return bundlemax.umap(o, u, s).amax(dim=(1, 2)).numpy()
        vals = bundlemax.forward(get_model(), u, H, um, tag='held pool')[0]
    assert vals.shape == (len(H),), (vals.shape, len(H))
    check_vals(H, vals, Hb, vHb, conds)
    if not cached:
        tmp = udir / 'held_vals.tmp.npy'
        np.save(tmp, vals)
        tmp.replace(f)
    return vals, M


# ---------- scores ----------

def class_auc(f, n):
    """P(random fire value > random near-miss value), ties 1/2, from exact pair counts; nan if a side is empty"""
    f, n = np.asarray(f, dtype=np.float64), np.asarray(n, dtype=np.float64)
    if not len(f) or not len(n):
        return float('nan')
    gt = int((f[:, None] > n[None, :]).sum())
    eq = int((f[:, None] == n[None, :]).sum())
    return (2 * gt + eq) / (2 * len(f) * len(n))


def sample_scores(recs, act_of):
    """nearmiss.score of each valid sample, in sample order"""
    return [nearmiss.score(recs[s]['fire'], recs[s]['near_miss'], act_of)
            for s in sorted(recs, key=int) if recs[s]['valid']]


def stats(sc, k):
    v = [s[k] for s in sc]
    return {'mean': float(np.mean(v)), 'min': float(min(v)), 'max': float(max(v))} if v else None


def summarise(e, x, lst, recs, act_of, groups, ratio):
    r = {'id': e, 'source': x.get('source'), 'lists_valid': lst['valid'], 'fire_classes': lst['fire'],
         'near_miss_classes': lst['near_miss'], 'dropped': lst['dropped'], 'list_attempts': len(lst['attempts'])}
    if not lst['valid']:
        return r
    sc = sample_scores(recs, act_of)
    fp, nq = groups[f'{e}:fire']['photos'], groups[f'{e}:near_miss']['photos']
    rf, rn = ratio[fp], ratio[nq]
    r.update(samples={s: {'valid': recs[s]['valid'], 'fire': recs[s].get('fire'), 'near_miss': recs[s].get('near_miss')}
                      for s in sorted(recs, key=int)},
             sample_scores=sc, n_samples=len(recs), n_valid=len(sc),
             fire_hit=stats(sc, 'fire_hit'), near_ok=stats(sc, 'near_ok'), gap=stats(sc, 'gap'),
             gap_vs_chance=nearmiss.decision(sc),
             class_auc=class_auc(rf, rn), n_fire=int(len(fp)), n_near_miss=int(len(nq)),
             mean_ratio_fire=float(rf.mean()) if len(rf) else float('nan'),
             mean_ratio_near_miss=float(rn.mean()) if len(rn) else float('nan'),
             fire_photos=fp.tolist(), near_miss_photos=nq.tolist())
    return r


def unit_scores(name, M, expl, lists, groups, pool, replies, ratio):
    act_of = {p['number']: p['ratio'] for p in pool}
    gd = {g['id']: g for g in groups}
    rows = [summarise(e, expl[e], lists[e], replies.get(e, {}), act_of, gd, ratio) for e in expl]
    ranked = sorted([r for r in rows if r.get('n_valid')], key=lambda r: -r['fire_hit']['mean'])
    gs = [{'id': g['id'], 'classes': g['classes'], 'labels': g['labels'], 'n_held': int(len(g['photos'])),
           'drew': sum(p['drawn_by'] == g['id'] for p in pool), 'in_pool': sum(g['id'] in p['in_groups'] for p in pool)}
          for g in groups]
    return {'unit': name, 'M': M, 'chance': nearmiss.chance(act_of), 'groups': gs, 'pool': pool,
            'explanations': rows, 'ranking': [[r['id'], r['fire_hit']['mean']] for r in ranked]}


def row_line(r):
    head = f'{r["id"]}  source={r["source"]}'
    if not r['lists_valid']:
        return (f'{head}  class lists invalid (kept fire {len(r["fire_classes"])}, near_miss '
                f'{len(r["near_miss_classes"])}, after {r["list_attempts"]} calls)')
    if r['n_valid']:
        f = r['fire_hit']
        p = (f'fire_hit={f["mean"]:.3f} ({f["min"]:.3f}-{f["max"]:.3f})  near_ok={r["near_ok"]["mean"]:.3f}  '
             f'gap={r["gap"]["mean"]:+.3f}')
    else:
        p = 'no valid samples'
    return (f'{head}  {p}  valid={r["n_valid"]}/{SAMPLES}  class_auc={r["class_auc"]:.3f}  '
            f'n={r["n_fire"]}/{r["n_near_miss"]}  mean_ratio={r["mean_ratio_fire"]:.3f}/{r["mean_ratio_near_miss"]:.3f}  '
            f'gap_vs_chance={r["gap_vs_chance"]}')


def section(S):
    c = S['chance']
    L = [f'== {S["unit"]}  M={S["M"]:.4g}', f'chance  fire_hit={c["fire_hit"]:.3f}  near_ok={c["near_ok"]:.3f}']
    for g in S['groups']:
        L.append(f'pool {g["id"]}: drew {g["drew"]}, {g["in_pool"]} pool photos in its classes '
                 f'({len(g["classes"])} classes, {g["n_held"]} held photos)')
    L += [row_line(r) for r in S['explanations']]
    L.append('ranked by fire_hit mean: ' + (', '.join(f'{e} {m:.3f}' for e, m in S['ranking']) or 'none'))
    return '\n'.join(L)


# ---------- calls ----------

def pick_calls(path, ids, expl, sheet_b64):
    """as tools/nearmiss.py: SAMPLES fresh samples per explanation, up to WORKERS in parallel, cached in path"""
    cache = json.loads(path.read_text()) if path.exists() else {}
    jobs = [(e, s) for e in ids for s in range(SAMPLES) if str(s) not in cache.get(e, {})]
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(nearmiss.run_sample, expl[e]['text'], sheet_b64): (e, s) for e, s in jobs}
        for f in as_completed(futs):
            e, s = futs[f]
            try:
                rec = f.result()
            except Exception as err:  # not cached, so a rerun tries again
                print(f'{e} sample {s} failed: {err}', flush=True)
                continue
            cache.setdefault(e, {})[str(s)] = rec
            save_json(cache, path)
    return cache


def save_json(obj, path):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(path)


# ---------- one unit ----------

def run_unit(u, expl, H, labels, cats, get_model):
    name = u['name']
    udir = OUT / name
    udir.mkdir(parents=True, exist_ok=True)

    # 1. class lists, from the explanation text only
    print(name, 'class lists', flush=True)
    lists = class_lists(expl, cats, udir / 'classes.json')
    missing = [e for e in expl if e not in lists]
    if missing:
        return f'== {name}  class lists missing for {missing} (calls failed, not cached); rerun'

    # 2. pool, drawn from the class groups; no activation has been read yet
    groups = make_groups(expl, lists, cats, H, labels)
    ids, by = draw_pool([(g['id'], g['photos']) for g in groups], zlib.crc32(name.encode()))
    numbered = number_pool(ids, by)

    # 3. the unit's values on H, read only now
    vals, M = held_values(u, H, udir, get_model)
    ratio = np.full(len(labels), np.nan)
    ratio[H] = vals.astype(np.float64) / M
    pool = pool_records(numbered, groups, labels, cats, ratio)
    pf, rf = udir / 'pool.json', udir / 'replies.json'
    if pf.exists() and rf.exists():
        old = [(p['number'], p['index']) for p in json.loads(pf.read_text())]
        if old != [(p['number'], p['index']) for p in pool]:
            raise SystemExit(f'{pf}: the pool drawn now differs from the one {rf.name} was made on; stopping')
    save_json(pool, pf)
    sheet = udir / 'sheet.jpg'
    nearmiss.make_sheet(pool, str(sheet))
    b64 = base64.b64encode(sheet.read_bytes()).decode()

    # 4. blind picks, as tools/nearmiss.py
    print(name, 'picks', flush=True)
    replies = pick_calls(rf, [e for e in expl if lists[e]['valid']], expl, b64)

    # 5. scores
    S = unit_scores(name, M, expl, lists, groups, pool, replies, ratio)
    save_json(S, udir / 'scores.json')
    return section(S)


# ---------- selftest ----------

def selftest():
    cats = ['tabby', 'tiger cat', 'Persian cat', 'crane', 'goldfish', 'crane', 'lion', 'cougar', 'lynx', 'leopard',
            'jaguar', 'cheetah']

    # prompt: the template verbatim, {names} filled before {text}
    p = lists_prompt('likes {names} and {"x"}', ['a', 'b c'])
    assert p.startswith('Below are the 1000 ImageNet-1k class names, separated by semicolons. A unit (one neuron) in '
                        'an image classification network has been explained as follows:\n\nlikes {names} and {"x"}\n\n'
                        'Using only the explanation, judging each class by what its typical photo shows, list:\n'
                        '- "fire": 8 to 15 classes on whose'), p
    assert '\n- "near_miss": 8 to 15 classes whose typical photo looks related to what the explanation describes or ' \
           'shares obvious features with it, but on' in p
    assert p.endswith('the unit should barely fire.\nCopy names exactly as listed and put no class in both lists. Reply '
                      'with only JSON: {"fire": [...], "near_miss": [...]}\n\nClass names:\na; b c'), p

    # class-list parsing and filtering
    def lists_of(reply):
        return filter_lists(nearmiss.parse_reply(reply), cats)
    ok = '{"fire": ["tabby", "tiger cat", "Persian cat", "lion"], "near_miss": ["goldfish", "crane", "cheetah", "jaguar"]}'
    f = lists_of('Sure. {"fire": ["x"], "near_miss": ["y"]} then ```json\n' + ok + '\n``` {"note": 1}')
    assert f['valid'] and f['fire'] == ['tabby', 'tiger cat', 'Persian cat', 'lion'], f
    assert f['near_miss'] == ['goldfish', 'crane', 'cheetah', 'jaguar']
    assert f['dropped'] == {'not_a_class': [], 'in_both': [], 'repeated': []}, f
    f = lists_of('{"fire": ["tabby", "Tabby", "tiger cat", "lion ", "cougar", "lynx", "tabby"], '
                 '"near_miss": ["lynx", "goldfish", "crane", "leopard", "jaguar", 3, null]}')
    assert f['fire'] == ['tabby', 'tiger cat', 'cougar'] and f['near_miss'] == ['goldfish', 'crane', 'leopard', 'jaguar']
    assert f['dropped'] == {'not_a_class': ['Tabby', 'lion ', 3, None], 'in_both': ['lynx'], 'repeated': ['tabby']}, f
    assert not f['valid']  # 3 kept in fire
    f = lists_of('{"fire": ["tabby", "tiger cat", "Persian cat", "lion", "cougar"], '
                 '"near_miss": ["goldfish", "crane", "cheetah", "cougar", "jaguar"]}')
    assert f['valid'] and f['dropped']['in_both'] == ['cougar'] and 'cougar' not in f['fire'] + f['near_miss']
    assert len(f['fire']) == 4 and len(f['near_miss']) == 4
    f = lists_of('{"fire": "tabby", "near_miss": ["goldfish", "crane", "cheetah", "jaguar"]}')
    assert not f['valid'] and f['fire'] == [] and f['dropped']['not_a_class'] == ['tabby']
    for bad in ['no json', '{"fire": ["tabby", "tiger cat", "Persian cat", "lion"]}', '{"fire": [], "near_miss": []}']:
        assert not lists_of(bad)['valid'], bad
    assert not lists_of('no json')['parsed']

    # one retry in a fresh call; an error reply raises
    calls = []

    def fake(texts):
        it = iter(texts)

        def call(content):
            calls.append(content)
            return {'text': next(it), 'is_error': False}
        return call
    few = '{"fire": ["tabby"], "near_miss": ["goldfish"]}'
    g = get_lists('t', cats, fake([few, ok, 'unused']))
    assert g['valid'] and len(g['attempts']) == 2 and len(calls) == 2 and g['fire'][0] == 'tabby' and g['text'] == 't'
    assert calls[0] == [{'type': 'text', 'text': lists_prompt('t', cats)}] and calls[1] == calls[0]
    calls.clear()
    g = get_lists('t', cats, fake([ok, 'unused']))
    assert g['valid'] and len(g['attempts']) == 1 and len(calls) == 1
    calls.clear()
    g = get_lists('t', cats, fake([few, 'no json', 'unused']))
    assert not g['valid'] and len(g['attempts']) == 2 and len(calls) == 2 and g['fire'] == []
    try:
        get_lists('t', cats, lambda c: {'text': 'overloaded', 'is_error': True})
        raise AssertionError('error reply accepted')
    except RuntimeError:
        pass

    # labels and group photos
    assert label_set(['crane', 'tabby'], cats) == {0, 3, 5}
    H = np.array([0, 2, 4, 6, 8, 10])
    labels = np.array([0, 1, 3, 1, 5, 0, 2, 2, 0, 9, 6, 6])
    assert photos_of({0, 3, 5}, H, labels).tolist() == [0, 2, 4, 8] and photos_of(set(), H, labels).tolist() == []

    # pool drawing: determinism, round robin, the rng.integers rule, skipping an exhausted group
    g3 = [('a:fire', [10, 11, 12]), ('a:near_miss', list(range(20, 30))), ('b:fire', list(range(30, 40)))]
    ids, by = draw_pool(g3, 123, 10)
    assert (ids, by) == draw_pool(g3, 123, 10)
    assert by == ['a:fire', 'a:near_miss', 'b:fire'] * 3 + ['a:near_miss'], by
    assert sorted(i for i, b in zip(ids, by) if b == 'a:fire') == [10, 11, 12]
    assert all(i in dict(g3)[b] for i, b in zip(ids, by)) and len(set(ids)) == 10
    assert ids[0] == 10 + int(np.random.default_rng(123).integers(3))
    # no duplicates when groups share photos; exact fit; error when all run out
    ov = [('x', [1, 2, 3, 4]), ('y', [3, 4, 5, 6])]
    ids, by = draw_pool(ov, 5, 6)
    assert sorted(ids) == [1, 2, 3, 4, 5, 6] and all(i in dict(ov)[b] for i, b in zip(ids, by))
    ids, by = draw_pool([('x', [1, 2]), ('y', [3, 4])], 0, 4)
    assert sorted(ids) == [1, 2, 3, 4] and by == ['x', 'y', 'x', 'y']
    for groups, n in [([('x', [1, 2]), ('y', [2, 3])], 4), ([], 4), ([('x', [])], 1)]:
        try:
            draw_pool(groups, 0, n)
            raise AssertionError(f'short pool accepted {groups}')
        except ValueError:
            pass
    big = [(f'g{k}', list(range(100 * k, 100 * k + 20))) for k in range(6)]
    ids, by = draw_pool(big, zlib.crc32(b'mixed4c:447'), 48)
    assert all(by.count(f'g{k}') == 8 for k in range(6)) and len(set(ids)) == 48
    numbered = number_pool(ids, by)
    assert numbered == number_pool(ids, by) and [p['number'] for p in numbered] == list(range(1, 49))
    assert [p['index'] for p in numbered] == np.random.default_rng(1).permutation(np.array(ids)).tolist()
    assert all(p['drawn_by'] == f'g{p["index"] // 100}' for p in numbered)

    # check of the held values against values.npz
    Hc = np.arange(0, 300, 3)
    v = np.random.default_rng(4).random(len(Hc)).astype(np.float32) + 0.5
    sel = [1, 5, 7, 40]
    check_vals(Hc, v, Hc[sel], np.stack([v[sel] * 2, v[sel] * 1.0005]), ['b1', 'whole'])
    for ids_b, vH in [(Hc[sel], np.stack([v[sel] * 1.01])), (Hc[sel] + 1, np.stack([v[sel]])),
                      (np.array([297, 301]), np.stack([v[[99, 99]]]))]:
        try:
            check_vals(Hc, v, ids_b, vH, ['whole'])
            raise AssertionError('mismatch accepted')
        except ValueError:
            pass

    # class AUC against hand values, with ties
    assert class_auc([1, 2, 3], [0, 0]) == 1.0 and class_auc([0], [1]) == 0.0 and class_auc([1, 1], [1, 1]) == 0.5
    assert class_auc([0.5, 0.2], [0.2, 0.1]) == 0.875
    assert class_auc([0.3, 0.3, 0.1], [0.3, 0.2]) == 0.5
    assert class_auc([0.4, 0.1, 0.0], [0.0, 0.1]) == 2 / 3
    a, b = np.random.default_rng(9).random(30).round(1), np.random.default_rng(10).random(20).round(1)
    assert abs(class_auc(a, b) + class_auc(b, a) - 1) < 1e-12
    assert np.isnan(class_auc([], [1])) and np.isnan(class_auc([1], []))

    # pick scoring, as tools/nearmiss.py
    act_of = {k: 0.5 if k <= 16 else 0.2 if k <= 32 else 0.0 for k in range(1, 49)}
    recs = {'0': {'valid': True, 'fire': [1, 2, 3, 4, 5, 6], 'near_miss': [33, 34, 35, 36, 37, 38]},
            '1': {'valid': False, 'replies': ['x', 'y']},
            '2': {'valid': True, 'fire': [1, 2, 3, 17, 18, 19], 'near_miss': [20, 21, 22, 33, 34, 35]}}
    sc = sample_scores(recs, act_of)
    assert len(sc) == 2 and sc[0]['fire_hit'] == 1 and sc[0]['near_ok'] == 1 and np.isclose(sc[0]['gap'], 0.5)
    assert sc[1]['fire_hit'] == 0.5 and sc[1]['near_ok'] == 0.5 and np.isclose(sc[1]['gap'], 0.25)
    s = nearmiss.score([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], {k: 0.3 if k <= 6 else 0.1 for k in range(1, 49)})
    assert s['fire_hit'] == 1 and s['near_ok'] == 0
    c = nearmiss.chance(act_of)
    assert np.isclose(c['fire_hit'], 16 / 48) and np.isclose(c['near_ok'], 16 / 48) and c['gap'] == 0
    assert stats(sc, 'fire_hit') == {'mean': 0.75, 'min': 0.5, 'max': 1.0} and stats([], 'gap') is None
    assert nearmiss.decision(sc) == 'ABOVE' and nearmiss.decision([]) == 'NONE'

    # a synthetic unit end to end (no model, no Claude): groups, pool, scores, report
    cats2 = [f'c{k}' for k in range(12)]
    labels2 = np.repeat(np.arange(12), 10)
    H2 = np.arange(120)
    ratio2 = labels2 / 11.0
    expl = {'A': {'text': 'a', 'source': 's1'}, 'B': {'text': 'b', 'source': 's2'}, 'C': {'text': 'c', 'source': 's3'}}
    dr = {'not_a_class': [], 'in_both': [], 'repeated': []}
    lists = {'A': {'valid': True, 'fire': ['c0', 'c1', 'c2', 'c3'], 'near_miss': ['c4', 'c5', 'c6', 'c7'],
                   'dropped': dr, 'attempts': [{}]},
             'B': {'valid': False, 'fire': ['c0'], 'near_miss': [], 'dropped': dr, 'attempts': [{}, {}]},
             'C': {'valid': True, 'fire': ['c8', 'c9', 'c10', 'c11'], 'near_miss': ['c0', 'c1', 'c2', 'c3'],
                   'dropped': dr, 'attempts': [{}]}}
    groups = make_groups(expl, lists, cats2, H2, labels2)
    assert [g['id'] for g in groups] == ['A:fire', 'A:near_miss', 'C:fire', 'C:near_miss']
    assert all(len(g['photos']) == 40 for g in groups) and groups[0]['labels'] == [0, 1, 2, 3]
    ids, by = draw_pool([(g['id'], g['photos']) for g in groups], zlib.crc32(b'test'))
    pool = pool_records(number_pool(ids, by), groups, labels2, cats2, ratio2)
    for p in pool:
        lab = int(labels2[p['index']])
        assert p['label'] == f'c{lab}' and p['ratio'] == lab / 11.0
        assert p['in_groups'] == (['A:fire', 'C:near_miss'] if lab < 4 else ['A:near_miss'] if lab < 8 else ['C:fire'])
    replies = {'A': {'0': {'valid': True, 'fire': [1, 2, 3, 4, 5, 6], 'near_miss': [7, 8, 9, 10, 11, 12]},
                     '1': {'valid': True, 'fire': [13, 14, 15, 16, 17, 18], 'near_miss': [19, 20, 21, 22, 23, 24]},
                     '2': {'valid': False}},
               'C': {'0': {'valid': False}}}
    S = unit_scores('test', 2.0, expl, lists, groups, pool, replies, ratio2)
    rA, rB, rC = S['explanations']
    assert [r['id'] for r in S['explanations']] == ['A', 'B', 'C'] and not rB['lists_valid'] and 'class_auc' not in rB
    assert rA['class_auc'] == 0.0 and rC['class_auc'] == 1.0 and rA['n_fire'] == rA['n_near_miss'] == 40
    assert np.isclose(rC['mean_ratio_fire'], 9.5 / 11) and np.isclose(rC['mean_ratio_near_miss'], 1.5 / 11)
    act2 = {p['number']: p['ratio'] for p in pool}
    want = [nearmiss.score(replies['A'][k]['fire'], replies['A'][k]['near_miss'], act2) for k in ('0', '1')]
    assert rA['n_valid'] == 2 and rA['sample_scores'] == want and rC['n_valid'] == 0 and rC['gap_vs_chance'] == 'NONE'
    assert np.isclose(rA['fire_hit']['mean'], np.mean([w['fire_hit'] for w in want]))
    assert [g['drew'] for g in S['groups']] == [12, 12, 12, 12]
    assert S['groups'][0]['in_pool'] == S['groups'][3]['in_pool'] == 24 and S['groups'][1]['in_pool'] == 12
    assert S['ranking'] == [['A', rA['fire_hit']['mean']]] and S['chance'] == nearmiss.chance(act2)
    txt = section(S)
    assert txt.startswith('== test  M=2\nchance  fire_hit=') and 'B  source=s2  class lists invalid' in txt
    assert 'C  source=s3  no valid samples  valid=0/3  class_auc=1.000  n=40/40' in txt
    assert txt.splitlines()[-1] == f'ranked by fire_hit mean: A {rA["fire_hit"]["mean"]:.3f}'
    json.dumps(S)
    print('selftest ok')


def main():
    if '--selftest' in sys.argv[1:]:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--units', required=True)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    units = [bundlemax.parse_unit(s.strip()) for s in a.units.split(',') if s.strip()]
    expl_all = json.loads(EXPL.read_text())
    missing = [u['name'] for u in units if u['name'] not in expl_all]
    if missing:
        raise SystemExit(f'units not in {EXPL}: {missing}')
    names = json.loads(NAMES.read_text())
    _, H = bundlemax.pools(names)
    with np.load(ACTS) as z:
        labels = np.asarray(z['classes'], dtype=np.int64)
    assert len(labels) == len(names) and labels.min() >= 0 and labels.max() < 1000, (len(labels), len(names))
    from torchvision.models import GoogLeNet_Weights
    cats = list(GoogLeNet_Weights.IMAGENET1K_V1.meta['categories'])
    assert len(cats) == 1000, len(cats)
    held_model = []

    def get_model():
        if not held_model:
            import torch
            torch.set_num_threads(4)
            m = bundlemax._bank().load_model(J / 'raw/weights/inception5h.pth')
            for p in m.parameters():
                p.requires_grad_(False)
            held_model.append(m)
        return held_model[0]

    OUT.mkdir(parents=True, exist_ok=True)
    secs = []
    for u in units:
        print('==', u['name'], flush=True)
        secs.append(run_unit(u, expl_all[u['name']], H, labels, cats, get_model))
        (OUT / 'report.txt').write_text('\n\n'.join(secs) + '\n')
        gc.collect()
        print(secs[-1], flush=True)
    print('report ->', OUT / 'report.txt')


if __name__ == '__main__':
    main()
