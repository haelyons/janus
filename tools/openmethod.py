"""Look, then score: one pipeline run the same way on any image unit (spec: authors/spec_openmethod.md).

Per unit (one channel of mixed3a-mixed5b): its spatial max on every card-pool (C) and held-pool (H) photo; a card of
its 20 strongest C photos (a red box where it is most active) and 10 random C photos; readings (explanations) from
agents that see the card under the system prompts '' (bare), Q0 and F1, 2 samples each, plus "wrong" = the bare_s1
explanation of another unit. Every reading is scored as tools/simulate.py on a pool of 24 same-class pairs of held
photos, drawn from the readings' class lists (as tools/nearmiss2.py) before any held activation is read, and by a
within-pair score. Outputs: runs/openmethod/.

usage: python -I tools/openmethod.py [--units-file authors/round20_units.json] [--selftest]
"""
import argparse
import base64
import gc
import hashlib
import json
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))  # -I does not add the script dir
import bundlemax  # noqa: E402
import claude_call  # noqa: E402
import nearmiss  # noqa: E402
import nearmiss2  # noqa: E402
import run_subjects  # noqa: E402
import simulate  # noqa: E402

OUT = J / 'runs/openmethod'
UNITS_FILE = 'authors/round20_units.json'
WEIGHTS = J / 'raw/weights/inception5h.pth'
NCH = {'mixed3a': 256, 'mixed3b': 480, 'mixed4a': 508, 'mixed4b': 512, 'mixed4c': 512, 'mixed4d': 528,
       'mixed4e': 832, 'mixed5a': 832, 'mixed5b': 1024}
BATCH, TILE, WORKERS = 64, 224, 4
N_CARD, N_RANDOM, N_PAIRS, PAIR_DIFF = 20, 10, 24, 0.1
ARMS = (('bare', None), ('Q0', 'arms/Q0_shared.txt'), ('F1', 'arms/F1_forced.txt'))
SAMPLES = (1, 2)
READINGS = [f'{a}_s{k}' for a, _ in ARMS for k in SAMPLES] + ['wrong']

CARD = ('# Unit {name}\n\nRecorded from one unit (a single neuron) in an image classification network, run over '
        'several thousand natural photos. The first attached grid shows the 20 photos on which the unit is most '
        'active, strongest first; a red box marks where in each photo it is most active. Under each is its '
        'activation, scaled 0-10 relative to the strongest. The second grid shows 10 photos drawn at random from the '
        'same collection, with their activation on the same scale.')


# ---------- units ----------

def parse_unit(s):
    """'mixedNx:c' (a single channel of mixed3a-mixed5b) -> dict(name, layer, chan)"""
    p = s.split(':')
    if (len(p) != 2 or p[0] not in NCH or not (p[1].isascii() and p[1].isdigit()) or str(int(p[1])) != p[1]
            or int(p[1]) >= NCH[p[0]]):
        raise ValueError(f'bad unit {s!r}')
    return {'name': s, 'layer': p[0], 'chan': int(p[1])}


def check_units(data):
    """entries of the units file -> [{name, layer, chan, wrong_from}]; ValueError unless every unit is listed once and
    every wrong_from is another unit of the file"""
    if not isinstance(data, list) or not data:
        raise ValueError('the units file must be a non-empty list')
    out = []
    for x in data:
        if not isinstance(x, dict) or not isinstance(x.get('unit'), str) or not isinstance(x.get('wrong_from'), str):
            raise ValueError(f'bad entry {x!r}')
        out.append({**parse_unit(x['unit']), 'wrong_from': x['wrong_from']})
    names = [e['name'] for e in out]
    if len(set(names)) != len(names):
        raise ValueError('a unit is listed twice')
    for e in out:
        if e['wrong_from'] == e['name'] or e['wrong_from'] not in names:
            raise ValueError(f'{e["name"]}: wrong_from {e["wrong_from"]!r} is not another unit of the file')
    return out


# ---------- activations ----------

def spatial_max(m):
    """m: (n, s, s) maps -> each map's max, its row and col (first in row-major order on ties), and s"""
    m = np.asarray(m)
    n, h, w = m.shape
    assert h == w, ('map is not square', m.shape)
    flat = m.reshape(n, -1)
    idx = flat.argmax(axis=1)
    return flat[np.arange(n), idx], idx // w, idx % w, h


def forward_max(model, u, ids, tag=''):
    """spatial_max of channel u['chan'] of the output of the module named u['layer'], on photo ids (bundlemax.net_input,
    batches of 64), the forward pass stopped at that layer by img_acts.Stop"""
    import torch
    Stop = bundlemax._bank().Stop
    got = {}

    def stop(_m, _i, out):
        assert out.shape[1] == NCH[u['layer']], tuple(out.shape)
        got['m'] = out[:, u['chan']].detach().numpy().astype(np.float32)
        raise Stop

    vals, rows, cols, sizes = [], [], [], set()
    h = dict(model.named_modules())[u['layer']].register_forward_hook(stop)
    try:
        for b, s in enumerate(range(0, len(ids), BATCH)):
            x = bundlemax.net_input(ids[s:s + BATCH])
            got.clear()
            try:
                with torch.no_grad():
                    model(x)
            except Stop:
                pass
            v, r, c, n = spatial_max(got['m'])
            vals.append(v)
            rows.append(r)
            cols.append(c)
            sizes.add(n)
            if b % 20 == 19:
                print(f'  {tag} {min(s + BATCH, len(ids))}/{len(ids)}', flush=True)
    finally:
        h.remove()
    assert len(sizes) == 1, sizes
    return np.concatenate(vals), np.concatenate(rows), np.concatenate(cols), sizes.pop()


def unit_values(u, C, H, udir, get_model):
    """value, row and col of the max on every photo of C and H, and the map size; cached in values.npz"""
    f = udir / 'values.npz'
    if f.exists():
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        if str(d['unit']) != u['name'] or not np.array_equal(d['C'], C) or not np.array_equal(d['H'], H):
            raise SystemExit(f'{f}: made for another unit or other photo pools; stopping')
        print(u['name'], 'values: cached', flush=True)
        return d
    v, r, c, size = forward_max(get_model(), u, np.concatenate([C, H]), tag=u['name'])
    n = len(C)
    d = {'unit': np.array(u['name']), 'C': C, 'H': H, 'vC': v[:n], 'vH': v[n:], 'rC': r[:n], 'cC': c[:n],
         'rH': r[n:], 'cH': c[n:], 'size': np.array(size)}
    tmp = udir / 'values.tmp.npz'
    np.savez_compressed(tmp, **d)
    tmp.replace(f)
    return d


# ---------- card ----------

def box_px(row, col, size, tile=TILE):
    """inclusive pixel box (x0, y0, x1, y1) around the map cells (row-1, col-1) to (row+1, col+1), clipped to the map,
    each cell tile / size pixels"""
    cell = tile / size
    r0, r1 = max(row - 1, 0), min(row + 1, size - 1)
    c0, c1 = max(col - 1, 0), min(col + 1, size - 1)
    return (int(round(c0 * cell)), int(round(r0 * cell)), int(round((c1 + 1) * cell)) - 1,
            int(round((r1 + 1) * cell)) - 1)


def draw_box(img, box):
    """a 2 px red rectangle on img, inside box"""
    ImageDraw.Draw(img).rectangle(box, outline=(255, 0, 0), width=2)


def make_card_boxed(path, ids, vals, M, boxes):
    """bundlemax.make_card (5 columns of whole 224 px photos, '#n   activation k' in a white strip under each), with a
    box drawn on each photo"""
    assert len(boxes) == len(ids), (len(boxes), len(ids))
    cols, tile, lab = 5, TILE, 24
    rows = -(-len(ids) // cols)
    im = Image.new('RGB', (cols * tile, rows * (tile + lab)), 'white')
    dr, f = ImageDraw.Draw(im), bundlemax._font(16)
    for n, (i, v, b) in enumerate(zip(ids, vals, boxes)):
        x0, y0 = (n % cols) * tile, (n // cols) * (tile + lab)
        p = bundlemax.photo(i)
        draw_box(p, b)
        im.paste(p, (x0, y0))
        dr.text((x0 + 4, y0 + tile + 3), f'#{n + 1}   activation {bundlemax.act_k(v, M)}', fill='black', font=f)
    im.save(path, quality=90)


def top_positions(vC, n=N_CARD):
    """positions into C of the n highest values, strongest first (stable on ties)"""
    return np.argsort(-np.asarray(vC), kind='stable')[:n]


def random_positions(n_C, top, unit, n=N_RANDOM):
    """positions into C of n photos drawn with default_rng(crc32(unit)).choice, without replacement, among the C
    photos (in C order) outside the top; in draw order"""
    rest = np.setdiff1d(np.arange(n_C), np.asarray(top))
    return np.random.default_rng(zlib.crc32(unit.encode())).choice(rest, n, replace=False)


def card_content(name, card_block, random_block):
    return [{'type': 'text', 'text': CARD.replace('{name}', name)}, card_block, random_block,
            {'type': 'text', 'text': run_subjects.TASK}]


def make_cards(u, d, udir):
    """card.jpg, random.jpg and the reading prompt's content"""
    C, vC, size = d['C'], d['vC'].astype(np.float64), int(d['size'])
    M = float(vC.max())
    assert M > 0, ('M <= 0', M)
    top = top_positions(vC)
    boxes = [box_px(int(r), int(c), size) for r, c in zip(d['rC'][top], d['cC'][top])]
    make_card_boxed(udir / 'card.jpg', C[top], vC[top], M, boxes)
    rnd = random_positions(len(C), top, u['name'])
    bundlemax.make_card(udir / 'random.jpg', C[rnd], vC[rnd], M)
    name = bundlemax.card_name(u['name'], 'whole')
    content = card_content(name, run_subjects.img_block(udir / 'card.jpg'), run_subjects.img_block(udir / 'random.jpg'))
    meta = {'name': name, 'M': M, 'map_size': size, 'top_ids': C[top].tolist(), 'top_values': vC[top].tolist(),
            'top_rows': d['rC'][top].tolist(), 'top_cols': d['cC'][top].tolist(), 'boxes': [list(b) for b in boxes],
            'random_ids': C[rnd].tolist(), 'random_values': vC[rnd].tolist()}
    return meta, content


# ---------- readings ----------

def sha256(s):
    return hashlib.sha256(s.encode()).hexdigest()


def get_reading(system_prompt, content, call=None):
    """one fresh single-turn call, parsed as bundlemax.parse_expl; one fresh retry on a parse failure, then missing.
    An is_error reply raises (not cached, so a rerun tries again)."""
    call = call or (lambda sp, c: claude_call.converse(sp, [c])[0])
    texts = []
    for a in range(2):
        r = call(system_prompt, content)
        if r.get('is_error'):
            raise RuntimeError(f'error reply, not cached: {str(r.get("text"))[:80]}')
        texts.append(r['text'])
        o = bundlemax.parse_expl(r['text'])
        if o is not None:
            return {'valid': True, 'class': o.get('class'), 'explanation': o['explanation'], 'parsed': o,
                    'text': r['text'], 'texts': texts, 'attempts': a + 1}
    return {'valid': False, 'class': None, 'explanation': None, 'parsed': None, 'text': texts[-1], 'texts': texts,
            'attempts': 2}


def readings(u, content, arms, udir):
    """every (arm, sample) reading, up to WORKERS in parallel, cached in readings.json keyed '<arm>_s<k>'"""
    path = udir / 'readings.json'
    cache = json.loads(path.read_text()) if path.exists() else {}
    csha = sha256(json.dumps(content, sort_keys=True))
    jobs = []
    for arm, sp in arms:
        ssha = sha256(sp)
        for k in SAMPLES:
            key = f'{arm}_s{k}'
            if key not in cache:
                jobs.append((key, arm, k, sp, ssha))
            elif cache[key]['content_sha256'] != csha or cache[key]['system_prompt_sha256'] != ssha:
                raise SystemExit(f'{path}: cached {key} was made from a different card or system prompt; stopping')
    if jobs:
        print(u['name'], 'readings', [j[0] for j in jobs], flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(get_reading, sp, content): (key, arm, k, ssha) for key, arm, k, sp, ssha in jobs}
        for f in as_completed(futs):
            key, arm, k, ssha = futs[f]
            try:
                rec = f.result()
            except Exception as err:  # not cached, so a rerun tries again
                print(f'{u["name"]} {key} failed: {err}', flush=True)
                continue
            cache[key] = {'unit': u['name'], 'reading': key, 'arm': arm, 'sample': k, 'model': claude_call.MODEL,
                          'system_prompt_sha256': ssha, 'content_sha256': csha, **rec}
            nearmiss2.save_json(cache, path)
    return cache


def reading_texts(unit, wrong_from, recs):
    """recs = {unit: {'<arm>_s<k>': record}}. -> ({reading id: explanation or None} in READINGS order, {reading id:
    class}, [reading ids whose call failed and was not cached]); 'wrong' is wrong_from's bare_s1"""
    src = [(rid, unit, rid) for rid in READINGS[:-1]] + [('wrong', wrong_from, 'bare_s1')]
    texts, classes, failed = {}, {}, []
    for rid, owner, key in src:
        r = recs.get(owner, {}).get(key)
        if r is None:
            failed.append(rid)
        texts[rid] = None if r is None else r['explanation']
        classes[rid] = None if r is None else r['class']
    return texts, classes, failed


# ---------- pool of same-class pairs ----------

def held_by_label(H, labels):
    """{label: its held photo ids, ascending}"""
    lab = np.asarray(labels)[H]
    return {int(k): H[lab == k] for k in np.unique(lab)}


def draw_pairs(groups, photos, seed, n=N_PAIRS):
    """groups = [(group id, class labels)] in order; photos = {label: held photo ids}. Cycle through the groups, each
    turn drawing one class uniformly (rng.integers) from the group's classes not yet in the pool that have at least 2
    held photos, then 2 of its photos without replacement (rng.choice); a group with none left is skipped; stop at n
    classes. ValueError if every group runs out first. -> [{pair, label, ids, group}] in draw order"""
    rng = np.random.default_rng(seed)
    pairs, used = [], set()
    while len(pairs) < n:
        drew = False
        for gid, labs in groups:
            if len(pairs) == n:
                break
            left = [int(k) for k in labs if int(k) not in used and len(photos.get(int(k), ())) >= 2]
            if not left:
                continue
            k = left[int(rng.integers(len(left)))]
            two = rng.choice(np.asarray(photos[k], dtype=np.int64), 2, replace=False)
            pairs.append({'pair': len(pairs) + 1, 'label': k, 'ids': [int(i) for i in two], 'group': gid})
            used.add(k)
            drew = True
        if not drew:
            raise ValueError(f'all groups ran out at {len(pairs)} of {n} classes')
    return pairs


def number_pairs(pairs):
    """the 2n photos in draw order, shuffled and numbered by nearmiss2.number_pool, each with its pair id"""
    ids = [i for p in pairs for i in p['ids']]
    by = [p['group'] for p in pairs for _ in p['ids']]
    pid = {i: p['pair'] for p in pairs for i in p['ids']}
    numbered = nearmiss2.number_pool(ids, by)
    for q in numbered:
        q['pair'] = pid[q['index']]
    return numbered


def pair_numbers(pool):
    """[(number, number)] of each pair, in pair order"""
    by = {}
    for p in pool:
        by.setdefault(p['pair'], []).append(p['number'])
    if any(len(v) != 2 for v in by.values()):
        raise ValueError('a pair does not have 2 photos')
    return [tuple(sorted(by[k])) for k in sorted(by)]


# ---------- scores ----------

def within_pair(mean_pred, ratio, pairs, min_diff=PAIR_DIFF):
    """over the pairs whose two ratios differ by at least min_diff, the share where the mean prediction is higher for
    the photo with the higher ratio, ties 1/2. mean_pred and ratio are by photo number (number k at k - 1).
    -> (share or None, number of such pairs)"""
    s, n = 0.0, 0
    for a, b in pairs:
        ra, rb = ratio[a - 1], ratio[b - 1]
        if abs(ra - rb) < min_diff:
            continue
        n += 1
        if mean_pred is not None:
            hi, lo = (a, b) if ra > rb else (b, a)
            ph, pl = mean_pred[hi - 1], mean_pred[lo - 1]
            s += 1.0 if ph > pl else 0.5 if ph == pl else 0.0
    return (s / n if n and mean_pred is not None else None), n


def sim_calls(path, expl, sheet_b64):
    """as tools/simulate.py: SAMPLES fresh samples per reading, up to WORKERS in parallel, cached in path"""
    cache = json.loads(path.read_text()) if path.exists() else {}
    for rid, recs in cache.items():
        if rid in expl and any(r.get('text') != expl[rid]['text'] for r in recs.values()):
            raise SystemExit(f'{path}: cached samples of {rid} were made from a different text; stopping')
    jobs = [(rid, s) for rid in expl for s in range(simulate.SAMPLES) if str(s) not in cache.get(rid, {})]
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(simulate.run_sample, expl[rid]['text'], sheet_b64): (rid, s) for rid, s in jobs}
        for f in as_completed(futs):
            rid, s = futs[f]
            try:
                rec = f.result()
            except Exception as err:  # not cached, so a rerun tries again
                print(f'{rid} sample {s} failed: {err}', flush=True)
                continue
            cache.setdefault(rid, {})[str(s)] = {**rec, 'text': expl[rid]['text']}
            nearmiss2.save_json(cache, path)
    return cache


def reading_rows(texts, classes, lists, sim, pool):
    """per reading: simulate.summarize of its samples and the within-pair score; ids ranked by mean rho; the top id"""
    rv, pn = simulate.pool_ratios(pool), pair_numbers(pool)
    rows = {}
    for rid in READINGS:
        lv = lists.get(rid) if texts[rid] is not None else None
        row = {'explanation': texts[rid], 'class': classes[rid], 'lists_valid': None if lv is None else lv['valid'],
               'fire_classes': None if lv is None else lv['fire'],
               'near_miss_classes': None if lv is None else lv['near_miss']}
        if texts[rid] is not None:
            s = simulate.summarize(sim.get(rid, {}), rv)
            wp, n = within_pair(s['mean_prediction'], rv, pn)
            row.update(s, within_pair=wp, within_pair_n=n)
        rows[rid] = row
    ranked = sorted((rid for rid in READINGS if rows[rid].get('n_valid')), key=lambda rid: -rows[rid]['rho_mean'])
    return rows, ranked, (ranked[0] if ranked else None)


def section(S):
    fm = simulate.fmt
    rd = S['readings']
    L = [f'== {S["unit"]}  M={S["M"]:.4g}  map {S["map_size"]}x{S["map_size"]}',
         'bare classes: ' + '  '.join(f'{k}={rd[k]["class"]}' for k in ('bare_s1', 'bare_s2'))]
    for rid in READINGS:
        r = rd[rid]
        if r['explanation'] is None:
            L.append(f'{rid}  no reading')
            continue
        L.append(f'{rid}  rho={fm(r["rho_mean"])} ({fm(r["rho_min"])} to {fm(r["rho_max"])})  '
                 f'rho_mean_pred={fm(r["rho_mean_prediction"])}  consistency={fm(r["consistency"])}  '
                 f'top12={fm(r["top12_mean"])}  within_pair={fm(r["within_pair"])} (n={r["within_pair_n"]})  '
                 + ' '.join(r['explanation'].split())[:140])
    L.append('ranked by mean rho: ' + (', '.join(f'{k} ({rd[k]["rho_mean"]:.3f})' for k in S['ranked_by_mean_rho'])
                                       or 'none'))
    L.append(f'top: {S["top"] or "none"}')
    return '\n'.join(L)


# ---------- one unit, after every unit's readings ----------

def score_unit(e, d, card, texts, classes, failed, H, labels, cats):
    name = e['name']
    udir = OUT / name
    if failed:
        return f'== {name}  readings not available for {failed} (calls failed, not cached); rerun'
    expl = {rid: {'text': t} for rid, t in texts.items() if t is not None}  # READINGS order
    if not expl:
        return f'== {name}  no readings'

    # 1. class lists, from the reading texts only, as tools/nearmiss2.py
    print(name, 'class lists', flush=True)
    lists = nearmiss2.class_lists(expl, cats, udir / 'classes.json')
    missing = [rid for rid in expl if rid not in lists]
    if missing:
        return f'== {name}  class lists missing for {missing} (calls failed, not cached); rerun'

    # 2. the pool of same-class pairs; no activation is read here
    groups = nearmiss2.make_groups(expl, lists, cats, H, labels)
    try:
        pairs = draw_pairs([(g['id'], g['labels']) for g in groups], held_by_label(H, labels),
                           zlib.crc32(name.encode()))
    except ValueError as err:
        raise ValueError(f'{name}: {err}') from err
    numbered = number_pairs(pairs)

    # 3. the unit's held values, read only now
    M = float(np.max(d['vC']))
    ratio = np.full(len(labels), np.nan)
    ratio[d['H']] = d['vH'].astype(np.float64) / M
    pool = nearmiss2.pool_records(numbered, groups, labels, cats, ratio)
    for p, q in zip(pool, numbered):
        p['pair'] = q['pair']
    pf, sf = udir / 'pool.json', udir / 'sim.json'
    if pf.exists() and sf.exists():
        old = [(p['number'], p['index']) for p in json.loads(pf.read_text())]
        if old != [(p['number'], p['index']) for p in pool]:
            raise SystemExit(f'{pf}: the pool drawn now differs from the one {sf.name} was made on; stopping')
    nearmiss2.save_json(pool, pf)
    sheet = udir / 'sheet.jpg'
    nearmiss.make_sheet(pool, str(sheet))
    b64 = base64.b64encode(sheet.read_bytes()).decode()

    # 4. predicted activations, as tools/simulate.py
    print(name, 'simulation', flush=True)
    sim = sim_calls(sf, expl, b64)

    # 5. scores
    rows, ranked, top = reading_rows(texts, classes, lists, sim, pool)
    gs = [{'id': g['id'], 'classes': g['classes'], 'labels': g['labels'], 'n_held': int(len(g['photos'])),
           'pairs_drawn': sum(p['group'] == g['id'] for p in pairs)} for g in groups]
    S = {'unit': name, 'wrong_from': e['wrong_from'], 'M': M, 'map_size': int(d['size']), 'card': card,
         'n_photos': simulate.N, 'samples_per_reading': simulate.SAMPLES, 'pair_min_diff': PAIR_DIFF,
         'chance': simulate.CHANCE, 'groups': gs, 'pairs': pairs, 'pool': pool, 'readings': rows,
         'ranked_by_mean_rho': ranked, 'top': top}
    nearmiss2.save_json(S, udir / 'scores.json')
    return section(S)


# ---------- selftest ----------

def selftest():
    # units
    assert parse_unit('mixed4c:447') == {'name': 'mixed4c:447', 'layer': 'mixed4c', 'chan': 447}
    assert parse_unit('mixed3a:0')['layer'] == 'mixed3a' and parse_unit('mixed5b:1023')['chan'] == 1023
    for bad in ['mixed4c:512', 'mixed5b:1024', 'mixed4c:-1', 'mixed4c:07', 'mixed4c', 'mixed4c:x', 'mixed6a:1',
                'sum:mixed4c:1+2', 'mixed4c:1+2', 'mixed4c: 3', 'mixed4c:']:
        try:
            parse_unit(bad)
        except ValueError:
            continue
        raise AssertionError('accepted ' + bad)
    assert all(NCH[k] == v for k, v in bundlemax.NCH.items())
    ok = [{'unit': 'mixed4c:447', 'wrong_from': 'mixed5a:3'}, {'unit': 'mixed5a:3', 'wrong_from': 'mixed4c:447'}]
    assert [(x['name'], x['wrong_from']) for x in check_units(ok)] == [('mixed4c:447', 'mixed5a:3'),
                                                                        ('mixed5a:3', 'mixed4c:447')]
    for bad in [[{'unit': 'mixed4c:447', 'wrong_from': 'mixed4c:447'}], [ok[0]], ok + [ok[0]], [], {'a': 1},
                [{'unit': 'mixed4c:447'}]]:
        try:
            check_units(bad)
        except ValueError:
            continue
        raise AssertionError(f'accepted units {bad}')

    # spatial max, row and col
    m = np.zeros((2, 3, 3), np.float32)
    m[0, 1, 2] = m[0, 2, 0] = 5
    m[1] = -2
    m[1, 2, 1] = -1
    v, r, c, n = spatial_max(m)
    assert v.tolist() == [5, -1] and r.tolist() == [1, 2] and c.tolist() == [2, 1] and n == 3

    # box geometry: 14x14 (16 px cells), 7x7 (32 px), 28x28 (8 px), clipped at the edges
    assert box_px(5, 7, 14) == (96, 64, 143, 111)
    assert box_px(0, 0, 14) == (0, 0, 31, 31) and box_px(13, 13, 14) == (192, 192, 223, 223)
    assert box_px(0, 13, 14) == (192, 0, 223, 31) and box_px(13, 6, 14) == (80, 192, 127, 223)
    assert box_px(3, 3, 7) == (64, 64, 159, 159) and box_px(0, 6, 7) == (160, 0, 223, 63)
    assert box_px(6, 0, 7) == (0, 160, 63, 223) and box_px(10, 20, 28) == (152, 72, 175, 95)
    for size in (7, 14, 28):
        cell = TILE // size
        for row in range(size):
            for col in range(size):
                x0, y0, x1, y1 = box_px(row, col, size)
                assert 0 <= x0 <= col * cell and (col + 1) * cell - 1 <= x1 <= TILE - 1, (size, row, col)
                assert 0 <= y0 <= row * cell and (row + 1) * cell - 1 <= y1 <= TILE - 1, (size, row, col)
                assert (x1 + 1 - x0) // cell == min(col + 1, size - 1) - max(col - 1, 0) + 1
                assert (y1 + 1 - y0) // cell == min(row + 1, size - 1) - max(row - 1, 0) + 1
    for box in [(96, 64, 143, 111), (192, 192, 223, 223)]:
        img = Image.new('RGB', (TILE, TILE), 'white')
        draw_box(img, box)
        a = np.asarray(img)
        red = (a[..., 0] == 255) & (a[..., 1] == 0) & (a[..., 2] == 0)
        x0, y0, x1, y1 = box
        ys, xs = np.nonzero(red)
        assert (xs.min(), ys.min(), xs.max(), ys.max()) == box, (box, xs.min(), ys.min(), xs.max(), ys.max())
        w, h = x1 - x0 + 1, y1 - y0 + 1
        assert red.sum() == w * h - (w - 4) * (h - 4), red.sum()  # 2 px wide, inside the box
        assert not red[y0 + 2:y1 - 1, x0 + 2:x1 - 1].any() and red[y0:y0 + 2, x0:x1 + 1].all()

    # card and random-row draw
    C = np.arange(5, 3005, 3)
    vC = np.random.default_rng(3).random(len(C))
    top = top_positions(vC)
    assert top.tolist() == np.argsort(-vC)[:20].tolist() and vC[top[0]] == vC.max()
    pos = random_positions(len(C), top, 'mixed4c:447')
    assert np.array_equal(pos, random_positions(len(C), top, 'mixed4c:447')), 'random row not reproducible'
    assert len(pos) == 10 and len(set(pos.tolist())) == 10 and not set(pos.tolist()) & set(top.tolist())
    tops = set(top.tolist())
    rest = np.array([i for k, i in enumerate(C) if k not in tops])
    assert np.array_equal(C[pos], np.random.default_rng(zlib.crc32(b'mixed4c:447')).choice(rest, 10, replace=False))
    assert not np.array_equal(pos, random_positions(len(C), top, 'mixed4c:448'))
    C30, v30 = np.arange(30) * 7, np.random.default_rng(4).random(30)
    t30 = top_positions(v30)
    p30 = random_positions(30, t30, 'mixed4e:55')  # 10 photos outside the top 20: all of them, none of the top
    assert sorted(C30[p30].tolist()) == sorted(set(C30.tolist()) - set(C30[t30].tolist()))
    nm = bundlemax.card_name('mixed4c:447', 'whole')
    cc = card_content(nm, {'type': 'image', 'k': 1}, {'type': 'image', 'k': 2})
    assert [b['type'] for b in cc] == ['text', 'image', 'image', 'text'] and cc[1]['k'] == 1 and cc[2]['k'] == 2
    assert cc[3]['text'] == run_subjects.TASK
    assert cc[0]['text'] == (
        f'# Unit {nm}\n\nRecorded from one unit (a single neuron) in an image classification network, run over several '
        'thousand natural photos. The first attached grid shows the 20 photos on which the unit is most active, '
        'strongest first; a red box marks where in each photo it is most active. Under each is its activation, scaled '
        '0-10 relative to the strongest. The second grid shows 10 photos drawn at random from the same collection, '
        'with their activation on the same scale.'), cc[0]['text']

    # reading calls: one retry on a parse failure, then missing; an is_error reply raises
    calls = []

    def fake(outs):
        def call(sp, content):
            calls.append((sp, content))
            return outs.pop(0)
        return call
    good = 'Sure.\n```json\n{"class": "one", "explanation": "dog faces", "evidence": "x"}\n```'
    g = get_reading('SP', ['c'], fake([{'text': 'no json', 'is_error': False}, {'text': good, 'is_error': False}]))
    assert g['valid'] and g['explanation'] == 'dog faces' and g['class'] == 'one' and g['attempts'] == 2
    assert g['texts'] == ['no json', good] and g['text'] == good and calls == [('SP', ['c'])] * 2
    calls.clear()
    g = get_reading('', ['c'], fake([{'text': good, 'is_error': False}]))
    assert g['valid'] and g['attempts'] == 1 and len(calls) == 1
    g = get_reading('', ['c'], fake([{'text': '{"class": "one", "explanation": ""}', 'is_error': False},
                                     {'text': 'nope', 'is_error': False}, {'text': good, 'is_error': False}]))
    assert not g['valid'] and g['explanation'] is None and g['class'] is None and g['attempts'] == 2
    for outs in ([{'text': 'overloaded', 'is_error': True}],
                 [{'text': 'no json', 'is_error': False}, {'text': 'overloaded', 'is_error': True}]):
        try:
            get_reading('', ['c'], fake(outs))
        except RuntimeError:
            continue
        raise AssertionError('is_error reply accepted')

    # reading order and the wrong reading
    assert READINGS == ['bare_s1', 'bare_s2', 'Q0_s1', 'Q0_s2', 'F1_s1', 'F1_s2', 'wrong']

    def rec(t, cl='one'):
        return {'explanation': t, 'class': None if t is None else cl}
    recs = {'A': {'bare_s1': rec('a1'), 'bare_s2': rec(None), 'Q0_s1': rec('aq1'), 'Q0_s2': rec('aq2'),
                  'F1_s1': rec('af1'), 'F1_s2': rec('af2')},
            'B': {'bare_s1': rec('b1', 'unrelated'), 'Q0_s1': rec('bq1')},
            'C': {'bare_s1': rec(None)}}
    t, cl, f = reading_texts('A', 'B', recs)
    assert list(t) == READINGS and list(cl) == READINGS and f == []
    assert t == {'bare_s1': 'a1', 'bare_s2': None, 'Q0_s1': 'aq1', 'Q0_s2': 'aq2', 'F1_s1': 'af1', 'F1_s2': 'af2',
                 'wrong': 'b1'} and cl['wrong'] == 'unrelated' and cl['bare_s1'] == 'one' and cl['bare_s2'] is None
    t, cl, f = reading_texts('B', 'A', recs)
    assert t['wrong'] == 'a1' and t['bare_s1'] == 'b1' and t['Q0_s1'] == 'bq1'
    assert f == ['bare_s2', 'Q0_s2', 'F1_s1', 'F1_s2']
    t, cl, f = reading_texts('A', 'C', recs)
    assert t['wrong'] is None and f == []  # C's bare_s1 was asked and is missing
    t, cl, f = reading_texts('A', 'D', recs)
    assert t['wrong'] is None and f == ['wrong']  # D's bare_s1 call failed: not available

    # paired pool
    labels_all = np.repeat(np.arange(40), 10)  # label k: photos 10k..10k+9
    Hs = np.array([i for i in range(400) if not 51 <= i <= 69])  # label 5: one held photo; label 6: none
    byl = held_by_label(Hs, labels_all)
    assert byl[5].tolist() == [50] and 6 not in byl and byl[7].tolist() == list(range(70, 80))
    groups = [('A:fire', [0, 1, 2, 5, 6]), ('A:near_miss', list(range(7, 21))), ('B:fire', [1, 2] + list(range(21, 30))),
              ('B:near_miss', [3, 4])]
    seed = zlib.crc32(b'mixed4c:447')
    pairs = draw_pairs(groups, byl, seed)
    assert pairs == draw_pairs(groups, byl, seed), 'pool not reproducible'
    assert len(pairs) == 24 and [p['pair'] for p in pairs] == list(range(1, 25))
    labs = [p['label'] for p in pairs]
    assert len(set(labs)) == 24 and not {5, 6} & set(labs)
    hs = set(Hs.tolist())
    for p in pairs:
        assert len(p['ids']) == 2 and len(set(p['ids'])) == 2 and p['label'] in dict(groups)[p['group']]
        assert all(i in hs and labels_all[i] == p['label'] for i in p['ids'])
    assert [p['group'] for p in pairs[:4]] == ['A:fire', 'A:near_miss', 'B:fire', 'B:near_miss']
    assert sum(p['group'] == 'A:fire' for p in pairs) <= 3 and sum(p['group'] == 'B:near_miss' for p in pairs) <= 2
    rng = np.random.default_rng(seed)
    lab0 = [0, 1, 2][int(rng.integers(3))]
    assert pairs[0]['label'] == lab0 and pairs[0]['ids'] == [int(x) for x in rng.choice(byl[lab0], 2, replace=False)]
    assert draw_pairs(groups, byl, seed + 1) != pairs
    small = {0: np.array([1, 2]), 1: np.array([10, 11, 12]), 2: np.array([20, 21]), 3: np.array([30, 31]),
             5: np.array([50])}
    sg = [('x', [0]), ('y', [1, 2, 3]), ('z', [5, 9])]  # x exhausted after 1; z never has a class with 2 photos
    p = draw_pairs(sg, small, 7, n=4)
    assert [q['group'] for q in p] == ['x', 'y', 'y', 'y'] and sorted(q['label'] for q in p) == [0, 1, 2, 3]
    assert sorted(p[0]['ids']) == [1, 2]
    p = draw_pairs([('x', [0, 1]), ('y', [1, 0])], small, 3, n=2)
    assert sorted(q['label'] for q in p) == [0, 1] and [q['group'] for q in p] == ['x', 'y']
    for gs_, n_ in [(sg, 5), ([('x', [0, 1]), ('y', [1, 0])], 3), ([('z', [5, 9])], 1), ([], 1)]:
        try:
            draw_pairs(gs_, small, 7, n=n_)
        except ValueError:
            continue
        raise AssertionError(f'short pool accepted {gs_} {n_}')
    numbered = number_pairs(pairs)
    ids = [i for q in pairs for i in q['ids']]
    assert [q['number'] for q in numbered] == list(range(1, 49))
    assert [q['index'] for q in numbered] == np.random.default_rng(1).permutation(np.array(ids)).tolist()
    pool = [{'number': q['number'], 'index': q['index'], 'pair': q['pair'], 'ratio': 0.0} for q in numbered]
    pn = pair_numbers(pool)
    assert len(pn) == 24 and sorted(x for ab in pn for x in ab) == list(range(1, 49))
    for k, (a, b) in enumerate(pn, 1):
        assert labels_all[numbered[a - 1]['index']] == labels_all[numbered[b - 1]['index']] == pairs[k - 1]['label']

    # within-pair score on hand cases, with ties
    ratio = [0.9, 0.2, 0.1, 0.6, 0.5, 0.45, 0.8, 0.0]
    pred = [5, 3, 2, 2, 9, 1, 1, 4]
    pr = [(1, 2), (3, 4), (5, 6), (7, 8)]
    assert within_pair(pred, ratio, pr) == (0.5, 3)  # 1 + 1/2 (tie) + 0 over 3; (5, 6) differ by 0.05
    assert within_pair(pred, ratio, [(2, 1), (4, 3), (6, 5), (8, 7)]) == (0.5, 3)
    assert within_pair([1] * 8, ratio, pr) == (0.5, 3)
    assert within_pair(pred, ratio, [(1, 2)]) == (1.0, 1) and within_pair(pred, ratio, [(7, 8)]) == (0.0, 1)
    assert within_pair(pred, ratio, [(3, 4)]) == (0.5, 1) and within_pair(pred, ratio, [(5, 6)]) == (None, 0)
    assert within_pair(None, ratio, pr) == (None, 3)
    r48 = [0.9, 0.1] + [0.0] * 46
    s = simulate.summarize({'0': {'valid': True, 'predictions': [0.1, 0.3] + [0.0] * 46},
                            '1': {'valid': True, 'predictions': [0.2, 0.2] + [0.0] * 46},
                            '2': {'valid': True, 'predictions': [0.3, 0.1] + [0.0] * 46}}, r48)
    assert within_pair(s['mean_prediction'], r48, [(1, 2)]) == (0.5, 1)  # equal means tie exactly

    # rows, ranking, top, report section on a synthetic pool
    rp = []
    for j in range(24):  # pair j: numbers 2j+1, 2j+2; every third pair differs by only 0.02
        rp += [0.6 * j / 23, 0.6 * j / 23 + (0.02 if j % 3 == 0 else 0.35)]
    pool = [{'number': k, 'ratio': rp[k - 1], 'pair': (k + 1) // 2} for k in range(1, 49)]
    n_q = sum(abs(rp[2 * j] - rp[2 * j + 1]) >= PAIR_DIFF for j in range(24))
    assert n_q == 16, n_q
    up = {str(s): {'valid': True, 'predictions': [10 * x for x in rp]} for s in range(3)}
    down = {str(s): {'valid': True, 'predictions': [10 * (1 - x) for x in rp]} for s in range(3)}
    flat = {'0': {'valid': True, 'predictions': [5.0] * 48}, '1': {'valid': False}}
    texts = {rid: None for rid in READINGS}
    texts.update(bare_s1='reads\n  up', Q0_s1='down ' * 60, wrong='flat')
    classes = {rid: ('one' if texts[rid] else None) for rid in READINGS}
    rows, ranked, best = reading_rows(texts, classes, {}, {'bare_s1': up, 'Q0_s1': down, 'wrong': flat}, pool)
    assert ranked == ['bare_s1', 'wrong', 'Q0_s1'] and best == 'bare_s1', ranked
    assert np.isclose(rows['bare_s1']['rho_mean'], 1) and np.isclose(rows['Q0_s1']['rho_mean'], -1)
    assert rows['bare_s1']['within_pair'] == 1.0 and rows['Q0_s1']['within_pair'] == 0.0
    assert rows['wrong']['within_pair'] == 0.5 and rows['wrong']['n_valid'] == 1
    assert all(rows[k]['within_pair_n'] == n_q for k in ('bare_s1', 'Q0_s1', 'wrong'))
    assert 'rho_mean' not in rows['bare_s2'] and rows['bare_s2']['explanation'] is None
    S = {'unit': 'mixed4c:447', 'M': 2.5, 'map_size': 14, 'readings': rows, 'ranked_by_mean_rho': ranked, 'top': best}
    L = section(S).splitlines()
    assert L[0] == '== mixed4c:447  M=2.5  map 14x14' and L[1] == 'bare classes: bare_s1=one  bare_s2=None'
    assert [x.split()[0] for x in L[2:9]] == READINGS and L[3] == 'bare_s2  no reading'
    assert L[2].endswith(f'within_pair=1.000 (n={n_q})  reads up') and L[4].endswith(('down ' * 60)[:140])
    assert L[-2] == 'ranked by mean rho: bare_s1 (1.000), wrong (0.000), Q0_s1 (-1.000)', L[-2]
    assert L[-1] == 'top: bare_s1'
    json.dumps(rows)
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
    entries = check_units(json.loads((uf if uf.is_absolute() else J / uf).read_text()))
    arms = [(arm, '' if f is None else (J / f).read_text().strip()) for arm, f in ARMS]
    names = json.loads(nearmiss2.NAMES.read_text())
    C, H = bundlemax.pools(names)
    with np.load(nearmiss2.ACTS) as z:
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
            m = bundlemax._bank().load_model(WEIGHTS)
            for p in m.parameters():
                p.requires_grad_(False)
            held_model.append(m)
        return held_model[0]

    OUT.mkdir(parents=True, exist_ok=True)

    # 1. every unit, one at a time: values, card, readings
    vals, cards, recs = {}, {}, {}
    for e in entries:
        print('==', e['name'], flush=True)
        udir = OUT / e['name']
        udir.mkdir(parents=True, exist_ok=True)
        vals[e['name']] = unit_values(e, C, H, udir, get_model)
        cards[e['name']], content = make_cards(e, vals[e['name']], udir)
        recs[e['name']] = readings(e, content, arms, udir)
        del content
        gc.collect()
    held_model.clear()
    gc.collect()

    # 2. then the wrong readings
    got = {e['name']: reading_texts(e['name'], e['wrong_from'], recs) for e in entries}
    nearmiss2.save_json({n: g[0] for n, g in got.items()}, OUT / 'readings_all.json')

    # 3. class lists, pool, simulation and scores, one unit at a time
    secs = []
    for e in entries:
        n = e['name']
        print('==', n, 'scores', flush=True)
        secs.append(score_unit(e, vals[n], cards[n], *got[n], H, labels, cats))
        (OUT / 'report.txt').write_text('\n\n'.join(secs) + '\n')
        gc.collect()
        print(secs[-1], flush=True)
    print('report ->', OUT / 'report.txt')


if __name__ == '__main__':
    main()
