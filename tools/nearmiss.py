"""Blind maximum and near misses for image units (spec: authors/spec_nearmiss.md).

An agent that sees only an explanation picks, from a fixed pool of 48 unlabelled photos, 6 the unit should fire on
most and 6 near misses; the unit's recorded activations score the picks.

    python -I tools/nearmiss.py [--selftest]
"""
import base64
import json
import os
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I does not add the script dir
import claude_call  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPL = os.path.join(ROOT, 'authors/explanations/nearmiss.json')
NAMES = os.path.join(ROOT, 'bank/img/names.json')
THUMBS = os.path.join(ROOT, 'bank/img/thumbs')
OUT = os.path.join(ROOT, 'runs/nearmiss')
UNITS = {'U-121': ('bank/img/acts.npz', 'maps', 1527.0),
         'U-945': ('bank/img/maps_mixed4d_409.npy', None, 2288.9)}
SAMPLES, TILE, STRIP = 3, 112, 20

FIRST = ('A unit (one neuron) of an image classification network has been explained as follows:\n\n{text}\n\n'
         'The attached sheet shows 48 numbered photos. You have not seen how the unit responds to any of them.')
LAST = ('Using only the explanation, choose the 6 photos on which the unit should fire most strongly, and 6 near misses: '
        'photos that look related to what the explanation describes or share obvious features with it, but on which, '
        'according to the explanation, the unit should barely fire. Reply with only JSON: {"fire": [six numbers], '
        '"near_miss": [six numbers]}')


def unit_acts(card):
    """spatial max of max(map, 0), divided by card_max"""
    path, key, cmax = UNITS[card]
    m = np.load(os.path.join(ROOT, path))
    if key:
        m = m[key]
    return np.maximum(m.reshape(len(m), -1).max(1).astype(np.float64), 0) / cmax


def held_indices(names):
    return np.array([i for i, n in enumerate(names) if zlib.crc32(n.encode()) % 10 >= 7])


def build_pool(act, held):
    a = act[held]
    strata = [held[a >= 0.3], held[(a >= 0.05) & (a < 0.3)], held[a < 0.05]]
    rng = np.random.default_rng(0)
    picks = np.concatenate([rng.choice(s, 16, replace=False) for s in strata])
    picks = np.random.default_rng(1).permutation(picks)
    return [{'number': k + 1, 'index': int(i), 'act': float(act[i])} for k, i in enumerate(picks)]


def make_sheet(pool, path):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
    sheet = Image.new('RGB', (8 * TILE, 6 * (TILE + STRIP)), 'white')
    draw = ImageDraw.Draw(sheet)
    for p in pool:
        r, c = divmod(p['number'] - 1, 8)
        x, y = c * TILE, r * (TILE + STRIP)
        img = Image.open(os.path.join(THUMBS, f"{p['index']:05d}.jpg")).convert('RGB').resize((TILE, TILE))
        sheet.paste(img, (x, y))
        draw.text((x + 4, y + TILE + 2), str(p['number']), fill='black', font=font)
    sheet.save(path, quality=90)


def parse_reply(text):
    """the last JSON object in the reply that has both 'fire' and 'near_miss'"""
    dec, found = json.JSONDecoder(), None
    for i, ch in enumerate(text):
        if ch == '{':
            try:
                obj, _ = dec.raw_decode(text, i)
            except ValueError:
                continue
            if isinstance(obj, dict) and 'fire' in obj and 'near_miss' in obj:
                found = obj
    return found


def is_valid(obj):
    def ok(v):
        return (isinstance(v, list) and len(v) == 6 and len(set(v)) == 6
                and all(type(x) is int and 1 <= x <= 48 for x in v))
    return obj is not None and ok(obj['fire']) and ok(obj['near_miss']) and not set(obj['fire']) & set(obj['near_miss'])


def run_sample(text, sheet_b64):
    content = [{'type': 'text', 'text': FIRST.format(text=text)},
               {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': sheet_b64}},
               {'type': 'text', 'text': LAST}]
    replies = []
    for _ in range(2):  # one retry, fresh conversation
        reply = claude_call.converse('', [content])[0]['text']
        replies.append(reply)
        obj = parse_reply(reply)
        if is_valid(obj):
            return {'replies': replies, 'valid': True, 'fire': obj['fire'], 'near_miss': obj['near_miss']}
    return {'replies': replies, 'valid': False}


def score(fire, near, act_of):
    fa, na = [act_of[k] for k in fire], [act_of[k] for k in near]
    return {'fire_hit': float(np.mean([a >= 0.3 for a in fa])), 'near_ok': float(np.mean([a < 0.1 for a in na])),
            'gap': float(np.mean(fa) - np.mean(na))}


def chance(act_of):
    a = np.array(list(act_of.values()))
    return {'fire_hit': float(np.mean(a >= 0.3)), 'near_ok': float(np.mean(a < 0.1)), 'gap': 0.0}


def decision(scores):
    """gap against its chance level 0, over valid samples"""
    if not scores:
        return 'NONE'
    gaps = [s['gap'] for s in scores]
    return 'ABOVE' if min(gaps) > 0 else 'BELOW' if max(gaps) < 0 else 'MIXED'


def stat(scores, k):
    v = [s[k] for s in scores]
    return f'{k}={np.mean(v):.3f} ({min(v):.3f} to {max(v):.3f})'


def save_json(obj, path):
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(path + '.tmp', path)


def main():
    os.makedirs(OUT, exist_ok=True)
    expl = json.load(open(EXPL))
    names = json.load(open(NAMES))
    held = held_indices(names)
    pools, sheets = {}, {}
    for card in sorted({e['unit_card'] for e in expl.values()}):
        act = unit_acts(card)
        assert len(act) == len(names), (card, len(act), len(names))
        pool = build_pool(act, held)
        save_json(pool, os.path.join(OUT, f'pool_{card}.json'))
        sheet_path = os.path.join(OUT, f'sheet_{card}.jpg')
        make_sheet(pool, sheet_path)
        pools[card] = {p['number']: p['act'] for p in pool}
        sheets[card] = base64.b64encode(open(sheet_path, 'rb').read()).decode()

    cache_path = os.path.join(OUT, 'replies.json')
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    jobs = [(n, s) for n in expl for s in range(SAMPLES) if str(s) not in cache.get(n, {})]
    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(run_sample, expl[n]['text'], sheets[expl[n]['unit_card']]): (n, s) for n, s in jobs}
        for f in as_completed(futs):
            n, s = futs[f]
            try:
                rec = f.result()
            except Exception as e:  # not cached, so a rerun tries again
                print(f'{n} sample {s} failed: {e}')
                continue
            cache.setdefault(n, {})[str(s)] = rec
            save_json(cache, cache_path)

    lines = []
    for n, e in expl.items():
        card, recs = e['unit_card'], cache.get(n, {})
        sc = [score(r['fire'], r['near_miss'], pools[card]) for r in recs.values() if r['valid']]
        head = f'{n}  unit={card}  source={e["source"]}  valid={len(sc)}/{SAMPLES}'
        body = '  '.join(stat(sc, k) for k in ('fire_hit', 'near_ok', 'gap')) if sc else 'no valid samples'
        lines.append(f'{head}  {body}  gap_vs_chance={decision(sc)}')
    for card in pools:
        c = chance(pools[card])
        lines.append(f'chance {card}  fire_hit={c["fire_hit"]:.3f}  near_ok={c["near_ok"]:.3f}  gap={c["gap"]:.3f}')
    report = '\n'.join(lines) + '\n'
    with open(os.path.join(OUT, 'report.txt'), 'w') as f:
        f.write(report)
    print(report, end='')


def selftest():
    # pool: strata sizes and determinism on synthetic acts
    names = [f'img_{i}.jpg' for i in range(3000)]
    held = held_indices(names)
    act = np.random.default_rng(7).random(3000) ** 2
    pool = build_pool(act, held)
    assert pool == build_pool(act, held)
    assert [p['number'] for p in pool] == list(range(1, 49))
    idx = [p['index'] for p in pool]
    assert len(set(idx)) == 48 and set(idx) <= set(held.tolist())
    a = np.array([p['act'] for p in pool])
    assert (a >= 0.3).sum() == 16 and ((a >= 0.05) & (a < 0.3)).sum() == 16 and (a < 0.05).sum() == 16

    # reply validation
    good = 'Here: {"fire": [1, 2, 3, 4, 5, 6], "near_miss": [7, 8, 9, 10, 11, 12]} and {"note": 1}'
    assert is_valid(parse_reply(good)) and parse_reply(good)['fire'] == [1, 2, 3, 4, 5, 6]
    bad = ['{"fire": [1, 1, 3, 4, 5, 6], "near_miss": [7, 8, 9, 10, 11, 12]}',   # duplicate
           '{"fire": [1, 2, 3, 4, 5, 49], "near_miss": [7, 8, 9, 10, 11, 12]}',  # out of range
           '{"fire": [0, 2, 3, 4, 5, 6], "near_miss": [7, 8, 9, 10, 11, 12]}',   # out of range
           '{"fire": [1, 2, 3, 4, 5, 6], "near_miss": [6, 8, 9, 10, 11, 12]}',   # overlap
           '{"fire": [1, 2, 3, 4, 5], "near_miss": [7, 8, 9, 10, 11, 12]}',      # wrong count
           '{"fire": [1, 2, 3, 4, 5, 6], "near_miss": [7, 8, 9, 10, 11, 12, 13]}',
           '{"fire": [1, 2, 3, 4, 5, 6]}', 'no json']
    assert not any(is_valid(parse_reply(t)) for t in bad)

    # scoring on a synthetic pool
    act_of = {k: 0.5 if k <= 16 else 0.2 if k <= 32 else 0.0 for k in range(1, 49)}
    s = score([1, 2, 3, 4, 5, 6], [33, 34, 35, 36, 37, 38], act_of)
    assert s['fire_hit'] == 1 and s['near_ok'] == 1 and np.isclose(s['gap'], 0.5)
    s = score([1, 2, 3, 17, 18, 19], [20, 21, 22, 33, 34, 35], act_of)
    assert s['fire_hit'] == 0.5 and s['near_ok'] == 0.5 and np.isclose(s['gap'], 0.25)
    s = score([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], {k: 0.3 if k <= 6 else 0.1 for k in range(1, 49)})
    assert s['fire_hit'] == 1 and s['near_ok'] == 0
    c = chance(act_of)
    assert np.isclose(c['fire_hit'], 16 / 48) and np.isclose(c['near_ok'], 16 / 48) and c['gap'] == 0
    assert decision([{'gap': 0.1}, {'gap': 0.2}]) == 'ABOVE' and decision([{'gap': -0.1}]) == 'BELOW'
    assert decision([{'gap': -0.1}, {'gap': 0.1}]) == 'MIXED' and decision([]) == 'NONE'
    print('selftest ok')


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv[1:] else main()
