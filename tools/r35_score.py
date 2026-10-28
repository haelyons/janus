"""Round 35, Inception: Codex judges score the labels on held-out photos (PREDICTIONS.md, round 35).

Unit score: the judge sees one unit label and the unit's 8 scoring photos (4 strongest held-out, 4 near misses; numbered
tiles in seeded order) and picks the 4 the label describes; score = share of the 8 classified correctly. A label
"can't tell" scores 0.5 with no call. Group score: the judge sees the shared line and two sets of 8 held-out photos
(each unit's top 2), this group's and its pair's (same layer, next category in the cycle explained, cluster, random
neurons, random directions), and picks one; 1 or 0. "nothing in common" scores 0.5 with no call (the same rule as
"can't tell"); the named-failed group has no pair. Resumable: judgements are cached per call.

usage: python -I tools/r35_score.py agent|human [--workers 4]
"""
import argparse
import json
import re
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
from r35_pages import LETTERS, font, labels, load, photo  # noqa: E402

OUT = J / 'runs/r35/img'
CYCLE = ('explained', 'cluster', 'random_neurons', 'random_directions')
T = 200


def rng(tag):
    return np.random.default_rng(zlib.crc32(tag.encode()))


def blank(label):
    s = (label or '').strip().lower().replace('’', "'")
    return not s or s.startswith("can't") or s.startswith('cant') or s.startswith('nothing in common')


def tiles(ids, tag):
    """a 4 x 2 grid of photos numbered 1-8"""
    im = Image.new('RGB', (4 * T, 2 * T), 'white')
    d = ImageDraw.Draw(im)
    for k, i in enumerate(ids):
        x, y = (k % 4) * T, (k // 4) * T
        im.paste(photo(i, T - 6), (x + 3, y + 3))
        d.rectangle([x + 3, y + 3, x + 33, y + 33], fill='white')
        d.text((x + 9, y + 5), str(k + 1), font=font(22, True), fill=(0, 0, 0))
    return im


def codex(prompt, img, cache):
    if cache.exists():
        return json.loads(cache.read_text())
    out = cache.with_suffix('.txt')
    subprocess.run(['codex', 'exec', '--skip-git-repo-check', '--ephemeral', '-s', 'read-only', '-C', str(cache.parent),
                    '-o', str(out), '-i', str(img), prompt], stdin=subprocess.DEVNULL, capture_output=True,
                   timeout=900, check=True)
    text = out.read_text()
    m = re.findall(r'\{[^{}]*\}', text)
    assert m, text[:300]
    obj = json.loads(m[-1])
    res = {'reply': text, 'parsed': obj}
    cache.write_text(json.dumps(res))
    return res


def unit_job(who, u, label, d):
    order = [int(j) for j in rng(f'r35:judge:{u["name"]}').permutation(8)]
    ids = [(u['pos'] + u['near'])[j] for j in order]
    img = d / f'unit_{u["name"].replace(":", "_")}.png'
    tiles(ids, u['name']).save(img)
    prompt = (f'A unit of an image-classification network was described as: "{label}". The image shows 8 photos '
              'numbered 1-8. Exactly 4 of them make the unit strongly active and 4 do not. Which 4 does the '
              'description fit best? Reply with only a JSON object: {"picks": [four numbers]}.')
    r = codex(prompt, img, d / f'unit_{u["name"].replace(":", "_")}.json')['parsed']
    picks = {int(x) - 1 for x in r['picks']}
    assert len(picks) == 4 and picks <= set(range(8)), r
    hits = sum(order[k] < 4 for k in picks)
    return u['name'], hits / 4


def group_job(who, g, pair, shared, d):
    a_first = bool(rng(f'r35:judge:{g["code"]}').integers(2))
    sets = [[i for u in x['units'] for i in u['pos'][:2]] for x in ((g, pair) if a_first else (pair, g))]
    im = Image.new('RGB', (4 * T, 4 * T + 60), 'white')
    dr = ImageDraw.Draw(im)
    for k, ids in enumerate(sets):
        dr.text((8, k * (2 * T + 30) + 2), 'AB'[k], font=font(22, True), fill=(0, 0, 0))
        im.paste(tiles(ids, ''), (0, k * (2 * T + 30) + 30))
    img = d / f'group_{g["code"]}.png'
    im.save(img)
    prompt = (f'Four units of an image-classification network were described together as: "{shared}". The image '
              'shows two sets of photos, A (top) and B (bottom); each set holds the 2 strongest photos of each of '
              'four units. Which set does the description fit better? Reply with only a JSON object: {"pick": "A"} '
              'or {"pick": "B"}.')
    r = codex(prompt, img, d / f'group_{g["code"]}.json')['parsed']
    pick = str(r['pick']).strip().upper()
    assert pick in ('A', 'B'), r
    return g['code'], float((pick == 'A') == a_first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('who', choices=('agent', 'human'))
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    groups, lab = load(), labels(a.who)
    d = OUT / 'judge' / a.who
    d.mkdir(parents=True, exist_ok=True)
    by = {(g['category'], g['units'][0]['layer']): g for g in groups}
    units, grp = {}, {}
    jobs = []
    with ThreadPoolExecutor(a.workers) as ex:
        for g in groups:
            L = lab.get(g['code'], {})
            for r, name in enumerate(g['rows']):
                u = next(x for x in g['units'] if x['name'] == name)
                if blank(L.get(LETTERS[r])):
                    units[name] = 0.5
                else:
                    jobs.append(ex.submit(unit_job, a.who, u, L[LETTERS[r]], d))
            if g['category'] == 'named_failed':
                continue
            if blank(L.get('shared')):
                grp[g['code']] = 0.5
                continue
            nxt = CYCLE[(CYCLE.index(g['category']) + 1) % len(CYCLE)]
            jobs.append(ex.submit(group_job, a.who, g, by[(nxt, g['units'][0]['layer'])], L['shared'], d))
        for j in jobs:
            k, v = j.result()
            (grp if k.startswith('I') else units)[k] = v
    p = OUT / 'scores.json'
    sc = json.loads(p.read_text()) if p.exists() else {}
    sc[a.who], sc[f'{a.who}_group'] = units, grp
    p.write_text(json.dumps(sc, indent=1))
    print(a.who, 'units', len(units), 'mean', round(float(np.mean(list(units.values()))), 3), 'groups', grp)


if __name__ == '__main__':
    main()
