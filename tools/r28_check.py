"""Round 28 prediction checks (PREDICTIONS.md, round 28), on runs/refineloop/<unit>/stats.json. Written before the
results exist.

usage: python -I tools/r28_check.py
"""
import json
import re
from pathlib import Path

J = Path(__file__).resolve().parent.parent
OUT = J / 'runs/refineloop'
PREF = {'mixed4b:431': 'right', 'mixed4b:443': 'right', 'mixed4b:429': 'left', 'mixed4b:384': 'left'}
CONTROLS = ['mixed4b:409', 'mixed4b:418']
OTHER = {'left': 'right', 'right': 'left'}
AMBIG = re.compile(r"(\b(its|his|her|their|own)\s+|'s\s+)(left|right)\b", re.I)


def sides(text):
    return {s for s in ('left', 'right') if re.search(rf'\b{s}\b', text, re.I)}


def is_direction(p):
    a = p['answers']
    return bool(sides(p['question']) | sides(a[p['most']]) | sides(a[p['least']]))


def favoured(p):
    a = p['answers']
    hi, lo = a[p['most']], a[p['least']]
    if AMBIG.search(hi) or AMBIG.search(lo):
        return None
    if len(sides(hi)) == 1:
        return next(iter(sides(hi)))
    if len(sides(lo)) == 1:
        return OTHER[next(iter(sides(lo)))]
    return None


def refs(u):
    path = OUT / u / 'stats.json'
    return json.loads(path.read_text())['refinements'] if path.exists() else []


def main():
    units = list(PREF) + CONTROLS
    Rf = {u: refs(u) for u in units}
    for u in units:
        st = json.loads((OUT / u / 'stats.json').read_text()) if (OUT / u / 'stats.json').exists() else None
        if st:
            print(f'{u} triggers: ' + '; '.join(f'{a}: j={t.get("j")} {t.get("kind")} peer={t.get("peer")}'
                                                for a, t in st['triggers'].items() if t))
        for p in Rf[u]:
            r = p['real']
            print(f'  {p["call"]} dir={is_direction(p)} fav={favoured(p) if is_direction(p) else "-"} '
                  f'{r["outcome"]} n {r["n_hi"]}/{r["n_lo"]} pct {r["pct"]} distinct={r["distinctive"]} '
                  f'null={p["null"]["outcome"]} | {p["question"][:70]} | most: {p["answers"][p["most"]][:40]} | '
                  f'least: {p["answers"][p["least"]][:40]}')

    def ok(p, u, side):
        return is_direction(p) and favoured(p) == side and p['real']['outcome'] == 'held'

    wrote = [u for u in PREF if any(p['arm'] == 'F' and is_direction(p) for p in Rf[u])]
    print('1. F writes a direction refinement:', wrote, '->', len(wrote) >= 3)
    good = {a: [u for u in PREF if any(p['arm'] == a and ok(p, u, PREF[u]) for p in Rf[u])] for a in 'FR'}
    bad = [u for u in PREF if any(p['arm'] == 'F' and ok(p, u, OTHER[PREF[u]]) for p in Rf[u])]
    print('2. F correct side held:', good['F'], '; wrong side held:', bad, '->', len(good['F']) >= 2 and not bad)
    print('3. R correct side held:', good['R'], '->', len(good['F']) > len(good['R']))
    ctrl = {u: [p['call'] for p in Rf[u] if is_direction(p) and p['real']['outcome'] == 'held'] for u in CONTROLS}
    print('4. controls, held direction refinements:', ctrl, '->',
          not ctrl['mixed4b:409'] and sum(map(len, ctrl.values())) <= 1)
    cnt = {t: [0, 0] for t in ('real', 'null')}
    for u in units:
        for p in Rf[u]:
            for t in cnt:
                if p[t]['outcome'] != 'too few':
                    cnt[t][1] += 1
                    cnt[t][0] += p[t]['outcome'] == 'held'
    rr, nr = (cnt[t][0] / cnt[t][1] if cnt[t][1] else float('nan') for t in ('real', 'null'))
    print(f'5. held real {cnt["real"][0]}/{cnt["real"][1]} ({rr:.2f}), null {cnt["null"][0]}/{cnt["null"][1]} '
          f'({nr:.2f}) ->', nr <= 0.10 and rr >= 3 * nr)
    share = {}
    for a in 'FR':
        held = [p for u in units for p in Rf[u] if p['arm'] == a and p['real']['outcome'] == 'held']
        share[a] = (sum(p['real']['distinctive'] for p in held), len(held))
    f, r = (share[a][0] / share[a][1] if share[a][1] else float('nan') for a in 'FR')
    print(f'6. distinctive among held: F {share["F"][0]}/{share["F"][1]}, R {share["R"][0]}/{share["R"][1]} ->', f > r)


if __name__ == '__main__':
    main()
