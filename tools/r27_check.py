"""Round 27 prediction checks (PREDICTIONS.md, round 27), on runs/contrastloop/<unit>/stats.json and judge.json.
Written before the results exist.

usage: python -I tools/r27_check.py
"""
import json
import re
import statistics
from pathlib import Path

J = Path(__file__).resolve().parent.parent
OUT = J / 'runs/contrastloop'
PREF = {'mixed4b:431': 'right', 'mixed4b:443': 'right', 'mixed4b:429': 'left', 'mixed4b:384': 'left'}
CONTROLS = ['mixed4d:409', 'mixed4b:237', 'mixed4b:373', 'mixed4e:55', 'mixed5a:719']
OTHER = {'left': 'right', 'right': 'left'}


def sides(text):
    return {s for s in ('left', 'right') if re.search(rf'\b{s}\b', text, re.I)}


def is_direction(p):
    a = p['answers']
    return bool(sides(p['question']) | sides(a[p['most']]) | sides(a[p['least']]))


def favoured(p):
    a = p['answers']
    hi, lo = sides(a[p['most']]), sides(a[p['least']])
    if len(hi) == 1:
        return next(iter(hi))
    if len(lo) == 1:
        return OTHER[next(iter(lo))]
    return None


def preds(u):
    st = json.loads((OUT / u / 'stats.json').read_text())
    return [(s, p) for s, rec in st['samples'].items() for p in (rec['predictions'] or [])]


def main():
    units = list(PREF) + CONTROLS
    P = {u: preds(u) for u in units if (OUT / u / 'stats.json').exists()}
    for u, ps in P.items():
        for s, p in ps:
            if is_direction(p):
                print(f'{u} {s} dir fav={favoured(p)} real={p["real"]["outcome"]} {p["real"]["pct"]} '
                      f'null={p["null"]["outcome"]}  {p["question"]} | most: {p["answers"][p["most"]]} | '
                      f'least: {p["answers"][p["least"]]}')
    wrote = [u for u in PREF if any(is_direction(p) for _, p in P.get(u, []))]
    side = [u for u in PREF if any(is_direction(p) and favoured(p) == PREF[u] for _, p in P.get(u, []))]
    held = [u for u in PREF if any(is_direction(p) and favoured(p) == PREF[u] and p['real']['outcome'] == 'held'
                                   for _, p in P.get(u, []))]
    wrong = [u for u in PREF if any(is_direction(p) and favoured(p) == OTHER[PREF[u]] and
                                    p['real']['outcome'] == 'held' for _, p in P.get(u, []))]
    ctrl = [u for u in CONTROLS if any(is_direction(p) and p['real']['outcome'] == 'held' for _, p in P.get(u, []))]
    print('1. direction prediction written:', wrote, '->', len(wrote) >= 2)
    print('2. favours preferred side:', side, '->', len(side) >= 2)
    print('3. preferred side held:', held, '; wrong side held:', wrong, '->', len(held) >= 2 and not wrong)
    print('4. controls with a held direction prediction:', ctrl, '->', len(ctrl) <= 1)
    cnt = {t: [0, 0] for t in ('real', 'null')}
    for ps in P.values():
        for _, p in ps:
            for t in cnt:
                if p[t]['outcome'] != 'too few':
                    cnt[t][1] += 1
                    cnt[t][0] += p[t]['outcome'] == 'held'
    rr, nr = (cnt[t][0] / cnt[t][1] if cnt[t][1] else float('nan') for t in ('real', 'null'))
    print(f'5. held real {cnt["real"][0]}/{cnt["real"][1]} ({rr:.2f}), null {cnt["null"][0]}/{cnt["null"][1]} '
          f'({nr:.2f}) ->', nr <= 0.10 and rr >= 3 * nr)
    jp = OUT / 'judge.json'
    if not jp.exists():
        print('6-8. no judge.json yet')
        return
    jd = json.loads(jp.read_text())

    def named(t):
        return sum(bool(g['names_direction']) for u in PREF for rid, g in jd.get(u, {}).items() if rid.startswith(t))

    print(f'6. names direction, oriented: real {named("real")}, null {named("null")} ->',
          named('real') >= 5 and named('null') < named('real'))

    def mean(u, t='real'):
        x = [g['match'] for rid, g in jd.get(u, {}).items() if rid.startswith(t) and g['match'] is not None]
        return statistics.mean(x) if x else float('nan')

    for u in jd:
        print(f'   {u} judge real {mean(u):.2f} null {mean(u, "null"):.2f}')
    print('7. 443, 429 real >= 2.5:', all(mean(u) >= 2.5 for u in ('mixed4b:443', 'mixed4b:429')))
    print('8. 237, 373 real >= 4:', all(mean(u) >= 4 for u in ('mixed4b:237', 'mixed4b:373')))


if __name__ == '__main__':
    main()
