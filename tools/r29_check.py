"""Round 29 prediction checks (PREDICTIONS.md, round 29), on runs/pairtable/<unit>/stats.json. Written before the
results exist.

usage: python -I tools/r29_check.py
"""
import json
import re
from pathlib import Path

J = Path(__file__).resolve().parent.parent
OUT = J / 'runs/pairtable'
PREF = {'mixed4b:431': 'right', 'mixed4b:443': 'right', 'mixed4b:429': 'left', 'mixed4b:384': 'left'}
CONTROLS = ['mixed4b:409', 'mixed4b:418']
OTHER = {'left': 'right', 'right': 'left'}
AGENT_REAL = ['real_s1', 'real_s2', 'real_s3']
AGENT_NULL = ['null_s1', 'null_s2', 'null_s3']
RULE = ['rule_real', 'rule_null']
PICKING = AGENT_REAL + AGENT_NULL + RULE


def favoured(st, r):
    """the side a row favours when it is the direction row (Q1, left against right), else None"""
    if r['q'] != 0:
        return None
    a = st['questions'][0]['answers']
    first, second = a[r['first']].strip().lower(), a[r['second']].strip().lower()
    return first if {first, second} == {'left', 'right'} else None


def held(p, key='T'):
    return p[key]['outcome'] == 'held'


def rate(picks, key='T'):
    chk = [p for p in picks if p[key]['outcome'] != 'too few']
    return sum(held(p, key) for p in chk), len(chk)


def main():
    units = list(PREF) + CONTROLS
    S = {u: json.loads((OUT / u / 'stats.json').read_text()) for u in units}
    for u in units:
        st = S[u]
        print(f'== {u} peers ' + ', '.join(f'{p["unit"]} ({p["corr"]:.2f})' for p in st['peers']))
        dr = [r for r in st['tables']['real'] if favoured(st, r)]
        for r in dr:
            print(f'  D direction row {r["row"]}: favours {favoured(st, r)} n {r["n_first"]}/{r["n_second"]} '
                  f'd {r["d"]:+.2f} pct {r["pct"]:.1f} peers {[round(x, 1) for x in r["peer_pct"]]}')
        for p in st['picks']['every']:
            if favoured(st, p):
                t = p['T']
                print(f'  T direction row (every): favours {favoured(st, p)} {t["outcome"]} pct {t["pct"]} '
                      f'peers {t["peer_pct"]}')
        for k in AGENT_REAL + AGENT_NULL:
            rd = st['readings'].get(k, {})
            ex = (rd.get('explanation') or '')
            sides = sorted({s for s in ('left', 'right') if re.search(rf'\b{s}\b', ex, re.I)})
            picks = [(p['row'], favoured(st, p), p['T']['outcome']) for p in st['picks'].get(k, [])]
            print(f'  {k}: sides named {sides} picks {picks} | {ex[:150]}')
        for k in RULE:
            print(f'  {k}: ' + str([(p['row'], st['questions'][p['q']]['question'][:30], p['first_answer'][:20],
                                     p['second_answer'][:20], p['T']['outcome']) for p in st['picks'][k]]))

    def has(u, arms, side, need_held=True):
        st = S[u]
        return any(favoured(st, p) == side and (held(p) or not need_held) for k in arms for p in st['picks'].get(k, []))

    ok1 = [u for u in PREF if any(favoured(S[u], r) == PREF[u] and r['pct'] >= 95 for r in S[u]['tables']['real'])]
    bad1 = [u for u in CONTROLS if any(favoured(S[u], r) and r['pct'] >= 95 for r in S[u]['tables']['real'])]
    print('1. D direction row, preferred side >= 95:', ok1, '; controls >= 95:', bad1, '->',
          len(ok1) >= 3 and not bad1)
    ok2 = [u for u in PREF if has(u, AGENT_REAL, PREF[u])]
    wrong = [(u, k) for u in PREF for k in PICKING + ['every'] if has(u, [k], OTHER[PREF[u]])]
    print('2. agent real, held preferred-side direction row:', ok2, '; wrong side held:', wrong, '->',
          len(ok2) >= 3 and not wrong)
    ctrl = [(u, k) for u in CONTROLS for k in PICKING for s in ('left', 'right') if has(u, [k], s)]
    print('3. controls, held direction rows in picking arms:', ctrl, '->', not ctrl)
    ar = [p for u in units for k in AGENT_REAL for p in S[u]['picks'].get(k, [])]
    an = [p for u in units for k in AGENT_NULL for p in S[u]['picks'].get(k, [])]
    (hr, nr), (hn, nn) = rate(ar), rate(an)
    fr, fn = (hr / nr if nr else float('nan')), (hn / nn if nn else float('nan'))
    print(f'4. held on T: agent real {hr}/{nr} ({fr:.2f}), agent null {hn}/{nn} ({fn:.2f}) ->',
          fn <= 0.15 and fr >= 2 * fn)
    allp = ar + an + [p for u in units for k in RULE for p in S[u]['picks'][k]]
    h5, n5 = rate(allp, 'T_null')
    print(f'5. shuffled-T held over agent and rule picks: {h5}/{n5} ({h5 / n5 if n5 else float("nan"):.2f}) ->',
          n5 > 0 and h5 / n5 <= 0.10)
    ha = [p for p in ar if held(p)]
    he = [p for u in units for p in S[u]['picks']['every'] if p['D']['pct'] >= 95 and held(p)]
    da, de = sum(p['T']['distinctive'] for p in ha), sum(p['T']['distinctive'] for p in he)
    sa, se = (da / len(ha) if ha else float('nan')), (de / len(he) if he else float('nan'))
    print(f'6. distinctive among held: agent real {da}/{len(ha)} ({sa:.2f}), every D>=95 {de}/{len(he)} ({se:.2f}) ->',
          sa > se)
    ok7 = [u for u in PREF if has(u, ['rule_real'], PREF[u])]
    print('7. rule real, held preferred-side direction row:', ok7, '->', len(ok7) >= 3)


if __name__ == '__main__':
    main()
