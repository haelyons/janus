"""Round 29B prediction checks (PREDICTIONS.md, round 29B), on runs/finalread/<unit>/stats.json and
runs/pairtable/<unit>/stats.json. Written before the results exist.

usage: python -I tools/r29b_check.py
"""
import json
from pathlib import Path

J = Path(__file__).resolve().parent.parent
SRC = J / 'runs/pairtable'
OUT = J / 'runs/finalread'
PREF = {'mixed4b:431': 'right', 'mixed4b:443': 'right', 'mixed4b:429': 'left', 'mixed4b:384': 'left'}
CONTROLS = ['mixed4b:409', 'mixed4b:418']
SAMPLES = ('s1', 's2', 's3')


def direction_rows(st):
    """{row number: favoured side} for the real table's direction rows (Q1, left against right)"""
    a = st['questions'][0]['answers']
    out = {}
    for r in st['tables']['real']:
        if r['q'] == 0:
            f, s = a[r['first']].strip().lower(), a[r['second']].strip().lower()
            if {f, s} == {'left', 'right'}:
                out[r['row']] = f
    return out


def main():
    units = list(PREF) + CONTROLS
    P = {u: json.loads((SRC / u / 'stats.json').read_text()) for u in units}
    F = {u: json.loads((OUT / u / 'stats.json').read_text()) for u in units}
    keep = {}
    for u in units:
        dr = direction_rows(P[u])
        for arm in ('real', 'null'):
            for s in SAMPLES:
                rec = F[u].get(f'{arm}_{s}')
                if not rec or not rec['valid']:
                    keep[u, arm, s] = None
                    continue
                keep[u, arm, s] = {dr[r] for r in rec['rows'] if r in dr}
                print(f'{u} {arm}_{s}: picked {rec["picked"]} held {rec["held"]} final {rec["rows"]} '
                      f'direction kept {sorted(keep[u, arm, s])} | {rec["explanation"][:160]}')
    k1 = {u: sum(1 for s in SAMPLES if keep[u, 'real', s] and PREF[u] in keep[u, 'real', s]) for u in PREF}
    print('1. real, preferred-side direction row kept (samples of 3):', k1, '->', all(v >= 2 for v in k1.values()))
    k2 = sum(1 for s in SAMPLES if keep['mixed4b:409', 'real', s])
    print('2. real, 409 direction row kept in', k2, 'samples ->', k2 == 0)
    k3 = sum(1 for u in PREF for s in SAMPLES if keep[u, 'null', s])
    print('3. null, oriented units keep a direction row in', k3, 'of 12 samples ->', k3 <= 3)
    nh, nk, hh, hk = 0, 0, 0, 0
    for u in units:
        for s in SAMPLES:
            rec = F[u].get(f'real_{s}')
            if not rec or not rec['valid']:
                continue
            for r in rec['picked']:
                kept = r in rec['rows']
                if r in rec['held']:
                    hh, hk = hh + 1, hk + kept
                else:
                    nh, nk = nh + 1, nk + kept
    print(f'4. real: not-held picks kept {nk}/{nh}, held picks kept {hk}/{hh} ->',
          nh > 0 and hh > 0 and nk <= 0.2 * nh and hk >= 0.8 * hh)


if __name__ == '__main__':
    main()
