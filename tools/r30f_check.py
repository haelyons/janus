"""Round 30 final-reading checks (PREDICTIONS.md, round 30F), on runs/r30<stage>/finalread/<unit>/stats.json and
runs/r30<stage>/pairtable/<unit>/stats.json. Written before any round-30 final reading exists.

usage: python -I tools/r30f_check.py --stage a|b
"""
import argparse
import json
import sys
from pathlib import Path

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
from r30_check import SIDE, lr_side  # noqa: E402

SAMPLES = ('s1', 's2', 's3')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['a', 'b'])
    a = ap.parse_args()
    units = json.loads((J / f'authors/round30{a.stage}_units.json').read_text())
    src, out = J / f'runs/r30{a.stage}/pairtable', J / f'runs/r30{a.stage}/finalread'
    P = {u: json.loads((src / u / 'stats.json').read_text()) for u in units}
    F = {u: json.loads((out / u / 'stats.json').read_text()) for u in units}
    tally = {'real': [0, 0, 0, 0], 'null': [0, 0, 0, 0]}  # held picks, kept; not-held picks, kept
    lr_unheld_kept, adds, side_named = [], [], {}
    for u in units:
        ev = {p['row']: p for p in P[u]['picks']['every']}
        side_named[u] = 0
        for arm in ('real', 'null'):
            for s in SAMPLES:
                rec = F[u].get(f'{arm}_{s}')
                if not rec or not rec['valid']:
                    print(f'{u} {arm}_{s}: no valid final reading')
                    continue
                t = tally[arm]
                for r in rec['picked']:
                    kept = r in rec['rows']
                    if r in rec['held']:
                        t[0], t[1] = t[0] + 1, t[1] + kept
                    else:
                        t[2], t[3] = t[2] + 1, t[3] + kept
                if arm == 'real':
                    for r in rec['rows']:
                        p = ev.get(r)
                        if p and lr_side(P[u], p) and p['T']['outcome'] != 'held':
                            lr_unheld_kept.append((u, s, r, lr_side(P[u], p)))
                        if r not in rec['picked']:
                            adds.append((u, s, r, p['T']['outcome'] if p else 'no check'))
                    side_named[u] += bool(SIDE.search(rec['explanation'] or ''))
                print(f'{u} {arm}_{s}: picked {rec["picked"]} held {rec["held"]} final {rec["rows"]} | '
                      f'{(rec["explanation"] or "")[:200]}')
    rh, rhk, rn, rnk = tally['real']
    nh, nhk, nn, nnk = tally['null']
    print(f'F1. real: held picks kept {rhk}/{rh}, not-held picks kept {rnk}/{rn} ->',
          rh > 0 and rn > 0 and rhk >= 0.8 * rh and rnk <= 0.2 * rn)
    print(f'F2. real final rows that are left/right rows and did not hold on T (every arm): {lr_unheld_kept} ->',
          len(lr_unheld_kept) <= 2)
    print(f'F3. null: picks kept {nhk + nnk}/{nh + nn} ->', nh + nn > 0 and nhk + nnk <= 0.2 * (nh + nn))
    held_adds = sum(x[3] == 'held' for x in adds)
    print(f'recorded: unchecked additions {len(adds)}, held on T {held_adds}: {adds}')
    print(f'recorded: real final explanations with a left/right word (of 3; claims and retractions alike): '
          f'{json.dumps(side_named)}')


if __name__ == '__main__':
    main()
