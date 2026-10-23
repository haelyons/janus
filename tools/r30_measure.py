"""Round 30 measurement diagnostics (the sealed next step after 30A; no Claude calls): how well the pairwise table
replicates from the D half to the T half, on rounds 29 and 30A's every-arm rows.

- per unit: correlation of d on D with d on T (and with d on T_null, answers shuffled within class), the rows at 95
  or above on D that hold on T, and the smaller answer count per row;
- per question: the same correlations over all units' rows;
- by D percentile band: how often T keeps the sign, reaches 95, reaches 90, against T_null;
- per unit: heavy tails of the within-class residuals on each half (kurtosis, share of the 5 largest squares).

usage: python -I tools/r30_measure.py
"""
import json
import sys
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
import proploop  # noqa: E402

QN = ['Q1 facing', 'Q2 size', 'Q3 cut off', 'Q4 position', 'Q5 light', 'Q6 tint', 'Q7 tilt', 'Q8 blur']
NONE = ('no single main subject', 'no clear facing', 'no horizon')


def corr(x, y):
    k = [i for i in range(len(x)) if x[i] is not None and y[i] is not None]
    return float(np.corrcoef([x[i] for i in k], [y[i] for i in k])[0, 1]) if len(k) > 2 else float('nan')


def up(c):
    """d > 0 and pct >= 95 on a check (T or T_null)"""
    return c['d'] is not None and c['d'] > 0 and c['pct'] >= 95


def main():
    units = {f'r29 {d.name}': json.loads((d / 'stats.json').read_text())
             for d in sorted((J / 'runs/pairtable').glob('mixed*'))}
    units.update({f'r30a {d.name}': json.loads((d / 'stats.json').read_text())
                  for d in sorted((J / 'runs/r30a/pairtable').glob('mixed*'))})
    names = json.loads(proploop.NAMES.read_text())
    with np.load(proploop.ACTS) as z:
        labels = np.asarray(z['classes'], dtype=np.int64)

    print('unit                 rows  r(D,T)  r(D,Tnull)  D>=95 held on T  median min n | kurtosis D T  top5 share D T')
    byq = {q: ([], [], []) for q in range(8)}
    pooled = []
    for name, st in units.items():
        ev = st['picks']['every']
        D, T, N = ([p[k]['d'] for p in ev] for k in ('D', 'T', 'T_null'))
        hi = [p for p in ev if p['D']['pct'] >= 95 and p['D']['d'] > 0]
        n = np.median([min(p['D']['n_first'], p['D']['n_second']) for p in ev])
        layer, c = name.split()[1].split(':')
        v = np.load(proploop.OUT / f'values_{layer}.npy', mmap_mode='r')[:, int(c)]
        tails = []
        for half in ('D', 'T'):
            ids = np.asarray(st[half])
            x = np.asarray(v[ids], dtype=np.float64)
            lab = labels[ids]
            r = x.copy()
            for k in np.unique(lab):
                r[lab == k] -= x[lab == k].mean()
            z2 = (r / r.std()) ** 2
            tails.append((float((z2 ** 2).mean()), float(np.sort(z2)[::-1][:5].sum() / z2.sum())))
        print(f'{name:20s} {len(ev):4d}  {corr(D, T):+.2f}   {corr(D, N):+.2f}       {sum(up(p["T"]) for p in hi):2d} / '
              f'{len(hi):2d}         {n:3.0f}      | {tails[0][0]:4.1f} {tails[1][0]:4.1f}   {tails[0][1]:.2f} {tails[1][1]:.2f}')
        for p in ev:
            b = byq[p['q']]
            b[0].append(p['D']['d']), b[1].append(p['T']['d']), b[2].append(p['T_null']['d'])
            if p['T']['d'] is not None and p['T_null']['d'] is not None:
                pooled.append((p['D']['pct'], p['T'], p['T_null'], any(s in p['second_answer'] for s in NONE)))
    allD, allT, allN = ([x for b in byq.values() for x in b[i]] for i in range(3))
    print(f'all rows {len(allD)}: r(D,T) {corr(allD, allT):+.2f}, r(D,Tnull) {corr(allD, allN):+.2f}')
    for q, (D, T, N) in byq.items():
        print(f'  {QN[q]:12s} rows {len(D):3d}  r(D,T) {corr(D, T):+.2f}  r(D,Tnull) {corr(D, N):+.2f}')

    def band(sel, label):
        same = np.mean([t['d'] > 0 for _, t, _, _ in sel])
        same_n = np.mean([t['d'] > 0 for _, _, t, _ in sel])
        h95 = np.mean([up(t) for _, t, _, _ in sel])
        h95_n = np.mean([up(t) for _, _, t, _ in sel])
        h90 = np.mean([t['d'] > 0 and t['pct'] >= 90 for _, t, _, _ in sel])
        med = np.median([t['pct'] for _, t, _, _ in sel])
        print(f'{label:40s} n {len(sel):3d}: T same sign {same:.2f} (null {same_n:.2f}); T >= 95 {h95:.2f} '
              f'(null {h95_n:.2f}); T >= 90 {h90:.2f}; median T pct {med:.0f}')
    print('rows by D percentile (every arm, rows with a T and T_null value):')
    band([r for r in pooled if r[0] >= 95], 'D >= 95')
    band([r for r in pooled if 90 <= r[0] < 95], 'D 90-95')
    band([r for r in pooled if 75 <= r[0] < 90], 'D 75-90')
    band([r for r in pooled if r[0] < 75], 'D < 75')
    band([r for r in pooled if r[0] >= 95 and r[3]], 'D >= 95, second answer a "none" answer')
    band([r for r in pooled if r[0] >= 95 and not r[3]], 'D >= 95, two real answers')


if __name__ == '__main__':
    main()
