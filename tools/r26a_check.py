"""Round 26A predictions 1-4 (mine; rules as sealed in PREDICTIONS.md), on runs/rangeprofile/<layer>.csv and
runs/lrcheck/table.csv.

usage: python -I tools/r26a_check.py
"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

J = Path(__file__).resolve().parent.parent
ORIENTED_LATE = ('mixed4b:443', 'mixed4b:429')
ORIENTED_TOP = ('mixed4b:431', 'mixed4b:384')
NOT_ORIENTED = ('mixed4b:409', 'mixed4b:418', 'mixed4c:254', 'mixed4c:407', 'mixed4b:237', 'mixed4b:373',
                'mixed4d:409', 'mixed4e:55')


def rows(layer):
    with open(J / f'runs/rangeprofile/{layer}.csv') as f:
        return {r['unit']: r for r in csv.DictReader(f)}


def num(s):
    return float(s) if s != '' else float('nan')


def chg_pct(r, s):
    return num(r[f'mirror_chg_pct_s{s}'])


def main():
    tab = {}
    for L in ('mixed4b', 'mixed4c', 'mixed4d', 'mixed4e'):
        tab.update(rows(L))
    for u in ORIENTED_LATE + ORIENTED_TOP + NOT_ORIENTED:
        print(u, 'mirror change pct s1-s3:', [round(chg_pct(tab[u], s), 1) for s in (1, 2, 3)],
              'retention pct:', [round(num(tab[u][f'mirror_ret_pct_s{s}']), 1) for s in (1, 2, 3)])
    p1 = all(max(chg_pct(tab[u], 2), chg_pct(tab[u], 3)) >= 95 for u in ORIENTED_LATE)
    p2 = all(chg_pct(tab[u], 1) >= 95 for u in ORIENTED_TOP)
    n3 = sum(all(chg_pct(tab[u], s) < 95 for s in (1, 2, 3)) for u in NOT_ORIENTED)
    print('1', 'held' if p1 else 'failed')
    print('2', 'held' if p2 else 'failed')
    print('3', f'{n3} of 8 below 95 in all strata:', 'held' if n3 >= 7 else 'failed')
    with open(J / 'runs/lrcheck/table.csv') as f:
        lr = {r['unit']: num(r['absd_pct_D']) for r in csv.DictReader(f)}
    ok_s2, ok_gt = 0, 0
    for L in ('mixed4b', 'mixed4c', 'mixed4d', 'mixed4e'):
        R = rows(L)
        us = [u for u in R if u in lr]
        res = {}
        for s in (1, 2, 3):
            x = np.array([chg_pct(R[u], s) for u in us]); y = np.array([lr[u] for u in us])
            m = np.isfinite(x) & np.isfinite(y)
            res[s] = spearmanr(x[m], y[m])[0]
        print(L, 'Spearman with |d| pct on dogs: s1 %.3f s2 %.3f s3 %.3f' % (res[1], res[2], res[3]), f'n={len(us)}')
        ok_s2 += res[2] >= 0.3
        ok_gt += res[2] > res[1]
    print('4', f's2 >= 0.3 in {ok_s2} of 4, s2 > s1 in {ok_gt} of 4:', 'held' if ok_s2 >= 3 and ok_gt >= 3 else 'failed')


if __name__ == '__main__':
    main()
