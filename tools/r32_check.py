"""Round 32 checks (PREDICTIONS.md, round 32). Written before tools/r32_basis.py has run.

  python -I tools/r32_check.py [--basis runs/r32/basis.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

J = Path(__file__).resolve().parent.parent
ORIENTED = {'mixed4b:431': 'right', 'mixed4b:443': 'right', 'mixed4b:429': 'left', 'mixed4b:384': 'left'}


def share(recs, key):
    return float(np.mean([r[key] >= 1 for r in recs])) if recs else None


def mean(recs, key):
    return float(np.mean([r[key] for r in recs])) if recs else None


def facing_row(rec, side, answers):
    """the row comparing `side` (first) against the other side on question 0, or None"""
    other = 'left' if side == 'right' else 'right'
    got = [r for r in rec['rows'] if r['q'] == 0 and answers[r['first']] == side and answers[r['second']] == other]
    return got[0] if got else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--basis', default=str(J / 'runs/r32/basis.json'))
    a = ap.parse_args()
    recs = json.loads(Path(a.basis).read_text())['records']
    G = {g: [r for r in recs if r['group'] == g] for g in ('selected', 'channel', 'direction')}
    layers = sorted({r['layer'] for r in recs})

    sc, sd = share(G['channel'], 'n_held'), share(G['direction'], 'n_held')
    p = mannwhitneyu([r['n_held'] for r in G['channel']], [r['n_held'] for r in G['direction']],
                     alternative='greater').pvalue
    print(f'B1. share with a held row: channels {sc:.3f} ({len(G["channel"])}), directions {sd:.3f} '
          f'({len(G["direction"])}); Mann-Whitney p {p:.2g} ->', sc - sd >= 0.10 and p < 0.05)

    per = {L: (mean([r for r in G['channel'] if r['layer'] == L], 'n_held'),
               mean([r for r in G['direction'] if r['layer'] == L], 'n_held')) for L in layers}
    wins = sum(c > d for c, d in per.values())
    print('B2. mean held rows per layer (channel, direction):',
          json.dumps({L: [round(c, 3), round(d, 3)] for L, (c, d) in per.items()}), f'-> channel ahead in {wins}:',
          wins >= 5)

    ds, dc = share(G['selected'], 'n_distinctive'), share(G['channel'], 'n_distinctive')
    print(f'B3. share with a held distinctive row: selected {ds:.3f}, channels {dc:.3f} ->', ds >= dc + 0.20)

    mc = mean(G['channel'], 'n_held')
    nulls = {g: mean(rs, 'n_held_null') for g, rs in G.items()}
    print('B4. mean held null rows per group:', json.dumps({g: round(v, 3) for g, v in nulls.items()}),
          f'bar {0.5 * mc:.3f} ->', all(v <= 0.5 * mc for v in nulls.values()))

    kw = sum(np.median([r['kurt_all'] for r in G['channel'] if r['layer'] == L and r['kurt_all'] is not None]) >
             np.median([r['kurt_all'] for r in G['direction'] if r['layer'] == L and r['kurt_all'] is not None])
             for L in layers)
    print(f'K. median excess kurtosis, channel above direction in {kw} of {len(layers)} layers ->', kw >= 6)

    import sys
    sys.path.insert(0, str(J / 'tools'))
    import pairtable
    answers = pairtable.load_questions()[0]['answers']
    by = {r['unit']: r for r in G['selected']}
    rows = {u: facing_row(by[u], s, answers) if u in by else None for u, s in ORIENTED.items()}
    j1 = {u: None if r is None else [r['T_outcome'], r['jT_outcome'], r['jT_pct']] for u, r in rows.items()}
    print('J1. oriented units, facing row (T outcome, joint T outcome, joint T pct):', json.dumps(j1), '->',
          sum(r is not None and r['T_outcome'] == 'held' and r['jT_outcome'] == 'held' for r in rows.values()) >= 3)

    held = [(r['unit'], x) for r in G['selected'] for x in r['rows'] if x['T_outcome'] == 'held']
    kept = [h for h in held if h[1]['jT_outcome'] == 'held']
    print(f'recorded: selected units\' held rows that also hold jointly: {len(kept)} of {len(held)}')
    qs = pairtable.load_questions()
    for u, x in held:
        if x['jT_outcome'] != 'held':
            Q = qs[x['q']]
            print(f'  lost: {u} q{x["q"] + 1} {Q["answers"][x["first"]]} > {Q["answers"][x["second"]]} '
                  f'T {x["T_pct"]:.0f}, joint T {x["jT_pct"]}')


if __name__ == '__main__':
    main()
