"""Round 33 full-round checks (PREDICTIONS.md, round 33 full). Written before any full-round session exists.

  python -I tools/r33f_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
from r33_check import keys, score  # noqa: E402

ARMS = ('A0', 'F2_pool', 'P1_active', 'Y1_yoked')
PILOT = ('U-733', 'U-421', 'O-290', 'O-158')  # pilot cards: their sample 1 is the pilot, not counted here
PLANTED = ('U-862', 'U-518', 'U-733', 'U-421')
NEURONS = ('O-848', 'O-142', 'O-975', 'O-293', 'O-290', 'O-158')
DIRECTIONS = ('O-111', 'O-981', 'O-425', 'O-540')
CARDS = PLANTED + NEURONS + DIRECTIONS


def samples(c):
    return (2, 3) if c in PILOT else (1, 2)


def cell(c, arm, K):
    """mean held-out and mean simulator rho over the card's two samples; None when a session is missing"""
    acc, rho = [], []
    for s in samples(c):
        p = J / 'runs' / arm / c / f's{s}.json'
        q = J / 'runs' / 'S0_sim' / arm / c / f's{s}.json'
        if not p.exists() or not q.exists():
            return None
        r = score(json.loads(p.read_text()), json.loads(q.read_text()), K[c])
        acc.append(r['acc'])
        rho.append(r['rho'] if r['rho'] is not None else 0.0)
    return float(np.mean(acc)), float(np.mean(rho))


def main():
    K = keys()
    T = {(c, a): cell(c, a, K) for c in CARDS for a in ARMS}
    done = [c for c in CARDS if all(T[(c, a)] for a in ARMS)]
    print(f'cards complete: {len(done)} of {len(CARDS)}')
    for c in CARDS:
        print(f'  {c}', '  '.join('-' if T[(c, a)] is None else f'{a} {T[(c, a)][0]:.2f}/{T[(c, a)][1]:.2f}'
                                 for a in ARMS))
    if len(done) < len(CARDS):
        print('not all cards complete: predictions not evaluated')
        return
    d = [T[(c, 'P1_active')][1] - T[(c, 'Y1_yoked')][1] for c in CARDS]
    wins = sum(x > 0 for x in d)
    print(f'F1. P1 - Y1 simulator rho: mean {np.mean(d):+.3f}, P1 ahead on {wins} of {len(d)} '
          f'(sign test p {binomtest(wins, len(d), alternative="greater").pvalue:.3f}) ->',
          np.mean(d) >= 0.05 and wins >= 9)
    d2 = [T[(c, 'P1_active')][0] - T[(c, 'A0')][0] for c in PLANTED]
    print(f'F2. planted cards, P1 - A0 held-out: {[round(x, 3) for x in d2]}, mean {np.mean(d2):+.3f} ->',
          np.mean(d2) >= 0.10)
    d3 = [T[(c, 'Y1_yoked')][0] - T[(c, 'F2_pool')][0] for c in CARDS]
    print(f'F3. Y1 - F2 held-out: mean {np.mean(d3):+.3f} ->', np.mean(d3) >= 0.05)
    n, dr = np.mean([T[(c, 'A0')][1] for c in NEURONS]), np.mean([T[(c, 'A0')][1] for c in DIRECTIONS])
    print(f'F4. A0 simulator rho: neurons {n:.3f}, directions {dr:.3f} ->', n - dr >= 0.10)
    for a in ARMS:
        print(f'recorded: {a:10s} mean held-out {np.mean([T[(c, a)][0] for c in CARDS]):.3f}, '
              f'mean simulator rho {np.mean([T[(c, a)][1] for c in CARDS]):.3f}')


if __name__ == '__main__':
    main()
