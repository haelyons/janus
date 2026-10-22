"""Round 27 post hoc A (PREDICTIONS.md): each prediction's d for the unit's 3 peers, on the same checked photos and
real answers (runs/contrastloop/<unit>/stats.json, answers.json; values from runs/proploop/values_<layer>.npy).
No calls. Writes runs/contrastloop/peers_posthoc.txt.

usage: python -I tools/r27_peers.py
"""
import json
import sys
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
import contrastloop  # noqa: E402
import proploop  # noqa: E402

OUT = J / 'runs/contrastloop'
UNITS = json.loads((J / 'authors/round27_units.json').read_text())
FOCUS = ['mixed4b:443', 'mixed4b:429']
CONTROLS = ['mixed4d:409', 'mixed4b:237', 'mixed4b:373', 'mixed4e:55', 'mixed5a:719']


def main():
    with np.load(J / 'bank/img/acts.npz') as z:
        labels = np.asarray(z['classes'], dtype=np.int64)
    V = {}
    L, held_shared = [], {}
    for u in UNITS:
        st = json.loads((OUT / u / 'stats.json').read_text())
        layer, c = st['layer'], st['chan']
        if layer not in V:
            V[layer] = np.load(J / f'runs/proploop/values_{layer}.npy')
        T = np.asarray(st['T'], dtype=np.int64)
        R = proploop.centre(V[layer][T], labels[T])
        ac = json.loads((OUT / u / 'answers.json').read_text())
        row_of = {}
        for b in ac.values():
            row_of.update(zip(b['photos'], b['answers']))
        rows = [row_of[i] for i in T.tolist()]
        peers = [p['chan'] for p in st['peers']]
        L.append(f'== {u}  peers ' + ', '.join(f'{p["unit"]} ({p["corr"]:.2f})' for p in st['peers']))
        held_shared[u] = [0, 0]
        for s, rec in st['samples'].items():
            for j, p in enumerate(rec['predictions'] or [], 1):
                ans = [r[p['question_index']] for r in rows]
                got = contrastloop.contrast(R, ans, p['most'], p['least'])
                if got is None:
                    continue
                d = got[2]
                assert abs(d[c] - p['real']['d']) < 1e-9, (u, s, j)
                pp = [proploop.percentile(d, q) for q in peers]
                held = p['real']['outcome'] == 'held'
                shared = held and max(pp) >= 95
                if held:
                    held_shared[u][0] += 1
                    held_shared[u][1] += shared
                tag = 'HELD shared' if shared else 'HELD unit only' if held else ''
                L.append(f'{s} P{j} unit d {d[c]:+.2f} pct {proploop.percentile(d, c):5.1f} | peers '
                         + ', '.join(f'{d[q]:+.2f} ({x:.1f})' for q, x in zip(peers, pp)) + f'  {tag}')
                L.append(f'      {p["question"][:90]} | most: {p["answers"][p["most"]][:50]} | '
                         f'least: {p["answers"][p["least"]][:40]}')
    L.append('')
    L.append('held predictions, shared with a peer at >= 95: ' + ', '.join(
        f'{u} {s}/{h}' for u, (h, s) in held_shared.items()))
    fh, fs = (sum(held_shared[u][k] for u in FOCUS) for k in (0, 1))
    ch, cs = (sum(held_shared[u][k] for u in CONTROLS) for k in (0, 1))
    L.append(f'A1. 443 + 429: {fs} of {fh} held predictions shared -> {fh > 0 and fs * 2 >= fh}')
    L.append(f'A2. controls: {cs} of {ch} held predictions shared -> {ch > 0 and cs * 2 < ch}')
    text = '\n'.join(L)
    (OUT / 'peers_posthoc.txt').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
