"""Read the circuit into mixed4e:55: which mixed4d inputs carry it when it fires on cats, cars, other animals
and objects; how much those sets overlap; what each main input neuron's own top photos are.

usage: python -I tools/circuit_read.py   (reads runs/circuit/circuit.npz; writes runs/circuit/inputs.jpg, read.txt)
"""
import collections
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'bank'))
from img_cards import CATS, CARS
from torchvision.models import GoogLeNet_Weights

NAMES = GoogLeNet_Weights.IMAGENET1K_V1.meta['categories']
z = np.load(J / 'runs/circuit/circuit.npz')
SMAX, CON, W = z['smax'].astype(np.float32), z['contrib'].astype(np.float32), z['w']
PRE = np.load(J / 'bank/img/acts.npz')['maps'].astype(np.float32)
CLS = np.load(J / 'bank/img/acts.npz')['classes']
POS = np.maximum(PRE, 0).reshape(len(PRE), -1).max(1)
FIRE = POS >= 0.3 * 1527.0
GROUPS = {'cat': np.isin(CLS, list(CATS)), 'car': np.isin(CLS, list(CARS)),
          'other animal': (CLS <= 397) & ~np.isin(CLS, list(CATS)), 'object': (CLS > 397) & ~np.isin(CLS, list(CARS))}


def what(c, k=40):
    top = np.argsort(-SMAX[:, c])[:k]
    return ', '.join(f'{NAMES[x]} {n}' for x, n in collections.Counter(CLS[top]).most_common(4))


def main():
    lines, means = [], {}
    for g, m in GROUPS.items():
        sel = FIRE & m
        means[g] = CON[sel].mean(0)
        top = np.argsort(-means[g])[:8]
        share = means[g][top].sum() / means[g][means[g] > 0].sum()
        lines.append(f'{g} ({sel.sum()} firing photos): top inputs ' + ', '.join(f'{c}' for c in top) +
                     f'  (top 8 carry {share:.0%} of positive contribution)')
    gs = list(GROUPS)
    lines.append('cosine similarity of mean contribution vectors:')
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            a, b = means[gs[i]], means[gs[j]]
            lines.append(f'  {gs[i]} vs {gs[j]}: {a @ b / np.linalg.norm(a) / np.linalg.norm(b):.2f}')
    t = {g: set(np.argsort(-means[g])[:10]) for g in gs}
    lines.append(f'top-10 overlap cat & car: {sorted(t["cat"] & t["car"])}; cat & object: {sorted(t["cat"] & t["object"])}; '
                 f'car & object: {sorted(t["car"] & t["object"])}')
    allm = CON[FIRE].mean(0)
    main_in = list(np.argsort(-allm)[:8])
    neg_in = list(np.argsort(W * SMAX.mean(0))[:4])
    lines.append('main positive inputs over all firing photos (share of mean positive contribution; own top photos):')
    for c in main_in:
        lines.append(f'  {c}: {allm[c] / allm[allm > 0].sum():.1%}  w={W[c]:+.3f}  | {what(c)}')
    lines.append('strongest negative inputs (w x mean activity; own top photos):')
    for c in neg_in:
        lines.append(f'  {c}: w={W[c]:+.3f}  | {what(c)}')
    (J / 'runs/circuit/read.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    rows = main_in[:6] + neg_in[:2]
    g = Image.new('RGB', (8 * 140, len(rows) * 160), (255, 255, 255))
    d = ImageDraw.Draw(g)
    for r, c in enumerate(rows):
        for k, i in enumerate(np.argsort(-SMAX[:, c])[:8]):
            g.paste(Image.open(J / 'bank/img/thumbs' / f'{i:05d}.jpg').resize((140, 140)), (k * 140, r * 160))
        d.text((4, r * 160 + 142), f'input {c}  w={W[c]:+.3f}  {"(negative)" if c in neg_in else ""}', fill=(0, 0, 0))
    g.save(J / 'runs/circuit/inputs.jpg', quality=85)


if __name__ == '__main__':
    main()
