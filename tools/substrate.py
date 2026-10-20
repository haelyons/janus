"""Substrate checks on InceptionV1 mixed4e:55, no subjects involved.

1. Class ranking: mean spatial-max activation per ImageNet class; where objects that are neither animals nor
   vehicles land (pool table, fireboat, bullet train...).
2. Heatmaps: where in the photo the unit peaks (top-20) and troughs (most-negative 20), overlaid.
3. Drawn shapes: V/chevron, triangle, circle, ring, square, at several sizes and polarities.

usage: python -I tools/substrate.py   (writes runs/substrate/)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'bank'))
from img_acts import load_model, run_batch
from img_cards import CATS, CARS

OUT = J / 'runs/substrate'
OUT.mkdir(parents=True, exist_ok=True)
z = np.load(J / 'bank/img/acts.npz')
PRE, CLS = z['maps'].astype(np.float32), z['classes']
POS, NEG = np.maximum(PRE, 0).reshape(len(PRE), -1).max(1), PRE.reshape(len(PRE), -1).min(1)
from torchvision.models import GoogLeNet_Weights
NAMES = GoogLeNet_Weights.IMAGENET1K_V1.meta['categories']
MAX = 1527.0  # card max


def class_ranking():
    means = np.array([POS[CLS == c].mean() for c in range(1000)])
    order = np.argsort(-means)
    other = [c for c in order if c > 397 and c not in CARS]  # 0-397 are animals
    lines = ['top 25 classes overall: ' + ', '.join(f'{NAMES[c]} {means[c] / MAX:.2f}' for c in order[:25]),
             'top 25 non-animal, non-car classes: ' + ', '.join(f'{NAMES[c]} {means[c] / MAX:.2f}' for c in other[:25])]
    for name in ['fireboat', 'pool table', 'bullet train', 'volcano', 'torch', 'golf ball', 'soccer ball',
                 'orange', 'pug', 'hamster', 'jacamar', 'hummingbird', 'grille', 'convertible']:
        c = NAMES.index(name)
        lines.append(f'{name}: mean {means[c] / MAX:.2f} of max, rank {int(np.where(order == c)[0][0]) + 1}/1000')
    return lines


def overlay(idx, path, negative=False):
    tiles = []
    for i in idx:
        im = Image.open(J / 'bank/img/thumbs' / f'{i:05d}.jpg').convert('RGB')
        m = -PRE[i] if negative else np.maximum(PRE[i], 0)
        m = m / (m.max() + 1e-6)
        heat = Image.fromarray((m * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        red = Image.new('RGB', (224, 224), (255, 0, 0) if not negative else (0, 80, 255))
        tiles.append(Image.composite(red, im, heat.point(lambda v: int(v * 0.55))))
    g = Image.new('RGB', (5 * 224, ((len(tiles) + 4) // 5) * 224), (255, 255, 255))
    for n, t in enumerate(tiles):
        g.paste(t, ((n % 5) * 224, (n // 5) * 224))
    g.save(path, quality=88)


def shapes():
    model = load_model(J / 'raw/weights/inception5h.pth')
    def draw(kind, size, fg, bg):
        im = Image.new('RGB', (224, 224), bg)
        d = ImageDraw.Draw(im)
        c, r, w = 112, size // 2, max(6, size // 8)
        if kind == 'V':
            d.line([(c - r, c - r), (c, c + r), (c + r, c - r)], fill=fg, width=w, joint='curve')
        elif kind == 'Λ':
            d.line([(c - r, c + r), (c, c - r), (c + r, c + r)], fill=fg, width=w, joint='curve')
        elif kind == 'triangle':
            d.polygon([(c - r, c + r), (c, c - r), (c + r, c + r)], fill=fg)
        elif kind == 'two ears':
            for dx in (-r, r // 3):
                d.polygon([(c + dx, c + r // 2), (c + dx + r // 3, c - r), (c + dx + 2 * r // 3, c + r // 2)], fill=fg)
        elif kind == 'circle':
            d.ellipse([c - r, c - r, c + r, c + r], fill=fg)
        elif kind == 'ring':
            d.ellipse([c - r, c - r, c + r, c + r], outline=fg, width=w)
        elif kind == 'square':
            d.rectangle([c - r, c - r, c + r, c + r], fill=fg)
        return im
    kinds = ['V', 'Λ', 'triangle', 'two ears', 'circle', 'ring', 'square']
    rows, imgs, labels = [], [], []
    for fg, bg in [((0, 0, 0), (255, 255, 255)), ((255, 255, 255), (128, 128, 128)), ((150, 90, 40), (40, 110, 40))]:
        for size in (80, 140, 200):
            for k in kinds:
                imgs.append(draw(k, size, fg, bg))
                labels.append((k, size, fg))
    acts = run_batch(model, imgs)
    pos, neg = np.maximum(acts, 0).reshape(len(acts), -1).max(1), acts.reshape(len(acts), -1).min(1)
    by = {k: [] for k in kinds}
    for (k, s, fg), p, n in zip(labels, pos, neg):
        by[k].append((p / MAX, n))
    lines = ['drawn shapes (9 variants each: 3 colourings x 3 sizes): mean / max of spatial-max activation as share of card max; mean most-negative raw value']
    for k in kinds:
        v = np.array(by[k])
        lines.append(f'  {k:9s} mean {v[:, 0].mean():.3f}  max {v[:, 0].max():.3f}  most-negative mean {v[:, 1].mean():.0f}')
    grid = Image.new('RGB', (7 * 112, 112), (255, 255, 255))
    for n, k in enumerate(kinds):
        grid.paste(imgs[kinds.index(k) + 7].resize((112, 112)), (n * 112, 0))
    grid.save(OUT / 'shapes_sample.jpg')
    return lines


def main():
    names = json.loads((J / 'bank/img/names.json').read_text())
    import zlib
    card = np.where(np.array([zlib.crc32(n.encode()) % 10 < 7 for n in names]))[0]
    top = card[np.argsort(-POS[card])[:20]]
    bot = card[np.argsort(NEG[card])[:20]]
    overlay(top, OUT / 'top20_heat.jpg')
    overlay(bot, OUT / 'neg20_heat.jpg', negative=True)
    lines = class_ranking() + shapes()
    (OUT / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
