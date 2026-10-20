"""Display v2 for InceptionV1 mixed4e:55: the same photos and held-out set, seen through a better microscope.

- Positives: the top-20 photos, each cropped (128px window, resized to 224) around the unit's hotspot.
- Negatives: ranked by the most negative 3x3-average of the pre-ReLU map (a located trough, not one cell),
  cropped around that trough.
- Full card adds response shading: red where the unit is active, blue where it is pushed negative.
The held-out photos are unchanged (whole photos), so scores compare with rounds 1-4.

usage: python -I bank/img_cards_v2.py
"""
import json
import random
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

B = Path(__file__).resolve().parent


def tile(i, cell, m=None, rgb=(255, 0, 0)):
    im = Image.open(B / 'img/thumbs' / f'{i:05d}.jpg').convert('RGB')
    if m is not None:
        heat = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        im = Image.composite(Image.new('RGB', (224, 224), rgb), im, heat.point(lambda v: int(v * 0.5)))
    cy, cx = cell[0] * 16 + 8, cell[1] * 16 + 8
    l, t = min(max(cx - 64, 0), 96), min(max(cy - 64, 0), 96)
    return im.crop((l, t, l + 128, t + 128)).resize((224, 224), Image.BICUBIC)


def grid(tiles, labels, path, cols=5):
    W, H = 224, 250
    g = Image.new('RGB', (cols * W, ((len(tiles) + cols - 1) // cols) * H), (255, 255, 255))
    d = ImageDraw.Draw(g)
    for n, (t, lab) in enumerate(zip(tiles, labels)):
        x, y = (n % cols) * W, (n // cols) * H
        g.paste(t, (x, y))
        d.text((x + 6, y + 228), f'#{n + 1}   {lab}', fill=(0, 0, 0))
    g.save(path, quality=88)


def main():
    pre = np.load(B / 'img/acts.npz')['maps'].astype(np.float32)
    names = json.loads((B / 'img/names.json').read_text())
    pos = np.maximum(pre, 0).reshape(len(pre), -1).max(1)
    pad = np.pad(pre, ((0, 0), (1, 1), (1, 1)), mode='edge')
    pool = sum(pad[:, dy:dy + 14, dx:dx + 14] for dy in range(3) for dx in range(3)) / 9
    trough = pool.reshape(len(pre), -1).min(1)
    card = np.where(np.array([zlib.crc32(n.encode()) % 10 < 7 for n in names]))[0]
    top, bot = card[np.argsort(-pos[card])[:20]], card[np.argsort(trough[card])[:20]]
    cmax = float(pos[top[0]])
    hot = {i: np.unravel_index(pre[i].argmax(), (14, 14)) for i in top}
    cold = {i: np.unravel_index(pool[i].argmin(), (14, 14)) for i in bot}
    old = json.loads((B / 'KEY_img.json').read_text())
    held = next(iter(old.values()))['heldout']
    key, ids = {}, random.Random(552).sample(range(100, 999), 2)
    for fmt, num in zip(('field', 'full'), ids):
        cid = f'U-{num}'
        if fmt == 'field':
            grid([tile(i, hot[i]) for i in top], [f'activation {round(pos[i] / cmax * 10)}' for i in top], B / 'cards' / f'{cid}_pos.jpg')
            text = (f'# Unit {cid}\n\nRecorded from one unit (a single neuron) in an image classification network, run over several '
                    'thousand natural photos.\nThe attached grid shows the 20 photos on which the unit is most active, strongest first, '
                    'each cropped around the spot where it is most active. Under each is its activation, scaled 0-10 relative to the strongest.\n')
            imgs = [f'{cid}_pos.jpg']
        else:
            grid([tile(i, hot[i], np.maximum(pre[i], 0) / cmax) for i in top], [f'activation {pos[i]:.1f}' for i in top], B / 'cards' / f'{cid}_pos.jpg')
            grid([tile(i, cold[i], -pre[i] / -trough[bot[0]], (0, 80, 255)) for i in bot], [f'activation {trough[i]:.1f}' for i in bot], B / 'cards' / f'{cid}_neg.jpg')
            text = (f'# Unit {cid}\n\nRecorded from one unit (a single neuron) in an image classification network, run over several '
                    'thousand natural photos.\nFirst attached grid: the 20 photos on which the unit is most active, strongest first, each '
                    'cropped around the spot where it is most active, with red shading showing where it responds and raw activation '
                    'values. Second attached grid: the 20 photos containing the spots that push the unit most strongly negative (its value '
                    'before the ReLU, averaged over a small area), most negative first, each cropped around that spot, with blue shading '
                    'showing where it is pushed negative.\n'
                    f'Across all photos the unit is above 30% of its maximum on {np.mean(pos[card] >= 0.3 * cmax) * 100:.1f}% of them.\n')
            imgs = [f'{cid}_pos.jpg', f'{cid}_neg.jpg']
        (B / 'cards' / f'{cid}.md').write_text(text)
        (B / 'heldout' / f'{cid}.json').write_text(json.dumps([{'id': t['id'], 'image': f'img_{t["id"]:02d}.jpg'} for t in held]))
        key[cid] = {'unit': 'inceptionv1_mixed4e_55', 'format': fmt, 'display': 'v2', 'truth_class': None, 'images': imgs,
                    'card_max': cmax, 'heldout': held}
        print(cid, fmt, imgs)
    (B / 'KEY_img_v2.json').write_text(json.dumps(key, indent=1))


if __name__ == '__main__':
    main()
