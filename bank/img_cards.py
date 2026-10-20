"""Build the InceptionV1 mixed4e:55 cards (field and full) and their 20 held-out images.

Card pool and held-out pool are disjoint (70/30 split by file name). Positive score = spatial max
of the post-ReLU map; negative score = spatial min of the pre-ReLU map.

usage: python -I bank/img_cards.py
"""
import json
import random
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

B = Path(__file__).resolve().parent
CATS = {281, 282, 283, 284, 285}
CARS = {407, 436, 468, 479, 511, 581, 609, 627, 656, 661, 717, 734, 751, 817}


def grid(idx, labels, path, cols=5):
    rows = (len(idx) + cols - 1) // cols
    W, H = 224, 224 + 26
    g = Image.new('RGB', (cols * W, rows * H), (255, 255, 255))
    d = ImageDraw.Draw(g)
    for n, (i, lab) in enumerate(zip(idx, labels)):
        x, y = (n % cols) * W, (n // cols) * H
        g.paste(Image.open(B / 'img/thumbs' / f'{i:05d}.jpg'), (x, y))
        d.text((x + 6, y + 228), f'#{n + 1}   {lab}', fill=(0, 0, 0))
    g.save(path, quality=88)


def main():
    z = np.load(B / 'img/acts.npz')
    pre, cls = z['maps'].astype(np.float32), z['classes']
    names = json.loads((B / 'img/names.json').read_text())
    pos, neg = np.maximum(pre, 0).reshape(len(pre), -1).max(1), pre.reshape(len(pre), -1).min(1)
    held = np.array([zlib.crc32(n.encode()) % 10 >= 7 for n in names])
    card = np.where(~held)[0]
    from torchvision.models import GoogLeNet_Weights
    cat_names = GoogLeNet_Weights.IMAGENET1K_V1.meta['categories']

    top = card[np.argsort(-pos[card])[:20]]
    bot = card[np.argsort(neg[card])[:20]]
    cmax = float(pos[top[0]])
    print('anchor: top-20 classes:', [cat_names[c] for c in cls[top]])
    print('negative-20 classes:', [cat_names[c] for c in cls[bot]])
    print(f'card max {cmax:.2f}; held-out pool {held.sum()}')

    (B / 'cards').mkdir(exist_ok=True)
    (B / 'heldout').mkdir(exist_ok=True)
    rng = random.Random(55)
    ids = rng.sample(range(100, 999), 2)
    key = {}
    for fmt, num in zip(('field', 'full'), ids):
        cid = f'U-{num}'
        if fmt == 'field':
            grid(top, [f'activation {round(pos[i] / cmax * 10)}' for i in top], B / 'cards' / f'{cid}_pos.jpg')
            text = (f'# Unit {cid}\n\nRecorded from one unit (a single neuron) in an image classification network, run over '
                    'several thousand natural photos.\nThe attached grid shows the 20 photos on which the unit is most active, '
                    'strongest first. Under each is its activation, scaled 0-10 relative to the strongest.\n')
            imgs = [f'{cid}_pos.jpg']
        else:
            grid(top, [f'activation {pos[i]:.1f}' for i in top], B / 'cards' / f'{cid}_pos.jpg')
            grid(bot, [f'activation {neg[i]:.1f}' for i in bot], B / 'cards' / f'{cid}_neg.jpg')
            text = (f'# Unit {cid}\n\nRecorded from one unit (a single neuron) in an image classification network, run over '
                    'several thousand natural photos.\nFirst attached grid: the 20 photos on which the unit is most active, '
                    'strongest first, with raw activation values. Second attached grid: the 20 photos that push the unit '
                    'most strongly negative (its value before the ReLU), most negative first, with raw values.\n'
                    f'Across all photos the unit is above 30% of its maximum on {np.mean(pos[card] >= 0.3 * cmax) * 100:.1f}% of them.\n')
            imgs = [f'{cid}_pos.jpg', f'{cid}_neg.jpg']
        (B / 'cards' / f'{cid}.md').write_text(text)
        key[cid] = {'unit': 'inceptionv1_mixed4e_55', 'format': fmt, 'truth_class': None, 'images': imgs, 'card_max': cmax}
    # one held-out set, shared by both cards
    hp = np.where(held)[0]
    cat, car = np.isin(cls[hp], list(CATS)), np.isin(cls[hp], list(CARS))
    animal = cls[hp] <= 397  # ImageNet classes 0-397 are animals
    other = ~cat & ~car
    fire, no = pos[hp] >= 0.3 * cmax, pos[hp] <= 0.1 * cmax
    quads = {  # cats and cars nearly always fire, so there is no cat/car "no" bucket
        'cat_fire': (hp[cat & fire], 2), 'car_fire': (hp[car & fire], 2),
        'animal_fire': (hp[other & animal & fire], 3), 'object_fire': (hp[other & ~animal & fire], 3),
        'animal_no': (hp[other & animal & no], 5), 'object_no': (hp[other & ~animal & no], 5),
    }
    items = []
    for q, (pool, k) in quads.items():
        print(q, len(pool))
        items += [(int(i), q) for i in rng.sample(list(pool), k)]
    rng.shuffle(items)
    truth = [{'id': n, 'fires': q.endswith('fire'), 'tag': q, 'act': round(float(pos[i]) / cmax, 3),
              'class': cat_names[cls[i]], 'thumb': int(i)} for n, (i, q) in enumerate(items, 1)]
    for t in truth:
        Image.open(B / 'img/thumbs' / f'{t["thumb"]:05d}.jpg').save(B / 'heldout' / f'img_{t["id"]:02d}.jpg', quality=88)
    for cid in key:
        key[cid]['heldout'] = truth
        (B / 'heldout' / f'{cid}.json').write_text(json.dumps([{'id': t['id'], 'image': f'img_{t["id"]:02d}.jpg'} for t in truth]))
    (B / 'KEY_img.json').write_text(json.dumps(key, indent=1))
    print('held-out:', [(t['tag'], t['class']) for t in truth])


if __name__ == '__main__':
    main()
