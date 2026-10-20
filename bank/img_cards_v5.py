"""Round 8 for InceptionV1 mixed4e:55: a mixed cat/car grid, and a disagreement held-out set.

- Mixed card (field format, v2 crops): the 10 strongest card-pool photos with cat labels and the 10 strongest with car
  labels, strongest first, activation scaled 0-10 against the strongest photo overall. The famous grid shows cats and
  cars side by side; our top 20 has one car.
- Disagreement set: 20 held-out-pool photos from the shape groups in shape_classes.json (fixed before activations
  were viewed), 4 per group, 2 firing (>= 30% of max) and 2 silent (< 10%) where the pool allows, else filled from
  whichever side has photos. Readings disagree here: pointed objects (V says fire, faces say no), flat-fronted
  vehicles (cars say fire, V says no), round-faced animals (furry faces say fire, V says no), pointed-face dogs.
  Shown as a third turn, after the usual held-out turn, on this card and on U-121.

usage: python -I bank/img_cards_v5.py
"""
import collections
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.models import GoogLeNet_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent))
from img_cards import CARS, CATS
from img_cards_v2 import grid, tile

B = Path(__file__).resolve().parent
N = GoogLeNet_Weights.IMAGENET1K_V1.meta['categories']


def main():
    z = np.load(B / 'img/acts.npz')
    pre, cls = z['maps'].astype(np.float32), z['classes']
    names = json.loads((B / 'img/names.json').read_text())
    pos = np.maximum(pre, 0).reshape(len(pre), -1).max(1)
    in_card = np.array([zlib.crc32(n.encode()) % 10 < 7 for n in names])
    card, pool = np.where(in_card)[0], np.where(~in_card)[0]
    cmax = float(pos[card].max())
    old = json.loads((B / 'KEY_img_v2.json').read_text())
    held = old['U-121']['heldout']
    used = {h['thumb'] for h in held}

    # disagreement set
    rng = random.Random(858)
    dis = []
    for tag, cl in json.loads((B / 'shape_classes.json').read_text()).items():
        m = pool[np.isin(cls[pool], [N.index(n) for n in cl])]
        fire = [int(i) for i in m if pos[i] >= 0.3 * cmax and i not in used]
        no = [int(i) for i in m if pos[i] < 0.1 * cmax and i not in used]
        rng.shuffle(fire), rng.shuffle(no)
        nf = min(2, len(fire))
        pick = [(i, True) for i in fire[:nf + max(0, 2 - min(2, len(no)))]] + [(i, False) for i in no[:4 - nf]]
        dis += [{'tag': f'{tag}_{"fire" if f else "no"}', 'thumb': i, 'fires': f, 'act': round(float(pos[i] / cmax), 3),
                 'class': N[cls[i]]} for i, f in pick[:4]]
        print(tag, 'eligible fire', len(fire), 'no', len(no))
    rng.shuffle(dis)
    for n, d in enumerate(dis, 1):
        d['id'] = n
        Image.open(B / 'img/thumbs' / f'{d["thumb"]:05d}.jpg').save(B / 'heldout' / f'dis_{n:02d}.jpg', quality=88)

    # mixed card
    top = lambda s: card[np.isin(cls[card], list(s))][np.argsort(-pos[card[np.isin(cls[card], list(s))]])[:10]]
    sel = np.concatenate([top(CATS), top(CARS)])
    sel = sel[np.argsort(-pos[sel])]
    cid = f'U-{random.Random(859).randrange(100, 999)}'
    hot = lambda i: np.unravel_index(pre[i].argmax(), (14, 14))
    grid([tile(i, hot(i)) for i in sel], [f'activation {round(pos[i] / cmax * 10)}' for i in sel], B / 'cards' / f'{cid}_pos.jpg')
    (B / 'cards' / f'{cid}.md').write_text(
        f'# Unit {cid}\n\nRecorded from one unit (a single neuron) in an image classification network, run over several '
        'thousand natural photos.\nThe attached grid shows 20 of the photos on which the unit is strongly active (a selection, '
        'not simply the strongest 20), strongest first, each cropped around the spot where it is most active. Under each is '
        'its activation, scaled 0-10 relative to the strongest photo overall.\n')
    dis_items = json.dumps([{'id': d['id'], 'image': f'dis_{d["id"]:02d}.jpg'} for d in dis])
    for c in (cid, 'U-121'):
        (B / 'heldout' / f'{c}_dis.json').write_text(dis_items)
    (B / 'heldout' / f'{cid}.json').write_text((B / 'heldout/U-121.json').read_text())
    key = {cid: {**old['U-121'], 'format': 'mixed', 'display': 'v5', 'images': [f'{cid}_pos.jpg'], 'disagree': dis},
           'U-121': {**old['U-121'], 'disagree': dis}}
    (B / 'KEY_img_v5.json').write_text(json.dumps(key, indent=1))
    print(cid, 'cats', int(np.isin(cls[sel], list(CATS)).sum()), 'cars', int(np.isin(cls[sel], list(CARS)).sum()),
          'disagreement', dict(collections.Counter(d['tag'] for d in dis)))


if __name__ == '__main__':
    main()
