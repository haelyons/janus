"""Display v4 for InceptionV1 mixed4e:55: U-860's two grids plus feature-visualisation renders of the unit, its
strongest inputs and readers by weight (runs/featvis, tools/featvis.py), relabelled so no layer name is shown.
One factor changed from U-860: the renders. Held-out photos unchanged.

usage: python -I bank/img_cards_v4.py
"""
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from img_cards_v3 import grid, titled

B = Path(__file__).resolve().parent
J = B.parent
JOBS = [('mixed4e_55', 'the unit itself')] + \
       [(f'mixed4d_{c}', f'input {n + 1}, w={w}') for n, (c, w) in
        enumerate([(493, '+0.311'), (511, '+0.128'), (61, '+0.116'), (352, '+0.102'), (334, '+0.080'), (383, '+0.079'),
                   (345, '-0.072'), (456, '-0.067')])] + \
       [(f'mixed5a_{c}', f'reader {n + 1}, w={w}') for n, (c, w) in enumerate([(94, '+0.100'), (240, '+0.058'), (81, '+0.057')])]


def main():
    tiles = [Image.open(J / 'runs/featvis' / f'{f}.png').convert('RGB') for f, _ in JOBS]
    rg = grid(tiles, [lab for _, lab in JOBS], 'Grid 3: renders. Synthetic images optimised to maximise each unit.', cols=4)
    old = json.loads((B / 'KEY_img_v2.json').read_text())['U-860']
    cid = f'U-{random.Random(754).randrange(100, 999)}'
    out = {'a_top': titled(B / 'cards/U-860_pos.jpg', 'Grid 1: strongest photos.'),
           'b_trough': titled(B / 'cards/U-860_neg.jpg', 'Grid 2: most negative spots.'), 'c_renders': rg}
    for k, im in out.items():
        im.save(B / 'cards' / f'{cid}_{k}.jpg', quality=88)
    text = (B / 'cards/U-860.md').read_text().replace('# Unit U-860', f'# Unit {cid}')
    text += ('Third attached grid: feature visualisations, synthetic images optimised to make one unit as active as '
             'possible. First the unit itself; then its eight strongest inputs from the layer below, by weight (the unit\'s '
             'value before the ReLU is exactly a weighted sum of 528 such inputs plus a constant; six positive, two '
             'negative); then the three units in the next layer that read from it with the largest positive weights.\n')
    (B / 'cards' / f'{cid}.md').write_text(text)
    (B / 'heldout' / f'{cid}.json').write_text((B / 'heldout/U-860.json').read_text())
    key = {cid: {**old, 'format': 'renders', 'display': 'v4', 'images': [f'{cid}_{k}.jpg' for k in out]}}
    (B / 'KEY_img_v4.json').write_text(json.dumps(key, indent=1))
    print(cid)


if __name__ == '__main__':
    main()
