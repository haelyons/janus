"""Display v3 for InceptionV1 mixed4e:55: U-860's two grids plus contrast, and optionally the circuit.

- Contrast card: grids 1-2 as U-860 (strongest photos, located troughs), then
  grid 3: same-class pairs. For each of the 12 classes with the highest mean activation, the class's strongest
  photo (cropped at the hotspot, red shading) beside its weakest (whole photo).
  grid 4: off-class positives. Photos at >= 30% of max whose class ranks outside the top 200 by mean activation.
- Circuit card: the same four grids, plus grid 5: the unit's main inputs from mixed4d (pre = sum_c w_c a_c + b,
  exact; see tools/circuit.py). The six with the largest mean contribution on firing photos, then the two with
  the most negative mean w_c a_c. Each row is that input's own six strongest photos, cropped at its own hotspot.
Everything is chosen from the card pool; the held-out photos are unchanged, so scores compare with rounds 1-5.

usage: python -I bank/img_cards_v3.py   (needs runs/circuit/circuit.npz and raw/weights/inception5h.pth)
"""
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from img_acts import Stop, load_model
from img_cards_v2 import tile

B = Path(__file__).resolve().parent
J = B.parent


def whole(i):
    return Image.open(B / 'img/thumbs' / f'{i:05d}.jpg').convert('RGB')


def grid(tiles, labels, title, cols, size=224):
    H = size + 26
    g = Image.new('RGB', (cols * size, 30 + ((len(tiles) + cols - 1) // cols) * H), (255, 255, 255))
    d = ImageDraw.Draw(g)
    d.text((6, 8), title, fill=(0, 0, 0))
    for n, (t, lab) in enumerate(zip(tiles, labels)):
        x, y = (n % cols) * size, 30 + (n // cols) * H
        g.paste(t.resize((size, size)), (x, y))
        d.text((x + 6, y + size + 4), lab, fill=(0, 0, 0))
    return g


def titled(path, title):
    im = Image.open(path).convert('RGB')
    g = Image.new('RGB', (im.width, im.height + 30), (255, 255, 255))
    g.paste(im, (0, 30))
    ImageDraw.Draw(g).text((6, 8), title, fill=(0, 0, 0))
    return g


def input_hotspots(model, idx, chans):
    """mixed4d argmax cell of each requested channel on each photo (thumbs, so near the recorded values)."""
    got = {}

    def grab(_m, _i, out):
        got['a'] = out.detach()

    def stop(_m, _i, out):
        raise Stop

    model.mixed4d.register_forward_hook(grab)
    model.mixed4e_1x1_pre_relu_conv.register_forward_hook(stop)
    x = torch.from_numpy(np.stack([np.asarray(whole(i), dtype=np.float32) for i in idx])).permute(0, 3, 1, 2) - 117
    try:
        with torch.no_grad():
            model(x)
    except Stop:
        pass
    a = got['a'][torch.arange(len(idx)), torch.tensor(chans)].flatten(1).argmax(1).numpy()
    return [divmod(int(k), 14) for k in a]


def main():
    z = np.load(B / 'img/acts.npz')
    pre, cls = z['maps'].astype(np.float32), z['classes']
    names = json.loads((B / 'img/names.json').read_text())
    pos = np.maximum(pre, 0).reshape(len(pre), -1).max(1)
    card = np.where(np.array([zlib.crc32(n.encode()) % 10 < 7 for n in names]))[0]
    held = next(iter(json.loads((B / 'KEY_img.json').read_text()).values()))['heldout']
    cmax = float(pos[card].max())
    hot = lambda i: np.unravel_index(pre[i].argmax(), (14, 14))
    shade = lambda i: np.maximum(pre[i], 0) / cmax

    # grid 3: same-class pairs
    cm = np.array([pos[card[cls[card] == c]].mean() if (cls[card] == c).any() else -1 for c in range(1000)])
    rank = np.argsort(np.argsort(-cm))
    tiles, labels = [], []
    for c in np.argsort(-cm)[:12]:
        m = card[cls[card] == c]
        hi, lo = m[pos[m].argmax()], m[pos[m].argmin()]
        tiles += [tile(hi, hot(hi), shade(hi)), whole(lo)]
        labels += [f'strongest: {pos[hi]:.0f}', f'weakest, same class: {pos[lo]:.0f}']
    pairs = grid(tiles, labels, 'Grid 3: same-class pairs. Left: the class photo with the highest activation (cropped). '
                 'Right: the class photo with the lowest (whole photo).', cols=4)

    # grid 4: off-class positives
    off = card[(pos[card] >= 0.3 * cmax) & (rank[cls[card]] >= 200)]
    off = off[np.argsort(-pos[off])][:15]
    offg = grid([tile(i, hot(i), shade(i)) for i in off], [f'{pos[i]:.0f} (class average {cm[cls[i]]:.0f})' for i in off],
                'Grid 4: strongly active photos from classes on which the unit is usually weak.', cols=5)
    print('off-class positives:', len(off), 'pair classes:', list(np.argsort(-cm)[:12]))

    # grid 5: the main inputs
    c = np.load(J / 'runs/circuit/circuit.npz')
    smax, con, w = c['smax'].astype(np.float32), c['contrib'].astype(np.float32), c['w']
    fire = card[pos[card] >= 0.3 * cmax]
    mean_con = con[fire].mean(0)
    main_in = list(np.argsort(-mean_con)[:6])
    neg_in = list(np.argsort(w * smax[card].mean(0))[:2])
    share = mean_con / mean_con[mean_con > 0].sum()
    rows = [(ch, card[np.argsort(-smax[card, ch])[:6]]) for ch in main_in + neg_in]
    cells = iter(input_hotspots(load_model(J / 'raw/weights/inception5h.pth'), [i for _, idx in rows for i in idx],
                                [ch for ch, idx in rows for _ in idx]))
    tiles, labels = [], []
    for ch, idx in rows:
        tiles += [tile(i, next(cells)) for i in idx]
        what = (f'{share[ch]:.0%} of the drive when the unit fires' if ch in main_in else 'pushes the unit down')
        labels += [f'input {ch}, w={w[ch]:+.3f}' if k == 0 else (what if k == 1 else '') for k in range(len(idx))]
    inputs = grid(tiles, labels, 'Grid 5: main inputs from the layer below. Each row: one input unit\'s own six most '
                  'active photos, cropped where it is most active.', cols=6, size=160)
    print('inputs:', [(int(ch), round(float(w[ch]), 3)) for ch, _ in rows])

    assert not {h['thumb'] for h in held} & set(card.tolist())  # held-out come from the other pool
    key, ids = {}, random.Random(653).sample(range(100, 999), 2)
    base = ('Recorded from one unit (a single neuron) in an image classification network, run over several thousand '
            'natural photos. Activations are raw values; the strongest on these photos is '
            f'{cmax:.0f}.\n'
            'Grid 1: the 20 photos on which the unit is most active, strongest first, each cropped around the spot where it '
            'is most active, with red shading where it responds.\n'
            'Grid 2: the 20 photos containing the spots that push the unit most strongly negative (its value before the '
            'ReLU, averaged over a small area), most negative first, each cropped around that spot, with blue shading.\n'
            'Grid 3: pairs of photos with the same label in the dataset. For each of the 12 labels on which the unit is most '
            'active on average, the label\'s photo with the highest activation (cropped, red shading) beside its photo '
            'with the lowest activation (the whole photo).\n'
            'Grid 4: photos on which the unit is strongly active (at least 30% of its maximum) although their label is not '
            'among the 200 labels (of 1000) on which it is most active on average; each with its label\'s average.\n')
    circuit = ('Grid 5: the unit\'s value before the ReLU is exactly a weighted sum of the 528 units in the layer below it, '
               'plus a constant. Each row is one of those input units: the six that contribute most when the unit fires '
               '(with each one\'s share of the positive contribution), then the two whose weighted input pushes the unit '
               'most negative on average. Each row shows that input unit\'s own six most active photos, cropped around '
               'where the input unit is most active, with its weight.\n')
    rate = f'Across all photos the unit is above 30% of its maximum on {np.mean(pos[card] >= 0.3 * cmax) * 100:.1f}% of them.\n'
    for fmt, num in zip(('contrast', 'circuit'), ids):
        cid = f'U-{num}'
        out = {'a_top': titled(B / 'cards/U-860_pos.jpg', 'Grid 1: strongest photos.'),
               'b_trough': titled(B / 'cards/U-860_neg.jpg', 'Grid 2: most negative spots.'),
               'c_pairs': pairs, 'd_offclass': offg}
        if fmt == 'circuit':
            out['e_inputs'] = inputs
        for k, im in out.items():
            im.save(B / 'cards' / f'{cid}_{k}.jpg', quality=88)
        (B / 'cards' / f'{cid}.md').write_text(f'# Unit {cid}\n\n' + base + (circuit if fmt == 'circuit' else '') + rate)
        (B / 'heldout' / f'{cid}.json').write_text(json.dumps([{'id': t['id'], 'image': f'img_{t["id"]:02d}.jpg'} for t in held]))
        key[cid] = {'unit': 'inceptionv1_mixed4e_55', 'format': fmt, 'display': 'v3', 'truth_class': None,
                    'images': [f'{cid}_{k}.jpg' for k in out], 'card_max': cmax, 'heldout': held}
        print(cid, fmt, list(out))
    (B / 'KEY_img_v3.json').write_text(json.dumps(key, indent=1))


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
