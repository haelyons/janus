"""Pose test on Zoom In's superposition circuit: car neuron 4c:447 and its readers in mixed4d. Received reading:
the car is stored in dog neurons. Essay's reading: the readers are generalized pose neurons, shared by cars and dogs.

Label-free check: a pose neuron's top photos all face one way, so mirroring them should cut its activation for cars
and for dogs alike; a car or dog detector shouldn't care much.
1. Readers: mixed4d channels ranked by d(sum 4d_j)/d(4c_447) on car photos (an expanded weight through the block).
2. For each reader (and controls 4c:447, 4e:55): share of dogs among its top 50 car+dog photos, and the median
   flip drop 1 - a(mirror x)/a(x) over its top 20 car photos and top 20 dog photos (spatial max).

usage: python -I tools/pose.py   (writes runs/pose/report.txt, runs/pose/readers.jpg)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'bank'))
from img_acts import Stop, load_model
from img_cards import CARS

DOGS = set(range(151, 269))
CAR = 447


def to_x(ims):
    return torch.from_numpy(np.stack([np.asarray(i, dtype=np.float32) for i in ims])).permute(0, 3, 1, 2) - 117


def record(model, ims):
    """Spatial max of mixed4c, mixed4d, mixed4e for each photo, in batches."""
    got, out = {}, []
    hs = [getattr(model, n).register_forward_hook(lambda _m, _i, o, n=n: got.__setitem__(n, o.detach())) for n in
          ('mixed4c', 'mixed4d', 'mixed4e')]
    hs.append(model.maxpool10.register_forward_hook(lambda *_: (_ for _ in ()).throw(Stop)))
    for k in range(0, len(ims), 64):
        try:
            with torch.no_grad():
                model(to_x(ims[k:k + 64]))
        except Stop:
            pass
        out.append(torch.cat([got[n].flatten(2).max(2).values for n in ('mixed4c', 'mixed4d', 'mixed4e')], 1).numpy())
    for h in hs:
        h.remove()
    return np.concatenate(out)


def readers(model, ims):
    leaf = {}

    def make_leaf(_m, _i, o):
        leaf['x'] = o.detach().requires_grad_()
        return leaf['x']

    got = {}
    h1 = model.mixed4c.register_forward_hook(make_leaf)
    h2 = model.mixed4d.register_forward_hook(lambda _m, _i, o: got.__setitem__('d', o))
    h3 = model.mixed4e.register_forward_hook(lambda *_: (_ for _ in ()).throw(Stop))
    try:
        model(to_x(ims))
    except Stop:
        pass
    for h in (h1, h2, h3):
        h.remove()
    d = got['d']
    g = [torch.autograd.grad(d[:, j].sum(), leaf['x'], retain_graph=True)[0][:, CAR].sum().item() for j in range(d.shape[1])]
    return np.array(g)


def main():
    torch.set_num_threads(4)
    model = load_model(J / 'raw/weights/inception5h.pth')
    cls = np.load(J / 'bank/img/acts.npz')['classes']
    idx = np.where(np.isin(cls, list(CARS | DOGS)))[0]
    is_dog = np.isin(cls[idx], list(DOGS))
    thumbs = [Image.open(J / 'bank/img/thumbs' / f'{i:05d}.jpg').convert('RGB') for i in idx]
    a = record(model, thumbs)
    af = record(model, [ImageOps.mirror(t) for t in thumbs])
    car_ims = [thumbs[k] for k in np.where(~is_dog)[0][np.argsort(-a[~is_dog, CAR])[:48]]]
    g = readers(model, car_ims)
    rd = list(np.argsort(-g)[:8])
    cols = [('4c', CAR, CAR)] + [('4d', j, 512 + j) for j in rd] + [('4e', 55, 512 + 528 + 55)]
    lines = [f'{len(idx)} photos ({is_dog.sum()} dog, {(~is_dog).sum()} car). Readers of 4c:447 = top mixed4d channels by '
             'd(sum 4d_j)/d(4c_447) over its 48 strongest car photos.',
             'unit      grad   dogs-in-top50  flip-drop cars  flip-drop dogs']
    allfd = []
    for lay, ch, c in cols:
        top50 = np.argsort(-a[:, c])[:50]
        fd = lambda m: float(np.median(1 - af[m, c] / np.maximum(a[m, c], 1e-6)))
        tc = np.where(~is_dog)[0][np.argsort(-a[~is_dog, c])[:20]]
        td = np.where(is_dog)[0][np.argsort(-a[is_dog, c])[:20]]
        gr = f'{g[ch]:+.2f}' if lay == '4d' else '   - '
        lines.append(f'{lay}:{ch:<4d} {gr}   {is_dog[top50].mean():.2f}           {fd(tc):+.2f}            {fd(td):+.2f}')
    for c in range(512, 512 + 528):
        tc = np.where(~is_dog)[0][np.argsort(-a[~is_dog, c])[:20]]
        td = np.where(is_dog)[0][np.argsort(-a[is_dog, c])[:20]]
        allfd.append((np.median(1 - af[tc, c] / np.maximum(a[tc, c], 1e-6)), np.median(1 - af[td, c] / np.maximum(a[td, c], 1e-6))))
    allfd = np.array(allfd)
    lines.append(f'all 528 mixed4d channels: flip-drop cars median {np.median(allfd[:, 0]):+.2f} (90th pct '
                 f'{np.percentile(allfd[:, 0], 90):+.2f}), dogs median {np.median(allfd[:, 1]):+.2f} (90th pct {np.percentile(allfd[:, 1], 90):+.2f})')
    both = np.argsort(-np.minimum(allfd[:, 0], allfd[:, 1]))[:10]
    lines.append('mixed4d channels with the largest flip drop on both cars and dogs: ' +
                 ', '.join(f'{j} ({allfd[j, 0]:+.2f}/{allfd[j, 1]:+.2f})' for j in both))
    out = J / 'runs/pose'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    S = 112
    im = Image.new('RGB', (12 * S, len(rd) * (S + 14)), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    for r, j in enumerate(rd):
        c = 512 + j
        tc = np.where(~is_dog)[0][np.argsort(-a[~is_dog, c])[:6]]
        td = np.where(is_dog)[0][np.argsort(-a[is_dog, c])[:6]]
        for k, m in enumerate(list(tc) + list(td)):
            im.paste(thumbs[m].resize((S, S)), (k * S, r * (S + 14)))
        dr.text((4, r * (S + 14) + S + 1), f'4d:{j}  top cars | top dogs', fill=(0, 0, 0))
    im.save(out / 'readers.jpg', quality=85)


if __name__ == '__main__':
    main()
