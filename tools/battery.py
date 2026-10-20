"""Transformation battery over every channel of InceptionV1 mixed4e.

For each of the 832 channels c and each transform T, R[c, T] is the nanmedian over c's top-20 pool photos x of
a_c(T(x)) / a_c(x), with a_c the spatial max of the post-ReLU 14x14 map (NaN where a_c(x) <= 0).
Pool: names[i] with crc32 % 10 < 7. Input range follows img_acts.run_batch: pixels (0-255) - 117.
Outputs and caches go to runs/battery/.

usage: python -I tools/battery.py [--selftest] [--report 1,2,3]
"""
import json
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

J = Path(__file__).resolve().parent.parent
OUT = J / 'runs/battery'
K = 20
BATCH = 64
NCH = 832
PERM = [10, 7, 13, 0, 15, 2, 8, 5, 3, 12, 1, 14, 6, 9, 4, 11]


def _arr(img):
    return np.asarray(img, dtype=np.float64)


def _u8(y):
    return Image.fromarray(np.clip(np.round(y), 0, 255).astype(np.uint8))


def _pil(a):
    return Image.fromarray(np.ascontiguousarray(a))


def grayscale(img):
    return img.convert('L').convert('RGB')


def low_pass(img):
    x = _arr(img)
    return _u8(ndimage.gaussian_filter(x, sigma=(4, 4, 0), mode='reflect'))


def high_pass(img):
    x = _arr(img)
    return _u8(x - ndimage.gaussian_filter(x, sigma=(4, 4, 0), mode='reflect') + x.mean(axis=(0, 1), keepdims=True))


def phase_scramble(img):
    x = _arr(img)
    rng = np.random.default_rng(0)
    theta = np.angle(np.fft.fft2(rng.standard_normal((224, 224))))
    theta[0, 0] = 0
    y = np.empty_like(x)
    for c in range(3):
        F = np.fft.fft2(x[:, :, c])
        y[:, :, c] = np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * (np.angle(F) + theta))))
    return _u8(y)


def _tile(a, t):
    r, c = divmod(t, 4)
    return a[56 * r:56 * r + 56, 56 * c:56 * c + 56]


def patch_shuffle_4x4(img):
    x = np.asarray(img)
    y = np.empty_like(x)
    for p in range(16):
        _tile(y, p)[...] = _tile(x, PERM[p])
    return _pil(y)


def rotate_90(img):
    return _pil(np.rot90(np.asarray(img), k=1, axes=(0, 1)))


def edge_map(img):
    g = ndimage.gaussian_filter(np.asarray(img.convert('L'), dtype=np.float64), sigma=2)
    mag = np.hypot(ndimage.sobel(g, axis=1), ndimage.sobel(g, axis=0))
    e = mag > np.percentile(mag, 90)
    out = np.where(e, 0, 255).astype(np.uint8)
    return _pil(np.repeat(out[:, :, None], 3, axis=2))


def luminance_polarity_inversion(img):
    x = _arr(img)
    L = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
    return _u8(x + (255 - 2 * L)[:, :, None])


def rotate_180(img):
    return _pil(np.rot90(np.asarray(img), k=2, axes=(0, 1)))


def mirror(img):
    return _pil(np.asarray(img)[:, ::-1, :])


TRANSFORMS = {
    'grayscale': grayscale,
    'low_pass': low_pass,
    'high_pass': high_pass,
    'phase_scramble': phase_scramble,
    'patch_shuffle_4x4': patch_shuffle_4x4,
    'rotate_90': rotate_90,
    'edge_map': edge_map,
    'luminance_polarity_inversion': luminance_polarity_inversion,
    'rotate_180': rotate_180,
    'mirror': mirror,
}


def decide(r):
    """Neutral label on one retention value: R < 0.70 ATTENUATED, R > 1.30 AMPLIFIED, else RETAINED."""
    if np.isnan(r):
        return 'UNDEFINED'
    return 'ATTENUATED' if r < 0.70 else 'AMPLIFIED' if r > 1.30 else 'RETAINED'


def _bank():
    p = str(J / 'bank')
    if p not in sys.path:
        sys.path.insert(0, p)
    import img_acts
    return img_acts


def pool(names):
    return np.array([i for i, n in enumerate(names) if zlib.crc32(n.encode()) % 10 < 7], dtype=np.int64)


def photo(i):
    with Image.open(J / 'bank/img/thumbs' / f'{int(i):05d}.jpg') as im:
        img = im.convert('RGB')
    assert img.size == (224, 224), (i, img.size)
    return img


def acts(model, ids, fn=None):
    """a(x) or a(fn(x)) for photo ids: spatial max of every mixed4e channel, (len(ids), 832) float32."""
    import torch
    Stop = _bank().Stop
    got = {}

    def keep(_m, _i, out):
        got['a'] = out.detach().clone()

    def stop(_m, _i, _o):
        raise Stop

    hs = [model.mixed4e.register_forward_hook(keep), model.maxpool10.register_forward_hook(stop)]
    res = []
    try:
        for s in range(0, len(ids), BATCH):
            imgs = [photo(i) for i in ids[s:s + BATCH]]
            if fn is not None:
                imgs = [fn(im) for im in imgs]
            x = torch.from_numpy(np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in imgs])).permute(0, 3, 1, 2)
            x = x * 255 - 117
            got.clear()
            try:
                with torch.no_grad():
                    model(x)
            except Stop:
                pass
            a = got['a']
            assert tuple(a.shape[1:]) == (NCH, 14, 14), tuple(a.shape)
            res.append(a.amax(dim=(2, 3)).numpy().astype(np.float32))
    finally:
        for h in hs:
            h.remove()
    return np.concatenate(res)


def ratio(t, o):
    """a_c(T x) / a_c(x), NaN where a_c(x) <= 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(o > 0, t / np.where(o > 0, o, 1), np.nan)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    names = json.loads((J / 'bank/img/names.json').read_text())
    P = pool(names)
    held = []

    def model():
        if not held:
            import torch
            torch.set_num_threads(4)
            held.append(_bank().load_model(J / 'raw/weights/inception5h.pth'))
        return held[0]

    # 1. a(x) on the pool
    if (OUT / 'orig.npy').exists() and (OUT / 'pool.npy').exists():
        assert np.array_equal(np.load(OUT / 'pool.npy'), P), 'cached pool.npy differs from P'
        orig = np.load(OUT / 'orig.npy')
        print('orig: cached', orig.shape, flush=True)
    else:
        print('orig: running on', len(P), 'photos', flush=True)
        orig = acts(model(), P)
        np.save(OUT / 'pool.npy', P)
        np.save(OUT / 'orig.npy', orig)
    assert orig.shape == (len(P), NCH), orig.shape

    # 2. top-20 per channel, U = union
    order = np.argsort(-orig, axis=0, kind='stable')[:K].T  # (832, 20) positions in P, descending
    topk = P[order]
    np.save(OUT / 'topk.npy', topk)
    U = np.unique(topk)
    if (OUT / 'U.npy').exists():
        assert np.array_equal(np.load(OUT / 'U.npy'), U), 'cached U.npy differs from U'
    else:
        np.save(OUT / 'U.npy', U)

    # 3. a(T x) on U for each transform
    t = {}
    for k, (name, fn) in enumerate(TRANSFORMS.items()):
        f = OUT / f't_{name}.npy'
        if f.exists():
            t[name] = np.load(f)
            print(f'[{k + 1}/{len(TRANSFORMS)}] {name}: cached', flush=True)
        else:
            print(f'[{k + 1}/{len(TRANSFORMS)}] {name}: running on {len(U)} photos', flush=True)
            t[name] = acts(model(), U, fn)
            np.save(f, t[name])
            print(f'[{k + 1}/{len(TRANSFORMS)}] {name}: done', flush=True)
        assert t[name].shape == (len(U), NCH), (name, t[name].shape)

    # 4. retention
    o = np.take_along_axis(orig.T, order, axis=1)
    upos = np.searchsorted(U, topk)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        R = np.stack([np.nanmedian(ratio(np.take_along_axis(t[n].T, upos, axis=1), o), axis=1)
                      for n in TRANSFORMS], axis=1)
        med = [float(np.nanmedian(R[:, j])) for j in range(len(TRANSFORMS))]
    np.save(OUT / 'retention.npy', R)
    (OUT / 'transforms.json').write_text(json.dumps(list(TRANSFORMS)))
    print('retention', R.shape, '->', OUT / 'retention.npy')
    for j, name in enumerate(TRANSFORMS):
        print(f'{name:30s} n={int(np.isfinite(R[:, j]).sum())} layer_median={med[j]:.2f} -> {decide(med[j])}')
    return dict(P=P, orig=orig, order=order, topk=topk, U=U, t=t, R=R)


def _wrap(name, width=16):
    lines, cur = [], ''
    for w in name.split('_'):
        nxt = f'{cur}_{w}' if cur else w
        if cur and len(nxt) > width:
            lines.append(cur + '_')
            cur = w
        else:
            cur = nxt
    return '\n'.join(lines + [cur])


def sheet(c, d, n=6, tile=112, head=40, lab=14):
    cols = ['original'] + list(TRANSFORMS)
    im = Image.new('RGB', (tile * len(cols), head + n * (tile + lab)), 'white')
    dr = ImageDraw.Draw(im)
    for j, name in enumerate(cols):
        dr.multiline_text((j * tile + 2, 2), _wrap(name), fill='black', spacing=1)
    ids = d['topk'][c, :n]
    o = d['orig'][d['order'][c, :n], c]
    upos = np.searchsorted(d['U'], ids)
    for r, i in enumerate(ids):
        x = photo(i)
        y0 = head + r * (tile + lab)
        vals = [1.0] + [float(ratio(d['t'][nm][upos[r], c], o[r])) for nm in TRANSFORMS]
        imgs = [x] + [fn(x) for fn in TRANSFORMS.values()]
        for j, (img, v) in enumerate(zip(imgs, vals)):
            im.paste(img.resize((tile, tile), Image.LANCZOS), (j * tile, y0))
            dr.text((j * tile + 2, y0 + tile + 1), f'{v:.2f}', fill='black')
    path = OUT / f'sheet_{c}.jpg'
    im.save(path, quality=90)
    return path


def report(chs, d):
    R = d['R']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for c in chs:
            for j, name in enumerate(TRANSFORMS):
                col = R[:, j]
                v = col[~np.isnan(col)]
                r = float(R[c, j])
                pct = 100 * float(np.mean(v <= r)) if not np.isnan(r) else float('nan')
                print(f'{c:4d} {name:30s} R={r:.2f} pct={pct:.0f} layer_median={float(np.nanmedian(col)):.2f}'
                      f' -> {decide(r)}')
            print('sheet', sheet(c, d))


def selftest():
    rng = np.random.default_rng(12345)
    a = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(a)
    out = {}
    for name, fn in TRANSFORMS.items():
        y1, y2 = fn(img), fn(img)
        assert y1.size == (224, 224) and y1.mode == 'RGB', (name, y1.size, y1.mode)
        assert np.array_equal(np.asarray(y1), np.asarray(y2)), name + ': two calls differ'
        out[name] = np.asarray(y1)

    g = out['grayscale']
    assert (g[:, :, 0] == g[:, :, 1]).all() and (g[:, :, 1] == g[:, :, 2]).all(), 'grayscale: R, G, B differ'

    def times(fn, k):
        y = img
        for _ in range(k):
            y = fn(y)
        return np.asarray(y)

    assert np.array_equal(times(rotate_90, 4), a), 'rotate_90 x4'
    assert np.array_equal(times(rotate_180, 2), a), 'rotate_180 x2'
    assert np.array_equal(times(mirror, 2), a), 'mirror x2'

    s = out['patch_shuffle_4x4']
    assert np.array_equal(np.sort(s, axis=None), np.sort(a, axis=None)), 'patch_shuffle: pixel values changed'
    for p in range(16):
        assert not np.array_equal(_tile(s, p), _tile(a, p)), f'patch_shuffle: tile {p} not moved'

    e = out['edge_map']
    black, white = (e == 0).all(axis=2), (e == 255).all(axis=2)
    assert (black | white).all(), 'edge_map: not pure black/white'
    assert abs(black.mean() - 0.10) <= 0.01, ('edge_map black fraction', black.mean())

    # low-saturation image in 64-191 so none of the tests below clip
    base = rng.integers(96, 160, (224, 224, 1))
    m = (base + rng.integers(-16, 17, (224, 224, 3))).astype(np.uint8)
    assert m.min() >= 64 and m.max() <= 191
    mi = Image.fromarray(m)
    x = m.astype(np.float64)

    lp, hp = _arr(low_pass(mi)), _arr(high_pass(mi))
    d = np.abs(lp + hp - (x + x.mean(axis=(0, 1), keepdims=True))).max()
    assert d <= 1 + 1e-6, ('low_pass + high_pass vs x + mean', d)

    d = np.abs(_arr(luminance_polarity_inversion(luminance_polarity_inversion(mi))) - x).max()
    assert d <= 1, ('luminance_polarity_inversion x2', d)

    ps = _arr(phase_scramble(mi))
    for c in range(3):
        a0 = np.abs(np.fft.fft2(x[:, :, c])).ravel()[1:]
        a1 = np.abs(np.fft.fft2(ps[:, :, c])).ravel()[1:]
        err = np.linalg.norm(a1 - a0) / np.linalg.norm(a0)
        assert err <= 0.02, ('phase_scramble amplitude', c, err)
    print('selftest ok')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    d = run()
    if '--report' in sys.argv:
        report([int(c) for c in sys.argv[sys.argv.index('--report') + 1].split(',')], d)


if __name__ == '__main__':
    main()
