"""Synthetic curve stimuli: which channels of an InceptionV1 layer respond to curves?

Stimuli, 224x224 RGB, white background, black stroke 3 px wide (drawn at 4x as a distance field, LANCZOS down).
Coordinates are continuous pixel units (pixel i spans [i, i + 1)), so the image centre is (112, 112).
  curves: 90-degree arcs of radius r in {16, 32, 48}; the arc midpoint is the image centre and the circle centre lies
          r px from it in direction theta (0 = right, counter-clockwise on screen), theta = 0, 15, ..., 345;
  lines:  50 px segments centred on the image centre at angle 0, 15, ..., 165 (same angle convention);
  blank:  white.
a_c(x) is the spatial max of channel c of the post-ReLU output of --layer (default mixed3b).
Input range follows tools/battery.py: pixels (0-255) - 117. Outputs go to runs/curves/.

usage: python -I tools/curves.py [--layer mixed3b] [--report 1,2] [--sheet 3] [--selftest]
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

J = Path(__file__).resolve().parent.parent
OUT = J / 'runs/curves'
BATCH = 64
N = 224
SS = 4                         # draw at 4x
WIDTH = 3                      # stroke width after downsampling, px
RADII = (16, 32, 48)
R32 = RADII.index(32)
THETAS = np.arange(0, 360, 15)
ANGLES = np.arange(0, 180, 15)
LINE_LEN = 50
NC = len(RADII) * len(THETAS)  # curve stimuli come first, r-major: index = ri * 24 + ti


def _grid():
    u = (np.arange(N * SS) + 0.5) / SS  # 4x pixel centres in 224-px units
    return np.meshgrid(u, u)            # x, y (y down)


def _render(dist):
    """Black where the 4x pixel centre is within WIDTH / 2 of the stroke, LANCZOS down to 224x224 RGB."""
    big = Image.fromarray(np.where(dist <= WIDTH / 2, 0, 255).astype(np.uint8))
    return big.resize((N, N), Image.LANCZOS).convert('RGB')


def curve(r, theta):
    x, y = _grid()
    t = np.deg2rad(theta)
    cx, cy = N / 2 + r * np.cos(t), N / 2 - r * np.sin(t)
    dx, dy = x - cx, y - cy
    d = (np.arctan2(-dy, dx) - t) % (2 * np.pi) - np.pi  # angle from the arc's midpoint direction t + pi
    ends = [(cx + r * np.cos(t + np.pi + s), cy - r * np.sin(t + np.pi + s)) for s in (-np.pi / 4, np.pi / 4)]
    to_end = np.minimum(*[np.hypot(x - ex, y - ey) for ex, ey in ends])
    return _render(np.where(np.abs(d) <= np.pi / 4, np.abs(np.hypot(dx, dy) - r), to_end))


def line(angle):
    x, y = _grid()
    a = np.deg2rad(angle)
    ux, uy = np.cos(a), -np.sin(a)
    px, py = x - N / 2, y - N / 2
    s = np.clip(px * ux + py * uy, -LINE_LEN / 2, LINE_LEN / 2)
    return _render(np.hypot(px - s * ux, py - s * uy))


def blank():
    return Image.new('RGB', (N, N), 'white')


def stimuli():
    """72 curves (r-major), 12 lines, 1 blank."""
    return [curve(r, t) for r in RADII for t in THETAS] + [line(a) for a in ANGLES] + [blank()]


def _bank():
    p = str(J / 'bank')
    if p not in sys.path:
        sys.path.insert(0, p)
    import img_acts
    return img_acts


def acts(model, layer, imgs):
    """Spatial max of every channel of the post-ReLU output of `layer`, (len(imgs), C)."""
    import torch
    Stop = _bank().Stop
    got = {}

    def keep(_m, _i, out):
        got['a'] = out.detach().clone()
        raise Stop

    h = getattr(model, layer).register_forward_hook(keep)
    res = []
    try:
        for s in range(0, len(imgs), BATCH):
            x = torch.from_numpy(np.stack([np.asarray(im, dtype=np.float32) / 255.0
                                           for im in imgs[s:s + BATCH]])).permute(0, 3, 1, 2)
            x = x * 255 - 117
            got.clear()
            try:
                with torch.no_grad():
                    model(x)
            except Stop:
                pass
            a = got['a']
            assert a.ndim == 4 and float(a.min()) >= 0, (tuple(a.shape), float(a.min()))
            res.append(a.amax(dim=(2, 3)).numpy().astype(np.float32))
    finally:
        h.remove()
    return np.concatenate(res)


def osi(t):
    """|sum_k t_k exp(i theta_k)| / sum_k t_k over axis 0, t clipped at 0; NaN where sum_k t_k = 0."""
    t = np.clip(t, 0, None)
    z = np.abs(np.tensordot(np.exp(1j * np.deg2rad(THETAS)), t, axes=(0, 0)))
    s = t.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(s > 0, z / np.where(s > 0, s, 1), np.nan)


def measure(A):
    A = np.asarray(A, dtype=np.float64)
    cur = A[:NC].reshape(len(RADII), len(THETAS), -1)
    lines, bl = A[NC:NC + len(ANGLES)], A[NC + len(ANGLES)]
    cp, lp = cur.max(axis=(0, 1)), lines.max(axis=0)
    tun = cur[R32]
    k = tun.argmax(axis=0)
    best = np.array(RADII)[cur[:, k, np.arange(A.shape[1])].argmax(axis=0)]
    return dict(acts=A, curves=cur, lines=lines, blank=bl, curve_peak=cp, line_peak=lp, ratio=cp / (lp + 1e-6),
                tuning=tun, pref=THETAS[k], osi=osi(tun), best_r=best)


def decide(r):
    """Neutral label on one ratio value: r < 0.70 LINE>CURVE, r > 1.30 CURVE>LINE, else SIMILAR."""
    if np.isnan(r):
        return 'UNDEFINED'
    return 'LINE>CURVE' if r < 0.70 else 'CURVE>LINE' if r > 1.30 else 'SIMILAR'


def row(d, c):
    cp, lp, r = d['curve_peak'][c], d['line_peak'][c], d['ratio'][c]
    lab = decide(r) if max(cp, lp) > 0 else 'UNDEFINED'
    return (f'{c:4d} curve_peak={cp:.3f} line_peak={lp:.3f} blank={d["blank"][c]:.3f} ratio={r:.2f}'
            f' pref={int(d["pref"][c]):3d} osi={d["osi"][c]:.2f} best_r={int(d["best_r"][c]):2d} -> {lab}')


def pct(col, v):
    col = col[np.isfinite(col)]
    return 100 * float(np.mean(col <= v)) if np.isfinite(v) else float('nan')


def report(d, layer, chs):
    cp, r, o = d['curve_peak'], d['ratio'], d['osi']
    q = float(np.percentile(cp, 75))
    el = np.flatnonzero(cp > q)
    top = el[np.argsort(-r[el], kind='stable')][:20]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mo = float(np.nanmedian(o))
    mr = float(np.median(r))
    out = [f'layer {layer}: {cp.size} channels; {NC} curves, {len(ANGLES)} lines, 1 blank',
           f'layer median curve_peak={float(np.median(cp)):.3f} ratio={mr:.2f} -> {decide(mr)}'
           f' osi={mo:.2f} (osi defined for {int(np.isfinite(o).sum())})',
           f'top {len(top)} by ratio among {len(el)} channels with curve_peak > p75 = {q:.3f}']
    out += [row(d, c) for c in top]
    if chs:
        out.append('requested channels (pct = % of layer channels <= value)')
    out += [row(d, c) + f' | pct curve_peak={pct(cp, cp[c]):.0f} ratio={pct(r, r[c]):.0f} osi={pct(o, o[c]):.0f}'
            for c in chs]
    text = '\n'.join(out)
    print(text)
    (OUT / 'report.txt').write_text(text + '\n')


def sheet(c, d, imgs, tile=64, lab=14):
    im = Image.new('RGB', (tile * len(THETAS), tile + lab), 'white')
    dr = ImageDraw.Draw(im)
    cp = float(d['curve_peak'][c])
    for k in range(len(THETAS)):
        v = float(d['tuning'][k, c]) / cp if cp > 0 else float('nan')
        im.paste(imgs[R32 * len(THETAS) + k].resize((tile, tile), Image.LANCZOS), (k * tile, 0))
        dr.text((k * tile + 2, tile + 1), f'{v:.2f}', fill='black')
    path = OUT / f'sheet_{c}.jpg'
    im.save(path, quality=90)
    return path


def run(layer):
    assert re.fullmatch(r'mixed\d[a-z]', layer), layer
    import torch
    torch.set_num_threads(4)
    model = _bank().load_model(J / 'raw/weights/inception5h.pth')
    assert hasattr(model, layer), layer
    imgs = stimuli()
    d = measure(acts(model, layer, imgs))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f'{layer}.npz', radii=np.array(RADII), thetas=THETAS, line_angles=ANGLES, **d)
    print('saved', OUT / f'{layer}.npz')
    return d, imgs


def selftest():
    imgs = stimuli()
    assert len(imgs) == NC + len(ANGLES) + 1, len(imgs)
    for im in imgs:
        a = np.asarray(im)
        assert im.size == (N, N) and im.mode == 'RGB' and a.dtype == np.uint8, (im.size, im.mode, a.dtype)
        assert a.min() >= 0 and a.max() == 255 and (a[[0, -1]] == 255).all() and (a[:, [0, -1]] == 255).all()
    assert (np.asarray(imgs[-1]) == 255).all(), 'blank not white'
    for im in imgs[:-1]:
        assert np.asarray(im).min() < 64, 'stroke not dark'

    for r in RADII:  # theta = 0: circle centre to the right of the midpoint
        ys, xs = np.nonzero(np.asarray(curve(r, 0))[:, :, 0] < 128)
        assert abs(int(xs.min()) - 112) <= 2, ('leftmost dark column', r, int(xs.min()))
        assert xs.mean() > 112, ('mean x of dark pixels', r, float(xs.mean()))

    for a in ANGLES:
        g = np.asarray(line(a)).astype(int)
        assert np.abs(g - np.rot90(g, 2, axes=(0, 1))).max() <= 1, ('line not 180-symmetric', a)

    one = np.zeros(len(THETAS))
    one[5] = 2.0
    assert abs(float(osi(one)) - 1) < 1e-9, float(osi(one))
    assert abs(float(osi(np.ones(len(THETAS))))) < 1e-9, float(osi(np.ones(len(THETAS))))

    A = np.zeros((len(imgs), 2))
    A[R32 * len(THETAS) + 6, 0] = 2.0  # r = 32, theta = 90
    A[2 * len(THETAS) + 6, 0] = 3.0    # r = 48, theta = 90
    A[NC:NC + len(ANGLES), 0] = 1.0
    d = measure(A)
    assert d['curve_peak'][0] == 3 and d['line_peak'][0] == 1 and d['pref'][0] == 90 and d['best_r'][0] == 48
    assert abs(d['osi'][0] - 1) < 1e-9 and np.isnan(d['osi'][1]) and d['ratio'][1] == 0
    print('selftest ok')


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    if '--selftest' in sys.argv:
        return selftest()
    layer = _arg('--layer', 'mixed3b')
    d, imgs = run(layer)
    report(d, layer, [int(c) for c in _arg('--report', '').split(',') if c])
    if '--sheet' in sys.argv:
        print('sheet', sheet(int(_arg('--sheet')), d, imgs))


if __name__ == '__main__':
    main()
