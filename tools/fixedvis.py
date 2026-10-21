"""Feature visualisation of single units, plain and lightly biased toward the unit's natural input paths.

For a unit layer:ch, with input layer = the previous module (mixed4d for mixed4e, mixed4c for mixed4d):
  plain: maximise mean(a_ch(layer))
  fixed: maximise mean(a_ch(layer)) + lam * cos(m(T), m_top)
m(T) is the spatial mean of the input layer over the render (a vector over its channels), m_top the same vector
averaged over the unit's 50 strongest photos of bank/img/thumbs (ranked by the unit's spatial max; input as
tools/battery.py: pixels (0-255) - 117), lam = 0.1 x the unit's mean spatial-mean activation over those photos.
Seeds 0 and 1. Each saved PNG is measured like a photo (upsampled to 224 as lucent's render does) and then described
by a fresh, single-turn Claude call that sees only that image. Outputs go to runs/fixedvis/.

usage: python -I tools/fixedvis.py --units mixed4d:409,mixed4e:55,mixed4e:700 [--selftest]
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

J = Path(__file__).resolve().parent.parent
OUT = J / 'runs/fixedvis'
PREV = {'mixed4e': 'mixed4d', 'mixed4d': 'mixed4c'}
TOP = 50
LAM_FRAC = 0.1
SEEDS = (0, 1)
CONDS = ('plain', 'fixed')
TILE = 160
PROMPT = ('This is a synthetic image produced by optimisation. In one sentence, say what objects, animals, parts or '
          'shapes it most resembles, and from what viewpoint, if any. Reply with only JSON: {"description": "..."}')


def _tools():
    p = str(J / 'tools')
    if p not in sys.path:
        sys.path.insert(0, p)
    import battery
    import claude_call
    return battery, claude_call


def _lucent():
    import kornia
    import kornia.geometry.transform as kt
    kornia.get_rotation_matrix2d = kt.get_rotation_matrix2d
    kornia.warp_affine = kt.warp_affine
    from lucent.optvis import objectives, param, render, transform
    return objectives, param, render, transform


def parse_units(s):
    out = []
    for u in s.split(','):
        layer, ch = u.strip().split(':')
        assert layer in PREV, f'unit layer must be one of {list(PREV)}: {u}'
        out.append((layer, int(ch)))
    return out


def png_name(layer, ch, cond, seed):
    return f'{layer}_{ch}_{cond}_s{seed}.png'


def top_photos(mx, k=TOP):
    """positions of the k largest values, descending (stable)"""
    return np.argsort(-mx, kind='stable')[:k]


def lam_of(means):
    """0.1 x the unit's mean spatial-mean activation over its top photos"""
    return LAM_FRAC * float(np.mean(means))


def plain_value(a, ch):
    return a[:, ch].mean()


def cos_term(a_in, m_top):
    """cos(m(T), m_top), m(T) = spatial mean of the input-layer map (a vector over channels); mean over the batch"""
    import torch.nn.functional as F
    m = a_in.mean(dim=(2, 3))
    return F.cosine_similarity(m, m_top.reshape(1, -1).to(m), dim=1).mean()


def fixed_value(a, a_in, ch, lam, m_top):
    return plain_value(a, ch) + lam * cos_term(a_in, m_top)


def objective(cond, layer, ch, lam, m_top):
    objectives = _lucent()[0]
    if cond == 'plain':
        return objectives.channel(layer, ch)

    @objectives.wrap_objective()
    def fixed(layer, ch, prev, lam, m_top):
        return lambda T: -fixed_value(T(layer), T(prev), ch, lam, m_top)

    return fixed(layer, ch, PREV[layer], lam, m_top)


def to_input(arrs, dev):
    """RGB float arrays in [0, 1] -> network input: upsampled to 224 if smaller (bilinear, as lucent), x*255 - 117"""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(np.stack(arrs).astype(np.float32)).permute(0, 3, 1, 2).to(dev)
    if tuple(x.shape[2:]) != (224, 224):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
    return x * 255 - 117


def layer_stats(model, x, layers):
    """{layer: (spatial max, spatial mean)} of every channel of each module output, (N, C) numpy each"""
    import torch
    Stop = _tools()[0]._bank().Stop
    got = {}

    def keep(name):
        def f(_m, _i, out):
            got[name] = out.detach()
        return f

    def stop(_m, _i, _o):
        raise Stop

    hs = [getattr(model, l).register_forward_hook(keep(l)) for l in layers]
    hs.append(model.maxpool10.register_forward_hook(stop))
    try:
        with torch.no_grad():
            model(x)
    except Stop:
        pass
    finally:
        for h in hs:
            h.remove()
    return {l: (got[l].amax(dim=(2, 3)).cpu().numpy(), got[l].mean(dim=(2, 3)).cpu().numpy()) for l in layers}


def photo_stats(model, layers, dev):
    battery = _tools()[0]
    ids = np.array(sorted(int(p.stem) for p in (J / 'bank/img/thumbs').glob('*.jpg')), dtype=np.int64)
    acc = {l: ([], []) for l in layers}
    for s in range(0, len(ids), battery.BATCH):
        arrs = [np.asarray(battery.photo(i), dtype=np.float32) / 255.0 for i in ids[s:s + battery.BATCH]]
        for l, (mx, mn) in layer_stats(model, to_input(arrs, dev), layers).items():
            acc[l][0].append(mx)
            acc[l][1].append(mn)
        if (s // battery.BATCH) % 20 == 0:
            print(f'photos {s}/{len(ids)}', flush=True)
    return ids, {l: (np.concatenate(a), np.concatenate(b)) for l, (a, b) in acc.items()}


def _load(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0


def render_png(model, obj, seed, path):
    import torch
    _, param, render, transform = _lucent()
    torch.manual_seed(seed)
    np.random.seed(seed)  # lucent's jitter/scale/rotate draw from numpy
    imgs = render.render_vis(model, obj, param_f=lambda: param.image(128, fft=True, decorrelate=True),
                             transforms=transform.standard_transforms, thresholds=(256,), show_image=False,
                             progress=False)
    Image.fromarray(np.clip(np.round(imgs[-1][0] * 255), 0, 255).astype(np.uint8)).save(path)


def sheet(units, acts, head=16, lab=14):
    cols = [(c, s) for c in CONDS for s in SEEDS]
    im = Image.new('RGB', (TILE * len(cols), head + len(units) * (TILE + lab)), 'white')
    dr = ImageDraw.Draw(im)
    for j, (c, s) in enumerate(cols):
        dr.text((j * TILE + 2, 2), f'{c} s{s}', fill='black')
    for r, (layer, ch) in enumerate(units):
        y0 = head + r * (TILE + lab)
        for j, (c, s) in enumerate(cols):
            name = png_name(layer, ch, c, s)
            with Image.open(OUT / name) as t:
                im.paste(t.convert('RGB').resize((TILE, TILE), Image.LANCZOS), (j * TILE, y0))
            v = acts[f'{layer}:{ch}']['renders'][name]['max']
            dr.text((j * TILE + 2, y0 + TILE + 1), f'{layer}:{ch} act={v:.1f}', fill='black')
    im.save(OUT / 'sheet.jpg', quality=90)
    return OUT / 'sheet.jpg'


def describe(names):
    cc = _tools()[1]
    order = [names[i] for i in np.random.default_rng(0).permutation(len(names))]
    desc = {}
    for k, name in enumerate(order):
        data = base64.b64encode((OUT / name).read_bytes()).decode()
        blocks = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': data}},
                  {'type': 'text', 'text': PROMPT}]
        text = cc.converse('', [blocks])[0]['text']
        try:
            desc[name] = str(cc.parse_json(text)['description'])
        except (ValueError, KeyError, TypeError):
            desc[name] = 'UNPARSED: ' + text.strip()
        (OUT / 'descriptions.json').write_text(json.dumps(desc, indent=1))
        print(f'[{k + 1}/{len(order)}] described {name}', flush=True)
    return desc


def run(units):
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    battery = _tools()[0]
    _lucent()  # kornia patch before lucent.optvis is used
    torch.set_num_threads(4)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # lucent puts the image on this device
    model = battery._bank().load_model(J / 'raw/weights/inception5h.pth').to(dev)
    layers = sorted({l for l, _ in units} | {PREV[l] for l, _ in units})
    ids, st = photo_stats(model, layers, dev)

    acts, names = {}, []
    for layer, ch in units:
        prev, key = PREV[layer], f'{layer}:{ch}'
        top = top_photos(st[layer][0][:, ch])
        lam = lam_of(st[layer][1][top, ch])
        m_top = torch.tensor(st[prev][1][top].mean(axis=0), dtype=torch.float32, device=dev)
        u = acts[key] = dict(layer=layer, ch=ch, input_layer=prev, top50=ids[top].tolist(),
                             photo_max_mean=float(st[layer][0][top, ch].mean()),
                             photo_mean_mean=float(st[layer][1][top, ch].mean()), lam=lam, renders={})
        print(f'{key} input={prev} photo_max_mean={u["photo_max_mean"]:.2f} lam={lam:.4g}', flush=True)
        for cond in CONDS:
            for seed in SEEDS:
                name = png_name(layer, ch, cond, seed)
                render_png(model, objective(cond, layer, ch, lam, m_top), seed, OUT / name)
                r = layer_stats(model, to_input([_load(OUT / name)], dev), [layer, prev])
                cos = float(cos_term(torch.from_numpy(r[prev][1]).view(1, -1, 1, 1), m_top))
                u['renders'][name] = dict(condition=cond, seed=seed, max=float(r[layer][0][0, ch]),
                                          mean=float(r[layer][1][0, ch]), cos_input=cos)
                names.append(name)
                print(f'  {name} max={u["renders"][name]["max"]:.2f} cos_input={cos:.3f}', flush=True)
        for seed in SEEDS:
            p = u['renders'][png_name(layer, ch, 'plain', seed)]['max']
            f = u['renders'][png_name(layer, ch, 'fixed', seed)]['max']
            u[f'fixed_over_plain_s{seed}'] = rr = float(battery.ratio(np.float64(f), np.float64(p)))
            u[f'decision_s{seed}'] = battery.decide(rr)
        (OUT / 'acts.json').write_text(json.dumps(acts, indent=1))
    print('sheet', sheet(units, acts))

    desc = describe(names)
    print(f'\n{"unit":14s} {"condition":9s} {"seed":>4s} {"activation":>10s}  description')
    for key, u in acts.items():
        for name, r in u['renders'].items():
            print(f'{key:14s} {r["condition"]:9s} {r["seed"]:4d} {r["max"]:10.2f}  {desc[name]}')
    for key, u in acts.items():
        for seed in SEEDS:
            print(f'{key} s{seed}: fixed/plain spatial max = {u[f"fixed_over_plain_s{seed}"]:.2f}'
                  f' -> {u[f"decision_s{seed}"]}')


def selftest():
    import torch
    u = parse_units('mixed4d:409, mixed4e:55,mixed4e:700')
    assert u == [('mixed4d', 409), ('mixed4e', 55), ('mixed4e', 700)], u
    try:
        parse_units('mixed5a:3')
    except AssertionError:
        pass
    else:
        raise AssertionError('parse_units accepted a layer with no input layer')
    assert png_name('mixed4e', 55, 'fixed', 1) == 'mixed4e_55_fixed_s1.png'
    names = [png_name(l, c, k, s) for l, c in u for k in CONDS for s in SEEDS]
    assert len(set(names)) == 12, names

    # top-50 ranking and lam scaling
    rng = np.random.default_rng(0)
    mx = rng.permutation(200).astype(np.float64)
    top = top_photos(mx)
    assert np.array_equal(mx[top], np.arange(199, 149, -1)), 'top_photos: not the 50 largest, descending'
    means = rng.uniform(0, 5, 200)
    lam = lam_of(means[top])
    assert np.isclose(lam, 0.1 * means[top].mean()), lam
    assert np.isclose(lam_of(3 * means[top]), 3 * lam), 'lam not proportional to the activation scale'

    # cos term
    C, H = 16, 7
    m_top = torch.rand(C) + 0.1
    a_in = m_top.view(1, C, 1, 1) * (1 + torch.rand(1, 1, H, H))
    assert abs(float(cos_term(a_in, m_top)) - 1) < 1e-5, 'cos: parallel != 1'
    assert abs(float(cos_term(-a_in, m_top)) + 1) < 1e-5, 'cos: antiparallel != -1'
    e0, e1 = torch.eye(C)[0], torch.eye(C)[1]
    assert abs(float(cos_term(e1.view(1, C, 1, 1).expand(1, C, H, H), e0))) < 1e-6, 'cos: orthogonal != 0'
    a2 = torch.zeros(1, C, 2, 2)
    a2[0, 0, 0, :] = 1
    a2[0, 1, 1, :] = 1
    assert abs(float(cos_term(a2, e0 + e1)) - 1) < 1e-6, 'cos must use the spatial-mean vector'

    # fixed = plain + lam * cos; gradients
    a = torch.rand(1, 32, H, H, requires_grad=True)
    b = torch.rand(1, C, H, H, requires_grad=True)
    f = fixed_value(a, b, 5, 0.37, m_top)
    assert torch.isclose(plain_value(a, 5), a[0, 5].mean())
    assert torch.isclose(f - plain_value(a, 5), 0.37 * cos_term(b, m_top)), 'fixed - plain != lam * cos'
    f.backward()
    g = torch.zeros_like(a)
    g[0, 5] = 1 / (H * H)
    assert torch.allclose(a.grad, g), 'main term must touch only channel ch'
    assert float(b.grad.abs().sum()) > 0, 'cos term passes no gradient to the input layer'
    print('selftest ok')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    run(parse_units(sys.argv[sys.argv.index('--units') + 1]))


if __name__ == '__main__':
    main()
