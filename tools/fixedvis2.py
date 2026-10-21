"""Feature visualisation of single units with stronger pulls toward natural paths, and rendering "through" an input.

For a unit layer:ch, input layer = the previous module (as tools/fixedvis.py), seeds 0 and 1 per condition:
  pull5:     tools/fixedvis.py's "fixed" objective with lam x 50 (= 5 x the unit's mean spatial-mean activation over
             its top 50 photos)
  init:      plain objective; image parameterised directly as pixels (lucent param.image, fft=False,
             decorrelate=False, i.e. sigmoid(pixels)), initialised from the unit's strongest photo resized to 128 px
  through:i: maximise mean(unit) + w * mean(input-layer channel i), w = unit's mean spatial-mean activation over its
             top 50 photos / channel i's mean spatial-mean activation over the same photos
Plain renders are not redone (runs/fixedvis/). Each render's spatial max (and channel i's for through) is measured
as in tools/fixedvis.py; over_plain = render max / runs/fixedvis plain max (same unit, same seed), labelled with
tools/battery.py's decide (< 0.70 ATTENUATED, > 1.30 AMPLIFIED, else RETAINED; UNDEFINED if no plain value).
Outputs go to runs/fixedvis2/.

usage: python -I tools/fixedvis2.py --units mixed4d:409,mixed4e:55,mixed4e:700 \
           --through "mixed4d:409=447,393;mixed4e:55=493,352" [--selftest]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))  # -I drops the script directory from sys.path
import fixedvis as fv  # noqa: E402

OUT = J / 'runs/fixedvis2'
PLAIN = J / 'runs/fixedvis/acts.json'
PULL = 50
SIZE = 128
EPS = 1e-3  # pixel clip before logit; round(EPS * 255) == 0, so the init image saves back unchanged


def parse_through(s):
    """'layer:ch=i,j;layer:ch=k' -> {(layer, ch): [i, j], ...}"""
    out = {}
    for part in s.split(';'):
        if not part.strip():
            continue
        u, chans = part.split('=')
        (unit,) = fv.parse_units(u)
        out[unit] = [int(c) for c in chans.split(',')]
    return out


def conditions(unit, through):
    return ['pull5', 'init'] + [f'through:{i}' for i in through.get(unit, [])]


def png_name(layer, ch, cond, seed):
    return fv.png_name(layer, ch, cond.replace(':', '-'), seed)


def pull_lam(means):
    """fixedvis lam x 50"""
    return PULL * fv.lam_of(means)


def weight(unit_means, in_means):
    """unit's mean spatial-mean activation / channel i's, over the same photos"""
    d = float(np.mean(in_means))
    assert d > 0, f'input channel has mean activation {d} on the top photos'
    return float(np.mean(unit_means)) / d


def through_value(a, a_in, ch, i, w):
    return a[:, ch].mean() + w * a_in[:, i].mean()


def through_objective(layer, ch, i, w):
    objectives = fv._lucent()[0]

    @objectives.wrap_objective()
    def through(layer, ch, prev, i, w):
        return lambda T: -through_value(T(layer), T(prev), ch, i, w)

    return through(layer, ch, fv.PREV[layer], i, w)


def render_init(model, obj, seed, path, img):
    """plain pixel parameterisation (sigmoid of a pixel tensor), started from img (PIL, SIZE px)"""
    import torch
    _, param, render, transform = fv._lucent()
    x = np.clip(np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0, EPS, 1 - EPS)
    logit = torch.from_numpy(np.log(x / (1 - x))).permute(2, 0, 1)[None]

    def param_f():
        params, image_f = param.image(SIZE, fft=False, decorrelate=False)
        with torch.no_grad():
            params[0].copy_(logit.to(params[0]))
        return params, image_f

    torch.manual_seed(seed)
    np.random.seed(seed)
    imgs = render.render_vis(model, obj, param_f=param_f, transforms=transform.standard_transforms,
                             thresholds=(256,), show_image=False, progress=False)
    Image.fromarray(np.clip(np.round(imgs[-1][0] * 255), 0, 255).astype(np.uint8)).save(path)


def sheet(rows, acts, head=16, lab=14):
    T = fv.TILE
    im = Image.new('RGB', (T * len(fv.SEEDS), head + len(rows) * (T + 2 * lab)), 'white')
    dr = ImageDraw.Draw(im)
    for j, s in enumerate(fv.SEEDS):
        dr.text((j * T + 2, 2), f's{s}', fill='black')
    for r, (layer, ch, cond) in enumerate(rows):
        y0 = head + r * (T + 2 * lab)
        dr.text((2, y0 + 1), f'{layer}:{ch} {cond}', fill='black')
        for j, s in enumerate(fv.SEEDS):
            name = png_name(layer, ch, cond, s)
            with Image.open(OUT / name) as t:
                im.paste(t.convert('RGB').resize((T, T), Image.LANCZOS), (j * T, y0 + lab))
            v = acts[f'{layer}:{ch}']['renders'][name]['max']
            dr.text((j * T + 2, y0 + lab + T + 1), f'act={v:.1f}', fill='black')
    im.save(OUT / 'sheet.jpg', quality=90)
    return OUT / 'sheet.jpg'


def run(units, through):
    import torch
    assert set(through) <= set(units), 'every --through unit must be in --units'
    OUT.mkdir(parents=True, exist_ok=True)
    battery = fv._tools()[0]
    fv._lucent()  # kornia patch before lucent.optvis is used
    torch.set_num_threads(4)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = battery._bank().load_model(J / 'raw/weights/inception5h.pth').to(dev)
    layers = sorted({l for l, _ in units} | {fv.PREV[l] for l, _ in units})
    ids, st = fv.photo_stats(model, layers, dev)
    plain = json.loads(PLAIN.read_text()) if PLAIN.exists() else {}

    acts, names, rows = {}, [], []
    for layer, ch in units:
        prev, key = fv.PREV[layer], f'{layer}:{ch}'
        top = fv.top_photos(st[layer][0][:, ch])
        lam = pull_lam(st[layer][1][top, ch])
        m_top = torch.tensor(st[prev][1][top].mean(axis=0), dtype=torch.float32, device=dev)
        init_id = int(ids[top[0]])
        u = acts[key] = dict(layer=layer, ch=ch, input_layer=prev, top50=ids[top].tolist(),
                             photo_max_mean=float(st[layer][0][top, ch].mean()),
                             photo_mean_mean=float(st[layer][1][top, ch].mean()), pull5_lam=lam,
                             init_photo=init_id, through_w={}, renders={})
        print(f'{key} input={prev} pull5_lam={lam:.4g} init_photo={init_id}', flush=True)
        for cond in conditions((layer, ch), through):
            i = None
            if cond == 'pull5':
                obj = fv.objective('fixed', layer, ch, lam, m_top)
            elif cond == 'init':
                obj = fv.objective('plain', layer, ch, lam, m_top)
            else:
                i = int(cond.split(':')[1])
                assert i < st[prev][1].shape[1], f'{prev} has no channel {i}'
                w = u['through_w'][str(i)] = weight(st[layer][1][top, ch], st[prev][1][top, i])
                obj = through_objective(layer, ch, i, w)
            rows.append((layer, ch, cond))
            for seed in fv.SEEDS:
                name = png_name(layer, ch, cond, seed)
                if cond == 'init':
                    render_init(model, obj, seed, OUT / name,
                                battery.photo(init_id).resize((SIZE, SIZE), Image.LANCZOS))
                else:
                    fv.render_png(model, obj, seed, OUT / name)
                r = fv.layer_stats(model, fv.to_input([fv._load(OUT / name)], dev), [layer, prev])
                rec = dict(condition=cond, seed=seed, max=float(r[layer][0][0, ch]), mean=float(r[layer][1][0, ch]))
                if i is not None:
                    rec.update(input_ch=i, input_max=float(r[prev][0][0, i]), input_mean=float(r[prev][1][0, i]))
                pm = plain.get(key, {}).get('renders', {}).get(fv.png_name(layer, ch, 'plain', seed), {})
                rec['plain_max'] = float(pm.get('max', np.nan))
                rec['over_plain'] = float(battery.ratio(np.float64(rec['max']), np.float64(rec['plain_max'])))
                rec['decision'] = battery.decide(rec['over_plain'])
                u['renders'][name] = rec
                names.append(name)
                extra = f' input_max={rec["input_max"]:.2f}' if i is not None else ''
                print(f'  {name} max={rec["max"]:.2f}{extra}', flush=True)
        (OUT / 'acts.json').write_text(json.dumps(acts, indent=1))
    print('sheet', sheet(rows, acts))

    fv.OUT = OUT  # fixedvis.describe reads PNGs from and writes descriptions.json to its OUT
    desc = fv.describe(names)
    print(f'\n{"unit":14s} {"condition":12s} {"seed":>4s} {"activation":>10s}  description')
    for key, u in acts.items():
        for name, r in u['renders'].items():
            print(f'{key:14s} {r["condition"]:12s} {r["seed"]:4d} {r["max"]:10.2f}  {desc[name]}')
    for key, u in acts.items():
        for name, r in u['renders'].items():
            print(f'{key} {r["condition"]} s{r["seed"]}: render/plain spatial max = {r["over_plain"]:.2f}'
                  f' -> {r["decision"]}')


def selftest():
    import torch
    # --through parsing
    t = parse_through('mixed4d:409=447,393;mixed4e:55=493,352')
    assert t == {('mixed4d', 409): [447, 393], ('mixed4e', 55): [493, 352]}, t
    assert parse_through(' mixed4e:700=3 ;') == {('mixed4e', 700): [3]}
    assert parse_through('') == {}
    try:
        parse_through('mixed5a:3=1')
    except AssertionError:
        pass
    else:
        raise AssertionError('parse_through accepted a layer with no input layer')

    # file naming
    units = fv.parse_units('mixed4d:409,mixed4e:55,mixed4e:700')
    assert conditions(('mixed4e', 700), t) == ['pull5', 'init']
    assert conditions(('mixed4d', 409), t) == ['pull5', 'init', 'through:447', 'through:393']
    assert png_name('mixed4d', 409, 'through:447', 1) == 'mixed4d_409_through-447_s1.png'
    assert png_name('mixed4e', 55, 'pull5', 0) == 'mixed4e_55_pull5_s0.png'
    names = [png_name(l, c, k, s) for l, c in units for k in conditions((l, c), t) for s in fv.SEEDS]
    assert len(set(names)) == len(names) == 20, names
    assert all(':' not in n for n in names)

    # weight scaling
    rng = np.random.default_rng(0)
    um, im = rng.uniform(0, 5, 50), rng.uniform(0.1, 2, 50)
    assert np.isclose(pull_lam(um), 5 * um.mean()), 'pull5 lam != 5 x mean activation'
    assert np.isclose(pull_lam(um), 50 * fv.lam_of(um))
    w = weight(um, im)
    assert np.isclose(w * im.mean(), um.mean()), 'w x channel mean != unit mean'
    assert np.isclose(weight(3 * um, im), 3 * w) and np.isclose(weight(um, 3 * im), w / 3)
    try:
        weight(um, np.zeros(50))
    except AssertionError:
        pass
    else:
        raise AssertionError('weight accepted a zero input-channel mean')

    # through = mean(unit) + w * mean(channel i); gradient only on those two channels
    a = torch.rand(1, 32, 7, 7, requires_grad=True)
    b = torch.rand(1, 16, 7, 7, requires_grad=True)
    v = through_value(a, b, 5, 3, 0.37)
    assert torch.isclose(v, a[0, 5].mean() + 0.37 * b[0, 3].mean())
    v.backward()
    ga, gb = torch.zeros_like(a), torch.zeros_like(b)
    ga[0, 5] = 1 / 49
    gb[0, 3] = 0.37 / 49
    assert torch.allclose(a.grad, ga) and torch.allclose(b.grad, gb), 'through gradient on the wrong channels'
    print('selftest ok')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    units = fv.parse_units(sys.argv[sys.argv.index('--units') + 1])
    through = parse_through(sys.argv[sys.argv.index('--through') + 1]) if '--through' in sys.argv else {}
    run(units, through)


if __name__ == '__main__':
    main()
