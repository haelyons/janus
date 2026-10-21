"""Two added views of a unit's data, in a 2x2 design (spec: authors/spec_lookviews.md).

Per unit (one channel of mixed3a-mixed5b): photos, pools C and H, values, M, card.jpg, random.jpg and the card text
exactly as tools/openmethod.py. View C: the 20 card photos cropped to the unit's effective receptive field (crops.jpg).
View P: an invariance profile over battery.TRANSFORMS on a random 1000 card-pool photos, against every channel of the
unit's layer. Explanations come from agents shown the base card, card + C, card + P and card + C + P, 3 samples each.
Nothing is scored here. Outputs: runs/lookviews/.

usage: python -I tools/lookviews.py [--units-file authors/round21_units.json] [--selftest]
"""
import argparse
import gc
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))  # -I does not add the script dir
import battery  # noqa: E402
import bundlemax  # noqa: E402
import claude_call  # noqa: E402
import openmethod  # noqa: E402
import run_subjects  # noqa: E402

OUT = J / 'runs/lookviews'
UNITS_FILE = 'authors/round21_units.json'
REPLY = J / 'authors/battery.reply.json'
NCH = openmethod.NCH
LAYERS = list(NCH)  # network order, mixed3a first
TILE, BATCH, WORKERS, N_CARD = openmethod.TILE, openmethod.BATCH, openmethod.WORKERS, openmethod.N_CARD
RF_FRAC, RF_MIN, RF_STEP, RF_BATCH = 0.8, 8, 2, 5
S_ROUND, S_MIN, S_MAX = 8, 32, 224
CROP, LAB = 160, 24
N_PROFILE, K_PROFILE = 1000, 20
CONDITIONS = ('base', 'C', 'P', 'CP')
SAMPLES = (1, 2, 3)
READINGS = [f'{c}_s{k}' for c in CONDITIONS for k in SAMPLES]
FIXED_DESC = {'rotate_180': 'upright orientation', 'mirror': 'left-right orientation'}

CROP_SENTENCE = ('The third grid shows the same 20 photos, in the same order, each cropped to a {S} px square of the '
                 '224 px photo around the spot where the unit is most active.')
PROFILE_HEAD = ("How the unit's activation changes when photos are transformed. For each transformation: the median "
                'share of its activation the unit keeps on its 20 strongest photos out of a random 1000, the median of '
                'the same number over all {n} units of its layer (each on its own 20 strongest photos), and this '
                "unit's percentile among them (low = this unit loses more than most units do).")
PROFILE_LINE = '- {name} (removes {description}): keeps {u}%; layer median {m}%; percentile {p}'


def save_json(obj, path):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(path)


# ---------- units ----------

def load_units(data):
    """the units file: a non-empty JSON list of unit strings, each listed once -> [openmethod.parse_unit(s)]"""
    if not isinstance(data, list) or not data:
        raise ValueError('the units file must be a non-empty list of unit strings')
    out = []
    for x in data:
        if not isinstance(x, str):
            raise ValueError(f'bad entry {x!r}')
        out.append(openmethod.parse_unit(x))
    names = [u['name'] for u in out]
    if len(set(names)) != len(names):
        raise ValueError('a unit is listed twice')
    return out


# ---------- effective receptive field and crops ----------

def cell_centre(row, col, size, tile=TILE):
    """(x, y) in pixels of the centre of map cell (row, col)"""
    return (col + 0.5) * tile / size, (row + 0.5) * tile / size


def _start(c, side):
    """first pixel of a run of `side` pixels centred on c (pixel k covers [k, k + 1)); exact for the integer centres of
    7, 14 and 28 px maps and even sides"""
    return int(math.floor(c - side / 2 + 0.5))


def rf_side(g, cx, cy, frac=RF_FRAC, lo=RF_MIN, hi=TILE, step=RF_STEP):
    """the smallest side s in lo, lo + step, ..., hi such that the square of side s centred on (cx, cy), clipped to the
    photo, holds at least frac of the sum of g; hi if no side does"""
    g = np.asarray(g, dtype=np.float64)
    h, w = g.shape
    ii = np.zeros((h + 1, w + 1))
    ii[1:, 1:] = g.cumsum(axis=0).cumsum(axis=1)
    need = frac * ii[h, w]
    for s in range(lo, hi + 1, step):
        x0, y0 = _start(cx, s), _start(cy, s)
        a0, a1 = min(max(x0, 0), w), min(max(x0 + s, 0), w)
        b0, b1 = min(max(y0, 0), h), min(max(y0 + s, 0), h)
        if ii[b1, a1] - ii[b0, a1] - ii[b1, a0] + ii[b0, a0] >= need:
            return s
    return hi


def side_S(s):
    """the median of the sides, rounded to the nearest multiple of 8 (halves up), clamped to [32, 224]"""
    med = float(np.median(np.asarray(s, dtype=np.float64)))
    S = S_ROUND * int(math.floor(med / S_ROUND + 0.5))
    return int(min(max(S, S_MIN), S_MAX))


def crop_box(cx, cy, S, tile=TILE):
    """(x0, y0, x1, y1), x1 and y1 exclusive: the square of side S centred on (cx, cy), shifted (not shrunk) to lie
    inside the photo"""
    assert 0 < S <= tile, S
    x0 = min(max(_start(cx, S), 0), tile - S)
    y0 = min(max(_start(cy, S), 0), tile - S)
    return x0, y0, x0 + S, y0 + S


def crops_grid(imgs, boxes, tile=CROP, lab=LAB, cols=5):
    """each photo cropped to its box and resized to tile x tile (bicubic); 5 columns in the given order, a white strip
    under each with '#n' (as bundlemax.make_card)"""
    assert len(imgs) == len(boxes), (len(imgs), len(boxes))
    rows = -(-len(imgs) // cols)
    im = Image.new('RGB', (cols * tile, rows * (tile + lab)), 'white')
    dr, f = ImageDraw.Draw(im), bundlemax._font(16)
    for n, (p, b) in enumerate(zip(imgs, boxes)):
        x0, y0 = (n % cols) * tile, (n // cols) * (tile + lab)
        im.paste(p.crop(tuple(b)).resize((tile, tile), Image.BICUBIC), (x0, y0))
        dr.text((x0 + 4, y0 + tile + 3), f'#{n + 1}', fill='black', font=f)
    return im


def input_grad(model, u, ids, rows, cols, size):
    """per photo n: g = |d out[n, chan, row_n, col_n] / d input| summed over the 3 colour channels (224 x 224), with
    input = bundlemax.net_input(ids) and out = the output of the module named u['layer'], the forward pass stopped
    there; and out[n, chan, row_n, col_n] itself"""
    import torch
    Stop = bundlemax._bank().Stop
    got = {}

    def stop(_m, _i, out):
        assert out.shape[1] == NCH[u['layer']], tuple(out.shape)
        got['o'] = out
        raise Stop

    h = dict(model.named_modules())[u['layer']].register_forward_hook(stop)
    try:
        x = bundlemax.net_input(ids).requires_grad_(True)
        with torch.enable_grad():
            try:
                model(x)
            except Stop:
                pass
            o = got['o'][:, u['chan']]
            assert tuple(o.shape[1:]) == (size, size), (tuple(o.shape), size)
            v = o[torch.arange(len(ids)), torch.as_tensor(np.asarray(rows, dtype=np.int64)),
                  torch.as_tensor(np.asarray(cols, dtype=np.int64))]
            gx, = torch.autograd.grad(v.sum(), x)
    finally:
        h.remove()
    assert tuple(gx.shape) == (len(ids), 3, TILE, TILE), tuple(gx.shape)
    return gx.abs().sum(dim=1).numpy().astype(np.float64), v.detach().numpy().astype(np.float64)


def receptive_field(u, d, udir, get_model):
    """s for each of the 20 card photos (card order) and S; cached in rf.json"""
    top = openmethod.top_positions(d['vC'])
    ids, rows, cols = d['C'][top], d['rC'][top], d['cC'][top]
    size = int(d['size'])
    centres = [cell_centre(int(r), int(c), size) for r, c in zip(rows, cols)]
    key = {'unit': u['name'], 'top_ids': ids.tolist(), 'rows': rows.tolist(), 'cols': cols.tolist(), 'map_size': size}
    f = udir / 'rf.json'
    if f.exists():
        rf = json.loads(f.read_text())
        if any(rf.get(k) != v for k, v in key.items()):
            raise SystemExit(f'{f}: made for another unit or other card photos; stopping')
        assert rf['S'] == side_S(rf['s']), (rf['S'], rf['s'])
        print(u['name'], 'receptive field: cached', flush=True)
        return rf
    print(u['name'], 'receptive field', flush=True)
    model = get_model()
    parts = [input_grad(model, u, ids[s:s + RF_BATCH], rows[s:s + RF_BATCH], cols[s:s + RF_BATCH], size)
             for s in range(0, len(ids), RF_BATCH)]  # photos are independent; small batches bound the memory
    g, v = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    ref = d['vC'][top].astype(np.float64)
    M = float(d['vC'].max())
    assert np.allclose(v, ref, rtol=1e-3, atol=1e-3 * M), ('value at the stored (row, col) differs from values.npz',
                                                           v.tolist(), ref.tolist())
    s = [rf_side(g[n], *centres[n]) for n in range(len(ids))]
    S = side_S(s)
    rf = {**key, 'S': S, 's': s, 'centres': [list(c) for c in centres], 'g_total': g.sum(axis=(1, 2)).tolist(),
          'crop_boxes': [list(crop_box(cx, cy, S)) for cx, cy in centres]}
    save_json(rf, f)
    return rf


# ---------- invariance profile ----------

def to_input(imgs):
    """as bundlemax.net_input, from PIL images: pixels (0-255) - 117, NCHW"""
    import torch
    x = torch.from_numpy(np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in imgs])).permute(0, 3, 1, 2)
    return x * 255 - 117


def layer_maxes(model, layers, imgs):
    """{layer: (len(imgs), channels) spatial max of every channel}, in one forward pass that hooks every layer in
    layers and is stopped after the deepest"""
    import torch
    Stop = bundlemax._bank().Stop
    deepest = max(layers, key=LAYERS.index)
    mods = dict(model.named_modules())
    got = {}

    def hook(L):
        def f(_m, _i, out):
            assert out.shape[1] == NCH[L], (L, tuple(out.shape))
            got[L] = out.amax(dim=(2, 3)).detach().numpy().astype(np.float32)
            if L == deepest:
                raise Stop
        return f

    hs = [mods[L].register_forward_hook(hook(L)) for L in layers]
    try:
        try:
            with torch.no_grad():
                model(to_input(imgs))
        except Stop:
            pass
    finally:
        for h in hs:
            h.remove()
    assert set(got) == set(layers), (sorted(got), layers)
    return got


def profiles(layers, P0, get_model):
    """{layer: {'orig': (len(P0), channels), 't_<transform>': ...}}: spatial maxes on the original photos and under
    each transform of battery.TRANSFORMS; cached per layer in profile_<layer>.npz"""
    names = list(battery.TRANSFORMS)
    out, todo = {}, []
    for L in layers:
        f = OUT / f'profile_{L}.npz'
        if not f.exists():
            todo.append(L)
            continue
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        if str(d['layer']) != L or not np.array_equal(d['P0'], P0) or d['transforms'].tolist() != names:
            raise SystemExit(f'{f}: made for another layer, photo set or transform list; stopping')
        print(L, 'profile: cached', flush=True)
        out[L] = d
    if not todo:
        return out
    model = get_model()
    conds = [('orig', None)] + [(f't_{n}', fn) for n, fn in battery.TRANSFORMS.items()]
    res = {L: {c: [] for c, _ in conds} for L in todo}
    print('profile', todo, f'{len(P0)} photos x {len(conds)} conditions', flush=True)
    for s in range(0, len(P0), BATCH):
        imgs = [bundlemax.photo(i) for i in P0[s:s + BATCH]]
        for c, fn in conds:
            m = layer_maxes(model, todo, imgs if fn is None else [fn(im) for im in imgs])
            for L in todo:
                res[L][c].append(m[L])
        print(f'  profile {min(s + BATCH, len(P0))}/{len(P0)}', flush=True)
    for L in todo:
        d = {'layer': np.array(L), 'P0': P0, 'transforms': np.array(names)}
        for c, _ in conds:
            d[c] = np.concatenate(res[L][c])
            assert d[c].shape == (len(P0), NCH[L]), (L, c, d[c].shape)
        tmp = OUT / f'profile_{L}.tmp.npz'
        np.savez_compressed(tmp, **d)
        tmp.replace(OUT / f'profile_{L}.npz')
        out[L] = d
    return out


def retention(orig, trans, k=K_PROFILE):
    """per channel: its k photos with the highest original value, photos whose original value is 0 skipped (stable
    order on ties); the median over them of transformed / original. NaN for a channel with no photo above 0."""
    orig, trans = np.asarray(orig, dtype=np.float64), np.asarray(trans, dtype=np.float64)
    assert orig.shape == trans.shape, (orig.shape, trans.shape)
    R = np.full(orig.shape[1], np.nan)
    for c in range(orig.shape[1]):
        pos = np.flatnonzero(orig[:, c] > 0)
        if len(pos):
            top = pos[np.argsort(-orig[pos, c], kind='stable')[:k]]
            R[c] = float(np.median(trans[top, c] / orig[top, c]))
    return R


def retention_table(d, names=None):
    """(channels, transforms) retention of every channel of a layer, transforms in battery.TRANSFORMS order"""
    names = list(battery.TRANSFORMS) if names is None else names
    return np.stack([retention(d['orig'], d[f't_{n}']) for n in names], axis=1)


def unit_profile(R, chan, names=None):
    """-> (n, [(transform, u, m, p)]): n = the layer's channels that have a retention; per transform the unit's
    retention u, the median m over those channels, and p = 100 x their share with retention below u"""
    names = list(battery.TRANSFORMS) if names is None else names
    assert R.shape[1] == len(names), (R.shape, names)
    ok = np.isfinite(R)
    n = int(ok[:, 0].sum())
    assert (ok.sum(axis=0) == n).all(), ok.sum(axis=0)
    rows = []
    for j, name in enumerate(names):
        v = R[ok[:, j], j]
        u = float(R[chan, j])
        m = float(np.median(v)) if len(v) else float('nan')
        p = 100.0 * float(np.mean(v < u)) if len(v) and np.isfinite(u) else float('nan')
        rows.append((name, u, m, p))
    return n, rows


def describe(name, reply):
    """a transform's description: the reply's "removes" for it, cut at the first ':' and then at the first '.',
    lower-cased; fixed for rotate_180 and mirror"""
    if name in FIXED_DESC:
        return FIXED_DESC[name]
    hits = [t for t in reply['transforms'] if t['name'] == name]
    if len(hits) != 1:
        raise ValueError(f'{name}: {len(hits)} entries in the reply')
    return hits[0]['removes'].split(':')[0].split('.')[0].strip().lower()


def whole(x):
    """a whole number (halves up), or 'n/a' if undefined"""
    return 'n/a' if not np.isfinite(x) else str(int(math.floor(x + 0.5)))


def profile_lines(rows, desc):
    return [PROFILE_LINE.format(name=name, description=desc[name], u=whole(100 * u), m=whole(100 * m), p=whole(p))
            for name, u, m, p in rows]


def profile_text(n, rows, desc):
    return PROFILE_HEAD.format(n=n) + '\n' + '\n'.join(profile_lines(rows, desc))


# ---------- conditions and readings ----------

def condition_contents(card_text, card_block, random_block, crops_block, profile, S):
    """{condition: content blocks} in CONDITIONS order"""
    def t(s):
        return {'type': 'text', 'text': s}
    crop_text = card_text + ' ' + CROP_SENTENCE.replace('{S}', str(S))
    task = t(run_subjects.TASK)
    return {'base': [t(card_text), card_block, random_block, task],
            'C': [t(crop_text), card_block, random_block, crops_block, task],
            'P': [t(card_text), card_block, random_block, t(profile), task],
            'CP': [t(crop_text), card_block, random_block, crops_block, t(profile), task]}


def readings(name, contents, udir, call=None):
    """every (condition, sample) reading as openmethod.get_reading with system prompt '', up to WORKERS in parallel,
    cached in readings.json keyed '<condition>_s<k>'; a failed call (is_error) is not cached"""
    path = udir / 'readings.json'
    cache = json.loads(path.read_text()) if path.exists() else {}
    sha = {c: openmethod.sha256(json.dumps(contents[c], sort_keys=True)) for c in CONDITIONS}
    jobs = []
    for c in CONDITIONS:
        for k in SAMPLES:
            key = f'{c}_s{k}'
            if key not in cache:
                jobs.append((key, c, k))
            elif cache[key]['content_sha256'] != sha[c]:
                raise SystemExit(f'{path}: cached {key} was made from different content; stopping')
    if jobs:
        print(name, 'readings', [j[0] for j in jobs], flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(openmethod.get_reading, '', contents[c], call): (key, c, k) for key, c, k in jobs}
        for f in as_completed(futs):
            key, c, k = futs[f]
            try:
                rec = f.result()
            except Exception as err:  # not cached, so a rerun tries again
                print(f'{name} {key} failed: {err}', flush=True)
                continue
            cache[key] = {'unit': name, 'reading': key, 'condition': c, 'sample': k, 'model': claude_call.MODEL,
                          'system_prompt': '', 'content_sha256': sha[c], **rec}
            save_json(cache, path)
    return cache


# ---------- report ----------

def reading_line(rid, r):
    if r is None:
        return f'{rid}  no reading (call failed, not cached)'
    if r.get('explanation') is None:
        return f'{rid}  {r.get("class")}  (reply not parsed)'
    return f'{rid}  {r["class"]}  ' + ' '.join(r['explanation'].split())[:160]


def decision_line(rows):
    """neutral label per transform on the unit's retention u, by battery.decide"""
    return ('retention labels (battery.decide on u: < 0.70 ATTENUATED, > 1.30 AMPLIFIED, else RETAINED): '
            + ', '.join(f'{name} {battery.decide(u)}' for name, u, _, _ in rows))


def section(name, S, plines, rows, recs):
    L = [f'== {name}  S={S}'] + list(plines) + [decision_line(rows)]
    L += [reading_line(rid, recs.get(rid)) for rid in READINGS]
    return '\n'.join(L)


# ---------- selftest ----------

def selftest():
    import tempfile
    import threading
    from collections import OrderedDict

    # units file
    us = load_units(['mixed4c:447', 'mixed5b:1023', 'mixed3a:0'])
    assert [(u['name'], u['layer'], u['chan']) for u in us] == [('mixed4c:447', 'mixed4c', 447),
                                                                ('mixed5b:1023', 'mixed5b', 1023),
                                                                ('mixed3a:0', 'mixed3a', 0)]
    for bad in [[], {'a': 1}, 'mixed4c:447', ['mixed4c:447', 'mixed4c:447'], [{'unit': 'mixed4c:447'}],
                ['mixed4c:512'], ['mixed6a:1'], ['sum:mixed4c:1+2']]:
        try:
            load_units(bad)
        except ValueError:
            continue
        raise AssertionError(f'accepted units {bad!r}')
    assert LAYERS == ['mixed3a', 'mixed3b', 'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a',
                      'mixed5b']

    # receptive-field rule on synthetic g
    assert cell_centre(7, 7, 14) == (120.0, 120.0) and cell_centre(0, 0, 7) == (16.0, 16.0)
    assert cell_centre(27, 0, 28) == (4.0, 220.0) and cell_centre(3, 3, 7) == (112.0, 112.0)
    z = np.zeros((TILE, TILE))
    pm = z.copy()
    pm[120, 120] = 1
    assert rf_side(pm, 120, 120) == RF_MIN == 8  # a point mass at the centre: the minimum
    pm = z.copy()
    pm[123, 116] = 1  # (y, x) inside [116, 124) x [116, 124)
    assert rf_side(pm, 120, 120) == 8
    pm = z.copy()
    pm[130, 120] = 1  # y = 130 first inside [120 - s/2, 120 + s/2) at s = 22
    assert rf_side(pm, 120, 120) == 22
    pm = z.copy()
    pm[0, 0] = 1
    assert rf_side(pm, 8, 8) == 16
    one = np.ones((TILE, TILE))
    need = math.ceil(math.sqrt(RF_FRAC) * TILE)  # 201: the side of a square holding 80% of the area
    assert need == 201
    assert rf_side(one, 112, 112) == 202 and rf_side(one, 120, 120) == 202
    for size in (7, 14, 28):
        for r in range(size):
            for c in range(size):
                assert rf_side(one, *cell_centre(r, c, size)) >= need, (size, r, c)
    blk = z.copy()
    blk[100:124, 100:124] = 1
    assert rf_side(blk, 112, 112) == 22  # 22 x 22 of a 24 x 24 block: 484 / 576 >= 0.8; 20 x 20 is not
    blk = z.copy()
    blk[:24, :24] = 1
    assert rf_side(blk, 8, 8) == 28  # a peak at a corner: the square [-6, 22) is clipped to [0, 22)
    assert rf_side(one, 8, 8) == TILE  # no side up to 224 reaches 80% from a corner cell: 224
    assert rf_side(z, 120, 120) == 8  # zero g: 80% of 0 is reached at once
    assert side_S([8] * 20) == 32 and side_S([224] * 20) == 224 and side_S([300] * 20) == 224
    assert side_S([50] * 10 + [54] * 10) == 56  # median 52: halves round up
    assert side_S([48] * 10 + [52] * 10) == 48  # median 50
    assert side_S(list(range(138, 98, -2))) == 120  # median 119
    assert side_S([200] * 11 + [8] * 9) == 200

    # crop placement: shifted inside, side S
    assert crop_box(120, 120, 64) == (88, 88, 152, 152)
    assert crop_box(8, 8, 64) == (0, 0, 64, 64) and crop_box(216, 216, 64) == (160, 160, 224, 224)
    assert crop_box(216, 8, 48) == (176, 0, 224, 48) and crop_box(4, 220, 32) == (0, 192, 32, 224)
    assert crop_box(112, 112, 224) == (0, 0, 224, 224) and crop_box(8, 216, 224) == (0, 0, 224, 224)
    for size in (7, 14, 28):
        for r in range(size):
            for c in range(size):
                cx, cy = cell_centre(r, c, size)
                for S in range(S_MIN, S_MAX + 1, S_ROUND):
                    x0, y0, x1, y1 = crop_box(cx, cy, S)
                    assert x1 - x0 == S and y1 - y0 == S and x0 >= 0 and y0 >= 0 and x1 <= TILE and y1 <= TILE
                    assert x0 <= cx < x1 and y0 <= cy < y1
                    if S / 2 <= cx <= TILE - S / 2:
                        assert x0 == cx - S / 2
                    if S / 2 <= cy <= TILE - S / 2:
                        assert y0 == cy - S / 2
    rng = np.random.default_rng(0)
    photos = [Image.fromarray(rng.integers(0, 256, (TILE, TILE, 3), dtype=np.uint8)) for _ in range(N_CARD)]
    boxes = [crop_box(*cell_centre(k % 14, (3 * k) % 14, 14), 48) for k in range(N_CARD)]
    grid = crops_grid(photos, boxes)
    assert grid.size == (5 * CROP, 4 * (CROP + LAB)), grid.size
    a = np.asarray(grid)
    for k in (0, 7, 19):
        x0, y0 = (k % 5) * CROP, (k // 5) * (CROP + LAB)
        want = np.asarray(photos[k].crop(boxes[k]).resize((CROP, CROP), Image.BICUBIC))
        assert np.array_equal(a[y0:y0 + CROP, x0:x0 + CROP], want), k
        strip = a[y0 + CROP:y0 + CROP + LAB, x0:x0 + CROP]
        assert (strip.min(axis=2) < 128).any() and (strip == 255).all(axis=2).mean() > 0.8, k

    # input gradient and the one-pass hooks, on small stand-in modules (not the network)
    import torch
    torch.manual_seed(0)

    class Boom(torch.nn.Module):
        def forward(self, x):
            raise AssertionError('ran past the stop')

    fm = torch.nn.Sequential(OrderedDict([('mixed4c', torch.nn.Conv2d(3, 512, 16, stride=16)), ('boom', Boom())]))
    for q in fm.parameters():
        q.requires_grad_(False)
    rc = [(2, 9), (13, 0)]
    g, v = input_grad(fm.eval(), openmethod.parse_unit('mixed4c:5'), np.array([0, 1]), [2, 13], [9, 0], 14)
    with torch.no_grad():
        ref = fm.mixed4c(bundlemax.net_input([0, 1]))[:, 5]
    assert np.allclose(v, [float(ref[0, 2, 9]), float(ref[1, 13, 0])], rtol=1e-5, atol=1e-6)
    for n, (r, c) in enumerate(rc):
        mask = np.zeros((TILE, TILE), bool)
        mask[16 * r:16 * r + 16, 16 * c:16 * c + 16] = True
        assert (g[n][mask] > 0).all() and (g[n][~mask] == 0).all(), n
        assert rf_side(g[n], *cell_centre(r, c, 14)) <= 16
    assert torch.equal(to_input([bundlemax.photo(0), bundlemax.photo(1)]), bundlemax.net_input([0, 1]))
    fm2 = torch.nn.Sequential(OrderedDict([('mixed3a', torch.nn.Conv2d(3, 256, 16, stride=16)),
                                           ('mixed3b', torch.nn.Conv2d(256, 480, 1)), ('boom', Boom())])).eval()
    imgs = photos[:3]
    m2 = layer_maxes(fm2, ['mixed3b', 'mixed3a'], imgs)
    with torch.no_grad():
        a3 = fm2.mixed3a(to_input(imgs))
        b3 = fm2.mixed3b(a3)
    assert m2['mixed3a'].shape == (3, 256) and m2['mixed3b'].shape == (3, 480)
    assert np.allclose(m2['mixed3a'], a3.amax(dim=(2, 3)).numpy()) and np.allclose(m2['mixed3b'],
                                                                                    b3.amax(dim=(2, 3)).numpy())
    seen = []
    h = fm2.mixed3b.register_forward_pre_hook(lambda *_: seen.append(1))
    m1 = layer_maxes(fm2, ['mixed3a'], imgs)
    h.remove()
    assert seen == [] and list(m1) == ['mixed3a']  # stopped after the deepest hooked layer

    # retention and percentile on a synthetic layer, zero originals skipped
    n_ph = 30
    orig, trans = np.zeros((n_ph, 5)), np.zeros((n_ph, 5))
    p = np.arange(n_ph, dtype=np.float64)
    orig[:, 0], trans[:, 0] = p, p * p / 29  # ratio p / 29; photo 0 (value 0) skipped; top 20 = photos 10-29
    orig[:5, 1] = [5, 4, 3, 2, 1]  # only 5 photos above 0
    trans[:5, 1] = [0.5, 0.8, 0.9, 0.8, 0.5]  # ratios 0.1-0.5, median 0.3
    trans[5:, 1] = 100  # on photos with original 0: must be skipped
    trans[:, 2] = 7  # no photo above 0: NaN
    v3 = np.arange(1, 31, dtype=np.float64)
    zero = [0, 7, 14, 21, 28]
    v3[zero] = 0
    orig[:, 3] = v3
    trans[:, 3] = 0.8 * v3
    trans[1:6, 3] = 5 * v3[1:6]  # the 5 lowest values above 0: outside the top 20
    trans[zero, 3] = 50
    orig[:, 4], trans[:, 4] = orig[:, 3], trans[:, 3]
    R1 = retention(orig, trans)
    assert np.allclose(R1[[0, 1, 3, 4]], [19.5 / 29, 0.3, 0.8, 0.8]) and np.isnan(R1[2]), R1
    assert np.allclose(retention(orig, trans, k=5)[[0, 3]], [27 / 29, 0.8])
    R = retention_table({'orig': orig, 't_a': trans, 't_b': orig.copy()}, ['a', 'b'])
    assert R.shape == (5, 2) and np.isnan(R[2]).all() and np.allclose(R[[0, 1, 3, 4], 1], 1)
    n, rows = unit_profile(R, 0, ['a', 'b'])
    med_a = (19.5 / 29 + 0.8) / 2
    assert n == 4 and [x[0] for x in rows] == ['a', 'b']
    assert np.allclose(rows[0][1:], [19.5 / 29, med_a, 25.0]) and np.allclose(rows[1][1:], [1, 1, 0])
    assert unit_profile(R, 1, ['a', 'b'])[1][0][3] == 0.0
    assert unit_profile(R, 3, ['a', 'b'])[1][0][3] == 50.0  # below, not at: the equal channel 4 is not counted
    _, rows2 = unit_profile(R, 2, ['a', 'b'])
    assert np.isnan(rows2[0][1]) and np.isnan(rows2[0][3]) and np.isclose(rows2[0][2], med_a)

    # description rule
    fake_reply = {'transforms': [{'name': 'a', 'removes': 'Foo Bar: baz. qux'},
                                 {'name': 'b', 'removes': 'One, two. Three: four'},
                                 {'name': 'mirror', 'removes': 'Not used: x'}]}
    assert describe('a', fake_reply) == 'foo bar' and describe('b', fake_reply) == 'one, two'
    assert describe('mirror', fake_reply) == 'left-right orientation'
    assert describe('rotate_180', fake_reply) == 'upright orientation'
    try:
        describe('c', fake_reply)
        raise AssertionError('missing transform accepted')
    except ValueError:
        pass
    want = {'grayscale': 'all chromatic information', 'low_pass': 'high spatial frequencies',
            'high_pass': 'low spatial frequencies below about 10 cycles per image',
            'phase_scramble': 'all spatial structure carried by fourier phase',
            'patch_shuffle_4x4': 'global configuration', 'rotate_90': 'absolute orientation',
            'edge_map': ('colour, shading, surface reflectance, texture inside regions, edge contrast magnitude and '
                         'contrast polarity'),
            'luminance_polarity_inversion': 'the sign of luminance contrast',
            'rotate_180': 'upright orientation', 'mirror': 'left-right orientation'}
    reply = json.loads(REPLY.read_text())
    assert list(battery.TRANSFORMS) == list(want)
    got = {nm: describe(nm, reply) for nm in battery.TRANSFORMS}
    assert got == want, got

    # profile text format
    assert whole(62.5) == '63' and whole(85.4) == '85' and whole(100 * 1.006) == '101' and whole(0) == '0'
    assert whole(float('nan')) == 'n/a'
    head = ' '.join([
        "How the unit's activation changes when photos are transformed. For each transformation: the median share of "
        'its',
        'activation the unit keeps on its 20 strongest photos out of a random 1000, the median of the same number over '
        'all',
        "4 units of its layer (each on its own 20 strongest photos), and this unit's percentile among them (low = this "
        'unit',
        'loses more than most units do).'])
    desc = {'a': 'x things', 'b': 'y'}
    pt = profile_text(n, rows, desc)
    assert pt == (head + '\n- a (removes x things): keeps 67%; layer median 74%; percentile 25'
                  '\n- b (removes y): keeps 100%; layer median 100%; percentile 0'), pt
    assert profile_lines(rows2, desc)[0] == '- a (removes x things): keeps n/a%; layer median 74%; percentile n/a'

    # content order of each condition
    cb, rb, xb = {'type': 'image', 'k': 'card'}, {'type': 'image', 'k': 'random'}, {'type': 'image', 'k': 'crops'}
    nm = bundlemax.card_name('mixed4c:447', 'whole')
    base = openmethod.card_content(nm, cb, rb)
    ct = base[0]['text']
    cs = condition_contents(ct, cb, rb, xb, 'PROFILE', 48)
    assert list(cs) == list(CONDITIONS) and cs['base'] == base
    sent = ' '.join(['The third grid shows the same 20 photos, in',
                     'the same order, each cropped to a 48 px square of the 224 px photo around the spot where the '
                     'unit is most active.'])
    ctc = ct + ' ' + sent
    assert ct.endswith('on the same scale.') and '. The third grid shows' in ctc

    def kinds(c):
        return [b['k'] if 'k' in b else b['text'] for b in c]
    T = run_subjects.TASK
    assert kinds(cs['base']) == [ct, 'card', 'random', T]
    assert kinds(cs['C']) == [ctc, 'card', 'random', 'crops', T]
    assert kinds(cs['P']) == [ct, 'card', 'random', 'PROFILE', T]
    assert kinds(cs['CP']) == [ctc, 'card', 'random', 'crops', 'PROFILE', T]
    assert all(b['type'] == 'text' for c in cs.values() for b in c if 'k' not in b)
    assert READINGS == ['base_s1', 'base_s2', 'base_s3', 'C_s1', 'C_s2', 'C_s3', 'P_s1', 'P_s2', 'P_s3',
                        'CP_s1', 'CP_s2', 'CP_s3']

    # readings cache: 12 keys, an is_error reply not cached, content changes stop
    lock, calls = threading.Lock(), []

    def fake(n_err):
        state = {'err': n_err}

        def call(sp, content):
            assert sp == ''
            with lock:
                calls.append(len(content))
                if state['err'] and len(content) == 6:
                    state['err'] -= 1
                    return {'text': 'overloaded', 'is_error': True}
            return {'text': '{"class": "one", "explanation": "n=%d"}' % len(content), 'is_error': False}
        return call
    with tempfile.TemporaryDirectory() as td:
        rec = readings('mixed4c:447', cs, Path(td), call=fake(1))
        assert len(rec) == 11 and len(calls) == 12 and sum(k.startswith('CP_') for k in rec) == 2
        assert rec['base_s2']['explanation'] == 'n=4' and rec['P_s3']['explanation'] == 'n=5'
        assert rec['C_s1']['class'] == 'one' and rec['C_s1']['condition'] == 'C' and rec['C_s1']['sample'] == 1
        calls.clear()
        rec = readings('mixed4c:447', cs, Path(td), call=fake(0))
        assert sorted(rec) == sorted(READINGS) and calls == [6]
        calls.clear()
        cs2 = dict(cs, P=cs['P'][:3] + [{'type': 'text', 'text': 'OTHER'}] + cs['P'][4:])
        try:
            readings('mixed4c:447', cs2, Path(td), call=fake(0))
            raise AssertionError('changed content accepted')
        except SystemExit:
            pass
        assert calls == []

    # report lines
    assert reading_line('base_s1', None) == 'base_s1  no reading (call failed, not cached)'
    assert reading_line('C_s2', {'class': None, 'explanation': None}) == 'C_s2  None  (reply not parsed)'
    assert reading_line('P_s3', {'class': 'one', 'explanation': 'a\n  b'}) == 'P_s3  one  a b'
    assert reading_line('CP_s1', {'class': 'one', 'explanation': 'word  ' * 60}) == 'CP_s1  one  ' + ('word ' * 60)[:160]
    dl = decision_line([('a', 0.5, 0, 0), ('b', 1.0, 0, 0), ('c', 1.5, 0, 0), ('d', float('nan'), 0, 0)])
    assert dl.endswith(': a ATTENUATED, b RETAINED, c AMPLIFIED, d UNDEFINED'), dl
    sec = section('mixed4c:447', 48, profile_lines(rows, desc), rows, {'base_s1': rec['base_s1']}).splitlines()
    assert sec[0] == '== mixed4c:447  S=48' and sec[1].startswith('- a (removes') and sec[3].startswith('retention')
    assert sec[4] == 'base_s1  one  n=4' and sec[5] == 'base_s2  no reading (call failed, not cached)'
    assert len(sec) == 4 + len(READINGS)
    print('selftest ok')


# ---------- main ----------

def main():
    if '--selftest' in sys.argv[1:]:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--units-file', default=UNITS_FILE, help='relative paths are taken from the repo root')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    uf = Path(a.units_file)
    units = load_units(json.loads((uf if uf.is_absolute() else J / uf).read_text()))
    reply = json.loads(REPLY.read_text())
    desc = {n: describe(n, reply) for n in battery.TRANSFORMS}
    names = json.loads(openmethod.nearmiss2.NAMES.read_text())
    C, H = bundlemax.pools(names)
    P0 = np.sort(np.random.default_rng(0).choice(C, N_PROFILE, replace=False))
    held_model = []

    def get_model():
        if not held_model:
            import torch
            torch.set_num_threads(4)
            m = bundlemax._bank().load_model(openmethod.WEIGHTS)
            for q in m.parameters():
                q.requires_grad_(False)
            held_model.append(m)
        return held_model[0]

    OUT.mkdir(parents=True, exist_ok=True)

    # 1. invariance profile of every layer that has a unit
    layers = sorted({u['layer'] for u in units}, key=LAYERS.index)
    prof = profiles(layers, P0, get_model)
    R = {L: retention_table(prof[L]) for L in layers}

    # 2. every unit: values, receptive field, crops
    vals, rfs = {}, {}
    for u in units:
        n = u['name']
        print('==', n, flush=True)
        udir = OUT / n
        udir.mkdir(parents=True, exist_ok=True)
        d = openmethod.unit_values(u, C, H, udir, get_model)
        pos = np.searchsorted(d['C'], P0)
        assert np.array_equal(d['C'][pos], P0), 'P0 is not inside C'
        M = float(d['vC'].max())
        assert np.allclose(prof[u['layer']]['orig'][:, u['chan']], d['vC'][pos], rtol=1e-3, atol=1e-3 * M), \
            f'{n}: profile originals differ from values.npz'
        rf = receptive_field(u, d, udir, get_model)
        boxes = [crop_box(cx, cy, rf['S']) for cx, cy in rf['centres']]
        crops_grid([bundlemax.photo(i) for i in rf['top_ids']], boxes).save(udir / 'crops.jpg', quality=90)
        vals[n], rfs[n] = d, rf
    del prof
    held_model.clear()
    gc.collect()

    # 3. every unit: card, the four contents, readings; report
    secs, allr = [], {}
    for u in units:
        n = u['name']
        udir = OUT / n
        meta, content = openmethod.make_cards(u, vals[n], udir)
        assert meta['top_ids'] == rfs[n]['top_ids'], f'{n}: card photos differ from rf.json'
        assert [b['type'] for b in content] == ['text', 'image', 'image', 'text']
        assert content[3]['text'] == run_subjects.TASK
        cnt, rows = unit_profile(R[u['layer']], u['chan'])
        contents = condition_contents(content[0]['text'], content[1], content[2],
                                      run_subjects.img_block(udir / 'crops.jpg'), profile_text(cnt, rows, desc),
                                      rfs[n]['S'])
        recs = readings(n, contents, udir)
        allr[n] = {rid: (recs[rid]['explanation'] if rid in recs else None) for rid in READINGS}
        secs.append(section(n, rfs[n]['S'], profile_lines(rows, desc), rows, recs))
        save_json(allr, OUT / 'readings_all.json')
        (OUT / 'report.txt').write_text('\n\n'.join(secs) + '\n')
        print(secs[-1], flush=True)
        del content, contents
        gc.collect()
    print('report ->', OUT / 'report.txt')


if __name__ == '__main__':
    main()
