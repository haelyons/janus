"""Affected-vs-spared view for a unit's most distinctive transform (spec: authors/spec_sparedview.md).

Per unit (one channel of mixed3a-mixed5b): values, card, card text, receptive-field side S, crops, invariance profile
and profile text exactly as tools/lookviews.py. T* = the transform with the lowest profile percentile (ties: lower
retention, then battery.TRANSFORMS order), if that percentile is at most 20. Its view: of the unit's 200 strongest
card-pool photos, the 10 that lose the most activation under T* (affected.jpg) and, matched one to one on activation,
10 from the third that lose the least (spared.jpg), each cropped as in crops.jpg with 'activation k, loses q%' under
it. Explanations come from agents shown condition CP (lookviews) and CP+A (CP plus the view), 3 samples each. Nothing
is scored here. Outputs: runs/sparedview/.

usage: python -I tools/sparedview.py [--units-file authors/round23_units.json] [--selftest]
"""
import argparse
import gc
import json
import math
import shutil
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
import lookviews  # noqa: E402
import openmethod  # noqa: E402
import run_subjects  # noqa: E402

OUT = J / 'runs/sparedview'
LV = lookviews.OUT  # runs/lookviews: its caches are copied here when present and not here yet
UNITS_FILE = 'authors/round23_units.json'
NCH, LAYERS = lookviews.NCH, lookviews.LAYERS
TILE, BATCH, RF_BATCH = lookviews.TILE, lookviews.BATCH, lookviews.RF_BATCH
CROP, LAB = lookviews.CROP, lookviews.LAB
N_X, N_PAIR, P_MAX = 200, 10, 20
N_CHECK, RTOL_CHECK = 64, 1e-3
WORKERS = 6
CONDITIONS = ('CP', 'CPA')
SAMPLES = (1, 2, 3)
READINGS = [f'{c}_s{k}' for c in CONDITIONS for k in SAMPLES]
FONT_SIZES = range(16, 7, -1)  # caption font: the largest of these at which every caption fits inside its tile

VIEW_TEXT = ('The next two grids are about the transformation this unit is most unusually sensitive to: {T} '
             '(removes {desc}; percentile {p} in the profile above). The first shows the 10 of its 200 strongest '
             'photos that lose the most activation under it; the second shows 10 photos of similar activation that '
             'lose the least. Each is cropped as in the third grid; under each are its activation and the share it '
             'loses.')


# ---------- T* and the selection ----------

def profile_rows(rows):
    """lookviews.unit_profile rows [(name, u, m, p)] -> [{'name', 'u', 'p'}]"""
    return [{'name': name, 'u': u, 'p': p} for name, u, _m, p in rows]


def pick_tstar(rows, p_max=P_MAX):
    """rows = [{'name', 'u', 'p'}] in battery.TRANSFORMS order -> (name, p) of T*: the lowest percentile p, ties broken
    by lower retention u, then row order; None when that percentile is above p_max. Rows without a finite percentile
    are skipped (None if no row has one)."""
    cand = []
    for j, r in enumerate(rows):
        p, u = float(r['p']), float(r['u'])
        if math.isfinite(p):
            cand.append((p, u if math.isfinite(u) else math.inf, j))
    if not cand:
        return None
    p, _, j = min(cand)
    return None if p > p_max else (rows[j]['name'], rows[j]['p'])


def drop_of(a, b):
    """1 - b / a, elementwise"""
    return 1.0 - np.asarray(b, dtype=np.float64) / np.asarray(a, dtype=np.float64)


def select_pairs(a, drop, n=N_PAIR):
    """a, drop: arrays over X -> (affected_idx, spared_idx), lists of indices into X.
    Affected: the n with the largest drop (ties: higher a, then lower index), largest first.
    Spared candidates: the len(X) // 3 with the smallest drop (ties: higher a, then lower index), less the affected.
    For each affected photo in order, the unused candidate with the smallest |log(a) - log(a_affected)| (ties: the
    candidate with the smaller drop, as ordered above); spared_idx[i] is the match of affected_idx[i].
    ValueError when there are fewer than n candidates."""
    a, drop = np.asarray(a, dtype=np.float64), np.asarray(drop, dtype=np.float64)
    if a.ndim != 1 or a.shape != drop.shape:
        raise ValueError(f'a and drop must be 1-d arrays of one length, got {a.shape} and {drop.shape}')
    if not (np.isfinite(a).all() and np.isfinite(drop).all()):
        raise ValueError('a and drop must be finite')
    N = len(a)
    if not 0 < n <= N:
        raise ValueError(f'n = {n} for {N} photos')
    affected = sorted(range(N), key=lambda i: (-drop[i], -a[i], i))[:n]
    taken = set(affected)
    cand = [i for i in sorted(range(N), key=lambda i: (drop[i], -a[i], i))[:N // 3] if i not in taken]
    if len(cand) < n:
        raise ValueError(f'{len(cand)} spared candidates for {n} affected photos')
    if not all(a[i] > 0 for i in affected + cand):
        raise ValueError('a must be > 0 on the affected photos and the candidates (log match)')
    la = {i: math.log(a[i]) for i in affected + cand}
    spared, used = [], set()
    for i in affected:
        j = min((c for c in cand if c not in used), key=lambda c: abs(la[c] - la[i]))  # first minimum in cand order
        spared.append(j)
        used.add(j)
    return [int(i) for i in affected], [int(j) for j in spared]


def strip_text(k, q):
    """the caption under a view photo: its activation k (the card's 0-10 scale of M) and its drop q (a fraction),
    shown as a whole percentage, halves up (lookviews.whole)"""
    return f'activation {k}, loses {lookviews.whole(100 * float(q))}%'


def view_text(T, desc, p):
    """the text block before the two grids; the percentile shown as in the profile (lookviews.whole)"""
    return VIEW_TEXT.format(T=T, desc=desc, p=lookviews.whole(p))


# ---------- grids ----------

def caption_font(captions, width=CROP - 8):
    """bundlemax._font at the largest size in FONT_SIZES at which every caption is at most width px wide (the
    smallest size if none is)"""
    dr = ImageDraw.Draw(Image.new('RGB', (1, 1), 'white'))
    for s in FONT_SIZES:
        f = bundlemax._font(s)
        if max(dr.textlength(c, font=f) for c in captions) <= width:
            break
    return f


def captioned_grid(imgs, boxes, captions, font, tile=CROP, lab=LAB, cols=5):
    """lookviews.crops_grid (each photo cropped to its box and resized to tile x tile, bicubic; 5 columns in the given
    order; a white strip under each), with the caption in the strip in place of '#n'"""
    assert len(imgs) == len(boxes) == len(captions), (len(imgs), len(boxes), len(captions))
    rows = -(-len(imgs) // cols)
    im = Image.new('RGB', (cols * tile, rows * (tile + lab)), 'white')
    dr = ImageDraw.Draw(im)
    for n, (p, b, c) in enumerate(zip(imgs, boxes, captions)):
        x0, y0 = (n % cols) * tile, (n // cols) * (tile + lab)
        im.paste(p.crop(tuple(b)).resize((tile, tile), Image.BICUBIC), (x0, y0))
        dr.text((x0 + 4, y0 + tile + 3), c, fill='black', font=font)
    return im


def build_view(name, d, S, ts, desc_T, b, photo=bundlemax.photo):
    """ts = T* as (name, p); b = the unit's spatial max under T* on X = its N_X card-pool photos with the highest value
    (strongest first, as openmethod.top_positions). -> (the spared.json record, {'affected': grid, 'spared': grid})"""
    T, p = ts
    X = openmethod.top_positions(d['vC'], N_X)
    ids, a = d['C'][X], d['vC'][X].astype(np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert b.shape == a.shape, (b.shape, a.shape)
    drop = drop_of(a, b)
    aff, spa = select_pairs(a, drop)
    M, size = float(np.max(d['vC'])), int(d['size'])

    def one(j):
        row, col = int(d['rC'][X[j]]), int(d['cC'][X[j]])
        k = bundlemax.act_k(a[j], M)
        return {'id': int(ids[j]), 'rank': j + 1, 'a': float(a[j]), 'b': float(b[j]), 'drop': float(drop[j]), 'k': k,
                'caption': strip_text(k, drop[j]), 'row': row, 'col': col,
                'crop_box': list(lookviews.crop_box(*lookviews.cell_centre(row, col, size), S))}
    A, B = [one(j) for j in aff], [one(j) for j in spa]
    for x, y in zip(A, B):
        y['matched_to'] = x['id']
    font = caption_font([r['caption'] for r in A + B])  # one size for both grids
    grids = {nm: captioned_grid([photo(r['id']) for r in rs], [r['crop_box'] for r in rs], [r['caption'] for r in rs],
                                font)
             for nm, rs in (('affected', A), ('spared', B))}
    rec = {'unit': name, 'tstar': T, 'X_ids': ids.tolist(), 'percentile': float(p), 'description': desc_T,
           'S': int(S), 'M': M, 'n_X': int(len(X)), 'third': int(len(X) // 3), 'X_a': a.tolist(), 'X_b': b.tolist(),
           'affected': A, 'spared': B, 'mean_drop_affected': float(np.mean(drop[aff])),
           'mean_drop_spared': float(np.mean(drop[spa]))}
    return rec, grids


# ---------- network on the compute device ----------

class Nets:
    """the network, each copy loaded on first use: on the CPU (openmethod's computation, for the check) and on the
    compute device (CUDA when torch.cuda.is_available(), else the CPU copy itself)"""

    def __init__(self):
        self.held = {}

    @staticmethod
    def _load(dev):
        import torch
        torch.set_num_threads(4)
        m = bundlemax._bank().load_model(openmethod.WEIGHTS)
        for q in m.parameters():
            q.requires_grad_(False)
        return m.to(dev)

    def cpu(self):
        if 'cpu' not in self.held:
            self.held['cpu'] = self._load('cpu')
        return self.held['cpu']

    def dev(self):
        import torch
        if not torch.cuda.is_available():
            return self.cpu()
        if 'cuda' not in self.held:
            torch.backends.cudnn.allow_tf32 = False  # full fp32 convolutions and matmuls, as on the CPU
            torch.backends.cuda.matmul.allow_tf32 = False
            self.held['cuda'] = self._load('cuda')
        return self.held['cuda']

    def clear(self):
        self.held.clear()


def _device(model):
    return next(model.parameters()).device


def _no_model():
    raise AssertionError('no model is needed: the cache is present')


def _unit_maps(model, u, x):
    """the maps of channel u['chan'] of the output of the module named u['layer'] on input x (moved to the model's
    device), the forward pass stopped there by img_acts.Stop; on the CPU, float32"""
    import torch
    Stop = bundlemax._bank().Stop
    got = {}

    def stop(_m, _i, out):
        assert out.shape[1] == NCH[u['layer']], tuple(out.shape)
        got['m'] = out[:, u['chan']].detach().cpu().numpy().astype(np.float32)
        raise Stop

    h = dict(model.named_modules())[u['layer']].register_forward_hook(stop)
    try:
        try:
            with torch.no_grad():
                model(x.to(_device(model)))
        except Stop:
            pass
    finally:
        h.remove()
    return got['m']


def dev_forward_max(model, u, ids, tag=''):
    """openmethod.forward_max on the model's device: spatial_max of the unit's map on photo ids (bundlemax.net_input,
    batches of 64)"""
    vals, rows, cols, sizes = [], [], [], set()
    for b, s in enumerate(range(0, len(ids), BATCH)):
        v, r, c, n = openmethod.spatial_max(_unit_maps(model, u, bundlemax.net_input(ids[s:s + BATCH])))
        vals.append(v)
        rows.append(r)
        cols.append(c)
        sizes.add(n)
        if b % 20 == 19:
            print(f'  {tag} {min(s + BATCH, len(ids))}/{len(ids)}', flush=True)
    assert len(sizes) == 1, sizes
    return np.concatenate(vals), np.concatenate(rows), np.concatenate(cols), sizes.pop()


def dev_transformed_max(model, u, ids, fn):
    """the unit's spatial max on fn(photo) for each photo id (lookviews.to_input, batches of 64), float64"""
    out = []
    for s in range(0, len(ids), BATCH):
        imgs = [fn(bundlemax.photo(i)) for i in ids[s:s + BATCH]]
        out.append(openmethod.spatial_max(_unit_maps(model, u, lookviews.to_input(imgs)))[0])
    return np.concatenate(out).astype(np.float64)


def dev_layer_maxes(model, layers, imgs):
    """lookviews.layer_maxes on the model's device, results on the CPU"""
    import torch
    Stop = bundlemax._bank().Stop
    deepest = max(layers, key=LAYERS.index)
    mods = dict(model.named_modules())
    got = {}

    def hook(L):
        def f(_m, _i, out):
            assert out.shape[1] == NCH[L], (L, tuple(out.shape))
            got[L] = out.amax(dim=(2, 3)).detach().cpu().numpy().astype(np.float32)
            if L == deepest:
                raise Stop
        return f

    hs = [mods[L].register_forward_hook(hook(L)) for L in layers]
    try:
        try:
            with torch.no_grad():
                model(lookviews.to_input(imgs).to(_device(model)))
        except Stop:
            pass
    finally:
        for h in hs:
            h.remove()
    assert set(got) == set(layers), (sorted(got), layers)
    return got


def dev_input_grad(model, u, ids, rows, cols, size):
    """lookviews.input_grad on the model's device, results on the CPU"""
    import torch
    Stop = bundlemax._bank().Stop
    dev = _device(model)
    got = {}

    def stop(_m, _i, out):
        assert out.shape[1] == NCH[u['layer']], tuple(out.shape)
        got['o'] = out
        raise Stop

    h = dict(model.named_modules())[u['layer']].register_forward_hook(stop)
    try:
        x = bundlemax.net_input(ids).to(dev).requires_grad_(True)
        with torch.enable_grad():
            try:
                model(x)
            except Stop:
                pass
            o = got['o'][:, u['chan']]
            assert tuple(o.shape[1:]) == (size, size), (tuple(o.shape), size)
            v = o[torch.arange(len(ids), device=dev),
                  torch.as_tensor(np.asarray(rows, dtype=np.int64), device=dev),
                  torch.as_tensor(np.asarray(cols, dtype=np.int64), device=dev)]
            gx, = torch.autograd.grad(v.sum(), x)
    finally:
        h.remove()
    assert tuple(gx.shape) == (len(ids), 3, TILE, TILE), tuple(gx.shape)
    return (gx.abs().sum(dim=1).detach().cpu().numpy().astype(np.float64),
            v.detach().cpu().numpy().astype(np.float64))


# ---------- caches, values, receptive field, profiles (as lookviews) ----------

def adopt(src, dst):
    """copy a runs/lookviews cache src to dst when dst is missing and src exists; True if copied (the copy is then
    checked as lookviews checks its own cache)"""
    if dst.exists() or not src.exists():
        return False
    tmp = dst.with_name(dst.stem + '.copy' + dst.suffix)
    shutil.copyfile(src, tmp)
    tmp.replace(dst)
    print('copied', src, '->', dst, flush=True)
    return True


def check_device(u, ids, nets):
    """stop unless the compute device's values on ids match openmethod.forward_max on the CPU (np.allclose, rtol
    1e-3)"""
    ref = openmethod.forward_max(nets.cpu(), u, ids)[0]
    got = dev_forward_max(nets.dev(), u, ids)[0]
    if got.shape != ref.shape or not np.allclose(got, ref, rtol=RTOL_CHECK):
        raise SystemExit(f'{u["name"]}: values on {_device(nets.dev())} differ from the CPU on the first {len(ids)} '
                         'card-pool photos; stopping')
    print(u['name'], f'device check: {len(ids)} card-pool photos match the CPU', flush=True)


def unit_values(u, C, H, udir, nets):
    """values.npz as openmethod.unit_values: copied from runs/lookviews/<unit>/ when present there and not here, and
    checked as openmethod does; else computed on the compute device, after check_device on the first 64 C photos"""
    f = udir / 'values.npz'
    adopt(LV / u['name'] / 'values.npz', f)
    if f.exists():
        return openmethod.unit_values(u, C, H, udir, _no_model)
    check_device(u, C[:N_CHECK], nets)
    v, r, c, size = dev_forward_max(nets.dev(), u, np.concatenate([C, H]), tag=u['name'])
    n = len(C)
    d = {'unit': np.array(u['name']), 'C': C, 'H': H, 'vC': v[:n], 'vH': v[n:], 'rC': r[:n], 'cC': c[:n],
         'rH': r[n:], 'cH': c[n:], 'size': np.array(size)}
    tmp = udir / 'values.tmp.npz'
    np.savez_compressed(tmp, **d)
    tmp.replace(f)
    return d


def receptive_field(u, d, udir, nets):
    """rf.json as lookviews.receptive_field: copied from runs/lookviews/<unit>/ when present there and not here, and
    checked by lookviews; else computed with the gradients on the compute device"""
    f = udir / 'rf.json'
    adopt(LV / u['name'] / 'rf.json', f)
    if f.exists():
        return lookviews.receptive_field(u, d, udir, _no_model)
    top = openmethod.top_positions(d['vC'])
    ids, rows, cols = d['C'][top], d['rC'][top], d['cC'][top]
    size = int(d['size'])
    centres = [lookviews.cell_centre(int(r), int(c), size) for r, c in zip(rows, cols)]
    key = {'unit': u['name'], 'top_ids': ids.tolist(), 'rows': rows.tolist(), 'cols': cols.tolist(), 'map_size': size}
    print(u['name'], 'receptive field', flush=True)
    model = nets.dev()
    parts = [dev_input_grad(model, u, ids[s:s + RF_BATCH], rows[s:s + RF_BATCH], cols[s:s + RF_BATCH], size)
             for s in range(0, len(ids), RF_BATCH)]
    g, v = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    ref = d['vC'][top].astype(np.float64)
    M = float(d['vC'].max())
    assert np.allclose(v, ref, rtol=1e-3, atol=1e-3 * M), ('value at the stored (row, col) differs from values.npz',
                                                           v.tolist(), ref.tolist())
    s = [lookviews.rf_side(g[n], *centres[n]) for n in range(len(ids))]
    S = lookviews.side_S(s)
    rf = {**key, 'S': S, 's': s, 'centres': [list(c) for c in centres], 'g_total': g.sum(axis=(1, 2)).tolist(),
          'crop_boxes': [list(lookviews.crop_box(cx, cy, S)) for cx, cy in centres]}
    lookviews.save_json(rf, f)
    return rf


def profiles(layers, P0, nets):
    """lookviews.profiles with its per-layer caches in runs/sparedview/ (runs/lookviews/profile_<layer>.npz copied when
    present there and not here), checked as lookviews does; new profiles on the compute device"""
    names = list(battery.TRANSFORMS)
    out, todo = {}, []
    for L in layers:
        f = OUT / f'profile_{L}.npz'
        adopt(LV / f'profile_{L}.npz', f)
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
    model = nets.dev()
    conds = [('orig', None)] + [(f't_{n}', fn) for n, fn in battery.TRANSFORMS.items()]
    res = {L: {c: [] for c, _ in conds} for L in todo}
    print('profile', todo, f'{len(P0)} photos x {len(conds)} conditions', flush=True)
    for s in range(0, len(P0), BATCH):
        imgs = [bundlemax.photo(i) for i in P0[s:s + BATCH]]
        for c, fn in conds:
            m = dev_layer_maxes(model, todo, imgs if fn is None else [fn(im) for im in imgs])
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


# ---------- the view of one unit ----------

def unit_view(u, d, S, prows, desc, udir, nets):
    """T* from the unit's profile rows; spared.json, affected.jpg and spared.jpg, or a 'no T*' record. b on X is
    cached in spared.json; a spared.json made for another unit, T* or X stops the run. -> the spared.json record"""
    name = u['name']
    ts = pick_tstar(prows)
    f = udir / 'spared.json'
    old = json.loads(f.read_text()) if f.exists() else None
    X = openmethod.top_positions(d['vC'], N_X)
    key = {'unit': name, 'tstar': None if ts is None else ts[0], 'X_ids': None if ts is None else d['C'][X].tolist()}
    if old is not None and any(old.get(k) != v for k, v in key.items()):
        raise SystemExit(f'{f}: made for another unit, T* or photo set; stopping')
    if ts is None:
        rec = {**key, 'record': 'no T*', 'lowest_percentile': pick_tstar(prows, p_max=math.inf)}
        lookviews.save_json(rec, f)
        print(name, 'no T*', flush=True)
        return rec
    a = d['vC'][X]
    if not (a > 0).all():
        raise SystemExit(f'{name}: a value <= 0 among its {N_X} strongest card-pool photos; stopping')
    if old is not None:
        b = np.asarray(old['X_b'], dtype=np.float64)
        print(name, f'view: {ts[0]} values cached', flush=True)
    else:
        print(name, f'view: {ts[0]} on its {N_X} strongest card-pool photos', flush=True)
        b = dev_transformed_max(nets.dev(), u, d['C'][X], battery.TRANSFORMS[ts[0]])
    rec, grids = build_view(name, d, S, ts, desc[ts[0]], b)
    for nm, im in grids.items():
        im.save(udir / f'{nm}.jpg', quality=90)
    lookviews.save_json(rec, f)
    return rec


# ---------- conditions and readings ----------

def cpa_contents(cp, text, affected_block, spared_block):
    """CP's blocks with a text block, affected.jpg and spared.jpg inserted just before the run_subjects.TASK block"""
    at = [j for j, b in enumerate(cp) if b.get('type') == 'text' and b.get('text') == run_subjects.TASK]
    if len(at) != 1:
        raise ValueError(f'{len(at)} task blocks in the CP content')
    j = at[0]
    return cp[:j] + [{'type': 'text', 'text': text}, affected_block, spared_block] + cp[j:]


def conditions(card_text, card_block, random_block, crops_block, profile, S, view=None):
    """{'CP': exactly lookviews.condition_contents' CP, 'CPA': CP with the view inserted}; 'CPA' only when view =
    (text, affected block, spared block) is given"""
    cp = lookviews.condition_contents(card_text, card_block, random_block, crops_block, profile, S)['CP']
    out = {'CP': cp}
    if view is not None:
        out['CPA'] = cpa_contents(cp, *view)
    return out


def reading_ids(contents):
    return [f'{c}_s{k}' for c in CONDITIONS if c in contents for k in SAMPLES]


def readings(name, contents, udir, call=None):
    """as lookviews.readings, over the conditions in contents (CONDITIONS order): every (condition, sample) reading as
    openmethod.get_reading with system prompt '', up to WORKERS = 6 in parallel, cached in readings.json keyed
    '<condition>_s<k>'; a failed call (is_error) is not cached"""
    path = udir / 'readings.json'
    cache = json.loads(path.read_text()) if path.exists() else {}
    conds = [c for c in CONDITIONS if c in contents]
    sha = {c: openmethod.sha256(json.dumps(contents[c], sort_keys=True)) for c in conds}
    jobs = []
    for c in conds:
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
            lookviews.save_json(cache, path)
    return cache


# ---------- report ----------

def decision_line(drop_affected, drop_spared):
    """neutral label on each grid's mean retention, 1 - mean drop, by battery.decide"""
    return ('view labels (battery.decide on 1 - mean drop: < 0.70 ATTENUATED, > 1.30 AMPLIFIED, else RETAINED): '
            f'affected {battery.decide(1 - drop_affected)}, spared {battery.decide(1 - drop_spared)}')


def section(name, S, view, recs, rids):
    L = [f'== {name}  S={S}']
    if view.get('tstar') is None:
        L.append('no T*')
    else:
        da, ds = view['mean_drop_affected'], view['mean_drop_spared']
        L += [f'T* {view["tstar"]}  percentile {view["percentile"]:.1f}',
              f'mean drop: affected {da:.3f}, spared {ds:.3f}', decision_line(da, ds)]
    L += [lookviews.reading_line(rid, recs.get(rid)) for rid in rids]
    return '\n'.join(L)


# ---------- selftest ----------

def selftest():
    import tempfile
    import threading

    # T*: the lowest percentile, ties by lower retention, then battery order; none above 20
    tn = list(battery.TRANSFORMS)
    assert len(tn) == 10 and tn[:5] == ['grayscale', 'low_pass', 'high_pass', 'phase_scramble', 'patch_shuffle_4x4']

    def rows_of(ps, us=None):
        us = [1.0] * len(ps) if us is None else us
        return [{'name': n, 'u': u, 'p': p} for n, u, p in zip(tn, us, ps)]
    assert pick_tstar(rows_of([50, 5, 30, 60, 70, 80, 90, 40, 45, 55])) == ('low_pass', 5)
    assert pick_tstar(rows_of([50, 30, 30, 60, 70, 80, 90, 40, 45, 3])) == ('mirror', 3)
    assert pick_tstar(rows_of([50, 20, 30] + [99] * 7)) == ('low_pass', 20)  # at 20: a view
    assert pick_tstar(rows_of([50, 20.01, 30] + [99] * 7)) is None  # above 20: none
    assert pick_tstar(rows_of([21] * 10)) is None and pick_tstar(rows_of([0] * 10)) == ('grayscale', 0)
    us = [0.9, 0.8, 0.5, 0.5] + [0.1] * 6
    assert pick_tstar(rows_of([10, 10, 10, 10] + [50] * 6, us)) == ('high_pass', 10)  # lower u, then battery order
    assert pick_tstar(rows_of([10, 3, 10, 3] + [50] * 6, [0.2, 0.9, 0.1, 0.4] + [1] * 6)) == ('phase_scramble', 3)
    nan = float('nan')
    assert pick_tstar(rows_of([nan, 15] + [50] * 8)) == ('low_pass', 15) and pick_tstar(rows_of([nan] * 10)) is None
    assert pick_tstar(rows_of([50] * 9 + [30])) is None
    assert pick_tstar(rows_of([50] * 9 + [30]), p_max=math.inf) == ('mirror', 30)
    R = np.ones((5, 10))
    R[0, 6] = 0.1  # channel 0: no channel below it anywhere (p = 0 everywhere), lowest retention under edge_map
    R[1] = 2.0  # channel 1: 4 of the 5 channels below it everywhere (p = 80)
    _, r0 = lookviews.unit_profile(R, 0)
    assert pick_tstar(profile_rows(r0)) == ('edge_map', 0.0)
    _, r1 = lookviews.unit_profile(R, 1)
    assert np.allclose([x[3] for x in r1], 80) and pick_tstar(profile_rows(r1)) is None
    assert [r['name'] for r in profile_rows(r0)] == tn and profile_rows(r0)[6]['u'] == 0.1

    # affected and spared photos
    a1 = np.array([4.0, 4.1, 4.0, 5, 5, 5, 5, 5, 4.05, 1.0, 6.5, 2.0])
    d1 = np.array([0.9, 0.8, 0.3, 0.35, 0.4, 0.45, 0.5, 0.42, 0.0, 0.01, 0.02, 0.03])
    # third = photos 8-11. Photo 8 is the closest match of both affected photos: photo 0 takes it, photo 1 gets 10
    # (|log| difference; by |a| difference it would get 11). Photo 2 matches photo 0 exactly but is not in the third.
    assert select_pairs(a1, d1, n=2) == ([0, 1], [8, 10])
    a2 = np.array([5, 9, 1, 2, 3, 4, 6, 7, 8, 1.5, 2.5, 3.5])
    d2 = np.array([0.5] + [0.0] * 11)
    # tied drops: affected = 0 and the highest a among the ties (1); the third (ties: higher a) = 1, 8, 7, 6 less 1
    aff, spa = select_pairs(a2, d2, n=2)
    assert aff == [0, 1] and spa == [6, 8] and 1 not in spa
    a3 = np.array([1, 3, 2, 5, 5, 5, 1.1, 2.9, 2.1])
    d3 = np.array([0.7, 0.7, 0.7, 0.2, 0.3, 0.25, 0.0, 0.05, 0.1])
    assert select_pairs(a3, d3, n=2) == ([1, 2], [7, 8])  # drop ties among the affected: higher a first
    a4 = np.array([3, 5, 5, 5, 5, 5, 3, 3, 9])
    d4 = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.02, 0.01, 0.0])
    assert select_pairs(a4, d4, n=1) == ([0], [7])  # equal matches (6 and 7): the smaller drop
    a5 = np.array([2, 2.1, 5, 5, 2.05, 9])
    d5 = np.array([0.9, 0.8, 0.5, 0.4, 0.0, 0.1])
    assert select_pairs(a5, d5, n=2) == ([0, 1], [4, 5])  # no reuse: photo 1 gets the only candidate left
    for bad in [(np.arange(1.0, 7.0), np.linspace(0, 1, 6), 3), (np.ones(6), np.ones(5), 1),
                (np.array([1.0, 2, 3, 0, 5, 6]), np.array([0.9, 0.8, 0.7, 0.0, 0.1, 0.2]), 1),
                (np.ones(6), np.array([0.1, nan, 0.2, 0.3, 0.4, 0.5]), 1)]:
        try:
            select_pairs(*bad)
        except ValueError:
            continue
        raise AssertionError(f'select_pairs accepted {bad}')
    rng = np.random.default_rng(0)
    ar, dr_ = np.exp(rng.normal(size=N_X)), rng.uniform(-0.2, 1, N_X)
    aff, spa = select_pairs(ar, dr_)
    assert len(aff) == len(spa) == N_PAIR == 10 and len(set(spa)) == N_PAIR and not set(aff) & set(spa)
    assert aff == np.argsort(-dr_)[:N_PAIR].tolist() and set(spa) <= set(np.argsort(dr_)[:N_X // 3].tolist())
    assert all(type(i) is int for i in aff + spa)

    # drop and caption
    assert np.allclose(drop_of(np.array([2.0, 4.0, 5.0]), np.array([1.0, 5.0, 5.0])), [0.5, -0.25, 0.0])
    assert drop_of(4.0, 1.0) == 0.75 and drop_of([2, 4], [1, 1]).shape == (2,)
    assert strip_text(7, 0.634) == 'activation 7, loses 63%'
    assert strip_text(10, 0.625) == 'activation 10, loses 63%'  # halves up
    assert strip_text(3, 0.375) == 'activation 3, loses 38%' and strip_text(0, 0.004) == 'activation 0, loses 0%'
    assert strip_text(5, 1.0) == 'activation 5, loses 100%' and strip_text(5, -0.125) == 'activation 5, loses -12%'
    assert strip_text(np.int64(7), np.float64(0.634)) == 'activation 7, loses 63%'
    assert strip_text(bundlemax.act_k(3.6, 10.0), 0.5) == 'activation 4, loses 50%'

    # the view text and the content order of each condition
    vt = view_text('low_pass', 'high spatial frequencies', 6.6)
    assert vt == ' '.join([
        'The next two grids are about the transformation this unit is most unusually sensitive to: low_pass (removes',
        'high spatial frequencies; percentile 7 in the profile above). The first shows the 10 of its 200 strongest',
        'photos that lose the most activation under it; the second shows 10 photos of similar activation that lose',
        'the least. Each is cropped as in the third grid; under each are its activation and the share it loses.']), vt
    cb, rb, xb = {'type': 'image', 'k': 'card'}, {'type': 'image', 'k': 'random'}, {'type': 'image', 'k': 'crops'}
    ab, sb = {'type': 'image', 'k': 'affected'}, {'type': 'image', 'k': 'spared'}
    ct = openmethod.card_content(bundlemax.card_name('mixed4c:447', 'whole'), cb, rb)[0]['text']
    ctc = ct + ' ' + lookviews.CROP_SENTENCE.replace('{S}', '48')
    T = run_subjects.TASK
    cs = conditions(ct, cb, rb, xb, 'PROFILE', 48, (vt, ab, sb))

    def kinds(c):
        return [b['k'] if 'k' in b else b['text'] for b in c]
    assert list(cs) == ['CP', 'CPA'] and cs['CP'] == lookviews.condition_contents(ct, cb, rb, xb, 'PROFILE', 48)['CP']
    assert kinds(cs['CP']) == [ctc, 'card', 'random', 'crops', 'PROFILE', T]
    assert kinds(cs['CPA']) == [ctc, 'card', 'random', 'crops', 'PROFILE', vt, 'affected', 'spared', T]
    assert cs['CPA'][5] == {'type': 'text', 'text': vt} and cs['CPA'][-1] == {'type': 'text', 'text': T}
    cs0 = conditions(ct, cb, rb, xb, 'PROFILE', 48)
    assert list(cs0) == ['CP'] and cs0['CP'] == cs['CP']
    assert READINGS == ['CP_s1', 'CP_s2', 'CP_s3', 'CPA_s1', 'CPA_s2', 'CPA_s3']
    assert reading_ids(cs) == READINGS and reading_ids(cs0) == ['CP_s1', 'CP_s2', 'CP_s3']
    for bad in (cs['CP'][:-1], cs['CP'] + [cs['CP'][-1]]):
        try:
            cpa_contents(bad, vt, ab, sb)
        except ValueError:
            continue
        raise AssertionError('content without exactly one task block accepted')

    # captions fit their tile; the view record and grids on synthetic values and photos
    caps = ['activation 10, loses 100%', 'activation 10, loses -20%', 'activation 0, loses 0%']
    f = caption_font(caps)
    dr = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    assert max(dr.textlength(c, font=f) for c in caps) <= CROP - 8
    n_c = 300
    Cs = np.arange(1000, 1000 + n_c)
    vC = rng.uniform(0.5, 10, n_c).astype(np.float32)
    d = {'C': Cs, 'vC': vC, 'rC': rng.integers(0, 14, n_c), 'cC': rng.integers(0, 14, n_c), 'size': np.array(14)}
    X = openmethod.top_positions(vC, N_X)
    b = vC[X].astype(np.float64) * rng.uniform(0, 1.2, N_X)
    pics = {int(i): Image.fromarray(rng.integers(0, 256, (TILE, TILE, 3), dtype=np.uint8)) for i in Cs[X]}
    rec, grids = build_view('mixed4c:447', d, 48, ('low_pass', 6.6), 'high spatial frequencies', b,
                            photo=lambda i: pics[int(i)])
    dX = drop_of(vC[X], b)
    ia, ib = select_pairs(vC[X], dX)
    A, B = rec['affected'], rec['spared']
    assert [r['rank'] - 1 for r in A] == ia and [r['rank'] - 1 for r in B] == ib
    assert [r['matched_to'] for r in B] == [r['id'] for r in A]
    M = float(vC.max())
    for r in A + B:
        j = r['rank'] - 1
        assert r['id'] == int(Cs[X[j]]) and r['a'] == float(vC[X[j]]) and r['b'] == float(b[j])
        assert np.isclose(r['drop'], dX[j]) and r['k'] == bundlemax.act_k(r['a'], M)
        assert r['caption'] == strip_text(r['k'], r['drop'])
        assert r['crop_box'] == list(lookviews.crop_box(*lookviews.cell_centre(int(d['rC'][X[j]]),
                                                                               int(d['cC'][X[j]]), 14), 48))
    assert np.isclose(rec['mean_drop_affected'], np.mean(dX[ia])) and np.isclose(rec['mean_drop_spared'],
                                                                                  np.mean(dX[ib]))
    assert rec['tstar'] == 'low_pass' and rec['percentile'] == 6.6 and rec['X_ids'] == Cs[X].tolist()
    assert rec['third'] == 66 and len(rec['X_b']) == N_X and rec['S'] == 48
    json.dumps(rec)
    for nm, rs in (('affected', A), ('spared', B)):
        g = grids[nm]
        assert g.size == (5 * CROP, 2 * (CROP + LAB)), g.size
        arr = np.asarray(g)
        for k in (0, 4, 7):
            x0, y0 = (k % 5) * CROP, (k // 5) * (CROP + LAB)
            want = np.asarray(pics[rs[k]['id']].crop(tuple(rs[k]['crop_box'])).resize((CROP, CROP), Image.BICUBIC))
            assert np.array_equal(arr[y0:y0 + CROP, x0:x0 + CROP], want), (nm, k)
            strip = arr[y0 + CROP:y0 + CROP + LAB, x0:x0 + CROP]
            assert (strip.min(axis=2) < 128).any() and (strip == 255).all(axis=2).mean() > 0.6, (nm, k)
        for k in range(len(rs)):
            if k % 5 < 4:  # the caption stays inside its own tile
                x1, y0 = (k % 5 + 1) * CROP, (k // 5) * (CROP + LAB)
                assert (arr[y0 + CROP:y0 + CROP + LAB, x1 - 2:x1 + 2] == 255).all(), (nm, k)

    # readings cache: 6 keys with a view, 3 without; an is_error reply not cached; content changes stop
    lock, calls = threading.Lock(), []

    def fake(n_err):
        state = {'err': n_err}

        def call(sp, content):
            assert sp == ''
            with lock:
                calls.append(len(content))
                if state['err'] and len(content) == 9:
                    state['err'] -= 1
                    return {'text': 'overloaded', 'is_error': True}
            return {'text': '{"class": "one", "explanation": "n=%d"}' % len(content), 'is_error': False}
        return call
    with tempfile.TemporaryDirectory() as td:
        got = readings('mixed4c:447', cs, Path(td), call=fake(1))
        assert len(got) == 5 and len(calls) == 6 and sum(k.startswith('CPA_') for k in got) == 2
        assert got['CP_s2']['explanation'] == 'n=6' and got['CP_s2']['condition'] == 'CP' and got['CP_s2']['sample'] == 2
        assert all(got[k]['explanation'] == 'n=9' and got[k]['condition'] == 'CPA' for k in got if k[:4] == 'CPA_')
        assert all(r['system_prompt'] == '' and r['model'] == claude_call.MODEL for r in got.values())
        calls.clear()
        got = readings('mixed4c:447', cs, Path(td), call=fake(0))
        assert sorted(got) == sorted(READINGS) and calls == [9]
        calls.clear()
        cs2 = dict(cs, CPA=cs['CPA'][:5] + [{'type': 'text', 'text': 'OTHER'}] + cs['CPA'][6:])
        try:
            readings('mixed4c:447', cs2, Path(td), call=fake(0))
            raise AssertionError('changed content accepted')
        except SystemExit:
            pass
        assert calls == []
    with tempfile.TemporaryDirectory() as td:
        got = readings('mixed4c:447', cs0, Path(td), call=fake(0))
        assert sorted(got) == ['CP_s1', 'CP_s2', 'CP_s3'] and calls == [6, 6, 6]

    # caches from runs/lookviews: copied only when missing here
    with tempfile.TemporaryDirectory() as td:
        lv, sv = Path(td) / 'lv', Path(td) / 'sv'
        lv.mkdir()
        sv.mkdir()
        (lv / 'rf.json').write_text('x')
        assert adopt(lv / 'rf.json', sv / 'rf.json') and (sv / 'rf.json').read_text() == 'x'
        (lv / 'rf.json').write_text('y')
        assert not adopt(lv / 'rf.json', sv / 'rf.json') and (sv / 'rf.json').read_text() == 'x'
        assert not adopt(lv / 'values.npz', sv / 'values.npz') and not (sv / 'values.npz').exists()
        assert sorted(p.name for p in sv.iterdir()) == ['rf.json']

    # report lines
    assert decision_line(0.5, 0.1).endswith(': affected ATTENUATED, spared RETAINED')
    assert decision_line(-0.4, 0.29).endswith(': affected AMPLIFIED, spared RETAINED')
    sec = section('mixed4c:447', 48, rec, {'CP_s1': {'class': 'one', 'explanation': 'a\n  b'}}, READINGS).splitlines()
    assert sec[0] == '== mixed4c:447  S=48' and sec[1] == 'T* low_pass  percentile 6.6'
    assert sec[2] == f'mean drop: affected {rec["mean_drop_affected"]:.3f}, spared {rec["mean_drop_spared"]:.3f}'
    assert sec[3].startswith('view labels (battery.decide') and sec[4] == 'CP_s1  one  a b'
    assert sec[5] == 'CP_s2  no reading (call failed, not cached)' and len(sec) == 4 + len(READINGS)
    sec = section('mixed4c:447', 48, {'unit': 'mixed4c:447', 'tstar': None, 'record': 'no T*'}, {},
                  reading_ids(cs0)).splitlines()
    assert sec[:2] == ['== mixed4c:447  S=48', 'no T*'] and len(sec) == 5 and sec[2].startswith('CP_s1  no reading')
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
    units = lookviews.load_units(json.loads((uf if uf.is_absolute() else J / uf).read_text()))
    reply = json.loads(lookviews.REPLY.read_text())
    desc = {n: lookviews.describe(n, reply) for n in battery.TRANSFORMS}
    names = json.loads(openmethod.nearmiss2.NAMES.read_text())
    C, H = bundlemax.pools(names)
    P0 = np.sort(np.random.default_rng(0).choice(C, lookviews.N_PROFILE, replace=False))
    nets = Nets()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. every unit: values (a unit whose values are computed is first checked against the CPU)
    vals = {}
    for u in units:
        print('==', u['name'], flush=True)
        udir = OUT / u['name']
        udir.mkdir(parents=True, exist_ok=True)
        vals[u['name']] = unit_values(u, C, H, udir, nets)

    # 2. invariance profile of every layer that has a unit
    layers = sorted({u['layer'] for u in units}, key=LAYERS.index)
    prof = profiles(layers, P0, nets)
    R = {L: lookviews.retention_table(prof[L]) for L in layers}

    # 3. every unit: profile check, receptive field, crops, T* and its view
    rfs, profs, views = {}, {}, {}
    for u in units:
        n = u['name']
        print('==', n, flush=True)
        udir = OUT / n
        d = vals[n]
        pos = np.searchsorted(d['C'], P0)
        assert np.array_equal(d['C'][pos], P0), 'P0 is not inside C'
        M = float(d['vC'].max())
        assert np.allclose(prof[u['layer']]['orig'][:, u['chan']], d['vC'][pos], rtol=1e-3, atol=1e-3 * M), \
            f'{n}: profile originals differ from values.npz'
        rf = receptive_field(u, d, udir, nets)
        boxes = [lookviews.crop_box(cx, cy, rf['S']) for cx, cy in rf['centres']]
        lookviews.crops_grid([bundlemax.photo(i) for i in rf['top_ids']], boxes).save(udir / 'crops.jpg', quality=90)
        profs[n] = lookviews.unit_profile(R[u['layer']], u['chan'])
        views[n] = unit_view(u, d, rf['S'], profile_rows(profs[n][1]), desc, udir, nets)
        rfs[n] = rf
    del prof
    nets.clear()
    gc.collect()

    # 4. every unit: card, the contents, readings; report
    secs, allr = [], {}
    for u in units:
        n = u['name']
        udir = OUT / n
        meta, content = openmethod.make_cards(u, vals[n], udir)
        assert meta['top_ids'] == rfs[n]['top_ids'], f'{n}: card photos differ from rf.json'
        assert [b['type'] for b in content] == ['text', 'image', 'image', 'text']
        assert content[3]['text'] == run_subjects.TASK
        cnt, rows = profs[n]
        v = views[n]
        view = None if v['tstar'] is None else (view_text(v['tstar'], desc[v['tstar']], v['percentile']),
                                                run_subjects.img_block(udir / 'affected.jpg'),
                                                run_subjects.img_block(udir / 'spared.jpg'))
        contents = conditions(content[0]['text'], content[1], content[2], run_subjects.img_block(udir / 'crops.jpg'),
                              lookviews.profile_text(cnt, rows, desc), rfs[n]['S'], view)
        recs = readings(n, contents, udir)
        rids = reading_ids(contents)
        allr[n] = {rid: (recs[rid]['explanation'] if rid in recs else None) for rid in rids}
        secs.append(section(n, rfs[n]['S'], v, recs, rids))
        lookviews.save_json(allr, OUT / 'readings_all.json')
        (OUT / 'report.txt').write_text('\n\n'.join(secs) + '\n')
        print(secs[-1], flush=True)
        del content, contents, view
        gc.collect()
    print('report ->', OUT / 'report.txt')


if __name__ == '__main__':
    main()
