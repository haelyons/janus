"""Stratified exemplars and same-label near misses, in a 2x2 design (spec: authors/spec_rangeview.md).

Per unit (one channel of mixed3a-mixed5b): values, M, card.jpg, random.jpg, the card text, the receptive-field size S,
crops.jpg, the invariance profile and its text exactly as tools/lookviews.py, which give its condition CP. View S
(range.jpg): photos drawn at random from four ranges of value / M over the card pool. View N (nearmiss.jpg): for the
labels most common among the 20 card photos, each label's 2 least-active card-pool photos. Explanations come from
agents shown CP, CP + S, CP + N and CP + S + N, 3 samples each. Nothing is scored here. Outputs: runs/rangeview/.

usage: python -I tools/rangeview.py [--units-file authors/round24_units.json] [--selftest]
"""
import argparse
import gc
import json
import math
import shutil
import sys
import zlib
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
import nearmiss2  # noqa: E402
import openmethod  # noqa: E402
import run_subjects  # noqa: E402

OUT = J / 'runs/rangeview'
UNITS_FILE = 'authors/round24_units.json'
SOURCES = (J / 'runs/lookviews', J / 'runs/sparedview')  # caches are copied from the first that has them
UNIT_CACHES = ('values.npz', 'rf.json')
NCH, TILE, BATCH = openmethod.NCH, openmethod.TILE, openmethod.BATCH
COLS, LAB = 5, 24  # as openmethod.make_card_boxed
N_CHECK, CHECK_RTOL = 64, 1e-3
WORKERS = 6
BINS = ((0.6, 1.0), (0.4, 0.6), (0.2, 0.4), (0.05, 0.2))  # [lo, hi); the first bin also holds r = 1.0
BIN_NAMES = ('60-100%', '40-60%', '20-40%', '5-20%')
PER_BIN, N_LABELS, PER_LABEL = 5, 5, 2
CONDITIONS = ('CP', 'CPS', 'CPN', 'CPSN')
SAMPLES = (1, 2, 3)
READINGS = [f'{c}_s{k}' for c in CONDITIONS for k in SAMPLES]

S_TEXT = ('The next grid shows photos drawn at random from four activation ranges below the strongest photos, one row '
          'each: 60-100%, 40-60%, 20-40% and 5-20% of the strongest activation (fewer when a range has fewer photos). '
          'Same box and scale as the first grid.')
N_TEXT = ('The next grid shows near misses: for each of the 5 kinds of photo most common among the strongest photos, '
          'the 2 photos of that kind on which the unit is least active (#1-#2 are one kind, #3-#4 the next, and so '
          'on). Same box and scale as the first grid.')


# ---------- caches copied from earlier runs ----------

def copy_file(src, dst):
    tmp = dst.with_name(dst.name + '.copy.tmp')
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def copy_first(name, srcs, dst):
    """dst / name, if missing, copied from the first of the dirs srcs that has it; -> that dir, or None if nothing was
    copied"""
    if (dst / name).exists():
        return None
    for s in srcs:
        if (s / name).is_file():
            dst.mkdir(parents=True, exist_ok=True)
            copy_file(s / name, dst / name)
            return s
    return None


def copy_unit_caches(udir, srcs):
    """only if udir has no values.npz: values.npz and rf.json (each when present there, rf.json only if udir has none)
    copied from the first of the unit dirs srcs that has values.npz; -> that dir, or None if nothing was copied"""
    if (udir / 'values.npz').exists():
        return None
    for s in srcs:
        if (s / 'values.npz').is_file():
            udir.mkdir(parents=True, exist_ok=True)
            for fn in UNIT_CACHES:
                if (s / fn).is_file() and not (udir / fn).exists():
                    copy_file(s / fn, udir / fn)
            return s
    return None


def profiles(layers, P0, get_model, out=OUT):
    """lookviews.profiles, with its per-layer cache (profile_<layer>.npz) in out instead of runs/lookviews"""
    keep = lookviews.OUT
    lookviews.OUT = out
    try:
        return lookviews.profiles(layers, P0, get_model)
    finally:
        lookviews.OUT = keep


# ---------- values and gradients on a device ----------

def no_tf32():
    """full float32 on CUDA (no TF32 in convolutions or matmuls), so device values can match the CPU's"""
    import torch
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def forward_max_dev(model, u, ids, device, tag=''):
    """openmethod.forward_max with the input on device and the unit's maps moved to the CPU"""
    import torch
    Stop = bundlemax._bank().Stop
    got = {}

    def stop(_m, _i, out):
        assert out.shape[1] == NCH[u['layer']], tuple(out.shape)
        got['m'] = out[:, u['chan']].detach().cpu().numpy().astype(np.float32)
        raise Stop

    vals, rows, cols, sizes = [], [], [], set()
    h = dict(model.named_modules())[u['layer']].register_forward_hook(stop)
    try:
        for b, s in enumerate(range(0, len(ids), BATCH)):
            x = bundlemax.net_input(ids[s:s + BATCH]).to(device)
            got.clear()
            try:
                with torch.no_grad():
                    model(x)
            except Stop:
                pass
            v, r, c, n = openmethod.spatial_max(got['m'])
            vals.append(v)
            rows.append(r)
            cols.append(c)
            sizes.add(n)
            if b % 20 == 19:
                print(f'  {tag} {min(s + BATCH, len(ids))}/{len(ids)}', flush=True)
    finally:
        h.remove()
    assert len(sizes) == 1, sizes
    return np.concatenate(vals), np.concatenate(rows), np.concatenate(cols), sizes.pop()


def values_agree(a, b, rtol=CHECK_RTOL):
    """same shape and np.allclose(a, b, rtol)"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=rtol))


def check_device(u, ids, cpu_model, dev_model, device):
    """the unit's values on ids by forward_max_dev on device against openmethod.forward_max on the CPU; stops unless
    they agree (np.allclose, rtol 1e-3)"""
    ref = openmethod.forward_max(cpu_model, u, ids)[0].astype(np.float64)
    got = forward_max_dev(dev_model, u, ids, device)[0].astype(np.float64)
    diff = float(np.abs(got - ref).max()) if got.shape == ref.shape and len(ref) else float('nan')
    if not values_agree(got, ref):
        raise SystemExit(f'{u["name"]}: values on {device} differ from the CPU on the first {len(ids)} card-pool '
                         f'photos (max abs diff {diff:.4g}); stopping')
    print(f'{u["name"]} values on {device} match the CPU on the first {len(ids)} card-pool photos '
          f'(max abs diff {diff:.3g})', flush=True)


def unit_values(u, C, H, udir, get_model, device):
    """openmethod.unit_values (values.npz, same keys, cache check); new values by forward_max_dev on device, after
    check_device on the first N_CHECK card-pool photos"""
    if (udir / 'values.npz').exists():
        return openmethod.unit_values(u, C, H, udir, None)  # cached: checked there, no model is used
    model = get_model(device)
    check_device(u, C[:N_CHECK], get_model('cpu'), model, device)
    v, r, c, size = forward_max_dev(model, u, np.concatenate([C, H]), device, tag=u['name'])
    n = len(C)
    d = {'unit': np.array(u['name']), 'C': C, 'H': H, 'vC': v[:n], 'vH': v[n:], 'rC': r[:n], 'cC': c[:n],
         'rH': r[n:], 'cH': c[n:], 'size': np.array(size)}
    tmp = udir / 'values.tmp.npz'
    np.savez_compressed(tmp, **d)
    tmp.replace(udir / 'values.npz')
    return d


def input_grad_dev(model, u, ids, rows, cols, size, device):
    """lookviews.input_grad with the input on device and the outputs moved to the CPU"""
    import torch
    Stop = bundlemax._bank().Stop
    got = {}

    def stop(_m, _i, out):
        assert out.shape[1] == NCH[u['layer']], tuple(out.shape)
        got['o'] = out
        raise Stop

    h = dict(model.named_modules())[u['layer']].register_forward_hook(stop)
    try:
        x = bundlemax.net_input(ids).to(device).requires_grad_(True)
        with torch.enable_grad():
            try:
                model(x)
            except Stop:
                pass
            o = got['o'][:, u['chan']]
            assert tuple(o.shape[1:]) == (size, size), (tuple(o.shape), size)
            dv = o.device
            v = o[torch.arange(len(ids), device=dv), torch.as_tensor(np.asarray(rows, dtype=np.int64), device=dv),
                  torch.as_tensor(np.asarray(cols, dtype=np.int64), device=dv)]
            gx, = torch.autograd.grad(v.sum(), x)
    finally:
        h.remove()
    assert tuple(gx.shape) == (len(ids), 3, TILE, TILE), tuple(gx.shape)
    return (gx.abs().sum(dim=1).detach().cpu().numpy().astype(np.float64),
            v.detach().cpu().numpy().astype(np.float64))


def receptive_field(u, d, udir, get_model, device):
    """lookviews.receptive_field (rf.json, same keys, cache check); new gradients by input_grad_dev on device"""
    if (udir / 'rf.json').exists():
        return lookviews.receptive_field(u, d, udir, None)  # cached: checked there, no model is used
    top = openmethod.top_positions(d['vC'])
    ids, rows, cols = d['C'][top], d['rC'][top], d['cC'][top]
    size = int(d['size'])
    centres = [lookviews.cell_centre(int(r), int(c), size) for r, c in zip(rows, cols)]
    key = {'unit': u['name'], 'top_ids': ids.tolist(), 'rows': rows.tolist(), 'cols': cols.tolist(), 'map_size': size}
    print(u['name'], 'receptive field on', device, flush=True)
    model = get_model(device)
    B = lookviews.RF_BATCH
    parts = [input_grad_dev(model, u, ids[s:s + B], rows[s:s + B], cols[s:s + B], size, device)
             for s in range(0, len(ids), B)]
    g, v = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    ref = d['vC'][top].astype(np.float64)
    M = float(d['vC'].max())
    assert np.allclose(v, ref, rtol=1e-3, atol=1e-3 * M), ('value at the stored (row, col) differs from values.npz',
                                                           v.tolist(), ref.tolist())
    s = [lookviews.rf_side(g[n], *centres[n]) for n in range(len(ids))]
    S = lookviews.side_S(s)
    rf = {**key, 'S': S, 's': s, 'centres': [list(c) for c in centres], 'g_total': g.sum(axis=(1, 2)).tolist(),
          'crop_boxes': [list(lookviews.crop_box(cx, cy, S)) for cx, cy in centres]}
    lookviews.save_json(rf, udir / 'rf.json')
    return rf


# ---------- view S ----------

def bin_of(r):
    """index 0-3 of the bin of ratio r: [0.6, 1.0], [0.4, 0.6), [0.2, 0.4), [0.05, 0.2); None outside them (below
    0.05, above 1.0, NaN)"""
    x = float(r)
    for b, (lo, hi) in enumerate(BINS):
        if lo <= x < hi or (b == 0 and x == hi):
            return b
    return None


def bin_members(ids, r, exclude):
    """the ids of each bin, in bin order, each list ascending; ids in exclude left out. ids and r: aligned arrays"""
    ids = np.asarray(ids, dtype=np.int64)
    r = np.asarray(r, dtype=np.float64)
    assert ids.ndim == 1 and ids.shape == r.shape, (ids.shape, r.shape)
    ex = {int(i) for i in exclude}
    out = [[] for _ in BINS]
    for i, x in zip(ids.tolist(), r.tolist()):
        b = bin_of(x)
        if b is not None and i not in ex:
            out[b].append(i)
    return [sorted(m) for m in out]


def stratified_pick(ids, r, exclude, rng, per_bin=PER_BIN):
    """from each bin in bin order, min(per_bin, its size) of its ids drawn without replacement by rng.choice on the
    bin's ids in ascending order; ids in exclude are in no bin. -> 4 lists of ids, each in draw order"""
    out = []
    for m in bin_members(ids, r, exclude):
        k = min(per_bin, len(m))
        out.append([int(i) for i in rng.choice(np.asarray(m, dtype=np.int64), k, replace=False)] if k else [])
    return out


# ---------- view N ----------

def top_labels(card_labels, card_r, k=N_LABELS):
    """up to k labels of the card photos, the most common first; ties: the higher sum of r over the card photos of the
    label, then the lower label"""
    labs = [int(x) for x in card_labels]
    rs = [float(x) for x in card_r]
    assert len(labs) == len(rs), (len(labs), len(rs))
    by = {}
    for lab, x in zip(labs, rs):
        by.setdefault(lab, []).append(x)
    return sorted(by, key=lambda lab: (-len(by[lab]), -math.fsum(by[lab]), lab))[:k]


def least_active(ids, values, labels, label, exclude, n=PER_LABEL):
    """the n ids of that label with the lowest value (ties: lower id), ids in exclude left out. ids and values: aligned
    arrays; labels: aligned with ids or, if its length differs, the label of every photo, indexed by id"""
    ids = np.asarray(ids, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    assert ids.ndim == 1 and ids.shape == values.shape, (ids.shape, values.shape)
    lab = labels if len(labels) == len(ids) else labels[ids]
    ex = np.array(sorted({int(i) for i in exclude}), dtype=np.int64)
    m = (lab == label) & ~np.isin(ids, ex)
    cand, v = ids[m], values[m]
    order = np.lexsort((cand, v))  # by value, then by id
    return [int(i) for i in cand[order[:n]]]


# ---------- grids ----------

def range_slots(picks, per_bin=PER_BIN):
    """cells of view S: one row of per_bin cells per bin, its photos first, then empty cells; photos numbered from 1
    across the grid. -> [(number, id) or None]"""
    out, n = [], 0
    for row in picks:
        assert len(row) <= per_bin, (len(row), per_bin)
        for j in range(per_bin):
            if j < len(row):
                n += 1
                out.append((n, int(row[j])))
            else:
                out.append(None)
    return out


def nearmiss_slots(groups, per_label=PER_LABEL):
    """cells of view N: per_label cells per label in order, label j's photos numbered per_label * j + 1, ...; a label
    with fewer photos leaves the rest of its cells empty. -> [(number, id) or None]"""
    out = []
    for j, g in enumerate(groups):
        assert len(g) <= per_label, (len(g), per_label)
        for k in range(per_label):
            out.append((per_label * j + k + 1, int(g[k])) if k < len(g) else None)
    return out


def boxed_grid(cells, M, path=None, cols=COLS, load=bundlemax.photo):
    """openmethod.make_card_boxed on cells in row-major order: 5 columns of whole 224 px photos, the box drawn on each,
    '#n   activation k' (bundlemax.act_k on M) in a white strip under each; a None cell stays white"""
    rows = -(-len(cells) // cols)
    im = Image.new('RGB', (cols * TILE, rows * (TILE + LAB)), 'white')
    dr, f = ImageDraw.Draw(im), bundlemax._font(16)
    for k, cell in enumerate(cells):
        if cell is None:
            continue
        n, i, v, b = cell
        x0, y0 = (k % cols) * TILE, (k // cols) * (TILE + LAB)
        p = load(i)
        openmethod.draw_box(p, b)
        im.paste(p, (x0, y0))
        dr.text((x0 + 4, y0 + TILE + 3), f'#{n}   activation {bundlemax.act_k(v, M)}', fill='black', font=f)
    if path is not None:
        im.save(path, quality=90)
    return im


def make_views(name, d, labels, udir, load=bundlemax.photo):
    """range.jpg (view S) and nearmiss.jpg (view N) in udir, on the card's scale M and with the card's peak boxes;
    labels = the label of every photo, indexed by id. -> what they show"""
    C, vC, size = np.asarray(d['C'], dtype=np.int64), np.asarray(d['vC'], dtype=np.float64), int(d['size'])
    M = float(vC.max())
    assert M > 0, ('M <= 0', M)
    r = vC / M
    top = openmethod.top_positions(vC)
    card = set(C[top].tolist())
    at = {i: k for k, i in enumerate(C.tolist())}

    def cells(slots):
        out = []
        for s in slots:
            if s is None:
                out.append(None)
                continue
            n, i = s
            k = at[i]
            out.append((n, i, vC[k], openmethod.box_px(int(d['rC'][k]), int(d['cC'][k]), size)))
        return out

    sizes = [len(m) for m in bin_members(C, r, card)]
    picks = stratified_pick(C, r, card, np.random.default_rng(zlib.crc32(name.encode())))
    boxed_grid(cells(range_slots(picks)), M, udir / 'range.jpg', load=load)
    lC = np.asarray(labels)[C]
    labs = top_labels(lC[top], r[top])
    near = [least_active(C, vC, lC, lab, card) for lab in labs]
    boxed_grid(cells(nearmiss_slots(near)), M, udir / 'nearmiss.jpg', load=load)
    return {'M': M, 'card_ids': C[top].tolist(), 'bin_sizes': sizes, 'range_ids': picks, 'near_labels': labs,
            'near_ids': near, 'near_ratios': [[float(r[at[i]]) for i in g] for g in near]}


# ---------- conditions and readings ----------

def condition_contents(cp, range_block, near_block):
    """{condition: content blocks} in CONDITIONS order: CP as given (lookviews' CP, ending with run_subjects.TASK);
    CPS, CPN and CPSN with the S blocks, the N blocks, or S then N inserted just before the task"""
    cp = list(cp)
    if not cp or cp[-1] != {'type': 'text', 'text': run_subjects.TASK}:
        raise ValueError('CP must end with run_subjects.TASK')
    head, task = cp[:-1], cp[-1]
    s = [{'type': 'text', 'text': S_TEXT}, range_block]
    n = [{'type': 'text', 'text': N_TEXT}, near_block]
    return {'CP': cp, 'CPS': head + s + [task], 'CPN': head + n + [task], 'CPSN': head + s + n + [task]}


def readings(name, contents, udir, call=None):
    """every (condition, sample) reading as openmethod.get_reading with system prompt '', up to WORKERS in parallel,
    cached in readings.json keyed '<condition>_s<k>' (as lookviews.readings); a failed call (is_error) is not cached"""
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
            lookviews.save_json(cache, path)
    return cache


# ---------- report ----------

def near_rows(v, cats):
    """[(label, class name, [(number, photo id, ratio)])] of view N, numbered as nearmiss_slots"""
    out = []
    for j, (lab, ids, rs) in enumerate(zip(v['near_labels'], v['near_ids'], v['near_ratios'])):
        out.append((lab, cats[lab], [(PER_LABEL * j + k + 1, i, x) for k, (i, x) in enumerate(zip(ids, rs))]))
    return out


def section(name, sizes, near, recs):
    L = [f'== {name}', 'S bin sizes (card pool without the 20 card photos): '
         + ', '.join(f'{nm} {n}' for nm, n in zip(BIN_NAMES, sizes))]
    for j, (lab, cname, items) in enumerate(near, 1):
        L.append(f'N {j}: {cname} (class {lab})  '
                 + (', '.join(f'#{n} photo {i} ratio {x:.3f}' for n, i, x in items) or 'no photos'))
    L += [lookviews.reading_line(rid, recs.get(rid)) for rid in READINGS]
    return '\n'.join(L)


# ---------- selftest ----------

def selftest():
    import copy
    import tempfile
    import threading
    from collections import OrderedDict

    # bins: every boundary and just below each
    def below(x):
        return float(np.nextafter(x, -np.inf))
    for x, b in [(1.0, 0), (0.6, 0), (0.4, 1), (0.2, 2), (0.05, 3), (0.8, 0), (0.5, 1), (0.3, 2), (0.1, 3)]:
        assert bin_of(x) == b, (x, bin_of(x))
    assert [bin_of(below(x)) for x in (1.0, 0.6, 0.4, 0.2, 0.05)] == [0, 1, 2, 3, None]
    assert bin_of(0.0) is None and bin_of(-1.0) is None and bin_of(float('nan')) is None
    assert bin_of(float(np.nextafter(1.0, 2.0))) is None  # above 1.0 (cannot occur on the card pool): no bin
    assert bin_of(np.float32(0.6)) == 0 and bin_of(np.float64(0.05)) == 3

    # stratified pick: bins, exclusion, short and empty bins, the rng rule, determinism
    ids = np.arange(100, 160)
    r = np.concatenate([np.linspace(0.6, 1.0, 12), [0.4, 0.5, below(0.6)], np.linspace(0.05, below(0.2), 25),
                        np.linspace(0.0, below(0.05), 20)])
    assert len(r) == 60 and [bin_of(x) for x in r] == [0] * 12 + [1] * 3 + [3] * 25 + [None] * 20
    ex = {100, 101, 113, 115, 999}
    members = bin_members(ids, r, ex)
    assert members == [list(range(102, 112)), [112, 114], [], list(range(116, 140))], members
    picks = stratified_pick(ids, r, ex, np.random.default_rng(7))
    assert [len(p) for p in picks] == [5, 2, 0, 5] and sorted(picks[1]) == [112, 114]
    assert all(set(p) <= set(m) and len(set(p)) == len(p) for p, m in zip(picks, members))
    assert not ex & {i for p in picks for i in p}
    assert picks == stratified_pick(ids, r, ex, np.random.default_rng(7)), 'not reproducible'
    g = np.random.default_rng(7)
    want = [g.choice(np.array(members[0]), 5, replace=False).tolist(),
            g.choice(np.array(members[1]), 2, replace=False).tolist(), [],
            g.choice(np.array(members[3]), 5, replace=False).tolist()]
    assert picks == want, (picks, want)
    perm = np.random.default_rng(3).permutation(60)
    assert stratified_pick(ids[perm], r[perm], ex, np.random.default_rng(7)) == picks  # input order does not matter
    assert stratified_pick(ids, r, ex, np.random.default_rng(8)) != picks
    assert [len(p) for p in stratified_pick(ids, r, set(), np.random.default_rng(7), per_bin=3)] == [3, 3, 0, 3]
    assert stratified_pick(ids, r, set(range(100, 140)), np.random.default_rng(7)) == [[], [], [], []]

    # top labels: count, then the sum of r, then the lower label; fewer when fewer distinct labels
    cl = [7, 7, 7, 2, 2, 2, 9, 9, 4, 4, 5, 5, 1, 3, 8, 6, 0, 11, 12, 13]
    cr = [0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.5, 0.375] + [1.0] * 8
    assert top_labels(cl, cr) == [7, 2, 5, 4, 9], top_labels(cl, cr)
    assert top_labels(cl, cr, k=7) == [7, 2, 5, 4, 9, 0, 1] and top_labels(cl, cr, k=2) == [7, 2]
    assert top_labels(np.array(cl), np.array(cr), k=100) == [7, 2, 5, 4, 9, 0, 1, 3, 6, 8, 11, 12, 13]
    assert top_labels([3, 3, 1], [0.1, 0.2, 0.9]) == [3, 1]
    assert top_labels([8, 2, 6], [0.5, 0.5, 0.75]) == [6, 2, 8]
    assert top_labels([4, 4, 4, 4], [1.0, 0.9, 0.8, 0.7]) == [4]

    # least active: lowest value, ties by lower id, exclusion, fewer when short
    li = np.array([10, 11, 12, 13, 14, 15, 16, 17])
    lv = np.array([0.3, 0.1, 0.1, 0.0, 0.2, 0.0, 0.5, 0.1])
    ll = np.array([1, 1, 1, 2, 1, 1, 1, 2])
    assert least_active(li, lv, ll, 1, set()) == [15, 11]
    assert least_active(li, lv, ll, 1, set(), n=3) == [15, 11, 12]
    assert least_active(li, lv, ll, 1, {15, 11}) == [12, 14]
    assert least_active(li, lv, ll, 2, set()) == [13, 17] and least_active(li, lv, ll, 2, {13}) == [17]
    assert least_active(li, lv, ll, 3, set()) == [] and least_active(li, lv, ll, 1, set(li.tolist())) == []
    q = np.random.default_rng(5).permutation(8)
    assert least_active(li[q], lv[q], ll[q], 1, set(), n=6) == [15, 11, 12, 14, 10, 16]
    full = np.full(30, -1)
    full[li] = ll
    assert least_active(li, lv, full, 1, {15}) == [11, 12]  # labels of every photo, indexed by id

    # grid cells and numbering
    rs_ = range_slots([[5, 6, 7, 8, 9], [1, 2], [], [3]])
    assert len(rs_) == 20 and rs_[:7] == [(1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 1), (7, 2)]
    assert rs_[7:15] == [None] * 8 and rs_[15] == (8, 3) and rs_[16:] == [None] * 4
    assert nearmiss_slots([[4, 5], [6], [], [7, 8]]) == [(1, 4), (2, 5), (3, 6), None, None, None, (7, 7), (8, 8)]

    # the boxed grid, on synthetic photos
    rng = np.random.default_rng(0)
    pics = {i: Image.fromarray(rng.integers(0, 256, (TILE, TILE, 3), dtype=np.uint8)) for i in (3, 8, 11)}

    def load(i):
        return pics[int(i)].copy()
    bx = [openmethod.box_px(0, 0, 14), openmethod.box_px(5, 7, 14), openmethod.box_px(13, 13, 14)]
    cells = [(1, 3, 2.0, bx[0]), None, (2, 8, 1.0, bx[1]), None, None, (3, 11, 0.5, bx[2])]
    im = boxed_grid(cells, 2.0, load=load)
    assert im.size == (COLS * TILE, 2 * (TILE + LAB)), im.size
    a = np.asarray(im)
    for k in (0, 2, 5):
        _, i, _, b = cells[k]
        x0, y0 = (k % COLS) * TILE, (k // COLS) * (TILE + LAB)
        want_im = load(i)
        openmethod.draw_box(want_im, b)
        assert np.array_equal(a[y0:y0 + TILE, x0:x0 + TILE], np.asarray(want_im)), k
        assert (a[y0 + TILE:y0 + TILE + LAB, x0:x0 + TILE].min(axis=2) < 128).any(), k
    for k in (1, 3, 4, 6, 7, 8, 9):
        x0, y0 = (k % COLS) * TILE, (k // COLS) * (TILE + LAB)
        assert (a[y0:y0 + TILE + LAB, x0:x0 + TILE] == 255).all(), k

    # both views on a synthetic unit
    rng = np.random.default_rng(11)
    Cs = np.arange(0, 900, 3)
    vCs = (rng.random(300) ** 2).astype(np.float32)
    ds = {'C': Cs, 'vC': vCs, 'rC': rng.integers(0, 14, 300), 'cC': rng.integers(0, 14, 300), 'size': np.array(14)}
    labs_all = (np.arange(900) // 3) % 7

    def flat(i):
        return Image.new('RGB', (TILE, TILE), (int(i) % 256, 80, 160))
    with tempfile.TemporaryDirectory() as td:
        v = make_views('mixed4c:447', ds, labs_all, Path(td), load=flat)
        assert v == make_views('mixed4c:447', ds, labs_all, Path(td), load=flat), 'views not reproducible'
        with Image.open(Path(td) / 'range.jpg') as im:
            assert im.size == (COLS * TILE, 4 * (TILE + LAB)), im.size
        with Image.open(Path(td) / 'nearmiss.jpg') as im:
            assert im.size == (COLS * TILE, -(-PER_LABEL * len(v['near_labels']) // COLS) * (TILE + LAB)), im.size
    r_s = vCs.astype(np.float64) / float(vCs.max())
    top_s = np.argsort(-vCs.astype(np.float64), kind='stable')[:20]
    card_s = set(Cs[top_s].tolist())
    assert v['card_ids'] == Cs[top_s].tolist() and len(card_s) == 20
    sz = [sum(1 for k in range(300) if int(Cs[k]) not in card_s and bin_of(r_s[k]) == b) for b in range(4)]
    assert v['bin_sizes'] == sz and all(s > 5 for s in sz), sz
    for b, row in enumerate(v['range_ids']):
        assert len(row) == min(PER_BIN, sz[b]) and len(set(row)) == len(row)
        assert all(i not in card_s and bin_of(r_s[i // 3]) == b for i in row)
    assert v['range_ids'] == stratified_pick(Cs, r_s, card_s, np.random.default_rng(zlib.crc32(b'mixed4c:447')))
    assert v['near_labels'] == top_labels(labs_all[Cs[top_s]], r_s[top_s]) and len(v['near_labels']) == 5
    for lab, got_ids, rs in zip(v['near_labels'], v['near_ids'], v['near_ratios']):
        rest = [k for k in range(300) if labs_all[Cs[k]] == lab and int(Cs[k]) not in card_s]
        assert len(got_ids) == min(PER_LABEL, len(rest)) and all(labs_all[i] == lab for i in got_ids)
        others = [vCs[k] for k in rest if int(Cs[k]) not in got_ids]
        assert all(vCs[i // 3] <= min(others) for i in got_ids)
        assert rs == [float(r_s[i // 3]) for i in got_ids]

    # content order of each condition; the verbatim texts
    assert S_TEXT == ' '.join([
        'The next grid shows photos drawn at random from four activation ranges below the strongest photos, one',
        'row each: 60-100%, 40-60%, 20-40% and 5-20% of the strongest activation (fewer when a range has fewer',
        'photos). Same box and scale as the first grid.'])
    assert N_TEXT == ' '.join([
        'The next grid shows near misses: for each of the 5 kinds of photo most common among the strongest',
        'photos, the 2 photos of that kind on which the unit is least active (#1-#2 are one kind, #3-#4 the next,',
        'and so on). Same box and scale as the first grid.'])
    cb, rb, xb, sb, nb = ({'type': 'image', 'k': k} for k in ('card', 'random', 'crops', 'range', 'nearmiss'))
    nm = bundlemax.card_name('mixed4c:447', 'whole')
    ct = openmethod.card_content(nm, cb, rb)[0]['text']
    cp = lookviews.condition_contents(ct, cb, rb, xb, 'PROFILE', 48)['CP']
    cs = condition_contents(cp, sb, nb)

    def kinds(c):
        return [b['k'] if 'k' in b else b['text'] for b in c]
    T = run_subjects.TASK
    head = [ct + ' ' + lookviews.CROP_SENTENCE.replace('{S}', '48'), 'card', 'random', 'crops', 'PROFILE']
    assert list(cs) == list(CONDITIONS) and cs['CP'] == cp
    assert kinds(cs['CP']) == head + [T]
    assert kinds(cs['CPS']) == head + [S_TEXT, 'range', T]
    assert kinds(cs['CPN']) == head + [N_TEXT, 'nearmiss', T]
    assert kinds(cs['CPSN']) == head + [S_TEXT, 'range', N_TEXT, 'nearmiss', T]
    assert all(b['type'] == 'text' for c in cs.values() for b in c if 'k' not in b)
    assert kinds(cp) == head + [T]  # CP itself is not changed
    try:
        condition_contents(cp[:-1], sb, nb)
        raise AssertionError('CP without the task accepted')
    except ValueError:
        pass
    assert READINGS == ['CP_s1', 'CP_s2', 'CP_s3', 'CPS_s1', 'CPS_s2', 'CPS_s3', 'CPN_s1', 'CPN_s2', 'CPN_s3',
                        'CPSN_s1', 'CPSN_s2', 'CPSN_s3'] and WORKERS == 6

    # readings cache: 12 keys, an is_error reply not cached, content changes stop
    lock, calls = threading.Lock(), []

    def fake(n_err):
        state = {'err': n_err}

        def call(sp, content):
            assert sp == ''
            with lock:
                calls.append(len(content))
                if state['err'] and len(content) == 10:
                    state['err'] -= 1
                    return {'text': 'overloaded', 'is_error': True}
            return {'text': '{"class": "one", "explanation": "n=%d"}' % len(content), 'is_error': False}
        return call
    with tempfile.TemporaryDirectory() as td:
        rec = readings('mixed4c:447', cs, Path(td), call=fake(1))
        assert len(rec) == 11 and len(calls) == 12 and sum(k.startswith('CPSN_') for k in rec) == 2
        assert rec['CP_s3']['explanation'] == 'n=6' and rec['CPS_s2']['explanation'] == 'n=8'
        assert rec['CPN_s1']['explanation'] == 'n=8' and rec['CPN_s1']['condition'] == 'CPN'
        assert rec['CPN_s1']['sample'] == 1 and rec['CPN_s1']['system_prompt'] == ''
        calls.clear()
        rec = readings('mixed4c:447', cs, Path(td), call=fake(0))
        assert sorted(rec) == sorted(READINGS) and calls == [10]
        calls.clear()
        cs2 = dict(cs, CPN=cs['CPN'][:-1] + [{'type': 'text', 'text': 'OTHER'}])
        try:
            readings('mixed4c:447', cs2, Path(td), call=fake(0))
            raise AssertionError('changed content accepted')
        except SystemExit:
            pass
        assert calls == []

    # report lines
    vv = {'near_labels': [281, 5], 'near_ids': [[12, 30], [7]], 'near_ratios': [[0.25, 0.5], [0.125]]}
    cats_t = [f'c{k}' for k in range(1000)]
    assert near_rows(vv, cats_t) == [(281, 'c281', [(1, 12, 0.25), (2, 30, 0.5)]), (5, 'c5', [(3, 7, 0.125)])]
    sec = section('mixed4c:447', [3, 10, 0, 40], [(281, 'tabby', [(1, 12, 0.25), (2, 30, 0.5)]), (5, 'x y', [])],
                  {'CP_s1': {'class': 'one', 'explanation': 'a\n  b'}}).splitlines()
    assert sec[0] == '== mixed4c:447'
    assert sec[1] == 'S bin sizes (card pool without the 20 card photos): 60-100% 3, 40-60% 10, 20-40% 0, 5-20% 40'
    assert sec[2] == 'N 1: tabby (class 281)  #1 photo 12 ratio 0.250, #2 photo 30 ratio 0.500', sec[2]
    assert sec[3] == 'N 2: x y (class 5)  no photos'
    assert sec[4] == 'CP_s1  one  a b' and sec[5] == 'CP_s2  no reading (call failed, not cached)'
    assert len(sec) == 4 + len(READINGS)

    # copied caches
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        a_, b_, dst = td / 'a' / 'u', td / 'b' / 'u', td / 'dst' / 'u'
        for p in (a_, b_, dst):
            p.mkdir(parents=True)
        (b_ / 'values.npz').write_text('B values')
        (b_ / 'rf.json').write_text('B rf')
        assert copy_unit_caches(dst, [a_, b_]) == b_
        assert (dst / 'values.npz').read_text() == 'B values' and (dst / 'rf.json').read_text() == 'B rf'
        (a_ / 'values.npz').write_text('A values')
        (a_ / 'rf.json').write_text('A rf')
        (dst / 'rf.json').unlink()
        assert copy_unit_caches(dst, [a_, b_]) is None and not (dst / 'rf.json').exists()  # values here: none copied
        (a_ / 'rf.json').unlink()
        dst2 = td / 'dst2' / 'u'
        assert copy_unit_caches(dst2, [a_, b_]) == a_  # the first with values.npz; rf.json only from it
        assert (dst2 / 'values.npz').read_text() == 'A values' and not (dst2 / 'rf.json').exists()
        assert copy_unit_caches(td / 'dst3', [td / 'none1', td / 'none2']) is None and not (td / 'dst3').exists()
        (td / 'b' / 'profile_mixed4c.npz').write_text('P')
        assert copy_first('profile_mixed4c.npz', [td / 'a', td / 'b'], td / 'out') == td / 'b'
        assert (td / 'out' / 'profile_mixed4c.npz').read_text() == 'P'
        assert copy_first('profile_mixed4c.npz', [td / 'a', td / 'b'], td / 'out') is None  # already here
        assert copy_first('profile_mixed5b.npz', [td / 'a', td / 'b'], td / 'out') is None

        # profiles: lookviews.profiles with the cache in another dir, lookviews.OUT restored
        P0t = np.arange(0, 30, 3)
        pd = {'layer': np.array('mixed4c'), 'P0': P0t, 'transforms': np.array(list(battery.TRANSFORMS)),
              'orig': np.ones((10, 512), np.float32)}
        np.savez_compressed(td / 'profile_mixed4c.npz', **pd)
        keep = lookviews.OUT
        got = profiles(['mixed4c'], P0t, None, out=td)
        assert lookviews.OUT is keep and np.array_equal(got['mixed4c']['orig'], pd['orig'])
        try:
            profiles(['mixed4c'], P0t + 1, None, out=td)
            raise AssertionError('profile for other photos accepted')
        except SystemExit:
            pass
        assert lookviews.OUT is keep

    # values and gradients on a device, on a small stand-in module (not the network)
    import torch
    torch.manual_seed(0)
    no_tf32()

    class Boom(torch.nn.Module):
        def forward(self, x):
            raise AssertionError('ran past the stop')

    fm = torch.nn.Sequential(OrderedDict([('mixed4c', torch.nn.Conv2d(3, 512, 16, stride=16)),
                                          ('boom', Boom())])).eval()
    for p in fm.parameters():
        p.requires_grad_(False)
    u = openmethod.parse_unit('mixed4c:5')
    ids3 = np.arange(3)
    ref = openmethod.forward_max(fm, u, ids3)
    got = forward_max_dev(fm, u, ids3, 'cpu')
    assert all(np.array_equal(x, y) for x, y in zip(ref[:3], got[:3])) and ref[3] == got[3] == 14
    assert values_agree([1.0, 2.0], [1.0005, 2.0]) and not values_agree([1.0, 2.0], [1.002, 2.0])
    assert not values_agree([1.0, 2.0], [1.0, 2.0, 3.0])
    check_device(u, ids3, fm, fm, 'cpu')
    off = copy.deepcopy(fm)
    with torch.no_grad():
        off.mixed4c.weight.mul_(1.01)
        off.mixed4c.bias.mul_(1.01)
    try:
        check_device(u, ids3, fm, off, 'cpu')
        raise AssertionError('differing device values accepted')
    except SystemExit:
        pass
    g1, v1 = input_grad_dev(fm, u, np.array([0, 1]), [2, 13], [9, 0], 14, 'cpu')
    g2, v2 = lookviews.input_grad(fm, u, np.array([0, 1]), [2, 13], [9, 0], 14)
    assert g1.shape == (2, TILE, TILE) and np.array_equal(g1, g2) and np.array_equal(v1, v2)

    def gm(dev='cpu'):
        return fm
    with tempfile.TemporaryDirectory() as td:
        p1, p2 = Path(td) / 'a', Path(td) / 'b'
        p1.mkdir()
        p2.mkdir()
        Cu, Hu = np.arange(8), np.arange(8, 12)
        d1 = unit_values(u, Cu, Hu, p1, gm, 'cpu')
        d2 = openmethod.unit_values(u, Cu, Hu, p2, lambda: fm)
        assert sorted(d1) == sorted(d2) and all(np.array_equal(d1[k], d2[k]) for k in d1)
        d3 = unit_values(u, Cu, Hu, p1, None, 'cpu')  # cached
        assert sorted(d3) == sorted(d1) and all(np.array_equal(d3[k], d1[k]) for k in d1)
        rf1 = receptive_field(u, d1, p1, gm, 'cpu')
        rf2 = lookviews.receptive_field(u, d1, p2, lambda: fm)
        assert rf1 == rf2
        assert receptive_field(u, d1, p1, None, 'cpu') == rf1  # cached
    if torch.cuda.is_available():
        fc = copy.deepcopy(fm).to('cuda')
        gotc = forward_max_dev(fc, u, ids3, 'cuda')
        assert gotc[3] == 14 and np.allclose(gotc[0], ref[0], rtol=1e-3, atol=1e-3 * float(np.abs(ref[0]).max()))
        gc1, vc1 = input_grad_dev(fc, u, np.array([0, 1]), [2, 13], [9, 0], 14, 'cuda')
        assert np.allclose(vc1, v1, rtol=1e-3, atol=1e-3 * float(np.abs(v1).max()))
        assert np.allclose(gc1, g1, rtol=1e-3, atol=1e-3 * float(g1.max()))
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
    names = json.loads(nearmiss2.NAMES.read_text())
    C, H = bundlemax.pools(names)
    with np.load(nearmiss2.ACTS) as z:
        labels = np.asarray(z['classes'], dtype=np.int64)
    assert len(labels) == len(names) and labels.min() >= 0 and labels.max() < 1000, (len(labels), len(names))
    from torchvision.models import GoogLeNet_Weights
    cats = list(GoogLeNet_Weights.IMAGENET1K_V1.meta['categories'])
    assert len(cats) == 1000, len(cats)
    P0 = np.sort(np.random.default_rng(0).choice(C, lookviews.N_PROFILE, replace=False))
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    no_tf32()
    print('new values and gradients on', device, flush=True)
    held = {}

    def get_model(dev='cpu'):
        if dev not in held:
            torch.set_num_threads(4)
            m = bundlemax._bank().load_model(openmethod.WEIGHTS)
            for q in m.parameters():
                q.requires_grad_(False)
            held[dev] = m.to(dev)
        return held[dev]

    OUT.mkdir(parents=True, exist_ok=True)

    # 1. invariance profile of every layer that has a unit, as lookviews (CPU)
    layers = sorted({u['layer'] for u in units}, key=lookviews.LAYERS.index)
    for L in layers:
        src = copy_first(f'profile_{L}.npz', SOURCES, OUT)
        if src is not None:
            print(L, 'profile cache copied from', src, flush=True)
    prof = profiles(layers, P0, get_model)
    R = {L: lookviews.retention_table(prof[L]) for L in layers}

    # 2. every unit: values, receptive field, crops
    vals, rfs = {}, {}
    for u in units:
        n = u['name']
        print('==', n, flush=True)
        udir = OUT / n
        udir.mkdir(parents=True, exist_ok=True)
        src = copy_unit_caches(udir, [s / n for s in SOURCES])
        if src is not None:
            print(n, 'caches copied from', src, flush=True)
        d = unit_values(u, C, H, udir, get_model, device)
        pos = np.searchsorted(d['C'], P0)
        assert np.array_equal(d['C'][pos], P0), 'P0 is not inside C'
        M = float(d['vC'].max())
        assert np.allclose(prof[u['layer']]['orig'][:, u['chan']], d['vC'][pos], rtol=1e-3, atol=1e-3 * M), \
            f'{n}: profile originals differ from values.npz'
        rf = receptive_field(u, d, udir, get_model, device)
        boxes = [lookviews.crop_box(cx, cy, rf['S']) for cx, cy in rf['centres']]
        lookviews.crops_grid([bundlemax.photo(i) for i in rf['top_ids']], boxes).save(udir / 'crops.jpg', quality=90)
        vals[n], rfs[n] = d, rf
    del prof
    held.clear()
    gc.collect()

    # 3. every unit: card, views S and N, the four contents, readings; report
    secs, allr = [], {}
    for u in units:
        n = u['name']
        udir = OUT / n
        d, rf = vals[n], rfs[n]
        meta, content = openmethod.make_cards(u, d, udir)
        assert meta['top_ids'] == rf['top_ids'], f'{n}: card photos differ from rf.json'
        assert [b['type'] for b in content] == ['text', 'image', 'image', 'text']
        assert content[3]['text'] == run_subjects.TASK
        cnt, rows = lookviews.unit_profile(R[u['layer']], u['chan'])
        cp = lookviews.condition_contents(content[0]['text'], content[1], content[2],
                                          run_subjects.img_block(udir / 'crops.jpg'),
                                          lookviews.profile_text(cnt, rows, desc), rf['S'])['CP']
        v = make_views(n, d, labels, udir)
        assert v['card_ids'] == meta['top_ids'], f'{n}: view photos excluded differ from the card photos'
        contents = condition_contents(cp, run_subjects.img_block(udir / 'range.jpg'),
                                      run_subjects.img_block(udir / 'nearmiss.jpg'))
        recs = readings(n, contents, udir)
        allr[n] = {rid: (recs[rid]['explanation'] if rid in recs else None) for rid in READINGS}
        secs.append(section(n, v['bin_sizes'], near_rows(v, cats), recs))
        lookviews.save_json(allr, OUT / 'readings_all.json')
        (OUT / 'report.txt').write_text('\n\n'.join(secs) + '\n')
        print(secs[-1], flush=True)
        del content, contents, cp
        gc.collect()
    print('report ->', OUT / 'report.txt')


if __name__ == '__main__':
    main()
