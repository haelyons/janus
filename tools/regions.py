"""Do labelled image regions push a channel up or down?

80 photos from the held pool (names[i] with crc32 % 10 >= 7): 60 drawn at random plus the 20 other held photos with
the highest spatial max of the recorded pre-ReLU map (bank/img/acts.npz, channel given as --unit). A vision model that
never sees any activation names, on a 7x7 grid over each photo, the cells holding a pointed shape and the cells holding
a round shape. Cell value = mean of the 2x2 block of the 14x14 map inside the cell; P = pointed only, R = round only,
O = neither, cells in both lists dropped. Per photo with a P and an R cell d_PR = mean(P) - mean(R); on those photos,
where O is non-empty, also d_PO = mean(P) - mean(O) and d_RO = mean(R) - mean(O). Each is summarised with a two-sided
sign test; d_PR is also compared with every mixed4e channel's d_PR on its post-ReLU map, divided by the channel's
strong level (median spatial max over its top-20 battery pool photos, runs/battery/orig.npy).
Outputs and caches go to runs/regions/.

usage: python -I tools/regions.py --unit N [--selftest]
"""
import base64
import importlib
import io
import json
import re
import sys
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

J = Path(__file__).resolve().parent.parent
OUT = J / 'runs/regions'
NCH = 832
K = 20
N_RAND, N_TOP = 60, 20
GRID, SIDE = 7, 448
CELL = SIDE // GRID
PER_CALL, PARALLEL = 5, 4
ALPHA = 0.05
STATS = ('PR', 'PO', 'RO')
GROUPS = ('all', 'random', 'top')
PROMPT = ('Each image has a 7x7 grid. Cells are named by row letter (A-G, top to bottom) and column number (1-7, left to '
          'right). For each image, list the cells that contain a pointed shape: a tip, a point, a corner, or a V- or '
          'wedge-shaped part of something. Then list the cells that contain a round shape: a circle, disc, ring, dome '
          'or a round part of something. A cell can be in both lists or in neither. Reply with only JSON:\n'
          '{"images": [{"id": n, "pointed": ["A1", ...], "round": ["B2", ...]}]}')


def _tool(name):
    p = str(J / 'tools')
    if p not in sys.path:
        sys.path.insert(0, p)
    return importlib.import_module(name)


def held(names):
    return np.array([i for i, n in enumerate(names) if zlib.crc32(n.encode()) % 10 >= 7], dtype=np.int64)


def select(names, maps):
    """The 80 photo indices in sheet order (id k + 1 at position k) and the subset ('random' / 'top') of each."""
    H = held(names)
    rand = np.random.default_rng(0).choice(H, N_RAND, replace=False)
    rest = np.setdiff1d(H, rand)
    peak = maps[rest].astype(np.float32).max(axis=(1, 2))
    top = rest[np.argsort(-peak, kind='stable')[:N_TOP]]
    ids = np.concatenate([rand, top])
    sub = np.array(['random'] * N_RAND + ['top'] * N_TOP)
    perm = np.random.default_rng(1).permutation(len(ids))
    return ids[perm], sub[perm]


def cell_name(r, c):
    return 'ABCDEFG'[r] + str(c + 1)


def parse_cell(s):
    """'B3' -> (1, 2): row A-G top to bottom, column 1-7 left to right. None if s is not a cell name."""
    m = re.fullmatch(r'([A-G])([1-7])', s.strip()) if isinstance(s, str) else None
    return None if m is None else (ord(m[1]) - ord('A'), int(m[2]) - 1)


def pool2(m):
    """(..., 14, 14) map -> (..., 7, 7) cell values, each the mean of the 2x2 block inside the cell."""
    m = np.asarray(m, dtype=np.float64)
    assert m.shape[-2:] == (14, 14), m.shape
    return m.reshape(m.shape[:-2] + (GRID, 2, GRID, 2)).mean(axis=(-3, -1))


def masks(lab):
    """7x7 boolean P (pointed only), R (round only), O (neither) for one label entry; a cell in both is in none."""
    p, r = np.zeros((GRID, GRID), bool), np.zeros((GRID, GRID), bool)
    for key, m in (('pointed', p), ('round', r)):
        for s in lab[key]:
            rc = parse_cell(s)
            assert rc is not None, s
            m[rc] = True
    return p & ~r, r & ~p, ~p & ~r


def diffs(cells, P, R, O):
    """d_PR, d_PO, d_RO over the trailing 7x7 axes of cells, None where undefined.
    d_PR needs a P and an R cell; d_PO and d_RO are taken on those photos when they also have an O cell."""
    d = dict.fromkeys(STATS)
    if P.any() and R.any():
        mp, mr = cells[..., P].mean(axis=-1), cells[..., R].mean(axis=-1)
        d['PR'] = mp - mr
        if O.any():
            mo = cells[..., O].mean(axis=-1)
            d['PO'], d['RO'] = mp - mo, mr - mo
    return d


def sheet(img):
    """448x448 upscale with a 7x7 grid (1 px white over 1 px black) and each cell's name at its top-left."""
    im = img.convert('RGB').resize((SIDE, SIDE), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    for k in range(1, GRID):
        p = k * CELL
        dr.line([(p - 1, 0), (p - 1, SIDE - 1)], fill='black')
        dr.line([(p, 0), (p, SIDE - 1)], fill='white')
        dr.line([(0, p - 1), (SIDE - 1, p - 1)], fill='black')
        dr.line([(0, p), (SIDE - 1, p)], fill='white')
    for r in range(GRID):
        for c in range(GRID):
            xy = (c * CELL + 3, r * CELL + 3)
            x0, y0, x1, y1 = dr.textbbox(xy, cell_name(r, c))
            dr.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), fill='black')
            dr.text(xy, cell_name(r, c), fill='white')
    return im


def _jpeg(im):
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=90)
    return buf.getvalue()


def content(batch):
    """Anthropic blocks for one call: 'Image {id}:' and the sheet for each (id, base64 jpeg), then PROMPT."""
    out = []
    for i, data in batch:
        out.append({'type': 'text', 'text': f'Image {i}:'})
        out.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': data}})
    out.append({'type': 'text', 'text': PROMPT})
    return out


def parse_reply(text):
    """The reply's {"images": [...]} object. claude_call.parse_json returns the last JSON object in the text, which
    for this nested reply is the last inner image entry; when it lacks 'images', take the last object that has it."""
    obj = _tool('claude_call').parse_json(text)
    if 'images' in obj:
        return obj
    dec, best = json.JSONDecoder(), None
    for i, ch in enumerate(text):
        if ch == '{':
            try:
                o, _ = dec.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(o, dict) and 'images' in o:
                best = o
    if best is None:
        raise ValueError('no "images" object in reply')
    return best


def _id(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().isdigit():
        return int(v)
    return None


def _entry(e):
    """{'pointed': [...], 'round': [...]} with sorted canonical names, or None if a list is missing or a name invalid."""
    out = {}
    for key in ('pointed', 'round'):
        v = e.get(key)
        if not isinstance(v, list):
            return None
        rc = [parse_cell(s) for s in v]
        if any(x is None for x in rc):
            return None
        out[key] = sorted({cell_name(*x) for x in rc})
    return out


def check_reply(text, ids):
    """({id: entry} for the ids with a present, valid entry, [ids missing or with an invalid entry])."""
    try:
        imgs = parse_reply(text).get('images')
    except ValueError:
        return {}, list(ids)
    if not isinstance(imgs, list):
        return {}, list(ids)
    got = {}
    for e in imgs:
        k = _id(e.get('id')) if isinstance(e, dict) else None
        if k in ids and k not in got:
            got[k] = e
    good, bad = {}, []
    for k in ids:
        lab = _entry(got[k]) if k in got else None
        if lab is None:
            bad.append(k)
        else:
            good[k] = lab
    return good, bad


def label(ids, subs):
    """Pointed and round cells of every photo from the vision model; writes labels.json, replies.json, sheets/."""
    battery, claude_call = _tool('battery'), _tool('claude_call')
    (OUT / 'sheets').mkdir(parents=True, exist_ok=True)
    data = {}
    for k, i in enumerate(ids):
        b = _jpeg(sheet(battery.photo(i)))
        (OUT / 'sheets' / f'{k + 1:02d}.jpg').write_bytes(b)
        data[k + 1] = base64.b64encode(b).decode('ascii')
    batches = [list(range(s, min(s + PER_CALL, len(ids) + 1))) for s in range(1, len(ids) + 1, PER_CALL)]

    def one(b):
        tries = []
        for _ in range(2):  # retry once if an id is missing or a cell name is invalid; the last reply is kept
            r = claude_call.converse('', [content([(k, data[k]) for k in b])])[0]
            tries.append(r)
            good, bad = check_reply(r.get('text') or '', b)
            if not bad:
                break
        print(f'batch {b[0]}-{b[-1]}: {len(tries)} call(s), unlabelled {bad}', flush=True)
        return tries, good

    print('labels: running', len(batches), 'batches', flush=True)
    with ThreadPoolExecutor(PARALLEL) as ex:
        res = list(ex.map(one, batches))
    (OUT / 'replies.json').write_text(json.dumps([dict(ids=b, attempts=t) for b, (t, _g) in zip(batches, res)],
                                                 indent=1))
    good = {k: v for _t, g in res for k, v in g.items()}
    photos = [dict(id=k + 1, index=int(i), subset=str(s), **good.get(k + 1, dict(pointed=None, round=None)))
              for k, (i, s) in enumerate(zip(ids, subs))]
    (OUT / 'labels.json').write_text(json.dumps(dict(photos=photos), indent=1))
    return photos


def labels(ids, subs):
    f = OUT / 'labels.json'
    if not f.exists():
        return label(ids, subs)
    photos = json.loads(f.read_text())['photos']
    want = [(k + 1, int(i), str(s)) for k, (i, s) in enumerate(zip(ids, subs))]
    assert [(p['id'], p['index'], p['subset']) for p in photos] == want, 'cached labels.json differs from the selection'
    print('labels: cached', flush=True)
    return photos


def post_maps(model, ids):
    """Post-ReLU mixed4e maps of every channel for photo ids, (len(ids), 832, 14, 14) float32; input as battery.acts."""
    import torch
    battery = _tool('battery')
    Stop = battery._bank().Stop
    got = {}

    def keep(_m, _i, out):
        got['a'] = out.detach().clone()

    def stop(_m, _i, _o):
        raise Stop

    hs = [model.mixed4e.register_forward_hook(keep), model.maxpool10.register_forward_hook(stop)]
    res = []
    try:
        for s in range(0, len(ids), battery.BATCH):
            imgs = [battery.photo(i) for i in ids[s:s + battery.BATCH]]
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
            assert float(a.min()) >= 0, 'mixed4e output is not post-ReLU'
            res.append(a.numpy().astype(np.float32))
    finally:
        for h in hs:
            h.remove()
    return np.concatenate(res)


def strong_levels(names):
    """Per channel median spatial max over its top-20 battery pool photos, from runs/battery/{orig,pool}.npy."""
    battery = _tool('battery')
    f, g = battery.OUT / 'orig.npy', battery.OUT / 'pool.npy'
    if not (f.exists() and g.exists()):
        raise SystemExit(f'missing {f} or {g}: run tools/battery.py first')
    P = battery.pool(names)
    assert np.array_equal(np.load(g), P), 'cached pool.npy differs from the battery pool'
    orig = np.load(f)
    assert orig.shape == (len(P), NCH), orig.shape
    order = np.argsort(-orig, axis=0, kind='stable')[:K]
    return np.median(np.take_along_axis(orig, order, axis=0), axis=0).astype(np.float64)


def sign_stats(v):
    """n, median, fraction > 0 (of all n), and the two-sided sign test p with zeros dropped."""
    from scipy.stats import binomtest
    v = np.asarray(v, dtype=np.float64)
    n, pos, neg = len(v), int((v > 0).sum()), int((v < 0).sum())
    return dict(n=n, pos=pos, neg=neg, median=float(np.median(v)) if n else float('nan'),
                frac=pos / n if n else float('nan'),
                p=float(binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else float('nan'))


def decide(st):
    """Neutral label on one sign test: p < 0.05 -> POSITIVE if more values > 0 than < 0, else NEGATIVE;
    p >= 0.05 -> UNRESOLVED; no non-zero values -> UNDEFINED."""
    if st['pos'] + st['neg'] == 0:
        return 'UNDEFINED'
    if st['p'] < ALPHA:
        return 'POSITIVE' if st['pos'] > st['neg'] else 'NEGATIVE'
    return 'UNRESOLVED'


def pct(v, x):
    v = v[np.isfinite(v)]
    return 100 * float(np.mean(v <= x)) if np.isfinite(x) and len(v) else float('nan')


def analyse(unit, photos, pre, post, strong):
    """Report text. pre: (80, 14, 14) pre-ReLU maps of the unit from acts.npz; post: (80, 832, 14, 14) post-ReLU
    mixed4e maps; strong: (832,) strong levels; photos: labels.json entries in sheet order."""
    nan = float('nan')
    d = {(s, g): [] for s in STATS for g in GROUPS}
    cv = {(m, g): [] for m in 'PRO' for g in GROUPS}
    ok = strong > 0
    scale = np.where(ok, strong, 1.0)
    lay, rows, unl, nboth = [], [], [], 0
    for k, lab in enumerate(photos):
        head = f'{lab["id"]:3d} {lab["index"]:6d} {lab["subset"]:6s}'
        if lab['pointed'] is None:
            unl.append(lab['id'])
            rows.append(head + ' unlabelled')
            continue
        P, R, O = masks(lab)
        both = GRID * GRID - int(P.sum() + R.sum() + O.sum())
        nboth += both
        cells = pool2(pre[k])
        groups = ('all', lab['subset'])
        for m, M in zip('PRO', (P, R, O)):
            for g in groups:
                cv[m, g].extend(cells[M].tolist())
        x = diffs(cells, P, R, O)
        for s in STATS:
            if x[s] is not None:
                for g in groups:
                    d[s, g].append(float(x[s]))
        y = diffs(pool2(post[k]), P, R, O)['PR']
        if y is not None:
            lay.append(y / scale)
        vals = ' '.join(f'{nan if x[s] is None else float(x[s]):8.3f}' for s in STATS)
        rows.append(f'{head} {int(P.sum()):3d} {int(R.sum()):3d} {int(O.sum()):3d} {both:4d} {vals}')

    n = len(photos)
    nr = sum(p['subset'] == 'random' for p in photos)
    out = [f'regions: unit {unit} of mixed4e; {n} photos from the held pool ({nr} random, {n - nr} top by spatial max '
           f'of the acts.npz map); cell value = mean of the 2x2 block of the 14x14 map inside each 7x7 grid cell',
           f'labelled {n - len(unl)}/{n}; unlabelled ids: {unl}',
           f'cells: P (pointed only) {len(cv["P", "all"])}, R (round only) {len(cv["R", "all"])}, '
           f'O (neither) {len(cv["O", "all"])}, in both lists (dropped) {nboth}',
           '',
           'pre-ReLU map of the unit (bank/img/acts.npz)',
           f'decision per row: two-sided sign test (zeros dropped) p < {ALPHA} -> POSITIVE if more values > 0 than < 0, '
           f'else NEGATIVE; p >= {ALPHA} -> UNRESOLVED',
           f'{"stat":5s} {"subset":6s} {"n":>3s} {"median":>8s} {"frac>0":>6s} {"p":>9s}  decision']
    for s in STATS:
        for g in GROUPS:
            st = sign_stats(d[s, g])
            out.append(f'd_{s:3s} {g:6s} {st["n"]:3d} {st["median"]:8.3f} {st["frac"]:6.2f} {st["p"]:9.3g}  '
                       f'{decide(st)}')

    out += ['', 'fraction of cells with pre-ReLU value < 0']
    for g in GROUPS:
        out.append(f'{g:6s} ' + '  '.join(
            f'{m}={float(np.mean(np.array(cv[m, g]) < 0)) if cv[m, g] else nan:.3f} (n={len(cv[m, g])})' for m in 'PRO'))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        r = float(np.corrcoef(np.maximum(pre, 0).ravel(), post[:, unit].ravel())[0, 1])
    out += ['', f'check: corr(relu(acts.npz map), post-ReLU mixed4e[{unit}] on the thumbnails) over {n} photos = '
                f'{r:.3f}']

    out += ['', f'layer: d_PR on each channel\'s post-ReLU map / its strong level (median spatial max over its top-{K} '
                f'battery pool photos), {len(lay)} photos with a P and an R cell; '
                f'{int(ok.sum())}/{NCH} channels with strong level > 0; strong level of {unit} = {strong[unit]:.3f}']
    if lay:
        A = np.array(lay)
        pos, nz = (A > 0).sum(axis=0), (A != 0).sum(axis=0)
        med = np.where(ok, np.median(A, axis=0), np.nan)
        frac = np.where(ok & (nz > 0), pos / np.maximum(nz, 1), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for name, v in (('median d_PR / strong', med), ('frac d_PR > 0 (zeros dropped)', frac)):
                out.append(f'{name:30s} unit {unit}={v[unit]:.3f} pct={pct(v, v[unit]):.0f} '
                           f'layer_median={float(np.nanmedian(v)):.3f} (n={int(np.isfinite(v).sum())})')

    out += ['', 'per photo, pre-ReLU: id index subset nP nR nO nBoth d_PR d_PO d_RO'] + rows
    return '\n'.join(out) + '\n'


def run(unit):
    OUT.mkdir(parents=True, exist_ok=True)
    names = json.loads((J / 'bank/img/names.json').read_text())
    with np.load(J / 'bank/img/acts.npz') as z:
        maps = z['maps']
    assert maps.shape == (len(names), 14, 14), maps.shape
    ids, subs = select(names, maps)
    strong = strong_levels(names)
    photos = labels(ids, subs)
    import torch
    torch.set_num_threads(4)
    model = _tool('battery')._bank().load_model(J / 'raw/weights/inception5h.pth')
    post = post_maps(model, ids)
    text = analyse(unit, photos, maps[ids].astype(np.float64), post, strong)
    (OUT / 'report.txt').write_text(text)
    print(text, end='')


def selftest():
    # cell names
    assert parse_cell('A1') == (0, 0) and parse_cell('G7') == (6, 6) and parse_cell('B3') == (1, 2)
    assert parse_cell(' C4 ') == (2, 3)
    for s in ['H1', 'A0', 'A8', '1A', 'AA', 'A', '', 'A10', 'a1', 'B 3', None, 5, ['A1']]:
        assert parse_cell(s) is None, s
    assert all(parse_cell(cell_name(r, c)) == (r, c) for r in range(GRID) for c in range(GRID))

    # 2x2 pooling
    m = np.arange(196, dtype=np.float64).reshape(14, 14)
    c = pool2(m)
    assert c.shape == (7, 7)
    assert c[0, 0] == (0 + 1 + 14 + 15) / 4 and c[6, 6] == (180 + 181 + 194 + 195) / 4
    for r in range(GRID):
        for k in range(GRID):
            assert c[r, k] == m[2 * r:2 * r + 2, 2 * k:2 * k + 2].mean(), (r, k)
    st3 = np.stack([m, -m, 2 * m])
    assert np.array_equal(pool2(st3)[1], -c) and np.array_equal(pool2(st3[None])[0, 2], 2 * c)

    # known answers: A1, A2 pointed only (3, 5 -> 4), B1 round only (-2), C3 in both (100, dropped), rest O (1)
    v = np.ones((GRID, GRID))
    v[0, 0], v[0, 1], v[1, 0], v[2, 2] = 3, 5, -2, 100
    syn = np.kron(v, np.ones((2, 2)))
    assert syn.shape == (14, 14) and np.array_equal(pool2(syn), v)
    lab = dict(pointed=['A1', 'A2', 'C3'], round=['B1', 'C3'])
    P, R, O = masks(lab)
    assert P.sum() == 2 and R.sum() == 1 and O.sum() == 45 and not (P | R | O)[2, 2], 'cell in both lists not dropped'
    d = diffs(pool2(syn), P, R, O)
    # with C3 kept in P d_PR would be 38; with C3 counted as O d_PO would be about 0.85
    assert np.isclose(d['PR'], 6) and np.isclose(d['PO'], 3) and np.isclose(d['RO'], -3), d
    d2 = diffs(pool2(syn), *masks(dict(pointed=['A1', 'A2'], round=['B1'])))
    assert np.isclose(d2['PR'], d['PR']), 'cell in both lists changed P or R'
    sw = diffs(pool2(syn), *masks(dict(pointed=lab['round'], round=lab['pointed'])))
    assert np.isclose(sw['PR'], -d['PR']) and np.isclose(sw['PO'], d['RO']) and np.isclose(sw['RO'], d['PO']), sw
    assert diffs(v, *masks(dict(pointed=['A1'], round=[]))) == dict(PR=None, PO=None, RO=None)
    full = dict(pointed=[cell_name(r, k) for r in range(GRID) for k in range(4)],
                round=[cell_name(r, k) for r in range(GRID) for k in range(4, GRID)])
    e = diffs(v, *masks(full))
    assert e['PR'] is not None and e['PO'] is None and e['RO'] is None, e

    # many channels at once == one channel at a time
    rng = np.random.default_rng(0)
    many = rng.standard_normal((5, 14, 14))
    vec = diffs(pool2(many), P, R, O)
    for j in range(5):
        one = diffs(pool2(many[j]), P, R, O)
        assert all(np.isclose(vec[s][j], one[s]) for s in STATS), j

    # reply parsing
    ids = [6, 7, 8]
    body = {'images': [{'id': 6, 'pointed': ['A1'], 'round': []}, {'id': '7', 'pointed': [], 'round': ['G7', 'G7']},
                       {'id': 8, 'pointed': ['B2'], 'round': ['B2']}, {'id': 99, 'pointed': [], 'round': []}]}
    good, bad = check_reply('Here:\n```json\n' + json.dumps(body) + '\n```\n', ids)
    assert bad == [] and good == {6: dict(pointed=['A1'], round=[]), 7: dict(pointed=[], round=['G7']),
                                  8: dict(pointed=['B2'], round=['B2'])}, (good, bad)
    assert check_reply(json.dumps({'images': body['images'][:2]}), ids)[1] == [8], 'missing id'
    for broken in (['A8'], ['a1'], 'A1', None):
        b2 = json.loads(json.dumps(body))
        b2['images'][0]['pointed'] = broken
        assert check_reply(json.dumps(b2), ids)[1] == [6], ('invalid entry', broken)
    b3 = json.loads(json.dumps(body))
    del b3['images'][1]['round']
    assert check_reply(json.dumps(b3), ids)[1] == [7], 'missing list'
    assert check_reply('no json here', ids) == ({}, ids) and check_reply('{"images": "x"}', ids) == ({}, ids)

    # sign test and decision
    from scipy.stats import binomtest
    st = sign_stats([1.0, 2.0, 3.0, -1.0, 0.0])
    assert (st['n'], st['pos'], st['neg'], st['frac'], st['median']) == (5, 3, 1, 0.6, 1.0), st
    assert np.isclose(st['p'], binomtest(3, 4, 0.5).pvalue)
    a, b = sign_stats(np.arange(1, 13)), sign_stats(-np.arange(1, 13))
    assert np.isclose(a['p'], b['p']) and decide(a) == 'POSITIVE' and decide(b) == 'NEGATIVE'
    assert decide(sign_stats([1, -1, 2, -2])) == 'UNRESOLVED' and decide(sign_stats([0, 0])) == 'UNDEFINED'
    assert np.isnan(sign_stats([])['median'])

    # selection
    names = [f'n{i:04d}.jpeg' for i in range(600)]
    maps = rng.standard_normal((600, 14, 14)).astype(np.float16)
    sel, sub = select(names, maps)
    H = held(names)
    assert len(H) >= N_RAND + N_TOP and len(sel) == 80 and len(set(sel.tolist())) == 80
    assert set(sel.tolist()) <= set(H.tolist()) and (sub == 'random').sum() == N_RAND
    base = np.empty_like(sel)
    base[np.random.default_rng(1).permutation(80)] = sel
    assert np.array_equal(base[:N_RAND], np.random.default_rng(0).choice(H, N_RAND, replace=False)), 'random draw'
    peak = maps.astype(np.float32).max(axis=(1, 2))
    assert peak[sel[sub == 'top']].min() >= peak[np.setdiff1d(H, sel)].max(), 'top 20'

    # sheet and content
    img = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
    sh = np.asarray(sheet(img))
    assert sh.shape == (SIDE, SIDE, 3), sh.shape
    for k in range(1, GRID):
        p = k * CELL
        assert (sh[40, p] == 255).all() and (sh[40, p - 1] == 0).all(), ('vertical line', k)
        assert (sh[p, 40] == 255).all() and (sh[p - 1, 40] == 0).all(), ('horizontal line', k)
    blocks = content([(1, 'AAAA'), (2, 'BBBB')])
    assert [x['type'] for x in blocks] == ['text', 'image', 'text', 'image', 'text']
    assert blocks[0]['text'] == 'Image 1:' and blocks[2]['text'] == 'Image 2:' and blocks[-1]['text'] == PROMPT
    assert blocks[3]['source'] == {'type': 'base64', 'media_type': 'image/jpeg', 'data': 'BBBB'}
    print('selftest ok')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    if '--unit' not in sys.argv:
        raise SystemExit(__doc__)
    unit = int(sys.argv[sys.argv.index('--unit') + 1])
    assert 0 <= unit < NCH, unit
    run(unit)


if __name__ == '__main__':
    main()
