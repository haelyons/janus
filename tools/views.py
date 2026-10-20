"""View labels for vehicle and dog photos; view tuning of every channel of InceptionV1 mixed4c and mixed4d.

Photos: every photo of 14 vehicle classes, plus 300 photos drawn without replacement (rng 0) from dog classes 151-268,
shuffled together (rng 1) and numbered 1..n. Claude labels each photo's facing and framing, 10 photos per call.
a_c(x) is the spatial max of channel c of the post-ReLU module output, divided by the max of a_c over the selected
photos (NaN throughout for channels whose max is 0). Per channel and per kind (V vehicles, D dogs): mean a_c per folded
facing bin (front, three_quarter_front, side, rear; NaN with < 5 photos), eta2 = between-bin SS / total SS, Pearson
corr of the V and D bin means (NaN with < 3 shared bins), the bin with the highest mean, the same eta2 and corr for
framing (head, part, whole), left minus right mean, and the share of dogs among the 50 top photos.
Neutral labels: eta2 >= 0.14 LARGE, >= 0.06 MEDIUM, >= 0.01 SMALL, else NEGLIGIBLE (Cohen's conventions);
corr >= 0.5 POSITIVE, <= -0.5 NEGATIVE, else WEAK.
Input range follows img_acts.run_batch: pixels (0-255) - 117. Outputs and caches go to runs/views/.

usage: python -I tools/views.py [--selftest] [--report 4d:N,...] [--sheet 4d:N]
"""
import base64
import csv
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

J = Path(__file__).resolve().parent.parent
OUT = J / 'runs/views'
VEHICLES = (407, 436, 468, 479, 511, 581, 609, 627, 656, 661, 717, 734, 751, 817)
DOG_LO, DOG_HI = 151, 268
NDOG = 300
BATCH = 64
PER_CALL = 10
PARALLEL = 4
MIN = 5
TOP = 50
LAYERS = (('4c', 512), ('4d', 528))
OFF = {'4c': 0, '4d': 512}
NCH = 1040
FACING = ('toward', 'toward_left', 'left', 'away_left', 'away', 'away_right', 'right', 'toward_right', 'unclear')
FRAMING = ('head', 'part', 'whole', 'unclear')
MISSING = 'missing'
FOLD = {
    'front': ('toward',),
    'three_quarter_front': ('toward_left', 'toward_right'),
    'side': ('left', 'right'),
    'rear': ('away', 'away_left', 'away_right'),
}
FRAME_BINS = ('head', 'part', 'whole')
LEFT = ('toward_left', 'left', 'away_left')
RIGHT = ('toward_right', 'right', 'away_right')
KEYS = ('eta2_V', 'eta2_D', 'corr', 'pref_V', 'pref_D', 'eta2f_V', 'eta2f_D', 'corrf', 'lr_V', 'lr_D', 'dogshare')
COLS = ['layer', 'channel'] + [f't_{k}_{b}' for k in 'VD' for b in FOLD] + list(KEYS)
MEDIANS = ('eta2_V', 'eta2_D', 'corr', 'eta2f_V', 'eta2f_D', 'corrf')
PROMPT = ' '.join([
    'For each image, consider the main animal or vehicle. Give "facing": which way its front (face, or the front of',
    'the vehicle) points, as seen in the image: "toward" (facing the camera), "toward_left" (front visible, turned',
    'toward the image\'s left), "left" (side view, front pointing to the image\'s left), "away_left", "away" (seen from',
    'behind), "away_right", "right", "toward_right", or "unclear". Give "framing": "head" (only the head or the front',
    'fills most of the frame), "part" (head and part of the body), "whole" (the whole animal or vehicle is visible),',
    'or "unclear". Reply with only JSON: {"images": [{"id": n, "facing": "...", "framing": "..."}]}',
])


def decide_eta2(e):
    """Neutral label on one eta2 value (Cohen): >= 0.14 LARGE, >= 0.06 MEDIUM, >= 0.01 SMALL, else NEGLIGIBLE."""
    if np.isnan(e):
        return 'UNDEFINED'
    return 'LARGE' if e >= 0.14 else 'MEDIUM' if e >= 0.06 else 'SMALL' if e >= 0.01 else 'NEGLIGIBLE'


def decide_corr(r):
    """Neutral label on one correlation: r >= 0.5 POSITIVE, r <= -0.5 NEGATIVE, else WEAK."""
    if np.isnan(r):
        return 'UNDEFINED'
    return 'POSITIVE' if r >= 0.5 else 'NEGATIVE' if r <= -0.5 else 'WEAK'


def _bank():
    p = str(J / 'bank')
    if p not in sys.path:
        sys.path.insert(0, p)
    import img_acts
    return img_acts


def _tools():
    p = str(J / 'tools')
    if p not in sys.path:
        sys.path.insert(0, p)
    import claude_call
    return claude_call


def select(classes):
    """selected photo indices in id order (id = position + 1) and their kind, 'V' or 'D'"""
    classes = np.asarray(classes)
    veh = np.flatnonzero(np.isin(classes, VEHICLES))
    dogpool = np.flatnonzero((classes >= DOG_LO) & (classes <= DOG_HI))
    dogs = np.random.default_rng(0).choice(dogpool, NDOG, replace=False)
    sel = np.random.default_rng(1).permutation(np.concatenate([veh, dogs]))
    return sel, np.where(np.isin(classes[sel], VEHICLES), 'V', 'D')


def photo(i):
    with Image.open(J / 'bank/img/thumbs' / f'{int(i):05d}.jpg') as im:
        img = im.convert('RGB')
    assert img.size == (224, 224), (i, img.size)
    return img


def b64(i):
    return base64.b64encode((J / 'bank/img/thumbs' / f'{int(i):05d}.jpg').read_bytes()).decode('ascii')


def content(ids, datas):
    """one user turn: 'Image {id}:' and the image for each photo, then PROMPT"""
    blocks = []
    for i, data in zip(ids, datas):
        blocks.append({'type': 'text', 'text': f'Image {i}:'})
        blocks.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': data}})
    blocks.append({'type': 'text', 'text': PROMPT})
    return blocks


def parse_reply(text):
    """claude_call.parse_json; if that object has no 'images' (parse_json falls back to the last object found, which
    for this reply shape is the last inner entry), the last object in the reply that carries 'images'"""
    obj = _tools().parse_json(text)
    if 'images' in obj:
        return obj
    dec, found = json.JSONDecoder(), []
    for i, ch in enumerate(text):
        if ch == '{':
            try:
                o, _ = dec.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(o, dict) and 'images' in o:
                found.append(o)
    return found[-1] if found else obj


def _id(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().isdigit():
        return int(v)
    return None


def check(text, ids):
    """{id: (facing, framing)} from one reply, and whether every id is there with both values in the lists"""
    try:
        obj = parse_reply(text)
    except ValueError:
        return {}, False
    ents = obj.get('images')
    got = {}
    for e in ents if isinstance(ents, list) else []:
        i = _id(e.get('id')) if isinstance(e, dict) else None
        if i in ids and i not in got:
            got[i] = (e.get('facing'), e.get('framing'))
    ok = all(i in got and got[i][0] in FACING and got[i][1] in FRAMING for i in ids)
    return got, ok


def label_batch(ids, datas, call):
    """one batch, a fresh single-turn conversation; retried once if an id is missing or a value is not in the lists.
    Whatever is still missing or invalid after the retry is recorded as MISSING."""
    attempts = []
    for _ in range(2):
        rep = call('', [content(ids, datas)])[0]
        got, ok = check(rep.get('text') or '', ids)
        attempts.append(dict(rep, ok=ok))
        if ok:
            break
    labs = {}
    for i in ids:
        f, g = got.get(i, (None, None))
        labs[i] = (f if f in FACING else MISSING, g if g in FRAMING else MISSING)
    return labs, attempts


def label(sel, kind, classes, names):
    f = OUT / 'labels.json'
    photos = [int(i) for i in sel]
    if f.exists():
        d = json.loads(f.read_text())
        assert d['photos'] == photos, 'cached labels.json photos differ from the selection'
        print('labels: cached', len(d['labels']), flush=True)
        return d['labels']
    cc = _tools()
    ids = list(range(1, len(sel) + 1))
    batches = [ids[s:s + PER_CALL] for s in range(0, len(ids), PER_CALL)]
    print(f'labels: {len(batches)} calls of up to {PER_CALL} photos, {PARALLEL} in parallel', flush=True)
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(label_batch, b, [b64(sel[i - 1]) for i in b], cc.converse) for b in batches]
        done = []
        for k, fu in enumerate(futs):
            done.append(fu.result())
            print(f'[{k + 1}/{len(batches)}] attempts={len(done[-1][1])}', flush=True)
    replies, labels = [], []
    for k, (b, (labs, att)) in enumerate(zip(batches, done)):
        replies.append({'batch': k + 1, 'ids': b, 'attempts': att})
        for i in b:
            p = int(sel[i - 1])
            labels.append({'id': i, 'photo': p, 'name': names[p], 'class': int(classes[p]), 'kind': str(kind[i - 1]),
                           'facing': labs[i][0], 'framing': labs[i][1]})
    (OUT / 'replies.json').write_text(json.dumps(replies, indent=1))
    f.write_text(json.dumps({'photos': photos, 'labels': labels}, indent=1))
    cost = sum(a.get('cost_usd') or 0 for r in replies for a in r['attempts'])
    print(f'labels: done, cost_usd={cost:.2f}', flush=True)
    return labels


def acts(model, ids):
    """spatial max of every mixed4c and mixed4d channel (post-ReLU module outputs), (len(ids), 1040) float32"""
    import torch
    Stop = _bank().Stop
    got = {}

    def keep(name):
        def hook(_m, _i, out):
            got[name] = out.detach().clone()
        return hook

    def stop(_m, _i, _o):
        raise Stop

    hs = [model.mixed4c.register_forward_hook(keep('4c')), model.mixed4d.register_forward_hook(keep('4d')),
          model.mixed4e.register_forward_hook(stop)]
    res = []
    try:
        for s in range(0, len(ids), BATCH):
            imgs = [photo(i) for i in ids[s:s + BATCH]]
            x = torch.from_numpy(np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in imgs])).permute(0, 3, 1, 2)
            x = x * 255 - 117
            got.clear()
            try:
                with torch.no_grad():
                    model(x)
            except Stop:
                pass
            parts = []
            for name, n in LAYERS:
                a = got[name]
                assert tuple(a.shape[1:]) == (n, 14, 14), (name, tuple(a.shape))
                parts.append(a.amax(dim=(2, 3)).numpy().astype(np.float32))
            res.append(np.concatenate(parts, axis=1))
    finally:
        for h in hs:
            h.remove()
    return np.concatenate(res)


def normalise(a):
    """a divided by its per-channel max over the photos; NaN throughout for channels whose max is 0"""
    a = np.asarray(a, dtype=np.float64)
    mx = a.max(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(mx > 0, a / np.where(mx > 0, mx, 1), np.nan)


def tuning(A, groups):
    """t (C, len(groups)): mean of A over each group with >= MIN photos, else NaN. eta2 (C,): between-group SS over
    total SS across the photos of the defined groups; NaN with < 2 defined groups or no spread in those photos."""
    C = A.shape[1]
    t = np.full((C, len(groups)), np.nan)
    ok = [g for g in groups if g.sum() >= MIN]
    eta2 = np.full(C, np.nan)
    with warnings.catch_warnings(), np.errstate(divide='ignore', invalid='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        for b, g in enumerate(groups):
            if g.sum() >= MIN:
                t[:, b] = A[g].mean(axis=0)
        if len(ok) >= 2:
            X = A[np.logical_or.reduce(ok)]
            m = X.mean(axis=0)
            sst = ((X - m) ** 2).sum(axis=0)
            ssb = sum(g.sum() * (A[g].mean(axis=0) - m) ** 2 for g in ok)
            spread = X.max(axis=0) > X.min(axis=0)
            eta2 = np.where(spread, ssb / np.where(spread, sst, 1), np.nan)
    return t, eta2


def pearson(x, y):
    """row-wise Pearson r over the columns defined in both; NaN with < 3 such columns or a constant row"""
    r = np.full(len(x), np.nan)
    for c in range(len(x)):
        ok = np.isfinite(x[c]) & np.isfinite(y[c])
        if ok.sum() < 3:
            continue
        a, b = x[c, ok], y[c, ok]
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            continue
        a, b = a - a.mean(), b - b.mean()
        r[c] = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    return r


def prefs(t, names):
    """name of the defined bin with the highest mean; 'tie' if shared by several bins, 'none' if no bin is defined"""
    out = []
    for row in t:
        if not np.isfinite(row).any():
            out.append('none')
            continue
        best = np.nanmax(row)
        out.append(names[int(np.nanargmax(row))] if int((row == best).sum()) == 1 else 'tie')
    return out


def lr(A, m, facing):
    """mean of A over m & LEFT minus over m & RIGHT; NaN if either side has < MIN photos"""
    L, R = m & np.isin(facing, LEFT), m & np.isin(facing, RIGHT)
    if L.sum() < MIN or R.sum() < MIN:
        return np.full(A.shape[1], np.nan)
    return A[L].mean(axis=0) - A[R].mean(axis=0)


def dogshare(A, isdog):
    """share of dogs among each channel's TOP highest photos (NaN for NaN channels)"""
    k = min(TOP, len(A))
    order = np.argsort(-A, axis=0, kind='stable')[:k]
    s = isdog[order].mean(axis=0)
    s[np.isnan(A).any(axis=0)] = np.nan
    return s


def analyse(A, kind, facing, framing):
    res = {}
    for k in 'VD':
        m = kind == k
        res[f't_{k}'], res[f'eta2_{k}'] = tuning(A, [m & np.isin(facing, fs) for fs in FOLD.values()])
        res[f'pref_{k}'] = prefs(res[f't_{k}'], list(FOLD))
        res[f'tf_{k}'], res[f'eta2f_{k}'] = tuning(A, [m & (framing == b) for b in FRAME_BINS])
        res[f'lr_{k}'] = lr(A, m, facing)
    res['corr'] = pearson(res['t_V'], res['t_D'])
    res['corrf'] = pearson(res['tf_V'], res['tf_D'])
    res['dogshare'] = dogshare(A, kind == 'D')
    return res


def col(layer, ch):
    n = dict(LAYERS)[layer]
    assert 0 <= ch < n, (layer, ch)
    return OFF[layer] + ch


def _spec(s):
    """'4d:12,4c:7' -> [('4d', 12), ('4c', 7)]"""
    out = []
    for x in s.split(','):
        layer, ch = x.strip().split(':')
        col(layer, int(ch))
        out.append((layer, int(ch)))
    return out


def table(res):
    rows = []
    for layer, n in LAYERS:
        for ch in range(n):
            c = OFF[layer] + ch
            r = {'layer': layer, 'channel': ch}
            for k in 'VD':
                for b, name in enumerate(FOLD):
                    r[f't_{k}_{name}'] = float(res[f't_{k}'][c, b])
            for key in KEYS:
                v = res[key][c]
                r[key] = v if isinstance(v, str) else float(v)
            rows.append(r)
    return rows


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    names = json.loads((J / 'bank/img/names.json').read_text())
    with np.load(J / 'bank/img/acts.npz') as z:
        classes = z['classes']
    assert len(names) == len(classes), (len(names), len(classes))
    sel, kind = select(classes)
    print(f'photos: {len(sel)} ({int((kind == "V").sum())} vehicles, {int((kind == "D").sum())} dogs)', flush=True)

    # 1. labels
    labels = label(sel, kind, classes, names)
    assert [x['id'] for x in labels] == list(range(1, len(sel) + 1)), 'labels not in id order'
    facing = np.array([x['facing'] for x in labels])
    framing = np.array([x['framing'] for x in labels])

    # 2. activations
    if (OUT / 'acts.npy').exists() and (OUT / 'photos.npy').exists():
        assert np.array_equal(np.load(OUT / 'photos.npy'), sel), 'cached photos.npy differs from the selection'
        a = np.load(OUT / 'acts.npy')
        print('acts: cached', a.shape, flush=True)
    else:
        import torch
        torch.set_num_threads(4)
        model = _bank().load_model(J / 'raw/weights/inception5h.pth')
        print('acts: running on', len(sel), 'photos', flush=True)
        a = acts(model, sel)
        np.save(OUT / 'photos.npy', sel)
        np.save(OUT / 'acts.npy', a)
    assert a.shape == (len(sel), NCH), a.shape

    # 3. per-channel numbers
    A = normalise(a)
    res = analyse(A, kind, facing, framing)
    rows = table(res)
    with open(OUT / 'table.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: f'{v:.6f}' if isinstance(v, float) else v for k, v in r.items()})
    print('table', len(rows), 'rows ->', OUT / 'table.csv')
    return dict(sel=sel, kind=kind, labels=labels, facing=facing, framing=framing, A=A, res=res, rows=rows)


def _min(a, b):
    return float(np.minimum(a, b))


QUANT = (
    ('min_eta2', lambda r: _min(r['eta2_V'], r['eta2_D']), decide_eta2),
    ('corr', lambda r: r['corr'], decide_corr),
    ('min_eta2f', lambda r: _min(r['eta2f_V'], r['eta2f_D']), decide_eta2),
    ('corrf', lambda r: r['corrf'], decide_corr),
)


def pct(v, x):
    """100 * share of the defined values of v that are <= x (NaN if x is NaN)"""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return 100 * float(np.mean(v <= x)) if np.isfinite(x) and len(v) else float('nan')


def line(r):
    t = {k: ' '.join(f'{r[f"t_{k}_{b}"]:.2f}' for b in FOLD) for k in 'VD'}
    return (f'{r["layer"]}:{r["channel"]:<4d} eta2_V={r["eta2_V"]:.3f} eta2_D={r["eta2_D"]:.3f} corr={r["corr"]:+.2f}'
            f' pref_V={r["pref_V"]} pref_D={r["pref_D"]} eta2f_V={r["eta2f_V"]:.3f} eta2f_D={r["eta2f_D"]:.3f}'
            f' corrf={r["corrf"]:+.2f} lr_V={r["lr_V"]:+.3f} lr_D={r["lr_D"]:+.3f} dogshare={r["dogshare"]:.2f}'
            f' t_V=[{t["V"]}] t_D=[{t["D"]}]')


def counts(kind, facing, framing):
    out = []
    cols = list(FRAMING) + [MISSING]
    for k, nm in (('V', 'vehicles'), ('D', 'dogs')):
        m = kind == k
        out.append(f'{nm}: n={int(m.sum())}, facing (rows) x framing (columns)')
        out.append(f'  {"":14s}' + ''.join(f'{c:>9s}' for c in cols + ['total']))
        for f in list(FACING) + [MISSING]:
            row = [int((m & (facing == f) & (framing == g)).sum()) for g in cols]
            out.append(f'  {f:14s}' + ''.join(f'{v:9d}' for v in row + [sum(row)]))
        tot = [int((m & (framing == g)).sum()) for g in cols]
        out.append(f'  {"total":14s}' + ''.join(f'{v:9d}' for v in tot + [sum(tot)]))
        fb = ' '.join(f'{b}={int((m & np.isin(facing, fs)).sum())}' for b, fs in FOLD.items())
        gb = ' '.join(f'{b}={int((m & (framing == b)).sum())}' for b in FRAME_BINS)
        sides = f'left={int((m & np.isin(facing, LEFT)).sum())} right={int((m & np.isin(facing, RIGHT)).sum())}'
        out.append(f'  folded facing: {fb}; framing: {gb}; {sides} (bins with < {MIN} photos are NaN)')
    return out


def report_lines(d, chs):
    rows, kind = d['rows'], d['kind']
    out = [f'photos: n={len(kind)} vehicles={int((kind == "V").sum())} dogs={int((kind == "D").sum())}', '']
    out += counts(kind, d['facing'], d['framing'])
    out += ['', 'labels: eta2 >= 0.14 LARGE, >= 0.06 MEDIUM, >= 0.01 SMALL, else NEGLIGIBLE;'
            ' corr >= 0.5 POSITIVE, <= -0.5 NEGATIVE, else WEAK', '']
    by = {layer: [r for r in rows if r['layer'] == layer] for layer, _ in LAYERS}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for layer, _ in LAYERS:
            parts = []
            for key in MEDIANS:
                v = np.array([r[key] for r in by[layer]])
                v = v[np.isfinite(v)]
                med = float(np.median(v)) if len(v) else float('nan')
                dec = decide_corr(med) if key.startswith('corr') else decide_eta2(med)
                parts.append(f'{key}={med:.3f} (n={len(v)}) -> {dec}')
            out.append(f'{layer} medians: ' + ', '.join(parts))
        out += ['', 't_V, t_D bins: ' + ', '.join(FOLD)]
        for layer, _ in LAYERS:
            key = np.array([_min(r['eta2_V'], r['eta2_D']) for r in by[layer]])
            fin = np.flatnonzero(np.isfinite(key))
            top = fin[np.argsort(-key[fin], kind='stable')[:15]]
            out.append(f'{layer}: top {len(top)} of {len(fin)} defined channels by min(eta2_V, eta2_D)')
            out += ['  ' + line(by[layer][i]) + f' -> {decide_eta2(key[i])}' for i in top]
        if chs:
            out += ['', 'requested channels (pct = percentile within the layer)']
            for layer, ch in chs:
                r = by[layer][ch]
                out.append(line(r))
                for name, f, dec in QUANT:
                    x = f(r)
                    out.append(f'    {name:10s} {x:+.3f} pct={pct([f(q) for q in by[layer]], x):.0f} -> {dec(x)}')
    return out


def sheet(layer, ch, d, n=8, tile=112, head=16, lab=28):
    c = col(layer, ch)
    A, kind, sel, labels = d['A'], d['kind'], d['sel'], d['labels']
    im = Image.new('RGB', (tile * n, head + 2 * (tile + lab)), 'white')
    dr = ImageDraw.Draw(im)
    dr.text((2, 2), f'{layer}:{ch}  row 1: top {n} vehicles, row 2: top {n} dogs', fill='black')
    for r, k in enumerate('VD'):
        pos = np.flatnonzero(kind == k)
        y0 = head + r * (tile + lab)
        for j, p in enumerate(pos[np.argsort(-A[pos, c], kind='stable')[:n]]):
            im.paste(photo(sel[p]).resize((tile, tile), Image.LANCZOS), (j * tile, y0))
            cap = f'{labels[p]["facing"]}/\n{labels[p]["framing"]} {A[p, c]:.2f}'
            dr.multiline_text((j * tile + 2, y0 + tile + 1), cap, fill='black', spacing=1)
    path = OUT / f'sheet_{layer}_{ch}.jpg'
    im.save(path, quality=90)
    return path


def selftest():
    nan = np.nan

    # selection
    classes = np.random.default_rng(12345).integers(0, 1000, 20000)
    sel, kind = select(classes)
    sel2, kind2 = select(classes)
    assert np.array_equal(sel, sel2) and np.array_equal(kind, kind2), 'select: not deterministic'
    assert len(np.unique(sel)) == len(sel), 'select: repeated photo'
    assert np.array_equal(np.sort(sel[kind == 'V']), np.flatnonzero(np.isin(classes, VEHICLES))), 'select: vehicles'
    dc = classes[sel[kind == 'D']]
    assert len(dc) == NDOG and ((dc >= DOG_LO) & (dc <= DOG_HI)).all(), 'select: dogs'
    assert not np.array_equal(sel, np.sort(sel)), 'select: not shuffled'

    # request content
    b = content([3, 4], ['AAA', 'BBB'])
    assert [x['type'] for x in b] == ['text', 'image', 'text', 'image', 'text'], 'content: block order'
    assert b[0]['text'] == 'Image 3:' and b[2]['text'] == 'Image 4:', 'content: id text'
    assert b[1] == {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': 'AAA'}}
    assert b[4]['text'] == PROMPT, 'content: prompt'
    assert PROMPT.startswith('For each image, consider') and PROMPT.endswith('"framing": "..."}]}'), 'prompt ends'
    assert '\n' not in PROMPT and '  ' not in PROMPT, 'prompt whitespace'
    for v in FACING + FRAMING:
        assert f'"{v}"' in PROMPT, ('prompt lacks', v)

    # label validation
    ids = [7, 8, 9]
    ent = [{'id': 7, 'facing': 'left', 'framing': 'whole'}, {'id': 8, 'facing': 'toward', 'framing': 'head'},
           {'id': 9, 'facing': 'unclear', 'framing': 'part'}]
    want = {7: ('left', 'whole'), 8: ('toward', 'head'), 9: ('unclear', 'part')}
    good = json.dumps({'images': ent})
    extra = json.dumps({'images': ent + [{'id': 10, 'facing': 'left', 'framing': 'head'}]})
    for text in (good, f'```json\n{good}\n```', f'Here you go.\n{good}\nDone.', extra):
        got, ok = check(text, ids)
        assert ok and got == want, ('check: valid reply rejected', text)
    miss = json.dumps({'images': ent[:2]})
    bad = json.dumps({'images': ent[:2] + [{'id': 9, 'facing': 'front', 'framing': 'part'}]})
    badf = json.dumps({'images': ent[:2] + [{'id': 9, 'facing': 'left', 'framing': 'body'}]})
    for text in (miss, bad, badf, 'no json here', '{"images": "x"}', '{"other": 1}'):
        assert not check(text, ids)[1], ('check: invalid reply accepted', text)

    def fake(replies):
        log = []

        def call(system, turns):
            log.append((system, turns))
            return [{'text': replies[len(log) - 1]}]
        return call, log

    datas = ['a', 'b', 'c']
    call, log = fake([good])
    labs, att = label_batch(ids, datas, call)
    assert len(log) == 1 and log[0] == ('', [content(ids, datas)]) and labs == want, 'label_batch: valid'
    call, log = fake([miss, good])
    labs, att = label_batch(ids, datas, call)
    assert len(log) == 2 and [a['ok'] for a in att] == [False, True] and labs == want, 'label_batch: retry'
    call, log = fake([miss, bad])
    labs, att = label_batch(ids, datas, call)
    assert len(log) == 2 and labs[7] == ('left', 'whole') and labs[9] == (MISSING, 'part'), 'label_batch: bad value'
    call, log = fake([bad, miss])
    labs, att = label_batch(ids, datas, call)
    assert len(log) == 2 and labs[9] == (MISSING, MISSING), 'label_batch: missing id'

    # folding
    mirror = {'toward_left': 'toward_right', 'left': 'right', 'away_left': 'away_right'}
    mirror.update({v: k for k, v in list(mirror.items())})
    fold = {f: b for b, fs in FOLD.items() for f in fs}
    assert sum(len(fs) for fs in FOLD.values()) == len(fold), 'fold: a facing in two bins'
    assert set(fold) == set(FACING) - {'unclear'}, 'fold: coverage'
    assert fold['toward'] == 'front' and fold['toward_right'] == 'three_quarter_front'
    assert fold['left'] == 'side' and fold['away'] == 'rear' and fold['away_left'] == 'rear'
    for f in fold:
        assert fold[f] == fold[mirror.get(f, f)], ('fold: left/right not merged', f)
    assert [mirror[f] for f in LEFT] == list(RIGHT), 'LEFT/RIGHT not mirror images'
    assert set(LEFT) | set(RIGHT) == set(FACING) - {'toward', 'away', 'unclear'}, 'LEFT/RIGHT coverage'

    # eta2 on synthetic data: groups of 5, 5, 4 (NaN bin) and 0 photos, 10 photos in no group
    n = 24
    idx = np.arange(n)
    groups = [(idx >= 0) & (idx < 5), (idx >= 5) & (idx < 10), (idx >= 10) & (idx < 14), np.zeros(n, bool)]
    A = np.zeros((n, 4))
    A[:, 0] = [0, 1, 2, 1, 1, 2, 3, 4, 3, 3] + [9] * 4 + [50] * 10  # SSB 10, SST 14
    A[:, 1] = 0.3  # constant channel
    A[:, 2] = nan  # channel whose max was 0
    A[:, 3] = [0, 2, 1, 0, 2, 2, 0, 1, 2, 0] + [5] * 14  # equal bin means
    t, e = tuning(A, groups)
    assert np.allclose(t[0], [1, 3, nan, nan], equal_nan=True) and np.isclose(e[0], 10 / 14), ('eta2 10/14', t[0], e[0])
    assert np.allclose(t[1, :2], 0.3) and np.isnan(t[1, 2:]).all() and np.isnan(e[1]), ('constant channel', t[1], e[1])
    assert np.isnan(t[2]).all() and np.isnan(e[2]), 'NaN channel'
    assert np.isclose(e[3], 0), ('eta2 0', e[3])
    assert np.isnan(tuning(A, [groups[0], groups[2]])[1]).all(), 'one defined bin must give NaN eta2'

    # corr on synthetic data
    x = np.array([[1, 2, 3, nan], [1, 2, nan, nan], [1, 1, 1, 1], [1, 2, 3, 4], [1, 2, 3, nan]])
    y = np.array([[2, 4, 6, 8], [1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 9]])
    r = pearson(x, y)
    assert np.isclose(r[0], 1) and np.isnan(r[1]) and np.isnan(r[2]), ('corr', r)
    assert np.isclose(r[3], -1) and np.isclose(r[4], 0.5), ('corr', r)

    # pref
    p = prefs(np.array([[1, nan, 3, nan], [nan] * 4, [2, 2, nan, 1]]), list(FOLD))
    assert p == ['side', 'none', 'tie'], p

    # lr sign
    m = np.ones(15, bool)
    A = np.array([0.9] * 5 + [0.2] * 5 + [0.5] * 5)[:, None]
    for fac in (np.array(['left'] * 5 + ['right'] * 5 + ['toward'] * 5),
                np.array(['toward_left', 'left', 'away_left', 'left', 'toward_left',
                          'toward_right', 'right', 'away_right', 'right', 'away_right'] + ['away'] * 5)):
        assert np.isclose(lr(A, m, fac)[0], 0.7), ('lr', lr(A, m, fac))
        assert np.isclose(lr(A, m, np.array([mirror.get(f, f) for f in fac]))[0], -0.7), 'lr mirrored'
    fac = np.array(['left'] * 4 + ['unclear'] + ['right'] * 5 + ['toward'] * 5)
    assert np.isnan(lr(A, m, fac)).all(), 'lr with 4 left photos must be NaN'

    # normalisation
    an = normalise(np.array([[0, 2, 1], [0, 4, 0.5]], dtype=np.float32))
    assert np.isnan(an[:, 0]).all() and np.allclose(an[:, 1:], [[0.5, 1], [1, 0.5]]), an

    # dogshare
    isdog = np.arange(60) < 30
    A = np.stack([np.where(isdog, 2.0, 1.0) + np.arange(60) * 1e-3, np.where(isdog, 1.0, 2.0), np.full(60, nan)], 1)
    s = dogshare(A, isdog)
    assert np.isclose(s[0], 0.6) and np.isclose(s[1], 0.4) and np.isnan(s[2]), s

    # everything through analyse: each raw facing x 5 photos per kind, channel value = folded bin code
    code = {f: float(j + 1) for j, fs in enumerate(FOLD.values()) for f in fs}
    fac = np.concatenate([np.repeat(FACING, MIN)] * 2)
    kind = np.repeat(['V', 'D'], MIN * len(FACING))
    v = np.array([code.get(f, 100.0) for f in fac])
    A = np.stack([v, np.where((kind == 'V') | (fac == 'unclear'), v, 5 - v)], axis=1)
    res = analyse(A, kind, fac, np.full(len(fac), 'whole'))
    assert np.allclose(res['t_V'][0], [1, 2, 3, 4]) and np.isclose(res['eta2_V'][0], 1), 'analyse: folding'
    assert np.isclose(res['corr'][0], 1), ('analyse corr', res['corr'])
    assert np.allclose(res['t_D'][1], [4, 3, 2, 1]) and np.isclose(res['corr'][1], -1), 'analyse: corr -1'
    assert res['pref_V'][0] == 'rear' and res['pref_D'][1] == 'front', 'analyse: pref'
    assert np.isnan(res['eta2f_V']).all() and np.isnan(res['corrf']).all(), 'analyse: one framing bin -> NaN'
    assert np.allclose(res['lr_V'], 0) and np.allclose(res['lr_D'], 0), 'analyse: lr'

    # channel specs, percentile, decisions
    assert _spec('4d:12,4c:7') == [('4d', 12), ('4c', 7)] and col('4d', 12) == 524 and col('4c', 7) == 7
    assert np.isclose(pct([0.1, 0.2, 0.3, nan], 0.2), 200 / 3) and np.isnan(pct([0.1], nan)), 'pct'
    assert [decide_eta2(x) for x in (nan, 0.005, 0.01, 0.06, 0.14)] == \
        ['UNDEFINED', 'NEGLIGIBLE', 'SMALL', 'MEDIUM', 'LARGE'], 'decide_eta2'
    assert [decide_corr(x) for x in (nan, -0.5, 0.0, 0.5)] == ['UNDEFINED', 'NEGATIVE', 'WEAK', 'POSITIVE']
    print('selftest ok')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    d = run()
    chs = _spec(sys.argv[sys.argv.index('--report') + 1]) if '--report' in sys.argv else []
    text = '\n'.join(report_lines(d, chs)) + '\n'
    (OUT / 'report.txt').write_text(text)
    print(text, end='')
    print('report ->', OUT / 'report.txt')
    if '--sheet' in sys.argv:
        for layer, ch in _spec(sys.argv[sys.argv.index('--sheet') + 1]):
            print('sheet', sheet(layer, ch, d))


if __name__ == '__main__':
    main()
