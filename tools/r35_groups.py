"""Round 35, Inception: the 13 groups, their values on the 10,000 photos, the photos of each page (PREDICTIONS.md, round 35).

Groups: 3 explained, 1 named-failed, and per layer 3b, 4a, 4b (drawn in this order, each excluding every unit already
placed in its layer) random neurons, a cluster (seed + its 3 most correlated channels over card-half photos) and
random directions. A unit's value on a photo is the spatial max of the layer output projected on its vector (one-hot
for a channel). Photo halves: crc32("r35:" + name) % 2, 0 = card half, 1 = held-out half.
Per unit: top 5 card-half photos with the argmax position of the unit's map (blind page); 4 highest held-out photos
and a near miss for each (same class, lowest value; else a random held-out photo below the unit's held-out median).
Pages coded I01-I13 in seeded random order, rows a-d in seeded random order.

Outputs runs/r35/img/: groups.json, values.npy (units x photos, in groups.json order), dirs_<layer>.npy.

usage: python -I tools/r35_groups.py [--device cpu] [--selftest]
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
import bundlemax  # noqa: E402
import proploop  # noqa: E402
import r32_basis  # noqa: E402

OUT = J / 'runs/r35/img'
CTRL_LAYERS = ('mixed3b', 'mixed4a', 'mixed4b')
EXPLAINED = {
    'mixed3b': ([379, 385, 343, 342],
                'Curve detectors: each responds to a curve at its own preferred orientation (Curve Detectors). All four '
                'are at the 98th-100th percentile of the layer for curve response on synthetic arcs (round 17a); '
                'natural photos untested.'),
    'mixed4a': ([222, 234, 308, 325],
                "Dog heads, by facing (Zoom In's 4a pathways): 222 prefers dogs facing left, 234 and 308 facing right, "
                'consistent with the Equivariance flip pair 222/308 (round 30A); 325 a dog-head unit whose facing row '
                'does not survive the joint control (round 32).'),
    'mixed4b': ([443, 431, 429, 384],
                'Oriented dog heads: 443 and 431 prefer dogs facing right, 429 and 384 facing left; 99th percentile of '
                'the layer, dog-specific, mirroring the photos flips the sign (round 23B); each keeps its facing row '
                'with all eight questions in one fit (round 32).'),
}
FAILED = ([('mixed5a', 160), ('mixed5a', 23), ('mixed5b', 573), ('mixed5b', 907)],
          'Car-circuit labels: side views of cars. Not supported: no stable within-class side-view effect on either '
          'half of the photos (round 30A).')
EVIDENCE = {'mixed3b': 'strong on synthetic stimuli', 'mixed4a': 'strong (3 of 4 units)', 'mixed4b': 'strong'}
N_TOP, N_POS = 5, 4


def rng(tag):
    return np.random.default_rng(zlib.crc32(tag.encode()))


def half(name):
    return zlib.crc32(f'r35:{name}'.encode()) % 2


def draw_chans(layer, placed, tag, k):
    pool = np.array([c for c in range(proploop.NCH[layer]) if c not in placed], dtype=np.int64)
    return [int(x) for x in rng(f'r35:{tag}:{layer}').choice(pool, size=k, replace=False)]


def neighbours(V, seed, placed, k=3):
    """the k channels (not placed, not seed) with the highest Pearson correlation with seed over V's rows"""
    Z = (V - V.mean(0)) / np.where(V.std(0) > 0, V.std(0), np.inf)
    r = Z.T @ Z[:, seed] / len(V)
    r[[seed, *placed]] = -np.inf
    r[~np.isfinite(r)] = -np.inf
    return [int(x) for x in np.argsort(-r, kind='stable')[:k]]


def scoring(v, ho, classes, tag):
    """(pos, nm): the N_POS highest held-out photos; for each, the held-out photo of its class with the lowest value
    (not already used), else a random held-out photo below the held-out median"""
    pos = [int(i) for i in ho[np.argsort(-v[ho], kind='stable')[:N_POS]]]
    used, nm, r = set(pos), [], rng(f'r35:nm:{tag}')
    med = np.median(v[ho])
    for p in pos:
        same = [int(i) for i in ho if classes[i] == classes[p] and int(i) not in used]
        if same:
            q = min(same, key=lambda i: (v[i], i))
        else:
            low = [int(i) for i in ho if v[i] < med and int(i) not in used]
            q = int(r.choice(low))
        used.add(q)
        nm.append(q)
    return pos, nm


def argmax_pos(model, layer, vec, ids, device):
    """per photo id, the (x, y) pixel centre of the argmax of the unit's projected map on the 224 px photo"""
    import torch
    mod, got = dict(model.named_modules())[layer], {}
    h = mod.register_forward_hook(lambda _m, _i, out: got.__setitem__('F', out.detach().cpu().numpy()))
    try:
        with torch.no_grad():
            model(bundlemax.net_input(ids).to(device))
    finally:
        h.remove()
    M = np.tensordot(got['F'], vec, axes=([1], [0]))  # n x h x w
    res = []
    for m in M:
        r, c = np.unravel_index(int(np.argmax(m)), m.shape)
        res.append([round((c + 0.5) / m.shape[1] * 224, 1), round((r + 0.5) / m.shape[0] * 224, 1)])
    return res


def one_pass(model, dirs, ids, device):
    """{layer: len(ids) x 4 direction values} for CTRL_LAYERS, from one forward pass per batch hooked at every
    CTRL_LAYER and stopped after the deepest; the spatial max of every mixed3b channel goes to proploop's cache
    (values_mixed3b.npy, as proploop.layer_values would write it) when missing"""
    import torch
    Stop, mods, got = bundlemax._bank().Stop, dict(model.named_modules()), {}
    need3b = not (proploop.OUT / 'values_mixed3b.npy').exists()

    def hook_for(L):
        def hook(_m, _i, out):
            got[L] = out.detach().cpu().numpy()
            if L == CTRL_LAYERS[-1]:
                raise Stop
        return hook

    dv, v3b = {L: [] for L in CTRL_LAYERS}, []
    hs = [mods[L].register_forward_hook(hook_for(L)) for L in CTRL_LAYERS]
    try:
        for b, s in enumerate(range(0, len(ids), proploop.BATCH)):
            got.clear()
            try:
                with torch.no_grad():
                    model(bundlemax.net_input(ids[s:s + proploop.BATCH]).to(device))
            except Stop:
                pass
            for L in CTRL_LAYERS:
                dv[L].append(r32_basis.project_max(got[L], dirs[L]))
            if need3b:
                v3b.append(got['mixed3b'].max(axis=(2, 3)).astype(np.float32))
            if b % 20 == 19:
                print(f'  one pass {min(s + proploop.BATCH, len(ids))}/{len(ids)}', flush=True)
    finally:
        for h in hs:
            h.remove()
    if need3b:
        tmp = proploop.OUT / 'values_mixed3b.tmp.npy'
        np.save(tmp, np.ascontiguousarray(np.concatenate(v3b), dtype=np.float32))
        tmp.replace(proploop.OUT / 'values_mixed3b.npy')
    return {L: np.concatenate(dv[L]) for L in CTRL_LAYERS}


def build(device):
    load_model = bundlemax._bank().load_model
    names = json.loads(proploop.NAMES.read_text())
    with np.load(proploop.ACTS) as z:
        classes = np.asarray(z['classes'], dtype=np.int64)
    ids = list(range(proploop.N_PHOTOS))
    card = np.array([i for i in ids if half(names[i]) == 0])
    ho = np.array([i for i in ids if half(names[i]) == 1])
    model = load_model(proploop.WEIGHTS).to(device)
    dirs = {}
    for layer in CTRL_LAYERS:
        W = rng(f'r35:dir:{layer}').standard_normal((proploop.NCH[layer], 4))
        dirs[layer] = W / np.linalg.norm(W, axis=0, keepdims=True)
    OUT.mkdir(parents=True, exist_ok=True)
    dv = one_pass(model, dirs, ids, device)
    vals = proploop.layer_values(['mixed3b', 'mixed4a', 'mixed4b', 'mixed5a', 'mixed5b'], lambda d: model, ids, device)
    groups = []

    def chan_units(layer, chans):
        return [{'name': f'{layer}:{c}', 'layer': layer, 'kind': 'chan', 'index': c} for c in chans]

    for layer, (chans, text) in EXPLAINED.items():
        groups.append({'gid': f'explained_{layer}', 'category': 'explained', 'units': chan_units(layer, chans),
                       'explanation': text, 'evidence': EVIDENCE[layer]})
    groups.append({'gid': 'failed_side', 'category': 'named_failed',
                   'units': [u for L, c in FAILED[0] for u in chan_units(L, [c])],
                   'explanation': FAILED[1], 'evidence': 'failed'})
    for layer in CTRL_LAYERS:
        placed = set(EXPLAINED[layer][0])
        rn = draw_chans(layer, placed, 'chan', 4)
        placed |= set(rn)
        seed = draw_chans(layer, placed, 'clu', 1)[0]
        placed.add(seed)
        clu = [seed] + neighbours(vals[layer][card], seed, placed)
        groups.append({'gid': f'random_{layer}', 'category': 'random_neurons', 'units': chan_units(layer, rn)})
        groups.append({'gid': f'cluster_{layer}', 'category': 'cluster', 'units': chan_units(layer, clu)})
        groups.append({'gid': f'direction_{layer}', 'category': 'random_directions',
                       'units': [{'name': f'{layer}:dir{j}', 'layer': layer, 'kind': 'dir', 'index': j}
                                 for j in range(4)]})
    for layer in CTRL_LAYERS:
        np.save(OUT / f'dirs_{layer}.npy', dirs[layer].astype(np.float32))

    def vec(u):
        if u['kind'] == 'dir':
            return dirs[u['layer']][:, u['index']]
        e = np.zeros(proploop.NCH[u['layer']])
        e[u['index']] = 1.0
        return e

    rows = []
    for g in groups:
        for u in g['units']:
            v = dv[u['layer']][:, u['index']] if u['kind'] == 'dir' else vals[u['layer']][:, u['index']]
            v = np.asarray(v, dtype=np.float32)
            rows.append(v)
            top = [int(i) for i in card[np.argsort(-v[card], kind='stable')[:N_TOP]]]
            pos, nm = scoring(v, ho, classes, u['name'])
            u.update(top=top, top_xy=argmax_pos(model, u['layer'], vec(u), top, device), pos=pos, near=nm,
                     card_max=float(v[card].max()))
    codes = rng('r35:codes:img').permutation(len(groups))
    for g, c in zip(groups, codes):
        g['code'] = f'I{int(c) + 1:02d}'
        g['rows'] = [g['units'][int(j)]['name'] for j in rng(f'r35:rows:{g["gid"]}').permutation(4)]
    np.save(OUT / 'values.npy', np.stack(rows))
    (OUT / 'groups.json').write_text(json.dumps({'card_n': len(card), 'heldout_n': len(ho), 'groups': groups},
                                                indent=1))
    for g in sorted(groups, key=lambda g: g['code']):
        print(g['code'], g['gid'], [u['name'] for u in g['units']], flush=True)


def selftest():
    assert half('x') == zlib.crc32(b'r35:x') % 2
    V = np.random.default_rng(0).standard_normal((200, 6))
    V[:, 3] = V[:, 0] * 2 + 0.01 * V[:, 3]
    V[:, 5] = V[:, 0] + V[:, 5]
    assert neighbours(V, 0, {3}, k=1) == [5]
    v = np.array([9, 1, 8, 2, 7, 3, 0, 5], dtype=np.float32)
    classes = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    pos, nm = scoring(v, np.arange(8), classes, 't')
    assert pos == [0, 2, 4, 7] and nm == [1, 3, 5, 6], (pos, nm)
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import torch
    torch.set_num_threads(4)
    build(a.device)


if __name__ == '__main__':
    main()
