"""Round 20 mechanical checks, written by the orchestrator (not claim-blind), after the open-method results:
1. Mirror test for Zoom In's oriented dog-head pairs (4b:443/431, 4c:254/407): flip drop = 1 - a(mirror x)/a(x),
   median over each channel's top 20 of 1500 random card-pool photos, as a percentile among all channels of its
   layer; and the mirror-partner check: corr over the 1500 of A(mirror x) with B(x), against corr of A(x) with B(x).
2. Classes the essay names for 5a:719: rank of their mean activation among the 1000 classes (card + held pools).

usage: python -I tools/r20_checks.py   (writes runs/r20_checks/report.txt)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import ImageOps
from torchvision.models import GoogLeNet_Weights

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
sys.path.insert(0, str(J / 'bank'))
import bundlemax  # noqa: E402
from img_acts import Stop, load_model  # noqa: E402

PAIRS = [('mixed4b', 443, 431), ('mixed4c', 254, 407)]


def maxes(model, ims):
    got, out = {}, {'mixed4b': [], 'mixed4c': []}
    hs = [getattr(model, n).register_forward_hook(lambda _m, _i, o, n=n: got.__setitem__(n, o.detach())) for n in out]
    hs.append(model.mixed4d.register_forward_hook(lambda *_: (_ for _ in ()).throw(Stop)))
    for k in range(0, len(ims), 64):
        x = torch.from_numpy(np.stack([np.asarray(i, dtype=np.float32) for i in ims[k:k + 64]])).permute(0, 3, 1, 2) - 117
        try:
            with torch.no_grad():
                model(x)
        except Stop:
            pass
        for n in out:
            out[n].append(got[n].flatten(2).max(2).values.numpy())
    for h in hs:
        h.remove()
    return {n: np.concatenate(v) for n, v in out.items()}


def main():
    torch.set_num_threads(4)
    names = json.loads((J / 'bank/img/names.json').read_text())
    C, H = bundlemax.pools(names)
    ids = np.sort(np.random.default_rng(0).choice(C, 1500, replace=False))
    ims = [bundlemax.photo(i) for i in ids]
    model = load_model(J / 'raw/weights/inception5h.pth')
    a, af = maxes(model, ims), maxes(model, [ImageOps.mirror(i) for i in ims])
    L = ['1. Mirror test (1500 random card-pool photos; flip drop on each channel\'s own top 20)']
    for lay, p, q in PAIRS:
        top = np.argsort(-a[lay], axis=0)[:20]
        fd = np.array([np.median(1 - af[lay][top[:, c], c] / np.maximum(a[lay][top[:, c], c], 1e-6)) for c in range(a[lay].shape[1])])
        for c in (p, q):
            L.append(f'   {lay}:{c} flip drop {fd[c]:+.2f}, percentile {(fd < fd[c]).mean() * 100:.0f} (layer median {np.median(fd):+.2f}, 90th {np.percentile(fd, 90):+.2f})')
        r = lambda x, y: float(np.corrcoef(x, y)[0, 1])
        L.append(f'   partner {lay}:{p} vs {q}: corr A(x),B(x) {r(a[lay][:, p], a[lay][:, q]):.2f}; A(mirror x),B(x) '
                 f'{r(af[lay][:, p], a[lay][:, q]):.2f}; B(mirror x),A(x) {r(af[lay][:, q], a[lay][:, p]):.2f}')
    cls = np.load(J / 'bank/img/acts.npz')['classes']
    N = GoogLeNet_Weights.IMAGENET1K_V1.meta['categories']
    z = np.load(J / 'runs/openmethod/mixed5a:719/values.npz')
    ids5, v5 = np.concatenate([z['C'], z['H']]), np.concatenate([z['vC'], z['vH']])
    M = z['vC'].max()
    means = np.array([v5[cls[ids5] == k].mean() / M if (cls[ids5] == k).any() else np.nan for k in range(1000)])
    rank = {k: int((means > means[k]).sum()) + 1 for k in range(1000)}
    L.append('2. mixed5a:719 class means (share of card max, all 10k photos), rank among 1000')
    L.append('   top 10: ' + ', '.join(f'{N[k]} {means[k]:.2f}' for k in np.argsort(-means)[:10]))
    named = ['bow tie', 'admiral', 'ringlet', 'monarch', 'cabbage butterfly', 'sulphur butterfly', 'lycaenid', 'bikini',
             'maillot', 'maillot tank suit', 'brassiere', 'Loafer', 'sandal', 'purse', 'swimming trunks']
    for n in named:
        ks = [k for k in range(1000) if N[k] == n]
        L.append('   ' + '; '.join(f'{n} {means[k]:.2f} rank {rank[k]}' for k in ks))
    top20 = z['C'][np.argsort(-z['vC'])[:20]]
    L.append('   card top-20 labels: ' + ', '.join(N[cls[i]] for i in top20))
    (J / 'runs/r20_checks/report.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
