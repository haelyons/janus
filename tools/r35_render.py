"""Round 35, Inception: one feature visualisation per unit of runs/r35/img/groups.json (PREDICTIONS.md, round 35).

lucent, standard parameterisation and transforms, 256 steps, 128 px, torch seed 0 before each unit; objective: mean
over positions of the layer output dotted with the unit's vector (one-hot for a channel, the same as lucent's channel
objective). Resumable: a unit whose png exists is skipped.

usage: python -I tools/r35_render.py [shard/n]   (writes runs/r35/img/vis/<layer>_<unit>.png; on CUDA when available)
"""
import json
import sys
from pathlib import Path

import kornia
import numpy as np
import torch
from PIL import Image

kornia.get_rotation_matrix2d = kornia.geometry.transform.get_rotation_matrix2d  # lucent wants kornia <= 0.4.1
kornia.warp_affine = kornia.geometry.transform.warp_affine
from lucent.optvis import objectives, render  # noqa: E402

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
sys.path.insert(0, str(J / 'bank'))
import proploop  # noqa: E402
from img_acts import load_model  # noqa: E402

OUT = J / 'runs/r35/img'


@objectives.wrap_objective()
def dot(layer, vec):
    v = torch.tensor(vec, dtype=torch.float32).view(1, -1, 1, 1)

    def inner(model):
        a = model(layer)
        return -(a * v.to(a.device)).sum(1).mean()
    return inner


def vec_of(u):
    if u['kind'] == 'dir':
        return np.load(OUT / f'dirs_{u["layer"]}.npy')[:, u['index']]
    e = np.zeros(proploop.NCH[u['layer']], dtype=np.float32)
    e[u['index']] = 1.0
    return e


def main():
    shard, nshard = (int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '0/1').split('/'))
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(proploop.WEIGHTS).to(device)
    (OUT / 'vis').mkdir(parents=True, exist_ok=True)
    units = [u for g in json.loads((OUT / 'groups.json').read_text())['groups'] for u in g['units']]
    for n, u in enumerate(units):
        p = OUT / 'vis' / f'{u["name"].replace(":", "_")}.png'
        if p.exists() or n % nshard != shard:
            continue
        torch.manual_seed(0)
        im = render.render_vis(model, dot(u['layer'], vec_of(u)), thresholds=(256,), show_image=False,
                               progress=False)[-1][0]
        Image.fromarray((im * 255).astype(np.uint8)).save(p)
        print(f'{n + 1}/{len(units)} {u["name"]}', flush=True)


if __name__ == '__main__':
    main()
