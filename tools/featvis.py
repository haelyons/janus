"""Feature-visualisation renders (lucent, standard settings, 256 steps, 128px) for InceptionV1 mixed4e:55, its
strongest inputs by weight (mixed4d, 6 positive and 2 negative) and its strongest readers by weight (mixed5a's 1x1
branch, 3). The essay's circuit view is weights plus renders; this reproduces that view with standard tools.

usage: python -I tools/featvis.py   (writes runs/featvis/renders.jpg)
"""
import sys
from pathlib import Path

import kornia
import numpy as np
import torch
from PIL import Image, ImageDraw

kornia.get_rotation_matrix2d = kornia.geometry.transform.get_rotation_matrix2d  # lucent wants kornia <= 0.4.1
kornia.warp_affine = kornia.geometry.transform.warp_affine
from lucent.optvis import render

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'bank'))
from img_acts import CH, load_model


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = load_model(J / 'raw/weights/inception5h.pth')
    w_in = model.mixed4e_1x1_pre_relu_conv.weight[CH, :, 0, 0].detach().numpy()
    w_out = model.mixed5a_1x1_pre_relu_conv.weight[:, CH, 0, 0].detach().numpy()
    jobs = [(f'mixed4e:{CH}', 'the unit')]
    jobs += [(f'mixed4d:{c}', f'input w={w_in[c]:+.3f}') for c in np.argsort(-w_in)[:6]]
    jobs += [(f'mixed4d:{c}', f'input w={w_in[c]:+.3f}') for c in np.argsort(w_in)[:2]]
    jobs += [(f'mixed5a:{c}', f'reader w={w_out[c]:+.3f}') for c in np.argsort(-w_out)[:3]]
    out = J / 'runs/featvis'
    out.mkdir(parents=True, exist_ok=True)
    S = 256
    g = Image.new('RGB', (4 * S, 3 * (S + 24)), (255, 255, 255))
    d = ImageDraw.Draw(g)
    for n, (obj, lab) in enumerate(jobs):
        im = render.render_vis(model, obj, thresholds=(256,), show_image=False, progress=False)[-1][0]
        tile = Image.fromarray((im * 255).astype(np.uint8)).resize((S, S), Image.BICUBIC)
        tile.save(out / f'{obj.replace(":", "_")}.png')
        x, y = (n % 4) * S, (n // 4) * (S + 24)
        g.paste(tile, (x, y))
        d.text((x + 6, y + S + 5), f'{obj}  {lab}', fill=(0, 0, 0))
        print(obj, lab, flush=True)
    g.save(out / 'renders.jpg', quality=90)


if __name__ == '__main__':
    main()
