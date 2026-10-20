"""Record the circuit into InceptionV1 mixed4e:55. Its pre-ReLU value is a 1x1 conv over mixed4d's 528 channels:
pre(y, x) = sum_c w[c] * mixed4d[c, y, x] + b. For every photo, save each input channel's spatial max and its
contribution w[c] * mixed4d[c] at the unit's own hotspot.

usage: python -I tools/circuit.py <imagenetv2 tar> (writes runs/circuit/circuit.npz; checks the sum against
bank/img/acts.npz)
"""
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'bank'))
from img_acts import CH, Stop, crop224, load_model


def main():
    torch.set_num_threads(4)
    model = load_model(J / 'raw/weights/inception5h.pth')
    w = model.mixed4e_1x1_pre_relu_conv.weight[CH, :, 0, 0].detach()
    b = float(model.mixed4e_1x1_pre_relu_conv.bias[CH])
    stored = np.load(J / 'bank/img/acts.npz')['maps'].astype(np.float32)
    got = {}

    def grab(_m, _i, out):
        got['a'] = out.detach()

    def stop(_m, _i, out):
        raise Stop

    model.mixed4d.register_forward_hook(grab)
    model.mixed4e_1x1_pre_relu_conv.register_forward_hook(stop)
    smax, contrib, err, buf = [], [], [], []

    def flush():
        x = torch.from_numpy(np.stack([np.asarray(i, dtype=np.float32) for i in buf])).permute(0, 3, 1, 2) - 117
        try:
            with torch.no_grad():
                model(x)
        except Stop:
            pass
        a = got['a']                                     # (n, 528, 14, 14)
        pre = torch.einsum('c,ncyx->nyx', w, a) + b      # the unit, rebuilt from its inputs
        n0 = len(smax and np.concatenate(smax)) if smax else 0
        err.append(float((pre.numpy() - stored[n0:n0 + len(buf)]).__abs__().max()))
        loc = pre.flatten(1).argmax(1)
        af = a.flatten(2)                                # (n, 528, 196)
        contrib.append((w[None] * af[torch.arange(len(buf)), :, loc]).numpy().astype(np.float16))
        smax.append(af.max(2).values.numpy().astype(np.float16))
        buf.clear()

    with tarfile.open(sys.argv[1], 'r|') as tf:
        for mem in tf:
            if mem.isfile() and mem.name.lower().endswith(('.jpeg', '.jpg', '.png')):
                buf.append(crop224(Image.open(io.BytesIO(tf.extractfile(mem).read()))))
                if len(buf) == 64:
                    flush()
    if buf:
        flush()
    out = J / 'runs/circuit'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'circuit.npz', smax=np.concatenate(smax), contrib=np.concatenate(contrib), w=w.numpy(), b=b)
    print('max |rebuilt - stored| per batch, worst:', max(err), '(float16 storage; stored max is ~1527)')


if __name__ == '__main__':
    main()
