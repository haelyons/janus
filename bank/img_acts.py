"""Record InceptionV1 mixed4e channel 55 on every ImageNetV2 (matched-frequency) image.

Saves a 224px center-crop thumbnail per image and the pre-ReLU 14x14 map of the channel.
Channel 55 sits in the 1x1 branch (channels 0-255 of mixed4e), so its pre-ReLU value is
mixed4e_1x1_pre_relu_conv[:, 55]. Input range follows lucid/lucent: x*255 - 117.

usage: python -I bank/img_acts.py <tar> <out_dir> [--selftest]
"""
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CH = 55


class Stop(Exception):
    pass


def load_model(weights):
    from lucent.modelzoo.inceptionv1.InceptionV1 import InceptionV1
    m = InceptionV1(pretrained=False)
    m.load_state_dict(torch.load(weights, map_location='cpu'))
    return m.eval()


def crop224(img):
    img = img.convert('RGB')
    w, h = img.size
    s = 224 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return img.crop((l, t, l + 224, t + 224))


def run_batch(model, imgs):
    x = torch.from_numpy(np.stack([np.asarray(i, dtype=np.float32) / 255.0 for i in imgs])).permute(0, 3, 1, 2)
    x = x * 255 - 117
    got = {}

    def hook(_m, _i, out):
        got['a'] = out[:, CH].detach().clone()
        raise Stop

    h = model.mixed4e_1x1_pre_relu_conv.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(x)
    except Stop:
        pass
    finally:
        h.remove()
    return got['a'].numpy()


def selftest(weights):
    model = load_model(weights)
    a = run_batch(model, [Image.new('RGB', (224, 224), (128, 128, 128))] * 2)
    assert a.shape == (2, 14, 14), a.shape
    assert np.allclose(a[0], a[1]), 'same image must give same map'
    print('selftest ok', a.shape, float(a.max()), float(a.min()))


def main():
    tar_path, out = sys.argv[1], Path(sys.argv[2])
    weights = Path(__file__).resolve().parent.parent / 'raw/weights/inception5h.pth'
    if '--selftest' in sys.argv:
        return selftest(weights)
    torch.set_num_threads(4)
    model = load_model(weights)
    (out / 'thumbs').mkdir(parents=True, exist_ok=True)
    names, classes, maps = [], [], []
    buf, meta = [], []

    def flush():
        if not buf:
            return
        maps.append(run_batch(model, buf).astype(np.float16))
        for img, (name, cls) in zip(buf, meta):
            img.save(out / 'thumbs' / f'{len(names):05d}.jpg', quality=88)
            names.append(name)
            classes.append(cls)
        buf.clear()
        meta.clear()
        if len(names) % 1024 == 0:
            print(len(names), flush=True)

    with tarfile.open(tar_path, 'r|') as tf:
        for mem in tf:
            if not mem.isfile() or not mem.name.lower().endswith(('.jpeg', '.jpg', '.png')):
                continue
            cls = int(Path(mem.name).parent.name)
            img = crop224(Image.open(io.BytesIO(tf.extractfile(mem).read())))
            buf.append(img)
            meta.append((mem.name, cls))
            if len(buf) == 64:
                flush()
    flush()
    m = np.concatenate(maps)
    np.savez_compressed(out / 'acts.npz', maps=m, classes=np.array(classes))
    (out / 'names.json').write_text(json.dumps(names))
    print('done', m.shape, 'max', float(m.max()), 'min', float(m.min()))


if __name__ == '__main__':
    main()
