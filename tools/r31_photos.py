"""A larger photo pool from ImageNet train, prepared as bank/img/thumbs, with the network's values (spec:
authors/spec_r31_photos.md).

The ImageNet train mirror is a directory of Parquet shards (train-*.parquet, read in sorted file-name order), each row
an image (struct with bytes) and a label (0-999). A first pass reads only the label column and collects the key
('<shard file name>:<row>') of every row of the classes in --classes; per class the --per-class keys with the lowest
zlib.crc32 are chosen (ties by key). A second pass reads the chosen rows' images: decoded, converted to RGB,
centre-cropped to 224 x 224 (img_acts.crop224), saved as JPEG quality 88. Pool ids are 0, 1, ... by class in --classes
order, then the class's chosen order. Then the spatial max of every channel of each layer per pool photo
(proploop.forward_values; network input as bundlemax.net_input).

Outputs in --out: thumbs/<id:06d>.jpg, names.json (the key of each pool id), labels.npy (int64 class per pool id),
values_<layer>.npy (float32, pool ids x channels).

usage: python -I tools/r31_photos.py --shards <dir> --classes <file.json> --out <dir> [--per-class 50]
       [--layers mixed4a,mixed4b,mixed4c,mixed4d,mixed4e,mixed5a,mixed5b] [--device cuda] [--selftest]
"""
import argparse
import io
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))  # -I does not add the script dir
import bundlemax  # noqa: E402
import proploop  # noqa: E402

PER_CLASS = 50
LAYERS = ('mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b')
SHARD_GLOB = 'train-*.parquet'
QUALITY = 88  # as bank/img_acts.py saves thumbnails


# ---------- keys and choice ----------

def row_key(shard_name, row):
    """'<shard file name>:<row>', row the row's index in its shard (from 0, over all row groups)"""
    return f'{shard_name}:{row}'


def _order(key):
    return zlib.crc32(key.encode()), key


def choose(keys_by_class, per_class):
    """{class: its per_class keys with the lowest zlib.crc32(key.encode()), ties by key (all when fewer), in that
    order}"""
    return {c: sorted(keys, key=_order)[:per_class] for c, keys in keys_by_class.items()}


def load_classes(path):
    """the class indices of the JSON list at path, in file order: distinct ints 0-999"""
    cl = json.loads(Path(path).read_text())
    assert isinstance(cl, list) and cl, f'{path}: expected a non-empty list'
    assert all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c < 1000 for c in cl), cl
    assert len(set(cl)) == len(cl), f'{path}: a class is repeated'
    return cl


# ---------- photos ----------

def thumb(image_bytes):
    """the decoded image, converted to RGB, then img_acts.crop224 (224 x 224 RGB)"""
    with Image.open(io.BytesIO(image_bytes)) as im:
        img = im.convert('RGB')
    return bundlemax._bank().crop224(img)


def make_input(thumb_dir):
    """a function of a list of pool ids -> the network input as bundlemax.net_input (pixels 0-255 - 117, NCHW,
    float32), loading thumb_dir/<id:06d>.jpg (each 224 x 224 RGB)"""
    thumb_dir = Path(thumb_dir)

    def photo(i):
        with Image.open(thumb_dir / f'{int(i):06d}.jpg') as im:
            img = im.convert('RGB')
        assert img.size == (224, 224), (i, img.size)
        return img

    def net_input(ids):
        import torch
        x = torch.from_numpy(np.stack([np.asarray(photo(i), dtype=np.float32) / 255.0 for i in ids])).permute(0, 3, 1, 2)
        return x * 255 - 117
    return net_input


# ---------- shards ----------

def shard_paths(shards):
    """the train-*.parquet files of the directory shards, sorted by file name"""
    paths = sorted(Path(shards).glob(SHARD_GLOB), key=lambda p: p.name)
    assert paths, f'no {SHARD_GLOB} in {shards}'
    return paths


def collect(paths, classes):
    """pass 1: {class: the keys of its rows, in shard then row order} for each of classes, reading only the label
    column"""
    found = {int(c): [] for c in classes}
    want = np.array(sorted(found), dtype=np.int64)
    for n, p in enumerate(paths, 1):
        with open(p, 'rb') as f:
            lab = pq.ParquetFile(f).read(columns=['label']).column('label').to_numpy()
        for r in np.flatnonzero(np.isin(lab, want)).tolist():
            found[int(lab[r])].append(row_key(p.name, r))
        if n % 20 == 0 or n == len(paths):
            print(f'  labels {n}/{len(paths)} shards', flush=True)
    return found


def read_images(paths, keys):
    """pass 2: (key, image bytes, label) for each of keys, in shard then row order; per shard only the row groups that
    hold one of its rows are read, and of them only the image and label columns"""
    rows_of = {}
    for k in keys:
        s, r = k.rsplit(':', 1)
        rows_of.setdefault(s, set()).add(int(r))
    for p in paths:
        rows = rows_of.pop(p.name, None)
        if not rows:
            continue
        with open(p, 'rb') as f:
            pf = pq.ParquetFile(f)
            start = 0
            for g in range(pf.num_row_groups):
                n = pf.metadata.row_group(g).num_rows
                local = sorted(r - start for r in rows if start <= r < start + n)
                if local:
                    t = pf.read_row_group(g, columns=['image', 'label']).take(pa.array(local, type=pa.int64()))
                    for r, im, lab in zip(local, t.column('image').to_pylist(), t.column('label').to_pylist()):
                        yield row_key(p.name, start + r), im['bytes'], int(lab)
                start += n
    assert not rows_of, f'keys of shards not found: {sorted(rows_of)[:5]}'


# ---------- pool ----------

def build_pool(paths, classes, per_class, out):
    """pass 1, choose, pass 2: writes out/thumbs/<id:06d>.jpg, out/names.json, out/labels.npy; prints the classes with
    fewer than per_class rows and the pool size. -> (names, labels)"""
    out = Path(out)
    found = collect(paths, classes)
    chosen = choose(found, per_class)
    for c in classes:
        if len(found[c]) < per_class:
            print(f'class {c}: {len(found[c])} rows, fewer than {per_class}', flush=True)
    names = [k for c in classes for k in chosen[c]]
    labels = np.array([c for c in classes for _ in chosen[c]], dtype=np.int64)
    print('pool size', len(names), flush=True)
    assert names, 'no photos of these classes'
    pid = {k: i for i, k in enumerate(names)}
    tdir = out / 'thumbs'
    tdir.mkdir(parents=True, exist_ok=True)
    done = set()
    for k, b, lab in read_images(paths, names):
        i = pid[k]
        assert lab == labels[i], (k, lab, int(labels[i]))
        thumb(b).save(tdir / f'{i:06d}.jpg', quality=QUALITY)
        done.add(i)
        if len(done) % 1000 == 0:
            print(f'  thumbs {len(done)}/{len(names)}', flush=True)
    assert len(done) == len(names), (len(done), len(names))
    (out / 'names.json').write_text(json.dumps(names))
    np.save(out / 'labels.npy', labels)
    return names, labels


def write_values(out, n, layers, device):
    """out/values_<layer>.npy for each layer: proploop.forward_values over pool ids 0..n-1 (thumbnails in out/thumbs),
    float32, n x channels; the model as bundlemax._bank().load_model(proploop.WEIGHTS), gradients off, eval mode;
    full float32 on CUDA (proploop.no_tf32, as proploop.layer_values)"""
    import torch
    out = Path(out)
    torch.set_num_threads(4)
    proploop.no_tf32()
    model = bundlemax._bank().load_model(proploop.WEIGHTS)
    for q in model.parameters():
        q.requires_grad_(False)
    model = model.to(device)
    print('values of', list(layers), 'on', device, flush=True)
    vals = proploop.forward_values(model, layers, np.arange(n), device, make_input=make_input(out / 'thumbs'))
    for L in layers:
        v = np.ascontiguousarray(vals[L], dtype=np.float32)
        assert v.shape == (n, proploop.NCH[L]), (L, v.shape)
        tmp = out / f'values_{L}.tmp.npy'
        np.save(tmp, v)
        tmp.replace(out / f'values_{L}.npy')


# ---------- selftest ----------

def selftest():
    import tempfile
    import torch

    def enc(img, fmt, **kw):
        b = io.BytesIO()
        img.save(b, format=fmt, **kw)
        return b.getvalue()

    def crc(k):
        return zlib.crc32(k.encode())

    # row keys
    assert row_key('train-00003-of-00257.parquet', 17) == 'train-00003-of-00257.parquet:17'
    assert row_key(Path('/data/in1k/train-00000-of-00257.parquet').name, 0) == 'train-00000-of-00257.parquet:0'

    # choose: lowest crc32 first, ties by key, all when fewer, each list in that order, input unchanged
    kbc = {7: [row_key(f'train-{s:05d}.parquet', r) for s in range(3) for r in range(40)] + ['plumless', 'buckeroo'],
           2: ['b:1', 'a:1', 'c:9'], 5: []}
    before = {c: list(v) for c, v in kbc.items()}
    for per in (1, 10, 122, 500):
        ch = choose(kbc, per)
        assert kbc == before and sorted(ch) == [2, 5, 7], sorted(ch)
        for c, got in ch.items():
            assert len(got) == min(per, len(kbc[c])) and len(set(got)) == len(got) and set(got) <= set(kbc[c])
            for a, b in zip(got, got[1:]):
                assert crc(a) < crc(b) or (crc(a) == crc(b) and a < b), (a, b)
            for k in set(kbc[c]) - set(got):  # every key left out comes after the last one chosen
                assert crc(k) > crc(got[-1]) or (crc(k) == crc(got[-1]) and k > got[-1]), (k, got[-1])
    assert len(choose(kbc, 500)[7]) == 122 and choose(kbc, 2)[5] == [] and choose({}, 5) == {}
    pair = ['plumless', 'buckeroo']  # a crc32 collision pair, if equal: the tie goes to the lower key
    assert choose({0: pair}, 1) == {0: ['buckeroo'] if crc(pair[0]) == crc(pair[1]) else [min(pair, key=crc)]}

    # thumb: decoded, RGB, crop224 (a 300 x 200 photo is resized to 336 x 224 and cropped from x = 56)
    a = np.zeros((200, 300, 3), np.uint8)
    a[:, :100, 0] = 255
    a[:, 100:200, 1] = 255
    a[:, 200:, 2] = 255
    src = Image.fromarray(a)  # 300 wide, 200 high: red, green, blue thirds
    t = thumb(enc(src, 'PNG'))
    assert isinstance(t, Image.Image) and t.mode == 'RGB' and t.size == (224, 224), (t.mode, t.size)
    want_t = np.asarray(src.resize((336, 224), Image.BICUBIC).crop((56, 0, 280, 224)))
    assert np.array_equal(np.asarray(t), want_t)
    ta = np.asarray(t).astype(np.int64)
    for x, c in ((0, 0), (112, 1), (223, 2)):
        px = ta[112, x]
        assert px[c] >= 250 and all(px[j] <= 5 for j in range(3) if j != c), (x, px.tolist())
    src_p = Image.fromarray(np.ascontiguousarray(a.transpose(1, 0, 2)))  # 200 wide, 300 high: cropped from y = 56
    tp = thumb(enc(src_p, 'PNG'))
    assert tp.size == (224, 224)
    assert np.array_equal(np.asarray(tp), np.asarray(src_p.resize((224, 336), Image.BICUBIC).crop((0, 56, 224, 280))))
    for img, fmt in ((src.convert('L'), 'JPEG'), (src.convert('L'), 'PNG'), (src.convert('RGBA'), 'PNG'),
                     (src.convert('P'), 'PNG'), (src.convert('CMYK'), 'JPEG'), (src, 'JPEG')):
        t = thumb(enc(img, fmt))
        assert t.mode == 'RGB' and t.size == (224, 224), (img.mode, fmt, t.mode, t.size)
    g = np.asarray(thumb(enc(src.convert('L'), 'PNG')))
    assert np.array_equal(g[..., 0], g[..., 1]) and np.array_equal(g[..., 1], g[..., 2])
    tj = np.asarray(thumb(enc(src, 'JPEG', quality=95))).astype(np.float64)
    assert np.abs(tj - want_t).mean() < 10, np.abs(tj - want_t).mean()

    # make_input: shape, dtype, pixel values, row axis, and the same tensor as bundlemax.net_input
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cols = [(200, 30, 90), (10, 250, 128), (77, 77, 77)]
        for i, c in enumerate(cols):
            Image.new('RGB', (224, 224), c).save(td / f'{i:06d}.jpg', quality=QUALITY)
        b = np.zeros((224, 224, 3), np.uint8)
        b[:112] = (250, 20, 20)
        b[112:] = (20, 20, 250)
        Image.fromarray(b).save(td / f'{3:06d}.jpg', quality=QUALITY)  # top half red, bottom half blue
        f = make_input(td)
        ids = np.array([2, 0, 3])
        x = f(ids)
        assert isinstance(x, torch.Tensor) and tuple(x.shape) == (3, 3, 224, 224) and x.dtype == torch.float32
        dec = {}
        for i in (0, 2, 3):
            with Image.open(td / f'{i:06d}.jpg') as im:
                dec[i] = np.asarray(im.convert('RGB')).astype(np.int64)
        for k, i in enumerate(ids.tolist()):  # one pixel, each channel: its decoded value - 117
            for c in range(3):
                assert abs(float(x[k, c, 100, 50]) - (dec[i][100, 50, c] - 117)) < 1e-3, (i, c)
        for i in (0, 2):
            for c in range(3):
                assert abs(int(dec[i][100, 50, c]) - cols[i][c]) <= 4, (i, c, dec[i][100, 50].tolist())
        assert abs(float(x[2, 0, 10, 200]) - (250 - 117)) <= 6 and abs(float(x[2, 2, 10, 200]) - (20 - 117)) <= 6
        assert abs(float(x[2, 0, 200, 10]) - (20 - 117)) <= 6 and abs(float(x[2, 2, 200, 10]) - (250 - 117)) <= 6

        def tphoto(i):
            with Image.open(td / f'{int(i):06d}.jpg') as im:
                return im.convert('RGB')
        orig = bundlemax.photo
        bundlemax.photo = tphoto  # net_input reading these thumbnails instead of bank/img/thumbs
        try:
            ref = bundlemax.net_input(ids)
        finally:
            bundlemax.photo = orig
        assert torch.equal(x, ref)
        Image.new('RGB', (100, 100)).save(td / f'{4:06d}.jpg')
        try:
            f([4])
            rejected = False
        except AssertionError:
            rejected = True
        assert rejected, 'a thumbnail not 224 x 224 accepted'

    # the two passes and the pool on small synthetic shards (several row groups each; one file not train-*)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sd = td / 'shards'
        sd.mkdir()
        st = pa.struct([('bytes', pa.binary()), ('path', pa.string())])
        layout = {'train-00001-of-00002.parquet': ([5, 3, 3, 1, 5], 2),
                  'train-00000-of-00002.parquet': ([3, 5, 3, 9, 3, 5, 3], 3),
                  'validation-00000-of-00001.parquet': ([3, 5], 2)}
        raw, n = {}, 0
        for name, (labs, rg) in layout.items():
            ims = []
            for r in range(len(labs)):
                img = Image.new('RGB', (60 + 7 * n, 40 + 3 * n), (20 + 15 * n, 220 - 13 * n, 40 + 9 * n))
                raw[row_key(name, r)] = enc(img, 'PNG' if n % 2 else 'JPEG')
                ims.append({'bytes': raw[row_key(name, r)], 'path': f'{n}.jpeg'})
                n += 1
            pq.write_table(pa.table({'image': pa.array(ims, type=st), 'label': pa.array(labs, type=pa.int64())}),
                           str(sd / name), row_group_size=rg)
            with open(sd / name, 'rb') as fh:
                assert pq.ParquetFile(fh).num_row_groups == -(-len(labs) // rg), name
        paths = shard_paths(sd)
        assert [p.name for p in paths] == ['train-00000-of-00002.parquet', 'train-00001-of-00002.parquet']
        A, B = (p.name for p in paths)
        found = collect(paths, [5, 3, 7])
        assert found == {5: [f'{A}:1', f'{A}:5', f'{B}:0', f'{B}:4'],
                         3: [f'{A}:0', f'{A}:2', f'{A}:4', f'{A}:6', f'{B}:1', f'{B}:2'], 7: []}, found
        assert list(found) == [5, 3, 7]
        got = list(read_images(paths, [f'{B}:4', f'{A}:6', f'{A}:3', f'{B}:0']))
        assert [(k, lab) for k, _, lab in got] == [(f'{A}:3', 9), (f'{A}:6', 3), (f'{B}:0', 5), (f'{B}:4', 5)], got
        assert all(bb == raw[k] for k, bb, _ in got)
        out = td / 'pool'
        names, labels = build_pool(paths, [5, 3, 7], 3, out)
        ch = choose(found, 3)
        assert names == ch[5] + ch[3] + ch[7] and len(names) == 6, names
        assert labels.dtype == np.int64 and labels.tolist() == [5, 5, 5, 3, 3, 3]
        assert json.loads((out / 'names.json').read_text()) == names
        lab_f = np.load(out / 'labels.npy')
        assert lab_f.dtype == np.int64 and lab_f.tolist() == labels.tolist()
        assert sorted(p.name for p in (out / 'thumbs').iterdir()) == [f'{i:06d}.jpg' for i in range(6)]
        for i, k in enumerate(names):  # each pool id holds its key's photo: thumb() saved as JPEG quality 88
            with Image.open(out / 'thumbs' / f'{i:06d}.jpg') as im:
                assert im.format == 'JPEG' and im.mode == 'RGB' and im.size == (224, 224), (i, im.format, im.mode)
                got_px = np.asarray(im.convert('RGB'))
            with Image.open(io.BytesIO(enc(thumb(raw[k]), 'JPEG', quality=QUALITY))) as im:
                want_px = np.asarray(im.convert('RGB'))
            assert np.array_equal(got_px, want_px), (i, k)
        assert tuple(make_input(out / 'thumbs')(np.arange(6)).shape) == (6, 3, 224, 224)
    print('selftest ok')


# ---------- main ----------

def main():
    if '--selftest' in sys.argv[1:]:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', required=True, help='directory of train-*.parquet')
    ap.add_argument('--classes', required=True, help='JSON file: a list of class indices')
    ap.add_argument('--per-class', type=int, default=PER_CLASS)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', default=','.join(LAYERS))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    layers = [s.strip() for s in a.layers.split(',') if s.strip()]
    assert layers and len(set(layers)) == len(layers) and all(L in proploop.NCH for L in layers), layers
    assert a.per_class > 0, a.per_class
    classes = load_classes(a.classes)
    paths = shard_paths(a.shards)
    out = Path(a.out)
    names, _ = build_pool(paths, classes, a.per_class, out)
    write_values(out, len(names), layers, a.device)
    print('pool ->', out, flush=True)


if __name__ == '__main__':
    main()
