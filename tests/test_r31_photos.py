"""Black-box tests for tools/r31_photos.py.

Written from the Inputs and Interface sections of authors/spec_r31_photos.md only. Expected values come from the
definitions the spec names, never from the module's own helpers: zlib.crc32 (then the key) for choose,
bank/img_acts.py crop224 for thumb, the transform of tools/bundlemax.py net_input (pixels as float32 / 255, NCHW,
* 255 - 117) for make_input, and the spatial maximum per channel (proploop.forward_values) for main's values.

No model weights, GPU, network or real shards. main runs on two small Parquet shards built here (skipped without
pyarrow), with img_acts.load_model (what bundlemax._bank().load_model is) swapped for a small stand-in network whose
modules are named mixed4a and mixed4b, and torch.load made to fail so the real weights can never be read.
"""
import contextlib
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
import zlib
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import bundlemax  # noqa: E402
import proploop  # noqa: E402
import r31_photos  # noqa: E402

img_acts = bundlemax._bank()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pattern(w, h, seed=0):
    """an RGB uint8 array h x w x 3 that changes along both axes, so a shifted crop or a wrong scale moves pixels"""
    y, x = np.mgrid[0:h, 0:w]
    r = (x * 3 + seed * 37) % 256
    g = (y * 5 + seed * 11) % 256
    b = ((x + 2 * y) * 2 + seed * 53) % 256
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _alpha(w, h):
    y, x = np.mgrid[0:h, 0:w]
    return ((x + 3 * y) % 256).astype(np.uint8)


def _encode(img, fmt, **kw):
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def _image_bytes(kind, size, seed):
    """encoded image bytes: kind in jpeg, png, rgba (PNG with alpha), gray (PNG, mode L), p (PNG, palette),
    cmyk (JPEG, mode CMYK)"""
    w, h = size
    arr = _pattern(w, h, seed)
    img = Image.fromarray(arr)
    if kind == "jpeg":
        return _encode(img, "JPEG", quality=92)
    if kind == "png":
        return _encode(img, "PNG")
    if kind == "rgba":
        rgba = Image.fromarray(np.dstack([arr, _alpha(w, h)]))
        assert rgba.mode == "RGBA"
        return _encode(rgba, "PNG")
    if kind == "gray":
        return _encode(img.convert("L"), "PNG")
    if kind == "p":
        return _encode(img.convert("P"), "PNG")
    if kind == "cmyk":
        return _encode(img.convert("CMYK"), "JPEG", quality=92)
    raise ValueError(kind)


def _expected_thumb(b):
    """the decoded image, converted to RGB, then img_acts.crop224"""
    with Image.open(io.BytesIO(b)) as im:
        rgb = im.convert("RGB")
    return img_acts.crop224(rgb)


def _jpeg88(img):
    """pixels of img after saving as JPEG at quality 88 (as img_acts saves thumbnails) and decoding"""
    data = _encode(img, "JPEG", quality=88)
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


def _expected_input(thumb_dir, ids):
    """bundlemax.net_input's transform on thumb_dir/<id:06d>.jpg: float32 pixels / 255, NCHW, * 255 - 117"""
    arrs = []
    for i in ids:
        with Image.open(Path(thumb_dir) / f"{int(i):06d}.jpg") as im:
            arrs.append(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0)
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
    return x * 255 - 117


def _np(x):
    return x.detach().cpu().numpy()


def _order(keys):
    """keys sorted by zlib.crc32(key.encode()), ties by key"""
    return sorted(keys, key=lambda k: (zlib.crc32(k.encode()), k))


def _keys(shard, rows):
    return [f"{shard}:{r}" for r in rows]


@lru_cache(maxsize=None)
def _crc_tie():
    """two different keys with the same zlib.crc32, found by search (a birthday collision, about 1e5 tries)"""
    seen = {}
    for i in range(4_000_000):
        k = "train-" + hashlib.sha256(str(i).encode()).hexdigest()[:16] + ".parquet:0"
        c = zlib.crc32(k.encode())
        if c in seen and seen[c] != k:
            return tuple(sorted((seen[c], k)))
        seen[c] = k
    raise AssertionError("no crc32 collision found")


# ---------------------------------------------------------------------------
# row_key
# ---------------------------------------------------------------------------

def test_row_key_format():
    assert r31_photos.row_key("train-00003-of-00294.parquet", 17) == "train-00003-of-00294.parquet:17"


def test_row_key_row_zero():
    assert r31_photos.row_key("train-00000-of-00001.parquet", 0) == "train-00000-of-00001.parquet:0"


def test_row_key_numpy_row_and_str_type():
    k = r31_photos.row_key("train-00041-of-00294.parquet", np.int64(1043))
    assert isinstance(k, str)
    assert k == "train-00041-of-00294.parquet:1043"


# ---------------------------------------------------------------------------
# choose
# ---------------------------------------------------------------------------

SHARD_X = "train-00000-of-00004.parquet"
SHARD_Y = "train-00001-of-00004.parquet"


def test_choose_lowest_crc_per_class_in_crc_order():
    ka = _keys(SHARD_X, range(40))
    kb = _keys(SHARD_Y, range(5, 25))
    exp_a, exp_b = _order(ka)[:5], _order(kb)[:5]
    got = r31_photos.choose({3: list(reversed(_order(ka))), 917: list(kb)}, 5)
    assert set(got) == {3, 917}
    assert list(got[3]) == exp_a
    assert list(got[917]) == exp_b


def test_choose_order_does_not_depend_on_input_order():
    keys = _keys(SHARD_X, range(30)) + _keys(SHARD_Y, range(30))
    expected = _order(keys)[:7]
    shuffled = list(keys)
    random.Random(0).shuffle(shuffled)
    for given in (list(keys), list(reversed(keys)), shuffled, sorted(keys)):
        assert list(r31_photos.choose({12: given}, 7)[12]) == expected


def test_choose_fewer_than_per_class_gives_all_in_crc_order():
    keys = _keys(SHARD_Y, [4, 9, 2])
    got = r31_photos.choose({600: list(reversed(keys))}, 50)
    assert list(got[600]) == _order(keys)
    assert len(got[600]) == 3


def test_choose_exactly_per_class_gives_all_in_crc_order():
    keys = _keys(SHARD_X, range(6))
    assert list(r31_photos.choose({1: list(keys)}, 6)[1]) == _order(keys)


def test_choose_empty_class():
    got = r31_photos.choose({4: [], 5: _keys(SHARD_X, [0, 1])}, 3)
    assert set(got) == {4, 5}
    assert list(got[4]) == []
    assert list(got[5]) == _order(_keys(SHARD_X, [0, 1]))


def test_choose_classes_are_independent():
    ka = _keys(SHARD_X, range(20))
    kb = _keys(SHARD_Y, range(20))
    both = r31_photos.choose({0: list(ka), 999: list(kb)}, 4)
    alone_a = r31_photos.choose({0: list(ka)}, 4)
    alone_b = r31_photos.choose({999: list(kb)}, 4)
    assert list(both[0]) == list(alone_a[0]) == _order(ka)[:4]
    assert list(both[999]) == list(alone_b[999]) == _order(kb)[:4]


def test_choose_ties_by_key():
    a, b = _crc_tie()
    assert a < b and zlib.crc32(a.encode()) == zlib.crc32(b.encode())
    assert list(r31_photos.choose({5: [b, a]}, 2)[5]) == [a, b]
    assert list(r31_photos.choose({5: [a, b]}, 2)[5]) == [a, b]
    assert list(r31_photos.choose({5: [b, a]}, 1)[5]) == [a]
    assert list(r31_photos.choose({5: [a, b]}, 1)[5]) == [a]


def test_choose_tie_at_the_cut_keeps_the_smaller_key():
    a, b = _crc_tie()
    tie = zlib.crc32(a.encode())
    extras = _keys("train-00002-of-00004.parquet", range(60))
    below = [k for k in extras if zlib.crc32(k.encode()) < tie]
    pool = [b] + extras + [a]
    n = len(below) + 1
    got = list(r31_photos.choose({8: list(pool)}, n)[8])
    assert got == _order(pool)[:n]
    assert a in got and b not in got
    assert list(r31_photos.choose({8: list(pool)}, len(pool))[8]) == _order(pool)


# ---------------------------------------------------------------------------
# thumb
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", [(300, 200), (256, 256), (200, 300), (100, 80)])
@pytest.mark.parametrize("kind", ["jpeg", "png"])
def test_thumb_is_crop224_of_the_rgb_image(size, kind):
    b = _image_bytes(kind, size, seed=size[0] + size[1])
    got = r31_photos.thumb(b)
    assert isinstance(got, Image.Image)
    assert got.mode == "RGB"
    assert got.size == (224, 224)
    assert np.array_equal(np.asarray(got), np.asarray(_expected_thumb(b)))


@pytest.mark.parametrize("size", [(300, 200), (256, 256)])
@pytest.mark.parametrize("kind", ["rgba", "gray", "p", "cmyk"])
def test_thumb_converts_other_modes_to_rgb(kind, size):
    b = _image_bytes(kind, size, seed=7)
    got = r31_photos.thumb(b)
    assert isinstance(got, Image.Image)
    assert got.mode == "RGB"
    assert got.size == (224, 224)
    assert np.array_equal(np.asarray(got), np.asarray(_expected_thumb(b)))


def test_thumb_grayscale_has_equal_channels():
    b = _image_bytes("gray", (256, 256), seed=9)
    got = np.asarray(r31_photos.thumb(b))
    assert got.shape == (224, 224, 3)
    assert np.array_equal(got[..., 0], got[..., 1])
    assert np.array_equal(got[..., 1], got[..., 2])


# ---------------------------------------------------------------------------
# make_input
# ---------------------------------------------------------------------------

MI_IDS = [0, 1, 7, 1234, 98765]
DECOY = (255, 0, 255)


@pytest.fixture
def thumb_dir(tmp_path):
    d = tmp_path / "thumbs"
    d.mkdir()
    for n, i in enumerate(MI_IDS):
        Image.fromarray(_pattern(224, 224, seed=3 * n + 5)).save(d / f"{i:06d}.jpg", quality=88)
        # the 5-digit name img_acts uses, with other pixels: reading it instead gives wrong values
        Image.new("RGB", (224, 224), DECOY).save(d / f"{i:05d}.jpg", quality=88)
    return d


def test_make_input_shape_and_dtype(thumb_dir):
    x = r31_photos.make_input(thumb_dir)([7, 0, 1234])
    assert isinstance(x, torch.Tensor)
    assert tuple(x.shape) == (3, 3, 224, 224)
    assert x.dtype == torch.float32


def test_make_input_values_equal_net_input_transform(thumb_dir):
    ids = [7, 0, 1234]
    x = r31_photos.make_input(thumb_dir)(ids)
    assert np.allclose(_np(x), _np(_expected_input(thumb_dir, ids)), rtol=0, atol=1e-4)


def test_make_input_one_pixel(thumb_dir):
    x = _np(r31_photos.make_input(thumb_dir)([7, 1234]))
    with Image.open(thumb_dir / "001234.jpg") as im:
        px = np.asarray(im.convert("RGB"))[10, 200].astype(np.float32)
    assert np.allclose(x[1, :, 10, 200], px - 117.0, rtol=0, atol=1e-4)
    with Image.open(thumb_dir / "000007.jpg") as im:
        px = np.asarray(im.convert("RGB"))[223, 0].astype(np.float32)
    assert np.allclose(x[0, :, 223, 0], px - 117.0, rtol=0, atol=1e-4)


def test_make_input_order_and_repeats_follow_ids(thumb_dir):
    ids = [98765, 1, 1, 0, 7]
    x = _np(r31_photos.make_input(thumb_dir)(ids))
    assert x.shape == (5, 3, 224, 224)
    for k, i in enumerate(ids):
        assert np.allclose(x[k], _np(_expected_input(thumb_dir, [i]))[0], rtol=0, atol=1e-4), (k, i)
    assert np.array_equal(x[1], x[2])
    assert not np.allclose(x[0], x[3], atol=1.0)


def test_make_input_reads_six_digit_names(thumb_dir):
    x = _np(r31_photos.make_input(thumb_dir)([7]))
    decoy = np.array(DECOY, dtype=np.float32) - 117.0
    assert not np.allclose(x[0, :, 112, 112], decoy, atol=3.0)
    assert np.allclose(x, _np(_expected_input(thumb_dir, [7])), rtol=0, atol=1e-4)


def test_make_input_numpy_ids_like_list(thumb_dir):
    f = r31_photos.make_input(thumb_dir)
    from_list = _np(f([0, 7, 1234]))
    from_array = _np(f(np.array([0, 7, 1234], dtype=np.int64)))
    from_slice = _np(f(np.array(MI_IDS, dtype=np.int64)[[0, 2, 3]]))
    assert np.allclose(from_array, from_list, rtol=0, atol=1e-6)
    assert np.allclose(from_slice, from_list, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# main on two small shards, with a stand-in network
#
# Shard rows (label, image kind, size). --classes [913, 207] (not sorted), --per-class 3:
#   class 913: rows A:2 and B:2 -> 2 rows, fewer than 3, both kept;
#   class 207: rows A:0, A:3, A:4, A:6, B:1, B:4 -> the 3 with the lowest crc32 of the key.
# Pool ids 0-1 are class 913, 2-4 class 207, each class in its crc32 order. Labels 530 and 1 are never chosen.
# Keys use the row's 0-based index within its shard file.
# ---------------------------------------------------------------------------

SHARD_A = "train-00000-of-00002.parquet"
SHARD_B = "train-00001-of-00002.parquet"
SHARDS = [
    (SHARD_B, [(530, "jpeg", (300, 200)), (207, "rgba", (256, 256)), (913, "jpeg", (260, 240)),
               (1, "png", (256, 256)), (207, "png", (100, 80))]),
    (SHARD_A, [(207, "jpeg", (300, 200)), (530, "png", (256, 256)), (913, "rgba", (200, 300)),
               (207, "gray", (256, 256)), (207, "png", (300, 200)), (530, "jpeg", (240, 260)),
               (207, "jpeg", (320, 240))]),
]
MAIN_CLASSES = [913, 207]
MAIN_PER_CLASS = 3
MAIN_LAYERS = ["mixed4a", "mixed4b"]


def _expected_pool():
    by = {c: [] for c in MAIN_CLASSES}
    for name, rows in sorted(SHARDS):
        for r, (label, _kind, _size) in enumerate(rows):
            if label in by:
                by[label].append(f"{name}:{r}")
    names, labels = [], []
    for c in MAIN_CLASSES:
        for k in _order(by[c])[:MAIN_PER_CLASS]:
            names.append(k)
            labels.append(c)
    return names, labels


def _fake_model():
    g = torch.Generator().manual_seed(31)
    n4a, n4b = proploop.NCH["mixed4a"], proploop.NCH["mixed4b"]
    m = torch.nn.Sequential(OrderedDict([
        ("mixed4a", torch.nn.Conv2d(3, n4a, kernel_size=16, stride=16)),
        ("mixed4b", torch.nn.Conv2d(n4a, n4b, kernel_size=1)),
    ]))
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.01)
            p.requires_grad_(False)
    return m.eval()


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    base = tmp_path_factory.mktemp("r31")
    shards = base / "shards"
    shards.mkdir()
    blobs = {}
    image_type = pa.struct([("bytes", pa.binary())])
    for name, rows in SHARDS:
        si = 0 if name == SHARD_A else 1
        imgs, labels = [], []
        for r, (label, kind, size) in enumerate(rows):
            b = _image_bytes(kind, size, seed=10 * si + r + 1)
            blobs[f"{name}:{r}"] = b
            imgs.append({"bytes": b})
            labels.append(label)
        table = pa.table({"image": pa.array(imgs, type=image_type), "label": pa.array(labels, type=pa.int64())})
        pq.write_table(table, shards / name)
    classes_path = base / "classes.json"
    classes_path.write_text(json.dumps(MAIN_CLASSES))
    out = base / "out"
    out.mkdir()

    model = _fake_model()
    calls = []

    def fake_load_model(weights):
        calls.append(weights)
        return model

    def no_weights(*_a, **_k):
        raise AssertionError("the test must not load real weights")

    argv = ["r31_photos.py", "--shards", str(shards), "--classes", str(classes_path),
            "--per-class", str(MAIN_PER_CLASS), "--out", str(out), "--layers", ",".join(MAIN_LAYERS),
            "--device", "cpu"]
    buf = io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(img_acts, "load_model", fake_load_model)
        mp.setattr(torch, "load", no_weights)
        mp.setattr(sys, "argv", argv)
        with contextlib.redirect_stdout(buf):
            r31_photos.main()
    return {"out": out, "blobs": blobs, "model": model, "calls": calls, "stdout": buf.getvalue()}


def test_main_names_json(ran):
    names, _labels = _expected_pool()
    got = json.loads((ran["out"] / "names.json").read_text())
    assert got == names


def test_main_labels_npy(ran):
    _names, labels = _expected_pool()
    got = np.load(ran["out"] / "labels.npy")
    assert got.dtype == np.int64
    assert got.shape == (len(labels),)
    assert got.tolist() == labels


def test_main_thumb_file_names(ran):
    n = len(_expected_pool()[0])
    assert sorted(os.listdir(ran["out"] / "thumbs")) == [f"{i:06d}.jpg" for i in range(n)]


def test_main_thumbs_are_thumb_saved_as_jpeg_quality_88(ran):
    names, _labels = _expected_pool()
    for i, key in enumerate(names):
        with Image.open(ran["out"] / "thumbs" / f"{i:06d}.jpg") as im:
            assert im.format == "JPEG"
            assert im.size == (224, 224)
            got = np.asarray(im.convert("RGB"))
        assert np.array_equal(got, _jpeg88(_expected_thumb(ran["blobs"][key]))), (i, key)


def test_main_values_per_layer(ran):
    n = len(_expected_pool()[0])
    x = _expected_input(ran["out"] / "thumbs", list(range(n)))
    m = ran["model"]
    with torch.no_grad():
        a = m.mixed4a(x)
        b = m.mixed4b(a)
    expected = {"mixed4a": _np(a.amax(dim=(2, 3))), "mixed4b": _np(b.amax(dim=(2, 3)))}
    for layer in MAIN_LAYERS:
        got = np.load(ran["out"] / f"values_{layer}.npy")
        assert got.dtype == np.float32, layer
        assert got.shape == (n, proploop.NCH[layer]), layer
        assert np.allclose(got, expected[layer], rtol=1e-4, atol=1e-3), layer


def test_main_loads_model_from_proploop_weights(ran):
    assert ran["calls"], "img_acts.load_model was not called"
    assert all(Path(w) == Path(proploop.WEIGHTS) for w in ran["calls"])


def test_main_prints_short_class_and_pool_size(ran):
    text = ran["stdout"]
    assert "913" in text
    assert str(len(_expected_pool()[0])) in re.findall(r"\d+", text)


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def test_selftest_function_prints_ok(capsys):
    r31_photos.selftest()
    assert "selftest ok" in capsys.readouterr().out


def test_selftest_command_line():
    p = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "r31_photos.py"), "--selftest"],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "selftest ok" in p.stdout
