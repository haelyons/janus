# Spec: tools/battery.py (transformation battery over every channel of one layer)

Write one new file, `tools/battery.py`. Read only these files: this spec, `authors/battery.reply.json`, and
`bank/img_acts.py` (for `load_model`, `Stop` and the input convention). Do not open any other file in the repo.

## What it measures
For every channel c of InceptionV1 layer `mixed4e` (832 channels; the module output is post-ReLU, shape
(N, 832, 14, 14)), and every transform T, the retention

    R[c, T] = median over c's top-20 photos x of  a_c(T(x)) / a_c(x)

where a_c(x) is the spatial max over the 14x14 map of channel c on photo x. Where a_c(x) <= 0 (dead channel),
the ratio is NaN; use nanmedian.

## Data and model
- Repo root J = parent of `tools/`. Weights: `J/raw/weights/inception5h.pth`, loaded with
  `img_acts.load_model` (add `J/bank` to sys.path; scripts are run with `python -I`).
- Photos: `names = json.load(J/bank/img/names.json)`; photo i is `J/bank/img/thumbs/{i:05d}.jpg`, already 224x224.
  Open with `.convert('RGB')`.
- Pool: P = sorted indices i with `zlib.crc32(names[i].encode()) % 10 < 7`. Use only P.
- Input to the network, as in `img_acts.run_batch`: float32 pixel values (0-255) minus 117, NCHW.
- Activations: forward hook on `model.mixed4e` storing its output; another hook on `model.maxpool10` that raises
  `Stop` so the rest of the network is skipped. torch.no_grad(), batches of 64, `torch.set_num_threads(4)`.

## Steps
1. a(x) for every photo in P (len(P) x 832 float32). Cache to `J/runs/battery/orig.npy` with the pool ids in
   `J/runs/battery/pool.npy`; reuse the cache if it exists.
2. top-20 photos in P for each channel by a_c(x) (`topk.npy`, 832 x 20 photo ids). U = union of all top-20 sets.
3. For each transform in order, a(T(x)) for every x in U. Cache each to `J/runs/battery/t_{name}.npy` (len(U) x 832)
   with U in `J/runs/battery/U.npy`; skip a transform whose cache exists. Print progress per transform.
4. R (832 x n_transforms) to `J/runs/battery/retention.npy`; transform names in order to `transforms.json`.

## Transforms
A module-level ordered dict `TRANSFORMS = {name: fn}`, fn: PIL RGB 224x224 -> PIL RGB 224x224, in this order:
the 8 in `authors/battery.reply.json` "transforms", implemented exactly as each "how" says, then the 2 in
"added_not_blind" (rotate_180, mirror). Other code may import TRANSFORMS, so keep it importable without side effects.

## Report
`--report c1,c2,...` (after the run, or from caches): for each listed channel print one line per transform:
name, R[c,T] (2 decimals), its percentile among the non-NaN channels (100 x fraction of channels with retention
<= this one, 0 decimals), and the layer median of R[:,T]. Also write `J/runs/battery/sheet_{c}.jpg`: one row per
photo for the channel's top 6, columns = original then each transform, tiles 112 px, each labelled under it with
a_c(T x)/a_c(x) to 2 decimals (1.00 for the original), and the transform names as a header row.

## Selftest
`--selftest`, model-free and fast, asserts for each transform on a fixed random image (seeded): output is
224x224 RGB, same output on two calls. Plus: grayscale has R == G == B; rotate_90 applied 4 times, rotate_180
twice and mirror twice give the input back; patch_shuffle_4x4 keeps the sorted pixel values and moves every
tile; edge_map is black on 10% (+/- 1%) of pixels and white elsewhere; low_pass(x) + high_pass(x) equals
x + mean within 1 per pixel on an image whose values stay in 64-191 (so nothing clips);
luminance_polarity_inversion twice gives the input back within 1 on such an image; phase_scramble keeps
each channel's Fourier amplitude within 2% on such an image (compare before rounding if needed, or use a
tolerance that covers rounding). Print `selftest ok`.

## Usage line (docstring)
    python -I tools/battery.py [--selftest] [--report 1,2,3]
