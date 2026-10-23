# Spec: tools/r31_photos.py (a larger photo pool from ImageNet train, with the network's values)

Write one new file, `tools/r31_photos.py`. Read only this spec and `tools/proploop.py`, `tools/bundlemax.py` and
`bank/img_acts.py`. Do not open any other file, and do not run anything. It will run on a GPU machine where the
ImageNet train mirror is a directory of Parquet shards.

## Purpose
The repo's photo pool has 10 photos per class (`bank/img/thumbs`, values in `proploop.OUT/values_<layer>.npy`). Build
a second pool with more photos per class for chosen classes, prepared the same way, with the same values.

## Inputs
- `--shards`: a directory of Parquet files `train-*.parquet` (read in sorted file-name order); each row has an
  `image` column (a struct with `bytes`, a JPEG or PNG) and a `label` column (int 0-999, the standard ImageNet class
  index, the index used by `proploop.ACTS` `classes`).
- `--classes`: a JSON file, a list of class indices.
- `--per-class` (default 50), `--out` (a directory), `--layers` (default
  `mixed4a,mixed4b,mixed4c,mixed4d,mixed4e,mixed5a,mixed5b`), `--device` (default `cuda`).

## Interface
- `row_key(shard_name, row)` -> `f"{shard_name}:{row}"` (shard_name the file name without directory).
- `choose(keys_by_class, per_class)`: `keys_by_class` `{class: [keys]}` -> `{class: the per_class keys with the
  lowest zlib.crc32(key.encode()), ties by key}` (all of them when fewer), each list sorted by that order.
- `thumb(image_bytes)` -> a PIL RGB image: the decoded image, converted to RGB, then `img_acts.crop224` (as
  `bank/img_acts.py`).
- `make_input(thumb_dir)` -> a function of a list of pool ids returning the network input as
  `bundlemax.net_input` does, but loading `thumb_dir/<id:06d>.jpg` (each 224 x 224 RGB).
- `main()`: one pass over the shards collects the keys of every row whose label is in `--classes` (reading only the
  `label` column); `choose` picks the photos; a second pass reads the chosen rows' images. Pool ids are 0, 1, ...
  in the order (class in `--classes` order, then the class's chosen order). Writes into `--out`:
  `thumbs/<id:06d>.jpg` (`thumb(...)` saved as JPEG, quality 88, as `bank/img_acts.py` saves thumbnails),
  `names.json` (the key of each pool id, a list), `labels.npy` (int64 class per pool id), and
  `values_<layer>.npy` for each layer: `proploop.forward_values(model, layers, ids, device, make_input=...)` over all
  pool ids, float32, pool ids x channels, the model as `bundlemax._bank().load_model(proploop.WEIGHTS)` with
  gradients off, in eval mode as that function returns it. Prints the counts per class that had fewer than
  `--per-class` rows, and the pool size.
- `selftest()` (`--selftest`): checks `row_key`, `choose`, `thumb` (on a small image made in memory, e.g. 300 x 200)
  and `make_input` (on thumbnails written to a temporary directory: shape, dtype, the value of one pixel), without
  the model or the shards; prints "selftest ok".

## Constraints
- Read Parquet with `pyarrow.parquet` (column selection, row groups or batches as needed; do not load a whole shard's
  images when only labels are needed).
- Match the repo's style: module docstring with usage, small functions, plain asserts in the selftest.
