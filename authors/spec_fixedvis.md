# Spec: tools/fixedvis.py (feature visualisation, plain and lightly biased toward the unit's natural input paths)

Write one new file, `tools/fixedvis.py`. Read only this spec, `tools/battery.py`, `tools/claude_call.py` and `bank/img_acts.py`.

## Rendering (lucent, installed in the venv)
Before importing lucent.optvis, patch kornia: `import kornia, kornia.geometry.transform as kt;
kornia.get_rotation_matrix2d = kt.get_rotation_matrix2d; kornia.warp_affine = kt.warp_affine`.
Model: `img_acts.load_model(J/'raw/weights/inception5h.pth')`. Lucent layer names are the module names, e.g.
`mixed4d`, `mixed4e`. Use `render.render_vis(model, obj, param_f=lambda: param.image(128, fft=True,
decorrelate=True), transforms=transform.standard_transforms, thresholds=(256,), show_image=False,
progress=False)`, with `torch.manual_seed(seed)` before each render. Save each image as PNG.

## Objectives, for a unit `layer:ch`; input layer = the previous module (mixed4d for mixed4e, mixed4c for mixed4d)
- plain: maximise the mean of channel ch of `layer`.
- fixed: maximise mean(channel ch of `layer`) + lam * cos(m(T), m_top), where m(T) is the spatial mean of the input
  layer's activation over the rendered image (a vector over its channels) and m_top is the mean of the same vector
  over the unit's 50 strongest photos (all of `bank/img/thumbs`, ranked by the unit's spatial max; network input as in
  tools/battery.py). lam = 0.1 x the unit's mean spatial-mean activation over those 50 photos, so the extra term is
  small next to the main one. Use lucent's `objectives.wrap_objective` so the objective sees the hooked layers.

## Runs
`--units` (comma-separated `layer:ch`), seeds 0 and 1, plain and fixed: one PNG each in
`runs/fixedvis/{layer}_{ch}_{plain|fixed}_s{seed}.png`, plus `sheet.jpg` (one row per unit: plain s0, plain s1,
fixed s0, fixed s1, 160 px, labelled). Also record the unit's activation on each render (as for photos) in
`runs/fixedvis/acts.json`.

## Blind description
Then, for each PNG separately and in random order (seed 0), a fresh single-turn `claude_call.converse('', [image
block, text block])` (read `tools/claude_call.py` for converse; image block {"type": "image", "source": {"type":
"base64", "media_type": "image/png", "data": ...}}) with this text, verbatim:

  This is a synthetic image produced by optimisation. In one sentence, say what objects, animals, parts or shapes it
  most resembles, and from what viewpoint, if any. Reply with only JSON: {"description": "..."}

Save `runs/fixedvis/descriptions.json` (file name: description) and print a table: unit, condition, seed,
activation, description.

## Selftest
`--selftest`, model-free: the lam scaling and the cos term on synthetic tensors, file naming. Print `selftest ok`.

## Usage
    python -I tools/fixedvis.py --units mixed4d:409,mixed4e:55,mixed4e:700 [--selftest]
