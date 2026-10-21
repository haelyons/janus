# Spec: tools/fixedvis2.py (stronger pulls toward natural paths, and rendering "through" an input)

Write one new file, `tools/fixedvis2.py`. Read only this spec, `tools/fixedvis.py` (reuse its functions by import:
model loading, top photos, lam, description, sheet), `tools/battery.py`, `tools/claude_call.py`, `bank/img_acts.py`,
and lucent's installed source if needed.

## Conditions, per unit `layer:ch` (input layer = previous module, as in tools/fixedvis.py)
- `pull5`: tools/fixedvis.py's "fixed" objective with lam multiplied by 50 (5 x the unit's mean spatial-mean
  activation over its top 50 photos).
- `init`: the plain objective, starting from the unit's strongest photo: parameterise the image directly as pixels
  (no FFT, no decorrelation), initialised from that photo resized to 128 px, with the same transforms and 256 steps.
- `through:i` (from `--through layer:ch=i,j;...`): maximise mean(unit) + w * mean(input-layer channel i), where
  w = (unit's mean spatial-mean activation over its top 50 photos) / (channel i's mean spatial-mean activation over the
  same photos), so both terms have similar size.
Seeds 0 and 1 for each condition (seed torch and numpy). Plain renders are not redone (they are in runs/fixedvis/).

## Output
`runs/fixedvis2/{layer}_{ch}_{condition}_s{seed}.png` (":" in a condition becomes "-"), acts.json (the unit's
spatial max on each render measured as in tools/fixedvis.py, and for through renders also channel i's), sheet.jpg (one
row per unit and condition, 160 px), then blind one-sentence descriptions of every new PNG exactly as in
tools/fixedvis.py (same prompt, random order with seed 0), saved to descriptions.json and printed as a table.

## Selftest
`--selftest`, model-free: the weight scaling, file naming, `--through` parsing. Print `selftest ok`.

## Usage
    python -I tools/fixedvis2.py --units mixed4d:409,mixed4e:55,mixed4e:700 \
        --through "mixed4d:409=447,393;mixed4e:55=493,352"
