# Spec: tools/bundles.py (do two kinds of input reach a unit through shared or separate input channels?)

Write one new file, `tools/bundles.py`. Read only this spec, `tools/battery.py` and `bank/img_acts.py`.

## Units (from `--units`, comma-separated)
- `mixed4e:N` or `mixed4d:N`: that channel's post-ReLU 14x14 map (module output).
- `sum:mixed4e:A+B`: a planted unit, map = map_A / s_A + map_B / s_B, where s_c is channel c's largest spatial max
  over all photos. It mixes two separate channels on purpose, as a control with a known answer.
The input layer is the previous module: mixed4d for mixed4e units, mixed4c for mixed4d units.

## Kinds (from `--kinds`, one per unit, `;`-separated, each `A=lo-hi|i,j,.../B=...`)
Photo kind from its class (`np.load(bank/img/acts.npz)['classes']`); other classes are "other".

## Steps
1. Photos: all of `bank/img/thumbs/{i:05d}.jpg` (`bank/img/names.json` gives the count). Input to the network as in
   tools/battery.py. Top 150 photos per unit by spatial max of its map.
2. Attribution per top photo: v = the unit's map at its argmax location. A_i = sum over spatial positions of
   (dv/dx * x) for input-layer channel i (x = input-layer post-ReLU output, made a leaf with requires_grad). Unit-L2
   normalise each A. Batches of 16 are fine.
3. Shared-input score: cos(mean A over kind-A photos, mean A over kind-B photos) and, as references, the split-half
   cos within A and within B (photos of that kind alternately assigned to two halves in rank order). ratio =
   cos_between / mean of the two within values. Report the counts of A, B and other among the top 150.
4. Bundles: k-means on the normalised A of all 150 (`scipy.cluster.vq.kmeans2(..., k, seed=0, minit='++')`) for
   k = 2, 3, 4. For each k: the contingency table cluster x kind, normalised mutual information (arithmetic-mean
   normalisation) between cluster and kind over the A and B photos only, and for each cluster its 6 input channels
   with the largest mean attribution.
5. Write `runs/bundles/report.txt` and print it; save attributions to `runs/bundles/{unit}.npz` (ids, A, kinds).

## Selftest
`--selftest`, model-free: kinds parsing, the sum-unit normalisation on synthetic maps, split-half assignment, cos and
ratio on synthetic vectors with known answers (two disjoint supports give ratio near 0; one support gives near 1),
NMI on known tables (perfect, independent). Print `selftest ok`.

## Usage
    python -I tools/bundles.py --units mixed4d:409,sum:mixed4e:809+804,mixed4e:55 \
        --kinds "A=151-268/B=407,436,...;A=281-285/B=...;A=281-285/B=..." [--selftest]
