# Spec: tools/bundlemax.py (input bundles of a unit; blind maximum per bundle; are the bundles related?)

Write one new file, `tools/bundlemax.py`. Read only this spec, `tools/battery.py`, `tools/claude_call.py`,
`tools/run_subjects.py` (for TASK and img_block only) and `bank/img_acts.py`.

## Purpose
A method for telling a unit that has grown related offshoots from one with unrelated things packed into it: split the
unit's input into bundles of upstream channels, explain the unit with only one bundle left on, ask a second agent who
sees only that explanation to pick inputs that should drive it most, and compare against the strongest real input under
the same ablation. Then ask whether the bundle explanations are related. Nothing in the code may favour either answer.

## Units (`--units`, comma-separated)
- `mixedNx:c`: channel c of that module's post-ReLU output.
- `sum:mixedNx:a+b`: planted control, map = map_a / s_a + map_b / s_b, s = the channel's largest spatial max over
  the card pool.
Input layer = previous module: mixed3a for mixed3b, mixed4b for mixed4c, mixed4c for mixed4d, mixed4d for mixed4e.

## Photos
names = json `bank/img/names.json`; photo i = `bank/img/thumbs/{i:05d}.jpg`; card pool = `zlib.crc32(name) % 10 < 7`,
held pool = the rest. Network input as in tools/battery.py.

## Steps per unit
1. Attribution: top 150 card-pool photos by the unit's spatial max. v = the unit's map at its argmax;
   A_i = sum over positions of (dv/dx * x) for input channel i (x = input-layer post-ReLU output as a leaf).
   Unit-L2 normalise each photo's A.
2. Bundles: the K = 24 input channels with the largest mean A. Each channel's profile = its A over the 150 photos,
   centred and unit-L2 normalised. `scipy.cluster.vq.kmeans2(profiles, 3, seed=0, minit='++')`; empty clusters are
   dropped. Record each bundle's channels and its share of the summed mean A of the 24.
3. Ablation for bundle b: zero the input-layer output, at all positions, for the 24 channels not in b; everything
   else runs as normal. Also a "whole" condition with no ablation. Compute the unit's spatial max under each
   condition on: C = the top 1500 card-pool photos by unablated activation plus 500 other card-pool photos
   (`np.random.default_rng(0)`), and H = 1000 held-pool photos (`np.random.default_rng(1)`). M_b = the largest value
   on C under condition b.
4. Explanations: for each condition (each bundle and whole), a card = the 20 photos of C with the highest value under
   that condition, whole photos in a 5-column grid (224 px tiles, a white strip under each with "#n   activation k",
   k = round(10 * value / M_b)). Card text, verbatim: "# Unit {name}\n\nRecorded from one unit (a single neuron) in an
   image classification network, run over several thousand natural photos.\nThe attached grid shows the 20 photos on
   which the unit is most active, strongest first. Under each is its activation, scaled 0-10 relative to the
   strongest.\n" with {name} = "U-" + three random digits per condition (`random.Random(unit string + condition)`).
   Call `claude_call.converse('', [[text block, image block, {"type": "text", "text": run_subjects.TASK}]])`, 2
   samples (fresh conversations). Parse the reply's JSON object that has "class" and "explanation".
5. Blind maximum: pool per condition = 48 photos of H by value under that condition relative to M_b: 16 with >= 0.3,
   16 with 0.05-0.3, 16 below 0.05 (`np.random.default_rng(2)`; if a stratum is short, fill from the next lower one),
   shuffled (`default_rng(3)`), shown as an 8x6 sheet of 112 px numbered tiles. For each explanation (both samples)
   and each condition's pool, 2 calls (fresh conversations) with text blocks, verbatim:
   "A unit (one neuron) of an image classification network has been explained as follows:\n\n{explanation}\n\nThe
   attached sheet shows 48 numbered photos. You have not seen how the unit responds to any of them. Using only the
   explanation, choose the 6 photos on which the unit should fire most strongly. Reply with only JSON: {\"fire\":
   [six numbers]}". Score each call under the pool's condition: best = max value of the picks / M_b; hit = share of
   picks with value >= 0.3 M_b. "own" = an explanation of condition b scored on b's pool; "cross" = an explanation of
   another bundle of the same unit scored on b's pool. Chance hit = share of the pool >= 0.3 M_b.
   Parse the last JSON object with key "fire"; valid = 6 distinct integers 1-48; retry once.
6. Relatedness judge: per unit, sample-1 explanations of its bundles (not "whole"), listed as "Pathway 1: ...",
   3 calls (fresh conversations), text verbatim: "These are explanations of different input pathways into one unit of
   a neural network. Each was written from the inputs that drive the unit through that pathway alone.\n\n{list}\n\n
   Do they describe one underlying thing (different parts, aspects or natural offshoots of the same thing), or
   unrelated things that happen to share the unit? Reply with only JSON: {\"related\": n, \"shared\": \"...\"},
   where n is 1 (clearly unrelated) to 5 (clearly one thing) and shared names what they have in common, or is empty."

## Output
`runs/bundlemax/{unit}/` with bundles.json, values.npz, cards (jpg), explanations.json, picks.json, judge.json, and
`runs/bundlemax/report.txt` (per unit: bundles with channels and shares; per condition: the explanations (first 200
characters), own best and hit (mean over calls), cross best and hit, chance hit; judge scores and "shared" texts).
Cache every Claude reply; skip cached ones. Process units one at a time and free memory between them.

## Selftest
`--selftest`, model-free and Claude-free: unit parsing, ablation masks, strata fill, reply parsing, own/cross
bookkeeping on synthetic data. Print `selftest ok`.

## Usage
    python -I tools/bundlemax.py --units mixed4c:447,mixed4d:409 [--selftest]
