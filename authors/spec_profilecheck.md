# Spec: tools/profilecheck.py (do readers' claims follow what an invariance profile shows?)

Write one new file, `tools/profilecheck.py`. Read only this spec and `tools/claude_call.py`, `tools/run_subjects.py`,
`tools/bundlemax.py`, `tools/battery.py`, `tools/openmethod.py`, `tools/lookviews.py` and `bank/img_acts.py`. Do not
open any other file except `authors/battery.reply.json`, and do not run anything.

## Purpose
Readers shown an invariance profile (a table of how a unit's activation survives 10 transforms, against its layer)
may add claims to their explanations. Measure whether those claims match what the profile shows, on units chosen at
random, against readings without the profile.

## Units
For each layer in order mixed3a, mixed3b, mixed4b, mixed4d, mixed4e, mixed5a: draw channels with
`rng = np.random.default_rng(22)` (one generator for all layers, in that order), `rng.integers(n_channels)` repeatedly,
skipping excluded and already drawn channels, until 3 are kept. Excluded (studied before): mixed3a 136, 108, 132, 88,
110, 180, 153, 186, 86, 117, 112, 70, 106, 113, 147, 81, 145, 71; mixed3b 379, 385, 343, 342, 388, 340, 330, 349, 406,
324, 380, 404, 303; mixed4b 237, 491, 373, 443, 431, 429, 384, 409, 418; mixed4d 409, 417, 241, 277, 352, 455, 167, 295,
174, 393; mixed4e 55, 809, 804, 703, 315, 700; mixed5a 719. Also analysed (readings already exist): the 9 units of
`runs/lookviews/readings_all.json`, their base_s1-3 and P_s1-3 readings, with their profiles from runs/lookviews.

## Activations, cards, profiles
Unit values, cards and card text as tools/lookviews.py / tools/openmethod.py, with caches under
`runs/profilecheck/<unit>/` in exactly openmethod's values.npz format, so openmethod's cache check accepts them.
Compute the values yourself on CUDA when `torch.cuda.is_available()` (model and inputs on the GPU, outputs moved to
CPU), else on CPU; same definition as openmethod (spatial max, its row and col, map size); before saving, check the
first 64 card-pool photos against openmethod's own computation on CPU (np.allclose, rtol 1e-3) and stop if they
differ. Profiles: from the cached `runs/lookviews/profile_<layer>.npz` files, as lookviews (retention_table,
unit_profile, profile_text).

## Readings
Conditions base and P exactly as lookviews.condition_contents builds them, 3 samples each, bare system prompt, parsed
and cached as lookviews (`readings.json` per unit). Up to WORKERS = 6 calls in parallel.

## Claim extraction (one call per reading, text only)
A fresh single-turn `claude_call.converse('', [[text block]])`, verbatim with {text} the explanation:

  Here is an explanation of what one unit (neuron) of an image classification network responds to.

  {text}

  For each property below, say whether the explanation claims that the unit's response depends on it ("depends"),
  claims that the response is unaffected by it ("unaffected"), or does not say ("not_said"). Count only what the
  explanation states or clearly implies about the unit's response, not descriptions of the photos.
  - colour: colour or hue
  - fine_detail: fine detail and sharp edges
  - coarse_shading: smooth shading and large-scale brightness gradients
  - shape: the spatial structure or shape of what is shown
  - arrangement: the overall arrangement of parts across the image
  - orientation: orientation in general (which way edges or objects are tilted)
  - upright: being upright rather than upside down
  - left_right: left-right direction (which way something faces or points)
  - inner_texture: surface texture and shading inside regions, beyond outlines
  - polarity: whether something is light on dark or dark on light
  Reply with only JSON with exactly these keys: {"colour": ..., "fine_detail": ..., ...}

(join wrapped lines with spaces; keep the list lines.) Valid: exactly the 10 keys, each value one of the three
strings; one retry in a fresh conversation, then the reading's claims are missing. Cache in `claims.json` per unit
(for the 9 round-21 units, in `runs/profilecheck/<unit>/claims.json` too), keyed by reading id, storing the text hash.

## Agreement
Property -> transform: colour grayscale, fine_detail low_pass, coarse_shading high_pass, shape phase_scramble,
arrangement patch_shuffle_4x4, orientation rotate_90, upright rotate_180, left_right mirror, inner_texture edge_map,
polarity luminance_polarity_inversion. With p the unit's percentile for that transform (lookviews.unit_profile):
"depends" is supported if p <= 20, contradicted if p >= 80, else unremarkable; "unaffected" is supported if p >= 80,
contradicted if p <= 20, else unremarkable. "not_said" is no claim.

## Report
`runs/profilecheck/report.txt`, printed, and `summary.json`. For each condition (base, P), over all readings with
claims, for three groups (random units, round-21 units, all): readings, claims per reading, supported / unremarkable /
contradicted per reading and as shares of claims. Also, for random units whose 10 percentiles all lie in (20, 80):
claims per reading by condition. Then per unit: its percentiles (one line) and, per reading, its claims as
property:value:verdict.

## Interface (a separate tester writes tests against these; keep the names and behaviour)
- `draw_units(nch, exclude, seed=22)`: nch an ordered dict {layer: n_channels}, exclude {layer: set}; returns
  [(layer, channel), ...] by the rule above.
- `parse_claims(text)`: the dict of 10 values from a reply, or None when invalid.
- `verdict(claim, p)`: 'supported' | 'unremarkable' | 'contradicted' for claim 'depends' or 'unaffected'; None for
  'not_said'.
- `summarize(records)`: records = list of {"group": str, "condition": str, "claims": {prop: value}, "pct": {transform:
  p}}; returns {condition: {group: {"readings": n, "claims_per_reading": x, "supported_per_reading": x,
  "unremarkable_per_reading": x, "contradicted_per_reading": x, "supported_share": x, "unremarkable_share": x,
  "contradicted_share": x}}} with group "all" included; shares are None when there are no claims.
- `PROPERTY_TRANSFORM`: the property -> transform dict above.

## Selftest
`--selftest`, no model or Claude: the unit draw (determinism, exclusions, no repeats), claim parsing (good, missing key,
extra key, bad value), verdicts at the boundaries 20 and 80, and summarize on a hand case. Print `selftest ok`.

## Usage line
    python -I tools/profilecheck.py [--selftest]
