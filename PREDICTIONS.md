# Sealed predictions (written 2026-10-20, before any subject run)

Model for every subject: claude-opus-5-5. Three samples per arm and card. "Modal" = the class given in at
least 2 of 3 samples.

Cards (the subjects see only the ID):
- U-892 dotnext (truth one), U-570 listend (one): structural rules over a heterogeneous surface.
- U-863 coffee (one_plus_related): coffee, cafés, and "Java" mostly as the programming language.
- U-767 cheese_cpp, U-937 bird_month (unrelated): true OR-packing controls.
- U-421 sail_lag field (underdetermined), U-317 sail_lag full (one).
- Image: InceptionV1 mixed4e:55, field (positives only) and full (+ negatives); IDs set when built.

## Per card

1. U-892 and U-570: most arms, including A0 and C0, give modal one or one_plus_related. A modern model
   sees the structural position. C1 is the arm most likely to say unrelated, and I expect it to do so in at
   most 1 of the 2 cards.
2. U-863: A0, C0, C2, C3 and C4 give one_plus_related. C1 gives unrelated (coffee + Java programming) in
   at least 2 of 3 samples.
3. U-767 and U-937: A0, C0, C1 and C3 give unrelated, 3 of 3. C2 and C4 are the risk arms. I predict
   at least one sample across those two arms and two cards rationalises a link (not unrelated). C4
   reaches modal unrelated on both anyway.
4. U-421 (field): few arms say cannot_tell. Most say one ("clause-final punctuation"). C4 is the most
   likely arm to say cannot_tell or to name long context in missing_data. Held-out accuracy is near
   chance (0.4-0.6) in every arm, because the deciding information is outside the window.
5. U-317 (full): at least half the arms name nautical or sailing context. Held-out accuracy is above 0.75
   for those.
6. Image field card: A0, C0 and C1 give modal unrelated (cats and cars). C2 and C3 split. C4 is
   contaminated (its author read the essay, which names this neuron's reading) and gives one or
   one_plus_related with a shape-based explanation.
7. Image full card (+ negatives): more arms move off unrelated than on the field card, because the
   negative pole is visible. At least one non-C4 arm names pointed or V shapes against round ones.
8. Image held-out: explanations naming a shape (pointed, V, triangular ears or grilles) score at least
   0.15 higher than "cats and cars" explanations. "Cats and cars" scores near 0.5 by construction.

## Overall

- No arm is perfect on both one-job and packed cards. The trade-off shows up as C2/C4 against C1.
- A0 (bare) behaves closer to C0 than to C2: an unprimed modern model is not already "Nick by default".
  I hold this one weakly.
- If C4 scores at least as well as C1 on the packed controls while beating C0 on the iconic card, it is
  the best candidate. Its iconic-card result does not count as reproduction, because of contamination.

# Round 2: arm P0_probe (sealed 2026-10-20, before any P0 run)

P0 = A0 (empty system prompt, same card, same held-out turn) plus one paragraph offering up to 3 requests for
more data: ImageNet category names for the image unit (4 card-pool photos each), or self-written texts for text
units (exact rule). No instruction on how to reason. Cards: U-192, U-300 (InceptionV1 mixed4e:55), U-421
(hidden-context sail), U-937 (packed control). Three samples each.

1. At least 4 of the 6 image sessions request a vehicle category. The convertible in the top grid invites it.
2. Image held-out accuracy beats round 1's best cell (0.70) in at least 3 of 6 sessions, and car_fire is
   above 0 in at least 3 of 6.
3. A shape-level reading (pointed, V, angular, triangular, against round) appears in at most 2 of 6 image
   sessions. Probing finds that cars fire, but the modal reading becomes "animal faces + cars", with class
   one_plus_related or unrelated.
4. U-937 stays modal unrelated (or two named triggers); probing creates no false unity.
5. U-421: no session finds the sailing context, because guessing the topic would be needed. Held-out stays
   0.4-0.7. At least 1 of 3 says cannot_tell or names context in missing_data after seeing written
   punctuation fail to fire.

# Round 3: recognition diagnostic R0 (sealed 2026-10-20, before any R0 run)

R0: empty system prompt and the same InceptionV1 card (U-192 field, U-300 full; 3 samples each). Turn 1
replaces the open question with six candidate readings in shuffled order, or "your own": furry mammal faces;
cat/fox faces plus cars (unrelated); pointed/V shapes against round; fur texture; front-facing pose;
distinctive characterful faces. Turn 2 is the same 20 held-out photos.

1. Field card: the pointed/V reading is chosen in at most 1 of 3. Modal choice: furry faces or faces+cars.
2. Full card: pointed/V is chosen in at least 1 of 3. The round negatives make it checkable.
3. Sessions that choose pointed/V predict car_fire better (above 0) than sessions that choose furry faces,
   but still miss the pool tables and throne. Overall held-out at most 0.80.
4. The vague false unity ("characterful faces") is chosen in 0 of 6.

# Round 4: failure feedback F0 (sealed 2026-10-20, before any F0 run)

F0: empty system prompt, same cards U-192 and U-300, 3 samples each. T1: answer as in round 1. T2: predict 20
feedback photos from the card pool (10 random with activation >= 30% of max, 10 random <= 10%, chosen without
regard to class, same batch in every sample of a card). T3: show the true outcome and activation for each,
plus the score; it may revise its answer. T4: the usual 20 held-out photos.
Substrate note, found before this run: the negative grid's troughs are diffuse background, not objects.

1. Feedback-batch accuracy 0.55-0.80. The misses are mostly vehicles and other objects that fire.
2. After the reveal, at least 4 of 6 revise to include cars (unrelated, or one with car offshoots). At most 1
   of 6 reaches a shape-level reading.
3. Held-out after revision: car_fire at least 0.5 in at least 4 of 6; object_fire (pool tables, throne) above
   0 in at most 2 of 6.
4. Confidence falls after the reveal in at least 4 of 6. None says cannot_tell.

# Round 5: display v2, bare arm (sealed 2026-10-20, before any v2 run)

The A0 arm (empty system prompt, round-1 task and held-out turn) on display v2: U-121 (field: top-20 cropped
around the hotspot) and U-860 (full: crops with red response shading, plus 20 negatives located by the
strongest 3x3 trough, cropped and shaded blue). The same photos and held-out set as rounds 1-4; only the
display changed. Three samples each.

1. U-121: the modal reading is still category-level (furry or cat/fox faces). A shape-level reading (pointed
   muzzle or face, V, triangular, narrow against broad) as the defining feature appears in at most 1 of 3.
2. U-860: a shape-level reading appears in at least 1 and at most 2 of 3. At least 2 of 3 describe the negatives
   as flat, round, broad or jowly faces.
3. Held-out stays within 0.65-0.80. car_fire rises above 0 only in sessions that give a shape-level reading.

# Round 6: contrast, circuit, and the no-nouns question (sealed 2026-10-20, before any round-6 run)

Cards (bank/img_cards_v3.py). U-857 (contrast): U-860's two grids, plus same-class pairs (for each of the 12
labels with the highest mean activation, its strongest photo beside its weakest) and 15 off-class positives
(>= 30% of max, label outside the top 200 by mean). U-605 (circuit): the same, plus the 8 main mixed4d inputs,
each shown by its own six strongest photos with its weight and share. Seen by eye before sealing: the weak
twins are mostly small, side-on or partial animals, and the car twins are weak, not silent (255-362 of 1527).
Arm N0_nouns: one sentence from the essay ("if you couldn't use groups of nouns at all, how would you describe
this unit?"), with the round-1 task unchanged. Runs: A0 x U-857, A0 x U-605, N0 x U-860 (compare round 5's
A0 x U-860), and N0 x U-767, U-937 (packed text controls; A0 gave U/U/U and 1+/U/1+). Three samples each.

A carrier-free shape reading (CFS): the explanation names a geometric configuration (V, pointed, tapering,
converging, wedge, triangular, angular) as what the unit responds to, and applies it beyond faces, not as a
property of faces only. "Car fronts look like faces" is face-anchored, not CFS.

1. A0 x U-857: CFS in at most 1 of 3. The modal reading is a large, frontal (face or face-like) view; the
   pairs are read as pose, size or visibility.
2. A0 x U-605: at least 2 of 3 cite the shared input (the fox/cat-face input) or the car-only input as
   evidence. CFS in at most 1 of 3. At most 1 of 3 mentions the pointed-things input or the flat-front negative.
3. N0 x U-860: a shape stated as the primary description in at least 2 of 3; CFS in at least 1 of 3.
4. Held-out 0.65-0.85 in every image session. object_fire above 0 in more N0 x U-860 sessions than round 5's
   A0 x U-860.
5. Controls: N0 x U-767 3/3 unrelated; N0 x U-937 false unity in at most 2 of 3 (A0 baseline). Held-out 1.00.

# Round 7: renders by weight (sealed 2026-10-20, before the run)

U-283 = U-860 plus one grid: lucent renders of the unit, its 8 strongest inputs by weight (6 positive, 2 negative)
and its 3 strongest readers (mixed5a 1x1 branch), labelled by role and weight only. A0, three samples. Seen by eye
before sealing: the unit and input 1 show fox/cat faces; the inputs show crickets, arcs, grille with a dog nose, round
domes and dog bodies; the negatives show shaggy round dog faces; reader 1 shows flames and jets.

1. CFS (as in round 6) in at most 1 of 3; the modal reading stays small-mammal faces.
2. At least 2 of 3 mention the flames, insects or domes, and treat them as noise or extra jobs, not as a shared shape.
3. Held-out 0.65-0.80.

# Round 8: Nick's question, the mixed grid, and a disagreement set (sealed 2026-10-20, before the run)

Substrate first (bank/shape_classes.json, lists fixed before activations were viewed; one misfiled class, paper towel,
removed before viewing): fire rate pointed objects 5%, round objects 1%, flat-fronted vehicles 15%, round-faced animals
12%, pointed-face dogs 32%, cats 54%, cars 63%; base rate 5.4%. Shape matters inside faces and vehicles; a pointed shape
outside them is near base rate (pool table 0.31 and bullet train 0.27 mean, the highest).

Q0_shared: "Spend a few seconds wondering what one thing this might be. What's the shared pattern here?" (both
phrasings from the essay). U-686: the 10 strongest cat-label and 10 strongest car-label photos, v2 crops, described as
"a selection, not simply the strongest 20". New turn 3 on U-686 and U-121: 20 disagreement photos (4 per shape group,
2 firing / 2 silent where possible), so group-level readings score 0.5 and only per-photo discrimination scores above.
Runs: A0 x U-686, Q0 x U-686, Q0 x U-121, A0 x U-121 (s4-s6, for the turn-3 baseline), Q0 x U-767 and U-937 (controls).

1. A0 x U-686: unrelated in at least 2 of 3. CFS in at most 1 of 3.
2. Q0 x U-686: one or one_plus_related in at least 2 of 3; the modal unifying reading is face-like fronts (headlights
   as eyes), not a shape; CFS in at most 1 of 3.
3. Q0 x U-121: one or one_plus_related 3/3; CFS 0/3.
4. Controls: false unity in at most 2 of 6; held-out 1.00.
5. Turn 3: every session 0.40-0.65; none above 0.70. pointed_obj above 0.5 in at most 2 sessions.
6. Turn 2 (usual held-out): 0.65-0.85.

# Round 9: experiments (E0_edit) (sealed 2026-10-20, after round 8, before any E0 run)

tools/run_edit.py: empty system prompt, same card and turns, plus up to 3 requests of up to 8 edits of grid photos
(occlude, keep, zoom, flip, gray, rotate, shrink); each edited photo is shown as the network sees it, with its
activation on the 0-10 scale. Cards U-121 and U-686, three samples each. Compare A0 and Q0 on the same cards (round 8).

1. All 6 sessions make at least one request.
2. Most edits occlude or keep face parts (eyes, nose, ears); CFS in at most 1 of 6.
3. Class unchanged in mode: U-121 one/one_plus_related, U-686 unrelated in at least 2 of 3.
4. Turn 3 within 0.50-0.75; pointed_obj exactly 0.50 in at least 4 of 6.
5. Turn 2 within 0.65-0.80.

# Round 10: blind transformation battery (sealed 2026-10-20, before battery.py exists or runs)

Separation for this round: the transform list was chosen by a tool-less agent told only "units of a mid-level layer
of an ImageNet CNN, each on its own top photos" (authors/brief_battery.md, reply authors/battery.reply.json), with no
unit, hypothesis or essay. rotate_180 and mirror were added after it by the user's request and are marked not blind.
The code is written by a claim-blind author from authors/spec_battery.md, which names no channel and no hypothesis;
I do not edit it. Measure: retention R = median over a channel's top-20 card-pool photos of a(Tx)/a(x), spatial max,
for all 832 mixed4e channels; channel 55 reported as a percentile among them.

What the battery can decide: configuration versus texture (shape-type readings, V or face layout, predict high
edge_map / low_pass retention and low patch_shuffle / phase_scramble / rotation retention relative to the layer;
the field's fur-and-chrome OR(cat, car) reading predicts the reverse). What it cannot decide: upright furry face
versus downward-pointing V, since both are configural and both predict an inversion drop.

Predictions for channel 55 (percentile bands: low < 33, mid 33-67, high > 67):
1. grayscale mid; luminance_polarity_inversion mid; mirror retention >= 0.85.
2. low_pass high; edge_map above the layer median (percentile > 50).
3. patch_shuffle_4x4 low; phase_scramble low with retention < 0.2; high_pass low or mid.
4. rotate_90 low; rotate_180 low with retention < 0.4.
5. Layer: rotate_180 median retention below 0.7 (inversion hurts most mid-level channels, so an inversion drop
   alone is weak evidence about any one unit); edge_map median retention below 0.3 (texture bias).

# Round 11: blind-authored scientist arm with the battery as edit ops (sealed 2026-10-20, before any E1 run)

arms/E1_scientist.txt was written verbatim by a tool-less author from authors/brief_rival.md (careful experimental
scientist; no hypothesis, unit or essay; 458 words against a 400 limit, kept unedited). Image runs use run_edit.py
with the E0 ops plus the 10 battery transforms as op "transform" (each listed by name and the first clause of the
blind list's "removes"), so E1 differs from E0 in both prompt and ops; if E1 moves, an ablation follows. Runs:
E1 x U-121 and U-686 with edits (3 each); E1 x U-767 and U-937 (text controls, no experiments; 3 each).

1. All 6 image sessions request edits; at least 4 of 6 use 2 or more requests.
2. At least 4 of 6 use a battery transform; edge_map in at least 3 of 6.
3. CFS in at most 1 of 6.
4. U-121 one/one_plus_related in at least 2 of 3; U-686 unrelated in at least 2 of 3.
5. Controls: false unity in at most 1 of 6; held-out >= 0.95.
6. Turn 3 within 0.50-0.75 and pointed_obj exactly 0.50 in at least 4 of 6; turn 2 within 0.65-0.85.

# Round 12: the three follow-ups (sealed 2026-10-21, before any of the code exists or runs)

12a. Shape-class lists redone by a tool-less agent from the 1000 class names only (authors/brief_classes.md; group
definitions as before, no unit, activations or essay). Fire rate = share of photos (all 10k) whose spatial max is at
least 30% of the card max, by group.
1. pointed_obj fire rate below 11% (2x the 5.4% base rate); round_obj below 5%.
2. pointed_face_dog above round_face_animal; pointed_obj above round_obj.
3. At least half of each of my groups' classes that the blind agent also places are in the same group.

12b. Regions (tools/regions.py, claim-blind author; labels from a vision model that never sees activations; 60
random held-pool photos + 20 strongest). Essay's claim: pointed regions drive the unit, round ones push it negative.
1. Random 60: d_PR not significant (sign test p > 0.05).
2. Top 20: d_PR median below 0 (the hotspot is eyes and muzzle, which a labeller will often call round).
3. All 80: fraction of R cells with negative pre-ReLU within 10 points of the fraction for O cells.
4. Unit's median d_PR (post-ReLU, normalised) percentile among the 832 channels between 20 and 80.

12c. E2_baseline: E1's prompt and ops, plus each edit's result followed by how other channels that respond strongly to
that photo fare under the same edit (tools/peers.py, claim-blind author). U-121 and U-686, 3 each.
1. Class unchanged: U-121 one/one_plus_related in at least 2 of 3; U-686 unrelated in at least 2 of 3.
2. CFS in at most 1 of 6.
3. At least 4 of 6 test an inversion (rotate_180 or rotate 180).
4. Turn 3: pointed_obj exactly 0.50 in at least 4 of 6; turn 2 within 0.65-0.85.

# Round 13: the pose example, ground truth first (sealed 2026-10-21, before tools/views.py exists or runs)

Essay's claim: the Zoom In "car stored in dog neurons" circuit is shared pose; some mixed4d readers of the car
neuron 4c:447 are generalized pose neurons, tuned to the same view for cars and dogs. Zoom In's reader IDs are only in
figure images, so the readers are the 8 chosen by gradient in tools/pose.py: 4d:417, 455, 352, 167, 277, 241, 295,
174. Views come from a vision model that never sees activations (all 140 vehicle photos, 300 random dog photos); the
analysis (tools/views.py) is by a claim-blind author and covers every 4c and 4d channel.

1. Layer median corr (vehicle vs dog tuning over 4 folded facing bins) is at least 0.3: framing and view statistics
   are shared, so a high corr alone is weak; readers are judged by percentile.
2. At least 2 of the 8 readers have corr percentile >= 75 and min(eta2_V, eta2_D) percentile >= 50 in 4d.
3. Readers' left-right differences small (|lr_V|, |lr_D| < 0.1), as the mirror test found.
4. Of the 15 4d channels with the highest min(eta2_V, eta2_D), at least 5 have corr >= 0.7.

# Round 14: subjects on a pose candidate (sealed 2026-10-21, after round 13, before any run on U-945)

Round 13 found one clear candidate: 4d:409, top of the claim-blind ranking by view tuning in both kinds (eta2 0.16 and
0.18, corr 0.71), the 9th-strongest reader of 4c:447 by gradient; its top photos are whole-body side and three-quarter
views of dogs and cars. U-945 = field card for 4d:409 (bank/img_cards_ch.py): top-20 card-pool photos, whole (not
cropped, so the view stays visible), 0-10. Held-out: 5 vehicle_fire, 5 dog_fire, 3 vehicle_no, 7 dog_no (< 10%), so a
reading by kind scores about 0.5. Pose reading = turn-1 explanation names a view, pose or framing shared across kinds
(regex: side view/on/profile, profile, three-quarter, whole/full body, facing, pose, posture, viewpoint, standing,
orientation). Runs: A0 and Q0, 3 each.

1. A0: pose reading in at least 2 of 3; class one or one_plus_related in at least 2 of 3.
2. Q0: pose reading in at least 2 of 3.
3. Held-out at least 0.70 in at least 4 of 6; dog_no correct in at least 4 of 6.

# Round 15: the essay's blind near-miss test, ground truth first (sealed 2026-10-21, before tools/nearmiss.py exists)

tools/nearmiss.py (claim-blind author): an agent sees only an explanation and a fixed sheet of 48 held-pool photos
(16 each with act >= 0.3, 0.05-0.3, < 0.05; no activations shown), picks 6 to fire and 6 near misses; recorded
activations score them. Explanations frozen in authors/explanations/nearmiss.json. Validation first, on 4d:409, where
round 13 gave blind ground truth (pose); the test is useful for 55 only if it passes there.

Validation (U-945 = 4d:409):
1. 409_pose gap above 409_noun gap, and near_ok higher (the pose reading can name frontal dogs as near misses).
2. 409_wrong (an explanation of the other unit) near chance: fire_hit within 0.15 of the pool's chance rate.
Then mixed4e:55 (U-121):
3. 55_face and 55_or fire_hit at least 0.6; 55_wrong near chance.
4. 55_V gap below 55_face gap (pointed objects rarely fire on photos, rounds 12a/12b).

# Round 16: the essay's attribution bundles and "fixed" feature visualisation (sealed 2026-10-21, before code exists)

Both tools by claim-blind authors (authors/spec_bundles.md, spec_fixedvis.md). Ground truth first: 4d:409 (shared
pose, round 13) and a planted unit sum:mixed4e:809+804 (the 4e channels, other than 55, most selective for cats and for
cars, by a fixed rule on the battery cache), whose answer is "two separate routes" by construction.

Bundles (kinds: 409 dogs vs vehicles; planted and 55 cats 281-285 vs the 14 car classes):
1. Planted: ratio (between-kind cos / within-kind cos) below 0.5; NMI at k=2 above 0.5.
2. 4d:409: ratio above 0.7; NMI at k=2 below 0.3 (dogs and vehicles share inputs).
3. mixed4e:55: ratio between the two (0.4-0.8), NMI at k=2 at least 0.3: cats and cars reach it partly through
   separate inputs (the car input 352 against the fox-face input 493, round 6).
Fixed feature visualisation (units 4d:409, 4e:55, control 4e:700; blind one-sentence descriptions):
4. 4d:409 fixed renders described with a side view or whole body in at least 1 of 2; plain renders mention dogs or
   animal parts in at least 1 of 2.
5. 4e:55 fixed renders mention a car, vehicle or pointed/V shape in at most 1 of 2 (bias toward cat-heavy top photos).
6. Fixed renders keep at least 50% of the plain renders' activation on the unit.

# Round 17: a ground-truth suite from the essay's own examples; blind maximum per bundle; stronger pulls
(sealed 2026-10-21, before tools/curves.py, bundlemax.py, fixedvis2.py exist; all by claim-blind authors)

Ground-truth units, each with its answer fixed before any test runs:
- Curve detector in mixed3b (the essay: "the layer with Curve Neurons"; Curve Detectors). Unit = the channel
  tools/curves.py ranks first by curve/line ratio among channels with curve_peak above the 75th percentile, provided
  its osi is above the layer median. Answer: one thing (a curve at one orientation).
- 4c:447, the car neuron built from 4b car-part neurons (windows, car bodies, wheels; essay footnote 4). Answer: one
  thing with related parts.
- 4d:409, shared pose (rounds 13-14). Answer: one thing (a view shared by dogs and vehicles).
- Planted sum:mixed4e:809+804 (cat-selective + car-selective) and sum:mixed4e:703+315 (bird-selective +
  food-selective, same fixed rule). Answer: two unrelated things.
Target: 4e:55.

Curves:
1. The chosen 3b unit's curve/line ratio is above 2 and its osi above 0.3.

Blind maximum per bundle (tools/bundlemax.py):
2. Judge: planted units score at most 2 (mean of 3); 4c:447 and the curve unit at least 4.
3. 4d:409 judge at least 3. Bundles split by kind (round 16), but each bundle's explanation should name the side view.
4. Cross-bundle transfer (cross hit / own hit): planted below 0.4; 4c:447 and the curve unit above 0.6.
5. Own blind-maximum best at least 0.5 of the corpus max for at least 80% of bundle conditions (the explanations are
   good enough to reach the bundle's centre).
6. 4e:55: judge between the planted and the car scores (no strong prior); reported, not predicted.

Stronger pulls (tools/fixedvis2.py):
7. 4d:409 through 4c:447: blind descriptions name a vehicle in at least 1 of 2; through 4c:393: name a dog or
   animal in at least 1 of 2.
8. pull5 and init renders keep at least 50% of the plain renders' activation.

# Round 18: a near-miss pool built from each reading's own examples (tools/nearmiss2.py, claim-blind author)
(sealed 2026-10-21, before nearmiss2.py, the Q0/F1 readings and the class lists exist)

Units: 4d:409 (one shared pose), 4c:447 (one thing), planted sum:mixed4e:809+804 and sum:mixed4e:703+315 (two
things), target 4e:55. Readings per unit, all on the round-17c whole card (top 20 whole photos):
- noun: the bare subject's explanation (runs/bundlemax/<unit>/explanations.json, whole, sample 1, verbatim).
- Q0: the essay's question as system prompt (arms/Q0_shared.txt), sample 1.
- F1: forced unity, "one shared visual pattern, no kinds of objects, animals or scenes" (arms/F1_forced.txt), sample 1.
- wrong: another unit's noun reading (703+315's for every unit; 447's for 703+315).
- 409 adds pose (round 14 Q0 s2, as in round 15). 55 adds V (the essay's words with its examples: flames, fireboat
  jets, fox faces, shinkansens, pool tables) and or (round 8 E1, cats or cars).
A blind agent turns each reading, alone, into fire and near-miss ImageNet classes. The pool is 48 held-pool photos
drawn at random from those classes (no activations used). A blind picker sees one reading and the sheet and picks 6
fire and 6 near misses, 3 samples per reading. Class AUC = AUC of the unit's activation on all held photos of the
reading's fire classes against its near-miss classes.

Validation (whether the 55 ranking is read at all; needs 1, 2 and 4):
1. 4d:409: pose beats noun on fire_hit by at least 0.15.
2. Both planted units: noun beats F1 on fire_hit by at least 0.15, and noun's class AUC is above F1's.
3. 4c:447: noun and Q0 fire_hit at least 0.67, within 0.15 of each other.
4. wrong has the lowest fire_hit of its unit in at least 4 of 5 units.
5. Class AUC of noun at least 0.7 on 447 and both planted units.
6. 4e:55: reported, not predicted. Prior: face and or above V on fire_hit; V's near_ok high (round 15).

# Round 19: predicted activation on the whole pool (tools/simulate.py, claim-blind author)
(sealed 2026-10-21, before simulate.py exists; same units, readings, pools and sheets as round 18)

A blind agent sees one reading and the round-18 sheet and predicts the unit's activation (0-10) on all 48 photos,
3 samples per reading. Score: Spearman rho between predictions and the unit's ratio over the 48. Unlike six picks,
this charges a half-right reading for the firing photos it rates low.

Validation (needs 1, 2 and 4):
1. 4d:409: pose rho above noun rho by at least 0.10.
2. Both planted units: noun rho above F1 rho by at least 0.10.
3. 4c:447: noun and Q0 rho at least 0.5, within 0.10 of each other.
4. wrong has the lowest rho of its unit in at least 4 of 5 units, and is at most 0.2 in each.
5. 4e:55: reported. Prior: V's rho below face (noun), Q0 and or.

# Round 20: the open method on new units (tools/openmethod.py, claim-blind author)
(sealed 2026-10-21, before openmethod.py exists or any unit below has been looked at; IDs in authors/literature_ids.md)

Method, identical for every unit: a card of the 20 strongest card-pool photos (whole, peak marked by a box, 0-10)
plus 10 random photos; readings from the bare prompt, Q0 and F1, 2 samples each; a wrong-unit reading (another
unit's bare sample 1, authors/round20_units.json); class lists per reading; a 48-photo pool of 24 classes x 2 photos
(same-class pairs) drawn from those lists; predicted activation scored by Spearman rho, 3 samples per reading. The
method's answer is the top reading by mean rho. A blind judge then rates each reading against the literature answer.

Ground truth (literature): 4b:237 windows, 4b:491 car body, 4b:373 wheels (Zoom In; the essay: "basically
monosemantic"); oriented dog heads 4b:443, 4b:431, 4c:254, 4c:407 (Zoom In's two mirrored pathways).
Claims, reported: 5a:719 (the essay: pinched-center shape with semantic residue); 4d:417, 241, 277 (Zoom In's
readers of 447; the essay: two of four are car neurons and two share a pose; round 13: none had shared view tuning).

1. Car parts: the top reading matches the part (judge >= 4) in at least 2 of 3.
2. Car parts: the bare samples' class is one or one_plus_related in at least 5 of 6.
3. Dog heads: the top reading names dog heads or faces in 4 of 4, and names a facing direction in at most 2 of 4.
4. The wrong-unit reading has the lowest rho in at least 10 of 11 units.
5. On the 7 ground-truth units, the top reading is a bare or Q0 sample in at least 5.
6. 5a:719 and the 4d readers: reported, not predicted. Priors: some 5a:719 reading names a bow-tie or pinched
   shape; no 4d reader reading names a view shared by cars and dogs.

# Round 21: which views help, by ground truth (tools/lookviews.py, claim-blind author)
(sealed 2026-10-21, before lookviews.py exists; views from the blind list in authors/views.reply.json, ranks 2 and 3)

Two views added to the round-20 card (whole photos, peak boxed, 10 random), in a 2x2 design: C = the same 20
photos cropped to the unit's effective receptive field (gradient rule, no per-unit choice); P = an invariance
profile, the unit's retention under the 10 battery transforms with its percentile among all units of its layer. Four
conditions (base, C, P, CP), 3 bare readings each. A judge frozen before any reading (tools/judge_r21.py) rates each
reading against the reference.

Units and references: 3a:136, 3a:110 (high-low frequency boundary; Distill); 3b:379 (a curve at one orientation;
Distill, round 17a); 4b:237 windows, 4b:373 wheels (Zoom In, round 20); 4b:431 (facing one direction: Zoom In's
dog-head pathway, mirror drop at the 100th percentile, round 20); 4d:409 (side-on pose shared across kinds, round 13).
Targets: 4e:55 (the essay's V), 5a:719 (the essay's pinched-center shape).

1. Crops help shallow units: mean judge with C (C, CP) at least 1 point above without (base, P) on at least 2 of
   3a:136, 3a:110, 3b:379.
2. The profile helps 4b:431: a facing direction named in at least 2 of the 6 P/CP readings, at most 1 of the 6
   base/C readings.
3. Car parts: mean judge at least 4 in every condition (the added views don't hurt).
4. 4d:409: judge at least 3 against shared pose in at least 2 of 3 readings in every condition.
5. Targets: V judge at most 2, and pinched-shape judge at most 2, in every condition (no view flips them).

# Round 22: does the invariance profile make readers claim things it doesn't show? (tools/profilecheck.py)
(sealed 2026-10-21, before profilecheck.py exists; implementer and tester are separate claim-blind agents)

18 random units (3 per layer of 3a, 3b, 4b, 4d, 4e, 5a; fixed seed; studied units excluded), base and P conditions
of round 21, 3 bare readings each; plus round 21's base and P readings of its 9 units. A blind call reads only the
explanation and marks, for 10 properties (one per transform), whether it claims the unit depends on it, is
unaffected by it, or doesn't say. A claim is supported when the unit's percentile for that transform is <= 20
(depends) or >= 80 (unaffected), contradicted when it is the opposite extreme, unremarkable otherwise.

1. In P readings, at least 60% of claims are supported.
2. In P readings, at most 10% of claims are contradicted.
3. From base to P, supported claims per reading rise by at least 0.5 and unremarkable claims per reading by at most 0.3.
4. Random units whose 10 percentiles all lie in 20-80: P readings make at most 0.5 claims per reading.

# Round 23: an affected-vs-spared view, and a within-class facing test (tools/sparedview.py, tools/lrcheck.py)
(sealed 2026-10-21, before either tool exists; the view is rank 4 of a blind follow-up list, authors/followup.reply.json)

23A. View A, generic: T* = the transform with the unit's lowest profile percentile, if at most 20. From the unit's
200 strongest card-pool photos: the 10 that lose most under T*, beside 10 of matched activation that lose least
(receptive-field crops, loss shown). Conditions CP (round 21's crops + profile, now the default) and CP+A, 3 bare
readings each, 10 units (round 21's 9 plus 4b:443), judged as round 21 (443 with 431's reference).
1. 4b:431: a facing direction named in at least 2 of 3 CP+A readings (round 21: 0 of 6 with the profile).
2. Mean judge with A at least that without on at least 6 of 10 units; no unit drops by more than 1.
3. 4e:55 and 5a:719: judge against the essay's readings at most 2 with A.

23B. Ground truth for Zoom In's dog-head pairs: round 13's blind facing labels (300 dogs, 140 vehicles). For every
channel of 4b-4e: d = (mean on left-facing - mean on right-facing) / sd, per kind; percentile of |d| in its layer.
4. 4b:431 |d| on dogs at least the 95th percentile of 4b.
5. Of 4b:443, 431, 429, 384, 409, 418, at least 3 at or above the 90th percentile on dogs; the figure's row pairs
   (443/431, 429/384, 409/418) have opposite signs in at least 2 of 3.
6. 4c:254 and 4c:407 at most the 75th percentile (round 20: they don't read or behave as dog heads).

# Round 24: stratified exemplars and same-label near misses (tools/rangeview.py; blind view list ranks 1 and 5)
(sealed 2026-10-21, before rangeview.py exists)

Default card = lookviews CP (card, random row, crops, profile). Added views, 2x2: S = 20 photos drawn at random from
four activation ranges below the top (5 each from 60-100%, 40-60%, 20-40%, 5-20% of the strongest); N = for the 5
labels most common among the 20 strongest photos, the 2 photos of each on which the unit is least active (no label
names shown). Conditions CP, CP+S, CP+N, CP+S+N; 3 bare readings each; 10 units: 3a:110, 3b:379, 4b:237, 4b:373,
4b:431, 4b:443, 4b:429, 4d:409, 4e:55, 5a:719; judged as round 23 (429 with 431's reference).
1. With N, a facing direction is named in at least 2 of 3 readings for at least 2 of 431, 443, 429 (CP+N or CP+S+N).
2. 4d:409: mean judge at least 4 in CP+N.
3. 4b:443: mean judge with S at least 0.5 above without (443's direction shows across dog photos, not in its top 20).
4. Car parts: mean judge at least 4 in every condition.
5. 4e:55 and 5a:719: judge against the essay's readings at most 2 in every condition.

# Round 25: readings become photo questions, answered blind, checked within class (tools/proploop.py)
(sealed 2026-10-22, before proploop.py exists)

Loop, per unit, nothing unit-specific: a text-only agent turns the unit's 3 round-24 CP readings into up to 8
questions about a photo (whether it holds what the readings describe, and how that can differ between photos that
hold it). A second agent, which never sees readings or activations, answers them for every photo of the unit's 12
classes with the highest mean activation (120 photos). Per question: between-answer share of variance of the unit's
class-centred activation (eta2), and per answer a one-vs-rest d, each as a percentile among all channels of the
layer on the same photos and answers. Then a text-only agent revises each CP reading given this table (real), or
given the same table with answers permuted within class (null). Judged as round 24. Units: 4b:431, 443, 429,
4d:409, 4b:237, 373, 4e:55, 5a:719.
A left/right question (my analysis rule): one with an answer containing the word "left" and not "right", and one
the reverse; its left (right) answer = the such answer with the most photos.
1. A left/right question is proposed for at least 3 of 431, 443, 429, 409.
2. For at least 2 of 431, 443, 429: the preferred side (right for 431 and 443, left for 429, from round 23B) has a
   d percentile of at least 90 and a higher d than the other side.
3. For at least 2 of 431, 443, 429: the left/right question's eta2 percentile at least 95.
4. 4d:409: if it has a left/right question, its left and right answers' d percentiles lie on the same side of 50.
5. Real-table revisions name a direction in at least 5 of 9 for 431, 443, 429; null-table ones in at most 2 of 9.
6. 443 and 429: mean judge of real-table revisions at least 3.0 (round 24 CP: 2.0 each); null-table at most 2.5.
7. 409, 237, 373: mean judge of real-table revisions at least 4.
8. 4e:55 and 5a:719: mean judge at most 2 with either table.

# Round 26A: the invariance profile across the activation range (tools/rangeprofile.py)
(sealed 2026-10-22, before rangeprofile.py exists)

Profile caches (1000 random card-pool photos P0, battery transforms, every channel). Per channel and transform, in
three strata of its own P0 photos by original value rank (1-20, 21-100, 101-300): median retention t/o, and median
change |t - o| / (channel max over P0); each as a percentile among the layer's channels. Mirror is my analysis
target; the tool treats every transform alike. Oriented (round 23B, dogs, |d| >= 99th): 4b:431, 443, 429, 384.
Not oriented (23B or lrcheck |d| <= 86th on dogs and vehicles): 4b:409, 418, 4c:254, 407, 4b:237, 373, 4d:409,
4e:55.
1. 4b:443 and 429: mirror change percentile at least 95 in stratum 2 or 3 (each unit).
2. 4b:431 and 384: mirror change percentile at least 95 in stratum 1.
3. At least 7 of the 8 not-oriented units: mirror change percentile below 95 in all three strata.
4. Over all channels of 4b, 4c, 4d, 4e: Spearman of mirror change percentile in stratum 2 with lrcheck's |d|
   percentile on dogs at least 0.3 in at least 3 of 4 layers, and higher than stratum 1's in at least 3 of 4.

# Round 26B: a standing question set from what training treats as irrelevant (tools/proploop2.py)
(sealed 2026-10-22, before the standing questions or proploop2.py exist)

A tool-less agent, told only the generic setting, writes 8 questions about ways photos of one class differ that a
classifier's training treats as irrelevant (e.g. augmentation). Each round-25 unit gets them on top of its own 8
reading-derived questions; same 120 photos, blind answers for the new questions, same statistics, real and null
tables, revisions of the CP readings, judged as round 25. Left/right question and sides as round 25.
5. The standing set has a left/right question.
6. For at least 2 of 431, 443, 429: the preferred side's d percentile at least 90 and above the other side's.
7. For at least 2 of 431, 443, 429: the left/right question's eta2 percentile at least 95.
8. 4b:237, 373, 4d:409, 4e:55, 5a:719: the left/right question's eta2 percentile below 95 for at least 4 of 5.
9. Real-table revisions name a direction in at least 5 of 9 for 431, 443, 429; null-table ones in at most 2 of 9.
10. 443 and 429: mean judge of real-table revisions at least 2.5 (round 25: 1.33 each).
11. 4b:237 and 373: mean judge of real-table revisions at least 4.

# Round 27: predictions from contrasts, checked blind on other photos (tools/contrastloop.py)
(sealed 2026-10-22, before contrastloop.py exists)

Loop, per unit, nothing unit-specific, no earlier readings: an agent sees (1) for each of the unit's 4 top classes,
photos of that class from most to least active, and (2) for each of the 3 channels of its layer with the most
similar class profile, the 2 photos on which the unit most exceeds that peer and the 2 on which it falls most below,
each relative to its class average. It writes up to 4 predictions: a question, its answers, and the answers whose
photos the unit will be most and least active on within a class; 3 independent samples. A blind agent answers the
questions for the other half of the photos of the unit's 24 top classes (crc32 split). A prediction holds when the
unit's within-class d (most- minus least-active answer) is at the 95th percentile or above among its layer's
channels. Then an agent sees its explanation with each prediction's outcome (real, or with answers shuffled within
class) and says what the unit responds to. Units: 4b:431, 443, 429, 384, 4d:409, 4b:237, 373, 4e:55, 5a:719.
A direction prediction (my analysis rule): its question, most- or least-active answer contains the word "left" or
"right". Its favoured side: the side its most-active answer names, if it names exactly one; else the side opposite
the one its least-active answer names, if that names exactly one; else none. Preferred sides (round 23B): 431 and
443 right, 429 and 384 left. Oriented: 431, 443, 429, 384; controls: the other 5.
1. A direction prediction is written (any sample) for at least 2 of the 4 oriented units.
2. For at least 2 oriented units, a direction prediction favours the preferred side.
3. For at least 2 oriented units, a direction prediction favouring the preferred side holds (real answers), and no
   direction prediction favouring the other side holds for any oriented unit.
4. At most 1 of the 5 controls has a direction prediction that holds.
5. Calibration, pooled over units, checkable predictions: null answers hold at most 10%; real answers hold at least
   3 times as often as null.
6. Real readings name a direction (judge majority) in at least 5 of the 12 for the oriented units; null readings in
   fewer than real.
7. 443 and 429: mean judge of real readings at least 2.5 (round 26B: 1.33 each).
8. 4b:237 and 373: mean judge of real readings at least 4.

# Round 27, post hoc A: the predictions scored for each unit's peers (no new calls)
(sealed 2026-10-22, after round 27's results, before this analysis exists)

The same d (most minus least active answer, within class, on the same checked photos and real answers) for each of
the unit's 3 peers, with each peer's percentile in the layer. A held prediction is shared when at least one peer is
also at the 95th percentile or above.
A1. For 443 and 429 together, at least half of the held predictions are shared.
A2. Over the 5 controls, fewer than half of the held predictions are shared.

# Round 28: conditional refinements, family-aware (F) against refinement alone (R) (tools/refineloop.py)
(sealed 2026-10-22, before refineloop.py exists, and before 4b:409 and 418 have round-27 outputs)

Stage A: 4b:409 and 4b:418 (hard controls: dog-head units, peers of the oriented units, not oriented in 23B: 0.8th,
79.7th) go through round 27's tool unchanged. Stage B, per unit (431, 443, 429, 384, 409, 418): every round-27
prediction is scored on T for the unit and its 3 peers. Arm F picks the unit's strongest prediction that held and is
shared with a peer (else the one on which it differs most from a peer, |z| >= 2) and shows photos within each of its
two answers on which the unit and that peer disagree, with the peer table; arm R picks the unit's best prediction and
shows the unit's most and least active photos within each answer. 3 samples per arm write up to 2 conditional
predictions, each splitting one answer. Checked on D (never shown to the refining agent): within photos given the
split answer, the unit's d at the 95th percentile of its layer or above. Distinctive: held, and some peer below the
50th. No prompt to the refining agents mentions direction; the blind answerer is told left and right are as seen.
Direction prediction and favoured side: round 27's rule on the refinements (question, most- and least-active answer);
an answer naming a side with "its", "his", "her", "their", "own" or "'s" before it is side-ambiguous (not counted for
either side). Sides as seen in the image (round 13's labels); preferred: 431, 443 right; 429, 384 left.
1. Arm F writes a direction refinement for at least 3 of the 4 oriented units.
2. Arm F: a held direction refinement favouring the preferred side for at least 2 of the 4 oriented units, and no
   held one favouring the other side.
3. Arm F gets item 2's held, correct-side refinement for more oriented units than arm R.
4. No held direction refinement for 4b:409 (either arm); at most one across 409 and 418.
5. Calibration on D, all checkable refinements of both arms pooled: null holds at most 10%; real at least 3 times null.
6. Of held refinements, the share that is distinctive is higher in arm F than in arm R.

# Round 29: a fixed question set answered blind, every answer pair compared, rows read by an agent (tools/pairtable.py)
(sealed 2026-10-23, before pairtable.py exists)

Units: round 28's (431, 443, 429, 384 oriented; 409, 418 hard controls). The 8 standing questions of round 26B
(written once, for every unit, from "what training treats as irrelevant") are answered blind for every photo of the
units' 24 top classes, with round 28's left/right convention. On D (round 27's split), every two answers of a
question given to at least 5 photos each are compared within class; each row is written with the answer the unit is
more active on first, with the unit's percentile in its layer and its 3 similar units' percentiles. An agent reads
the table (no photos; no prompt mentions direction), says what the unit responds to and picks up to 4 rows expected
to hold on T and to be particular to the unit; 3 samples on the real table, 3 on a table from D answers shuffled
within class. A fixed rule picks from each table: rows at the 95th percentile or above, by unit percentile minus the
peers' mean percentile, top 4. Every row of the real table is also checked on T ("every"). A row holds on T when
the unit's d with the same answer first is at the 95th percentile or above; distinctive: held, and some similar unit
below the 50th on T. Null check on T: answers shuffled within class (seed crc32(unit)).

Direction row: the facing question (Q1) with first and second answers "left" and "right", either order; it favours
its first answer. Sides as seen in the image (round 13's labels); preferred: 431, 443 right; 429, 384 left. Rows
comparing one side with "toward or away" or "no clear" are not direction rows.
1. On D, the real table's direction row favours the preferred side at the 95th percentile or above for at least 3 of
   the 4 oriented units; for neither control is the direction row at the 95th or above (either side).
2. Agent, real table: for at least 3 of the 4 oriented units, some sample picks the direction row favouring the
   preferred side and it holds on T. No held direction row favouring the other side in any arm.
3. No held direction row on T for 409 or 418 in any picking arm (agent real, agent null, rule real, rule null).
4. Picked rows held on T, pooled over units: agent real at least twice agent null; agent null at most 15%.
5. Calibration: over all rows picked by the agent and the rule (real and null tables), at most 10% hold on the
   shuffled T answers.
6. Of rows held on T, the share distinctive is higher for the agent's real picks than for every real-table row at
   the 95th or above on D.
7. The rule (real table) gets item 2's held, preferred-side direction row for at least 3 of the 4 oriented units.

# Round 29B: a final reading that sees the checks on new photos (tools/finalread.py)
(sealed 2026-10-23, after round 29's results, before finalread.py exists)

Per unit and per valid real-table reading of round 29 (s1-s3): an agent sees the same D table, that reading's
explanation and picked rows, and the picked rows' checks on T, then says what the unit responds to and lists the
table rows it is confident describe the unit (structured: row numbers). Arm "real": the checks on T's real answers.
Arm "null": the same picks, with their checks on T's answers shuffled within class (round 29's T_null), presented
identically. Direction row and preferred sides as round 29.
1. Real: for each of the 4 oriented units, the preferred-side direction row is among the final rows in at least 2 of
   3 samples.
2. Real: 4b:409's direction row (did not hold on T) is among the final rows in none of its 3 samples.
3. Null: the oriented units' direction row is among the final rows in at most 3 of their 12 samples.
4. Real, all units: of picked rows that did not hold on T, at most 20% are kept; of picked rows that held, at least
   80% are kept.

# Round 30: blind transfer of the round-29 method to unseen units (tools/r30_run.py, r30_check.py)
(sealed 2026-10-23, after the units were drawn and before any of their answers, readings or intervention profiles)
Method unchanged: round 29's pairtable (questions, prompts, scoring) and round 20's openmethod readings (bare, Q0
"what one thing", F1 "no nouns"; one sample each), run by tools/r30_run.py with outputs under runs/r30a, runs/r30b.
References (authors/round30_candidates.json, tools/r30_refs.py): Nick Cammarata's audited car-circuit labels, Zoom In
(4a dog heads), Distill Equivariance figures; units mentioned anywhere in rounds 1-29 excluded. A blind agent mapped
each label, with no unit name, to the 8 questions (authors/round30_mapping.reply.json; it read only the brief).
Units drawn by tools/r30_select.py (seed crc32("round30")), authors/round30_selection.json:
30A zoomin_4a 4a:222, 234, 308, 325 (facing, side not stated); nick_inside 5a:160, 5a:23, 5b:573, 5b:907 (each
"side view": left or right more than toward or away); nick_outside 4a:259, 4b:169, 4e:756; equiv_variants 4b:391;
equiv_human_animal 4d:427 (its label maps to no question, so it is outside too); anchors 4b:431, 4b:409.
30B random: 4b:12, 4c:33, 4d:503, 4d:508, 5a:88, 5b:551.
Intervention reference: min over strata of the mirror retention percentile (runs/rangeprofile, and runs/r30a/
rangeprofile for 4a and 5b): sensitive <= 2.5, insensitive >= 10 (cut-offs set on round 29's 6 units: oriented
0.0-1.4, 409 15.4, 418 5.7). A "left/right row": Q1 left against right, or Q4 left third against right third.
1. Anchors (30A): 431's right-over-left facing row is picked and held on T in at least 2 of 3 agent samples; 409 has
   no held left/right row in any picking arm (agent real, agent null, rule).
2. Zoom In 4a: at least 2 of the 4 have exactly one held facing side among every-arm rows on T, with no violation of
   234 = 308, 325 = 222 and 308 != 222 among those.
3. nick_inside: at least 2 of 4 have a held row matching the mapped reference in at least 2 of 3 agent samples.
4. Calibration (30A): agent null picks held <= 15%, shuffled-T held over agent and rule picks <= 10%, agent real held
   rate at least twice the null rate.
5. Mirror control (30A and 30B, units with a profile): mirror-insensitive units have no held left/right row in any
   picking arm; at least half of mirror-sensitive units have a held left/right row among every-arm rows.
6. Outside (nick_outside and equiv_variants, 4 units): a reference keyword (from the blind mapping) appears in at
   least one bare/Q0/F1 reading for at least 3 of 4, and in an agent table explanation for fewer units than that.
7. Unjustified side claims recur: at least one 30A unit with no held left/right row has an agent explanation that
   names a side.
8. Attribution (r30_check.py tree, every-arm rows on T first): among inside units with mapped relations (zoomin_4a,
   nick_inside) that are not successes, at most 1 is "interpretation: missed".
9. Calibration (30B): as item 4, on the random units.
10. Random (30B): at least 3 of 6 have a held, distinctive agent pick in at least 2 of 3 samples. (Nick's essay
   expects the original InceptionV1's last layer to be the least legible; recorded per unit, not tested: our measure
   is not legibility.)
Next step, fixed now: if most non-success inside units fail on measurement (did not replicate, too few photos, low on
both halves), work on measurement (answer agreement with round 29 on the 510 shared photos, photos per class); if most
are "interpretation: missed", replace the agent's choice by the rule and limit prose to held rows; if bare/Q0/F1 name
the outside references for at least 3 of 4 units and the table explanations for at most 1, build a proposal stage
from looks (readings into questions, checked the same way); if neither names them, a proposal stage from the network
(inputs and readers). Item 7 true makes round 29B's final reading the default reporting layer.
Calls: 30A 298 (163 answer, 90 reading, 45 bare; finalread up to 90 more), 30B 166 (112, 36, 18). Run one at a time,
30A then 30B, each after a usage check.

# Round 30F: round 29B's final reading on round 30's units (tools/finalread.py via r30_run.py, r30f_check.py)
(sealed 2026-10-23, after round 30A's results and before any round-30 final reading or any 30B call)
Round 30A item 7 held, so by the sealed rule the final reading is the default reporting layer: tools/finalread.py,
unchanged, on 30A and 30B (36 + 90 calls, under the round's call budget). Calls now run up to 3 at a time per process
(r30_run.py --workers; contents unchanged). 30B's mixed5b mirror profile is runs/r30a/rangeprofile/mixed5b.csv,
copied to runs/r30b/rangeprofile (it covers every channel of the layer).
F1. Real (each stage): the final reading keeps at least 80% of picks that held on T and at most 20% of picks that did
    not (29B: 31/32 and 0/30).
F2. Real (each stage): at most 2 kept rows across all samples are left/right rows that did not hold on T (every arm).
    In 30A this covers the 6 units whose table prose named a facing side with no held row.
F3. Null (each stage): at most 20% of picks are kept.
Recorded, not predicted: unchecked additions and their T outcome; left/right words in final explanations (a word
count cannot tell a claim from a retraction).

# Round 31A: measurement, an open VLM answerer and ten times the photos (tools/vlm_answer.py, r31_validate.py,
# r31_photos.py, r31_tables.py; r31_check.py)
(sealed 2026-10-24, after round 30's results and before any VLM answer, new photo or new table exists; the tools are
being written from frozen specs by claim-blind authors and tested by separate spec-only testers)
Why: round 30's diagnostics (tools/r30_measure.py) found the table underpowered at about 110 photos per half (rows at
95+ on D keep their sign on T 87% of the time but reach 95 only 36%), and position rows near noise.
Step 1, validation (no Claude calls). Photos: every photo Claude answered in rounds 29, 30A, 30B (Claude rows merged
r29, r30b, r30a; the last wins). Candidates, each through tools/vlm_answer.py unchanged, one photo per prompt:
Qwen/Qwen3-VL-32B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen3-VL-8B-Instruct, OpenGVLab/InternVL3_5-8B
(a candidate that does not load in vLLM is dropped and reported). Selection: on the calibration half (r31_validate
half 0), the highest mean over the 8 questions of agreement with Claude; ties by list order. Benchmark: Claude
against Claude, rounds 29 and 30A on the photos answered in both.
V1. Selected model, confirmation half: per-question agreement with Claude at least the benchmark minus 5 points on
    at least 6 of 8 questions, Q1 (facing) and Q2 (size) among them.
V2. Tables (r31_validate unit_compare, all photos, units of rounds 29, 30A, 30B): Pearson r of d, Claude answers
    against the selected model's, at least 0.8 x the same r for Claude r29 against Claude r30a answers (round-29
    units, photos answered in both).
V3. With the selected model's answers, the facing row for each oriented round-29 unit's side (4b:431 and 443 right
    over left, 429 and 384 left over right) is at 95 or above on D for at least 3 of the 4.
Step 2 runs only if V1, V2 and V3 hold. Otherwise the round stops and reports.
Step 2, ten times the photos. Units: authors/round31_units.json (rounds 29, 30A and 30B, 25 units). Classes:
authors/round31_classes.json (308, the union of their top-24 classes on the old pool). 100 photos per class from the
ImageNet train mirror benjamin-paine/imagenet-1k-256x256 (centre square, Lanczos to 256; our crop224 then gives the
old pool's field of view), via tools/r31_photos.py; values on the same InceptionV1 weights; answers by the selected
model; tables by tools/r31_tables.py (rule and every arms, no readings, no Claude calls). Classes and peers stay those
of the old pool.
P1. Power: over all units' every-arm rows at 95 or above on D, at least 50% are at 95 or above on T (old pool 36%);
    with T answers shuffled within class, at most 10%.
P2. Split-half r of d (every-arm rows, D against T, all units pooled) at least 0.70 (old 0.48); Q4 (position) rows
    alone at least 0.30 (old 0.09).
P3. Oriented units: 4b:431 and 443 have a held "right over left" facing row, 429 and 384 a held "left over right"
    (every arm), all 4; 409 and 418 have no held left-against-right facing row.
P4. Zoom In 4a: all of 222, 234, 308, 325 have exactly one held facing side (every arm), with 234 = 308, 325 = 222
    and 308 != 222.
P5. Car side view (5a:160, 5a:23, 5b:573, 5b:907): at most 1 of 4 has a held row with "left" or "right" over
    "toward or away from the camera" (the within-class reference stays unsupported with ten times the photos).
P6. Random units (30B): at least 3 of 6 have a held row that is distinctive (every arm) (round 30B's agent picks: 1).
Recorded, not predicted: 4b:12's position row; 4d:427's facing rows; the count of held rows per unit, old pool
against new.
Next step, fixed now: if P1 and P2 hold, the large pool with the VLM answerer becomes the default measurement and
round 31B (below) runs on it; if they fail, the noise is in the answers rather than the sample, and the next round
studies answer reliability (repeat answers per photo) first.
Registered next task, round 31B (not sealed yet): a proposal stage from looks. The bare/Q0/F1 readings of a unit are
turned by a tool-less agent into a few unit-specific yes/no or multiple-choice photo questions (for example "is there
small text in the photo?"), answered blind by the validated answerer on the unit's photos, and checked with the same
pairwise table on D and T. Aimed at the units whose content the 8 standing questions cannot ask about (30B's random
units, the car parts).

# Round 31A2: Qwen3-VL-32B, which did not load in 31A, run again with more GPU memory (addendum, post hoc)
(written 2026-10-24, after 31A's results and before any 32B answer exists)
31A's verdict stands as recorded (Qwen3-VL-8B selected; V1, V2 failed; step 2 not run). This addendum was decided
after seeing it, so it is a second attempt, not part of 31A's seal. Change: only the GPU layout. tools/vlm_answer.py
runs unchanged through tools/r31_tp.py, which adds tensor_parallel_size=4 to vllm.LLM (8 x A100 40 GB: two copies of
the model, each answering half of the 3,080 validation photos; outputs merged). Prompt, sampling and parsing as in
31A. Everything else uses 31A's sealed rules unchanged: selection between the candidates that answered (Qwen3-VL-8B,
Qwen3-VL-32B) by mean calibration-half agreement; V1, V2, V3 for the selected model; step 2 and P1-P6 only if all
three hold, with the photo pool already built (runs/r31/pool) and step 2's answers sharded the same way.

# Round 32: basis control and joint-answer control (no calls; tools/r32_basis.py, tools/r32_check.py)
(written 2026-10-24, before tools/r32_basis.py exists or has run)
Question: is what the pairwise table finds specific to the network's channels, or would any direction in the same
space show it? Per layer of the 25 round-31 units (7 layers): the 25 units ("selected"), 40 channels drawn at random
("channel"), and 40 random unit-norm directions in channel space ("direction", spatial max of the projected map). Each
candidate gets pairtable's every-arm check on its own 24 classes among the 308 answered classes, with the merged
Claude answers (rounds 29, 30B, 30A). Same rules for every group (held: T pct >= 95 among the layer's channels and
d > 0). Joint control (selected units): each row's contrast refitted with all 8 questions' answers in one least-squares
fit, then checked on T the same way.
B1. Privileged basis: the share of channels with at least one held row exceeds the share of directions by 0.10 or
    more, and channels' held-row counts exceed directions' (one-sided Mann-Whitney, p < 0.05).
B2. Channels have more held rows per candidate than directions in at least 5 of the 7 layers.
B3. The selected units are not typical channels: their share with a held distinctive row is at least 0.20 above the
    random channels'.
B4. Sanity: each group's mean held rows with within-class permuted answers is at most half the channels' mean held
    rows.
K.  Expected from the ReLU alone, recorded as a sanity check, not evidence: median excess kurtosis of channels above
    directions in at least 6 of 7 layers.
J1. Joint control: at least 3 of the 4 oriented 4b units (431, 443 right; 429, 384 left) have their facing row held on
    T both plainly and jointly.
Recorded: how many of the selected units' held rows survive the joint fit, and which are lost.
Reading fixed now: if B1 fails, readability by this table is not specific to channels, "neurons are clean" gets no
support from it, and random directions become a required control in every later round (OLMo included). If B1 holds,
the table finds structure that is aligned with channels more than with random directions.

# Round 33 (pilot): experiments on text units, OLMo-2-1124-7B and planted (tools/run_text_arms.py, bank/olmo_units.py)
(written 2026-10-24, before any round-33 tool has run, any OLMo activation exists or any session)
Question (the full round, later): does an agent that runs its own experiments (writes texts, sees the unit's
activation on every token) end with a more predictive explanation than one that only reads the card, and is that
from choosing the experiments rather than from seeing more data? This pilot only checks the plumbing and whether
agents use the experiments. 4 cards x 4 arms x 1 sample, empty system prompt, claude-opus-5-5:
- cards: U-421 (planted sail_lag, field: the hidden-context card, near chance in every earlier arm); U-733 (planted
  repeat_cap: a repeated capitalised word, whose strongest examples are mostly names); the first 2 usable neurons of
  OLMo-2-1124-7B (base) in olmo_units.draw_units order (seed 33, layers 4-27, MLP hidden activation), cards built
  mechanically from Pile-10k.
- arms: A0 (card only); F2_pool (card + 15 more natural windows); P1_active (card + an offer of 1-3 requests of up to
  5 self-written texts; offer paragraph written blind from authors/brief_active.md); Y1_yoked (card + exactly the
  texts and results of the P1 session for that card, not chosen by the subject).
- scores: held-out turn (20 items, as every earlier round); explanation-only simulator (a fresh session sees only
  the turn-1 explanation and predicts activation 0-10 on 40 test tokens stratified across the activation range;
  Spearman rho).
Q1. Plumbing: all 16 sessions saved, and all 16 simulator replies give at least 36 of 40 predictions.
Q2. Agents use the experiments: P1_active makes 2 or more requests on average.
Recorded, not predicted (n = 4 cards, no comparison is powered): held-out accuracy and simulator rho per arm and card;
whether P1 or Y1 mentions repetition on U-733 or distant context on U-421; discriminating probe texts (predictions
differ across hypotheses).
Next step, fixed now: if Q1 and Q2 hold, the full round is sealed before it runs, with more cards (planted cards
built to mislead, real neurons, and random directions in the same MLP space as the basis control), 3 samples per arm,
and the comparison P1 against Y1 as the primary test. If Q2 fails, the offer is the problem and the full round is
not run as designed.

# Round 33 full: experiments on text units (written 2026-10-25, after the pilot, before any full-round session)
Changes from the pilot, decided from it: simulator excerpts now show 240 tokens before the marked token (the same
tokens and truth as before; the pilot's 48-token items stay in bank/KEY_*_pilot.json). Cards (14): planted U-862
(paren_year, new), U-518 (num_plural, new), U-733, U-421; OLMo neurons O-848, O-142, O-975, O-293 (draw order 3-6),
O-290, O-158; random directions in the same MLP space O-111, O-981, O-425, O-540 (draw order 1-4). Arms A0, F2_pool,
P1_active, Y1_yoked; 2 samples per card (pilot cards: samples 2 and 3, the pilot's sample 1 not counted); per card the
mean over the two samples. Run one card at a time (all arms, then the simulator), so any stop leaves whole cards.
Primary question: does choosing the experiments help, beyond seeing their results?
F1. P1 beats Y1 on simulator rho: mean difference >= 0.05 and P1 ahead on 9 or more of 14 cards. (The pilot points
    the other way; I expect this to fail.)
F2. On the 4 planted cards, P1's held-out accuracy beats A0's by 0.10 or more on average.
F3. Agent-chosen texts are better data than random natural windows: Y1's held-out beats F2's by 0.05 or more on average.
F4. Basis control: A0's simulator rho on the 6 neurons is 0.10 or more above its rho on the 4 random directions.
Evaluated only when all 14 cards are complete (tools/r33f_check.py); before that, per-card cells are recorded only.

# Round 34: one meaning across many forms, in OLMo-2 7B (tools/r34_record.py, tools/r34_analyze.py, r34_check.py)
(written 2026-10-27, before any round-34 tool exists or has run)
Stimuli: 32 forms written blind (authors/brief_r34_stimuli.md), each in four parallel versions that differ only in
the event: postponed (target), cancelled, moved earlier, flight delayed. Forms vary hedges (I think, AFAIK, IIRC),
questions, reported speech, voice, wording, register. A Codex judge (independent model) keeps only forms whose
versions are parallel and faithful. Read at a fixed suffix token appended to every sentence ("\n", as token ids);
also at the sentence's own last token (recorded only). Forms split in half by crc32: units and directions are chosen
on one half and tested on the other.
P1. Some MLP layer 8-23 classifies held-out forms' postponed versions against the rest with its top-10 neurons
    (nearest centroid) at balanced accuracy >= 0.90.
P2. Early layers (0-3) do at least 0.15 worse than the best of 8-23 (meaning is assembled, not read off the input).
P3. The meaning is not neuron-sparse: in layers 8-23 the median share of the hidden-space concept direction carried by
    its top 1% of neurons is below 0.5.
P4. The 8 neurons selected for the target at the best layer, read from Pile cards (A0 arm), are mostly not about the
    event: a Codex judge labels at most 2 of 8 readings "event_meaning" (others: lexical, structural, other).
P5. Attention writes carry at least half of the final-layer separation (the readout token must import the content
    from earlier positions).
P6. A single residual-stream direction separates held-out forms with AUC >= 0.95 in some layer 8-23.
Recorded: per layer accuracies for k = 1, 10, 100; meaning vs form variance shares of the top units; accuracies for
stance and speech act (hedged vs plain, question vs assertion) as their own targets.

# Round 35: label the unit, blind then revealed (Inception and OLMo, separate sets; tools/r35_*.py)
(written 2026-10-28, before any round-35 tool exists or has run)
A human and our agent label the same blind pages. A reveal page per group then shows the origin, the labels, our
researched explanation and held-out evidence. Two sets that share nothing: Inception (groups of 4 units) and OLMo
(single units). The tools are mine (presentation and the mechanical rules below); labeller and judges are blind.
Each set has one sealed JSON behind its reveal pages, hashed in LEDGER.jsonl before the human labels.

Inception. Values: spatial max of the layer output over 10,000 ImageNetV2 photos (runs/proploop cache; mixed3b
computed the same way). Photos split by crc32("r35:" + name) % 2: 0 = card half (blind pages), 1 = held-out half
(reveal pages and scoring). 13 groups:
- explained (3): 3b curves 379, 385, 343, 342 (Curve Detectors' first four; round 17a); 4a dog heads 222, 234, 308,
  325 (Zoom In's pathways; rounds 30A, 32); 4b oriented dog heads 443, 431, 429, 384 (rounds 23B, 32).
- named, failed (1): the car-circuit side-view labels 5a:160, 5a:23, 5b:573, 5b:907 (round 30A: not supported).
- per layer 3b, 4a, 4b, drawn in this order, each excluding every unit already placed in its layer:
  random neurons: default_rng(crc32("r35:chan:" + layer)).choice(pool, 4, replace=False);
  cluster (unexplained): a seed channel drawn the same way with "r35:clu:", plus the 3 channels with the highest
  Pearson correlation with it over card-half photos;
  random directions: 4 unit-norm standard-normal columns over the layer's channels from "r35:dir:"; value = spatial
  max of the projection.
- blind page per group: rows a-d in seeded random order; per row one feature visualisation (lucent, standard
  parameterisation and transforms, 256 steps, 128 px, seed 0; objective: mean over positions of the layer output
  dotted with the unit's vector, one-hot for a channel) and the top 5 card-half photos (whole photo, a square outline
  one third of the side at the argmax position, value 0-10 relative to the strongest shown). Pages coded I01-I13
  in seeded random order; no IDs, layers or origins. Instructions: authors/r35_brief_img.md.
- scoring photos per unit (held-out half): the 4 highest; for each, the photo of the same class with the lowest
  value (near miss), else a random photo below the unit's held-out median.
- reveal page per group: IDs, layer, origin; agent label (later the human's); our explanation and evidence (none for
  controls); the 8 scoring photos with values.
- unit score: a Codex judge sees one unit label and the 8 scoring photos (numbered tiles, seeded order) and picks the
  4 the label describes; score = share of the 8 classified correctly (chance 0.5); "can't tell" scores 0.5 with no
  call. Group score: a Codex judge sees the shared line and two sets of 8 held-out photos (each unit's top 2), this
  group's and its pair's (same layer, next category in the cycle explained, cluster, random neurons, random
  directions), and picks one (chance 0.5). The failed group has no pair and no group score.
- agent: Claude (tools/claude_call.py, no tools), one call per blind page, the human's instructions verbatim.

OLMo: round 33's 14 cards (4 planted, 6 neurons, 4 random directions), single units.
- blind page per card: the card's text as round 33's A0 saw it, marked tokens shaded by activation; coded T01-T14
  in seeded random order. Instructions: authors/r35_brief_txt.md.
- reveal page: origin (planted rule; neuron or random direction, with layer), the A0 and P1 labels of the first
  counted sample with their held-out accuracy and simulator rho, the 20 held-out items with activations.
- agent labels: round 33's, unchanged (no new calls). Human labels: round 33's simulator (S0_sim, same items, same
  prompt), one call per card; on planted cards also a Codex judge: does the label state the rule's condition (yes or
  no), for the human and A0.

Predictions on the agent's labels (Inception); the same on the human's, prefixed H, recorded separately. The human
is partially blind (has seen some readings in this project's discussions).
I1. Mean unit score: explained >= random neurons + 0.10.
I2. Basis: random directions >= random neurons - 0.10.
I3. A facing side (the word "left" or "right") in at most 2 of the 7 facing units' labels (4b:443, 431, 429, 384;
    4a:222, 234, 308).
I4. A shared line other than "nothing in common" for 3 of 3 explained groups, at most 1 of 6 random groups, and at
    least 2 of 3 clusters.
I5. Group score: explained 3 of 3; random groups at most 4 of 6.
O1. Human mean simulator rho over the 14 cards is below A0's (round 33: 0.588).
O2. Human labels state the planted rule's condition on at most 1 of 4 planted cards.
O3. Human: random directions' mean simulator rho within 0.10 of the neurons'.
Recorded: per group and category, labelled share and confidence, unit and group scores, human against agent per unit.
