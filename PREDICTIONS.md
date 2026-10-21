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
