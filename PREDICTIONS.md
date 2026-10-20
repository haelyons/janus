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
