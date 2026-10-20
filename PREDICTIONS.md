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
