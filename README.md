# janus

Can a prompt, given plain recorded data from one unit of a network, read it the way Nick Cammarata's
"Neural Networks Are Already Clean" (Goodfire, 2026-10-15) does: one shared pattern where the field saw an
unrelated mixture? And does it get there by reading the data, rather than by assuming neurons are clean?

- `CONTRACT.md`: cards, subject turns, arms, truth, scoring, isolation.
- `PREDICTIONS.md`: sealed before any subject run; `LEDGER.jsonl` holds the hashes.
- `bank/`: planted text units over Pile-10k (`text_units.py`, exact rules) and the real InceptionV1
  mixed4e:55 unit over ImageNetV2 (`img_acts.py`, `img_cards.py`). `cards/` and `heldout/` are what subjects
  see; `KEY_*.json` is the truth.
- `arms/`: one system prompt per arm. C3 and C4 were written by blind, tool-less authors (`authors/`).
- `tools/`: `run_subjects.py` (two-turn bare `claude -p` sessions, no tools/settings/hooks/MCP),
  `score.py`.
- `runs/`: every transcript; `SCORES_round1.txt`.

Rebuild data: Pile-10k parquet, ImageNetV2 matched-frequency tar and `inception5h.pth` go under `raw/`
(URLs in the scripts' history); then `python -I bank/img_acts.py raw/inv2/mf.tar.gz bank/img`,
`python -I bank/img_cards.py`, `python -I bank/text_units.py raw/pile/pile10k.parquet`.

## Round 1 (claude-opus-5-5, 3 samples per cell)

Cells: class per sample (1 = one, 1+ = one plus related, U = unrelated, ? = cannot tell) and mean held-out
accuracy (chance 0.5).

```
card     unit/format/truth                  A0             C0             C1             C2             C3             C4
U-892    dotnext/field/one                  1+/1+/1 0.92   1+/1+/1+ 0.83  1+/1+/1+ 0.83  1/1/1+ 0.92    1+/1+/1 0.93   1+/1+/1 0.85  
U-570    listend/field/one                  1/1/1 0.72     1/1/1 0.70     1/1/1 0.72     1/1/1 0.73     1/1/1 0.72     1/1/1 0.72    
U-863    coffee/field/one_plus_related      1+/1+/1+ 0.95  1+/1+/1+ 1.00  1+/1+/1+ 1.00  1+/1+/1+ 0.97  1+/1+/1+ 0.97  1+/1+/1+ 0.97 
U-767    cheese_cpp/field/unrelated         U/U/U 1.00     U/U/U 1.00     U/U/U 1.00     U/-/U 1.00     U/U/U 1.00     U/U/U 1.00    
U-937    bird_month/field/unrelated         1+/U/1+ 1.00   1+/1+/U 1.00   U/U/U 1.00     U/U/U 1.00     U/U/U 1.00     U/U/U 1.00    
U-421    sail_lag/field/cannot_tell         1+/1+/1+ 0.63  1+/1/1 0.62    1/1/1 0.58     1/1/1 0.60     1/1/1 0.67     1/1/1 0.65    
U-317    sail_lag/full/one                  1+/1+/1+ 0.80  1+/1+/1 0.83   1/1+/1+ 0.83   1+/1/1+ 0.70   1+/1/1+ 0.83   1/1+/1 0.85   
U-192    inceptionv1_mi/field/None          1+/1+/U 0.67   1+/U/1+ 0.65   U/1/U 0.68     1+/1+/1+ 0.65  U/1/1+ 0.67    1/1+/1+ 0.65  
U-300    inceptionv1_mi/full/None           1+/1+/1+ 0.70  1+/1+/1+ 0.70  1+/1+/1+ 0.70  1+/1+/1+ 0.70  1+/1+/1+ 0.70  1+/1+/1+ 0.70 
```

- Planted units: every arm reads one-job and offshoot units correctly. No arm invents unity on the packed
  controls; A0 and C0 sometimes label bird+month "one plus related" while describing two triggers. No arm says
  "cannot tell" on the hidden-context card, and held-out accuracy there is near chance.
- InceptionV1 mixed4e:55: the unit behaves as the essay describes. Top photos are pointed-ear faces plus a
  convertible; the most negative are round faces (pugs, mastiffs) and balls. It fires on held-out jeeps, cabs
  and pool tables. Every arm reads "faces of small furry mammals" and misses every held-out car and pool table
  (held-out 0.65-0.70 in every arm). No arm reaches a shape reading, including C4, whose author read the essay.

## Round 2: P0_probe (bare prompt + up to 3 requests for more data)

`tools/run_probe.py`. Same card, empty system prompt, plus one paragraph: the subject may ask for the unit on
more inputs (ImageNet category names give 4 unseen card-pool photos each; text units take self-written texts).
No instruction on how to reason. Predictions sealed in `PREDICTIONS.md` (round 2).

```
card     unit/format/truth                  A0             C0             C1             C2             C3             C4             P0            
U-937    bird_month/field/unrelated         1+/U/1+ 1.00   1+/1+/U 1.00   U/U/U 1.00     U/U/U 1.00     U/U/U 1.00     U/U/U 1.00     U/U/U 1.00    
U-421    sail_lag/field/cannot_tell         1+/1+/1+ 0.63  1+/1/1 0.62    1/1/1 0.58     1/1/1 0.60     1/1/1 0.67     1/1/1 0.65     1/1/1+ 0.58   
U-192    inceptionv1_mi/field/None          1+/1+/U 0.67   1+/U/1+ 0.65   U/1/U 0.68     1+/1+/1+ 0.65  U/1/1+ 0.67    1/1+/1+ 0.65   U/U/U 0.78    
U-300    inceptionv1_mi/full/None           1+/1+/1+ 0.70  1+/1+/1+ 0.70  1+/1+/1+ 0.70  1+/1+/1+ 0.70  1+/1+/1+ 0.70  1+/1+/1+ 0.70  U/U/U 0.77    
```

- Image unit: all 6 sessions probe (convertible, sports car, cab, pickup, more cats and foxes, a few dogs) and
  all 6 conclude "cat/fox faces, and separately cars: unrelated". Held-out rises to 0.70-0.80 because cars are
  now predicted, but pool tables and the throne stay wrong in every session. No session asks for a category
  outside animals and vehicles, and none looks for what cars and cat faces share. With data and no prior, the
  agent rediscovers the field's 2017 reading of this neuron.
- Text units: no session uses the offer; all answer directly. Bird+month is now 3/3 unrelated; the
  hidden-context card is unchanged (near chance).

## Round 3: R0_recognize (pick a reading from six, or give your own)

`tools/run_recognize.py`. 6 of 6 chose "faces of small furry mammals"; none chose pointed/V. The reasons cited
real items in the cards: the round-faced hamster ranks high, and pointed bird beaks appear among the negatives.
Held-out 0.70 in all 6.

## Substrate check (no subjects; `tools/substrate.py`, `runs/substrate/`)

- Class means: cars co-dominate with cat/fox faces (grille 8th of 1000; beach wagon, sports car, jeep, cab,
  convertible, ambulance, pickup, minivan all in the top 25). Next come other vehicles, pool table (32nd) and
  bullet train (43rd). Fireboat is 489th. The top-20 grid understates cars because ImageNetV2 has 10 photos
  per class.
- Heatmaps: positives peak on the face centre and muzzle, the hamster included. Negative troughs are diffuse
  background, not objects, so the negative grid (ranked by spatial minimum, whole photo shown) misleads; the
  "pointed beaks are negative" argument comes from that display.
- Drawn shapes: everything stays under 10% of max (out of distribution). V is weakly highest, ahead of circle
  and square.

## Round 4: F0_feedback (predict a random batch, see outcomes, revise)

`tools/run_feedback.py`. Feedback accuracy was 0.55-0.90. After the reveal, 6 of 6 converge on "furry
mammal faces and, separately, vehicles/street scenes: unrelated". One says "polysemantic". Held-out was
0.75-0.85, the best so far: car_fire 1.0 everywhere, object_fire 0.3 in 2 of 6. Confidence fell in 4 of 6.
None says cannot tell; none reaches a shape reading. The update reweights within the received categories,
toward the field's two-feature reading.

## Round 5: display v2, bare arm (`bank/img_cards_v2.py`)

Same photos and held-out set; only the display changed: crops around the hotspot; on the full card, red response
shading and negatives located by the strongest 3x3 trough, shaded blue.
- U-121 (crops): 3 of 3 call the car front "face-like: headlights and grille resemble eyes and muzzle" and class
  it one_plus_related. This is the first stated link between the car and the animals in any round.
- U-860 (crops + located negatives): 3 of 3 read the negatives as dark, flat, wrinkled dog muzzles. Two describe a
  contrast: "light, narrow muzzles vs dark, flat ones"; "pointy feline/vulpine faces vs heavy dark dog snouts".
- Held-out unchanged (0.65-0.70): the readings don't predict cabs or pool tables.

## Circuit into mixed4e:55 (`tools/circuit.py`, `tools/circuit_read.py`, `runs/circuit/`)

pre = sum_c w_c * mixed4d_c + b over 528 inputs; the rebuild matches the stored maps to float16 precision.
- One input, 493, carries 29% of the positive drive and is the top input for cats, cars, other animals and objects
  alike. Its own top photos are fox, cat and red-panda faces, yet it sits at the 89th percentile on car photos. It
  carries a median 40% of the drive on firing cat photos and 20% on firing car photos. Cats and cars share the
  main route.
- A car-specific input (352: cabs, jeeps, wagons, convertibles) adds a median 17% on cars and 0% on cats.
- Input 61 (birdhouse roofs, penguin beak, a diamond-patterned drum, flamingos) looks like pointed patterns; small
  share (2%).
- The strongest negative input (456) is flat-fronted vehicles: locomotives, garbage and tow trucks.
- Cosine of mean contribution vectors: cat vs car 0.72; cat vs other animals 0.98; car vs objects 0.96.

## Round 6: contrast, circuit, and the no-nouns question (`bank/img_cards_v3.py`, `arms/N0_nouns.txt`)

Cards: U-857 adds same-class pairs (strongest beside weakest photo for the 12 top labels) and 15 off-class
positives to U-860; U-605 also adds the 8 main inputs (ranked by mean contribution on firing photos), each by its own
top photos. N0 is one sentence from the essay: "if you couldn't use groups of nouns at all, how would you describe
this unit?" Predictions 1, 2 and 5 held; 3 and the object_fire part of 4 failed.
- A0 x U-857 (contrast): 0/3 shape readings. The pairs are read as pose and size ("the weakest show the animal small,
  far away or turned away"). Held-out 0.75-0.80, up from 0.70, via car_fire.
- A0 x U-605 (+ circuit): 3/3 unrelated, all citing "a dedicated car input (352) alongside the fox/cat-face input
  (493)". None mentions input 61 or the flat-front negative 456. Held-out 0.80. The circuit, shown this way, gives
  Zoom In's reading.
- N0 x U-860 (no nouns): 0/3 shape-primary, 0/3 carrier-free; explanations still list nouns. "Pointed/narrow snouts"
  appears only in the evidence, as in round 5. Held-out 0.70, identical to A0.
- Controls: N0 on cheese/C++ and bird/month 6/6 unrelated, held-out 1.00 (A0 had 2/3 false unity on bird/month).
  The question doesn't impose unity; it also doesn't produce it.

Substrate check after the run: the input ranking was dataset-weighted. By weight alone, the second-strongest input
is 511 (w = +0.128, vs 0.311 for 493), and it is almost never active (mean activity rank 516/528). Its top photos are
crickets, bee-eaters, vine snakes, leafhoppers; pool tables drive it at 2.7x its average. Ranking by contribution on
firing photos hid it, the same way top-k photos favour ImageNet's many animal faces. No top-weighted input's top
photos are flames or water jets. The essay's are generic renders (feature visualisation), not photos.

## Renders of the unit, its inputs and readers by weight (`tools/featvis.py`, `runs/featvis/renders.jpg`)

Standard lucent settings, 256 steps, ranked by weight as in the essay's circuit figure. My reading by eye:
- Unit 55 and input 493 (w +0.311): fox/cat faces, pointed ears and muzzle.
- 511 (+0.128): green crickets with spindly, angular legs. 61 (+0.116): curved arcs and loops, not V's.
  352 (+0.102): grille and headlight textures merged with a dog's nose. 334 (+0.080): round domes with central holes,
  a round shape on the positive side. 383 (+0.079): spotted dog bodies and legs.
- Negatives 345 (-0.072): shaggy, round dog faces; 456 (-0.067): arched round shapes with fur.
- Strongest reader, mixed5a:94 (+0.100): small flames and jets on discs, plus a fox face. This is where the essay's
  "V-shaped flames" are. The flames come from a reader, which no subject has seen. Readers 240 and 81: primate
  faces; papillon-like dogs with pointed ears.
Partial support: pointed faces, flames and angular insects, with a round negative pole (345). Against: 334 renders round
domes and has positive weight.

## Round 7: renders by weight (U-283 = U-860 + renders of the unit, 8 inputs, 3 readers)

3/3 one_plus_related, "the central face of small, pointy-snouted furry mammals". Held-out 0.70, as on U-860. The renders
are used only as confirmation ("the unit's render and its strongest input both show fox and cat faces"); none mentions
the flames reader, the cricket input or the domes. Predictions 1 and 3 held; 2 failed: 0/3 mention them at all.
One session's missing_data asks for the experiment itself: "the same animal face with dark versus light fur, or fox
versus cat faces at equal framing, to separate face shape from fur colour."

## Pose test on Zoom In's car circuit (`tools/pose.py`, `runs/pose/`)

Readers of the car neuron 4c:447 = the 8 mixed4d channels with the largest d(sum 4d_j)/d(4c_447) over its strongest
car photos; not checked against Zoom In's own reader IDs. 1320 photos (1180 dogs, 140 cars).
- Mirror test: no reader loses activation when its top photos are mirrored (flip drop -0.01 to +0.07 on cars and on
  dogs; all-channel median +0.06). So they are not left/right-specific. This doesn't rule out pose: "the way they face
  the camera" (frontal, side-on) is mostly mirror-symmetric.
- Some mixed4d channels are direction-specific on both (449: +0.15 cars / +0.52 dogs; 429: +0.21 / +0.27), but they
  are not among 447's top readers.
- By eye (`runs/pose/readers.jpg`): the readers' top cars are the same few side-on vehicles in every row, a limit of
  the 140-car pool. Several readers' top dogs are side-on whole bodies (417, 167, 277), consistent with a shared view.
  Others' are shaggy dog faces (352, 241, 295). Inconclusive.
Next: Zoom In's exact reader IDs, a larger vehicle pool, and a view label (frontal / side / rear) rather than mirroring.

## Substrate: shape groups (`bank/shape_classes.json`, lists fixed before activations were viewed)

Fire rate (>= 30% of max): cats 54%, cars 63%, pointed-face dogs 32%, flat-fronted vehicles 15%, round-faced animals
12%, pointed objects 5%, round objects 1%; base rate 5.4%. Shape matters inside faces and vehicles (pointed vs round
dogs 32% vs 12%; cars vs flat fronts 63% vs 15%), but a pointed shape on its own is near base rate. Nick's examples top
the pointed objects (pool table 0.31, bullet train 0.27 mean, vs 0.13 overall). Exceptions: Persian cat 0.51 (round),
Doberman 0.07 (pointed).

## Round 8: Nick's question, a mixed cat/car grid, a disagreement turn (`bank/img_cards_v5.py`, `arms/Q0_shared.txt`)

Q0 = "Spend a few seconds wondering what one thing this might be. What's the shared pattern here?" U-686 = the 10
strongest cat and 10 strongest car photos (like the famous grid). Turn 3 = 20 photos from the shape groups, 2 firing
and 2 silent per group, so group-level readings score 0.5.
- A0 x U-686: 3/3 unrelated, "cat faces and, separately, cars". Q0 x U-686: also 3/3 unrelated. The essay's question on
  the essay's kind of grid doesn't produce a shared pattern here (prediction 2 failed).
- Q0 x U-121 and A0 x U-121: frontal-face readings, "the eyes-over-snout layout; the car front mimics it". No shape.
- Controls: Q0 6/6 unrelated, 1.00. The question doesn't impose unity.
- Turn 3: 0.55-0.75; the mixed grid does best (0.60-0.75) through flat-fronted vehicles. pointed_obj is exactly 0.50 in
  all 12 sessions: no reading separates the pointed objects that fire from those that don't.

## Round 9: experiments (`tools/run_edit.py`, arm E0_edit)

Empty prompt, the same card and turns, plus up to 3 requests of up to 8 edits of grid photos (occlude, keep, zoom,
flip, gray, rotate, shrink). Each edited photo comes back as the network sees it, with its activation. First run
invalid: parse_json returned the last nested edit item, so no edit was applied (moved to runs/broken; fixed, rerun).
- 5/6 ran experiments (2-16 edits). They are classic face-perception tests: hide the muzzle (fox 10 -> 3.7-4.7), hide
  the eyes (-> 5.8-7.2), keep only the face (-> 7.1-7.9), turn upside down (-> 1.8-2.7), remove colour (little change).
  For the car: keep its lower front (7.0) or hide it (2.1), zoom into parts (0.5-3.5).
- None tests shape: no edit removes ears or a point, and none compares an upright with an inverted V. The edits test
  the hypothesis already held. CFS 0/6.
- U-121: 2/3 one_plus_related (face plus a "pareidolic" car front). One goes the other way on its own evidence: the car
  "stayed high in greyscale and still gave 5.0 with its grille and headlights hidden, so it is not just a face-like
  car front", so unrelated. U-686: 3/3 unrelated.
- Held-out 0.70-0.75; turn 3 0.60-0.65; pointed_obj 0.50 in 6/6. Predictions 2-5 held; 1 failed (5/6, not 6/6).
- Upside-down faces lose most of their activation. That fits "upright face" and also "a V pointing down"; nobody asked
  which.

Pose: Zoom In names no reader IDs in its text (they are only in figure images), so the readers stay gradient-chosen.

## Separation audit (written before rounds 10-11)

Blind throughout: the subjects (no tools, sandbox, never see code, truth or other runs), the C3/C4 arm authors, the
Fable consult; truth comes from the network; predictions are sealed and hashed before each round; scoring is
mechanical. Not blind in rounds 3-9: one agent (the orchestrator, who has read the essay) chose card contents and
captions, the shape-class lists, the CFS keywords, the edit-op menu, and wrote the scorer and render readings. The
round-9 op menu (box occlusion) invites part-by-part tests, and the substrate result ("pointed objects at base rate")
rests on lists the orchestrator wrote. From round 10, measurement choices go to blind agents where possible.

## Round 10: blind transformation battery (`tools/battery.py`)

The transforms were chosen by a tool-less agent told only "mid-level units of an ImageNet CNN, each on its own top
photos" (`authors/brief_battery.md`, reply `authors/battery.reply.json`): grayscale, low_pass, high_pass,
phase_scramble, patch_shuffle_4x4, rotate_90, edge_map, luminance_polarity_inversion. rotate_180 and mirror were
added afterwards, not blind. The code was written by a claim-blind author from `authors/spec_battery.md` (it read only
the spec, the list and `bank/img_acts.py`; unedited, selftest ok). Retention R = median over a channel's top-20 card
photos of a(Tx)/a(x), for all 832 mixed4e channels. Channel 55 against the layer (`runs/battery/report_55.txt`,
`sheet_55.jpg`):

    transform                     R(55)  percentile  layer median
    rotate_180                    0.27     4           0.59
    grayscale                     0.73    19           0.84
    rotate_90                     0.40    21           0.54
    patch_shuffle_4x4             0.49    23           0.61
    high_pass                     0.63    29           0.71
    luminance_polarity_inversion  0.66    34           0.73
    low_pass                      0.25    39           0.28
    phase_scramble                0.03    39           0.04
    edge_map                      0.15    41           0.18
    mirror                        0.98    82           0.95

- The distinctive property is orientation: upside-down keeps 27% against a layer median of 59% (4th percentile).
  Most channels lose some activation when inverted, so the inversion drop on its own is only weak evidence;
  channel 55's drop, though, is far larger than typical.
- 55 is not unusually contour- or outline-driven: edge maps and blur hurt it about as much as the typical channel.
  Patch shuffle and grayscale hurt it a little more than typical.
- Predictions: 8 of 12 held. The orientation ones held; the "configural shape" ones (low_pass high, edge_map above
  median) failed, as did grayscale mid and phase_scramble low.
- The battery cannot separate "upright face" from "downward-pointing V": both predict the orientation result.

## Round 11: blind-authored scientist arm (E1_scientist) with the battery as edit ops

`arms/E1_scientist.txt` was written by a tool-less author from `authors/brief_rival.md` (careful experimental
scientist, rival hypotheses, discriminating experiments; no unit or essay). Image runs offer the E0 ops plus the 10
transforms. Every prediction held:
- Controls (U-767, U-937): 6/6 unrelated, held-out 1.00. The arm does not impose unity.
- U-121: 3/3 one_plus_related, "upright, front-facing faces of furry mammals; a face-like car front more weakly".
  U-686: 3/3 unrelated, "cat faces and, separately, glossy car bodywork". CFS 0/6.
- All 6 used battery transforms, and 5 of 6 tried edge_map. Replayed: edge map 9.5 -> 1.0-1.4, phase scramble -> 0.2,
  low_pass -> 3.4, rotate_180 -> 2.7, patch_shuffle -> 6.1, grayscale -> 6.5. Subjects read the edge-map drop as
  "needs fur texture", but it is typical for the layer (41st percentile). They have no baseline to tell typical
  drops from distinctive ones. The one distinctive drop (inversion) was read as "upright face".
- Held-out 0.70-0.75; turn 3 0.60-0.70; pointed_obj 0.50 in 6/6 (both pointed photos that fire predicted silent).

## Round 12: the three follow-ups

12a. Shape-class lists redone by a tool-less agent from the class names only (`authors/brief_classes.md`, reply
`authors/classes.reply.json`). Same five group definitions, no unit, activations or essay. Fire rate (spatial max >=
30% of card max, all 10k photos; base rate 5.4%):

    group               my lists   blind lists
    pointed_obj           5.0%        1.5%
    round_obj             1.0%        2.4%
    flat_vehicle         15.0%       16.2%
    round_face_animal    12.5%       30.0%   (25.2% without domestic cats)
    pointed_face_dog     31.7%       12.0%

- The blind lists reverse both of my shape contrasts. Where we both placed a class we agree; the difference is
  which classes get placed. My "pointed-face dogs" were fox-like spitz breeds (Chihuahua, husky, Pomeranian,
  basenji) and my round faces were flat-faced dogs and primates. The blind agent read "pointed muzzle" as sighthounds
  (borzoi 0.3, Doberman 0.0) and counted cats, lynx (0.9), hamster (0.7), guinea pig (0.6) and red panda (0.6) as
  round faces.
- So the earlier substrate result ("shape matters inside faces") came from how I drew the groups. A literal reading
  of "pointed" doesn't predict the unit; small furry frontal faces do, round ones included.
- Predictions: 1 held; 2 failed (both contrasts reversed); 3 held trivially.

12b. Regions (`tools/regions.py`, claim-blind author). A vision model that never sees activations marks pointed and
round cells on a 7x7 grid over 80 held-pool photos (60 random, 20 strongest). The essay's figure claims pointed things
drive the unit and round ones push it negative. Pre-ReLU, `runs/regions/report.txt`:
- Random 60: pointed minus round cells, median +6.8, 58% positive, p = 0.31. Round cells are not more negative than
  unlabelled ones (56% vs 59% of cells below zero).
- Strongest 20: round cells are the most active (round minus unlabelled +208, p = 0.03; pointed minus round -103,
  p = 0.10). The labeller calls eyes and noses round.
- Against the layer (post-ReLU, normalised): the unit's share of photos with pointed > round is 0.52, 30th
  percentile. It prefers pointed cells less than the typical channel.
- Predictions 1, 2 and 4 held; 3 failed narrowly (round cells 11 points less often negative than unlabelled, not
  within 10).
- Limit: 32 px cells and one labeller's sense of "pointed"; the essay's figure probably shows attribution, not cells.

12c. E2_baseline: E1's prompt and ops, and after each edit the median share kept by the layer's other channels that
respond strongly to that photo, with the unit's percentile among them (`tools/peers.py`, claim-blind author).
Example: upside-down, this unit keeps 29% and its 192 peers a median of 51% (9th percentile).
- No change: U-121 3/3 one_plus_related (upright frontal furry faces, face-like car front); U-686 3/3 unrelated;
  CFS 0/6. 4 of 6 tested inversion; one cited the baseline ("a much larger drop than other channels show").
- Held-out 0.70-0.75, turn 3 0.60-0.75, pointed_obj 0.50 in 6/6. All four predictions held.

## Round 13: the pose example, ground truth (`tools/views.py`)

Essay: the Zoom In "car stored in dog neurons" circuit is shared pose, and InceptionV1 has over a hundred generalized
pose neurons. A vision model that never sees activations labelled facing (8 ways) and framing for all 140 vehicle
photos and 300 random dog photos. A claim-blind author wrote the tuning analysis for every 4c and 4d channel (facing
folded to front / three-quarter front / side / rear; eta2 per kind; corr of the vehicle and dog tuning curves).
`runs/views/report.txt`, `table.csv`.
- The labels are skewed: dogs are mostly frontal (160 of 300) and vehicles mostly three-quarter front (80 of 140).
  Many channels prefer "three-quarter front" for vehicles and "front" for dogs, so the layer median corr is -0.11,
  not positive (prediction 1 failed).
- None of the 8 gradient readers of 4c:447 shows shared tuning (prediction 2 failed). 4d:417, 455, 241 and 295 are
  strongly view-tuned (eta2 89th-97th percentile), but differently for dogs and vehicles. 4d:167 and 277 have
  aligned curves (corr 0.8) but no dog tuning. Left-right differences are small, as the mirror test found (prediction
  3 held).
- One clear candidate stands out: 4d:409 has the highest view tuning in both kinds (eta2 0.16 / 0.18, corr 0.71).
  Its top photos (`sheet_4d_409.jpg`) are whole-body side and three-quarter views of cars and dogs. It is the 9th
  strongest reader of 4c:447 by gradient, just outside the 8 we had tested. Of the top 15 by view tuning, only it
  has corr >= 0.7 (prediction 4 failed). On these labels, generalized pose neurons in mixed4d are rare rather than
  common, but this measure is coarse (4 bins, facing confounded with framing, one labeller).

## Round 14: subjects on the pose candidate (U-945 = 4d:409, whole photos)

Field card (`bank/img_cards_ch.py`): top 20 card-pool photos, whole rather than cropped so the view stays visible,
0-10. Held-out: 5 vehicle_fire, 5 dog_fire, 3 vehicle_no, 7 dog_no, so a reading by kind scores about 0.5.
- A0 (bare): 3/3 unrelated, "dogs and, separately, road vehicles". All three notice the shared view ("both mostly
  appear as a whole object seen side-on"; "what they may share is visual: a whole object in side profile, standing on
  the ground") but class it as two things. This is the essay's "a single pattern described as a list of nouns".
- Q0 (the essay's question): 3/3 one, "a horizontally elongated object seen side-on, standing on the ground on legs
  or wheels; a viewpoint and shape feature rather than a semantic category".
- Held-out 0.80-0.90 in all 6; dog_no 1.00 in all three Q0 sessions (frontal heads predicted silent).
- Predictions: 1 half held (pose reading 3/3 counting "seen from the side", which the regex missed; class one 0/3,
  failed); 2 held; 3 held.
- Contrast with mixed4e:55: the same question that did nothing there unifies this unit at once. Here the shared
  pattern is visible in the photos, and blind ground truth (round 13) backs it. For 55, no blind photo-level check
  (rounds 10 and 12) backs the V.

## Round 15: the essay's blind near-miss test (`tools/nearmiss.py`, claim-blind author)

An agent sees only an explanation and a fixed sheet of 48 unlabelled held-pool photos (16 high, 16 mid, 16 low
activation), then picks 6 to fire and 6 near misses. Activations score the picks; 3 samples each; chance fire_hit 0.33.

    explanation                         fire_hit  near_ok  gap
    409 pose (subject, Q0)              0.83      0.33     +0.28
    409 nouns (dogs; cars)              0.17      0.00     -0.27
    409 wrong unit (55's face reading)  0.11      0.00     -0.37
    55 face (subject, E1)               0.83      0.56     +0.17
    55 cats or cars (subject, E1)       0.94      0.17     +0.04
    55 V (essay)                        0.33      0.83     +0.16
    55 wrong unit (409's pose reading)  1.00      0.39     +0.21

- On the ground-truth case it works: for 4d:409 the pose reading picks firing photos (0.83), and the noun reading
  falls below chance. Its "dogs" picks are frontal portraits, which stay silent, and the side-on animals it calls near
  misses do fire.
- For mixed4e:55 it fails its own control. The other unit's pose explanation scores best (fire_hit 1.00), because
  this pool's firing photos are mostly whole animals and cars. So the test can't rank 55's readings. The V reading
  picks pointed objects, which score at chance (0.33). Its high near_ok mostly reflects that round objects don't fire.
- Predictions: 1 and 4 held (4 only by 0.015); 2 failed (the wrong-unit control was below chance, not at it);
  3 held for fire_hit but failed for 55's wrong-unit control.

## To do
- Done in round 18 (validation failed at the top: the pick test rejects wrong readings but can't rank partly right
  ones). A finer score would ask for a ranking of the whole pool, or coverage of each half of a planted pair.
- Views from the blind list (`authors/views.reply.json`, brief names no unit or finding) not yet tried, each with
  its text analogue: stratified exemplars across the activation range (text: windows sampled by activation quantile);
  occlusion maps (text: leave-one-token-out on the preceding window); a low-level property report (text: how much
  token identity, position and casing alone explain); same-class or same-token near misses with low activation;
  activation distribution and base rates; output-side effects (text: logit attribution and ablation); clustered
  modes of the top crops (text: clusters of top contexts). Crops and edit profiles (ranks 2-3) are round 21; the text
  analogue of a receptive-field crop is the attribution window around the firing token.
- Later focus: neurons over training in a model with base, SFT, DPO and RLVR checkpoints (AI2 OLMo 2 / OLMo 3;
  check exact repos). Start with neurons that have ground truth (simple token neurons), then the essay's language
  examples (iy -> diy -> do it yourself; enemy; pet microchip). GPT-2 XL's 2023 neuron-explainer data is a
  field-era baseline.

## Round 16: the essay's attribution bundles and "fixed" feature visualisation (claim-blind authors)

Bundles (`tools/bundles.py`, `runs/bundles/report.txt`): per top-150 photo, gradient x input attribution over the
previous layer's channels at the unit's peak. ratio = cos(mean cat-or-dog attribution, mean car attribution) / split-half
cos within kind; k-means NMI between cluster and kind.

    unit                              ratio   NMI k=2   expected
    planted 4e:809 + 4e:804           0.04    1.00      separate (by construction)
    4d:409 (blind-confirmed pose)     0.41    0.92      shared (essay)
    4e:55                             0.72    0.85      ?

- The planted control passes. The pose unit, though, gets its dogs and its vehicles through separate inputs: vehicles
  through 4c:447 (the car neuron itself), 417 and 167, dogs through 393, 418, 166 and 435. A unit that generalises
  across kinds can still be fed by kind-specific routes, so "separate bundles" is not evidence of packing on its own.
  The test fails its ground-truth case (prediction 2 failed). The essay pairs bundles with a blind maximum per bundle,
  which we have not run.
- 55 shares more than the pose unit: its car photos also run mostly through 493, the input whose top photos are fox
  faces (+0.62, beside 352 at +0.54). Predictions 1 and 3 held.

Fixed feature visualisation (`tools/fixedvis.py`, `runs/fixedvis/sheet.jpg`): plain renders, and renders with an extra
term 0.1 x activation x cos(input-layer pattern, its mean over the top 50 photos).
- The bias changed nothing visible: fixed and plain renders look the same and keep 91-110% of the activation.
  The intervention was too weak to test the essay's claim (prediction 6 held, 4 failed, 5 held trivially).
- By eye, 4d:409's renders, plain and fixed, show side-on vehicles with wheels at the bottom among furry body
  textures. A blind describer called one plain render "trucks or train cars seen from a low side/three-quarter view"
  and the other three "scales and feathers". 55's renders are cat faces in every condition.

## Round 17a: ground truth for curves (`tools/curves.py`, claim-blind author)

Synthetic stimuli (arcs at 24 orientations and 3 radii, straight lines, blank) over every mixed3b channel. The ten
curve neurons of Curve Detectors (3b:379, 385, 343, 342, 388, 340, 330, 349, 406, 324) all fall at the 98th-100th
percentile for curve response, the 86th-97th for curve/line ratio and the 90th-98th for orientation selectivity. The
literature's IDs hold in this checkpoint. By the sealed rule the suite's curve unit is 3b:380 (ratio 11.4,
osi 0.42, preferred orientation 150 degrees). Prediction 1 held.

## Round 17b: stronger pulls in feature visualisation (`tools/fixedvis2.py`, claim-blind author)

`runs/fixedvis2/`, blind one-sentence descriptions, two seeds each. Every render keeps 81-115% of the plain render's
activation, except 55 through 352 (50-56%).
- 4d:409 through the car neuron 4c:447: "side-view fragments of vehicles: pickup trucks, vans" (2/2). Through 4c:393
  (a dog input, round 16): "furry quadruped bodies resembling shaggy dogs, seen mostly from the side" (2/2). Started
  from its top photo: "a vintage convertible seen from a three-quarter side view". pull5 (50x the earlier bias):
  animal bodies mixed with wheel and locomotive shapes. This reproduces the essay's "visualized through the car, ...
  share a pose between cars and dogs": the same unit renders as a side-on vehicle or a side-on dog depending on the
  route. Predictions 7 and 8 held.
- 4e:55: pull5, init and through 493 give fox or cat faces from the front. Through the car input 352 it gives "a
  vintage car seen from a three-quarter front-side view", at only about half the unit's activation. No render shows
  flames, jets or a V.
- Control 4e:700: food in dishes from above, in every condition.

## Round 17c: the essay's blind maximum per bundle (`tools/bundlemax.py`, claim-blind author)

`runs/bundlemax/report.txt`. Per unit: 3 input bundles by attribution clustering, each bundle alone (others zeroed)
explained twice from a card of its top 20 whole photos, a blind picker given only the explanation picks 6 of 48
held-pool photos, and a judge (3 calls) rates whether the three bundle explanations describe one thing (1-5).

unit                judge   bundle explanations
3b:380 (curve)      4 HIGH  dials and gauges / dials and portholes / wheels on vehicles
4c:447 (car)        5 HIGH  side-on road vehicles, all three
4d:409 (pose)       3 MID   side-on cars / side-on quadrupeds / side-on dogs and wolves
planted 809+804     2 LOW   cats and keyboards / road vehicles / laptops and dogs (1 channel, 2% share)
planted 703+315     1 LOW   tentacles and florets / bottle shapes / birds
4e:55               1 LOW   cars / a mix with no theme (8% share) / cat and fox faces (71% share)

- Prediction 2 held: planted 2 and 1; car 5, curve 4. Prediction 3 held: 409 judge 3, every bundle names the side
  view (shared: "elongated objects seen side-on"). The pose ground truth is read at the middle, not the top.
- Prediction 4 (cross hit / own hit, mean over bundles) half held: curve 0.98 and car 1.00 above 0.6; planted
  809+804 0.35 below 0.4; planted 703+315 0.50 misses (the bird bundle's pool is hit by the other explanations, 0.63
  vs chance 0.33).
- Prediction 5 missed narrowly: own best at least 0.5 of the corpus max in 14 of 18 bundle conditions (78%; 83% with
  the whole conditions). The four misses are 0.42-0.50.
- 4e:55 (prediction 6, reported): judge 1, at the planted end, with transfer 0.42 / 0.89 / 0.26, like the planted
  809+804. Its bundles read as a car route and a cat-face route; no explanation names a V, a point or flames.
- With a ground truth that shares one thing across kinds (409), the method finds it; on 55 it finds the received
  reading. This is the essay's own step (ablate the other bundles, then a blind maximum), and no shared shape comes
  out of it.

## Round 18: a near-miss pool built from each reading's own examples (`tools/nearmiss2.py`, claim-blind author)

`runs/nearmiss2/report.txt`, readings in `authors/explanations/nearmiss2.json` (`tools/readings_r18.py`). Each reading,
alone, is turned into fire and near-miss ImageNet classes by a blind agent; the 48-photo pool is drawn at random from
all readings' classes (no activations), so the V reading brings its own fireboats, bullet trains, pool tables and
flames. Then round 15's blind picker (6 fire, 6 near misses, 3 samples). fire_hit chance 0.27-0.35.

    unit             noun   Q0     F1     wrong  extra
    409 pose         1.00   0.94   1.00   0.00   pose 1.00
    447 car          0.94   0.94   0.83   0.00
    planted 809+804  1.00   0.89   0.33   0.28
    planted 703+315  0.89   0.89   0.83   0.00
    4e:55            1.00   1.00   0.83   0.00   V 0.39, or 1.00

- Validation failed (needed 1, 2 and 4): 1 failed (pose and noun both 1.00 on 409), 2 held for 809+804 (forced
  unity 0.33, rejected) but failed for 703+315 (F1, "rounded mass of repeated small bulges", 0.83, and its class
  AUC 0.69 beat noun's 0.61). 3 held (447). 4 held, 5 of 5: the wrong-unit reading is lowest everywhere. 5 held for
  447 (0.85) and 809+804 (0.85), failed for 703+315 (0.61).
- What failed is resolution at the top: six picks only need to find some firing photos, so a reading that is half
  right (one half of a planted pair) or right in a different way (nouns on 409) reaches the ceiling. The test rejects
  readings that are wrong (wrong unit 5/5, forced unity on cats + cars) but can't rank readings that are partly right.
  By the sealed rule, the ranking on 55 isn't read as a verdict.
- 4e:55, reported: noun (faces), Q0 and or reach 1.00; F1 ("fluffy texture around two dark spots above a centre",
  i.e. a face without nouns) 0.83; V 0.39 against chance 0.27, wrong 0.00. V is the only reading of a real unit
  that lands near the wrong-unit floor.
- The essay's own examples, on all held photos (held median 0.12, 95th percentile 0.31): pool table 0.35 (n=7, max
  0.52), bullet train 0.23 (n=2), fireboat 0.09 (n=2), torch 0.13, candle 0.21, volcano 0.13, foxes 0.25-0.59. V's
  round near misses: balls 0.07-0.21, near baseline rather than negative; Persian cat 0.48. Pool tables are a real
  point for the essay; fireboats and flames are not, and round cat faces fire.
- Q0 on 409 said "two separate categories" this time (round 14: 3/3 one). F1 named the side view on 409 and a
  face pattern on 55; it found a pattern on every unit, including the planted pairs.

## Round 19: predicted activation on the whole pool (`tools/simulate.py`, claim-blind author)

`runs/simulate/report.txt`. Same readings and round-18 pools. A blind agent sees one reading and the sheet and
predicts the unit's activation (0-10) on all 48 photos, 3 samples; score = Spearman rho with the recorded activation.
Consistency between samples is 0.93-1.00 everywhere.

    unit             noun   Q0     F1     wrong   extra
    409 pose         0.63   0.61   0.66   -0.47   pose 0.65; noun_only 0.55 (post hoc)
    447 car          0.90   0.89   0.87   -0.81
    planted 809+804  0.79   0.78   0.19   -0.41
    planted 703+315  0.77   0.69   0.60   -0.61
    4e:55            0.47   0.46   0.40   -0.29   V 0.20, or 0.48

- Prediction 2 held on both planted pairs: forced unity loses by 0.60 and 0.17. This fixes round 18's failure: the
  half-right reading of 703+315 is charged for the firing photos it rates low. 3 held (447). 4 held (wrong lowest
  and negative in 5 of 5). 1 failed: pose 0.65 against noun 0.63. So validation failed again, on 409 only.
- Why 1 failed: every sealed 409 reading names the side view, the bare one included ("the only plausible shared
  factor is a side-profile view"). Post hoc, round 15's pure-noun reading ("dogs, and separately cars") scores 0.55,
  0.10 below pose. The pool is drawn by class, so it has few same-class contrasts (frontal against side-on dogs) for
  pose to win on.
- 4e:55 (prior held): faces, Q0 and cats-or-cars 0.46-0.48; the face pattern without nouns 0.40; V 0.20, top12 at
  chance (0.25). V is the lowest reading of a real unit on every scorer so far (rounds 15, 18, 19).
- Of the scorers tried, this is the only one that ranks partly right readings and passes both planted controls.

## Round 20: the open method on 11 new units (`tools/openmethod.py`, claim-blind author)

Essay review and literature IDs in `authors/literature_ids.md` (Zoom In's IDs are svg ids in its figures). Method per
unit, unchanged across units: card of the top 20 whole photos (peak boxed) plus 10 random; readings bare, Q0, F1 x 2;
a wrong-unit reading; class lists per reading; a 48-photo pool of 24 same-class pairs; predicted activation scored by
Spearman rho (3 samples). Judge (`tools/judge_r20.py`, frozen before the readings): each reading against the reference.
`runs/openmethod/report.txt`, `judge.json`.

    unit        reference (source)            top reading (rho)                                   judge
    4b:237      windows (Zoom In)             bare: side windows of vehicles (0.73)                5
    4b:491      car body (Zoom In)            Q0: lower side bodywork near the wheel arch (0.84)   4
    4b:373      wheels (Zoom In)              bare: front wheel and wheel arch (0.67)              4
    4b:443      oriented dog head (Zoom In)   F1: furred patch beside the eye, frontal head (0.58) 2
    4b:431      oriented dog head (Zoom In)   F1: dense soft fibrous texture (0.66)                2
    4c:254      oriented dog head (Zoom In)   bare: thin upright vertical structures (0.61)        1
    4c:407      oriented dog head (Zoom In)   Q0: throat, chest, lower muzzle of dogs (0.64)       2
    5a:719      pinched-center shape (essay)  Q0: close-ups of handcrafted, decorated objects      1
    4d:417      reader of 447                 bare: vehicle side panels; separately dogs (0.77)    2 (vs shared pose)
    4d:241      reader of 447                 Q0: shaggy dog faces and vehicle bodywork (0.64)     2
    4d:277      reader of 447                 Q0: animal heads and vehicle side panels (0.78)      2

- Car parts reproduce: all three named correctly and classed one or one-plus-related (predictions 1 and 2 held).
- Dog heads don't: no reading of any unit names a facing direction (prediction 3: the direction half held, 0 of 4;
  the dog-head half failed, 2 of 4). Mechanical mirror test (`tools/r20_checks.py`, orchestrator-written, after the
  results): 4b:431 loses 57% of its activation when its top photos are mirrored, the 100th percentile of 4b (median
  5%). It is direction-specific, and every reading calls it fur texture. 4b:443, 4c:254 and 4c:407 sit at the
  28th-65th percentile, so the layout-based dog-head ground truth holds only for 431 (my mapping from figure ids to
  roles may be wrong for the rest; 4c:254 reads and scores as vertical structures).
- Prediction 4 failed (wrong lowest in 7 of 11): on 4c:254, 4d:241 and 4d:277 the forced-unity readings score below
  the wrong unit (down to -0.27), and on 4b:373 the wrong reading (4d:417, vehicle side panels) is nearly right.
  Prediction 5 held (5 of 7 top readings bare or Q0).
- 5a:719: no reading names a pinched or bow-tie shape; bare sample 1 says cannot tell. Class means: necklace, wallet,
  Christmas stocking lead; bow tie ranks 109th, butterflies 25th-300th, swimsuits 265th-332nd, while the essay's
  "residue" ranks high (Loafer 12th, purse 27th). On this checkpoint the essay's reading of 5a:719 isn't visible.
- The three readers of 447 read as cars plus dogs, unrelated, 6 of 6 bare samples; none names a shared view, as in
  round 13. F1 on 417 says "the side flank of a bulky side-on body" (judge 4 against shared pose), but scores 0.65
  against 0.77 for the noun reading.
- The miss on 431 is the method's: no reading proposed direction, so nothing tested it. The battery's mirror
  transform with layer baselines (round 10) would have flagged it; that is the one battery result that is
  distinctive, as rotate_180 was for 4e:55. It belongs in the look step as a hypothesis source.

## Round 21: which views help, against ground truth (`tools/lookviews.py`, claim-blind author)

The views were chosen from a blind list (`authors/brief_views.md` names no unit or finding; reply
`authors/views.reply.json`, ranks 2 and 3): C = the 20 top photos cropped to the unit's effective receptive field
(gradient mass rule, one size per unit, 32-224 px); P = an invariance profile (retention under the 10 battery
transforms, with the layer median and the unit's percentile). Base = round 20's card. 2x2, 3 bare readings per
condition, judged against references frozen before any reading (`tools/judge_r21.py`). Mean judge (1-5):

    unit          reference                   base   C     P     CP
    3a:136        high-low frequency edge     2.0   3.0   2.0   4.0
    3a:110        high-low frequency edge     1.7   4.0   2.0   4.0
    3b:379        curve at one orientation    2.7   4.0   3.0   4.0
    4b:237        windows                     4.7   4.7   4.3   4.7
    4b:373        wheels                      5.0   5.0   5.0   4.0
    4b:431        facing one direction        1.0   1.7   2.0   2.0
    4d:409        side-on pose across kinds   3.0   4.0   4.3   4.3
    4e:55         the essay's V               1.7   1.3   1.7   2.0
    5a:719        the essay's pinched shape   1.0   1.0   1.0   1.0

- Crops are the big effect on shallow units (prediction 1 held, 3 of 3, +1.2 to +2.2): with crops, 3a:110 reads as "a
  vertical boundary where a smooth region on the left meets fine dotted texture on the right", the Distill
  description with its orientation. Whole photos with a box give "knitted fabric" and similar nouns.
- Neither view hurts the car parts (3 held) or flips the targets (5 held: 55 stays an upright frontal furry face,
  "face-like" car grilles; 5a:719 stays handcrafted objects).
- Both views lift the pose unit from 3 to 4-4.3 (4 held). With the profile alone: "the side-on middle section of an
  upright, horizontally elongated body held up on legs or wheels ... covers both standing four-legged animals and
  cars seen from the side", without Nick's question.
- The profile did not make 4b:431 read as oriented (prediction 2 failed, 0 of 6 by the judge), although it showed
  mirror "keeps 41%; layer median 95%; percentile 0". Two of the six profile readings moved the response to "one side
  of the face" ("the viewer's-left side of a dog's head"); none without it did. Readers took the number as a
  location, not as a facing direction. Whether "fur on the left side of a dog's head" and "a dog head facing right"
  are the same unit at two levels of description needs a within-class view test (blind facing labels).
- Compute: the machine has a GTX 1070; `.venv-gpu` (torch 2.14.1+cu126, gitignored, 6.5 GB) runs InceptionV1 at 416
  photos/s, against ~25 on CPU. Claude calls now run 12 at a time (runtime setting, logged in the ledger).
