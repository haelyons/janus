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

## Round 22: does the invariance profile make readers claim what it doesn't show? (`tools/profilecheck.py`)

From here each tool has an implementer and a separate tester: both claim-blind, both reading only the spec; the
tester writes black-box tests of the spec's interface (`tests/`), run against the implementation before any real run.
18 random units (3 per layer; `runs/profilecheck/report.txt`) plus round 21's 9, base and P readings. A blind call
reads only an explanation and marks claims on 10 properties (one per transform); each claim is checked against the
unit's percentile (supported <= 20 or >= 80 in the claimed direction, contradicted at the opposite extreme,
unremarkable between).

    claims per reading        base    P      change
    supported                 0.59   0.95    +0.36
    unremarkable              1.20   1.26    +0.06
    contradicted              0.22   0.27    +0.05

- All four predictions failed as sealed (P supported share 0.38, not >= 0.60; contradicted 0.109, not <= 0.10;
  supported rose 0.36, not 0.5; the one flat-profile random unit made 2.7 claims per reading, not <= 0.5, though
  fewer than its 4.0 without the profile). The thresholds assumed readers make few claims without the profile; they
  make two per reading, mostly unremarkable ("depends on shape" for a unit typical of its layer), so the shares were
  set by the baseline, not by the profile.
- The profile's own effect is what the worry was about, and it is small and mostly right: of the ~0.5 claims per
  reading it adds, about three quarters are supported. It doesn't make readers invent orientation or colour stories
  on units where those are typical. It also doesn't correct claims readers take from the photos (contradicted stays
  ~11% of claims either way).

## Round 23B: Zoom In's dog-head pairs, against blind facing labels (`tools/lrcheck.py`)

Round 13's labels (a vision model that never saw activations; 134 dogs and 115 vehicles facing left or right), every
channel of 4b-4e, d = left-minus-right mean over pooled sd, per kind; `runs/lrcheck/report.txt`. 4b median |d| on
dogs 0.12, 95th percentile 0.34.

    unit      d dogs   pct   d dogs mirrored   d vehicles  pct
    4b:443    -0.95    99.2      +1.16           -0.38     93.9
    4b:431    -0.99    99.4      +1.08           +0.09     31.4
    4b:429    +1.00    99.6      -0.79           -0.01      3.9
    4b:384    +1.20    99.8      -1.10           -0.02      6.2
    4b:409    +0.00     0.8      +0.11           -0.22     74.6
    4b:418    +0.22    79.7      +0.36           -0.10     34.4
    4c:254    +0.02    10.5        -             -0.23     79.5
    4c:407    +0.25    85.9        -             +0.26     84.8

- Zoom In's oriented dog heads reproduce in 4 of the 6 4b IDs: 443 and 431 prefer dogs facing right, 429 and 384
  dogs facing left, all at the 99th percentile of the layer and dog-specific; mirroring the photos flips every sign
  (prediction 4 held; 5's first half held, 4 of 6). The figure's rows, not its columns, are the two directions, so
  my pairing was wrong (5's second half failed, 0 of 3). 4c:254 isn't a dog head (6 half held); 407 is at the 86th.
- Round 20's mirror test missed 443 (flip drop at the 40th percentile on its 20 strongest photos). Across ordinary
  dog photos the direction preference is as strong as 431's. A unit's strongest photos can hide what it does across
  its range: the blind view list's first item, stratified exemplars.
- 431 is a right-facing dog unit, as Zoom In says; the readers' "fur on the viewer's left side of the head" (round 21)
  described the same unit by where it fires, not by what it codes.

## Round 23A: an affected-vs-spared view (`tools/sparedview.py`; blind follow-up list rank 4)

Generic rule: T* = the transform with the unit's lowest profile percentile, if at most 20. From its 200 strongest
photos, the 10 that lose most under T* beside 10 of matched activation that lose least (receptive-field crops, loss
shown). CP (crops + profile, the default) against CP+A; 3 readings each; judge as round 21. `runs/sparedview/`.

    unit      T* (pct)              CP    CP+A   direction named (CP / CP+A)
    3a:136    rotate_180 (2.7)      4.0   3.3
    3a:110    mirror (2.7)          4.0   4.0
    3b:379    phase_scramble (3.8)  4.3   4.3
    4b:237    phase_scramble (0.0)  4.3   4.7
    4b:373    rotate_90 (0.2)       4.3   4.3
    4b:431    mirror (0.4)          2.7   3.0    1/3 / 3/3
    4b:443    rotate_180 (0.4)      2.0   2.0
    4d:409    phase_scramble (0.0)  5.0   5.0
    4e:55     rotate_180 (9.9)      1.7   2.0
    5a:719    grayscale (18.6)      1.0   1.0

- All three predictions held. With the view, 431 reads as "prefers heads turned or facing toward the right of the
  image" in 3 of 3, which is what the blind labels say (round 23B); the profile alone (CP) got there in 1 of 3. No
  unit drops by more than 0.7; 55 and 719 stay off the essay's readings.
- 443's T* is rotate_180, not mirror, so the view says nothing about its direction; its direction shows only across
  ordinary dog photos (23B), not in its 200 strongest.
- CP alone is now strong: 409 at 5.0 in 3 of 3, the shallow units at 4-4.3 (round 21's base was 1.7-2.7).

## Round 24: stratified exemplars and same-label near misses (`tools/rangeview.py`; blind view list ranks 1 and 5)

Default CP plus S (20 photos drawn from four activation ranges below the top) and/or N (for the top photos' 5 most
common labels, their 2 least-active photos; no label names), 2x2, 3 readings each. `runs/rangeview/`. Mean judge:

    unit      CP    CP+S  CP+N  CP+S+N   direction named (CP / S / N / SN)
    3a:110    4.0   4.0   4.0   4.0
    3b:379    4.0   4.0   3.7   4.0
    4b:237    4.3   5.0   4.7   4.7
    4b:373    4.3   4.7   4.7   4.7
    4b:431    2.0   3.0   2.0   2.3     0 / 2 / 0 / 1 of 3
    4b:443    2.0   2.0   2.0   2.0     0 everywhere
    4b:429    2.0   2.0   2.0   2.0     0 everywhere
    4d:409    4.0   4.0   5.0   4.7
    4e:55     2.0   1.7   1.7   1.7
    5a:719    1.3   1.3   1.0   1.0

- Held: 2 (409 at 5.0 with N), 4 (car parts >= 4.3 everywhere), 5 (targets <= 2). Failed: 1 (N names no direction
  for any dog-head unit) and 3 (S leaves 443 at 2.0).
- N's limit: for a broadly active unit, a label's least-active photos aren't near misses. 431's sit at 17-46% of its
  maximum. For 409 they are: frontal dogs, silent, which is the contrast that makes pose legible.
- 443 and 429 read in every condition as "upright, roughly frontal dog faces, eye and upper muzzle". That is what
  their strongest photos are. Their direction preference (23B, 99th percentile) lives in the middle of the range,
  across ordinary dog photos, and no view so far shows it; only the blind-label test does. Whether that makes them
  "oriented dog heads" or "frontal dog-face units with a direction bias" depends on how much of the range a reading
  is meant to cover.
- S gives 431 its direction in 2 of 3 ("dogs facing right"), as the affected-vs-spared view did in 3 of 3.

## Round 25: readings become photo questions, answered blind, checked within class (`tools/proploop.py`)

Per unit, nothing unit-specific: a text-only agent turns the 3 round-24 CP readings into 8 questions about a photo;
an agent that never sees readings or activations answers them for all 120 photos of the unit's 12 top classes; per
question, eta2 of the unit's class-centred activation and a one-vs-rest d per answer, each as a percentile among all
channels of the layer on the same photos and answers. A text-only agent then revises each CP reading from that table
(real) or from the same table with answers shuffled within class (null). Implementer and tester were separate agents
(tester from the spec only); `runs/proploop/`.

The check is calibrated: separation percentile above 95 in 27 of 64 real questions, 3 of 64 null (5% expected).
What it confirmed or corrected, within class:
- 4b:237: a side window clearly visible (d +0.87, 99th), frame and pillars distinct (99th). 4b:373: a wheel inside
  its arch (99th), at medium size (99.8th), front wheel (99.4th), angled view (99.2nd).
- 4d:409 (top classes are vehicles): three-quarter view +0.71 (99.4th), front or rear -1.05 (0.9th), straight side
  -0.14 (32nd). Round 13's independent facing labels agree: vehicles three-quarter 0.49, side 0.44, front 0.23.
- 4b:443 and 429: frontal face (98-99th), both eyes visible (97-98th). 4e:55: an upright, camera-facing furry muzzle
  (99.5th), whiskers (98th).
- 5a:719: every curved-contour question unremarkable (34th-72nd); only "object fills the frame" stands out (99th).
  The loop tested the readings' curve claim and found nothing.

Mean judge (direction named), real / null / round-24 CP:

    4b:431  2.00 / 1.67 / 2.00     (0 of 3 everywhere)
    4b:443  1.33 / 1.67 / 2.00
    4b:429  1.33 / 2.00 / 2.00
    4d:409  3.33 / 2.33 / 4.00
    4b:237  5.00 / 4.33 / 4.33
    4b:373  4.67 / 4.67 / 4.33
    4e:55   1.00 / 1.67 / 2.00
    5a:719  1.00 / 1.00 / 1.33

- Held: 8 (targets at most 2). Failed: 1 (no left/right question for any unit: 0 of 64 questions), so 2 and 3 fail
  too; 4 untestable; 5 (no direction named, real or null); 6 (443 and 429 fall to 1.33); 7 (409 at 3.33, because the
  revisions say "three-quarter view", which the data and round 13 support but the reference "side view" doesn't).
- The loop checks well what someone thinks to ask. Its questions follow the readings' frame: "which view of the
  head" (frontal, three-quarter, side) every time, never which way it faces. Left-right is a nuisance variable for
  recognising objects, and neither the readers nor the question writer treat it as a property.
- Revisions follow the table either way: null tables make revisions retreat to "class identity" or invert the
  claims (429 null: "tilted, three-quarter").
- Side check (profile caches, no new run): mirror retention on the 20 strongest photos puts 431 at the 0.4th
  percentile of 4b and 384 at the 0.2nd, but 443 at the 60th and 429 at the 7th. A profile shown to the question
  writer would prompt a direction question for 431, not for 443.

## Round 26A: the invariance profile across the activation range (`tools/rangeprofile.py`)

Profile caches (1000 random photos, all channels), per channel in three strata of its own photos by rank (1-20,
21-100, 101-300): median retention and median change |t - o| / max, each as a percentile in the layer. No Claude
calls. `runs/rangeprofile/`. Mirror, percentiles by stratum (change; retention):

    4b:443  change 94 / 98 / 95   retention 60 / 1.2 / 0.8    (oriented, 23B)
    4b:429  change 95 / 95 / 92   retention 7.4 / 4.7 / 1.4   (oriented)
    4b:431  change 99.8 / 99 / 98 retention 0.4 / 1.4 / 2.9   (oriented)
    4b:384  change 99.6 / 99.8 / 97                            (oriented)
    8 not-oriented units: change below 95 in every stratum

- Held: 2 (431, 384 at the top), 3 (8 of 8 not-oriented below 95). Failed: 1 (429's change peaks at 94.7 below the
  top), 4 (over whole layers, mirror change tracks round 23B's |d| on dogs weakly: Spearman 0.02-0.16).
- Retention, not sealed as the criterion, shows it more clearly: 443 goes from the 60th percentile on its top 20 to
  the 1st further down. The top-20 profile hid an asymmetry that ordinary mid-range photos show.
- Post hoc: mirror statistics pick out 4b's 25 most direction-tuned channels (23B) with AUC about 0.8 in every
  stratum, but only 0.45-0.6 in 4c-4e.

## Round 26B: a standing question set from what training treats as irrelevant (`tools/proploop2.py`)

A tool-less agent, told only "ways photos of a class differ that training treats as irrelevant, e.g. augmentation",
wrote 8 questions (`authors/standing.reply.json`): facing (left / right / toward or away / none), size, cut off,
horizontal position, lighting, tint, tilt, sharpness. Asked for every round-25 unit on its 120 photos, on top of its
8 own questions; same statistics, tables and revisions. `runs/proploop2/`. The facing question, d percentiles:

    unit     left   right  toward/away   eta2 pct   23B preference
    4b:431   0.2    81     92            92         right
    4b:443   0.0    65     97.5          99.6       right
    4b:429   12     0.2    99            99.2       left
    4d:409   98     80     1.1           99.4       none
    4b:373   97     94     0.2           99.8       none

- Held: 5 (the standing set has a facing question), 7 (443, 429 eta2 >= 95), 11 (car parts 5.0, 4.67). Failed: 6 (no
  preferred side at >= 90: with frontal faces in the "rest", the preference shows as the other side at 0.0-0.2), 8
  (373 and 4d:409 stand out through "toward or away", their front views), 9 (real tables: direction named 0 of 9;
  null tables: 3 of 9, all 443, from a chance 97.7 on "right"), 10 (443, 429 stay at 1.33).
- Post hoc, a right-versus-left contrast among side-facing photos only: 431 +0.75 (99.4th), 443 +1.12 (99.8th), 429
  -0.84 (0.8th). All three directions of 23B, at the extremes of their layer, from 120 photos and a question no one
  aimed at them. The five units without a known preference: 8th to 91st.
- So the generic question set finds it; the table format loses it (one-vs-rest against a frontal majority), and the
  reviser reads only high positive entries: it ignores "left: 0.0th" in the real table and names "right" from a
  chance 97.7 in the null one.

## Round 27: predictions from contrasts, checked blind on other photos (`tools/contrastloop.py`)

No earlier readings. Per unit, an agent sees photos of its 4 top classes from most to least active, and, for the 3
channels of its layer with the most similar class profile, the photos on which they disagree most (relative to each
one's class average). It writes up to 4 signed predictions (question, answers, most- and least-active answer); 3
samples. A blind agent answers them for the other half of the photos of the unit's 24 top classes (crc32 split,
117-131 photos). A prediction holds when the unit's within-class d (most minus least) is at the 95th percentile of
its layer or above. Then an agent sees its explanation with the outcomes (real, or answers shuffled within class)
and says what the unit responds to. `runs/contrastloop/`; judged as round 24, 384 with 431's reference.

    unit     peers (class-profile r)            held real/null  judge real/null  real readings
    4b:431   443 .88, 409 .84, 429 .83           0/12   0/12     1.33  1.00      dog-like animals, large in frame
    4b:443   409 .96, 429 .95, 418 .89           7/12   0/12     1.67  1.00      close-up dog face toward camera
    4b:429   443 .95, 409 .94, 384 .92           9/12   1/12     1.00  2.00      frontal dog face, eye contact
    4b:384   409 .93, 429 .92, 443 .89           0/12   0/12     2.00  1.00      dogs; framing shared by the layer
    4d:409   417 .76, 454 .68, 452 .68           3/10   0/10     3.00  2.00      whole vehicle, side or 3/4 view
    4b:237   209 .70, 373 .68, 256 .65           3/12   0/12     2.00  2.00      whole road vehicle
    4b:373   491 .75, 209 .75, 217 .73           4/11   0/11     2.33  2.00      whole vehicle side-on, wheels visible
    4e:55    197 .64, 24 .59, 4 .57              7/11   0/11     2.00  1.00      frontal furry face, upright ears
    5a:719   695 .70, 685 .69, 736 .68           0/11   1/11     1.67  1.33      ornaments, bows (cannot_tell)

- Held: 4 (no control has a direction prediction), 5 (calibration: real 33 of 103 checkable predictions hold, null 2
  of 103). Failed: 1-3 (0 of 108 predictions mention left or right), 6 (direction named 0 of 12, real and null), 7
  (443 1.67, 429 1.00), 8 (237 2.0, 373 2.33).
- The check is well calibrated and the readers use it: 384's readings say its framing preference "is shared by most
  units" (70th-90th percentile); null readings mostly say cannot_tell.
- The oriented units' peers are each other (431's include 443 and 429), so the mirror twin was in front of the agent.
  It found the axis, not the sign: "head turned to profile" against "facing the camera" was 431's main prediction in
  3 of 3 samples, and failed at d -0.31 (26th): profile pools the side 431 prefers with the side it avoids. 443 and
  429 hold on "frontal" at the 97th-99.8th and get the same reading; what separates them is never asked.
- Without CP readings to start from, part-level content is lost: within-class contrasts of car photos are about
  whole car against crop, so "window" and "wheel" (round 25: 5.0 and 4.67) become "whole vehicle" (2.0, 2.33).
- So contrasts plus signed predictions give measured, calibrated hypotheses about how activity varies, but one pass
  stops at the first axis that holds. Splitting a held or failed answer further, and asking what separates a unit
  from a peer that holds on the same prediction, is the missing step.

### Round 27, post hoc A: the predictions scored for each unit's peers (`tools/r27_peers.py`, no calls)

The same d on the same photos and real answers, for each unit's 3 class-profile peers. `runs/contrastloop/
peers_posthoc.txt`. A held prediction is shared when a peer is also at the 95th percentile or above.

    held, shared / all: 443 7/7, 429 8/9, 4d:409 3/3, 237 2/3, 373 4/4, 4e:55 1/7; 431, 384, 719 none held

- A1 held (443 + 429: 15 of 16 shared). A2 failed (controls 10 of 17): the car units share their "whole vehicle,
  side view" predictions with car peers too. Only 4e:55's are its own (6 of 7).
- So a layer percentile tests a unit against the whole layer and a held prediction often describes the family.
  Against peers, the oriented units do stand out where the layer percentile hides it. 431's failed "profile vs
  frontal" is d -0.31 (31st) but its peers 443, 409, 429 are at -0.78 to -1.01 (0th-4th): of the four, 431 alone
  hardly drops on profiles. 384's "frontal" is at the 68th-70th while 443 and 429 are at 98th-99th. The units that
  prefer one side differ from their family on exactly the axis whose sides the agent pooled.

## Round 28: conditional refinements, family-aware (F) against refinement alone (R) (`tools/refineloop.py`)

Stage A put 4b:409 and 418 (dog-head units, not oriented in 23B; 409's peers are 443, 429, 384) through round 27's
tool unchanged. Stage B, per unit, scores every round-27 prediction on T for the unit and its 3 peers. Arm F refines
the unit's strongest prediction shared with a peer (else the one where it differs most from a peer, by z), shown the
peer table and, within each of the prediction's two answers, the 3 photos where the unit most exceeds that peer and
the 3 where it falls most below. Arm R refines the unit's best prediction from its own most and least active photos
within each answer. 3 samples x up to 2 conditional predictions (a new question splitting one answer), checked on D
within the split answer. `runs/refineloop/`; the session limit hit mid-run and the missing calls were redone one at a
time (`run1.log`, `run2.log`).

    unit     F trigger (peer)       held F real/null  R real/null   trigger re-checked on D: unit pct, peers
    4b:431   differs (443)          0/6  0/6          0/6  0/6      F 14 (4, 10, 1)
    4b:443   shared (429)           1/6  0/6          2/6  0/6      95 (93, 100, 85)
    4b:429   shared (443)           0/6  0/6          3/6  0/6      96 (98, 90, 75)
    4b:384   differs (409)          0/6  3/6          2/6  0/6      F 91 (99, 93, 90)
    4b:409   shared (443)           1/6  0/6          4/6  0/6      R 99 (94, 94, 90)
    4b:418   shared (434)           3/6  0/6          4/6  1/6      R 100 (89, 54, 17)

- Held: 4 (no direction refinement at all for the controls), 5 (real 20 of 65 checkable hold, null 4 of 65), 6
  (distinctive among held: F 3 of 5, R 4 of 15; all of F's are 418's "whole body vs head close-up"). Failed: 1-3:
  0 of 72 refinements name a side, in either arm. (My rule also flags "right next to people", 431 R; too few.)
- The family structure replicates on fresh photos: 431's profile prediction is again its own (14th) against peers at
  the 1st-10th; 443 and 429 share "facing the camera" again (95th-96th, peers 75th-100th).
- Two failures, by inspection of the panels:
  - Peer choice. z scales the gap by the noise, which shrinks as two units correlate, so 431's trigger went to 443
    (r .88), which prefers the same side. Its disagreement photos (costume dog, play-fight, wolf heads) carry no side.
  - Naming. 429's F panel, within "looking elsewhere", is side-split where 26B's blind facing answers exist (429
    above 443: left; 443 above 429: right, right). The refiners wrote camera height, mouth open, head tilt. The
    signal was in front of the agent and was not named, as in round 27.
- So the mechanism gets the family right (which predictions are shared, which unit departs) but leaves discovery to
  an agent that does not describe a mirror difference unprompted. A blind answerer does answer it when asked (26B).

## Round 29: a fixed question set answered blind, every answer pair compared, rows read by an agent (`tools/pairtable.py`)

Measurement first, agent second. The 8 standing questions of 26B (unchanged) were answered blind, with round 28's
left/right convention, for all 570 photos of the 6 units' 24 top classes (48 calls, none failed; one pass shared by
all units). Per unit, on D, every two answers of a question given to at least 5 photos each are compared within
class. Each row is written with the answer the unit is more active on first, with the unit's percentile in its
layer and its 3 peers' percentiles (17 to 26 rows). An agent reads the table only, with no photos and no word about
direction. It says what the unit responds to and picks up to 4 rows it expects to hold on T and to be particular to
the unit. That is 3 samples on the real table and 3 on a table from D answers shuffled within class (36 text-only
calls). A fixed rule also picks rows: those at the 95th percentile or above, by the unit's percentile minus its
peers' mean, top 4. Everything is checked on T. Calls ran one at a time. OpenCode was tried for the implementer and
tester but its Go usage limit was exceeded, so both were Claude agents again. `runs/pairtable/`, checks in
`r29_check.txt`.

The facing row, "left" against "right":

    unit     D: favours, pct (peers)        agent picks it   T: pct (peers)              held  distinctive
    4b:431   right 99.4 (99.8, 95.5, 53.9)  3 of 3           97.9 (82.2, 24.4, 0.4)      yes   yes
    4b:443   right 99.8 (99.6, 18.4, 45.5)  3 of 3           99.0 (53.5, 0.4, 70.3)      yes   yes
    4b:429   left  98.2 (0.2, 8.6, 99.8)    3 of 3           99.6 (0.6, 53.1, 99.8)      yes   yes
    4b:384   left  99.8 (47.5, 99.4, 0.8)   3 of 3           99.8 (47.3, 99.6, 0.2)      yes   yes
    4b:409   right 98.6 (99.8, 7.4, 0.0)    3 of 3           60.7 (99.2, 0.2, 0.0)       no    -
    4b:418   left  84.6 (80.7, 0.0, 5.9)    0 of 3           53.5 (36.3, 0.4, 46.3)      no    -

- Held: 2 (the agent's real-table picks give a held, correct-side direction row for all 4 oriented units; none on
  the wrong side in any arm), 3 (no held direction row for either control), 4 (picked rows held on T: agent real 32
  of 62, agent null 9 of 64), 5 (shuffled T answers: 7 of 160), 6 (distinctive among held: agent real 21 of 32,
  every D row at 95 or above 8 of 16), 7 (the fixed rule also gets all 4).
- Failed: 1. On D, 409's facing row is at 98.6 for "right" (n 17 and 21). It does not replicate on T (60.7), so the
  held-out check, not the table, keeps the control clean.
- Every real-table explanation for the 4 oriented units names the correct side ("weakest when the animal faces
  left", 443; "especially weak for dogs facing right", 429). All 3 for 409 also claim a left/right asymmetry
  "particular to this unit", which is false by T. The agent's prose inherits D's noise; only checked rows are
  trustworthy.
- Null holds come from the agent picking real structure that survived the shuffle by chance. 384's shuffled table
  still had left over right at 94.3; all 3 null samples picked it and it held on real T. The others are 443's row
  11 and 418's medium-size row, also held for the real arm.
- Against rounds 27 and 28, the agent named no side in 0 of 72 refinements when it had to find the axis by looking.
  Given a generic, blind-answered measurement table with peers, it picks the side for all 4 oriented units. The
  agent's role shrinks to choosing among measured rows.
- Caveat: the standing set has a facing question because flips are standard augmentation. It was written in 26B by
  an agent told only "what training treats as irrelevant", but after direction was this project's target. A feature
  off the set's axes would not be found this way.

## Round 29B: a final reading that sees the checks on new photos (`tools/finalread.py`)

Per unit and per valid real-table reading of round 29, an agent sees the same D table, that reading's explanation
and picked rows, and each picked row's check on T ("held" or "did not hold", with T's d and percentiles), then says
what the unit responds to and lists the rows it is confident describe the unit. Arm "null" shows the same picks with
their checks on T answers shuffled within class (round 29's T_null). 36 text-only calls, one at a time, none failed.
`runs/finalread/`, checks in `r29b_check.txt`.

- Held: 1 (each oriented unit keeps its preferred-side facing row in 3 of 3 real samples), 2 (409 keeps no facing
  row in any real sample; its explanations now describe size, not direction), 3 (null: no oriented unit keeps a
  facing row in any of 12 samples), 4 (real: picks that did not hold kept 0 of 30, picks that held kept 31 of 32).
- The final reader does what a rule would: keep a row if and only if it held. It adds little judgment beyond that.
- It also adds rows it was never shown a check for (8 additions, all at 93 or above on D). Round 29's every arm
  checks those on T: 4 held (409 rows 6 and 8, 443 row 2, 384 row 4), 4 did not (409 row 7, 384 rows 6, 7, 8 in a
  null sample). The size rows it adds for 409 are at 89 to 98 for its peers, so they describe the family. The
  "confident" list is therefore not fully backed by checks.
- Caveat: round 29's "truth" for the sides is round 23B's blind facing labels with a mirror check, the same kind of
  measurement as the method, so rounds 29 and 29B are a strong replication, not an independent validation. Zoom In
  gives the grouping of these units, not their sides; it also draws 409 and 418 in the mirrored dog-head circuit,
  which our measurement does not support.

## Round 30A: the round-29 method, unchanged, on 15 units it has never seen (`tools/r30_run.py`, `r30_check.py`)

Sealed in PREDICTIONS.md before any data. References come from outside this project: Zoom In's 4a dog-head pathways
(which units share a facing, not which side), the Distill Equivariance flip pairs (4a:222/308, 4d:427/429), and
Nick Cammarata's audited car-circuit labels, mapped blind to the 8 questions. 298 calls in phases, one at a time, none
failed (at most one more lost when a phase was stopped for usage and resumed). `runs/r30a/`, check in
`r30_check.txt`.

- Held: 1 (431 keeps its right row in 2 of 3 samples; 409 no left/right row), 2 (Zoom In 4a: 222 left, 234 right,
  308 right, 325 none; 234 = 308 and the flip pair 222 != 308 both hold; 325 = 222 untestable), 4 (calibration: real
  42/154 held, null 19/159, shuffled T 31/376), 5 (no mirror-insensitive unit has a held left/right row; 4 of 6
  mirror-sensitive units do: 222, 234, 308, 431; not 907, 427), 6 (bare/Q0/F1 name the outside reference for 259,
  169, 756; table explanations for 756, 391), 7, 8 (no "interpretation: missed").
- Failed: 3 (car-circuit "side view" units 5a:160, 5a:23, 5b:573, 5b:907: 0 of 4).
- First independent check of the side measurement: the 4a facing sides agree with Zoom In's pathway grouping and the
  Equivariance flip pair. Neither source says which side is which, so only the consistency is tested.
- Side view on cars: no stable within-class effect on either half. 160's D row "right over left" (99.2) is d -0.20
  on T; 573 and 907 lean the other way on D (frontal over side, 97.4 and 89.9) and flip sign on T; 23 is flat on
  both. Either the labels describe what the top photos are (cars seen side-on) rather than a preference within
  cars, or Q1 ("which way does it face") measures vehicle viewpoint poorly. Nothing here separates the two: the
  mirror profile cannot test side against frontal.
- Check errata (the sealed output stands; corrected reading by hand from `stats.json`): the cause tree treats any
  Q1 relation naming left or right as a left/right relation, so 160 and 23 got "reference not supported ... by the
  intervention", but side-vs-frontal is mirror-symmetric; read "unresolved: low on both halves". 573 and 907 got
  "too few photos", because the tree collects only rows pointing the reference's way and theirs point the other
  way; read the same. All four are measurement-type, as the sealed attribution says.
- Unjustified claims: agent prose names a facing side with no held left/right row for 325 (left), 160, 259, 756
  (right), 409 (left, 2 of 3) and 427 (left); 160, 259, 756 and 409 are mirror-insensitive. The held-out check stays
  clean (item 5); the prose does not, as with 409 in round 29.
- 427 (human beside animal, mirror 0.38) is a near miss rather than a missing candidate: "left over right" is +0.53
  (94.5) on D and +0.40 (89.6) on T, its flip partner 429 is at 14 and 13, and "toward over right" held (99.4).
- Answer agreement with round 29 on the 510 shared photos: Q1 88%, Q2 85%, Q3 89%, Q4 90%, Q5 84%, Q6 80%, Q7 81%,
  Q8 84%.
- Sealed next step: the non-success inside units fail on measurement, so measurement comes next; item 7 makes
  round 29B's final reading the default reporting layer.

## Round 30F: the final reading (round 29B) on round 30's units (`tools/r30f_check.py`)

Sealed before any final reading. `tools/finalread.py` unchanged, 90 calls on 30A and 36 on 30B, 3 at a time
(`r30_run.py --workers`), none failed. Checks in `runs/r30a/r30f_check.txt` and `runs/r30b/r30f_check.txt`.

- All held in both stages. F1: real picks that held are kept 42/42 (30A) and 6/6 (30B); picks that did not hold are
  kept 3/113 and 6/65. F2: no kept left/right row that failed on T. F3: null picks kept 9/155 and 6/71.
- The 6 units whose table prose named a facing side with no held row: the final explanations now retract it ("the
  apparent preference for vehicles facing right did not replicate", 756; "reversed on new photos", 160) or state only
  a held row (325 and 427, "toward or away over right", held at 95 or above). One hedged leftover: 259 s2, "weaker
  tendencies toward right-facing".
- Unchecked additions again: 9 on 30A (4 held on T), 15 on 30B (9 held). The confident list stays partly unbacked.

## Round 30B: six random units (`r30_check.py --stage b`)

166 calls, none failed (30B mixed5b mirror profile copied from 30A's, which covers the whole layer). Check in
`runs/r30b/r30_check.txt`.

- Failed: 9 (calibration: real 6/54 held, null 4/62, shuffled T 1/148; real is 1.7x null, under the 2x bar), 10 (a
  held, distinctive pick in 2 of 3 samples for 1 of 6 units, 5b:551; predicted at least 3), 5 for 30B (4b:12, mirror
  11.1, has a held position row "right third over left third" in both rule arms; no random unit is mirror-sensitive).
- Only 5b:551 has something that holds: a clear, sharp, large main subject over "no single main subject" (peers low).
- Bare readings name content the 8 questions cannot ask about: small text and photographer signatures (4b:12), rows
  of small repeated elements along an edge (4c:33), pale smooth surfaces (4d:503), round glossy discs (4d:508),
  mottled snake-skin patterns (5a:88), furry heads plus gadgets (5b:551). The standing questions are nuisance
  variables; on random units the missing candidates dominate.
- Rare answers drive picks: 503's and 88's agents picked "dark" and "black and white" rows (at least 5 photos on D),
  which have too few photos on T to check.

## Round 30, measurement diagnostics (`tools/r30_measure.py`, no calls)

The sealed next step after 30A. Every-arm rows of rounds 29 and 30A (21 units, 503 rows), d on D against d on T.
Output in `runs/r30a/r30_measure.txt`.

- The table replicates better than "held" says. Rows at 95 or above on D keep their sign on T 87% of the time (47%
  against answers shuffled within class) but reach 95 again only 36% (3%); their median on T is the 90th percentile.
  With about 110 photos per half, the 95 bar on T misses most real but moderate effects.
- Split-half correlation of d over all rows +0.48 (shuffled -0.11). By question: size +0.70, facing +0.52, cut off
  +0.42, tilt +0.42, blur +0.39, tint +0.27, light +0.21, position +0.09. Position rows carry almost no signal.
- Rows against a "none" answer ("no single main subject", "no clear facing direction") are 27% of D-high rows and
  restate one fact across questions (5a:23: 10 of its 13 D-high rows; they keep their sign on T at the 81st-92nd).
- Not heavy tails: within-class residual kurtosis 2.0-4.4, the 5 largest photos at most 33% of the variance.
- The car side-view rows are different: they flip sign between halves (160, 573, 907) or are flat (23).

## Round 31A: an open VLM as the answerer, validated against Claude (`tools/vlm_answer.py`, `r31_validate.py`, `r31_check.py`)

Sealed before any answer. Four tools from frozen specs, each by a claim-blind author with separate spec-only tests
(174 pass). The 3,080 photos Claude answered in rounds 29-30 were answered by open VLMs on Lambda H100s (vLLM 0.11.0;
the boxes' driver is CUDA 12.8). Check in `runs/r31/validate/r31_check_validate.txt`.

- Candidates: Qwen3-VL-8B answered 3080/3080; Qwen2.5-VL-7B 11/3080 (its replies do not parse); Qwen3-VL-32B did not
  load (no memory left for the cache) and InternVL3.5-8B did not load (processor error). Qwen3-VL-8B is selected.
- V1 failed: agreement with Claude is far below Claude against Claude on every question (confirmation half, model
  against benchmark): facing 0.63/0.88, size 0.65/0.85, cut off 0.31/0.89, position 0.74/0.90, light 0.66/0.84,
  tint 0.63/0.80, tilt 0.46/0.81, blur 0.75/0.84.
- V2 failed: table d correlation 0.50 against 0.88 for Claude against Claude (bar 0.70).
- V3 held: with the VLM's answers, all four oriented units' facing rows are at 99 or above on D.
- By the seal, step 2 (ten times the photos) did not run. Its photo pool was built in parallel and kept (30,800
  ImageNet-train photos of the 308 classes, InceptionV1 values; `runs/r31/pool/`, gitignored, checksums in
  `remote.sha256`). The facing signal survives a weaker answerer; most other questions do not.
- The validation half split is not independent of D/T (crc32 is affine, so the "r31:" prefix only flips parity per
  name length); V1 needs only disjoint photo sets, and V2/V3 use all photos.

## Round 31A2: Qwen3-VL-32B with tensor parallelism (post-hoc addendum, `tools/r31_tp.py`)

The 32B candidate that did not load in 31A, run unchanged through `tools/vlm_answer.py` with tensor parallel 4 on
8 x A100 (two copies, half the photos each). 3062/3080 valid. Same sealed rules: it beats the 8B on the calibration
half (mean agreement 0.68 against 0.60) and is selected.
- V1 failed: confirmation half, model against Claude-vs-Claude: facing 0.74/0.88, size 0.68/0.85, cut off 0.71/0.89,
  position 0.68/0.90, light 0.69/0.84, tint 0.72/0.80, tilt 0.49/0.81, blur 0.70/0.84.
- V2 failed: table d correlation 0.59 against 0.88 (bar 0.70). V3 held (facing rows 98-100).
- Step 2 not run. A larger model closes part of the gap (cut off 0.31 to 0.71, r 0.50 to 0.59) but no question
  comes within 5 points of Claude's own repeatability. Box time about 20 minutes.
