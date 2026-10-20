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
