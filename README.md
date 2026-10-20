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
