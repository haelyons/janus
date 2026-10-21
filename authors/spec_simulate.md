# Spec: tools/simulate.py (predicted activation on a fixed pool, scored by rank correlation)

Write one new file, `tools/simulate.py`. Read only this spec, `tools/claude_call.py` and `tools/nearmiss.py`. Do not
open any other file, and do not run anything.

## Purpose
Score several explanations of the same unit by how well an agent that sees only the explanation can predict the
unit's activation on every photo of a fixed pool: the rank correlation between predicted and recorded activation.

## Inputs
- `authors/explanations/nearmiss2.json`: {unit: {explanation_id: {"text": ..., "source": ...}}}. Explanation ids and
  sources are never shown to any agent.
- Per unit, from an earlier run: `runs/nearmiss2/<unit>/pool.json`, a list of 48 records with keys "number" (1-48)
  and "ratio" (the unit's recorded activation on that photo; other keys are ignored and never shown), and
  `runs/nearmiss2/<unit>/sheet.jpg`, the 48 photos numbered as in pool.json. Use them as they are; check there are 48
  records with numbers 1-48.

## Calls
For each explanation, 3 samples, each a fresh single-turn `claude_call.converse('', [content])`, up to 4 in parallel.
Content: a text block, the sheet as an image block (as tools/nearmiss.py builds it), then a text block. The first
text block is tools/nearmiss.py's FIRST, verbatim with {text} filled in (import it). The last, verbatim:

  Using only the explanation, predict the unit's activation on every photo, from 0 (silent) to 10 (as strong as on
  the photos it responds to most). Reply with only JSON: {"predictions": {"1": n, "2": n, ..., "48": n}}

Parse with claude_call.parse_json (the object has a "predictions" key). Valid: keys exactly "1"-"48" (strings or
integers), every value a number (int or float, not bool) in 0-10. Retry once in a fresh conversation, then record the
sample as invalid. An is_error reply is not cached. Cache replies in `runs/simulate/<unit>/replies.json` keyed by
explanation id and sample; skip cached.

## Scores
Per sample: rho = Spearman rank correlation (average ranks for ties) between the 48 predictions and the 48 ratios;
if the predictions are all equal, rho = 0 and the sample is flagged constant. Also top12 = how many of the 12
photos with the highest ratio are among the 12 with the highest prediction (ties in predictions broken by photo
number), divided by 12. Per explanation: mean rho and min-max over valid samples; rho of the mean prediction over
valid samples; consistency = mean pairwise Spearman between the valid samples' predictions; mean top12. Chance:
rho 0, top12 0.25. Implement Spearman directly (rank with average ties, then Pearson); no scipy or sklearn.

## Report
`runs/simulate/report.txt`, printed. Per unit: a header; one line per explanation in file order: id, source,
rho mean (min-max), rho of mean prediction, consistency, top12, valid samples, constant flags; then the ids ranked
by mean rho. Save everything per unit in `runs/simulate/<unit>/scores.json`.

## Selftest
`--selftest`, without Claude: reply validation (good; missing key; extra key; out of range; bool; string number;
integer keys), Spearman against hand values with ties and the constant case, top12 with ties, consistency on a
synthetic case. Print `selftest ok`.

## Usage line
    python -I tools/simulate.py --units mixed4c:447,mixed4e:55 [--selftest]
