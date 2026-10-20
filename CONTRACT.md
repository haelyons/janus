# janus: contract for the prototype

Question: can a prompt or agent config, given plain data from one unit of a network, reliably read it the way
Cammarata's "Neural Networks Are Already Clean" (Goodfire, 2026-10-15) does, rather than the way the field did
before it? And does it get there by reading the data, rather than by assuming neurons are clean?

## Cards (subject-facing data)

A card is plain recorded data for one unit, with no name, model or labels.
- Text unit: windows of text, one per document, with tokens marked «token|activation». Sections: strongest
  activations (12), middle of the range (30-70% of strongest, up to 8).
  - field format: 48 tokens before the peak, 15 after; activations scaled 0-10 (the 2023 explainer view).
  - full format: 240 before, 15 after; raw values; a one-line activation distribution.
- Image unit: a numbered grid of the strongest images with activation values.
  - field format: strongest positives only, scaled 0-10.
  - full format: strongest positives and strongest negatives (pre-ReLU), raw values.

## Subject turns (identical user messages in every arm)

Turn 1: the card, then the task: reply with only a JSON object
`{"class", "explanation", "evidence", "missing_data", "confidence"}`, class one of
`one` | `one_plus_related` | `unrelated` | `cannot_tell`.

Turn 2: 20 new inputs (one marked token per text item; numbered images for the image unit). Predict for each
whether the unit fires strongly (>= 30% of its strongest activation in the card) or barely (< 5%). Reply
`{"predictions": [{"id", "fires"}]}`. Every held-out item is one or the other by construction.

## Arms (only the system prompt differs)

- A0 bare: empty system prompt.
- C0 field-default: the 2023 automated-interpretability explainer instruction.
- C1 superposition-primed: the pre-2026 consensus stated as background.
- C2 clean-primed: Cammarata's thesis stated as background, no methods.
- C3 naive-authored: written blind by a fresh model from a neutral brief (no essay).
- C4 informed-authored: written blind by a fresh model given the essay, methods only.

## Truth

- Planted text units: exact rules over Pile-10k; the subject is told it is a language-model neuron.
  Truth classes: `one` (heterogeneous surface, one latent rule), `one_plus_related` (core plus offshoots that
  share a word or meaning), `unrelated` (OR of two unrelated rules), and an underdetermined card whose rule
  depends on context outside the window. Held-out labels come from the rule itself.
- Iconic unit: InceptionV1 mixed4e:55 (the cat/car neuron), real. Class has no ground truth; held-out labels
  are its measured activations on images it was not shown. Its class is reported as agreement with the field
  reading (unrelated: cats and cars) and with Cammarata's (one: a pointed/V shape, round on the negative side).

## Scoring (code and my reading, no LLM judge)

- Class against truth. Correct sets: one -> {one, one_plus_related}; one_plus_related -> {one,
  one_plus_related}; unrelated -> {unrelated}; underdetermined field card -> {cannot_tell, one,
  one_plus_related} (the error is calling it unrelated); its full-format twin -> {one, one_plus_related}.
- Held-out accuracy (20 items, chance 0.5), by item type (on-surface, near miss, novel surface).
- Reliability: 3 samples per arm and card; report the modal class and agreement.

## Isolation

- I (orchestrator) built the bank and hold the key. I write no candidate prompts.
- Authors and subjects run as `claude -p` with no tools, no settings, hooks, plugins or MCP, no session
  persistence, from a neutral scratch directory. They see only the text they are sent. Authors never see
  cards, truth or scorer; the naive author never sees the essay.
- Bank and predictions are hashed into LEDGER.jsonl before any subject runs.
