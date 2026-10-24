# Spec: tools/run_text_arms.py (text-unit arms: bare, extra windows, experiments, yoked; and a simulator score)

Write one new file, `tools/run_text_arms.py`. Read only this spec and `tools/run_probe.py`, `tools/run_subjects.py`,
`tools/claude_call.py`, `bank/text_units.py`. Do not open any other file, and do not run anything (it makes Claude
calls through `claude_call.converse`).

## Purpose
Subjects (bare Claude sessions, empty system prompt) see a text unit's card and say what the unit does (turn 1,
`run_subjects.TASK`), then predict 20 held-out items (turn 2, as `run_subjects.turns_for` builds it). Four arms
differ only in what the subject sees before its turn-1 answer:
- `A0`: nothing more (exactly `run_subjects.turns_for(cid)`).
- `F2_pool`: the card's extra windows (from its key, `pool`), shown before the task.
- `P1_active`: an offer to run experiments: the subject writes texts and sees the unit's activation on every token.
- `Y1_yoked`: the texts and results that the `P1_active` session with the same card and sample produced, shown
  like F2_pool's windows; the subject did not choose them.
Then a separate score: a fresh session sees only the turn-1 explanation and predicts the unit's activation on 40
test items (`S0_sim`).

## Keys and units
- Keys: `bank/KEY_r33.json` and `bank/KEY_olmo.json` (either may be absent), merged; per card id: `unit` (name),
  `format` ("field"), `card_max`, `pool` (list of window texts), `sim` (list of `{"id", "text"}`), and others this
  tool does not use.
- A unit whose name starts with `olmo:` is measured by an HTTP server (`--server`, default
  `http://127.0.0.1:8765`): POST `/acts` with `{"unit": name, "texts": [...]}` returns `{"results": [{"text",
  "offs", "acts"}]}` (one activation per token, null where it does not count). Any other unit is a planted rule:
  import `bank/text_units_r33.py` (module `text_units_r33`, which registers extra rules in `text_units.UNITS`) and
  measure with `text_units.unit_acts(name, [text_units.tok(t) for t in texts])`.
- `measure(key_entry, texts, server)` -> list of `(doc, acts)` with doc in text_units' `(toks, text, offs)` form
  (for the server: toks a list of the same length as offs) and acts a 1-D float array (NaN for null).
- `marked(doc, acts, card_max)` -> `text_units.window(doc, 0, 0, len(doc[0]), marks)` with marks
  `str(min(10, round(a / card_max * 10)))` where that is at least 1, else None (NaN never marked).

## Turn 1 per arm (turn 2 is always turns_for's held-out turn, unchanged)
Let `card_blocks` be turns_for's turn 1 without its last block (the task), and TASK that last block's text.
- A0: turns_for's turn 1.
- F2_pool: `card_blocks + [text(POOL_INTRO + "\n\n" + numbered(pool) + TASK)]`, where `numbered(xs)` is
  `"\n\n".join(f"Text {n}:\n{x}" for n, x in enumerate(xs, 1))` and POOL_INTRO is exactly
  `"\n\nBelow is the same unit on further texts, marked the same way and on the same scale as above."`.
- Y1_yoked: as F2_pool with the list of result texts of the matching P1_active record (all requests, in order)
  instead of pool. Skipped with a message when that record is absent or has no results.
- P1_active: `card_blocks + [text("\n\n" + OFFER + TASK)]`, OFFER the text of `arms/P1_active.txt` (stripped).
  Then a loop on the subject's replies (`claude_call.parse_json` on the last reply; a parse failure counts as {}):
  - a request: a dict with key `show` (a list) while requests remain (REQUESTS = 3): count it, take the first
    PER = 5 entries, each a dict whose `text` is a non-empty string (an entry that is a plain string counts as its
    text; others are dropped); measure them; reply with the marked results as
    `numbered(results) + f"\n\n({left} requests left.)"`; log the whole request dict and the result texts.
  - a dict with key `class` before any request has been made: reply once with exactly `"Run at least one
    experiment before answering. Reply with only a request in the format above."`; a second time it is accepted.
  - a dict with key `class` otherwise: the answer; send turn 2; after turn 2's reply the conversation ends.
  - anything else: reply `f"{left} requests left. Reply with only a request or the JSON answer described above."`
    (when no requests remain: `"Reply with only the JSON answer described above."`); at most 3 such replies in a
    row, then the session fails (not saved).
- Every arm is one `claude_call.converse("", turns)` conversation (callable turns as run_probe does). Record saved
  to `runs/<arm>/<cid>/s<k>.json`: `{"arm", "card", "sample", "model", "requests" (P1: the logged request dicts),
  "results" (P1: result texts in order; Y1, F2: the texts shown), "replies"}`; an existing record is skipped; a
  session with an error reply is not saved (raise, as run_probe).

## Simulator score
- `answer_of(record)` -> the parsed turn-1 answer: the last reply before the held-out turn that parses with a
  `class` key (None if none).
- SIM is exactly: `"Below is a description of what one unit (a single neuron) in a language model responds to, then
  40 text excerpts, each with one token marked ⟦like this⟧. Using only the description, predict the unit's
  activation on each marked token, from 0 (not active) to 10 (as active as it ever gets). Reply with only JSON:
  {\"predictions\": [{\"id\": 1, \"act\": 0}, ...]}"`.
- `sim_turn(explanation, items)` -> `SIM + "\n\nDescription: " + explanation + "\n\n" + "\n\n".join(f"Item
  {it['id']}:\n{it['text']}" for it in items)`.
- For each saved record of the given arms and cards: one fresh `converse("", [[text(sim_turn(...))]])` with the
  answer's `explanation` and the key's `sim` items; saved to `runs/S0_sim/<arm>/<cid>/s<k>.json` (`{"arm", "card",
  "sample", "explanation", "replies"}`); skipped when present or when the record has no answer.

## Command line
`python -I tools/run_text_arms.py --arms A0,F2_pool,P1_active,Y1_yoked --cards C1,C2 [--samples 1] [--workers 4]
[--server URL] [--sim]`: arms run in the order given, each arm's sessions in a thread pool before the next arm
starts (so Y1 follows P1); with `--sim`, run the simulator score for those arms and cards instead. Print one line
per session (done, skip, or FAIL with the error).

## Selftest (`--selftest`)
Without Claude or a server: `marked` on a hand-made doc; the P1 reply loop driven by a fake list of replies
(a request with strings and dicts, a premature answer, a malformed reply, the answer, the turn-2 reply), with a fake
`measure`, checking the texts sent back and the log; Y1 and F2 turn construction; `sim_turn` and `answer_of`.
Prints "selftest ok".

## Constraints
- Match the repo's style (run_probe.py): module docstring with usage, small functions, plain asserts.
- No prompt text other than the strings given above and the files named.
