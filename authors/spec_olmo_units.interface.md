## Units
- A neuron: layer l, index i: the i-th entry of the MLP hidden activation of decoder layer l, i.e. the input of
  `model.model.layers[l].mlp.down_proj` (a forward pre-hook; `act_fn(gate_proj(x)) * up_proj(x)`, width 11008).
- A direction: layer l, unit-norm vector w (length 11008): the dot product of that hidden activation with w.
- `draw_units(n_neurons=16, n_dirs=8, seed=33, width=11008)` -> list of dicts, neurons first then directions:
  `rng = np.random.default_rng(seed)`; for each neuron j: `layer = int(rng.integers(4, 28))`,
  `index = int(rng.integers(0, width))`, name `f"olmo:L{layer}:n{index}"`; then for each direction j:
  `layer = int(rng.integers(4, 28))`, `w = rng.standard_normal(width)` divided by its norm, name
  `f"olmo:L{layer}:d{j}"`. Records: `{"name", "kind": "neuron"|"direction", "layer", "index" or None, "w" or None}`.

## Documents and tokens
- Pile-10k: the `text` column of the parquet file, in row order; document d is row d.
- `tokenize(tokenizer, text, maxtok=1024)` -> `(ids, text, offs)` in text_units' doc convention: `ids` the token ids
  of `tokenizer(text, add_special_tokens=True, return_offsets_mapping=True, truncation=True, max_length=maxtok)`,
  `text` the input text cut at the end offset of the last non-special token, `offs` one start offset per token
  (a special token, whose offset span is empty, gets the start offset of the next token, or len(text) at the end).
  `text_units.window` must work on the result.
- Position 0 and special tokens never count: their activation is stored as NaN, they are never selected, marked or
  shown as values.
- Parts by `zlib.crc32(f"olmo:{d}".encode()) % 20`: 0-9 card docs, 10-12 held-out docs, 13-15 pool docs, 16-19
  test docs.

## Interface
- `extract(model_dir, parquet, units, out, batch=8, maxtok=1024)`: one pass over all documents (batches of
  `batch` documents, right-padded with an attention mask, bf16 on CUDA, no gradients, the forward stopped after the
  deepest layer any unit needs), recording every unit's activation on every token. Writes `out/acts.npz` with one
  float32 array per document holding all units (`doc_<d>`: tokens x units, in units order) and `out/units.json`
  (the unit records, w as a list) and `out/tokens.json` (per document: ids and offs; the text is re-cut from the
  parquet by the last offset). Prints progress per 500 documents.
- `card(cid, docs, acts, card_docs, rng, left=48)` -> `(text, cmax, peaks)`: the card for one unit. `acts`: one
  1-D array per document (NaN at positions that never count). Top examples: the 12 card documents with the highest
  maximum, each at its argmax, strongest first; cmax = the highest of them. Middle examples: up to 8 card documents
  whose maximum is between 0.3 and 0.7 x cmax, chosen with `rng.sample` from those in document order, each at its
  argmax, shown strongest first. Text: exactly `text_units.card_text`'s field format (its header lines, scale note,
  sections, `text_units.window` with 15 tokens after the peak and marks `str(min(10, round(a / cmax * 10)))` where
  that is at least 1). `peaks`: the 12 top examples' peak token strings, each stripped and lowercased.
- `heldout_items(docs, acts, ho_docs, cmax, peaks, rng, left=48)` -> `(items, truth)` as text_units.heldout returns
  them (items `{"id", "text"}` with the token marked by `window(..., target=True)`; truth `{"id", "fires", "tag",
  "act"}`). Plan (fire: act >= 0.3 x cmax; no: act <= 0.05 x cmax; a token's string stripped and lowercased):
  `fire:surface` 5 (string in peaks), `fire:novel` 5 (not in peaks), `no:near` 6 (string in peaks), `no:any` 4
  (any token, sampled as text_units does for 'any'). One item per document where possible, as text_units.heldout.
  When a tag has too few, the shortfall is taken from the other tag of the same kind (fire or no) and the item
  keeps the tag it was found under; when a kind has too few overall, raise SystemExit naming the unit and kind.
  Items shuffled with rng, numbered 1-20.
- `pool_windows(docs, acts, pool_docs, cmax, rng, n_fire=8, n_any=7, left=48)` -> list of window texts: n_fire
  windows each around a token with act >= 0.3 x cmax (one per document), n_any around uniformly chosen counting
  tokens of other documents; each window `text_units.window(doc, i, left, 15, marks)` with the card's marks
  (0-10 scaled to cmax); shuffled with rng.
- `sim_items(docs, acts, test_docs, cmax, rng, left=48)` -> `(items, truth)`: 40 tokens of test documents, at most
  one per document: 10 with act < 0.05 x cmax, then 6 in each act / cmax bin [0.05, 0.2), [0.2, 0.4), [0.4, 0.6),
  [0.6, 0.8), [0.8, inf); a bin's shortfall taken from the next lower bin's tokens. Items `{"id", "text"}` with the
  token marked as in heldout_items, shuffled, numbered 1-40; truth `{"id", "act", "bin"}`.
- `build(parquet, out, model_dir=None)`: loads extract's outputs and, with `--select k` (default 2), takes the first
  k neurons in draw order that are usable (at least 40 card documents with a maximum >= 0.3 x that unit's cmax, and
  heldout_items does not raise), and every direction that is usable. For each taken unit, cid `"O-" +
  str(zlib.crc32(name.encode()) % 900 + 100)`: writes `bank/cards/<cid>.md`, `bank/heldout/<cid>.json` (items), and
  an entry in `bank/KEY_olmo.json` (merged with what is there): `{"unit": name, "kind", "format": "field",
  "card_max", "heldout": truth, "pool": windows, "sim": sim items, "sim_truth": sim truth, "peaks"}`. rng for a
  unit: `random.Random(zlib.crc32(name.encode()))`, used in the order card, heldout_items, pool_windows, sim_items.
  Prints, per drawn unit, usable or why not.
- `UnitServer`: `serve(model_dir, units_json, port)`: loads the model once and serves HTTP on 127.0.0.1:port.
  POST `/acts` with JSON `{"unit": name, "texts": [str, ...]}` -> `{"results": [{"text": cut text, "offs": [...],
  "acts": [float or null, ...]}]}` (tokenize each text as above, one batch, the unit's activation per token, null
  where it never counts). An unknown unit gives HTTP 400.
- `main()`: subcommands `extract --model DIR --parquet P --out DIR [--batch 8]`, `build --parquet P --out DIR
  [--select 2]`, `serve --model DIR --units DIR/units.json [--port 8765]`, and `--selftest`.
- `selftest()`: without a model or the corpus: `draw_units` (determinism, names, unit norms); `tokenize` with a
  tiny fake tokenizer object (a callable returning `input_ids` and `offset_mapping`, one special token first) and
  `text_units.window` on its result; `card`, `heldout_items`, `pool_windows` and `sim_items` on synthetic documents
  and activations (checks: top examples are the highest-maximum documents, marks scale, plan counts and tags, the
  shortfall rule, bins, NaN positions never chosen). Prints "selftest ok".

