# Spec: tools/r34_record.py (a language model's internal state at one readout token, for a list of sentences)

Write one new file, `tools/r34_record.py`. Read only this spec. Do not open any other file, and do not run
anything. It runs on a CUDA machine with `torch` and `transformers` 4.57 and OLMo-2-1124-7B (a local directory);
import torch and transformers inside functions, so the module imports without them.

## Purpose
For each sentence, run the model once on the sentence followed by a fixed suffix, and record at the readout token
(the last suffix token) and at the sentence's own last token: every layer's MLP hidden activation, the residual stream
before and after each layer and between attention and MLP, and the attention weights from the readout token.

## Model facts (OLMo-2)
`model.model.layers[l]` (l = 0..L-1) is a decoder layer. Its input is the residual stream `pre[l]`. It computes
`mid = pre + post_attention_layernorm(self_attn(pre))`, then `post = mid + post_feedforward_layernorm(mlp(mid))`;
there is no norm before attention or MLP inputs. So the input of `layers[l].mlp` is `mid[l]`, the layer's output is
`post[l]`, and the MLP hidden activation is the input of `layers[l].mlp.down_proj` (width 11008). `pre[0]` is the
token embedding.

## Interface
- `build_ids(tokenizer, text, suffix="\n")` -> `(ids, n_text)`: `tokenizer(text, add_special_tokens=True)
  ["input_ids"]` followed by `tokenizer(suffix, add_special_tokens=False)["input_ids"]` (token ids concatenated, so
  the suffix tokens are the same for every sentence); n_text = the number of ids before the suffix. Readout positions:
  `len(ids) - 1` ("suffix") and `n_text - 1` ("text").
- `record(model, tokenizer, sentences, suffix="\n", device="cuda")` -> dict of numpy arrays, one sentence per forward
  pass (no padding), bf16 weights, no gradients, `attn_implementation="eager"` so attention weights exist
  (`output_attentions=True`). For each sentence and each readout position r in ("suffix", "text"):
  `hidden_<r>` [n, L, 11008], `pre_<r>` [n, L, d], `mid_<r>` [n, L, d], `post_<r>` [n, L, d] (all float32);
  `attn_<r>`: per sentence a float32 array [L, heads, len(ids)] of attention weights from position r (zeros past r).
  Also `ids` (per sentence) and `n_text`. Hooks: forward pre-hooks on `layers[l]` (pre), on `layers[l].mlp` (mid),
  on `layers[l].mlp.down_proj` (hidden), forward hooks on `layers[l]` (post; the first element when the output is a
  tuple). On the first sentence, assert that `post[l]` at every position equals the model's
  `output_hidden_states=True` hidden state `l + 1` for l < L-1 (tolerance 1e-2 relative), and `pre[0]` equals hidden
  state 0.
- `decompose(pre, mid, post)` -> `(attn_write, mlp_write)`: `mid - pre` and `post - mid` (same shapes).
- `save(out, rec, sentences)`: `out/record.npz` (the fixed-shape arrays; `attn_<r>` and `ids` as object arrays) and
  `out/sentences.json` (the sentence records as given, plus per sentence the token strings of its ids).
- `main()`: `--model DIR --stimuli FILE --out DIR [--suffix "\n"] [--device cuda]`. The stimuli file is a JSON list
  of records with at least `sid` and `text`; they are recorded in file order.
- `selftest()` (`--selftest`): without a model: `build_ids` with a tiny fake tokenizer (a callable returning
  `input_ids`); `decompose` on small arrays; prints "selftest ok".

## Constraints
- Match the repo's style: module docstring with usage, small functions, plain asserts. No statistics here.
