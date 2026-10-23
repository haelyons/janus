# Spec: tools/vlm_answer.py (the standing questions answered by an open vision-language model)

Write one new file, `tools/vlm_answer.py`. Read only this spec and `tools/proploop.py`, `tools/refineloop.py`,
`tools/pairtable.py`, `tools/bundlemax.py` and `tools/claude_call.py`. Do not open any other file, and do not run
anything. It will run on a GPU machine with `vllm` installed; it must import on a machine without `vllm` (import it
inside `main` only).

## Purpose
The repo's standing questions (`pairtable.load_questions()`) are answered blind for photos by Claude, 12 photos per
call (`refineloop.d_content`, parsed by `proploop.parse_answers`). This tool asks the same questions of an open
vision-language model served by vLLM, one photo per prompt, so that many more photos can be answered.

## Interface
- `build_messages(path, questions)` -> a chat message list for one photo, exactly
  `[{"role": "user", "content": [{"type": "text", "text": "Image 1:"}, {"type": "image_url", "image_url":
  {"url": <data URI>}}, {"type": "text", "text": <prompt>}]}]`, where the data URI is
  `"data:image/jpeg;base64," + base64 of the file's bytes as stored` (the file is a JPEG; it is not re-encoded or
  resized) and `<prompt>` is `proploop.fill(refineloop.A_PROMPT, b=1, questions=proploop.questions_block(questions))`.
- `answer_photos(photos, questions, generate, model)`: `photos` a list of `{"id": str, "path": str}`; `generate`
  a callable `generate(list_of_message_lists, temperature) -> list of reply texts` (same length and order). One
  first pass over all photos at temperature 0.0; then one retry pass, at temperature 0.7, over the photos whose
  first reply did not parse. A reply parses when `proploop.parse_answers(text, 1, questions)` is not None; the
  photo's answer row is then its single entry (a list with one int or None per question). Returns
  `{id: {"answers": row or None, "texts": [reply texts in order], "valid": bool, "model": model}}`, where `answers`
  is None and `valid` false when neither reply parsed.
- `main()`: arguments `--model` (a local directory or model name for vLLM), `--photos` (a JSON file: the list of
  `{"id", "path"}` records), `--out` (the output JSON path), `--max-model-len` (default 8192),
  `--gpu-memory-utilization` (default 0.9). Loads the questions with `pairtable.load_questions()`. If `--out`
  exists, it is loaded and the photos whose id is in it with `valid` true are skipped. Builds
  `vllm.LLM(model=..., max_model_len=..., gpu_memory_utilization=..., limit_mm_per_prompt={"image": 1},
  trust_remote_code=True)` and a `generate` that calls `llm.chat(list_of_message_lists, SamplingParams(temperature=t,
  max_tokens=256, seed=0))` once for the whole list and returns each output's first text. Writes the merged
  result (previous records updated with the new ones) to `--out` as JSON (via a temporary file and rename), and
  prints the number of photos asked, valid after the first pass, valid after the retry.
- `selftest()` (run by `python tools/vlm_answer.py --selftest`): checks `build_messages` and `answer_photos` with a
  fake `generate`, without a model or GPU; prints "selftest ok".

## Constraints
- No other prompt text, no system prompt, nothing added to or removed from the user turn above.
- Model-free functions take everything they use as arguments (no globals read at call time besides constants).
- Match the repo's style: module docstring with usage, small functions, plain asserts in the selftest.
