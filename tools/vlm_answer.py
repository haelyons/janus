"""The standing questions answered by an open vision-language model, one photo per prompt (spec:
authors/spec_vlmanswer.md).

The repo's standing questions (pairtable.load_questions()) are asked about every photo of a list, with an open
vision-language model served by vLLM. One prompt per photo, one user turn and no system prompt: the text 'Image 1:',
the photo (its file's bytes as stored, base64 in a JPEG data URI, not re-encoded or resized), then refineloop.A_PROMPT
for 1 image with proploop.questions_block of the questions. A first pass over all photos at temperature 0.0, then one
retry pass at temperature 0.7 over the photos whose first reply does not parse (proploop.parse_answers with 1 image);
at most 256 tokens, seed 0. Per photo: its answer row (an answer index, 0 for A, or None for '?', per question; None
when neither reply parsed), the reply texts in order, valid, and the model. Records already in --out with valid true
are kept and not asked again; the others are asked and replace their old records.

Printed: the number of photos asked, how many of them are valid after the first pass, and after the retry.

usage: python -I tools/vlm_answer.py --model <local dir or model name> --photos <photos.json> --out <answers.json>
           [--max-model-len 8192] [--gpu-memory-utilization 0.9] [--selftest]
"""
import argparse
import base64
import json
import sys
from pathlib import Path

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))  # -I does not add the script dir
import pairtable  # noqa: E402
import proploop  # noqa: E402
import refineloop  # noqa: E402

DATA_URI = 'data:image/jpeg;base64,'
T_FIRST, T_RETRY = 0.0, 0.7
MAX_TOKENS = 256
SEED = 0
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.9


# ---------- messages and answers ----------

def build_messages(path, questions):
    """the chat messages for one photo: one user turn of 'Image 1:', the photo (the file's bytes as stored, base64 in
    a JPEG data URI) and refineloop.A_PROMPT for 1 image with the questions block; nothing else"""
    uri = DATA_URI + base64.b64encode(Path(path).read_bytes()).decode('ascii')
    prompt = proploop.fill(refineloop.A_PROMPT, b=1, questions=proploop.questions_block(questions))
    return [{'role': 'user', 'content': [{'type': 'text', 'text': 'Image 1:'},
                                         {'type': 'image_url', 'image_url': {'url': uri}},
                                         {'type': 'text', 'text': prompt}]}]


def _row(text, questions):
    """the answer row of a one-photo reply: the single entry of proploop.parse_answers(text, 1, questions), None when
    it does not parse"""
    p = proploop.parse_answers(text, 1, questions)
    return None if p is None else p[0]


def answer_photos(photos, questions, generate, model):
    """{id: {'answers': row or None, 'texts': [reply texts in order], 'valid', 'model'}}, in photo order. photos:
    [{'id': str, 'path': str}], ids distinct; generate(list of message lists, temperature) -> one reply text per
    message list, in order. One pass over all photos at T_FIRST, then one at T_RETRY over the photos whose first reply
    did not parse; no call for an empty list"""
    ids = [p['id'] for p in photos]
    assert all(isinstance(i, str) for i in ids) and len(set(ids)) == len(ids), ids
    msgs = [build_messages(p['path'], questions) for p in photos]
    texts, rows = {i: [] for i in ids}, {}
    todo = list(range(len(photos)))
    for t in (T_FIRST, T_RETRY):
        if not todo:
            break
        replies = list(generate([msgs[k] for k in todo], t))
        assert len(replies) == len(todo), (len(replies), len(todo))
        for k, text in zip(todo, replies):
            texts[ids[k]].append(text)
            rows[ids[k]] = _row(text, questions)
        todo = [k for k in todo if rows[ids[k]] is None]
    return {i: {'answers': rows[i], 'texts': texts[i], 'valid': rows[i] is not None, 'model': model} for i in ids}


# ---------- selftest ----------

def selftest():
    import tempfile

    assert (T_FIRST, T_RETRY, MAX_TOKENS, SEED) == (0.0, 0.7, 256, 0)
    assert (MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION) == (8192, 0.9) and DATA_URI == 'data:image/jpeg;base64,'

    # the prompt: refineloop.A_PROMPT for 1 image with the questions block
    qs = [{'question': 'Is there a dog?', 'answers': ['yes', 'no']},
          {'question': 'Where is it?', 'answers': ['indoors', 'outdoors', 'cannot tell']}]
    block = 'Q1. Is there a dog? A) yes B) no\nQ2. Where is it? A) indoors B) outdoors C) cannot tell'
    assert proploop.questions_block(qs) == block
    prompt = refineloop.A_PROMPT.replace('{b}', '1').replace('{questions}', block)
    assert prompt == proploop.fill(refineloop.A_PROMPT, b=1, questions=block)
    assert prompt.startswith('Answer every question below for each of the 1 images above. For each question choose')
    assert '\n\n' + block + '\n\nReply with only JSON: {"images": [{"id": n, "answers": ["A", ...]}]}' in prompt
    assert '{b}' not in prompt and '{questions}' not in prompt

    def ok(ans, i=1):
        return json.dumps({'images': [{'id': i, 'answers': ans}]})

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        photos, by_bytes = [], {}
        for k in range(6):  # bytes of a JPEG's start and end around filler: passed as stored, never decoded
            raw = b'\xff\xd8\xff\xe0' + bytes((37 * k + j) % 256 for j in range(300 + k)) + b'\xff\xd9'
            path = td / f'p{k}.jpg'
            path.write_bytes(raw)
            by_bytes[raw] = f'photo {k}'
            photos.append({'id': f'photo {k}', 'path': str(path)})

        # build_messages: exactly one user turn, the bytes as stored
        raw0 = Path(photos[0]['path']).read_bytes()
        m0 = build_messages(photos[0]['path'], qs)
        assert m0 == [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Image 1:'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(raw0).decode()}},
            {'type': 'text', 'text': prompt}]}], m0
        assert base64.b64decode(m0[0]['content'][1]['image_url']['url'][len(DATA_URI):]) == raw0
        assert build_messages(Path(photos[0]['path']), qs) == m0 and json.loads(json.dumps(m0)) == m0
        assert build_messages(photos[1]['path'], qs)[0]['content'][1] != m0[0]['content'][1]

        # answer_photos with a stand-in for the model: per photo (first reply, retry reply or None when not asked)
        script = {
            'photo 0': (ok(['A', 'b']), None),
            'photo 1': ('no json here', ok(['B', '?'])),
            'photo 2': ('sorry', 'still no'),
            'photo 3': (ok([' ? ', 'c']), None),
            'photo 4': (ok(['A', 'A'], i=2), ok(['A', 'D'])),  # a wrong id, then a letter beyond the answers
            'photo 5': (ok(['A', 'A', 'A']), 'Sure:\n```json\n' + ok(['b', 'B']) + '\n```')}
        want = {'photo 0': [0, 1], 'photo 1': [1, None], 'photo 2': None, 'photo 3': [None, 2], 'photo 4': None,
                'photo 5': [1, 1]}
        calls = []

        def fake(message_lists, temperature):
            got = []
            for ml in message_lists:
                assert len(ml) == 1 and ml[0]['role'] == 'user' and len(ml[0]['content']) == 3, ml
                assert ml[0]['content'][0] == {'type': 'text', 'text': 'Image 1:'}
                url = ml[0]['content'][1]['image_url']['url']
                assert url.startswith(DATA_URI)
                got.append(by_bytes[base64.b64decode(url[len(DATA_URI):])])
            calls.append((temperature, got, message_lists))
            which = {T_FIRST: 0, T_RETRY: 1}[temperature]
            return [script[i][which] for i in got]

        res = answer_photos(photos, qs, fake, 'test-model')
        assert [(t, ids) for t, ids, _ in calls] == [
            (0.0, [p['id'] for p in photos]), (0.7, ['photo 1', 'photo 2', 'photo 4', 'photo 5'])], calls
        assert calls[0][2] == [build_messages(p['path'], qs) for p in photos]
        assert calls[1][2] == [build_messages(photos[k]['path'], qs) for k in (1, 2, 4, 5)]
        assert list(res) == [p['id'] for p in photos]
        for i, row in want.items():
            first, retry = script[i]
            assert res[i] == {'answers': row, 'texts': [first] if retry is None else [first, retry],
                              'valid': row is not None, 'model': 'test-model'}, (i, res[i])
        assert json.loads(json.dumps(res)) == res
        assert sum(r['valid'] and len(r['texts']) == 1 for r in res.values()) == 2
        assert sum(r['valid'] for r in res.values()) == 4

        # every first reply parses: no retry pass; no photos: no call
        calls.clear()
        res2 = answer_photos([photos[3], photos[0]], qs, fake, 'm')
        assert [(t, ids) for t, ids, _ in calls] == [(0.0, ['photo 3', 'photo 0'])], calls
        assert list(res2) == ['photo 3', 'photo 0'] and all(r['valid'] and r['model'] == 'm' for r in res2.values())
        calls.clear()
        assert answer_photos([], qs, fake, 'm') == {} and calls == []

        # a reply list of the wrong length, and repeated ids, stop
        try:
            answer_photos(photos[:2], qs, lambda ml, t: [ok(['A', 'A'])], 'm')
        except AssertionError:
            pass
        else:
            raise AssertionError('a reply list of the wrong length accepted')
        try:
            answer_photos([photos[0], dict(photos[1], id='photo 0')], qs, fake, 'm')
        except AssertionError:
            pass
        else:
            raise AssertionError('repeated ids accepted')
        assert calls == []
    assert 'vllm' not in sys.modules
    print('selftest ok')


# ---------- main ----------

def main():
    if '--selftest' in sys.argv[1:]:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='a local directory or a model name, for vLLM')
    ap.add_argument('--photos', required=True, help='a JSON file: the list of {"id", "path"} records')
    ap.add_argument('--out', required=True, help='the output JSON; its records with valid true are not asked again')
    ap.add_argument('--max-model-len', type=int, default=MAX_MODEL_LEN)
    ap.add_argument('--gpu-memory-utilization', type=float, default=GPU_MEMORY_UTILIZATION)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    questions = pairtable.load_questions()
    photos = json.loads(Path(a.photos).read_text())
    out = Path(a.out)
    prev = json.loads(out.read_text()) if out.exists() else {}
    todo = [p for p in photos if not (p['id'] in prev and prev[p['id']].get('valid') is True)]

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization,
              limit_mm_per_prompt={'image': 1}, trust_remote_code=True)

    def generate(message_lists, temperature):
        outs = llm.chat(message_lists, SamplingParams(temperature=temperature, max_tokens=MAX_TOKENS, seed=SEED))
        return [o.outputs[0].text for o in outs]

    res = answer_photos(todo, questions, generate, a.model)
    merged = dict(prev)
    merged.update(res)
    proploop.save_json(merged, out)  # a temporary file, then a rename
    first = sum(r['valid'] and len(r['texts']) == 1 for r in res.values())
    after = sum(r['valid'] for r in res.values())
    print(f'asked {len(todo)} photos; valid after the first pass {first}, after the retry {after}; -> {out}',
          flush=True)


if __name__ == '__main__':
    main()
