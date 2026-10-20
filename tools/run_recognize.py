"""Arm R0: recognition diagnostic. Same card, empty system prompt; pick a reading from shuffled candidates
(or give your own), then the same held-out turn.

usage: python -I tools/run_recognize.py --cards U-192,U-300 --samples 3
"""
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claude_call import converse, parse_json, MODEL
from run_subjects import J, turns_for

ARM = 'R0_recognize'
READINGS = {
    'furry_faces': 'It responds to the faces of small furry mammals, such as cats, foxes and red pandas.',
    'faces_plus_cars': 'It responds to cat- and fox-like faces, and separately to cars: two unrelated things.',
    'pointed_vs_round': 'It responds to pointed, V-shaped forms (pointed ears and muzzles, the V of a car front), with round forms on its negative side.',
    'fur_texture': 'It responds to soft fur texture.',
    'frontal_pose': 'It responds to subjects facing the camera head-on.',
    'characterful_faces': 'It responds to distinctive, characterful faces.',
}
TASK = '''
---
Which of these best describes what the unit responds to?
{options}
G. None of these (give your own).
Reply with only a JSON object: {{"choice": "A" to "G", "own_reading": "only if G", "why": "at most two sentences", "confidence": a number from 0 to 1}}'''


def one(cid, s):
    out = J / 'runs' / ARM / cid / f's{s}.json'
    if out.exists():
        return f'skip {cid} s{s}'
    keys = list(READINGS)
    random.Random(f'{cid}-{s}').shuffle(keys)
    letters = dict(zip('ABCDEF', keys))
    t1, t2 = turns_for(cid)
    t1 = t1[:-1] + [{'type': 'text', 'text': TASK.format(options='\n'.join(f'{l}. {READINGS[k]}' for l, k in letters.items()))}]
    replies = converse('', [t1, t2])
    if any(r['is_error'] for r in replies):
        raise RuntimeError(f'error reply, not saved: {replies[-1]["text"][:80]}')
    rec = {'arm': ARM, 'card': cid, 'sample': s, 'model': MODEL, 'letters': letters, 'replies': replies}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    ch = parse_json(replies[0]['text']).get('choice')
    return f'done {cid} s{s} choice={ch} -> {letters.get(ch, ch)}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cards', required=True)
    ap.add_argument('--samples', type=int, default=3)
    a = ap.parse_args()
    jobs = [(c, s) for s in range(1, a.samples + 1) for c in a.cards.split(',')]
    with ThreadPoolExecutor(6) as ex:
        for f in as_completed([ex.submit(one, *j) for j in jobs]):
            try:
                print(f.result(), flush=True)
            except Exception as e:
                print('FAIL', repr(e)[:200], flush=True)


if __name__ == '__main__':
    main()
