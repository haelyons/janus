"""Arm F0: failure feedback. Answer, predict a random feedback batch, see the outcomes and revise, then the
usual held-out turn. Empty system prompt.

usage: python -I tools/run_feedback.py --cards U-192,U-300 --samples 3
"""
import argparse
import json
import random
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claude_call import converse, parse_json, MODEL
from run_subjects import BANK, J, TURN2_IMG, img_block, turns_for

ARM = 'F0_feedback'
KEY = json.loads((BANK / 'KEY_img.json').read_text())
_z = np.load(BANK / 'img/acts.npz')
PRE = _z['maps'].astype(np.float32)
POS, NEG = np.maximum(PRE, 0).reshape(len(PRE), -1).max(1), PRE.reshape(len(PRE), -1).min(1)
NAMES = json.loads((BANK / 'img/names.json').read_text())
CARD = np.where(np.array([zlib.crc32(n.encode()) % 10 < 7 for n in NAMES]))[0]
SHOWN = set(CARD[np.argsort(-POS[CARD])[:20]]) | set(CARD[np.argsort(NEG[CARD])[:20]])


def feedback_batch(cid):
    mx, rng = KEY[cid]['card_max'], random.Random(f'feedback-{cid}')
    pool = [i for i in CARD if i not in SHOWN]
    fire = rng.sample([i for i in pool if POS[i] >= 0.3 * mx], 10)
    no = rng.sample([i for i in pool if POS[i] <= 0.1 * mx], 10)
    batch = fire + no
    rng.shuffle(batch)
    return batch


def one(cid, s):
    out = J / 'runs' / ARM / cid / f's{s}.json'
    if out.exists():
        return f'skip {cid} s{s}'
    t1, t_held = turns_for(cid)
    batch, mx, fmt = feedback_batch(cid), KEY[cid]['card_max'], KEY[cid]['format']
    t2 = [{'type': 'text', 'text': TURN2_IMG}]
    for n, i in enumerate(batch, 1):
        t2 += [{'type': 'text', 'text': f'Photo {n}:'}, img_block(BANK / 'img/thumbs' / f'{i:05d}.jpg')]
    truth = [bool(POS[i] >= 0.3 * mx) for i in batch]

    def reveal(replies):
        preds = {p.get('id'): p.get('fires') for p in parse_json(replies[-1]['text']).get('predictions', [])}
        right = sum(preds.get(n) == t for n, t in enumerate(truth, 1))
        val = (lambda i: f'activation {round(POS[i] / mx * 10)}') if fmt == 'field' else (lambda i: f'activation {POS[i]:.1f}')
        lines = [f'Photo {n}: {"fired strongly" if t else "barely fired"} ({val(i)}); you predicted '
                 f'{"fires" if preds.get(n) else "barely"}.' for n, (i, t) in enumerate(zip(batch, truth), 1)]
        return [{'type': 'text', 'text': f'Here is what the unit actually did on those 20 photos. You got {right} of 20 right.\n'
                 + '\n'.join(lines) + '\n\nYou may revise your answer about what the unit responds to. '
                 'Reply with only the same JSON object as your first answer.'}]

    replies = converse('', [t1, t2, reveal, t_held])
    if any(r['is_error'] for r in replies):
        raise RuntimeError(f'error reply, not saved: {replies[-1]["text"][:80]}')
    rec = {'arm': ARM, 'card': cid, 'sample': s, 'model': MODEL, 'feedback_batch': [int(i) for i in batch],
           'feedback_truth': truth, 'replies': replies}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    a, b = parse_json(replies[0]['text']), parse_json(replies[2]['text'])
    return f'done {cid} s{s} {a.get("class")}({a.get("confidence")}) -> {b.get("class")}({b.get("confidence")})'


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
