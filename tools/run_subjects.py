"""Run every (arm, card, sample) as a two-turn bare conversation; skip ones already saved.

usage: python -I tools/run_subjects.py --arms A0_bare,C0_field --cards U-892,U-570 --samples 3 [--workers 6]
"""
import argparse
import base64
import hashlib
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claude_call import converse, parse_json, MODEL

J = Path(__file__).resolve().parent.parent
BANK = J / 'bank'

TASK = '''
---
Task: say what this unit does. Reply with only a JSON object:
{"class": "one" | "one_plus_related" | "unrelated" | "cannot_tell",
 "explanation": "at most two sentences: what the unit responds to",
 "evidence": "at most two sentences: what in the data supports this",
 "missing_data": "what other data would most change your answer, or \\"none\\"",
 "confidence": a number from 0 to 1}
Class meanings: "one" = it responds to one thing; "one_plus_related" = one main thing plus weaker, related extensions; "unrelated" = two or more things with no meaningful connection; "cannot_tell" = this data is not enough to say.'''

TURN2_TEXT = '''Here are 20 new inputs that were not in the data above. In each, one token is marked ⟦like this⟧. Using your explanation, predict whether the unit fires strongly on the marked token (at least 30% of its strongest activation in the data above) or barely or not at all (under 5%). Every item is one or the other.
Reply with only JSON: {"predictions": [{"id": 1, "fires": true}, ...]}'''

TURN2_IMG = '''Here are 20 new photos that were not in the data above, numbered 1 to 20. Using your explanation, predict whether the unit fires strongly on each photo (at least 30% of its strongest activation in the data above) or barely or not at all (under 10%). Every photo is one or the other.
Reply with only JSON: {"predictions": [{"id": 1, "fires": true}, ...]}'''


def img_block(path):
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                        'data': base64.b64encode(Path(path).read_bytes()).decode()}}


def turns_for(cid):
    card = (BANK / 'cards' / f'{cid}.md').read_text()
    items = json.loads((BANK / 'heldout' / f'{cid}.json').read_text())
    imgs = sorted(BANK.glob(f'cards/{cid}_*.jpg'), key=lambda p: (p.stem.endswith('neg'), p.name))
    t1 = [{'type': 'text', 'text': card}] + [img_block(p) for p in imgs] + [{'type': 'text', 'text': TASK}]
    if 'image' in items[0]:
        t2 = [{'type': 'text', 'text': TURN2_IMG}]
        for it in items:
            t2 += [{'type': 'text', 'text': f'Photo {it["id"]}:'}, img_block(BANK / 'heldout' / it['image'])]
    else:
        body = '\n\n'.join(f'Item {it["id"]}:\n{it["text"]}' for it in items)
        t2 = [{'type': 'text', 'text': TURN2_TEXT + '\n\n' + body}]
    dis = BANK / 'heldout' / f'{cid}_dis.json'
    if not dis.exists():
        return [t1, t2]
    t3 = [{'type': 'text', 'text': TURN2_IMG.replace('20 new photos', '20 more new photos')}]
    for it in json.loads(dis.read_text()):
        t3 += [{'type': 'text', 'text': f'Photo {it["id"]}:'}, img_block(BANK / 'heldout' / it['image'])]
    return [t1, t2, t3]


def one(arm, cid, k):
    out = J / 'runs' / arm / cid / f's{k}.json'
    if out.exists():
        return f'skip {arm} {cid} s{k}'
    sp = (J / 'arms' / f'{arm}.txt').read_text().strip()
    replies = converse(sp, turns_for(cid))
    if any(r['is_error'] for r in replies):
        raise RuntimeError(f'error reply, not saved: {replies[-1]["text"][:80]}')
    rec ={'arm': arm, 'card': cid, 'sample': k, 'model': MODEL,
           'system_prompt_sha256': hashlib.sha256(sp.encode()).hexdigest(), 'replies': replies}
    for n, r in enumerate(replies, 1):
        try:
            rec[f'turn{n}'] = parse_json(r['text'])
        except Exception as e:
            rec[f'turn{n}_parse_error'] = repr(e)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    c = rec.get('turn1', {}).get('class')
    return f'done {arm} {cid} s{k} class={c} cost={sum(r["cost_usd"] or 0 for r in replies):.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', required=True)
    ap.add_argument('--cards', required=True)
    ap.add_argument('--samples', type=int, default=3)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    jobs = [(arm, cid, k) for k in range(1, a.samples + 1) for cid in a.cards.split(',') for arm in a.arms.split(',')]
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, *j): j for j in jobs}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception:
                print('FAIL', futs[f], traceback.format_exc().splitlines()[-1], flush=True)


if __name__ == '__main__':
    main()
