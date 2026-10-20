"""Arm P0: bare system prompt, the same card, plus the option to ask for more data (up to 3 requests).

Images: the subject names ImageNet categories and sees 4 card-pool photos of each with activations.
Text: the subject writes texts and sees the unit's activation on every token (the unit's exact rule).
Then the same held-out turn as every other arm.

usage: python -I tools/run_probe.py --cards U-192,U-300,U-421,U-937 --samples 3 [--workers 4]
"""
import argparse
import io
import itertools
import json
import random
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bank'))
from claude_call import converse, parse_json, MODEL
from run_subjects import BANK, J, TASK, img_block, turns_for
import text_units as tu

ARM, REQUESTS, PER = 'P0_probe', 3, 5
KEY = {**json.loads((BANK / 'KEY_text.json').read_text()), **json.loads((BANK / 'KEY_img.json').read_text())}

from torchvision.models import GoogLeNet_Weights
CATS = GoogLeNet_Weights.IMAGENET1K_V1.meta['categories']
_z = np.load(BANK / 'img/acts.npz')
PRE, CLS = _z['maps'].astype(np.float32), _z['classes']
POS, NEG = np.maximum(PRE, 0).reshape(len(PRE), -1).max(1), PRE.reshape(len(PRE), -1).min(1)
NAMES = json.loads((BANK / 'img/names.json').read_text())
CARD_POOL = np.array([zlib.crc32(n.encode()) % 10 < 7 for n in NAMES])

OFFER_IMG = (f'\n\nBefore answering, you may ask to see the unit on more photos, up to {REQUESTS} times. To ask, reply with only '
             f'{{"show": ["category", ...]}} naming up to {PER} of the ImageNet categories listed below; you will get 4 new photos '
             'of each with the unit\'s activation on the same scale as above. When you are ready, reply with the JSON answer '
             'described below instead.\n\nCategories: ' + '; '.join(CATS))
OFFER_TXT = (f'\n\nBefore answering, you may ask to see the unit on more text, up to {REQUESTS} times. To ask, reply with only '
             f'{{"show": ["text", ...]}} with up to {PER} texts you write; you will get the unit\'s activation on every token, '
             'marked on the same scale as above. When you are ready, reply with the JSON answer described below instead.')


def show_images(cid, names, rng):
    k = KEY[cid]
    blocks = []
    for name in names[:PER]:
        hits = [i for i, c in enumerate(CATS) if c.lower() == str(name).strip().lower()]
        if not hits:
            blocks.append({'type': 'text', 'text': f'"{name}": not one of the listed categories.'})
            continue
        pool = list(np.where(CARD_POOL & (CLS == hits[0]))[0])
        pick = rng.sample(pool, min(4, len(pool)))
        g = Image.new('RGB', (4 * 224, 250), (255, 255, 255))
        d = ImageDraw.Draw(g)
        for n, i in enumerate(pick):
            g.paste(Image.open(BANK / 'img/thumbs' / f'{i:05d}.jpg'), (n * 224, 0))
            lab = (f'activation {round(POS[i] / k["card_max"] * 10)}' if k['format'] == 'field'
                   else f'activation {POS[i]:.1f}   most negative {NEG[i]:.1f}')
            d.text((n * 224 + 6, 228), lab, fill=(0, 0, 0))
        buf = io.BytesIO()
        g.save(buf, 'JPEG', quality=88)
        p = J / 'runs' / ARM / cid / f'probe_{zlib.crc32(buf.getvalue()):08x}.jpg'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(buf.getvalue())
        blocks += [{'type': 'text', 'text': f'{CATS[hits[0]]}:'}, img_block(p)]
    return blocks


def show_texts(cid, texts):
    k = KEY[cid]
    docs = [tu.tok(str(t)) for t in texts[:PER]]
    acts = tu.unit_acts(k['unit'], docs)
    out = []
    for n, (doc, (a, _)) in enumerate(zip(docs, acts), 1):
        if k['format'] == 'field':
            v = np.round(a / k['card_max'] * 10).astype(int)
            marks = [str(x) if x >= 1 else None for x in v]
        else:
            marks = [f'{x:.2f}' if x >= 0.02 * k['card_max'] else None for x in a]
        out.append(f'Text {n}:\n' + tu.window(doc, 0, 0, len(doc[0]), marks))
    return [{'type': 'text', 'text': '\n\n'.join(out) or '(no texts)'}]


def one(cid, s):
    out_path = J / 'runs' / ARM / cid / f's{s}.json'
    if out_path.exists():
        return f'skip {cid} s{s}'
    is_img = KEY[cid]['unit'].startswith('inceptionv1')
    t1, t2 = turns_for(cid)
    t1 = t1[:-1] + [{'type': 'text', 'text': (OFFER_IMG if is_img else OFFER_TXT) + TASK}]
    rng, state, log = random.Random(f'{cid}{s}'), {'left': REQUESTS, 'answered': False, 'done': False}, []

    def step(out):
        if state['done']:
            return None
        try:
            msg = parse_json(out[-1]['text'])
        except ValueError:
            msg = {}
        if not state['answered'] and 'show' in msg and state['left'] > 0:
            state['left'] -= 1
            log.append(msg['show'])
            blocks = show_images(cid, msg['show'], rng) if is_img else show_texts(cid, msg['show'])
            return blocks + [{'type': 'text', 'text': f'({state["left"]} requests left.)'}]
        if not state['answered']:
            if 'class' not in msg:
                ask = 'a {"show": [...]} request or ' if state['left'] > 0 else ''
                return [{'type': 'text', 'text': f'{state["left"]} requests left. Reply with only {ask}the JSON answer described above.'}]
            state['answered'] = True
            return t2
        state['done'] = True
        return None

    replies = converse('', itertools.chain([t1], itertools.repeat(step)))
    if any(r['is_error'] for r in replies):
        raise RuntimeError(f'error reply, not saved: {replies[-1]["text"][:80]}')
    rec = {'arm': ARM, 'card': cid, 'sample': s, 'model': MODEL, 'requests': log, 'replies': replies}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return f'done {cid} s{s} requests={log}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cards', required=True)
    ap.add_argument('--samples', type=int, default=3)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    jobs = [(c, s) for s in range(1, a.samples + 1) for c in a.cards.split(',')]
    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(one, *j) for j in jobs]):
            try:
                print(f.result(), flush=True)
            except Exception as e:
                print('FAIL', repr(e)[:200], flush=True)


if __name__ == '__main__':
    main()
