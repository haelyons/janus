"""Arm E0: bare system prompt, the same card, plus experiments: before answering, the subject may edit grid photos
(occlude, keep, zoom, flip, gray, rotate, shrink) and see the unit's activation on each edited photo, up to 3 requests
of up to 8 edits. Then the same held-out turns as every other arm (turn 2, and turn 3 where the card has one).

Box coordinates are fractions of the photo as shown in the grid (a crop around the hotspot); edits are applied to the
whole photo the network sees, and the subject is shown that whole edited photo.

usage: python -I tools/run_edit.py --cards U-121,U-686 --samples 3 [--workers 3]
"""
import argparse
import io
import itertools
import json
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bank'))
from claude_call import converse, parse_json, MODEL
from run_subjects import BANK, J, TASK, img_block, turns_for
from img_acts import load_model, run_batch

ARM, REQUESTS, PER = 'E0_edit', 3, 8
KEY = {**json.loads((BANK / 'KEY_img_v2.json').read_text()), **json.loads((BANK / 'KEY_img_v5.json').read_text())}
_z = np.load(BANK / 'img/acts.npz')
PRE = _z['maps'].astype(np.float32)
NAMES = json.loads((BANK / 'img/names.json').read_text())
CARD = np.where(np.array([zlib.crc32(n.encode()) % 10 < 7 for n in NAMES]))[0]
MODEL_NET, LOCK = load_model(J / 'raw/weights/inception5h.pth'), threading.Lock()
GREY = (128, 128, 128)

OFFER = (f'\n\nBefore answering, you may run experiments on the unit, up to {REQUESTS} times. To do so, reply with only '
         f'{{"edits": [...]}} listing up to {PER} edits of photos in the grid; you will see each edited photo exactly as the '
         'network sees it, with the unit\'s activation on the same 0-10 scale (one decimal). Each edit is '
         '{"photo": n, "op": ..., ...} where n is the grid number and op is one of:\n'
         '- "occlude" with "box": [left, top, right, bottom]: grey out that box\n'
         '- "keep" with "box": grey out everything outside that box\n'
         '- "zoom" with "box": crop to that box and enlarge it to the full frame\n'
         '- "flip": mirror left-right; "gray": remove colour\n'
         '- "rotate" with "degrees"\n'
         '- "shrink" with "factor" between 0.2 and 1: shrink the photo onto a grey frame\n'
         'Boxes are fractions (0-1) of the photo as shown in the grid. When you are ready, reply with the JSON answer '
         'described below instead.')


def grid_ids(cid):
    """The photos in the card's grid, in grid order, with each crop box (in whole-photo pixels)."""
    pos = np.maximum(PRE, 0).reshape(len(PRE), -1).max(1)
    if KEY[cid]['format'] == 'mixed':
        from img_cards import CARS, CATS
        cls = _z['classes']
        top = lambda s: CARD[np.isin(cls[CARD], list(s))][np.argsort(-pos[CARD[np.isin(cls[CARD], list(s))]])[:10]]
        ids = np.concatenate([top(CATS), top(CARS)])
        ids = ids[np.argsort(-pos[ids])]
    else:
        ids = CARD[np.argsort(-pos[CARD])[:20]]
    boxes = []
    for i in ids:
        cy, cx = [v * 16 + 8 for v in np.unravel_index(PRE[i].argmax(), (14, 14))]
        l, t = min(max(cx - 64, 0), 96), min(max(cy - 64, 0), 96)
        boxes.append((l, t))
    return list(ids), boxes


def apply(im, crop, e):
    l, t = crop
    op = e.get('op')
    if op in ('occlude', 'keep', 'zoom'):
        b = [float(v) for v in e['box']]
        x0, y0, x1, y1 = l + b[0] * 128, t + b[1] * 128, l + b[2] * 128, t + b[3] * 128
        if op == 'occlude':
            im = im.copy()
            ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], fill=GREY)
        elif op == 'keep':
            bg = Image.new('RGB', im.size, GREY)
            bg.paste(im.crop((int(x0), int(y0), int(x1), int(y1))), (int(x0), int(y0)))
            im = bg
        else:
            im = im.crop((int(x0), int(y0), int(x1), int(y1))).resize((224, 224), Image.BICUBIC)
    elif op == 'flip':
        im = ImageOps.mirror(im)
    elif op == 'gray':
        im = ImageOps.grayscale(im).convert('RGB')
    elif op == 'rotate':
        im = im.rotate(float(e['degrees']), resample=Image.BICUBIC, fillcolor=GREY)
    elif op == 'shrink':
        f = min(max(float(e['factor']), 0.2), 1.0)
        s = int(224 * f)
        bg = Image.new('RGB', (224, 224), GREY)
        bg.paste(im.resize((s, s), Image.BICUBIC), ((224 - s) // 2, (224 - s) // 2))
        im = bg
    else:
        raise ValueError(f'unknown op {op!r}')
    return im


def run_edits(cid, edits, s, n_req):
    ids, crops = grid_ids(cid)
    cmax = KEY[cid]['card_max']
    out, ims, labs = [], [], []
    for k, e in enumerate(edits[:PER], 1):
        try:
            n = int(e['photo'])
            im = apply(Image.open(BANK / 'img/thumbs' / f'{ids[n - 1]:05d}.jpg').convert('RGB'), crops[n - 1], e)
        except Exception as ex:
            out.append({'type': 'text', 'text': f'Edit {k}: could not apply ({ex}).'})
            continue
        ims.append(im)
        labs.append((k, n, e))
    if ims:
        with LOCK, torch.no_grad():
            acts = run_batch(MODEL_NET, ims)
        for (k, n, e), im, a in zip(labs, ims, acts):
            v = max(float(a.max()), 0) / cmax * 10
            p = J / 'runs' / ARM / cid / f's{s}_r{n_req}_e{k}.jpg'
            p.parent.mkdir(parents=True, exist_ok=True)
            im.save(p, quality=88)
            desc = {x: y for x, y in e.items() if x != 'photo'}
            out += [{'type': 'text', 'text': f'Edit {k}: photo {n}, {json.dumps(desc)}: activation {v:.1f}'}, img_block(p)]
    return out


def one(cid, s):
    out_path = J / 'runs' / ARM / cid / f's{s}.json'
    if out_path.exists():
        return f'skip {cid} s{s}'
    turns = turns_for(cid)
    t1 = turns[0][:-1] + [{'type': 'text', 'text': OFFER + TASK}]
    later = list(turns[1:])
    state, log = {'left': REQUESTS, 'answered': False}, []

    def step(out):
        try:
            msg = parse_json(out[-1]['text'])
        except ValueError:
            msg = {}
        if not state['answered'] and 'edits' in msg and state['left'] > 0:
            state['left'] -= 1
            log.append(msg['edits'])
            return run_edits(cid, msg['edits'], s, REQUESTS - state['left']) + \
                [{'type': 'text', 'text': f'({state["left"]} requests left.)'}]
        if not state['answered']:
            if 'class' not in msg:
                ask = 'an {"edits": [...]} request or ' if state['left'] > 0 else ''
                return [{'type': 'text', 'text': f'{state["left"]} requests left. Reply with only {ask}the JSON answer described above.'}]
            state['answered'] = True
        return later.pop(0) if later else None

    replies = converse('', itertools.chain([t1], itertools.repeat(step)))
    if any(r['is_error'] for r in replies):
        raise RuntimeError(f'error reply, not saved: {replies[-1]["text"][:80]}')
    first = len(replies) - len(turns)  # the answer and the held-out turns are the last replies
    rec = {'arm': ARM, 'card': cid, 'sample': s, 'model': MODEL, 'requests': log, 'replies': replies,
           'turn_idx': {f'turn{k + 1}': first + k for k in range(len(turns))}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return f'done {cid} s{s} requests={len(log)} replies={len(replies)}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cards', required=True)
    ap.add_argument('--samples', type=int, default=3)
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    jobs = [(c, s) for s in range(1, a.samples + 1) for c in a.cards.split(',')]
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, *j): j for j in jobs}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:
                print('FAIL', futs[f], repr(e)[:200], flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
