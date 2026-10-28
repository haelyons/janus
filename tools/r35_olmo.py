"""Round 35, OLMo: the sealed card list, blind pages and reveal pages for round 33's 14 cards (PREDICTIONS.md, round 35).

Blind page per card (code only): the card's examples as round 33's A0 saw them (its two sections), marked tokens
shaded by activation with the 0-10 value; the header and intro are dropped (they carry the ID; the brief,
authors/r35_brief_txt.md, carries the rest). Reveal page per card: origin, the A0 and P1 labels of the first counted
sample with their held-out accuracy and simulator rho, the human's label when runs/r35/txt/human.md is filled, and
the 20 held-out items (target token boxed) with their truth and activation (0-10 of the card's strongest).

usage: python -I tools/r35_olmo.py seal|blind|reveal|template
"""
import json
import re
import sys
import textwrap
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
import r33_check  # noqa: E402
import r33f_check  # noqa: E402
import run_text_arms  # noqa: E402

OUT = J / 'runs/r35/txt'
MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
SANS = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
W, CW, LH, M = 1100, 9, 22, 16  # page width, mono char width (15 px font), line height, margin
INK, GREY = (20, 20, 20), (110, 110, 110)
MARK = re.compile(r'«(.+?)\|(\d+)»|⟦(.+?)⟧', re.S)


def font(path, n):
    return ImageFont.truetype(path, n)


def seal():
    K = r33_check.keys()
    cards = list(r33f_check.CARDS)
    codes = np.random.default_rng(zlib.crc32(b'r35:codes:txt')).permutation(len(cards))
    res = []
    for cid, c in zip(cards, codes):
        k, s = K[cid], r33f_check.samples(cid)[0]
        labs = {}
        for arm in ('A0', 'P1_active'):
            rec = json.loads((J / 'runs' / arm / cid / f's{s}.json').read_text())
            sim = json.loads((J / 'runs/S0_sim' / arm / cid / f's{s}.json').read_text())
            sc = r33_check.score(rec, sim, k)
            labs[arm] = {'explanation': run_text_arms.answer_of(rec)['explanation'], 'heldout_acc': sc['acc'],
                         'sim_rho': sc['rho']}
        if cid.startswith('U-'):
            origin = f'planted rule (exact, over GPT-2 tokens): {k["rule"]}'
        else:
            m = re.fullmatch(r'olmo:L(\d+):([nd])(\d+)', k['unit'])
            origin = (f'OLMo-2 7B MLP neuron, layer {m[1]}, index {m[3]} (drawn at random)' if m[2] == 'n' else
                      f'OLMo-2 7B random direction in layer {m[1]}\'s MLP space')
        res.append({'code': f'T{int(c) + 1:02d}', 'card': cid, 'sample': s, 'origin': origin, 'labels': labs,
                    'card_max': float(k['card_max']), 'heldout': k['heldout']})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cards.json').write_text(json.dumps(sorted(res, key=lambda r: r['code']), indent=1, ensure_ascii=False))
    for r in sorted(res, key=lambda r: r['code']):
        print(r['code'], r['card'], r['origin'])


def layout(d, text, x0, y, act_max=10.0, box=False):
    """draw text with «tok|n» shaded (n of act_max) or ⟦tok⟧ boxed, wrapped at the page width; returns the next y"""
    f, fs = font(MONO, 15), font(SANS, 10)
    x, right = x0, W - M
    pos = 0
    pieces = []
    for m in MARK.finditer(text):
        pieces.append((text[pos:m.start()], None))
        pieces.append((m[1].replace('\n', '↵'), int(m[2])) if m[1] is not None else (m[3].replace('\n', '↵'), -1))
        pos = m.end()
    pieces.append((text[pos:], None))
    for s, a in pieces:
        for part in re.split(r'(\n|[^\S\n]+)', s) if a is None else [s]:
            if not part:
                continue
            if part == '\n':
                x, y = x0, y + LH
                continue
            w = len(part) * CW + (14 if a is not None and a >= 0 else 0)
            if x + w > right and x > x0:
                x, y = x0, y + LH
                if a is None and not part.strip():
                    continue
            if a is not None and a >= 0:
                t = a / act_max
                d.rectangle([x, y, x + len(part) * CW, y + LH - 3],
                            fill=(255, int(255 - 120 * t), int(255 - 220 * t)))
                d.text((x, y + 1), part, font=f, fill=INK)
                d.text((x + len(part) * CW + 1, y + 7), str(a), font=fs, fill=GREY)
            elif a == -1:
                d.rectangle([x - 1, y, x + len(part) * CW + 1, y + LH - 3], outline=(255, 45, 85), width=2)
                d.text((x, y + 1), part, font=f, fill=INK)
            else:
                d.text((x, y + 1), part, font=f, fill=INK)
            x += w
    return y + LH


def page(blocks, height=12000):
    im = Image.new('RGB', (W, height), 'white')
    d = ImageDraw.Draw(im)
    y = M
    for kind, text in blocks:
        if kind == 'title':
            d.text((M, y), text, font=font(BOLD, 24), fill=INK)
            y += 38
        elif kind == 'head':
            y += 8
            d.text((M, y), text, font=font(BOLD, 15), fill=GREY)
            y += 26
        elif kind == 'note':
            for line in textwrap.wrap(text, 120):
                d.text((M, y), line, font=font(SANS, 14), fill=INK)
                y += 20
            y += 4
        else:
            y = layout(d, text, M + 12, y) + 10
    return im.crop((0, 0, W, y + M))


def blind(r):
    text = (J / 'bank/cards' / f'{r["card"]}.md').read_text()
    blocks = [('title', r['code'])]
    for sec in re.split(r'^## ', text, flags=re.M)[1:]:
        head, body = sec.split('\n', 1)
        blocks.append(('head', head.strip()))
        for ex in re.split(r'^### (?:Middle e|E)xample \d+\n', body, flags=re.M)[1:]:
            blocks.append(('text', ex.strip()))
    return page(blocks)


def human():
    p = OUT / 'human.md'
    if not p.exists():
        return {}
    res, code = {}, None
    for line in p.read_text().splitlines():
        s = line.strip()
        if s.startswith('## '):
            code = s[3:].strip()
            res[code] = {}
        elif code and ':' in s and s.split(':', 1)[0].strip() in ('label', 'confidence'):
            k, v = s.split(':', 1)
            if v.strip():
                res[code][k.strip()] = v.strip()
    return res


def reveal(r, hum, hsc):
    items = {it['id']: it['text'] for it in json.loads((J / 'bank/heldout' / f'{r["card"]}.json').read_text())}
    blocks = [('title', f'{r["code"]}   {r["card"]}'), ('note', f'Origin: {r["origin"]}.')]
    for arm, name in (('A0', 'agent, card only'), ('P1_active', 'agent, after its own experiments')):
        L = r['labels'][arm]
        rho = 'n/a' if L['sim_rho'] is None else f'{L["sim_rho"]:.2f}'
        blocks.append(('note', f'{name} (held-out {L["heldout_acc"]:.2f}, simulator rho {rho}): {L["explanation"]}'))
    h = hum.get(r['code'], {})
    if h.get('label'):
        s = hsc.get(r['code'])
        blocks.append(('note', 'you' + (f' (simulator rho {s:.2f})' if s is not None else '') + f': {h["label"]}'))
    blocks.append(('head', 'held-out items: boxed token, truth, activation 0-10 of the card\'s strongest'))
    for t in r['heldout']:
        tag = f'{"FIRES" if t["fires"] else "quiet"} {10 * t["act"] / r["card_max"]:.1f}  ({t["tag"]})'
        blocks += [('note', f'item {t["id"]}: {tag}'), ('text', items[t['id']])]
    return page(blocks)


def main():
    what = sys.argv[1]
    if what == 'seal':
        return seal()
    cards = json.loads((OUT / 'cards.json').read_text())
    if what == 'template':
        p = OUT / 'human.md'
        assert not p.exists(), p
        p.write_text('Instructions: authors/r35_brief_txt.md\n\n' +
                     '\n'.join(f'## {r["code"]}\nlabel: \nconfidence: \n' for r in cards))
        return
    out = OUT / what
    out.mkdir(exist_ok=True)
    sp = OUT / 'scores.json'
    hsc = json.loads(sp.read_text()).get('human', {}) if sp.exists() else {}
    for r in cards:
        (blind(r) if what == 'blind' else reveal(r, human(), hsc)).save(out / f'{r["code"]}.png')
    print(what, len(cards), 'pages ->', out)


if __name__ == '__main__':
    main()
