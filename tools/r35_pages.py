"""Round 35, Inception: blind and reveal pages from runs/r35/img/groups.json (PREDICTIONS.md, round 35).

Blind page per group (code only): rows a-d, a feature visualisation, then the top 5 card-half photos with the argmax
square and the value 0-10 relative to the strongest shown. Reveal page per group: origin, our explanation, labels
(agent, and the human's once runs/r35/img/human.md is filled), unit scores when scored, and per row the 4 strongest
held-out photos and their near misses, valued 0-10 relative to the unit's strongest card-half photo.

usage: python -I tools/r35_pages.py blind|reveal|template
"""
import json
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
import proploop  # noqa: E402

OUT = J / 'runs/r35/img'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
LETTERS = 'abcd'
INK, GREY, MARK = (20, 20, 20), (110, 110, 110), (255, 45, 85)


def font(n, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, n)


def load():
    return json.loads((OUT / 'groups.json').read_text())['groups']


def unit_of(g, name):
    return next(u for u in g['units'] if u['name'] == name)


def photo(i, s):
    return Image.open(proploop.thumb(i)).convert('RGB').resize((s, s), Image.BICUBIC)


def blind(g):
    T, P, L, H = 168, 6, 36, 56
    im = Image.new('RGB', (L + 6 * (T + P), H + 4 * (T + 28)), 'white')
    d = ImageDraw.Draw(im)
    d.text((8, 12), g['code'], font=font(24, True), fill=INK)
    d.text((L + 2, 36), 'synthetic', font=font(14), fill=GREY)
    d.text((L + T + P + 2, 36), 'strongest photos, strongest first', font=font(14), fill=GREY)
    for r, name in enumerate(g['rows']):
        u, y = unit_of(g, name), H + r * (T + 28)
        d.text((10, y + T // 2 - 12), LETTERS[r], font=font(22, True), fill=INK)
        vis = Image.open(OUT / 'vis' / f'{name.replace(":", "_")}.png').convert('RGB').resize((T, T), Image.BICUBIC)
        im.paste(vis, (L, y))
        vals = [proploop_value(u, i) for i in u['top']]
        for k, (i, (x0, y0)) in enumerate(zip(u['top'], u['top_xy'])):
            x = L + (k + 1) * (T + P)
            im.paste(photo(i, T), (x, y))
            s, h = T / 224, 224 / 3 / 2
            d.rectangle([x + (x0 - h) * s, y + (y0 - h) * s, x + (x0 + h) * s, y + (y0 + h) * s], outline=MARK, width=2)
            d.text((x + 2, y + T + 3), f'{10 * vals[k] / vals[0]:.1f}', font=font(14), fill=INK)
    return im


_V = {}


def proploop_value(u, i):
    """the unit's value on photo i (runs/r35/img/values.npy, rows in groups.json order)"""
    if not _V:
        import numpy as np
        V = np.load(OUT / 'values.npy')
        names = [x['name'] for g in load() for x in g['units']]
        _V.update({n: V[k] for k, n in enumerate(names)})
    return float(_V[u['name']][i])


def labels(who):
    """{code: {'a'..'d', 'shared', 'confidence'}} from runs/r35/img/<who>/<code>.json, or human.md"""
    if who == 'human':
        p = OUT / 'human.md'
        return parse_md(p.read_text()) if p.exists() else {}
    return {p.stem: json.loads(p.read_text())['parsed'] for p in sorted((OUT / who).glob('I*.json'))}


def parse_md(text):
    res, code = {}, None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('## '):
            code = s[3:].strip()
            res[code] = {}
        elif code and ':' in s:
            k, v = s.split(':', 1)
            k, v = k.strip(' *-').lower(), v.strip(' *')
            if k in (*LETTERS, 'shared', 'confidence') and v:
                res[code][k] = v
    return res


def scores():
    p = OUT / 'scores.json'
    return json.loads(p.read_text()) if p.exists() else {}


def reveal(g, lab, sc):
    T, P, W, H0 = 118, 5, 380, 40
    text = (g.get('explanation') or 'No explanation: control.') + (f'  Evidence: {g["evidence"]}.' if 'evidence' in g
                                                                    else '')
    lines = textwrap.wrap(text, 150)
    shared = [f'{who}: {lab[who].get(g["code"], {}).get("shared", "-")}' for who in ('agent', 'human')]
    H = H0 + 20 * (len(lines) + len(shared)) + 30
    im = Image.new('RGB', (W + 8 * (T + P) + 10, H + 4 * (T + 26)), 'white')
    d = ImageDraw.Draw(im)
    layers = sorted({u['layer'] for u in g['units']}, key=proploop.LAYERS.index)
    d.text((8, 8), f'{g["code"]}   {g["category"].replace("_", " ")}   {", ".join(layers)}', font=font(22, True),
           fill=INK)
    for k, s in enumerate(lines + shared):
        d.text((8, H0 + 20 * k), s, font=font(14), fill=INK)
    d.text((W, H - 24), 'held-out: strongest 4', font=font(14), fill=GREY)
    d.text((W + 4 * (T + P), H - 24), 'near misses (same class, lowest), in the same order', font=font(14), fill=GREY)
    for r, name in enumerate(g['rows']):
        u, y = unit_of(g, name), H + r * (T + 26)
        row = [f'{LETTERS[r]}  {name}']
        for who in ('agent', 'human'):
            got = lab[who].get(g['code'], {}).get(LETTERS[r])
            s = sc.get(who, {}).get(name)
            if got:
                row += textwrap.wrap(f'{who}: {got}' + (f'  ({s:.2f})' if s is not None else ''), 48)
        d.text((8, y), row[0], font=font(16, True), fill=INK)
        for k, s in enumerate(row[1:7]):
            d.text((8, y + 24 + 16 * k), s, font=font(13), fill=INK)
        for k, i in enumerate(u['pos'] + u['near']):
            x = W + k * (T + P)
            im.paste(photo(i, T), (x, y))
            d.text((x + 2, y + T + 3), f'{10 * proploop_value(u, i) / u["card_max"]:.1f}', font=font(13), fill=INK)
    return im


def main():
    what = sys.argv[1]
    groups = sorted(load(), key=lambda g: g['code'])
    if what == 'template':
        p = OUT / 'human.md'
        assert not p.exists(), p
        p.write_text('Instructions: authors/r35_brief_img.md\n\n' + '\n'.join(
            f'## {g["code"]}\n' + ''.join(f'{x}: \n' for x in (*LETTERS, 'shared', 'confidence')) for g in groups))
        return
    out = OUT / what
    out.mkdir(exist_ok=True)
    lab, sc = {'agent': labels('agent'), 'human': labels('human')}, scores()
    for g in groups:
        (blind(g) if what == 'blind' else reveal(g, lab, sc)).save(out / f'{g["code"]}.png')
    print(what, len(groups), 'pages ->', out)


if __name__ == '__main__':
    main()
