"""Field card and held-out set for any InceptionV1 channel (round 14: the pose example, mixed4d:409).

- Maps: post-ReLU 14x14 map of the channel on every photo (cached in bank/img/maps_{layer}_{ch}.npy).
- Card: the 20 card-pool photos with the highest spatial max, strongest first, whole photos (no crop, so a view or
  framing stays visible), activation scaled 0-10 against the strongest. Same text as the other field cards.
- Held-out: 20 held-pool photos, 5 each of vehicle_fire, dog_fire, vehicle_no, dog_no (fire >= 30% of card max,
  no < 10%, as in the disagreement set), seed 14; where a kind has too few silent photos, dog_no fills the 20. A
  reading by kind alone then scores about 0.5; only within-kind discrimination scores above.

usage: python -I bank/img_cards_ch.py mixed4d 409
"""
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from img_acts import Stop, load_model
from img_cards_v2 import grid

B = Path(__file__).resolve().parent
VEH = {407, 436, 468, 479, 511, 581, 609, 627, 656, 661, 717, 734, 751, 817}
DOG = set(range(151, 269))


def maps(layer, ch, n):
    p = B / 'img' / f'maps_{layer}_{ch}.npy'
    if p.exists():
        return np.load(p)
    torch.set_num_threads(4)
    model, got, out = load_model(B.parent / 'raw/weights/inception5h.pth'), {}, []
    h1 = getattr(model, layer).register_forward_hook(lambda _m, _i, o: got.__setitem__('a', o[:, ch].detach()))
    h2 = model.mixed4e.register_forward_hook(lambda *_: (_ for _ in ()).throw(Stop))
    for k in range(0, n, 64):
        x = np.stack([np.asarray(Image.open(B / 'img/thumbs' / f'{i:05d}.jpg').convert('RGB'), dtype=np.float32)
                      for i in range(k, min(n, k + 64))])
        try:
            with torch.no_grad():
                model(torch.from_numpy(x).permute(0, 3, 1, 2) - 117)
        except Stop:
            pass
        out.append(got['a'].numpy())
    h1.remove(), h2.remove()
    m = np.concatenate(out)
    np.save(p, m)
    return m


def main():
    layer, ch = sys.argv[1], int(sys.argv[2])
    names = json.loads((B / 'img/names.json').read_text())
    cls = np.load(B / 'img/acts.npz')['classes']
    pos = maps(layer, ch, len(names)).reshape(len(names), -1).max(1)
    in_card = np.array([zlib.crc32(n.encode()) % 10 < 7 for n in names])
    card, pool = np.where(in_card)[0], np.where(~in_card)[0]
    top = card[np.argsort(-pos[card])[:20]]
    cmax = float(pos[top[0]])
    cid = f'U-{random.Random(ch).randrange(100, 999)}'
    assert not (B / 'cards' / f'{cid}.md').exists(), cid
    grid([Image.open(B / 'img/thumbs' / f'{i:05d}.jpg').convert('RGB') for i in top],
         [f'activation {round(pos[i] / cmax * 10)}' for i in top], B / 'cards' / f'{cid}_pos.jpg')
    (B / 'cards' / f'{cid}.md').write_text(
        f'# Unit {cid}\n\nRecorded from one unit (a single neuron) in an image classification network, run over several '
        'thousand natural photos.\nThe attached grid shows the 20 photos on which the unit is most active, strongest '
        'first. Under each is its activation, scaled 0-10 relative to the strongest.\n')
    rng, held, spare = random.Random(14), [], []
    for tag, kind in (('vehicle', VEH), ('dog', DOG)):
        m = pool[np.isin(cls[pool], list(kind))]
        for fires, sel in ((True, m[pos[m] >= 0.3 * cmax]), (False, m[pos[m] < 0.1 * cmax])):
            sel = [int(i) for i in sel]
            rng.shuffle(sel)
            held += [{'tag': f'{tag}_{"fire" if fires else "no"}', 'fires': fires, 'thumb': i,
                      'act': round(float(pos[i] / cmax), 3), 'class': int(cls[i])} for i in sel[:5]]
            spare = [{**held[-1], 'thumb': i, 'act': round(float(pos[i] / cmax), 3), 'class': int(cls[i])}
                     for i in sel[5:]] if tag == 'dog' and not fires else spare
    held += spare[:20 - len(held)]
    rng.shuffle(held)
    items = []
    for n, h in enumerate(held, 1):
        h['id'] = n
        Image.open(B / 'img/thumbs' / f'{h["thumb"]:05d}.jpg').save(B / 'heldout' / f'{cid}_{n:02d}.jpg', quality=88)
        items.append({'id': n, 'image': f'{cid}_{n:02d}.jpg'})
    (B / 'heldout' / f'{cid}.json').write_text(json.dumps(items))
    kp = B / 'KEY_img_v6.json'
    key = json.loads(kp.read_text()) if kp.exists() else {}
    key[cid] = {'unit': f'inceptionv1_{layer}_{ch}', 'format': 'field', 'display': 'whole', 'truth_class': None,
                'images': [f'{cid}_pos.jpg'], 'card_max': cmax, 'heldout': held}
    kp.write_text(json.dumps(key, indent=1))
    print(cid, 'card_max', round(cmax, 1), 'top20 dogs', int(np.isin(cls[top], list(DOG)).sum()), 'vehicles',
          int(np.isin(cls[top], list(VEH)).sum()), 'held', {t: sum(h['tag'] == t for h in held) for t in
                                                           ('vehicle_fire', 'dog_fire', 'vehicle_no', 'dog_no')})


if __name__ == '__main__':
    main()
