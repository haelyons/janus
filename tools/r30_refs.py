"""Round 30 reference candidates: units with an outside description, none examined in rounds 1-29, and the blind
mapping brief (labels only, no unit names). Written before any round-30 data exists.

Sources:
- Nick Cammarata's car-circuit map (https://already-clean.goodfire.com/assets/car-path/archive/data/
  grouped_car_relp_circuit.json, saved as authors/ext/car_relp_circuit.json): audited labels of InceptionV1 units.
- Distill Zoom In, "Oriented Dog Head Detection": mixed4a 234, 308, 325, 222 (authors/literature_ids.md).
- Distill Equivariance (https://distill.pub/2020/circuits/equivariance/, saved as authors/ext/equivariance.html),
  ids read from its figures: flip pair "dog snout detectors" mixed4a 222/308; flip pair "human beside animal"
  mixed4d 429/427; "miscellaneous" figure (snout length, human vs dog) mixed4b 411, 391, 434 (which is which is not
  recoverable from the page; its "perspective" pair is from a Places model and is not used).

usage: python -I tools/r30_refs.py
"""
import json
import random
import re
import zlib
from pathlib import Path

J = Path(__file__).resolve().parent.parent
NICK = J / 'authors/ext/car_relp_circuit.json'
OUT = J / 'authors/round30_candidates.json'
BRIEF = J / 'authors/brief_round30_mapping.md'
KEY = J / 'authors/round30_mapping_key.json'
LAYERS = ('mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b')
SEED = zlib.crc32(b'round30')

# where a unit counts as examined: any mention in these (literature_ids.md only lists the literature, so not it)
SCAN = ['PREDICTIONS.md', 'README.md', 'authors/*.json', 'authors/*.md', 'tools/*.py', 'tests/*.py']
SCAN_SKIP = {'authors/literature_ids.md', 'authors/round30_candidates.json', 'authors/brief_round30_mapping.md',
             'authors/round30_mapping_key.json', 'tools/r30_refs.py'}
UNIT_RE = re.compile(r'(?<![\w])(?:mixed)?(3[ab]|4[a-e]|5[ab])[:_](\d+)((?:\s*[/,]\s*\d+)*)')

ZOOMIN_4A = ('Detects dog heads in one orientation. It belongs to two mirrored pathways of units, one detecting dog '
             'heads facing to the left and the other dog heads facing to the right; which of the two this unit belongs '
             'to is not stated.')
SNOUT_PAIR = ('Detects dog snouts or heads in one orientation; another unit is a horizontally flipped version of it. '
              'Which way this unit faces is not stated.')
HUMAN_ANIMAL = ('Detects a human beside an animal. Another unit is its horizontally flipped version: one responds to '
                'the human on one side and the animal on the other, its counterpart to the reverse arrangement. Which '
                'arrangement this unit prefers is not stated.')
VARIANTS = ('Detects dog heads. It belongs to a set of dog-head units that come in variants of the same feature, such '
            'as short-snouted versus long-snouted dogs, or a human versus a dog version; which variant this unit is '
            'is not stated.')
LIT = [('mixed4a:234', 'zoomin_4a', ZOOMIN_4A), ('mixed4a:325', 'zoomin_4a', ZOOMIN_4A),
       ('mixed4a:308', 'zoomin_4a', ZOOMIN_4A + ' ' + SNOUT_PAIR), ('mixed4a:222', 'zoomin_4a', ZOOMIN_4A + ' ' + SNOUT_PAIR),
       ('mixed4d:429', 'equiv_human_animal', HUMAN_ANIMAL), ('mixed4d:427', 'equiv_human_animal', HUMAN_ANIMAL),
       ('mixed4b:411', 'equiv_variants', VARIANTS), ('mixed4b:391', 'equiv_variants', VARIANTS),
       ('mixed4b:434', 'equiv_variants', VARIANTS)]


def examined():
    """every unit mentioned in the scanned files, with continuations like 4b:443/431 or mixed4c:254,407 expanded"""
    seen = set()
    for pat in SCAN:
        for f in sorted(J.glob(pat)):
            if str(f.relative_to(J)) in SCAN_SKIP:
                continue
            for m in UNIT_RE.finditer(f.read_text(errors='replace')):
                layer = 'mixed' + m.group(1)
                for c in [m.group(2)] + re.findall(r'\d+', m.group(3)):
                    seen.add(f'{layer}:{int(c)}')
    for d in (J / 'runs').glob('*/*'):
        m = re.fullmatch(r'(mixed(?:3[ab]|4[a-e]|5[ab])):(\d+)', d.name)
        if d.is_dir() and m:
            seen.add(f'{m.group(1)}:{int(m.group(2))}')
    return seen


def nick_labels():
    """{unit: (label, truncated)} from the car-circuit map, the longest label seen for each unit"""
    d = json.loads(NICK.read_text())
    got = {}
    tops = [n for g in d['groups'] for n in g['top_neurons']] + [n for f in d['factors'] for n in f['top_neurons']]
    for n in tops:
        layer, _, c = n['id'].partition('#')
        if layer not in LAYERS:
            continue
        u = f'{layer}:{int(c)}'
        lab = n['label'].strip()
        trunc = lab.endswith('...')
        lab = lab[:-3].rstrip() if trunc else lab
        if u not in got or len(lab) > len(got[u][0]):
            got[u] = (lab, trunc)
    return got


def main():
    seen = examined()
    cands = [{'unit': u, 'source': s, 'label': lab, 'truncated': False} for u, s, lab in LIT]
    lit_units = {c['unit'] for c in cands}
    for u, (lab, trunc) in sorted(nick_labels().items()):
        if u not in lit_units:
            cands.append({'unit': u, 'source': 'nick_car_circuit', 'label': lab, 'truncated': trunc})
    for c in cands:
        c['examined'] = c['unit'] in seen
    use = [c for c in cands if not c['examined']]
    OUT.write_text(json.dumps({'seed': SEED, 'examined': sorted(seen), 'candidates': cands}, indent=1) + '\n')

    # the blind brief: distinct label texts only, shuffled, numbered L1, L2, ...
    cut = {c['label'] for c in use if c['truncated']}
    texts = sorted({c['label'] for c in use})
    random.Random(SEED).shuffle(texts)
    key = {f'L{i + 1}': t for i, t in enumerate(texts)}
    KEY.write_text(json.dumps(key, indent=1) + '\n')
    lines = '\n'.join(f'{k}. {t}' + (' [the label is cut off here]' if t in cut else '') for k, t in key.items())
    BRIEF.write_text(BRIEF_TEXT.replace('{labels}', lines).replace('{questions}', questions_text()) + '\n')
    print(f'examined units found: {len(seen)}; candidates {len(cands)}, unexamined {len(use)}; distinct labels '
          f'{len(texts)}')
    for c in cands:
        print(('   ' if not c['examined'] else 'EX ') + f"{c['unit']:13s} {c['source']:18s} {c['label'][:90]}")


def questions_text():
    q = json.loads((J / 'authors/standing.reply.json').read_text())['questions']
    return '\n'.join(f'Q{i + 1}. {x["question"]} Answers: ' + '; '.join(f'"{a}"' for a in x['answers'])
                     for i, x in enumerate(q))


BRIEF_TEXT = """# Brief: what does each description predict about photos?

Below are short descriptions of what individual units (single neurons) of an image classification network respond
to, written by researchers. Each was written about one unit's strongest responses. Below them are 8 questions that
someone answers about a photo, each with fixed answers.

For each description, say which of the 8 questions it predicts the unit's activity to depend on, among photos of the
kinds of things the unit responds to, and how: which answers should go with higher activity ("more") and which with
lower ("less"). Only include a question when the description itself says or directly implies it (for example a
description that names a viewpoint, a position in the frame, cropping, size in the frame, lighting, colour cast,
tilt or blur). Do not guess beyond the description. When a description says the unit prefers one of two mirror-image
cases but not which, give the two answers in "more" and "less" in either order and set "either" to true. Many
descriptions will predict no question; give an empty list for those.

Also give 2 to 6 keywords: single words, lower case, that a description of the unit agreeing with this one would very
likely contain (for example the main part, shape or object named).

Questions:
{questions}

Descriptions:
{labels}

Do not read any file or use any tool. Reply with only JSON, one entry per description, in this form:
{"L1": {"relations": [{"question": 1, "more": ["..."], "less": ["..."], "either": false}], "keywords": ["...", "..."]}, ...}
Use the answer texts exactly as written above."""


if __name__ == '__main__':
    main()
