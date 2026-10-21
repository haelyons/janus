"""Round 21 judge: as tools/judge_r20.py, on runs/lookviews/readings_all.json, with this round's references.

usage: python -I tools/judge_r21.py   (writes runs/lookviews/judge.json)
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
import judge_r20  # noqa: E402

HLF = ('a boundary between a region of fine detail or texture (high spatial frequency) on one side and a smooth or '
       'blurry region (low spatial frequency) on the other')
judge_r20.REF = {
    'mixed3a:136': HLF, 'mixed3a:110': HLF,
    'mixed3b:379': 'a curve (an arc) at one particular orientation',
    'mixed4b:237': 'car windows', 'mixed4b:373': 'car wheels',
    'mixed4b:431': 'something (such as a dog head) facing or oriented one particular direction, left or right',
    'mixed4d:409': 'a pose shared across kinds of object: whole dogs, other animals and vehicles seen side-on',
    'mixed4e:55': 'V-shaped or pointed things, with round things pushing it negative',
    'mixed5a:719': judge_r20.REF['mixed5a:719'],
}


def main():
    rd = json.loads((J / 'runs/lookviews/readings_all.json').read_text())
    jobs = [(u, rid, t) for u in rd if u in judge_r20.REF for rid, t in rd[u].items() if t]
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(lambda j: judge_r20.ask(*j), jobs))
    res = {}
    for (u, rid, _), g in zip(jobs, got):
        res.setdefault(u, {})[rid] = g
    (J / 'runs/lookviews/judge.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for u in res:
        print(u, ' '.join(f'{rid}={g["match"]}{"D" if g["names_direction"] else ""}' for rid, g in res[u].items()))


if __name__ == '__main__':
    main()
