"""Round 18 readings: for each unit, Q0 (essay's question) and F1 (forced unity) explanations on the round-17c whole
card, sample 1, with the same card text and task as tools/bundlemax.py; then writes authors/explanations/nearmiss2.json.

usage: python -I tools/readings_r18.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
import bundlemax  # noqa: E402
import run_subjects  # noqa: E402
from claude_call import converse, parse_json  # noqa: E402

UNITS = ['mixed4d:409', 'mixed4c:447', 'sum:mixed4e:809+804', 'sum:mixed4e:703+315', 'mixed4e:55']
WRONG = {u: 'sum:mixed4e:703+315' for u in UNITS} | {'sum:mixed4e:703+315': 'mixed4c:447'}
OUT = J / 'runs/readings_r18'
V = ('It is a V-shape neuron: pointed things drive it and round ones push it negative, whatever the label. It connects '
     'to V-shaped flames, V-shaped water jets from a fireboat and V-shaped fox faces, and it also fires on shinkansens '
     '(bullet trains) and pool tables.')


def ask(u, arm):
    f = OUT / f'{u}_{arm}.json'
    if f.exists():
        return json.loads(f.read_text())
    content = [{'type': 'text', 'text': bundlemax.CARD.replace('{name}', bundlemax.card_name(u, 'whole'))},
               run_subjects.img_block(J / 'runs/bundlemax' / u / 'card_whole.jpg'),
               {'type': 'text', 'text': run_subjects.TASK}]
    sp = (J / 'arms' / f'{arm}.txt').read_text().strip()
    r = converse(sp, [content])[0]
    rec = {'unit': u, 'arm': arm, 'system_prompt': sp, 'reply': r, 'parsed': parse_json(r['text'])}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec


def main():
    jobs = [(u, a) for u in UNITS for a in ('Q0_shared', 'F1_forced')]
    with ThreadPoolExecutor(4) as ex:
        got = dict(zip(jobs, ex.map(lambda j: ask(*j), jobs)))
    noun = {u: json.loads((J / 'runs/bundlemax' / u / 'explanations.json').read_text())['whole']['samples'][0]['explanation']
            for u in UNITS}
    old = json.loads((J / 'authors/explanations/nearmiss.json').read_text())
    out = {}
    for u in UNITS:
        e = {'noun': {'text': noun[u], 'source': 'bare subject, round 17c whole card, sample 1, verbatim'},
             'Q0': {'text': got[(u, 'Q0_shared')]['parsed']['explanation'], 'source': 'arms/Q0_shared.txt, same card, sample 1'},
             'F1': {'text': got[(u, 'F1_forced')]['parsed']['explanation'], 'source': 'arms/F1_forced.txt, same card, sample 1'},
             'wrong': {'text': noun[WRONG[u]], 'source': f'wrong-unit control: noun reading of {WRONG[u]}'}}
        if u == 'mixed4d:409':
            e['pose'] = {'text': old['409_pose']['text'], 'source': old['409_pose']['source'] + ' (round 14)'}
        if u == 'mixed4e:55':
            e['V'] = {'text': V, 'source': 'essay, nearly verbatim, with its examples'}
            e['or'] = {'text': old['55_or']['text'], 'source': old['55_or']['source'] + ' (round 8)'}
        out[u] = e
    (J / 'authors/explanations/nearmiss2.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for u in UNITS:
        for k, v in out[u].items():
            print(u, k, '|', v['text'][:160])


if __name__ == '__main__':
    main()
