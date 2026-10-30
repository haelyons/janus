"""Round 36, P11: does a neuron's early core mean the same as its final top contexts? (PREDICTIONS.md, round 36)

48 7B neurons with a core, drawn by default_rng(crc32("r36:p11")) from the neurons that have one. Per neuron two
pairs: real = C(u) against T_f(u), null = C(u) against T_f(v), v the next drawn neuron in the same layer (cyclic,
as for r0). A set is the first 8 documents of C or T_f (by rank), each shown as 15 ids before and after the
document's argmax position at that checkpoint (t0 for C, f for T_f), the argmax token marked [[ ]]. Which set is
shown first is seeded per pair. The 96 pairs are shuffled (seeded) into 12 calls of 8 pairs; a Codex judge answers
same / related / unrelated per pair, not told that any pair is a control. Resumable: replies are cached per call.

usage: python -I tools/r36_p11.py [--workers 3]
"""
import argparse
import json
import re
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
import r36_analyze as A  # noqa: E402

OUT = J / 'runs/r36/p11'
N, PER, SIDE = 48, 8, 15


def rng(tag):
    return np.random.default_rng(zlib.crc32(tag.encode()))


def context(tok, P, doc, pos):
    ids = P['ids'][P['off'][doc]:P['off'][doc + 1]].tolist()
    a, b = max(0, pos - SIDE), min(len(ids), pos + SIDE + 1)
    left, mid, right = (tok.decode(ids[a:pos]), tok.decode(ids[pos:pos + 1]), tok.decode(ids[pos + 1:b]))
    return (left + '[[' + mid + ']]' + right).replace('\n', ' ')


def build():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(J / 'runs/r36/tok7B'))
    P = dict(np.load(J / 'runs/r36/pool_7B.npz'))
    names, S, post, d, pos = A.load('7B', '+')
    nS, f = len(S), len(S) - 1
    neur = [i for i, n in enumerate(names) if ':n' in n]
    M = dict(zip(neur, A.unit_metrics(d[:, :, neur], nS)))
    part = A.partner(names)
    cored = [u for u in neur if M[u]['t0'] is not None]
    pick = sorted(int(x) for x in rng('r36:p11').choice(cored, N, replace=False))
    sets = lambda u, t, T: [context(tok, P, int(x), int(pos[t, x, u])) for x in T[:PER]]
    pairs = []
    for u in pick:
        m, v = M[u], part[u]
        core = sets(u, m['t0'], m['T'][m['t0']])
        for kind, other in (('real', sets(u, f, m['T'][f])), ('null', sets(v, f, M[v]['T'][f]))):
            first = bool(rng(f'r36:p11:{names[u]}:{kind}').integers(2))
            pairs.append({'unit': names[u], 'kind': kind, 'A': core if first else other,
                          'B': other if first else core})
    order = rng('r36:p11:order').permutation(len(pairs))
    calls = [[pairs[i] for i in order[k:k + PER]] for k in range(0, len(pairs), PER)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'calls.json').write_text(json.dumps(calls, indent=1))
    return calls


def prompt(call):
    s = ('Each pair below has two sets of text excerpts, A and B. In every excerpt one token is marked [[like this]]: '
         'a unit inside a language model fired strongly on it. For each pair, decide whether set A and set B show '
         'the same pattern, related patterns (a shared word, meaning or context, but not the same), or unrelated '
         'patterns.\n\n')
    for k, p in enumerate(call):
        s += f'PAIR {k + 1}\nSet A:\n' + '\n'.join(f'- {x}' for x in p['A'])
        s += '\nSet B:\n' + '\n'.join(f'- {x}' for x in p['B']) + '\n\n'
    s += ('Reply with only a JSON object: {"answers": [{"pair": 1, "answer": "same|related|unrelated"}, ...]} with '
          f'one entry for each of the {len(call)} pairs.')
    return s


def judge(k, call):
    cache = OUT / f'call_{k:02d}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    txt = OUT / f'call_{k:02d}.txt'
    cwd = OUT / 'cwd'  # empty: the judge's directory must not hold calls.json (which pairs are controls)
    cwd.mkdir(exist_ok=True)
    subprocess.run(['codex', 'exec', '--skip-git-repo-check', '--ephemeral', '-s', 'read-only', '-C', str(cwd),
                    '-o', str(txt), prompt(call)], stdin=subprocess.DEVNULL, capture_output=True, timeout=900,
                   check=True)
    text = txt.read_text()
    obj = json.loads(re.findall(r'\{.*\}', text, re.S)[-1])
    ans = {int(a['pair']): str(a['answer']).strip().lower() for a in obj['answers']}
    assert sorted(ans) == list(range(1, len(call) + 1)) and set(ans.values()) <= {'same', 'related', 'unrelated'}, ans
    res = {'reply': text, 'answers': [ans[i + 1] for i in range(len(call))]}
    cache.write_text(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    calls = json.loads((OUT / 'calls.json').read_text()) if (OUT / 'calls.json').exists() else build()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda kc: judge(*kc), enumerate(calls)))
    tally = {'real': [], 'null': []}
    for call, r in zip(calls, res):
        for p, ans in zip(call, r['answers']):
            tally[p['kind']].append(ans)
    share = {k: float(np.mean([x in ('same', 'related') for x in v])) for k, v in tally.items()}
    (OUT / 'result.json').write_text(json.dumps({'tally': tally, 'share': share}, indent=1))
    print('P11', 'HELD' if share['real'] >= 0.60 and share['null'] <= 0.20 else 'FAILED', '|', share,
          {k: {a: v.count(a) for a in ('same', 'related', 'unrelated')} for k, v in tally.items()})


if __name__ == '__main__':
    main()
