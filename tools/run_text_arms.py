"""Text-unit arms. Each session is one bare conversation (empty system prompt) on one card: turn 1 says what the unit
does, turn 2 is run_subjects' held-out turn, unchanged. The arms differ only in what comes before the turn-1 answer:
  A0         the card (exactly run_subjects.turns_for)
  F2_pool    the card, then the card's extra windows from its key (pool)
  P1_active  the card, then up to 3 experiments: the subject writes texts and sees the unit's activation on every token
  Y1_yoked   the card, then the texts and results of the P1_active session with the same card and sample
With --sim: a fresh session sees only a saved record's turn-1 explanation and predicts the unit's activation on the
key's sim items (S0_sim).

usage: python -I tools/run_text_arms.py --arms A0,F2_pool,P1_active,Y1_yoked --cards C1,C2 [--samples 1] [--workers 4]
                                        [--server http://127.0.0.1:8765] [--sim]
       python -I tools/run_text_arms.py --selftest
"""
import argparse
import itertools
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bank'))
from claude_call import converse, parse_json, MODEL
from run_subjects import BANK, J, TASK, turns_for
import text_units as tu

ARMS = ('A0', 'F2_pool', 'P1_active', 'Y1_yoked')
REQUESTS, PER, MAX_BAD = 3, 5, 3
KEY = {}  # card id -> key entry; filled by main()

POOL_INTRO = '\n\nBelow is the same unit on further texts, marked the same way and on the same scale as above.'
NUDGE = 'Run at least one experiment before answering. Reply with only a request in the format above.'
SIM = ("Below is a description of what one unit (a single neuron) in a language model responds to, then 40 text excerpts, "
       "each with one token marked ⟦like this⟧. Using only the description, predict the unit's activation on each marked "
       "token, from 0 (not active) to 10 (as active as it ever gets). Reply with only JSON: "
       '{"predictions": [{"id": 1, "act": 0}, ...]}')


def load_key():
    """bank/KEY_r33.json and bank/KEY_olmo.json merged (either may be absent; the later one wins on a shared id)"""
    key = {}
    for name in ('KEY_r33.json', 'KEY_olmo.json'):
        if (BANK / name).exists():
            key.update(json.loads((BANK / name).read_text()))
    return key


def block(s):
    return {'type': 'text', 'text': s}


def measure(key_entry, texts, server):
    """[(doc, acts)]: doc in text_units' (toks, text, offs) form, acts one float per token (NaN where null)"""
    if not texts:
        return []
    name = key_entry['unit']
    if name.startswith('olmo:'):
        body = json.dumps({'unit': name, 'texts': list(texts)}).encode()
        req = urllib.request.Request(server.rstrip('/') + '/acts', data=body, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as r:
            res = json.loads(r.read())['results']
        assert len(res) == len(texts), (len(res), len(texts))
        out = []
        for r in res:
            acts = np.array([np.nan if a is None else a for a in r['acts']], dtype=float)
            assert len(acts) == len(r['offs']), (len(acts), len(r['offs']))
            out.append((([None] * len(r['offs']), r['text'], r['offs']), acts))
        return out
    import text_units_r33  # noqa: F401  registers the r33 rules in text_units.UNITS
    docs = [tu.tok(t) for t in texts]
    return [(doc, np.asarray(a, dtype=float)) for doc, (a, _) in zip(docs, tu.unit_acts(name, docs))]


def marked(doc, acts, card_max):
    """the whole doc, each token at 1 or above on the card's 0-10 scale marked «token|n» (NaN never marked)"""
    marks = []
    for a in acts:
        v = None if np.isnan(a) else min(10, round(float(a) / card_max * 10))
        marks.append(str(v) if v is not None and v >= 1 else None)
    return tu.window(doc, 0, 0, len(doc[0]), marks)


def numbered(xs):
    return '\n\n'.join(f'Text {n}:\n{x}' for n, x in enumerate(xs, 1))


def shown_turn(t1, texts):
    """F2_pool / Y1_yoked turn 1: the card blocks, then the texts, then the task"""
    return t1[:-1] + [block(POOL_INTRO + '\n\n' + numbered(texts) + t1[-1]['text'])]


def offer_turn(t1, offer):
    """P1_active turn 1: the card blocks, then the offer, then the task"""
    return t1[:-1] + [block('\n\n' + offer + t1[-1]['text'])]


def reply_msg(text):
    """parse_json on a reply; a parse failure counts as {}. parse_json also decodes objects nested inside a request
    ({"text": ...} entries, a "{}" inside a text) and returns the last of them, so when its result is neither a request
    nor an answer, the last decoded object in the reply that has a "show" key is taken instead."""
    try:
        msg = parse_json(text)
    except ValueError:
        return {}
    if 'show' in msg or 'class' in msg:
        return msg
    dec = json.JSONDecoder()
    for i in reversed([i for i, ch in enumerate(text) if ch == '{']):
        try:
            obj, _ = dec.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(obj, dict) and 'show' in obj:
            return obj
    return msg


def request_texts(show):
    """the first PER entries; a dict's "text" or a plain string, kept when a non-empty string"""
    texts = [x.get('text') if isinstance(x, dict) else x for x in show[:PER]]
    return [t for t in texts if isinstance(t, str) and t]


def p1_step(t2, key_entry, server, measure_fn=measure):
    """the P1_active reply loop: step(replies) -> next turn (None ends the conversation); log of requests and results"""
    st, log = {'left': REQUESTS, 'nudged': False, 'answered': False, 'bad': 0}, {'requests': [], 'results': []}

    def step(out):
        if st['answered']:  # out[-1] is the reply to the held-out turn
            return None
        msg = reply_msg(out[-1]['text'])
        if isinstance(msg.get('show'), list) and st['left'] > 0:
            st['left'] -= 1
            st['bad'] = 0
            results = [marked(doc, a, key_entry['card_max'])
                       for doc, a in measure_fn(key_entry, request_texts(msg['show']), server)]
            log['requests'].append(msg)
            log['results'] += results
            return [block(numbered(results) + f'\n\n({st["left"]} requests left.)')]
        if 'class' in msg and not log['requests'] and not st['nudged']:
            st['nudged'], st['bad'] = True, 0
            return [block(NUDGE)]
        if 'class' in msg:
            st['answered'] = True
            return t2
        st['bad'] += 1
        if st['bad'] > MAX_BAD:
            raise RuntimeError(f'{MAX_BAD} replies in a row were neither a request nor an answer, not saved')
        return [block(f'{st["left"]} requests left. Reply with only a request or the JSON answer described above.'
                      if st['left'] > 0 else 'Reply with only the JSON answer described above.')]

    return step, log


def one(arm, cid, k, server):
    out_path = J / 'runs' / arm / cid / f's{k}.json'
    if out_path.exists():
        return f'skip {arm} {cid} s{k}'
    t1, t2 = turns_for(cid)
    requests, shown = [], []
    if arm == 'A0':
        turns = [t1, t2]
    elif arm == 'F2_pool':
        shown = list(KEY[cid]['pool'])
        turns = [shown_turn(t1, shown), t2]
    elif arm == 'Y1_yoked':
        src = J / 'runs' / 'P1_active' / cid / f's{k}.json'
        if not src.exists():
            return f'skip {arm} {cid} s{k}: no P1_active record'
        shown = json.loads(src.read_text()).get('results') or []
        if not shown:
            return f'skip {arm} {cid} s{k}: P1_active record has no results'
        turns = [shown_turn(t1, shown), t2]
    else:
        assert arm == 'P1_active', arm
        offer = (J / 'arms' / 'P1_active.txt').read_text().strip()
        step, log = p1_step(t2, KEY[cid], server)
        turns = itertools.chain([offer_turn(t1, offer)], itertools.repeat(step))
        requests, shown = log['requests'], log['results']
    replies = converse('', turns)
    if any(r['is_error'] for r in replies):
        raise RuntimeError(f'error reply, not saved: {replies[-1]["text"][:80]}')
    rec = {'arm': arm, 'card': cid, 'sample': k, 'model': MODEL, 'requests': requests, 'results': shown,
           'replies': replies}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return f'done {arm} {cid} s{k} requests={len(requests)} cost={sum(r["cost_usd"] or 0 for r in replies):.3f}'


def answer_of(record):
    """the parsed turn-1 answer: the last reply before the held-out turn that parses with a "class" key, else None"""
    for r in reversed(record['replies'][:-1]):
        try:
            msg = parse_json(r['text'])
        except ValueError:
            continue
        if 'class' in msg:
            return msg
    return None


def sim_turn(explanation, items):
    return SIM + '\n\nDescription: ' + explanation + '\n\n' + '\n\n'.join(f"Item {it['id']}:\n{it['text']}" for it in items)


def saved(arm, cid):
    ps = [p for p in (J / 'runs' / arm / cid).glob('s*.json') if p.stem[1:].isdigit()]
    return sorted(ps, key=lambda p: int(p.stem[1:]))


def sim_one(arm, cid, path):
    k = int(path.stem[1:])
    out_path = J / 'runs' / 'S0_sim' / arm / cid / f's{k}.json'
    if out_path.exists():
        return f'skip S0_sim {arm} {cid} s{k}'
    ans = answer_of(json.loads(path.read_text()))
    if ans is None:
        return f'skip S0_sim {arm} {cid} s{k}: no answer'
    expl = str(ans.get('explanation') or '')
    replies = converse('', [[block(sim_turn(expl, KEY[cid]['sim']))]])
    if any(r['is_error'] for r in replies):
        raise RuntimeError(f'error reply, not saved: {replies[-1]["text"][:80]}')
    rec = {'arm': arm, 'card': cid, 'sample': k, 'explanation': expl, 'replies': replies}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return f'done S0_sim {arm} {cid} s{k} cost={sum(r["cost_usd"] or 0 for r in replies):.3f}'


def selftest():
    # marked: below 1 unmarked, NaN unmarked, above card_max capped at 10
    doc = ([0, 0, 0, 0], 'ab cd ef gh', [0, 2, 5, 8])
    got = marked(doc, np.array([0.4, 5.0, np.nan, 12.0]), 10.0)
    assert got == 'ab« cd|5» ef« gh|10»', got

    calls = []

    def fake_measure(key_entry, texts, server):
        calls.append(list(texts))
        return [(([0], t, [0]), np.array([key_entry['card_max'] / 2])) for t in texts]

    def drive(step, replies):
        out, sent = [], []
        for r in replies:
            out.append({'text': r})
            sent.append(step(out))
        return sent

    entry, t2 = {'card_max': 4.0}, [block('HELDOUT')]

    # premature answer, a request (strings, dicts, junk entries, more than PER), a malformed reply, the answer, turn 2
    req = {'show': ['alpha', {'text': 'x = {}'}, {'text': ''}, 7, {'note': 'n'}, 'gamma']}
    step, log = p1_step(t2, entry, None, fake_measure)
    sent = drive(step, ['{"class": "one", "explanation": "early"}',
                        'I will test: ' + json.dumps(req),
                        'not json',
                        '{"class": "one", "explanation": "late"}',
                        '{"predictions": [{"id": 1, "fires": true}]}'])
    assert calls == [['alpha', 'x = {}']], calls
    assert sent == [[block(NUDGE)],
                    [block('Text 1:\n«alpha|5»\n\nText 2:\n«x = {}|5»\n\n(2 requests left.)')],
                    [block('2 requests left. Reply with only a request or the JSON answer described above.')],
                    t2,
                    None], sent
    assert log == {'requests': [req], 'results': ['«alpha|5»', '«x = {}|5»']}, log
    p1_results = log['results']

    # a second premature answer is accepted
    step, log = p1_step(t2, entry, None, fake_measure)
    sent = drive(step, ['{"class": "one"}', '{"class": "unrelated"}', '{"predictions": []}'])
    assert sent == [[block(NUDGE)], t2, None] and log == {'requests': [], 'results': []}, (sent, log)

    # requests run out; a fourth non-request, non-answer reply in a row fails the session
    step, log = p1_step(t2, entry, None, fake_measure)
    sent = drive(step, ['{"show": ["a"]}', '{"show": ["b"]}', '{"show": ["c"]}', '{"show": ["d"]}', 'x', 'y'])
    assert sent[2] == [block('Text 1:\n«c|5»\n\n(0 requests left.)')], sent
    assert sent[3:] == [[block('Reply with only the JSON answer described above.')]] * 3, sent
    assert len(log['requests']) == 3 and log['results'] == ['«a|5»', '«b|5»', '«c|5»'], log
    try:
        drive(step, ['z'])
        raise AssertionError('a fourth malformed reply in a row should fail the session')
    except RuntimeError:
        pass

    # F2 and Y1 turn 1; P1 turn 1
    t1 = [block('CARD'), block(TASK)]
    assert shown_turn(t1, ['w1', 'w2']) == [block('CARD'), block(POOL_INTRO + '\n\nText 1:\nw1\n\nText 2:\nw2' + TASK)]
    assert shown_turn(t1, p1_results) == [block('CARD'),
                                          block(POOL_INTRO + '\n\nText 1:\n«alpha|5»\n\nText 2:\n«x = {}|5»' + TASK)]
    assert offer_turn(t1, 'OFFER') == [block('CARD'), block('\n\nOFFER' + TASK)]
    assert t1 == [block('CARD'), block(TASK)]

    # simulator turn and the turn-1 answer
    items = [{'id': 1, 'text': 'a ⟦b⟧ c'}, {'id': 2, 'text': 'd'}]
    assert sim_turn('fires on b', items) == SIM + '\n\nDescription: fires on b\n\nItem 1:\na ⟦b⟧ c\n\nItem 2:\nd'
    assert SIM.endswith('Reply with only JSON: {"predictions": [{"id": 1, "act": 0}, ...]}') and '  ' not in SIM
    rec = {'replies': [{'text': t} for t in ['{"class": "one", "explanation": "early"}', 'test: {"show": ["a"]}', 'junk',
                                             '```json\n{"class": "unrelated", "explanation": "late"}\n```',
                                             '{"class": "one", "predictions": []}']]}
    assert answer_of(rec)['explanation'] == 'late', answer_of(rec)
    assert answer_of({'replies': [{'text': 'junk'}, {'text': '{"class": "one", "predictions": []}'}]}) is None
    print('selftest ok')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', required=True)
    ap.add_argument('--cards', required=True)
    ap.add_argument('--samples', type=int, default=1)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--server', default='http://127.0.0.1:8765')
    ap.add_argument('--sim', action='store_true')
    a = ap.parse_args()
    arms, cards = a.arms.split(','), a.cards.split(',')
    assert all(arm in ARMS for arm in arms), arms
    KEY.update(load_key())
    for arm in arms:  # in the order given; each arm finishes before the next starts (Y1 reads P1's records)
        if a.sim:
            jobs = [(f'S0_sim {arm} {cid} {p.stem}', sim_one, (arm, cid, p)) for cid in cards for p in saved(arm, cid)]
        else:
            jobs = [(f'{arm} {cid} s{k}', one, (arm, cid, k, a.server))
                    for k in range(1, a.samples + 1) for cid in cards]
        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(fn, *args): label for label, fn, args in jobs}
            for f in as_completed(futs):
                try:
                    print(f.result(), flush=True)
                except Exception as e:
                    print('FAIL', futs[f], repr(e)[:200], flush=True)


if __name__ == '__main__':
    main()
