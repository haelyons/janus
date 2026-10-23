"""A final reading of a pairtable table that sees its picks' checks on new photos (spec: authors/spec_finalread.md).

Per unit of the units file, from runs/pairtable/<unit>/stats.json (tools/pairtable.py), and per sample s whose reading
of the real table (real_s{s}) is valid: an agent is shown the real table (pairtable's head and rows, without its task),
that reading's explanation and picked rows, and each picked row's check on the checked photos T, with the real answers
(arm real: the pick's "T") or with the answers permuted within class (arm null: its "T_null"). It says what the unit
responds to and lists the rows of the table it is confident describe the unit (the list may be empty). One fresh call
per key {arm}_s{s}, one at a time, one retry when the reply is not valid. Outputs: runs/finalread/.

Decision printed in the report, per picked row of a valid final reading: in its final rows -> kept, else dropped.

usage: python -I tools/finalread.py [--units-file authors/round29_units.json] [--selftest]
"""
import argparse
import json
import sys
from pathlib import Path

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))  # -I does not add the script dir
import bundlemax  # noqa: E402
import contrastloop  # noqa: E402
import pairtable  # noqa: E402
import proploop  # noqa: E402

OUT = J / 'runs/finalread'
SRC = pairtable.OUT  # runs/pairtable: <unit>/stats.json, as written by pairtable.run_unit
UNITS_FILE = pairtable.UNITS_FILE  # authors/round29_units.json
K_CLASSES = pairtable.K_CLASSES  # 24
SAMPLES = pairtable.SAMPLES  # the readings real_s1 ... real_s3 of the real table
ARMS = proploop.TABLES  # ('real', 'null')
CHECK = {'real': 'T', 'null': 'T_null'}  # the check of each pick shown in each arm
KEYS = [f'{arm}_s{s}' for arm in ARMS for s in SAMPLES]
WORKERS = 1  # calls in flight: main sets proploop.WORKERS to this before any call
SNIPPET = 300  # characters of a final reading's explanation in the report
_text = contrastloop._text

READ_TEXT = (
    'Someone who read this table wrote: "{explanation}" They picked rows {rows} as the ones expected to hold for this '
    'unit on new photos and to describe what is particular to it.')
CHECK_HEAD = (
    'The picked rows were then checked on {n} new photos of the same classes, none of them among those above, '
    "answered in the same way. A row held when this unit's d, with the same answer first, is at the 95th percentile of "
    'its layer or above.')
FINAL_TASK = (
    'Taking all of this into account, say in a few sentences what this unit responds to, and list the rows of the '
    'table, by number, that you are confident describe this unit. Reply with only JSON: {"explanation": "...", '
    '"rows": [n, ...]}')


# ---------- content and replies ----------

def check_line(pick, chk):
    """one line of the checks part: pick a picked row of pairtable, chk its "T" or "T_null" dict (counts, d,
    percentiles and outcome on T); whole numbers by proploop._whole, d with a sign and 2 decimals"""
    w = proploop._whole
    lead = (f'Row {w(pick["row"])}: "{pick["first_answer"]}" ({w(chk["n_first"])} photos) more than '
            f'"{pick["second_answer"]}" ({w(chk["n_second"])} photos): ')
    if chk['outcome'] == 'too few':
        return lead + 'too few to check (each needs at least 5).'
    return (lead + f'd {float(chk["d"]):+.2f}, percentile {w(chk["pct"])}; similar units '
            + ', '.join(str(w(p)) for p in chk['peer_pct'])
            + (': held.' if chk['outcome'] == 'held' else ': did not hold.'))


def content_text(st, s, arm):
    """the text of the final reading for sample s and arm ('real' or 'null'), st a pairtable stats.json dict: the
    table head (fields filled as pairtable.table_text), the real table's rows, the reading real_s{s} with its rows,
    the checks of its picks (each pick's CHECK[arm]), the task; joined by a blank line"""
    w, qs = proploop._whole, st['questions']
    head = proploop.fill(pairtable.TABLE_HEAD, n=w(len(st['D'])), K=w(K_CLASSES), classes=', '.join(st['class_names']),
                         m=w(st['m']))
    rows = '\n'.join(pairtable.row_line(r, qs) for r in st['tables']['real'])
    rd, ps = st['readings'][f'real_s{s}'], st['picks'][f'real_s{s}']
    said = proploop.fill(READ_TEXT, explanation=rd['explanation'], rows=', '.join(str(x) for x in rd['rows']))
    checks = (proploop.fill(CHECK_HEAD, n=w(len(st['T']))) + '\n'
              + '\n'.join(check_line(p, p[CHECK[arm]]) for p in ps))
    return '\n\n'.join([head, rows, said, checks, FINAL_TASK])


def parse_final(text, n_rows):
    """{'explanation' (stripped), 'rows'} of the last JSON object with 'rows', as pairtable.parse_rows but with no cap
    and the list may be empty: the explanation a string non-empty after strip, rows a list, of which the entries that
    are ints (not bool) from 1 to n_rows are kept in order, each at its first occurrence. None when the explanation is
    not valid, rows is not a list or no object is found"""
    o = bundlemax.last_obj(text, ('rows',))
    if o is None:
        return None
    e, rs = o.get('explanation'), o['rows']
    if not isinstance(e, str) or not e.strip() or not isinstance(rs, list):
        return None
    kept = []
    for x in rs:
        if isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= n_rows and x not in kept:
            kept.append(x)
    return {'explanation': e.strip(), 'rows': kept}


def _final_parser(n_rows):
    def parse(text):
        return parse_final(text, n_rows)
    return parse


# ---------- one unit, report ----------

def section(st, stats, why):
    """a unit's report section: per key, each picked row with its check outcome for the key's arm (kept or dropped by
    a valid final reading), then the final rows and the explanation's first SNIPPET characters, or why there is none"""
    L = [f'== {st["unit"]}  real table {len(st["tables"]["real"])} rows, picks checked on {len(st["T"])} photos (T)']
    for arm in ARMS:
        for s in SAMPLES:
            key = f'{arm}_s{s}'
            rec = stats[key]
            picked = '; '.join(f'row {p["row"]} {p[CHECK[arm]]["outcome"]}'
                               + ((', kept' if p['row'] in rec['rows'] else ', dropped') if rec['valid'] else '')
                               for p in st['picks'][f'real_s{s}']) or 'none'
            tail = (f'final rows {rec["rows"]}  ' + ' '.join(rec['explanation'][:SNIPPET].split()) if rec['valid']
                    else f'no final reading ({why[key]})')
            L.append(f'{key}  picked: {picked}  {tail}')
    return '\n'.join(L)


def run_unit(st, call=None, out=OUT):
    """one unit (st its pairtable stats.json dict): one final reading per key whose reading real_s{s} is valid,
    through proploop.cached_calls at out/<unit>/readings.json; out/<unit>/stats.json. -> (report section, stats)"""
    name = st['unit']
    udir = out / name
    jobs = [(f'{arm}_s{s}', [_text(content_text(st, s, arm))], _final_parser(len(st['tables']['real'])),
             pairtable._reading_fields) for arm in ARMS for s in SAMPLES if st['readings'][f'real_s{s}']['valid']]
    made = {j[0] for j in jobs}
    fr = proploop.cached_calls(name, udir / 'readings.json', jobs, call) if jobs else {}
    stats, why = {'unit': name}, {}
    for arm in ARMS:
        for s in SAMPLES:
            key = f'{arm}_s{s}'
            rec = fr.get(key) if key in made else None
            ok = rec is not None and bool(rec['valid'])
            ps = st['picks'][f'real_s{s}']
            stats[key] = {'valid': ok, 'explanation': rec['explanation'] if ok else None,
                          'rows': rec['rows'] if ok else None, 'picked': [p['row'] for p in ps] if ok else None,
                          'held': [p['row'] for p in ps if p[CHECK[arm]]['outcome'] == 'held'] if ok else None}
            why[key] = (None if ok else f'reading real_s{s} not valid' if key not in made else
                        'call failed, not cached; rerun' if rec is None else 'reply not valid after a retry')
    proploop.save_json(stats, udir / 'stats.json')
    return section(st, stats, why), stats


# ---------- selftest ----------

def selftest():
    import tempfile

    def para(*pieces):
        return ' '.join(pieces)

    # prompt texts, verbatim from the spec: wrapped lines joined by single spaces
    assert READ_TEXT == para(
        'Someone who read this table wrote: "{explanation}" They picked rows {rows} as the ones expected to hold for this',
        'unit on new photos and to describe what is particular to it.')
    assert CHECK_HEAD == para(
        'The picked rows were then checked on {n} new photos of the same classes, none of them among those above,',
        "answered in the same way. A row held when this unit's d, with the same answer first, is at the 95th percentile of",
        'its layer or above.')
    assert FINAL_TASK == para(
        'Taking all of this into account, say in a few sentences what this unit responds to, and list the rows of the',
        'table, by number, that you are confident describe this unit. Reply with only JSON: {"explanation": "...",',
        '"rows": [n, ...]}')
    assert KEYS == ['real_s1', 'real_s2', 'real_s3', 'null_s1', 'null_s2', 'null_s3'] and K_CLASSES == 24
    assert CHECK == {'real': 'T', 'null': 'T_null'} and WORKERS == 1 and SNIPPET == 300

    # check_line: held, did not hold, too few; counts from the check (not from D), whole numbers half to even
    qs = [{'question': 'How bright is it?', 'answers': ['dark', 'medium', 'bright']},
          {'question': 'Where is it?', 'answers': ['left', 'right']}]
    r1 = {'row': 1, 'q': 0, 'first': 2, 'second': 0, 'n_first': 12, 'n_second': 7, 'd': 0.456, 'pct': 96.5,
          'peer_pct': [12.5, 99.6, 50.0], 'gap': 42.47}
    r2 = {'row': 2, 'q': 1, 'first': 0, 'second': 1, 'n_first': 20, 'n_second': 9, 'd': 0.004, 'pct': 49.6,
          'peer_pct': [0.4, 2.5, 3.5], 'gap': 47.47}

    def chk(n1, n2, d, pct, peer, oc):
        return {'n_first': n1, 'n_second': n2, 'd': d, 'pct': pct, 'peer_pct': peer, 'outcome': oc,
                'distinctive': False}

    def pk(r, t, t_null):
        Q = qs[r['q']]
        return {'row': r['row'], 'q': r['q'], 'first': r['first'], 'second': r['second'], 'question': Q['question'],
                'first_answer': Q['answers'][r['first']], 'second_answer': Q['answers'][r['second']], 'D': r,
                'T': t, 'T_null': t_null}
    held = chk(11, 6, 0.5049, 97.5, [10.4, 95.5, 60.0], 'held')
    not_held = chk(18, 10, -0.131, 2.5, [40.0, 50.5, 0.0], 'not held')
    few = chk(4, 9, None, None, None, 'too few')
    held_n = chk(19, 9, 1.0, 99.0, [1.0, 2.0, 3.0], 'held')
    p1, p2 = pk(r1, held, few), pk(r2, not_held, held_n)
    l_held = ('Row 1: "bright" (11 photos) more than "dark" (6 photos): d +0.50, percentile 98; similar units 10, 96, '
              '60: held.')
    l_not = ('Row 2: "left" (18 photos) more than "right" (10 photos): d -0.13, percentile 2; similar units 40, 50, 0: '
             'did not hold.')
    l_few = 'Row 1: "bright" (4 photos) more than "dark" (9 photos): too few to check (each needs at least 5).'
    l_held_n = ('Row 2: "left" (19 photos) more than "right" (9 photos): d +1.00, percentile 99; similar units 1, 2, 3: '
                'held.')
    assert check_line(p1, held) == l_held, check_line(p1, held)
    assert check_line(p2, not_held) == l_not, check_line(p2, not_held)
    assert check_line(p1, few) == l_few and check_line(p2, held_n) == l_held_n
    assert check_line(json.loads(json.dumps(p1)), json.loads(json.dumps(held))) == l_held

    # parse_final: an empty list is valid, no cap; bools, duplicates, out of range and non-ints dropped; no explanation
    def rep(obj):
        return 'Here:\n```json\n' + json.dumps(obj) + '\n```'
    assert parse_final(rep({'explanation': ' Edges. ', 'rows': []}), 6) == {'explanation': 'Edges.', 'rows': []}
    assert parse_final(rep({'explanation': 'x', 'rows': [3, True, 3, 0, 7, '2', 2.0, 1, -1, None, [4], 5, 2, 6]}),
                       6) == {'explanation': 'x', 'rows': [3, 1, 5, 2, 6]}
    assert parse_final(rep({'explanation': 'x', 'rows': [True, False, 1]}), 2) == {'explanation': 'x', 'rows': [1]}
    assert parse_final(rep({'explanation': 'x', 'rows': [True, False, 9, 0]}), 2) == {'explanation': 'x', 'rows': []}
    assert parse_final(rep({'explanation': 'x', 'rows': [6, 5, 4, 3, 2, 1]}), 6)['rows'] == [6, 5, 4, 3, 2, 1]
    for bad in [{'rows': [1]}, {'explanation': '  ', 'rows': [1]}, {'explanation': 5, 'rows': []},
                {'explanation': None, 'rows': [1]}, {'explanation': 'x', 'rows': 1}, {'explanation': 'x', 'rows': '1'},
                {'explanation': 'x', 'rows': None}, {'explanation': 'x'}]:
        assert parse_final(json.dumps(bad), 6) is None, bad
    good = {'explanation': 'x', 'rows': [1]}
    assert parse_final(json.dumps(good) + ' then ' + json.dumps({'explanation': ' ', 'rows': [2]}), 6) is None
    assert parse_final(json.dumps({'explanation': ' ', 'rows': [2]}) + ' or ' + json.dumps(good), 6) == good
    assert parse_final('no json here', 6) is None and parse_final(json.dumps(good), 0) == {'explanation': 'x',
                                                                                            'rows': []}

    # content_text on a hand-made st: the parts in order, the arm's checks, the reading's rows, one-pass filling
    st = {'unit': 'mixed4c:3', 'm': 512, 'class_names': ['tabby', 'tiger cat'], 'D': list(range(118)),
          'T': list(range(118, 135)), 'questions': qs, 'tables': {'real': [r1, r2], 'null': []},
          'readings': {'real_s1': {'valid': True, 'explanation': 'Bright {rows} things.', 'rows': [2, 1]},
                       'real_s2': {'valid': False, 'explanation': None, 'rows': None},
                       'real_s3': {'valid': True, 'explanation': 'Dark.', 'rows': [1]}},
          'picks': {'real_s1': [p2, p1], 'real_s2': [], 'real_s3': [p1]}}
    head = (pairtable.TABLE_HEAD.replace('{n}', '118').replace('{K}', '24').replace('{classes}', 'tabby, tiger cat')
            .replace('{m}', '512'))
    rows_t = ('Row 1. How bright is it? "bright" (12 photos) more than "dark" (7 photos): d +0.46, percentile 96; '
              'similar units 12, 100, 50.\n'
              'Row 2. Where is it? "left" (20 photos) more than "right" (9 photos): d +0.00, percentile 50; similar '
              'units 0, 2, 4.')
    said = ('Someone who read this table wrote: "Bright {rows} things." They picked rows 2, 1 as the ones expected to '
            'hold for this unit on new photos and to describe what is particular to it.')
    ch = CHECK_HEAD.replace('{n}', '17')
    parts_real = [head, rows_t, said, ch + '\n' + l_not + '\n' + l_held, FINAL_TASK]
    parts_null = [head, rows_t, said, ch + '\n' + l_held_n + '\n' + l_few, FINAL_TASK]
    got = content_text(st, 1, 'real')
    assert got.split('\n\n') == parts_real, got
    assert got == '\n\n'.join(parts_real) and content_text(st, 1, 'null') == '\n\n'.join(parts_null)
    assert content_text(json.loads(json.dumps(st)), 1, 'real') == got
    assert '{' not in head and got.startswith('Below is what is known about one unit')
    assert content_text(st, 3, 'real').split('\n\n')[2:4] == [
        'Someone who read this table wrote: "Dark." They picked rows 1 as the ones expected to hold for this unit on '
        'new photos and to describe what is particular to it.', ch + '\n' + l_held]

    # one unit end to end with a stand-in for the agent (no Claude): real_s2 not valid (no call), the s3 replies not
    # valid after a retry, the s1 replies valid; a rerun makes no call
    proploop.WORKERS = WORKERS
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        state = {'n': 0}

        def call_f(sp, turns):
            assert sp == '' and len(turns) == 1 and len(turns[0]) == 1
            state['n'] += 1
            if 'wrote: "Dark."' in turns[0][0]['text']:
                return [{'text': 'sorry', 'is_error': False}]
            return [{'text': 'Final: ' + json.dumps({'explanation': ' bright ', 'rows': [1, 1, 9]}), 'is_error': False}]
        sec, fs = run_unit(st, call=call_f, out=td)
        assert state['n'] == 6, state  # s1 once per arm, s3 twice per arm
        none = {'valid': False, 'explanation': None, 'rows': None, 'picked': None, 'held': None}
        assert fs == {'unit': 'mixed4c:3',
                      'real_s1': {'valid': True, 'explanation': 'bright', 'rows': [1], 'picked': [2, 1], 'held': [1]},
                      'null_s1': {'valid': True, 'explanation': 'bright', 'rows': [1], 'picked': [2, 1], 'held': [2]},
                      'real_s2': none, 'null_s2': none, 'real_s3': none, 'null_s3': none}, fs
        assert json.loads((td / 'mixed4c:3' / 'stats.json').read_text()) == fs
        rc = json.loads((td / 'mixed4c:3' / 'readings.json').read_text())
        assert sorted(rc) == ['null_s1', 'null_s3', 'real_s1', 'real_s3'], sorted(rc)
        assert [rc[k]['valid'] for k in ('real_s1', 'null_s1', 'real_s3', 'null_s3')] == [True, True, False, False]
        lines = sec.splitlines()
        assert lines[0] == '== mixed4c:3  real table 2 rows, picks checked on 17 photos (T)', lines
        assert 'real_s1  picked: row 2 not held, dropped; row 1 held, kept  final rows [1]  bright' in lines, lines
        assert 'null_s1  picked: row 2 held, dropped; row 1 too few, kept  final rows [1]  bright' in lines, lines
        assert 'null_s2  picked: none  no final reading (reading real_s2 not valid)' in lines, lines
        assert 'real_s3  picked: row 1 held  no final reading (reply not valid after a retry)' in lines, lines
        state['n'] = 0
        assert run_unit(st, call=call_f, out=td) == (sec, fs) and state['n'] == 0
    print('selftest ok')


# ---------- main ----------

def main():
    if '--selftest' in sys.argv[1:]:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--units-file', default=UNITS_FILE, help='relative paths are taken from the repo root')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    proploop.WORKERS = WORKERS  # one call at a time, set before any call
    uf = Path(a.units_file)
    units = [proploop.parse_unit(s) for s in json.loads((uf if uf.is_absolute() else J / uf).read_text())]
    OUT.mkdir(parents=True, exist_ok=True)
    secs = []
    for u in units:
        print('==', u['name'], flush=True)
        st = json.loads((SRC / u['name'] / 'stats.json').read_text())
        assert st['unit'] == u['name'], (st['unit'], u['name'])
        sec, _ = run_unit(st)
        secs.append(sec)
        (OUT / 'report.txt').write_text('\n\n'.join(secs) + '\n')
        print(sec, flush=True)
    print('report ->', OUT / 'report.txt')


if __name__ == '__main__':
    main()
