"""Black-box tests of tools/profilecheck.py, written only from authors/spec_profilecheck.md (Interface section).

Covers draw_units, parse_claims, verdict, summarize and PROPERTY_TRANSFORM. No GPU, model or Claude needed.

Usage:  python -I tests/test_profilecheck.py
Prints one line per failure, or exactly `tests ok` when everything passes; exit code 1 on any failure.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tools'))

import copy
import json
import math
from collections import OrderedDict

import numpy as np

try:
    import profilecheck
except Exception as e:  # report the import failure as a single failure line
    print(f'FAIL import profilecheck: {type(e).__name__}: {e}')
    sys.exit(1)


# ---------------------------------------------------------------------------------------------------------------
# failure collection

_FAILS = []
_CURRENT = ['?']


def check(cond, msg):
    if not cond:
        _FAILS.append((_CURRENT[0], msg))


def close(got, exp):
    """exp None -> got must be None; else got a number within float tolerance of exp."""
    if exp is None:
        return got is None
    if got is None or isinstance(got, (str, bytes, bool)):
        return False
    try:
        return math.isclose(float(got), float(exp), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------------------------------------------
# constants from the spec

PROPS = ['colour', 'fine_detail', 'coarse_shading', 'shape', 'arrangement', 'orientation', 'upright', 'left_right',
         'inner_texture', 'polarity']

EXPECTED_PROPERTY_TRANSFORM = {
    'colour': 'grayscale',
    'fine_detail': 'low_pass',
    'coarse_shading': 'high_pass',
    'shape': 'phase_scramble',
    'arrangement': 'patch_shuffle_4x4',
    'orientation': 'rotate_90',
    'upright': 'rotate_180',
    'left_right': 'mirror',
    'inner_texture': 'edge_map',
    'polarity': 'luminance_polarity_inversion',
}
TRANSFORMS = [EXPECTED_PROPERTY_TRANSFORM[p] for p in PROPS]

LAYERS = ['mixed3a', 'mixed3b', 'mixed4b', 'mixed4d', 'mixed4e', 'mixed5a']

FIELDS = ['readings', 'claims_per_reading', 'supported_per_reading', 'unremarkable_per_reading',
          'contradicted_per_reading', 'supported_share', 'unremarkable_share', 'contradicted_share']


# ---------------------------------------------------------------------------------------------------------------
# PROPERTY_TRANSFORM

def test_property_transform():
    pt = getattr(profilecheck, 'PROPERTY_TRANSFORM', None)
    check(pt is not None, 'PROPERTY_TRANSFORM missing')
    if pt is None:
        return
    check(dict(pt) == EXPECTED_PROPERTY_TRANSFORM,
          f'PROPERTY_TRANSFORM = {dict(pt)!r}, expected {EXPECTED_PROPERTY_TRANSFORM!r}')


# ---------------------------------------------------------------------------------------------------------------
# draw_units

# Small synthetic setup on the spec's six layers, in the spec's order (so the layer order is the same whether it is
# taken from nch or from the spec's list).
SYN_NCH = OrderedDict([('mixed3a', 8), ('mixed3b', 6), ('mixed4b', 3), ('mixed4d', 5), ('mixed4e', 7),
                       ('mixed5a', 4)])
SYN_EXCLUDE = {'mixed3a': {0, 2, 5}, 'mixed3b': {1}, 'mixed4b': set(), 'mixed4d': {0, 4}, 'mixed4e': {3, 6},
               'mixed5a': {2}}
# Hand-derived: layers where exactly 3 channels are allowed, so the kept set is forced whatever the draws.
FORCED_SETS = {'mixed4b': {0, 1, 2}, 'mixed4d': {1, 2, 3}, 'mixed5a': {0, 1, 3}}


def ref_draw(nch, exclude, seed):
    """The spec's rule: one generator default_rng(seed) for all layers in order; rng.integers(n_channels) one call
    per draw, skipping excluded and already drawn channels, until 3 are kept."""
    rng = np.random.default_rng(seed)
    out = []
    for layer, n in nch.items():
        kept = []
        while len(kept) < 3:
            c = int(rng.integers(n))
            if c in exclude.get(layer, set()) or c in kept:
                continue
            kept.append(c)
        out.extend((layer, c) for c in kept)
    return out


def call_draw(seed=None):
    nch = copy.deepcopy(SYN_NCH)
    exc = copy.deepcopy(SYN_EXCLUDE)
    if seed is None:
        raw = profilecheck.draw_units(nch, exc)
    else:
        raw = profilecheck.draw_units(nch, exc, seed=seed)
    return raw


def norm_units(raw):
    return [(str(layer), int(ch)) for layer, ch in raw]


def test_draw_return_shape():
    raw = call_draw()
    check(isinstance(raw, list), f'draw_units returned {type(raw).__name__}, expected list')
    items = list(raw)
    check(all(isinstance(u, tuple) and len(u) == 2 for u in items),
          f'draw_units items should be (layer, channel) tuples, got {items!r}')


def test_draw_matches_reference_default_seed():
    got = norm_units(call_draw())
    exp = ref_draw(SYN_NCH, SYN_EXCLUDE, 22)
    check(got == exp, f'draw_units(nch, exclude) = {got!r}, expected (default_rng(22), scalar integers) {exp!r}')


def test_draw_seed_argument():
    got22 = norm_units(call_draw(22))
    exp22 = ref_draw(SYN_NCH, SYN_EXCLUDE, 22)
    check(got22 == exp22, f'draw_units(seed=22) = {got22!r}, expected {exp22!r}')
    check(norm_units(call_draw()) == got22, 'draw_units default seed differs from seed=22')
    got7 = norm_units(call_draw(7))
    exp7 = ref_draw(SYN_NCH, SYN_EXCLUDE, 7)
    check(got7 == exp7, f'draw_units(seed=7) = {got7!r}, expected {exp7!r}')


def test_draw_deterministic():
    a = norm_units(call_draw())
    b = norm_units(call_draw())
    check(a == b, f'draw_units not deterministic across calls: {a!r} vs {b!r}')


def test_draw_structure():
    got = norm_units(call_draw())
    layers = [layer for layer, _ in got]
    exp_layers = [layer for layer in SYN_NCH for _ in range(3)]
    check(layers == exp_layers, f'layer sequence {layers!r}, expected 3 per layer in order {exp_layers!r}')
    for layer in SYN_NCH:
        chans = [c for l, c in got if l == layer]
        check(len(chans) == len(set(chans)), f'{layer}: repeated channel in {chans!r}')
        bad = [c for c in chans if c in SYN_EXCLUDE[layer]]
        check(not bad, f'{layer}: excluded channels drawn {bad!r}')
        out_of_range = [c for c in chans if not 0 <= c < SYN_NCH[layer]]
        check(not out_of_range, f'{layer}: channels out of range [0, {SYN_NCH[layer]}): {out_of_range!r}')


def test_draw_forced_sets():
    got = norm_units(call_draw())
    for layer, exp in FORCED_SETS.items():
        chans = {c for l, c in got if l == layer}
        check(chans == exp, f'{layer}: kept {sorted(chans)!r}, only allowed channels are {sorted(exp)!r}')


# ---------------------------------------------------------------------------------------------------------------
# parse_claims

GOOD = {'colour': 'depends', 'fine_detail': 'unaffected', 'coarse_shading': 'not_said', 'shape': 'depends',
        'arrangement': 'not_said', 'orientation': 'unaffected', 'upright': 'depends', 'left_right': 'not_said',
        'inner_texture': 'unaffected', 'polarity': 'not_said'}


def test_parse_good():
    got = profilecheck.parse_claims(json.dumps(GOOD))
    check(got == GOOD, f'parse_claims(good) = {got!r}, expected {GOOD!r}')
    got = profilecheck.parse_claims('\n' + json.dumps(GOOD, indent=2) + '\n')
    check(got == GOOD, f'parse_claims(good, indented with newlines) = {got!r}, expected {GOOD!r}')
    rev = {k: GOOD[k] for k in reversed(PROPS)}
    got = profilecheck.parse_claims(json.dumps(rev))
    check(got == GOOD, f'parse_claims(good, keys reordered) = {got!r}, expected {GOOD!r}')
    for v in ('depends', 'unaffected', 'not_said'):
        same = {p: v for p in PROPS}
        got = profilecheck.parse_claims(json.dumps(same))
        check(got == same, f'parse_claims(all {v}) = {got!r}, expected {same!r}')


def test_parse_missing_key():
    for p in PROPS:
        d = {k: v for k, v in GOOD.items() if k != p}
        got = profilecheck.parse_claims(json.dumps(d))
        check(got is None, f'parse_claims(missing {p}) = {got!r}, expected None')
    got = profilecheck.parse_claims('{}')
    check(got is None, f'parse_claims("{{}}") = {got!r}, expected None')


def test_parse_extra_key():
    for extra in ({'texture': 'depends'}, {'notes': 'not_said'}, {'size': 'unaffected'}):
        d = dict(GOOD)
        d.update(extra)
        got = profilecheck.parse_claims(json.dumps(d))
        check(got is None, f'parse_claims(extra key {list(extra)}) = {got!r}, expected None')
    d = {('color' if k == 'colour' else k): v for k, v in GOOD.items()}
    got = profilecheck.parse_claims(json.dumps(d))
    check(got is None, f'parse_claims(colour renamed color) = {got!r}, expected None')


def test_parse_bad_value():
    for bad in ('maybe', '', 'yes', None, 1, 0.5, True, ['depends'], {'v': 'depends'}):
        for p in ('colour', 'polarity'):
            d = dict(GOOD)
            d[p] = bad
            got = profilecheck.parse_claims(json.dumps(d))
            check(got is None, f'parse_claims({p}={bad!r}) = {got!r}, expected None')


def test_parse_no_json():
    good_text = json.dumps(GOOD)
    for text in ('The unit responds to colour and depends on shape.', '', '   ', good_text[:-1], '[1, 2, 3]',
                 'not_said'):
        got = profilecheck.parse_claims(text)
        check(got is None, f'parse_claims({text!r}) = {got!r}, expected None')


# ---------------------------------------------------------------------------------------------------------------
# verdict

def test_verdict_depends():
    cases = [(0, 'supported'), (10, 'supported'), (19.9, 'supported'), (20, 'supported'), (20.0, 'supported'),
             (np.float64(20.0), 'supported'), (20.001, 'unremarkable'), (50, 'unremarkable'),
             (79.999, 'unremarkable'), (80, 'contradicted'), (80.0, 'contradicted'),
             (np.float64(80.0), 'contradicted'), (80.1, 'contradicted'), (100, 'contradicted')]
    for p, exp in cases:
        got = profilecheck.verdict('depends', p)
        check(got == exp, f'verdict("depends", {p!r}) = {got!r}, expected {exp!r}')


def test_verdict_unaffected():
    cases = [(0, 'contradicted'), (10, 'contradicted'), (19.9, 'contradicted'), (20, 'contradicted'),
             (20.0, 'contradicted'), (np.float64(20.0), 'contradicted'), (20.001, 'unremarkable'),
             (50, 'unremarkable'), (79.999, 'unremarkable'), (80, 'supported'), (80.0, 'supported'),
             (np.float64(80.0), 'supported'), (80.1, 'supported'), (100, 'supported')]
    for p, exp in cases:
        got = profilecheck.verdict('unaffected', p)
        check(got == exp, f'verdict("unaffected", {p!r}) = {got!r}, expected {exp!r}')


def test_verdict_not_said():
    for p in (0, 10, 20, 20.0, 50, 80, 80.0, 90, 100):
        got = profilecheck.verdict('not_said', p)
        check(got is None, f'verdict("not_said", {p!r}) = {got!r}, expected None')


# ---------------------------------------------------------------------------------------------------------------
# summarize

def claims_with(**kw):
    d = {p: 'not_said' for p in PROPS}
    for k, v in kw.items():
        assert k in d, k
        d[k] = v
    return d


def pct_with(**kw):
    d = {t: 50.0 for t in TRANSFORMS}
    for k, v in kw.items():
        assert k in d, k
        d[k] = float(v)
    return d


# unit A: grayscale 10, low_pass 90, phase_scramble 20 (boundary), patch_shuffle_4x4 80 (boundary), rest 50
PCT_A = pct_with(grayscale=10, low_pass=90, phase_scramble=20, patch_shuffle_4x4=80)
# unit B: grayscale 85, low_pass 15, rest 50
PCT_B = pct_with(grayscale=85, low_pass=15)


def rec(group, condition, claims, pct):
    return {'group': group, 'condition': condition, 'claims': dict(claims), 'pct': dict(pct)}


def expected(n, n_claims, s, u, c):
    """Per-reading = totals / readings; shares = totals / claims (pooled), None when there are no claims."""
    assert s + u + c == n_claims
    out = {'readings': n, 'claims_per_reading': n_claims / n, 'supported_per_reading': s / n,
           'unremarkable_per_reading': u / n, 'contradicted_per_reading': c / n}
    if n_claims:
        out.update(supported_share=s / n_claims, unremarkable_share=u / n_claims, contradicted_share=c / n_claims)
    else:
        out.update(supported_share=None, unremarkable_share=None, contradicted_share=None)
    return out


def check_stats(got, cond, group, exp):
    if not isinstance(got, dict) or cond not in got:
        check(False, f'summarize: condition {cond!r} missing from {got!r}')
        return
    if not isinstance(got[cond], dict) or group not in got[cond]:
        check(False, f'summarize: group {group!r} missing under {cond!r}: {got[cond]!r}')
        return
    g = got[cond][group]
    for k in FIELDS:
        if not isinstance(g, dict) or k not in g:
            check(False, f'summarize[{cond}][{group}]: field {k!r} missing')
            continue
        check(close(g[k], exp[k]), f'summarize[{cond}][{group}][{k}] = {g[k]!r}, expected {exp[k]!r}')


HAND_RECORDS = [
    # 1 random/base, unit A: colour depends (p10 supported), fine_detail depends (p90 contradicted),
    #   coarse_shading unaffected (p50 unremarkable)                          -> 3 claims: S1 U1 C1
    rec('random', 'base', claims_with(colour='depends', fine_detail='depends', coarse_shading='unaffected'), PCT_A),
    # 2 random/base, unit A: shape unaffected (p20 contradicted), arrangement unaffected (p80 supported)
    #                                                                           -> 2 claims: S1 C1
    rec('random', 'base', claims_with(shape='unaffected', arrangement='unaffected'), PCT_A),
    # 3 round21/base, unit B: all not_said                                     -> 0 claims
    rec('round21', 'base', claims_with(), PCT_B),
    # 4 round21/base, unit B: colour unaffected (p85 supported), fine_detail unaffected (p15 contradicted)
    #                                                                           -> 2 claims: S1 C1
    rec('round21', 'base', claims_with(colour='unaffected', fine_detail='unaffected'), PCT_B),
    # 5 random/P, unit A: all 10 depends: colour p10 S, fine_detail p90 C, coarse_shading p50 U, shape p20 S,
    #   arrangement p80 C, orientation/upright/left_right/inner_texture/polarity p50 U   -> 10 claims: S2 U6 C2
    rec('random', 'P', {p: 'depends' for p in PROPS}, PCT_A),
    # 6 round21/P, unit B: all not_said                                        -> 0 claims
    rec('round21', 'P', claims_with(), PCT_B),
]

HAND_EXPECTED = {
    'base': {
        'random': expected(2, 5, 2, 1, 2),    # recs 1, 2: shares 0.4/0.2/0.4 (pooled; mean of ratios would differ)
        'round21': expected(2, 2, 1, 0, 1),   # recs 3, 4
        'all': expected(4, 7, 3, 1, 3),       # recs 1-4
    },
    'P': {
        'random': expected(1, 10, 2, 6, 2),   # rec 5
        'round21': expected(1, 0, 0, 0, 0),   # rec 6: no claims -> shares None
        'all': expected(2, 10, 2, 6, 2),      # recs 5, 6
    },
}


def test_summarize_hand_case():
    got = profilecheck.summarize(copy.deepcopy(HAND_RECORDS))
    for cond, groups in HAND_EXPECTED.items():
        for group, exp in groups.items():
            check_stats(got, cond, group, exp)


def test_summarize_no_claims():
    records = [rec('random', 'base', claims_with(), PCT_A), rec('random', 'base', claims_with(), PCT_B)]
    got = profilecheck.summarize(records)
    exp = expected(2, 0, 0, 0, 0)
    check_stats(got, 'base', 'random', exp)
    check_stats(got, 'base', 'all', exp)


def test_summarize_single_group_all_equals_group():
    records = [
        # unit B: colour depends (p85 contradicted), fine_detail depends (p15 supported), shape depends (p50 U)
        rec('random', 'P', claims_with(colour='depends', fine_detail='depends', shape='depends'), PCT_B),
        # unit A: arrangement depends (p80 contradicted), polarity unaffected (p50 U)
        rec('random', 'P', claims_with(arrangement='depends', polarity='unaffected'), PCT_A),
        # unit A: all not_said
        rec('random', 'P', claims_with(), PCT_A),
    ]
    got = profilecheck.summarize(records)
    exp = expected(3, 5, 1, 2, 2)
    check_stats(got, 'P', 'random', exp)
    check_stats(got, 'P', 'all', exp)


# ---------------------------------------------------------------------------------------------------------------

TESTS = [
    test_property_transform,
    test_draw_return_shape,
    test_draw_matches_reference_default_seed,
    test_draw_seed_argument,
    test_draw_deterministic,
    test_draw_structure,
    test_draw_forced_sets,
    test_parse_good,
    test_parse_missing_key,
    test_parse_extra_key,
    test_parse_bad_value,
    test_parse_no_json,
    test_verdict_depends,
    test_verdict_unaffected,
    test_verdict_not_said,
    test_summarize_hand_case,
    test_summarize_no_claims,
    test_summarize_single_group_all_equals_group,
]


def main():
    for t in TESTS:
        _CURRENT[0] = t.__name__
        try:
            t()
        except Exception as e:
            _FAILS.append((t.__name__, f'raised {type(e).__name__}: {e}'))
    if _FAILS:
        for name, msg in _FAILS:
            print(f'FAIL {name}: {msg}')
        print(f'{len(_FAILS)} failure(s)')
        sys.exit(1)
    print('tests ok')


if __name__ == '__main__':
    main()
