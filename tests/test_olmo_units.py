"""Black-box tests for bank/olmo_units.py: draw_units, tokenize, card, heldout_items, pool_windows, sim_items.

Written from authors/spec_olmo_units.interface.md only, with bank/text_units.py for the conventions it names (the doc
form (toks, text, offs), window, card_text's field format, heldout's items and truth). The implementation was not
read.

No model, GPU, torch, network or corpus: synthetic documents in text_units' doc form, synthetic activation arrays (NaN
at position 0 and at other positions, some of them on tokens that would be picked if NaN counted as a value), and a
tiny fake tokenizer object. Windows and items returned by the module are mapped back to (document, position) by
looking them up among every window text_units.window can make on the fixture documents; the fixtures are built so that
this lookup is one-to-one and so that the rules under test can tell right from wrong (the test_fixture_* tests).

Left open by the interface and so not tested: the container and order of card's `peaks` (compared as a multiset of
12 distinct strings), the form of sim truth "bin" (only that it groups the items exactly as the act / cmax bins do,
where no shortfall moves items), the SystemExit message beyond naming the kind (heldout_items is not given the unit's
name), whether 'any' held-out tokens may have a string in peaks, and whether the n_any pool windows share documents.
"""

import random
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bank"))

import olmo_units  # noqa: E402
import text_units as tu  # noqa: E402


CMAX = 6.4  # the card maximum of every fixture; synthetic activations are fractions of it times CMAX
RIGHT = 15  # tokens after the target in every window
PEAK_PIECES = [" Alpha", " BRAVO", "Charlie\n", " delta ", "\tEcho", " foxtrot", " Golf", " HOTEL", " India",
               " juliett", " Kilo", " Lima"]
PEAKS = [p.strip().lower() for p in PEAK_PIECES]  # 12 distinct strings
PLAN = {"surface": 5, "novel": 5, "near": 6, "any": 4}
FIRE_TAGS = ("surface", "novel")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _doc(pieces):
    """a text_units doc (toks, text, offs) whose token strings are `pieces`; pieces[0] is '' (the special first token)"""
    offs, n = [], 0
    for p in pieces:
        offs.append(n)
        n += len(p)
    return list(range(len(pieces))), "".join(pieces), offs


def _piece(doc, i):
    """token i's string, cut as text_units.window cuts it"""
    toks, text, offs = doc
    return text[offs[i]:offs[i + 1] if i + 1 < len(toks) else len(text)]


def _marks(a, cmax):
    """the card's marks: str(min(10, round(a / cmax * 10))) where that is at least 1; never at NaN"""
    out = []
    for x in np.asarray(a, dtype=np.float64):
        if np.isnan(x):
            out.append(None)
            continue
        v = round(float(x) / float(cmax) * 10)
        out.append(str(min(10, v)) if v >= 1 else None)
    return out


def _window_map(docs, doc_ids, left, acts=None, cmax=None, target=False):
    """every text_units.window(doc, i, left, 15, marks, target) of these documents -> (document, position); the marks
    are the card's marks when acts are given"""
    out = {}
    for d in doc_ids:
        mk = None if acts is None else _marks(acts[d], cmax)
        for i in range(len(docs[d][0])):
            w = tu.window(docs[d], i, left, RIGHT, mk, target=target)
            assert w not in out, f"fixture: window of ({d}, {i}) equals that of {out[w]}"
            out[w] = (d, i)
    return out


# ---------------------------------------------------------------------------
# draw_units
# ---------------------------------------------------------------------------

def _draw_reference(n_neurons=16, n_dirs=8, seed=33, width=11008):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_neurons):
        layer = int(rng.integers(4, 28))
        index = int(rng.integers(0, width))
        out.append({"name": f"olmo:L{layer}:n{index}", "kind": "neuron", "layer": layer, "index": index, "w": None})
    for j in range(n_dirs):
        layer = int(rng.integers(4, 28))
        w = rng.standard_normal(width)
        out.append({"name": f"olmo:L{layer}:d{j}", "kind": "direction", "layer": layer, "index": None,
                    "w": w / np.linalg.norm(w)})
    return out


def _check_units(got, want):
    got = list(got)
    assert len(got) == len(want), f"{len(got)} units, want {len(want)}"
    for k, (g, w) in enumerate(zip(got, want)):
        assert {"name", "kind", "layer", "index", "w"} <= set(g), f"unit {k}: keys {sorted(g)}"
        assert g["name"] == w["name"], f"unit {k}: name {g['name']!r} != {w['name']!r}"
        assert g["kind"] == w["kind"], f"unit {k}: kind {g['kind']!r} != {w['kind']!r}"
        assert isinstance(g["layer"], int) and g["layer"] == w["layer"], f"unit {k}: layer {g['layer']!r}"
        if w["kind"] == "neuron":
            assert isinstance(g["index"], int) and g["index"] == w["index"], f"unit {k}: index {g['index']!r}"
            assert g["w"] is None, f"unit {k}: a neuron's w is None"
        else:
            assert g["index"] is None, f"unit {k}: a direction's index is None"
            np.testing.assert_allclose(np.asarray(g["w"], dtype=np.float64), w["w"], rtol=1e-6, atol=1e-7,
                                       err_msg=f"unit {k}: w")


def test_draw_units_defaults():
    units = list(olmo_units.draw_units())
    _check_units(units, _draw_reference())
    assert [u["kind"] for u in units] == ["neuron"] * 16 + ["direction"] * 8
    assert all(np.asarray(u["w"]).shape == (11008,) for u in units[16:])


@pytest.mark.parametrize("args", [dict(n_neurons=5, n_dirs=3, seed=7, width=50),
                                  dict(n_neurons=3, n_dirs=0, seed=2, width=20),
                                  dict(n_neurons=0, n_dirs=2, seed=1, width=9)])
def test_draw_units_arguments(args):
    _check_units(olmo_units.draw_units(**args), _draw_reference(**args))


def test_draw_units_names_ranges_and_unit_norms():
    width = 300
    units = list(olmo_units.draw_units(n_neurons=40, n_dirs=10, seed=5, width=width))
    assert len(units) == 50
    for u in units[:40]:
        m = re.fullmatch(r"olmo:L(\d+):n(\d+)", u["name"])
        assert m and int(m.group(1)) == u["layer"] and int(m.group(2)) == u["index"], u["name"]
        assert 4 <= u["layer"] < 28 and 0 <= u["index"] < width
    for j, u in enumerate(units[40:]):
        assert u["name"] == f"olmo:L{u['layer']}:d{j}", u["name"]
        assert 4 <= u["layer"] < 28
        w = np.asarray(u["w"], dtype=np.float64)
        assert w.shape == (width,)
        assert np.linalg.norm(w) == pytest.approx(1.0, abs=1e-6)


def test_draw_units_deterministic_and_seeded():
    a, b = list(olmo_units.draw_units(seed=33)), list(olmo_units.draw_units(seed=33))
    assert [u["name"] for u in a] == [u["name"] for u in b]
    for u, v in zip(a[16:], b[16:]):
        assert np.array_equal(np.asarray(u["w"]), np.asarray(v["w"]))
    assert [u["name"] for u in a] != [u["name"] for u in olmo_units.draw_units(seed=34)]


# ---------------------------------------------------------------------------
# tokenize, with a fake tokenizer
# ---------------------------------------------------------------------------

class _Encoding(dict):
    """a dict with attribute access, like transformers' BatchEncoding"""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None


class FakeTokenizer:
    """Word-level stand-in for a Hugging Face fast tokenizer, called as tokenizer(text, add_special_tokens=...,
    return_offsets_mapping=..., truncation=..., max_length=...). With special tokens: BOS first and, if eos, EOS last,
    each with the empty offset span (0, 0); max_length counts them. offset_mapping only when asked for, as in HF.
    trim=False: a token spans its word and the whitespace before it; trim=True: the word only."""

    BOS, EOS = 1, 2

    def __init__(self, eos=False, trim=False):
        self.eos, self.trim = eos, trim
        self.bos_token_id, self.eos_token_id = self.BOS, self.EOS
        self.all_special_ids = [self.BOS, self.EOS]
        self.calls = []

    def spans(self, text):
        return [m.span() for m in re.finditer(r"\S+" if self.trim else r"\s*\S+", text)]

    @staticmethod
    def tid(piece):
        return 100 + zlib.crc32(piece.encode()) % 50000

    def _one(self, text, add_special_tokens, return_offsets_mapping, truncation, max_length,
             return_special_tokens_mask):
        spans = self.spans(text)
        n_special = (1 + int(self.eos)) if add_special_tokens else 0
        if max_length is not None and truncation not in (False, "do_not_truncate"):
            spans = spans[:max(0, max_length - n_special)]
        ids = [self.tid(text[s:e]) for s, e in spans]
        offsets = [(s, e) for s, e in spans]
        special = [0] * len(ids)
        if add_special_tokens:
            ids, offsets, special = [self.BOS] + ids, [(0, 0)] + offsets, [1] + special
            if self.eos:
                ids, offsets, special = ids + [self.EOS], offsets + [(0, 0)], special + [1]
        enc = _Encoding(input_ids=ids, attention_mask=[1] * len(ids))
        if return_offsets_mapping:
            enc["offset_mapping"] = offsets
        if return_special_tokens_mask:
            enc["special_tokens_mask"] = special
        return enc

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, truncation=None,
                 max_length=None, return_special_tokens_mask=False, **kw):
        args = (add_special_tokens, return_offsets_mapping, truncation, max_length, return_special_tokens_mask)
        self.calls.append(args)
        if isinstance(text, (list, tuple)):
            encs = [self._one(t, *args) for t in text]
            return _Encoding({k: [e[k] for e in encs] for k in encs[0]})
        return self._one(text, *args)


TOK_TEXT = "  Hello wörld,\tthis is a café test \U0001F642.  \n\n"
FAKES = {"bos": (False, False), "bos_trim": (False, True), "bos_eos": (True, False), "bos_eos_trim": (True, True)}


def _tok_reference(fake, text, maxtok=1024):
    """(ids, cut text, offs) by the interface's rule, for a FakeTokenizer"""
    spans = fake.spans(text)[:maxtok - 1 - int(fake.eos)]
    cut = text[:spans[-1][1]]
    ids = [fake.BOS] + [fake.tid(text[s:e]) for s, e in spans] + ([fake.EOS] if fake.eos else [])
    offs = [spans[0][0]] + [s for s, _ in spans] + ([len(cut)] if fake.eos else [])
    return ids, cut, offs


def _tokenize(fake, text, **kw):
    ids, cut, offs = olmo_units.tokenize(fake, text, **kw)
    return [int(x) for x in ids], cut, [int(x) for x in offs]


@pytest.mark.parametrize("fake", sorted(FAKES))
def test_tokenize_ids_text_offs(fake):
    tok = FakeTokenizer(*FAKES[fake])
    assert _tokenize(tok, TOK_TEXT) == _tok_reference(tok, TOK_TEXT)


def test_tokenize_special_tokens_take_the_next_start_or_the_text_end():
    ids, cut, offs = _tokenize(FakeTokenizer(), TOK_TEXT)
    assert cut == "  Hello wörld,\tthis is a café test \U0001F642."  # cut at the last token's end
    assert len(ids) == 9 and ids[0] == FakeTokenizer.BOS
    assert offs == [0, 0, 7, 14, 19, 22, 24, 29, 34]
    ids, cut, offs = _tokenize(FakeTokenizer(eos=True, trim=True), TOK_TEXT)
    assert cut == TOK_TEXT[:37]
    assert len(ids) == 10 and ids[0] == FakeTokenizer.BOS and ids[-1] == FakeTokenizer.EOS
    assert offs == [2, 2, 8, 15, 20, 23, 25, 30, 35, 37]


@pytest.mark.parametrize("maxtok", (4, 6))
@pytest.mark.parametrize("fake", sorted(FAKES))
def test_tokenize_truncates_to_maxtok(fake, maxtok):
    tok = FakeTokenizer(*FAKES[fake])
    got = _tokenize(tok, TOK_TEXT, maxtok=maxtok)
    assert len(got[0]) == maxtok == len(got[2])
    assert got == _tok_reference(tok, TOK_TEXT, maxtok)


def test_tokenize_default_maxtok_is_1024():
    tok = FakeTokenizer()
    text = " ".join(f"word{k}" for k in range(1500)) + "\n"
    got = _tokenize(tok, text)
    assert len(got[0]) == 1024 == len(got[2])
    assert got == _tok_reference(tok, text, 1024)
    assert got[1].endswith(" word1022")


def test_text_units_window_works_on_tokenize_result():
    doc = olmo_units.tokenize(FakeTokenizer(), TOK_TEXT)
    assert tu.window(doc, 2, 1, 1, target=True) == "…  Hello⟦ wörld,⟧\tthis…"
    assert tu.window(doc, 0, 0, 100) == TOK_TEXT[:37]
    doc = olmo_units.tokenize(FakeTokenizer(eos=True, trim=True), TOK_TEXT)
    assert tu.window(doc, 2, 1, 1, target=True) == "…Hello ⟦wörld,\t⟧this …"
    assert tu.window(doc, 9, 1, 0, target=True) == "…\U0001F642.⟦⟧"  # the EOS shows as an empty piece
    assert tu.window(doc, 0, 0, 100) == TOK_TEXT[2:37]
    for fake in FAKES.values():
        tok = FakeTokenizer(*fake)
        got, ref = olmo_units.tokenize(tok, TOK_TEXT), _tok_reference(tok, TOK_TEXT)
        for i in range(len(ref[0])):
            assert tu.window(got, i, 2, 3, target=True) == tu.window(ref, i, 2, 3, target=True)


# ---------------------------------------------------------------------------
# card
#
# Card documents (80 tokens): 12 "top" documents with maxima 1.00 .. 0.78 x CMAX, 2 "next" documents at 0.76 and 0.74
# (above 0.7 x cmax, outside the top 12, holding tokens inside the 0.3-0.7 band), n_mid "mid" documents with maxima
# inside the band, 6 "low" documents below it; 2 non-card documents, one at 3 x CMAX. Every document has NaN at
# position 0 (so a plain argmax lands there), right after its peak, and at one more position.
# ---------------------------------------------------------------------------

CARD_LEN = 80
TOP_F = [round(1.0 - 0.02 * k, 2) for k in range(12)]
NEXT_F, NEXT_PIECES = [0.76, 0.74], [" Mike", " NOVEMBER"]
MID_F = [0.33, 0.37, 0.41, 0.44, 0.48, 0.52, 0.56, 0.61, 0.63, 0.67, 0.69]
LOW_F = [0.27, 0.18, 0.12, 0.06, 0.02, -0.1]
NONCARD_F = [3.0, 0.5]
CID = "O-123"
SEEDS = (0, 1, 7)

CARD_HEAD = [
    "Recorded from one unit (a single neuron) in a language model, run over a broad sample of text "
    "(web pages, code, science abstracts, forums, law, Wikipedia).",
    "Each example is a window of text around the token where the unit is most active in one document. "
    "Marked tokens look like «token|activation». Activations are scaled 0-10 relative to the strongest example "
    "shown; only tokens at 1 or above are marked.",
]
TOP_HEADING = "## Strongest activations (one example per document)"
MID_HEADING = "## Examples from the middle of the activation range (30-70% of the strongest)"


def _card_doc(d, fmax, peak_piece):
    p = 6 + (17 * d) % 68  # 6..73: some windows reach the document start or end
    hi = min(0.6 * fmax, fmax - 0.05) if fmax > 0 else fmax - 0.05
    k_hi = int(np.floor(hi * 100))
    k_lo = min(-15, k_hi - 10)
    f = np.random.default_rng(1000 + d).integers(k_lo, k_hi, size=CARD_LEN) / 100 + 0.003  # all below fmax
    f[p] = fmax
    for j in (0, p + 1, 1 + (p + 23) % (CARD_LEN - 1)):
        f[j] = np.nan
    pieces = [""] + [f" c{d}x{j}" for j in range(1, CARD_LEN)]
    pieces[p] = peak_piece
    return _doc(pieces), f * CMAX, p


def _card_fixture(n_mid=11):
    roles = ([("top", k, TOP_F[k], PEAK_PIECES[k]) for k in range(12)]
             + [("next", k, NEXT_F[k], NEXT_PIECES[k]) for k in range(2)]
             + [("mid", k, MID_F[k], f" Mid{k}Word") for k in range(n_mid)]
             + [("low", k, LOW_F[k], f" Low{k}Word") for k in range(len(LOW_F))]
             + [("noncard", k, NONCARD_F[k], f" Oscar{k}") for k in range(2)])
    order = np.random.default_rng(7).permutation(len(roles))
    docs, acts, info = [], [], {"role": {}, "peak": {}}
    for d, r in enumerate(order):
        kind, k, fmax, piece = roles[r]
        doc, a, p = _card_doc(d, fmax, piece)
        docs.append(doc)
        acts.append(a)
        info["role"][d] = (kind, k)
        info["peak"][d] = p
    card_docs = [d for d in range(len(docs)) if info["role"][d][0] != "noncard"]
    return docs, acts, card_docs, info


def _card_text(cid, docs, acts, top, mid, cmax, left):
    """text_units.card_text's field format for the given (document, position) examples"""
    lines = [f"# Unit {cid}", "", CARD_HEAD[0], CARD_HEAD[1], "", TOP_HEADING]
    for n, (d, i) in enumerate(top, 1):
        lines += ["", f"### Example {n}", tu.window(docs[d], i, left, RIGHT, _marks(acts[d], cmax))]
    lines += ["", MID_HEADING]
    if not mid:
        lines += ["", "(none found)"]
    for n, (d, i) in enumerate(mid, 1):
        lines += ["", f"### Middle example {n}", tu.window(docs[d], i, left, RIGHT, _marks(acts[d], cmax))]
    return "\n".join(lines) + "\n"


def _card_reference(cid, docs, acts, card_docs, seed, left=48):
    m = {d: float(np.nanmax(acts[d])) for d in card_docs}
    arg = {d: int(np.nanargmax(acts[d])) for d in card_docs}
    top = sorted(card_docs, key=lambda d: -m[d])[:12]
    cmax = m[top[0]]
    cands = [d for d in sorted(card_docs) if 0.3 * cmax <= m[d] <= 0.7 * cmax]
    chosen = random.Random(seed).sample(cands, min(8, len(cands)))
    mid = sorted(chosen, key=lambda d: -m[d])
    text = _card_text(cid, docs, acts, [(d, arg[d]) for d in top], [(d, arg[d]) for d in mid], cmax, left)
    return {"text": text, "cmax": cmax, "top": top, "mid": mid, "cands": cands, "m": m, "arg": arg,
            "peaks": [_piece(docs[d], arg[d]).strip().lower() for d in top]}


def _run_card(n_mid=11, seed=0, left=None):
    docs, acts, card_docs, info = _card_fixture(n_mid)
    kw = {} if left is None else {"left": left}
    got = olmo_units.card(CID, docs, acts, card_docs, random.Random(seed), **kw)
    want = _card_reference(CID, docs, acts, card_docs, seed, 48 if left is None else left)
    return tuple(got), want, (docs, acts, card_docs, info)


@pytest.mark.parametrize("with_mid", (True, False))
def test_fixture_card_format_is_text_units_card_text(with_mid):
    # one NaN-free document: card_text's choices are then fixed, and _card_text must reproduce its text exactly
    n = 70
    f = np.array([(-0.2, 0.013, 0.123, 0.253, 0.093)[j % 5] for j in range(n)])
    f[40], f[50] = 1.0, 0.8
    if with_mid:
        f[20], f[25] = 0.55, 0.41
    a = f * CMAX
    docs = [_doc([""] + [f" g{j}" for j in range(1, n)])]
    text, cmax = tu.card_text(CID, docs, [(a, [None] * n)], [0], "field", random.Random(0))
    assert cmax == pytest.approx(a[40])
    assert text == _card_text(CID, docs, [a], [(0, 40)], [(0, 20)] if with_mid else [], cmax, 48)


def test_fixture_card():
    for n_mid in (11, 8, 5, 0):
        docs, acts, card_docs, info = _card_fixture(n_mid)
        want = _card_reference(CID, docs, acts, card_docs, 0)
        role, cmax = info["role"], want["cmax"]
        assert cmax == pytest.approx(CMAX)
        assert [role[d] for d in want["top"]] == [("top", k) for k in range(12)]
        assert want["cands"] == sorted(d for d in card_docs if role[d][0] == "mid") and len(want["cands"]) == n_mid
        assert sorted(want["peaks"]) == sorted(PEAKS)
        for d in card_docs:
            if role[d][0] == "next":
                assert want["m"][d] > 0.7 * cmax
                assert np.any((acts[d] >= 0.3 * cmax) & (acts[d] <= 0.7 * cmax))
        assert max(np.nanmax(acts[d]) for d in range(len(docs)) if d not in card_docs) > 2 * cmax
        for d in range(len(docs)):
            a = acts[d]
            assert np.isnan(a[0]) and int(np.argmax(a)) == 0  # a plain argmax lands on position 0
            assert info["peak"][d] == int(np.nanargmax(a)) and np.isnan(a[info["peak"][d] + 1])
        v = np.concatenate(acts) / CMAX * 10
        v = v[~np.isnan(v)]
        assert np.all(np.abs(v - np.floor(v) - 0.5) > 0.02)  # no mark sits on a rounding tie
        marks = [m for d in want["top"] for m in _marks(acts[d], cmax)]
        assert "10" in marks and None in marks and any(m not in (None, "10") for m in marks)
        _window_map(docs, range(len(docs)), 48, acts=acts, cmax=cmax)
        _window_map(docs, range(len(docs)), 5, acts=acts, cmax=cmax)


def test_card_returns_text_cmax_and_peaks():
    got, want, _ = _run_card()
    assert len(got) == 3
    text, cmax, peaks = got
    assert isinstance(text, str) and text.endswith("\n")
    assert float(cmax) == pytest.approx(want["cmax"])
    assert len(list(peaks)) == 12


def test_card_cmax_is_the_highest_card_document_maximum():
    (_, cmax, _), _, (docs, acts, card_docs, _) = _run_card()
    assert float(cmax) == pytest.approx(max(float(np.nanmax(acts[d])) for d in card_docs))
    assert float(cmax) == pytest.approx(TOP_F[0] * CMAX)  # not the non-card document at 3 x CMAX


def test_card_top_examples_are_the_12_highest_maximum_documents_at_their_argmax_strongest_first():
    (text, _, _), want, (docs, acts, card_docs, info) = _run_card()
    assert TOP_HEADING in text
    pos = []
    for n, d in enumerate(want["top"], 1):
        w = tu.window(docs[d], want["arg"][d], 48, RIGHT, _marks(acts[d], want["cmax"]))
        block = f"### Example {n}\n{w}\n"
        assert block in text, f"top example {n} (document {d}, maximum {want['m'][d]:.3f}) missing or out of place"
        pos.append(text.index(block))
    assert pos == sorted(pos)
    assert text.count("\n### Example ") == 12
    for d, (kind, _) in info["role"].items():
        if kind in ("next", "low", "noncard"):
            assert f" c{d}x" not in text, f"{kind} document {d} is shown"


@pytest.mark.parametrize("seed", SEEDS)
def test_card_middle_examples_rng_sample_in_document_order_shown_strongest_first(seed):
    (text, _, _), want, (docs, acts, card_docs, info) = _run_card(seed=seed)
    assert MID_HEADING in text
    body = text.split(MID_HEADING, 1)[1]
    pos = []
    for n, d in enumerate(want["mid"], 1):
        w = tu.window(docs[d], want["arg"][d], 48, RIGHT, _marks(acts[d], want["cmax"]))
        block = f"### Middle example {n}\n{w}\n"
        assert block in body, f"middle example {n} (document {d}) missing or out of place"
        pos.append(body.index(block))
    assert pos == sorted(pos)
    assert body.count("### Middle example ") == 8
    for d in set(want["cands"]) - set(want["mid"]):
        assert f" c{d}x" not in text, f"middle candidate {d} not drawn by rng.sample is shown"


@pytest.mark.parametrize("n_mid", (11, 8, 5, 0))
def test_card_middle_examples_up_to_8(n_mid):
    (text, _, _), want, _ = _run_card(n_mid=n_mid)
    assert MID_HEADING in text
    body = text.split(MID_HEADING, 1)[1]
    assert body.count("### Middle example ") == min(8, n_mid)
    assert ("(none found)" in body) == (n_mid == 0)
    if n_mid <= 8:  # every candidate is shown whatever the rng draws, so the whole text is fixed
        assert text == want["text"]


@pytest.mark.parametrize("seed", SEEDS)
def test_card_text_is_text_units_field_format(seed):
    (text, _, _), want, _ = _run_card(seed=seed)
    head = [f"# Unit {CID}", "", CARD_HEAD[0], CARD_HEAD[1], "", TOP_HEADING, ""]
    assert text.split("\n")[:7] == head
    assert text == want["text"]


def test_card_marks_scale_to_cmax_and_skip_nan_and_low_tokens():
    (text, _, _), want, (docs, acts, _, _) = _run_card()
    for d in want["top"] + want["mid"]:
        p = want["arg"][d]
        mark = _marks(acts[d], want["cmax"])[p]
        # the peak is marked on the 0-10 scale; its NaN neighbour right after it is shown plain
        assert f"«{_piece(docs[d], p)}|{mark}»{_piece(docs[d], p + 1)}" in text, f"document {d}"
    d0 = want["top"][0]
    assert f"«{_piece(docs[d0], want['arg'][d0])}|10»" in text
    assert text.count("«") == want["text"].count("«")
    assert "nan" not in text.lower()


def test_card_peaks_are_the_top_examples_peak_strings_stripped_lowercased():
    (_, _, peaks), want, _ = _run_card()
    peaks = list(peaks)
    assert all(isinstance(s, str) for s in peaks)
    assert sorted(peaks) == sorted(want["peaks"])


def test_card_left():
    (text, _, _), want, _ = _run_card(left=5)
    assert text == want["text"]


@pytest.mark.parametrize("seed", SEEDS)
def test_card_never_chooses_nan_positions(seed):
    (text, _, peaks), want, (docs, acts, _, _) = _run_card(seed=seed)
    for d in want["top"] + want["mid"]:
        mk = _marks(acts[d], want["cmax"])
        for j in np.flatnonzero(np.isnan(acts[d])):
            assert tu.window(docs[d], int(j), 48, RIGHT, mk) not in text, f"document {d}: example at NaN position {j}"
        assert tu.window(docs[d], want["arg"][d], 48, RIGHT, mk) in text
    assert "" not in list(peaks)


# ---------------------------------------------------------------------------
# heldout_items
#
# Held-out documents by kind: "surface" (tokens >= 0.3 x cmax whose string is in peaks), "novel" (>= 0.3 x cmax, not
# in peaks), "near" (<= 0.05 x cmax, in peaks), "any" (120 tokens, all <= 0.05 x cmax), "anyfew" (only a few low
# tokens). Other tokens of surface / novel / near / anyfew documents sit at 0.15 x cmax (neither fire nor no). There
# are tokens just past a threshold (0.28 and 0.27 x cmax at positions 2 and 26 of surface / novel documents, 0.07 x
# cmax at position 26 of near documents and every 10th position from 5 of any documents), and NaN at positions 0, 9
# and 17 of every document (9 and 17 on tokens whose string is in peaks). Some documents of every kind are not
# held-out documents.
# ---------------------------------------------------------------------------

HO_LEN, HO_ANY_LEN = 30, 120
LOW_GRID = (-0.2, -0.05, 0.0, 0.012, 0.031, 0.044)
HO_SCENARIOS = {  # kind: (held-out documents, candidate tokens per document)
    "plenty": {"surface": (10, 3), "novel": (10, 3), "near": (10, 3), "any": (30, 0)},
    "surface_short": {"surface": (3, 1), "novel": (10, 3), "near": (10, 3), "any": (30, 0)},
    "novel_short": {"surface": (10, 3), "novel": (2, 1), "near": (10, 3), "any": (30, 0)},
    "near_short": {"surface": (10, 3), "novel": (10, 3), "near": (2, 1), "any": (30, 0)},
    "any_short": {"surface": (10, 3), "novel": (10, 3), "near": (12, 1), "anyfew": (2, 2)},
    "fire_short": {"surface": (3, 1), "novel": (4, 1), "near": (10, 3), "any": (30, 0)},
    "no_short": {"surface": (10, 3), "novel": (10, 3), "near": (2, 1), "anyfew": (1, 3)},
}
HO_TRAPS = {"surface": (4, 3), "novel": (3, 3), "near": (4, 3), "any": (2, 0)}  # not held-out documents


def _ho_doc(d, kind, n_cand):
    n = HO_ANY_LEN if kind == "any" else HO_LEN
    pieces = [""] + [f" h{d}x{j}" for j in range(1, n)]
    if kind == "any":
        f = np.array([LOW_GRID[(7 * j + d) % len(LOW_GRID)] for j in range(n)])
        f[5::10] = 0.07  # just above the 'no' threshold
    else:
        f = np.full(n, 0.15)
    for k, p in enumerate(range(4, 4 + 8 * n_cand, 8)):
        if kind == "surface":
            pieces[p], f[p] = PEAK_PIECES[(d + k) % 12], (0.8, 1.4, 0.45)[k % 3]
        elif kind == "novel":
            pieces[p], f[p] = f" Nova{d}k{k}", (0.6, 0.35, 1.2)[k % 3]
        elif kind == "near":
            pieces[p], f[p] = PEAK_PIECES[(d + k) % 12], (0.02, -0.3, 0.04)[k % 3]
        elif kind == "anyfew":
            f[p] = (0.01, -0.1, 0.03)[k % 3]
    if kind in ("surface", "novel"):
        pieces[2], f[2] = PEAK_PIECES[(d + 5) % 12], 0.28  # just below the fire threshold, string in peaks
        pieces[26], f[26] = f" Almost{d}", 0.27  # just below the fire threshold, novel string
    if kind == "near":
        pieces[26], f[26] = PEAK_PIECES[(d + 7) % 12], 0.07  # just above the no threshold, string in peaks
    f[0] = np.nan
    for p in (9, 17):  # NaN on tokens whose string is in peaks
        pieces[p], f[p] = PEAK_PIECES[(d + 3) % 12], np.nan
    return _doc(pieces), f * CMAX


def _ho_fixture(scenario):
    specs = [(kind, c, True) for kind, (n, c) in HO_SCENARIOS[scenario].items() for _ in range(n)]
    specs += [(kind, c, False) for kind, (n, c) in HO_TRAPS.items() for _ in range(n)]
    order = np.random.default_rng(11).permutation(len(specs))
    docs, acts, ho_docs, kinds = [], [], [], {}
    for d, s in enumerate(order):
        kind, n_cand, held_out = specs[s]
        doc, a = _ho_doc(d, kind, n_cand)
        docs.append(doc)
        acts.append(a)
        kinds[d] = kind
        if held_out:
            ho_docs.append(d)
    return docs, acts, ho_docs, kinds


def _ho_candidates(docs, acts, doc_ids):
    """counting tokens by the plan's rules; 'any' here is every counting token at or below 0.05 x cmax"""
    c = {t: [] for t in PLAN}
    for d in doc_ids:
        for i, x in enumerate(acts[d]):
            if np.isnan(x):
                continue
            s = _piece(docs[d], i).strip().lower()
            if x >= 0.3 * CMAX:
                c["surface" if s in PEAKS else "novel"].append((d, i))
            elif x <= 0.05 * CMAX:
                if s in PEAKS:
                    c["near"].append((d, i))
                c["any"].append((d, i))
    return c


def _tag(t):
    return str(t["tag"]).split(":")[-1]  # the tag's form has its own test; elsewhere 'fire:surface' reads as 'surface'


def _run_heldout(scenario, seed=3, left=None):
    docs, acts, ho_docs, kinds = _ho_fixture(scenario)
    kw = {} if left is None else {"left": left}
    items, truth = olmo_units.heldout_items(docs, acts, ho_docs, CMAX, list(PEAKS), random.Random(seed), **kw)
    return list(items), list(truth), (docs, acts, ho_docs, kinds)


def _check_heldout(items, truth, docs, acts, ho_docs, left=48):
    """map items to (document, position), check every per-item rule; returns [(d, i, tag)] in id order"""
    wmap = _window_map(docs, range(len(docs)), left, target=True)
    assert [it["id"] for it in items] == list(range(1, 21))
    assert [t["id"] for t in truth] == list(range(1, 21))
    rows = []
    for it, t in zip(items, truth):
        n = it["id"]
        assert set(it) == {"id", "text"}, f"item {n}: keys {sorted(it)}"
        assert {"id", "fires", "tag", "act"} <= set(t), f"truth {n}: keys {sorted(t)}"
        assert it["text"] in wmap, f"item {n} is not window(doc, i, {left}, 15, target=True) of a fixture document"
        d, i = wmap[it["text"]]
        a, tag = float(acts[d][i]), _tag(t)
        assert d in ho_docs, f"item {n}: document {d} is not a held-out document"
        assert not np.isnan(a), f"item {n}: position {i} of document {d} never counts"
        assert tag in PLAN, f"item {n}: tag {t['tag']!r}"
        assert bool(t["fires"]) == (tag in FIRE_TAGS), f"item {n}: fires {t['fires']!r} with tag {tag}"
        assert float(t["act"]) == pytest.approx(a, abs=5e-4), f"item {n}: act {t['act']} != {a}"
        s = _piece(docs[d], i).strip().lower()
        if tag in FIRE_TAGS:
            assert a >= 0.3 * CMAX, f"item {n} ({tag}): act {a} below 0.3 x cmax"
        else:
            assert a <= 0.05 * CMAX, f"item {n} ({tag}): act {a} above 0.05 x cmax"
        if tag in ("surface", "near"):
            assert s in PEAKS, f"item {n} ({tag}): {s!r} not in peaks"
        if tag == "novel":
            assert s not in PEAKS, f"item {n} (novel): {s!r} in peaks"
        rows.append((d, i, tag))
    return rows


HO_GUARD = {  # scenario: (surface, novel, near) candidate tokens in held-out documents, bounds on low tokens
    "plenty": ((30, 30, 30), 3000, None),
    "surface_short": ((3, 30, 30), 3000, None),
    "novel_short": ((30, 2, 30), 3000, None),
    "near_short": ((30, 30, 2), 3000, None),
    "any_short": ((30, 30, 12), 16, 16),
    "fire_short": ((3, 4, 30), 3000, None),
    "no_short": ((30, 30, 2), 5, 5),
}


def test_fixture_heldout():
    assert set(HO_GUARD) == set(HO_SCENARIOS)
    for scen, (counts, lo, hi) in HO_GUARD.items():
        docs, acts, ho_docs, kinds = _ho_fixture(scen)
        c = _ho_candidates(docs, acts, ho_docs)
        assert (len(c["surface"]), len(c["novel"]), len(c["near"])) == counts, scen
        assert lo <= len(c["any"]) and (hi is None or len(c["any"]) <= hi), (scen, len(c["any"]))
        for t in ("surface", "novel", "near"):  # a short tag has one candidate per document
            if len(c[t]) < 10:
                assert len({d for d, _ in c[t]}) == len(c[t]), (scen, t)
        other = _ho_candidates(docs, acts, [d for d in range(len(docs)) if d not in ho_docs])
        assert all(other[t] for t in PLAN), scen
        for d in range(len(docs)):
            assert np.isnan(acts[d][0]) and np.isnan(acts[d][9]) and np.isnan(acts[d][17])
            assert _piece(docs[d], 9).strip().lower() in PEAKS
        a = np.concatenate([acts[d] for d in ho_docs])
        assert np.any((a > 0.25 * CMAX) & (a < 0.3 * CMAX)) and np.any((a > 0.05 * CMAX) & (a < 0.1 * CMAX))
        for left in (48, 6):
            _window_map(docs, range(len(docs)), left, target=True)


@pytest.mark.parametrize("seed", (3, 4))
def test_heldout_items_plan_counts_and_tags(seed):
    items, truth, (docs, acts, ho_docs, _) = _run_heldout("plenty", seed)
    rows = _check_heldout(items, truth, docs, acts, ho_docs)
    assert Counter(tag for _, _, tag in rows) == Counter(PLAN)


def test_heldout_truth_fields_as_text_units_heldout_returns_them():
    items, truth, (docs, acts, ho_docs, _) = _run_heldout("plenty")
    wmap = _window_map(docs, range(len(docs)), 48, target=True)
    for it, t in zip(items, truth):
        assert it["text"] in wmap
        d, i = wmap[it["text"]]
        assert t["tag"] in PLAN, f"tag {t['tag']!r}: text_units.heldout keeps the part after 'fire:' / 'no:'"
        assert isinstance(t["fires"], bool), f"fires {t['fires']!r}"
        assert t["act"] == pytest.approx(round(float(acts[d][i]), 3), abs=1e-12), "act rounded to 3 decimals"


@pytest.mark.parametrize("seed", (3, 4, 5))
def test_heldout_thresholds_and_peak_strings(seed):
    # fire >= 0.3 x cmax, no <= 0.05 x cmax, surface / near in peaks, novel not (per item, in _check_heldout);
    # the tokens just past a threshold are never chosen
    items, truth, (docs, acts, ho_docs, kinds) = _run_heldout("plenty", seed)
    for d, i, tag in _check_heldout(items, truth, docs, acts, ho_docs):
        trap = (i % 10 == 5) if kinds[d] == "any" else (i in (2, 26))
        assert not trap, f"threshold trap ({d}, {i}) chosen as {tag}"


@pytest.mark.parametrize("scenario", ("plenty", "surface_short", "novel_short", "near_short", "any_short"))
def test_heldout_one_item_per_document_within_a_tag(scenario):
    items, truth, (docs, acts, ho_docs, _) = _run_heldout(scenario)
    rows = _check_heldout(items, truth, docs, acts, ho_docs)
    for tag in PLAN:
        ds = [d for d, _, t in rows if t == tag]
        assert len(ds) == len(set(ds)), f"{tag}: two items from one document"


SHORTFALL = {
    "surface_short": {"surface": 3, "novel": 7, "near": 6, "any": 4},
    "novel_short": {"surface": 8, "novel": 2, "near": 6, "any": 4},
    "near_short": {"surface": 5, "novel": 5, "near": 2, "any": 8},
}


@pytest.mark.parametrize("scenario", sorted(SHORTFALL))
def test_heldout_shortfall_taken_from_the_other_tag_of_the_kind(scenario):
    items, truth, (docs, acts, ho_docs, _) = _run_heldout(scenario)
    rows = _check_heldout(items, truth, docs, acts, ho_docs)  # items keep the tag they were found under
    assert Counter(tag for _, _, tag in rows) == Counter(SHORTFALL[scenario])
    short = [t for t in PLAN if SHORTFALL[scenario][t] < PLAN[t]][0]
    c = _ho_candidates(docs, acts, ho_docs)
    assert {(d, i) for d, i, t in rows if t == short} == set(c[short]), f"{short}: not all its candidates used"


def test_heldout_shortfall_of_any_taken_from_near():
    # few low tokens outside the near ones: the 1-in-100 sampling of 'any' finds fewer than 4
    items, truth, (docs, acts, ho_docs, _) = _run_heldout("any_short")
    n = Counter(tag for _, _, tag in _check_heldout(items, truth, docs, acts, ho_docs))
    assert n["surface"] == 5 and n["novel"] == 5
    assert n["near"] + n["any"] == 10
    assert n["any"] < 4 and n["near"] > 6


def test_heldout_raises_systemexit_naming_the_kind_when_fire_is_short():
    docs, acts, ho_docs, _ = _ho_fixture("fire_short")  # 7 fire tokens in all
    with pytest.raises(SystemExit) as e:
        olmo_units.heldout_items(docs, acts, ho_docs, CMAX, list(PEAKS), random.Random(3))
    assert "fire" in str(e.value), str(e.value)


def test_heldout_raises_systemexit_naming_the_kind_when_no_is_short():
    docs, acts, ho_docs, _ = _ho_fixture("no_short")  # 5 low tokens in all
    with pytest.raises(SystemExit) as e:
        olmo_units.heldout_items(docs, acts, ho_docs, CMAX, list(PEAKS), random.Random(3))
    assert re.search(r"\bno\b", str(e.value)), str(e.value)


def test_heldout_left():
    items, truth, (docs, acts, ho_docs, _) = _run_heldout("plenty", left=6)
    rows = _check_heldout(items, truth, docs, acts, ho_docs, left=6)
    assert Counter(tag for _, _, tag in rows) == Counter(PLAN)


def test_heldout_items_shuffled_and_seeded():
    a = _run_heldout("plenty", 3)[:2]
    assert a == _run_heldout("plenty", 3)[:2]
    assert a != _run_heldout("plenty", 4)[:2]
    in_plan_order = ([True] * 10 + [False] * 10, [False] * 10 + [True] * 10)
    orders = [[bool(t["fires"]) for t in _run_heldout("plenty", s)[1]] for s in (3, 4)]
    assert any(o not in in_plan_order for o in orders)


@pytest.mark.parametrize("seed", range(5))
def test_heldout_never_chooses_nan_positions(seed):
    items, truth, (docs, acts, ho_docs, _) = _run_heldout("plenty", seed)
    for d, i, _ in _check_heldout(items, truth, docs, acts, ho_docs):
        assert i not in (0, 9, 17) and not np.isnan(acts[d][i])


# ---------------------------------------------------------------------------
# pool_windows
#
# Pool documents (80 tokens): 8 "fire" documents, each with 3 tokens >= 0.3 x cmax (1.3, 0.9, 0.5 x CMAX; the first
# is marked 10), and 20 "other" documents with every token below 0.3 x cmax and NaN at half their positions.
# 3 more fire documents are not pool documents.
# ---------------------------------------------------------------------------

POOL_LEN = 80
POOL_LOW = tuple(k / 100 + 0.003 for k in range(-20, 28, 3))  # -0.197 .. 0.253


def _pool_doc(d, kind):
    f = np.array([POOL_LOW[(5 * j + d) % len(POOL_LOW)] for j in range(POOL_LEN)])
    f[0] = np.nan
    if kind == "fire":
        p = 5 + (13 * d) % 30
        for k, x in enumerate((1.3, 0.9, 0.5)):
            f[p + 20 * k] = x
        f[p + 1] = f[77] = np.nan
    else:
        f[1::2] = np.nan
    pieces = [""] + [f" p{d}x{j}" for j in range(1, POOL_LEN)]
    return _doc(pieces), f * CMAX


def _pool_fixture():
    specs = [("fire", True)] * 8 + [("other", True)] * 20 + [("fire", False)] * 3
    order = np.random.default_rng(13).permutation(len(specs))
    docs, acts, pool_docs, kinds = [], [], [], {}
    for d, s in enumerate(order):
        kind, in_pool = specs[s]
        doc, a = _pool_doc(d, kind)
        docs.append(doc)
        acts.append(a)
        kinds[d] = kind
        if in_pool:
            pool_docs.append(d)
    return docs, acts, pool_docs, kinds


def _run_pool(seed=5, **kw):
    docs, acts, pool_docs, kinds = _pool_fixture()
    wins = olmo_units.pool_windows(docs, acts, pool_docs, CMAX, random.Random(seed), **kw)
    return list(wins), (docs, acts, pool_docs, kinds)


def _check_pool(wins, docs, acts, pool_docs, left=48):
    wmap = _window_map(docs, range(len(docs)), left, acts=acts, cmax=CMAX)
    rows = []
    for k, w in enumerate(wins):
        assert isinstance(w, str), f"window {k}: {type(w)}"
        assert w in wmap, f"window {k} is not text_units.window(doc, i, {left}, 15, card marks) of a fixture document"
        d, i = wmap[w]
        assert d in pool_docs, f"window {k}: document {d} is not a pool document"
        assert not np.isnan(acts[d][i]), f"window {k}: position {i} of document {d} never counts"
        rows.append((d, i))
    return rows


def test_fixture_pool():
    docs, acts, pool_docs, kinds = _pool_fixture()
    assert sum(kinds[d] == "fire" for d in pool_docs) == 8
    for d in range(len(docs)):
        a = acts[d]
        assert np.isnan(a[0])
        if kinds[d] == "fire":
            assert int(np.sum(a >= 0.3 * CMAX)) == 3 and np.nanmax(a) > CMAX
        else:
            assert np.nanmax(a) < 0.3 * CMAX and np.isnan(a).mean() > 0.5
    assert any(kinds[d] == "fire" for d in range(len(docs)) if d not in pool_docs)
    for left in (48, 10):
        _window_map(docs, range(len(docs)), left, acts=acts, cmax=CMAX)


@pytest.mark.parametrize("seed", (5, 6))
def test_pool_windows_fire_one_per_document_and_any_from_other_documents(seed):
    wins, (docs, acts, pool_docs, kinds) = _run_pool(seed)
    assert len(wins) == 15
    rows = _check_pool(wins, docs, acts, pool_docs)
    fire = [(d, i) for d, i in rows if acts[d][i] >= 0.3 * CMAX]
    rest = [(d, i) for d, i in rows if acts[d][i] < 0.3 * CMAX]
    assert len(fire) == 8 and len(rest) == 7
    fire_docs = [d for d, _ in fire]
    assert len(set(fire_docs)) == 8, "one fire window per document"
    assert set(fire_docs) == {d for d in pool_docs if kinds[d] == "fire"}
    assert not {d for d, _ in rest} & set(fire_docs), "the n_any windows come from other documents"


def test_pool_windows_counts_follow_n_fire_and_n_any():
    wins, (docs, acts, pool_docs, _) = _run_pool(5, n_fire=8, n_any=3)
    assert len(wins) == 11
    rows = _check_pool(wins, docs, acts, pool_docs)
    assert sum(bool(acts[d][i] >= 0.3 * CMAX) for d, i in rows) == 8


def test_pool_windows_left():
    wins, (docs, acts, pool_docs, _) = _run_pool(5, left=10)
    assert len(_check_pool(wins, docs, acts, pool_docs, left=10)) == 15


def test_pool_windows_marks_are_the_cards_scaled_to_cmax():
    # the lookup in _check_pool already requires the exact marks; values above cmax show as 10, NaN never shows
    seen10 = False
    for seed in (5, 6, 7):
        wins, (docs, acts, pool_docs, _) = _run_pool(seed)
        _check_pool(wins, docs, acts, pool_docs)
        for w in wins:
            seen10 |= "|10»" in w
            assert "nan" not in w.lower()
    assert seen10


def test_pool_windows_shuffled_and_seeded():
    a, _ = _run_pool(5)
    assert a == _run_pool(5)[0]
    assert a != _run_pool(6)[0]
    in_plan_order = ([True] * 8 + [False] * 7, [False] * 7 + [True] * 8)
    orders = []
    for seed in (5, 6, 7):
        wins, (docs, acts, pool_docs, _) = _run_pool(seed)
        orders.append([bool(acts[d][i] >= 0.3 * CMAX) for d, i in _check_pool(wins, docs, acts, pool_docs)])
    assert any(o not in in_plan_order for o in orders)


@pytest.mark.parametrize("seed", range(4))
def test_pool_windows_never_choose_nan_positions(seed):
    wins, (docs, acts, pool_docs, _) = _run_pool(seed)
    for d, i in _check_pool(wins, docs, acts, pool_docs):
        assert i != 0 and not np.isnan(acts[d][i])


# ---------------------------------------------------------------------------
# sim_items
#
# Test documents (40 tokens), either "mixed" (tokens in every act / cmax bin, at least 3 counting tokens per bin) or
# holding only the tokens of one bin. NaN at position 0 and at every 7th position from 3. No act / cmax lies near a
# bin edge. Some mixed, top-bin and second-bin documents are not test documents.
# ---------------------------------------------------------------------------

SIM_LEN = 40
SIM_VALUES = ((-0.4, 0.0, 0.013, 0.031, 0.046), (0.057, 0.09, 0.13, 0.17, 0.193), (0.21, 0.27, 0.33, 0.38),
              (0.41, 0.47, 0.53, 0.58), (0.61, 0.67, 0.73, 0.79), (0.81, 0.9, 1.0, 1.35))  # act / cmax, per bin
SIM_EDGES = (0.05, 0.2, 0.4, 0.6, 0.8)
SIM_PLAN = (10, 6, 6, 6, 6, 6)
SIM_COUNTS = {0: 14, 1: 9, 2: 10, 3: 9, 4: 12, 5: 9}  # one-bin test documents per bin
SIM_TRAPS = ["mixed"] * 4 + [5] * 4 + [4] * 2  # not test documents
SIM_SHORT = {
    "none": ({}, SIM_PLAN),
    "top_short": ({5: 2}, (10, 6, 6, 6, 10, 2)),
    "mid_short": ({3: 3}, (10, 6, 9, 3, 6, 6)),
}


def _bin(x):
    """0: act < 0.05 x cmax; 1..5: the bins [0.05, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, inf)"""
    return sum(float(x) / CMAX >= e for e in SIM_EDGES)


def _sim_doc(d, kind):
    f = np.empty(SIM_LEN)
    for j in range(SIM_LEN):
        vals = SIM_VALUES[j % 6 if kind == "mixed" else kind]
        f[j] = vals[(j // 6 + d) % len(vals)]
    f[0] = np.nan
    f[3::7] = np.nan
    pieces = [""] + [f" s{d}x{j}" for j in range(1, SIM_LEN)]
    return _doc(pieces), f * CMAX


def _sim_fixture(counts=None):
    """counts None: 60 mixed test documents; else {bin: number of test documents holding only that bin}"""
    if counts is None:
        specs = [("mixed", True)] * 60
    else:
        specs = [(b, True) for b, n in sorted(counts.items()) for _ in range(n)]
    specs += [(kind, False) for kind in SIM_TRAPS]
    order = np.random.default_rng(17).permutation(len(specs))
    docs, acts, test_docs, kinds = [], [], [], {}
    for d, s in enumerate(order):
        kind, is_test = specs[s]
        doc, a = _sim_doc(d, kind)
        docs.append(doc)
        acts.append(a)
        kinds[d] = kind
        if is_test:
            test_docs.append(d)
    return docs, acts, test_docs, kinds


def _run_sim(counts=None, seed=9, left=None):
    docs, acts, test_docs, kinds = _sim_fixture(counts)
    kw = {} if left is None else {"left": left}
    items, truth = olmo_units.sim_items(docs, acts, test_docs, CMAX, random.Random(seed), **kw)
    return list(items), list(truth), (docs, acts, test_docs, kinds)


def _check_sim(items, truth, docs, acts, test_docs, left=48):
    """map items to (document, position), check the per-item rules; returns [(d, i, act bin, truth bin)]"""
    wmap = _window_map(docs, range(len(docs)), left, target=True)
    assert [it["id"] for it in items] == list(range(1, 41))
    assert [t["id"] for t in truth] == list(range(1, 41))
    rows = []
    for it, t in zip(items, truth):
        n = it["id"]
        assert set(it) == {"id", "text"}, f"item {n}: keys {sorted(it)}"
        assert {"id", "act", "bin"} <= set(t), f"truth {n}: keys {sorted(t)}"
        assert it["text"] in wmap, f"item {n} is not window(doc, i, {left}, 15, target=True) of a fixture document"
        d, i = wmap[it["text"]]
        a = float(acts[d][i])
        assert d in test_docs, f"item {n}: document {d} is not a test document"
        assert not np.isnan(a), f"item {n}: position {i} of document {d} never counts"
        assert float(t["act"]) == pytest.approx(a, abs=5e-4), f"item {n}: act {t['act']} != {a}"
        rows.append((d, i, _bin(a), t["bin"]))
    return rows


def test_fixture_sim():
    docs, acts, test_docs, kinds = _sim_fixture()
    assert len(test_docs) == 60
    for d in range(len(docs)):
        a = acts[d]
        assert np.isnan(a[0]) and np.isnan(a).sum() >= 5
        if kinds[d] == "mixed":
            bins = Counter(_bin(x) for x in a if not np.isnan(x))
            assert all(bins[b] >= 3 for b in range(6)), (d, bins)
    for x in np.concatenate(acts):
        if not np.isnan(x):
            assert min(abs(float(x) / CMAX - e) for e in SIM_EDGES) > 0.003
    for left in (48, 3):
        _window_map(docs, range(len(docs)), left, target=True)
    for override, _ in SIM_SHORT.values():
        counts = {**SIM_COUNTS, **override}
        docs, acts, test_docs, kinds = _sim_fixture(counts)
        assert Counter(kinds[d] for d in test_docs) == Counter(counts)
        for d in range(len(docs)):
            if kinds[d] != "mixed":
                assert {_bin(x) for x in acts[d] if not np.isnan(x)} == {kinds[d]}
        assert any(kinds[d] == 5 for d in range(len(docs)) if d not in test_docs)
        _window_map(docs, range(len(docs)), 48, target=True)


@pytest.mark.parametrize("seed", (9, 10))
def test_sim_items_ten_low_then_six_per_bin(seed):
    items, truth, (docs, acts, test_docs, _) = _run_sim(seed=seed)
    rows = _check_sim(items, truth, docs, acts, test_docs)
    assert Counter(r[2] for r in rows) == Counter(dict(enumerate(SIM_PLAN)))


@pytest.mark.parametrize("seed", (9, 10))
def test_sim_items_at_most_one_per_document(seed):
    items, truth, (docs, acts, test_docs, _) = _run_sim(seed=seed)
    ds = [r[0] for r in _check_sim(items, truth, docs, acts, test_docs)]
    assert len(set(ds)) == 40


def test_sim_truth_bins_group_items_like_the_act_bins():
    items, truth, (docs, acts, test_docs, _) = _run_sim()
    groups = {}
    for _, _, b, tb in _check_sim(items, truth, docs, acts, test_docs):
        groups.setdefault(b, set()).add(repr(tb))
    assert sorted(groups) == list(range(6))
    assert all(len(v) == 1 for v in groups.values()), groups
    assert len({next(iter(v)) for v in groups.values()}) == 6, groups


@pytest.mark.parametrize("case", sorted(SIM_SHORT))
def test_sim_items_shortfall_from_the_next_lower_bin(case):
    override, want = SIM_SHORT[case]
    items, truth, (docs, acts, test_docs, kinds) = _run_sim(counts={**SIM_COUNTS, **override})
    rows = _check_sim(items, truth, docs, acts, test_docs)
    assert Counter(r[2] for r in rows) == Counter(dict(enumerate(want)))
    assert len({r[0] for r in rows}) == 40
    for b in range(6):
        if want[b] < SIM_PLAN[b]:
            assert {r[0] for r in rows if r[2] == b} == {d for d in test_docs if kinds[d] == b}


def test_sim_items_left():
    items, truth, (docs, acts, test_docs, _) = _run_sim(left=3)
    rows = _check_sim(items, truth, docs, acts, test_docs, left=3)
    assert Counter(r[2] for r in rows) == Counter(dict(enumerate(SIM_PLAN)))


def test_sim_items_shuffled_and_seeded():
    a = _run_sim(seed=9)[:2]
    assert a == _run_sim(seed=9)[:2]
    assert a != _run_sim(seed=10)[:2]
    items, truth, (docs, acts, test_docs, _) = _run_sim(seed=9)
    bins = [r[2] for r in _check_sim(items, truth, docs, acts, test_docs)]
    assert bins != sorted(bins) and bins != sorted(bins, reverse=True)


@pytest.mark.parametrize("seed", range(4))
def test_sim_items_never_choose_nan_positions(seed):
    items, truth, (docs, acts, test_docs, _) = _run_sim(seed=seed)
    for d, i, _, _ in _check_sim(items, truth, docs, acts, test_docs):
        assert i != 0 and i % 7 != 3 and not np.isnan(acts[d][i])


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def test_selftest_prints_ok(capsys):
    olmo_units.selftest()
    assert "selftest ok" in capsys.readouterr().out


def test_main_selftest_flag(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["olmo_units.py", "--selftest"])
    try:
        olmo_units.main()
    except SystemExit as e:
        assert e.code in (None, 0), e.code
    assert "selftest ok" in capsys.readouterr().out
