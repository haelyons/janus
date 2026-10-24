"""Black-box tests for tools/run_text_arms.py.

Written from authors/spec_run_text_arms.interface.md only, with tools/run_subjects.py (turns_for, TASK),
tools/claude_call.py (converse, parse_json, MODEL) and bank/text_units.py (window, tok) for the functions it names.
The implementation was not read.

No Claude call, no network, no server:
- claude_call.converse, and any name in run_text_arms bound to it, is replaced by a fake that plays one scripted list
  of replies per conversation, calls callable turns with the replies so far (None ends the conversation) as
  claude_call.converse does, and records the system prompt and every turn sent. As a safety net the real claude
  subprocess and every socket connection raise.
- run_text_arms.measure is replaced by a fake: whitespace tokens (toks a list as long as offs, as for the server) and
  a fixed activation per word (FRAC * SCALE; "void" is NaN).
- Every module-level path under the repository in run_subjects and run_text_arms is pointed at the same place in a
  temporary copy of the layout (bank/cards, bank/heldout, bank/KEY_r33.json, bank/KEY_olmo.json, arms/P1_active.txt,
  runs/), the working directory is that copy, and run_text_arms.OFFER (if it is a string) is set to the test offer.
  After each test nothing may exist under the real runs/ for the test cards.

Sessions run through main() (the command line) with one worker and --samples 1, so each record is s1.json (samples
numbered from 1, as run_subjects numbers them). Expected turns come from run_subjects.turns_for on the temporary bank,
the interface's formulas and text_units.window; the fixed strings (POOL_INTRO, the P1 replies, SIM) are typed out here.

Choices the interface leaves open are not tested: the order of "first 5" and "drop bad entries" (every request here
has at most 5 entries or only good ones), a request with no usable entry, a dict with both show and class, whether the
premature-answer reply counts toward a run of malformed replies, the form of loop replies (a string or text blocks:
only their text is compared), what A0 records under requests/results, and the wording of output lines beyond the words
done / skip / FAIL.
"""

import copy
import importlib
import json
import math
import os
import re
import socket
import sys
import threading
from pathlib import Path, PurePath

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import claude_call  # noqa: E402
import run_subjects  # noqa: E402
import run_text_arms  # noqa: E402

if str(ROOT / "bank") not in sys.path:
    sys.path.append(str(ROOT / "bank"))
import text_units  # noqa: E402

REAL_CONVERSE = claude_call.converse

# ---------------------------------------------------------------------------
# fixture data
# ---------------------------------------------------------------------------

CID, CID2 = "TT-9001", "TT-9002"  # test-only card ids (a planted-rule unit and a server unit)
CARDS = (CID, CID2)
ARMS = ("A0", "F2_pool", "P1_active", "Y1_yoked")
DEFAULT_SERVER = "http://127.0.0.1:8765"

CMAX = {CID: 4.2, CID2: 2.1}  # card_max per card: the same activations mark differently on the two cards
SCALE = 4.2  # the fake unit's activation of a word is FRAC[word] * SCALE (0 for other words)
FRAC = {"coffee": 1.0, "espresso": 1.3, "cafe": 0.67, "java": 0.43, "mug": 0.08, "tea": 0.03, "void": float("nan")}

KEYS = {
    CID: {"unit": "rule_tt", "format": "field", "card_max": CMAX[CID],
          "pool": ["first «coffee|10» window", "second« cafe|7» window", "third window, nothing marked"],
          "sim": [{"id": 1, "text": "a ⟦coffee⟧ cup"}, {"id": 2, "text": "the ⟦tea⟧ pot"},
                  {"id": 3, "text": "java ⟦island⟧ trip"}],
          "truth_class": "one_plus_related", "rule": "not used by this tool"},
    CID2: {"unit": "olmo:L7/N1234", "format": "field", "card_max": CMAX[CID2],
           "pool": ["olmo pool «espresso|10» one", "olmo pool two"],
           "sim": [{"id": 1, "text": "x ⟦y⟧ z"}, {"id": 2, "text": "p ⟦q⟧ r"}],
           "truth_class": "unrelated"},
}
LAYOUTS = {  # which key file holds which card ("either may be absent", merged)
    "r33": {"KEY_r33.json": (CID, CID2)},
    "olmo": {"KEY_olmo.json": (CID, CID2)},
    "split": {"KEY_r33.json": (CID,), "KEY_olmo.json": (CID2,)},
}
CARD_TEXT = {
    CID: "# Unit TT-9001\n\nTest card one.\n\n### Example 1\nI drink« coffee|10» daily\n",
    CID2: "# Unit TT-9002\n\nTest card two.\n\n### Example 1\nan« espresso|10» shot\n",
}
HELDOUT = {
    CID: [{"id": 1, "text": "a ⟦coffee⟧ cup"}, {"id": 2, "text": "the ⟦tea⟧ pot"}],
    CID2: [{"id": 1, "text": "one ⟦espresso⟧ shot"}, {"id": 2, "text": "the ⟦door⟧ shut"}],
}
OFFER_FILE = '\n  You may run experiments before answering (test offer): reply {"show": ["text", ...]}.  \n\n'
OFFER = OFFER_FILE.strip()

# fixed strings from the interface
POOL_INTRO = "\n\nBelow is the same unit on further texts, marked the same way and on the same scale as above."
PREMATURE = "Run at least one experiment before answering. Reply with only a request in the format above."
ONLY_ANSWER = "Reply with only the JSON answer described above."
SIM = ("Below is a description of what one unit (a single neuron) in a language model responds to, then 40 text "
       "excerpts, each with one token marked ⟦like this⟧. Using only the description, predict the unit's activation "
       "on each marked token, from 0 (not active) to 10 (as active as it ever gets). Reply with only JSON: "
       '{"predictions": [{"id": 1, "act": 0}, ...]}')


def _again(left):
    return f"{left} requests left. Reply with only a request or the JSON answer described above."


# subject replies (requests here hold no nested JSON objects unless they are about dict entries)
REQ_A = {"show": ["I like coffee and java", "a cafe on the corner"], "why": "coffee words and where they are drunk"}
REQ_B = {"show": ["tea or coffee void"]}
REQ_C = {"show": ["espresso in a mug", "plain words only"]}
REQ_D = {"show": ["java the island"]}
REQ_7 = {"show": ["coffee one", "a cafe two", "java three", "mug four", "tea five", "espresso six", "coffee seven"]}
REQ_DROP_S = {"show": ["", 7, None, ["java"], "coffee time"]}
REQ_DICT = {"show": [{"text": "I like coffee and java"}, "a cafe on the corner",
                     {"text": "espresso in a mug", "why": "strength"}]}
REQ_DROP_D = {"show": [{"text": ""}, {"text": 5}, {"note": "java"}, {"text": "tea or coffee"}, "mug"]}
ANSWER = {"class": "one_plus_related", "explanation": "Coffee words; weaker on cafes and on Java.",
          "evidence": "Coffee at 10, cafe at 7, java at 4.", "missing_data": "none", "confidence": 0.7}
EARLY = {"class": "cannot_tell", "explanation": "Too little data yet.", "evidence": "-",
         "missing_data": "more texts", "confidence": 0.2}
PREDS = {"predictions": [{"id": 1, "fires": True}, {"id": 2, "fires": False}]}
SIM_REPLY = {"predictions": [{"id": 1, "act": 6}, {"id": 2, "act": 0}, {"id": 3, "act": 2}]}
M_PROSE = "I would like to think about this a little more first."
M_SHOW_STR = '{"show": "coffee"}'
M_BROKEN = '{"show": ["coffee", "tea"'
M_OTHER = '{"note": "still thinking"}'
M_ARRAY = '["coffee", "tea"]'


def _j(obj):
    return json.dumps(obj, ensure_ascii=False)


def _fenced(obj):
    return "Here is my answer.\n```json\n" + _j(obj) + "\n```"


# ---------------------------------------------------------------------------
# reference computations (from the interface and text_units.window)
# ---------------------------------------------------------------------------

def _text(s):
    return {"type": "text", "text": s}


def _numbered(xs):
    return "\n\n".join(f"Text {n}:\n{x}" for n, x in enumerate(xs, 1))


def _doc(text):
    """server-style doc: toks a list as long as offs; one token per word with its leading whitespace"""
    offs = [m.start() for m in re.finditer(r"\s*\S+", text)] or [0]
    return list(range(len(offs))), text, offs


def _acts(text):
    toks, _, offs = _doc(text)
    out = []
    for j in range(len(toks)):
        piece = text[offs[j]:offs[j + 1] if j + 1 < len(toks) else len(text)]
        out.append(FRAC.get(piece.strip().strip(".,;:!?").lower(), 0.0) * SCALE)
    return np.asarray(out, dtype=float)


def _marks(acts, card_max):
    marks = []
    for a in acts:
        a = float(a)
        v = None if math.isnan(a) else round(a / card_max * 10)
        marks.append(str(min(10, v)) if v is not None and v >= 1 else None)
    return marks


def _marked_ref(doc, acts, card_max):
    return text_units.window(doc, 0, 0, len(doc[0]), _marks(acts, card_max))


def _shown(text, cid=CID):
    return _marked_ref(_doc(text), _acts(text), CMAX[cid])


def _results_msg(texts, left, cid=CID):
    return _numbered([_shown(t, cid) for t in texts]) + f"\n\n({left} requests left.)"


def _sim_turn(explanation, items):
    return (SIM + "\n\nDescription: " + explanation + "\n\n"
            + "\n\n".join(f"Item {it['id']}:\n{it['text']}" for it in items))


def _parts(cid=CID):
    """(card_blocks, TASK text, turn 2) from turns_for on the (temporary) bank"""
    t1, t2 = run_subjects.turns_for(cid)
    return t1[:-1], t1[-1]["text"], t2


def _p1_turn1(cid=CID):
    card, task, _ = _parts(cid)
    return card + [_text("\n\n" + OFFER + task)]


def _pool_turn1(texts, cid=CID):
    card, task, _ = _parts(cid)
    return card + [_text(POOL_INTRO + "\n\n" + _numbered(texts) + task)]


def _text_of(content):
    """text of one turn the module sent: a string or a list of text blocks"""
    if isinstance(content, str):
        return content
    assert isinstance(content, list) and content, f"turn is not a string or a list of blocks: {content!r}"
    assert all(isinstance(b, dict) and b.get("type") == "text" for b in content), content
    return "".join(b["text"] for b in content)


# ---------------------------------------------------------------------------
# fakes and the temporary layout
# ---------------------------------------------------------------------------

def _reply(text, is_error=False):
    return {"text": text, "cost_usd": None if is_error else 0.0125,
            "usage": None if is_error else {"input_tokens": 100, "output_tokens": 20},
            "is_error": is_error, "ms": 7}


ERROR = _reply("API Error: 529 overloaded", is_error=True)


def _rec(arm, cid, texts, **extra):
    rec = {"arm": arm, "card": cid, "sample": 1, "model": claude_call.MODEL, "requests": [], "results": [],
           "replies": [_reply(t) for t in texts]}
    rec.update(extra)
    return rec


class FakeClaude:
    """claude_call.converse stand-in: one scripted reply list per conversation, in the order conversations start."""

    def __init__(self):
        self.scripts, self.calls, self.lock = [], [], threading.Lock()

    def script(self, *replies):
        self.scripts.append([dict(r) if isinstance(r, dict) else _reply(r) for r in replies])

    def converse(self, system_prompt, turns, model=claude_call.MODEL, timeout=900):
        with self.lock:
            script = self.scripts.pop(0) if self.scripts else []
            call = {"system": system_prompt, "model": model, "sent": [], "replies": []}
            self.calls.append(call)
        out = []
        for content in turns:
            if callable(content):  # as claude_call.converse: built from the replies so far; None ends it
                content = content(out)
                if content is None:
                    break
            call["sent"].append(copy.deepcopy(content))
            if not script:
                raise RuntimeError("fake converse: no scripted reply left")
            rep = script.pop(0)
            out.append(rep)
            call["replies"].append(copy.deepcopy(rep))
        return out


class FakeMeasure:
    def __init__(self):
        self.calls, self.lock = [], threading.Lock()

    def __call__(self, key_entry, texts, server):
        texts = list(texts)
        with self.lock:
            self.calls.append({"key_entry": key_entry, "texts": texts, "server": server})
        return [(_doc(t), _acts(t)) for t in texts]

    @property
    def texts(self):
        return [t for c in self.calls for t in c["texts"]]


class _NoSubprocess:
    PIPE = -1

    def Popen(self, *a, **k):  # noqa: N802
        raise AssertionError("a real claude process was started")


def _no_network(*a, **k):
    raise AssertionError("a network connection was attempted")


def _moved(v, root):
    """v pointed into the temporary copy if it is a path (or list of paths) under the repository, else None"""
    if isinstance(v, PurePath):
        p = Path(v)
        if p.is_absolute() and (p == ROOT or ROOT in p.parents):
            return root / p.relative_to(ROOT)
    elif isinstance(v, str):
        if v == str(ROOT) or v.startswith(str(ROOT) + os.sep):
            return str(root / Path(v).relative_to(ROOT))
    elif isinstance(v, (list, tuple)) and v and all(isinstance(x, (PurePath, str)) for x in v):
        new = [_moved(x, root) for x in v]
        if any(n is not None for n in new):
            return type(v)(x if n is None else n for n, x in zip(new, v))
    return None


def _redirect(monkeypatch, mod, root):
    for name, v in list(vars(mod).items()):
        if name.startswith("__"):
            continue
        new = _moved(v, root)
        if new is not None:
            monkeypatch.setattr(mod, name, new)


def _leaks():
    out = []
    for arm in ARMS:
        for cid in CARDS:
            out += [ROOT / "runs" / arm / cid, ROOT / "runs" / "S0_sim" / arm / cid]
    return [p for p in out if p.exists()]


class Env:
    def __init__(self, tmp_path, monkeypatch, layout="split"):
        self.mp = monkeypatch
        self.root = tmp_path / "janus"
        self.bank = self.root / "bank"
        self.runs = self.root / "runs"
        for d in (self.bank / "cards", self.bank / "heldout", self.root / "arms", self.root / "tools", self.runs):
            d.mkdir(parents=True)
        for cid in CARDS:
            (self.bank / "cards" / f"{cid}.md").write_text(CARD_TEXT[cid], encoding="utf-8")
            (self.bank / "heldout" / f"{cid}.json").write_text(json.dumps(HELDOUT[cid], ensure_ascii=False, indent=1),
                                                               encoding="utf-8")
        (self.root / "arms" / "P1_active.txt").write_text(OFFER_FILE, encoding="utf-8")
        self.keys(layout)

        importlib.reload(run_text_arms)  # fresh module state for every test
        for mod in (run_subjects, run_text_arms):
            _redirect(monkeypatch, mod, self.root)
        monkeypatch.chdir(self.root)
        if isinstance(getattr(run_text_arms, "OFFER", None), str):
            monkeypatch.setattr(run_text_arms, "OFFER", OFFER)

        self.claude = FakeClaude()
        monkeypatch.setattr(claude_call, "converse", self.claude.converse)
        for name, v in list(vars(run_text_arms).items()):
            if v is REAL_CONVERSE:
                monkeypatch.setattr(run_text_arms, name, self.claude.converse)
        monkeypatch.setattr(claude_call, "subprocess", _NoSubprocess())
        monkeypatch.setattr(socket.socket, "connect", _no_network)
        monkeypatch.setattr(socket.socket, "connect_ex", _no_network)
        self.measure = FakeMeasure()
        monkeypatch.setattr(run_text_arms, "measure", self.measure)

    def keys(self, layout):
        for name in ("KEY_r33.json", "KEY_olmo.json"):
            if (self.bank / name).exists():
                (self.bank / name).unlink()
        for name, cids in LAYOUTS[layout].items():
            (self.bank / name).write_text(json.dumps({c: KEYS[c] for c in cids}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")

    def run(self, arms, cards=(CID,), extra=()):
        argv = ["run_text_arms.py", "--arms", ",".join(arms), "--cards", ",".join(cards),
                "--samples", "1", "--workers", "1", *extra]
        self.mp.setattr(sys, "argv", argv)
        try:
            run_text_arms.main()
        except SystemExit as e:
            assert e.code in (None, 0), f"main exited with {e.code!r}"

    def path(self, arm, cid=CID, k=1, sim=False):
        return (self.runs / "S0_sim" if sim else self.runs) / arm / cid / f"s{k}.json"

    def put(self, rec, arm, cid=CID, k=1, sim=False):
        p = self.path(arm, cid, k, sim)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")

    def saved(self):
        return sorted(str(p.relative_to(self.runs)) for p in self.runs.rglob("*.json"))

    def record(self, arm, cid=CID, k=1, sim=False, out=""):
        p = self.path(arm, cid, k, sim)
        assert p.exists(), f"no record {p.relative_to(self.runs)}; saved: {self.saved()}; output:\n{out}"
        return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    e = Env(tmp_path, monkeypatch)
    yield e
    leaked = _leaks()
    assert not leaked, f"written under the real runs/: {leaked}"


def _only_call(env, out):
    calls = env.claude.calls
    assert len(calls) == 1, f"{len(calls)} conversations, want 1; output:\n{out}"
    assert calls[0]["system"] == "", "the system prompt must be empty"
    return calls[0]


def _check_record(rec, arm, cid, call):
    assert rec["arm"] == arm
    assert rec["card"] == cid
    assert rec["sample"] == 1
    assert rec["model"] == claude_call.MODEL
    assert rec["replies"] == call["replies"]


def _check_measure(env, cid, server=DEFAULT_SERVER):
    assert env.measure.calls
    for c in env.measure.calls:
        assert c["server"] == server
        assert c["key_entry"]["unit"] == KEYS[cid]["unit"]
        assert c["key_entry"]["card_max"] == KEYS[cid]["card_max"]


# ---------------------------------------------------------------------------
# fixture guards
# ---------------------------------------------------------------------------

def test_fixture_reference_marks_and_messages():
    assert _shown("I like coffee and java") == "I like« coffee|10» and« java|4»"
    assert _shown("I like coffee and java", CID2) == "I like« coffee|10» and« java|9»"
    assert _shown("tea or coffee void") == "tea or« coffee|10» void"
    assert _shown("tea or coffee void", CID2) == "«tea|1» or« coffee|10» void"
    assert _results_msg(REQ_A["show"], 2) == ("Text 1:\nI like« coffee|10» and« java|4»\n\n"
                                              "Text 2:\na« cafe|7» on the corner\n\n(2 requests left.)")
    assert _numbered(["x", "y"]) == "Text 1:\nx\n\nText 2:\ny"


def test_fixture_replies_parse_as_intended():
    for req in (REQ_A, REQ_B, REQ_C, REQ_D, REQ_7, REQ_DROP_S):
        assert claude_call.parse_json(_j(req)) == req
    assert claude_call.parse_json("Let me look at more.\n" + _j(REQ_B)) == REQ_B
    for obj in (ANSWER, EARLY, PREDS):
        assert claude_call.parse_json(_j(obj)) == obj
    assert claude_call.parse_json(_fenced(ANSWER)) == ANSWER
    for bad in (M_PROSE, M_BROKEN, M_ARRAY):
        with pytest.raises(ValueError):
            claude_call.parse_json(bad)
    for bad in (M_SHOW_STR, M_OTHER, _j(REQ_D)):
        got = claude_call.parse_json(bad)
        assert "class" not in got


def test_fixture_turns_for_reads_the_temporary_bank(env):
    assert run_subjects.BANK == env.bank
    t1, t2 = run_subjects.turns_for(CID)
    assert t1 == [_text(CARD_TEXT[CID]), _text(run_subjects.TASK)]
    assert len(t2) == 1 and t2[0]["text"].startswith(run_subjects.TURN2_TEXT)


# ---------------------------------------------------------------------------
# marked
# ---------------------------------------------------------------------------

MARK_TEXT = "alpha beta gamma delta epsilon zeta eta theta iota"
MARK_ACTS = [0.0, 0.06, 0.16, float("nan"), 1.34, 2.0, 2.52, -0.8, 0.92]


def test_marked_literal():
    # card_max 2.0: x10/2 -> 0, 0.3, 0.8, NaN, 6.7, 10, 12.6, -4, 4.6
    got = run_text_arms.marked(_doc(MARK_TEXT), np.asarray(MARK_ACTS), 2.0)
    assert got == "alpha beta« gamma|1» delta« epsilon|7»« zeta|10»« eta|10» theta« iota|5»"


@pytest.mark.parametrize("card_max", (2.0, 4.0, 0.5, 13.0))
def test_marked_is_window_with_the_interface_marks(card_max):
    doc, acts = _doc(MARK_TEXT), np.asarray(MARK_ACTS)
    got = run_text_arms.marked(doc, acts, card_max)
    assert got == text_units.window(doc, 0, 0, len(doc[0]), _marks(acts, card_max))


@pytest.mark.parametrize("acts", ([0.0] * 9, [float("nan")] * 9, [0.04] * 9, [-3.0] * 9),
                         ids=("zero", "nan", "below_1", "negative"))
def test_marked_unmarked_doc_is_its_text(acts):
    assert run_text_arms.marked(_doc(MARK_TEXT), np.asarray(acts), 2.0) == MARK_TEXT


def test_marked_gpt2_doc():
    doc = text_units.tok("See www.example.com and Fig. 3 here. I like coffee at the cafe.\n- milk\n")
    acts = np.random.default_rng(7).uniform(-0.5, 5.0, len(doc[0]))
    acts[::5] = np.nan
    assert run_text_arms.marked(doc, acts, 4.2) == _marked_ref(doc, acts, 4.2)


# ---------------------------------------------------------------------------
# A0
# ---------------------------------------------------------------------------

def test_a0_turns_and_record(env, capsys):
    env.claude.script(_j(ANSWER), _j(PREDS))
    env.run(["A0"])
    out = capsys.readouterr().out
    call = _only_call(env, out)
    t1, t2 = run_subjects.turns_for(CID)
    assert call["sent"] == [t1, t2]
    _check_record(env.record("A0", out=out), "A0", CID, call)
    assert "done" in out.lower()


# ---------------------------------------------------------------------------
# P1_active: the reply loop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cid", CARDS)
def test_p1_string_requests_then_answer_then_turn2(env, capsys, cid):
    env.claude.script(_j(REQ_A), "Let me look at more.\n" + _j(REQ_B), _fenced(ANSWER), _j(PREDS))
    env.run(["P1_active"], [cid])
    out = capsys.readouterr().out
    call = _only_call(env, out)
    sent = call["sent"]
    _, _, t2 = _parts(cid)
    assert sent[0] == _p1_turn1(cid)
    assert [_text_of(s) for s in sent[1:-1]] == [_results_msg(REQ_A["show"], 2, cid),
                                                 _results_msg(REQ_B["show"], 1, cid)]
    assert sent[-1] == t2
    assert len(sent) == 4, "the conversation ends after turn 2's reply"
    assert env.measure.texts == REQ_A["show"] + REQ_B["show"]
    _check_measure(env, cid)
    rec = env.record("P1_active", cid, out=out)
    _check_record(rec, "P1_active", cid, call)
    assert rec["requests"] == [REQ_A, REQ_B]
    assert rec["results"] == [_shown(t, cid) for t in REQ_A["show"] + REQ_B["show"]]
    assert "done" in out.lower()


def test_p1_request_entries_as_dicts(env, capsys):
    """A dict entry counts by its text. Note: claude_call.parse_json, as it stands, returns the last JSON object of a
    reply when none has class/predictions/edits, which for this request is its last entry, not the request."""
    env.claude.script(_j(REQ_DICT), _j(ANSWER), _j(PREDS))
    env.run(["P1_active"])
    out = capsys.readouterr().out
    call = _only_call(env, out)
    sent = call["sent"]
    texts = ["I like coffee and java", "a cafe on the corner", "espresso in a mug"]
    assert [_text_of(s) for s in sent[1:-1]] == [_results_msg(texts, 2)]
    assert sent[-1] == _parts()[2] and len(sent) == 3
    assert env.measure.texts == texts
    rec = env.record("P1_active", out=out)
    assert rec["requests"] == [REQ_DICT]
    assert rec["results"] == [_shown(t) for t in texts]


def test_p1_request_drops_dict_entries_without_text(env, capsys):
    """Same parse_json note as test_p1_request_entries_as_dicts."""
    env.claude.script(_j(REQ_DROP_D), _j(ANSWER), _j(PREDS))
    env.run(["P1_active"])
    out = capsys.readouterr().out
    sent = _only_call(env, out)["sent"]
    assert [_text_of(s) for s in sent[1:-1]] == ["Text 1:\ntea or« coffee|10»\n\nText 2:\n«mug|1»\n\n(2 requests left.)"]
    assert env.measure.texts == ["tea or coffee", "mug"]
    rec = env.record("P1_active", out=out)
    assert rec["requests"] == [REQ_DROP_D]
    assert rec["results"] == ["tea or« coffee|10»", "«mug|1»"]


def test_p1_request_drops_entries_that_are_not_text(env, capsys):
    env.claude.script(_j(REQ_DROP_S), _j(ANSWER), _j(PREDS))
    env.run(["P1_active"])
    out = capsys.readouterr().out
    sent = _only_call(env, out)["sent"]
    assert [_text_of(s) for s in sent[1:-1]] == ["Text 1:\n«coffee|10» time\n\n(2 requests left.)"]
    assert env.measure.texts == ["coffee time"]
    rec = env.record("P1_active", out=out)
    assert rec["requests"] == [REQ_DROP_S], "the whole request dict is logged"
    assert rec["results"] == ["«coffee|10» time"]


def test_p1_request_takes_the_first_five_entries(env, capsys):
    env.claude.script(_j(REQ_7), _j(ANSWER), _j(PREDS))
    env.run(["P1_active"])
    out = capsys.readouterr().out
    sent = _only_call(env, out)["sent"]
    first5 = REQ_7["show"][:5]
    assert [_text_of(s) for s in sent[1:-1]] == [_results_msg(first5, 2)]
    assert sent[-1] == _parts()[2] and len(sent) == 3
    assert env.measure.texts == first5
    rec = env.record("P1_active", out=out)
    assert rec["requests"] == [REQ_7]
    assert rec["results"] == [_shown(t) for t in first5]


def test_p1_three_requests_then_only_the_answer(env, capsys):
    env.claude.script(_j(REQ_A), _j(REQ_B), _j(REQ_C), _j(REQ_D), _j(ANSWER), _j(PREDS))
    env.run(["P1_active"])
    out = capsys.readouterr().out
    sent = _only_call(env, out)["sent"]
    assert sent[0] == _p1_turn1()
    assert [_text_of(s) for s in sent[1:-1]] == [_results_msg(REQ_A["show"], 2), _results_msg(REQ_B["show"], 1),
                                                 _results_msg(REQ_C["show"], 0), ONLY_ANSWER]
    assert sent[-1] == _parts()[2] and len(sent) == 6
    assert env.measure.texts == REQ_A["show"] + REQ_B["show"] + REQ_C["show"]
    rec = env.record("P1_active", out=out)
    assert rec["requests"] == [REQ_A, REQ_B, REQ_C]
    assert rec["results"] == [_shown(t) for t in REQ_A["show"] + REQ_B["show"] + REQ_C["show"]]


def test_p1_premature_answer_is_nudged_once(env, capsys):
    env.claude.script(_j(EARLY), _j(REQ_A), _j(ANSWER), _j(PREDS))
    env.run(["P1_active"])
    out = capsys.readouterr().out
    sent = _only_call(env, out)["sent"]
    assert [_text_of(s) for s in sent[1:-1]] == [PREMATURE, _results_msg(REQ_A["show"], 2)]
    assert sent[-1] == _parts()[2] and len(sent) == 4
    rec = env.record("P1_active", out=out)
    assert rec["requests"] == [REQ_A]
    assert run_text_arms.answer_of(rec) == ANSWER


def test_p1_premature_answer_accepted_the_second_time(env, capsys):
    env.claude.script(_j(EARLY), _j(ANSWER), _j(PREDS))
    env.run(["P1_active"])
    out = capsys.readouterr().out
    call = _only_call(env, out)
    sent = call["sent"]
    assert sent[0] == _p1_turn1()
    assert [_text_of(s) for s in sent[1:-1]] == [PREMATURE]
    assert sent[-1] == _parts()[2] and len(sent) == 3
    assert env.measure.calls == []
    rec = env.record("P1_active", out=out)
    _check_record(rec, "P1_active", CID, call)
    assert rec["requests"] == [] and rec["results"] == []
    assert run_text_arms.answer_of(rec) == ANSWER


def test_p1_malformed_replies_up_to_three_in_a_row(env, capsys):
    env.claude.script(M_PROSE, M_SHOW_STR, M_BROKEN, _j(REQ_A), M_OTHER, M_ARRAY, M_PROSE, _j(ANSWER), _j(PREDS))
    env.run(["P1_active"])
    out = capsys.readouterr().out
    sent = _only_call(env, out)["sent"]
    assert sent[0] == _p1_turn1()
    assert [_text_of(s) for s in sent[1:-1]] == [_again(3)] * 3 + [_results_msg(REQ_A["show"], 2)] + [_again(2)] * 3
    assert sent[-1] == _parts()[2] and len(sent) == 9
    rec = env.record("P1_active", out=out)
    assert rec["requests"] == [REQ_A]
    assert rec["results"] == [_shown(t) for t in REQ_A["show"]]


@pytest.mark.parametrize("n_req", (0, 1, 3))
def test_p1_fourth_malformed_reply_in_a_row_fails(env, capsys, n_req):
    reqs = [REQ_A, REQ_B, REQ_C][:n_req]
    bad = [M_PROSE, M_OTHER, M_ARRAY, M_BROKEN] if n_req < 3 else [_j(REQ_D), M_PROSE, M_OTHER, M_ARRAY]
    env.claude.script(*[_j(r) for r in reqs], *bad, _j(ANSWER), _j(PREDS))  # the answer must never be reached
    env.run(["P1_active"])
    out = capsys.readouterr().out
    sent = _only_call(env, out)["sent"]
    nudge = _again(3 - n_req) if n_req < 3 else ONLY_ANSWER
    want = [_results_msg(r["show"], 2 - i) for i, r in enumerate(reqs)] + [nudge] * 3
    assert sent[0] == _p1_turn1()
    assert [_text_of(s) for s in sent[1:]] == want
    assert not env.path("P1_active").exists(), "a failed session is not saved"
    assert "FAIL" in out


def test_p1_measures_with_the_server_option(env, capsys):
    env.claude.script(_j(REQ_B), _j(ANSWER), _j(PREDS))
    env.run(["P1_active"], [CID2], extra=["--server", "http://10.9.8.7:4321"])
    out = capsys.readouterr().out
    _only_call(env, out)
    _check_measure(env, CID2, server="http://10.9.8.7:4321")
    assert env.record("P1_active", CID2, out=out)["results"] == [_shown(REQ_B["show"][0], CID2)]


# ---------------------------------------------------------------------------
# F2_pool and Y1_yoked turn 1
# ---------------------------------------------------------------------------

def test_f2_pool_turn1_exact(env, capsys):
    env.claude.script(_j(ANSWER), _j(PREDS))
    env.run(["F2_pool"])
    out = capsys.readouterr().out
    call = _only_call(env, out)
    card, task, t2 = _parts()
    last = ("\n\nBelow is the same unit on further texts, marked the same way and on the same scale as above."
            "\n\nText 1:\nfirst «coffee|10» window\n\nText 2:\nsecond« cafe|7» window"
            "\n\nText 3:\nthird window, nothing marked" + task)
    assert call["sent"] == [card + [_text(last)], t2]
    rec = env.record("F2_pool", out=out)
    _check_record(rec, "F2_pool", CID, call)
    assert rec["results"] == KEYS[CID]["pool"]
    assert "done" in out.lower()


@pytest.mark.parametrize("layout", tuple(LAYOUTS))
def test_f2_pool_keys_from_either_file_merged(env, capsys, layout):
    env.keys(layout)
    for _ in CARDS:
        env.claude.script(_j(ANSWER), _j(PREDS))
    env.run(["F2_pool"], CARDS)
    out = capsys.readouterr().out
    assert len(env.claude.calls) == 2, out
    by_card = {next(c for c in CARDS if call["sent"][0][0] == _text(CARD_TEXT[c])): call
               for call in env.claude.calls}
    for cid in CARDS:
        assert by_card[cid]["sent"] == [_pool_turn1(KEYS[cid]["pool"], cid), _parts(cid)[2]]
        rec = env.record("F2_pool", cid, out=out)
        assert rec["card"] == cid and rec["results"] == KEYS[cid]["pool"]


def test_y1_turn1_from_the_p1_record(env, capsys):
    p1_results = ["Y one« coffee|10» here", "Y two plain", "«java|4» Y three"]
    env.put(_rec("P1_active", CID, [_j(REQ_A), _j(REQ_B), _j(ANSWER), _j(PREDS)], requests=[REQ_A, REQ_B],
                 results=p1_results), "P1_active")
    env.claude.script(_j(ANSWER), _j(PREDS))
    env.run(["Y1_yoked"])
    out = capsys.readouterr().out
    call = _only_call(env, out)
    card, task, t2 = _parts()
    last = (POOL_INTRO + "\n\nText 1:\nY one« coffee|10» here\n\nText 2:\nY two plain\n\nText 3:\n«java|4» Y three"
            + task)
    assert call["sent"] == [card + [_text(last)], t2]
    rec = env.record("Y1_yoked", out=out)
    _check_record(rec, "Y1_yoked", CID, call)
    assert rec["results"] == p1_results


@pytest.mark.parametrize("p1", ("absent", "empty_results", "no_results_key"))
def test_y1_skipped_without_p1_results(env, capsys, p1):
    if p1 != "absent":
        rec = _rec("P1_active", CID, [_j(EARLY), _j(ANSWER), _j(PREDS)])
        if p1 == "no_results_key":
            del rec["results"]
        env.put(rec, "P1_active")
    before = env.saved()
    env.claude.script(_j(ANSWER), _j(PREDS))
    env.run(["Y1_yoked"])
    out = capsys.readouterr().out
    assert env.claude.calls == []
    assert not env.path("Y1_yoked").exists()
    assert env.saved() == before
    assert out.strip(), "skipped with a message"


def test_p1_then_y1_in_one_run(env, capsys):
    env.claude.script(_j(REQ_A), _j(REQ_B), _j(ANSWER), _j(PREDS))
    env.claude.script(_j(ANSWER), _j(PREDS))
    env.run(["P1_active", "Y1_yoked"])
    out = capsys.readouterr().out
    shown = [_shown(t) for t in REQ_A["show"] + REQ_B["show"]]
    assert env.record("P1_active", out=out)["results"] == shown
    assert len(env.claude.calls) == 2, out
    y1 = env.claude.calls[1]
    assert y1["sent"] == [_pool_turn1(shown), _parts()[2]]
    rec = env.record("Y1_yoked", out=out)
    _check_record(rec, "Y1_yoked", CID, y1)
    assert rec["results"] == shown


# ---------------------------------------------------------------------------
# saving: existing records, error replies
# ---------------------------------------------------------------------------

def _p1_for_y1(env):
    env.put(_rec("P1_active", CID, [_j(REQ_A), _j(ANSWER), _j(PREDS)], requests=[REQ_A],
                 results=["r one« coffee|10»", "r two"]), "P1_active")


@pytest.mark.parametrize("arm", ARMS)
def test_existing_record_is_skipped(env, capsys, arm):
    if arm == "Y1_yoked":
        _p1_for_y1(env)
    env.put({"sentinel": arm}, arm)
    before = env.path(arm).read_text(encoding="utf-8")
    env.claude.script(_j(REQ_A), _j(ANSWER), _j(PREDS))
    env.run([arm])
    out = capsys.readouterr().out
    assert env.claude.calls == []
    assert env.path(arm).read_text(encoding="utf-8") == before
    assert "skip" in out.lower()


@pytest.mark.parametrize("arm", ARMS)
def test_session_with_an_error_reply_is_not_saved(env, capsys, arm):
    if arm == "Y1_yoked":
        _p1_for_y1(env)
    if arm == "P1_active":
        env.claude.script(_j(REQ_A), _j(ANSWER), ERROR)
    else:
        env.claude.script(_j(ANSWER), ERROR)
    env.run([arm])
    out = capsys.readouterr().out
    assert len(env.claude.calls) == 1, out
    assert not env.path(arm).exists()
    assert "FAIL" in out


# ---------------------------------------------------------------------------
# simulator score: sim_turn, answer_of, --sim
# ---------------------------------------------------------------------------

def test_sim_turn_exact():
    items = [{"id": 1, "text": "a ⟦coffee⟧ cup"}, {"id": 2, "text": "the ⟦tea⟧ pot"}]
    want = ("Below is a description of what one unit (a single neuron) in a language model responds to, then 40 text "
            "excerpts, each with one token marked ⟦like this⟧. Using only the description, predict the unit's "
            "activation on each marked token, from 0 (not active) to 10 (as active as it ever gets). Reply with only "
            'JSON: {"predictions": [{"id": 1, "act": 0}, ...]}'
            "\n\nDescription: Coffee words.\n\nItem 1:\na ⟦coffee⟧ cup\n\nItem 2:\nthe ⟦tea⟧ pot")
    assert run_text_arms.sim_turn("Coffee words.", items) == want


def test_sim_turn_uses_the_given_ids_in_order():
    items = [{"id": 7, "text": "x ⟦y⟧"}, {"id": 3, "text": "⟦z⟧ w"}, {"id": 40, "text": "q"}]
    assert run_text_arms.sim_turn("E", items) == _sim_turn("E", items)
    assert run_text_arms.sim_turn("E", items).endswith("\n\nDescription: E\n\nItem 7:\nx ⟦y⟧\n\nItem 3:\n⟦z⟧ w"
                                                       "\n\nItem 40:\nq")


ANSWER_CASES = {
    "bare": ([_j(ANSWER), _j(PREDS)], ANSWER),
    "last_of_several": ([_j(EARLY), _j(REQ_A), _fenced(ANSWER), _j(PREDS)], ANSWER),
    "held_out_reply_ignored": ([_j(ANSWER), _j({"predictions": [{"id": 1, "fires": True}], "class": "one"})], ANSWER),
    "unparseable_skipped": ([M_PROSE, _j(ANSWER), M_BROKEN, _j(PREDS)], ANSWER),
    "none": ([_j(REQ_A), M_PROSE, _j(PREDS)], None),
    "only_held_out_has_class": ([M_PROSE, _j({"predictions": [], "class": "one"})], None),
}


@pytest.mark.parametrize("case", tuple(ANSWER_CASES))
def test_answer_of(case):
    texts, want = ANSWER_CASES[case]
    rec = _rec("P1_active", CID, texts)
    got = run_text_arms.answer_of(rec)
    if want is None:
        assert got is None
    else:
        assert got == want


@pytest.mark.parametrize("arm,cid", (("A0", CID), ("P1_active", CID2)))
def test_sim_scores_a_saved_record(env, capsys, arm, cid):
    texts = [_j(ANSWER), _j(PREDS)] if arm == "A0" else [_j(EARLY), _j(REQ_A), _j(ANSWER), _j(PREDS)]
    env.put(_rec(arm, cid, texts), arm, cid)
    before = env.path(arm, cid).read_text(encoding="utf-8")
    env.claude.script(_j(SIM_REPLY))
    env.run([arm], [cid], extra=["--sim"])
    out = capsys.readouterr().out
    call = _only_call(env, out)
    assert call["sent"] == [[_text(_sim_turn(ANSWER["explanation"], KEYS[cid]["sim"]))]]
    sim = env.record(arm, cid, sim=True, out=out)
    assert sim["arm"] == arm and sim["card"] == cid and sim["sample"] == 1
    assert sim["explanation"] == ANSWER["explanation"]
    assert sim["replies"] == call["replies"]
    assert env.path(arm, cid).read_text(encoding="utf-8") == before
    assert env.saved() == sorted([f"{arm}/{cid}/s1.json", f"S0_sim/{arm}/{cid}/s1.json"])


@pytest.mark.parametrize("why", ("present", "no_answer"))
def test_sim_skips(env, capsys, why):
    env.put(_rec("A0", CID, [_j(ANSWER), _j(PREDS)] if why == "present" else [M_PROSE, _j(PREDS)]), "A0")
    if why == "present":
        env.put({"sentinel": "sim"}, "A0", sim=True)
    before = env.saved()
    sim_before = env.path("A0", sim=True).read_text(encoding="utf-8") if why == "present" else None
    env.claude.script(_j(SIM_REPLY))
    env.run(["A0"], [CID], extra=["--sim"])
    capsys.readouterr()
    assert env.claude.calls == []
    assert env.saved() == before
    if why == "present":
        assert env.path("A0", sim=True).read_text(encoding="utf-8") == sim_before
