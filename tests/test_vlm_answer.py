"""Black-box tests for tools/vlm_answer.py.

Written from the Interface section of authors/spec_vlmanswer.md only. Expected
prompts come from the definition the spec gives (proploop.fill of
refineloop.A_PROMPT with b=1 and proploop.questions_block), plus one prompt
written out by hand; expected answer rows are worked out by hand from
proploop.parse_answers' rules. Nothing is computed with the module's own
helpers. No vllm, GPU or network: generate is a fake that answers each photo
from a script, recognising the photo by its data URI, and the photos are small
JPEG files written to tmp_path.
"""

import base64
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import proploop  # noqa: E402
import refineloop  # noqa: E402
import vlm_answer  # noqa: E402


MODEL = "some-org/Some-VL-7B-Instruct"
URI_PREFIX = "data:image/jpeg;base64,"

Q_HAND = [
    {"question": "Is there a dog in the photo?", "answers": ["yes", "no"]},
    {"question": "What colour is the sky?", "answers": ["blue", "grey", "no sky"]},
]
PROMPT_HAND = (
    'Answer every question below for each of the 1 images above. For each question choose exactly one of its '
    'answers, by letter, or "?" if you cannot tell. Unless a question says otherwise, left and right mean the left and '
    'right of the photo as you look at it.\n\n'
    'Q1. Is there a dog in the photo? A) yes B) no\n'
    'Q2. What colour is the sky? A) blue B) grey C) no sky\n\n'
    'Reply with only JSON: {"images": [{"id": n, "answers": ["A", ...]}]}, one entry per image, each with one answer '
    'per question, in question order.')

Q_THREE = Q_HAND + [{"question": "Is it daytime?", "answers": ["yes", "no", "cannot tell"]}]

Q_FIVE = [
    {"question": "How many people are visible?", "answers": ["none", "one", "two", "three to five", "more"]},
    {"question": "Is the photo taken indoors?", "answers": ["indoors", "outdoors"]},
    {"question": "What is the main surface?", "answers": ["grass", "water", "road", "other"]},
]

# Replies for one image with Q_HAND (2 questions: 2 and 3 answers), worked out by hand.
OK_BC = '{"images": [{"id": 1, "answers": ["B", "C"]}]}'                        # -> [1, 2]
OK_A_UNK = 'Here you go:\n{"images": [{"id": 1, "answers": ["A", "?"]}]}'      # -> [0, None]
OK_ALL_UNK = '{"images": [{"id": 1, "answers": ["?", "?"]}]}'                  # -> [None, None]
OK_FENCED = '```json\n{"images": [{"id": 1, "answers": [" b ", "a"]}]}\n```'   # -> [1, 0]
BAD_PROSE = "Sorry, I cannot answer questions about this image."
BAD_LETTER = '{"images": [{"id": 1, "answers": ["C", "A"]}]}'      # C is not an answer of Q1
BAD_ID = '{"images": [{"id": 2, "answers": ["A", "A"]}]}'          # only image 1 exists
BAD_SHORT = '{"images": [{"id": 1, "answers": ["A"]}]}'            # one answer for two questions
BAD_TWO = ('{"images": [{"id": 1, "answers": ["A", "B"]}, '
           '{"id": 2, "answers": ["B", "A"]}]}')                     # two entries for one image
BAD_EMPTY = ""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_jpeg(path, seed, trailing=b""):
    """a small noise JPEG (24 x 32), optionally with bytes appended after it; returns the file's bytes"""
    arr = np.random.default_rng(seed).integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(path), format="JPEG", quality=90)
    if trailing:
        with open(path, "ab") as f:
            f.write(trailing)
    return Path(path).read_bytes()


def _data_uri(data):
    return URI_PREFIX + base64.b64encode(data).decode("ascii")


def _prompt(questions):
    """the prompt as the spec defines it"""
    return proploop.fill(refineloop.A_PROMPT, b=1, questions=proploop.questions_block(questions))


def _expected_messages(path, questions):
    return [{"role": "user", "content": [
        {"type": "text", "text": "Image 1:"},
        {"type": "image_url", "image_url": {"url": _data_uri(Path(path).read_bytes())}},
        {"type": "text", "text": _prompt(questions)},
    ]}]


def _uri_of(messages):
    return messages[0]["content"][1]["image_url"]["url"]


def _photos(tmp_path, ids):
    """one distinct JPEG per id; returns the photo records and data URI -> id"""
    photos, uri_to_id = [], {}
    for k, pid in enumerate(ids):
        path = tmp_path / f"{pid}.jpg"
        data = _write_jpeg(path, seed=100 + k)
        photos.append({"id": pid, "path": str(path)})
        uri_to_id[_data_uri(data)] = pid
    assert len(uri_to_id) == len(ids)
    return photos, uri_to_id


class FakeGenerate:
    """generate(list_of_message_lists, temperature) -> one scripted reply per message list, in order.

    The photo of each message list is recognised by its data URI; each photo's replies are handed out
    in script order. Every call is recorded as (message lists, temperature)."""

    def __init__(self, uri_to_id, replies):
        self.uri_to_id = uri_to_id
        self.replies = {pid: list(texts) for pid, texts in replies.items()}
        self.calls = []

    def __call__(self, message_lists, temperature):
        message_lists = list(message_lists)
        self.calls.append((copy.deepcopy(message_lists), temperature))
        out = []
        for messages in message_lists:
            pid = self.uri_to_id[_uri_of(messages)]
            assert self.replies.get(pid), f"photo {pid} asked more often than scripted"
            out.append(self.replies[pid].pop(0))
        return out


def _run(tmp_path, ids, replies, questions=Q_HAND, model=MODEL):
    photos, uri_to_id = _photos(tmp_path, ids)
    gen = FakeGenerate(uri_to_id, replies)
    out = vlm_answer.answer_photos(photos, questions, gen, model)
    return photos, gen, out


def _rec(answers, texts, valid, model=MODEL):
    return {"answers": answers, "texts": texts, "valid": valid, "model": model}


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------

def test_build_messages_exact_structure(tmp_path):
    path = tmp_path / "p.jpg"
    _write_jpeg(path, seed=1)
    msgs = vlm_answer.build_messages(str(path), Q_HAND)
    assert msgs == _expected_messages(path, Q_HAND)
    # one user turn, three blocks in this order, nothing else
    assert isinstance(msgs, list) and len(msgs) == 1
    assert set(msgs[0]) == {"role", "content"}
    assert msgs[0]["role"] == "user"
    assert [b["type"] for b in msgs[0]["content"]] == ["text", "image_url", "text"]
    assert set(msgs[0]["content"][0]) == {"type", "text"}
    assert set(msgs[0]["content"][1]) == {"type", "image_url"}
    assert set(msgs[0]["content"][1]["image_url"]) == {"url"}
    assert set(msgs[0]["content"][2]) == {"type", "text"}


def test_build_messages_text_blocks_exact(tmp_path):
    path = tmp_path / "p.jpg"
    _write_jpeg(path, seed=2)
    content = vlm_answer.build_messages(str(path), Q_HAND)[0]["content"]
    assert content[0]["text"] == "Image 1:"
    assert content[2]["text"] == PROMPT_HAND
    assert content[2]["text"] == _prompt(Q_HAND)


@pytest.mark.parametrize("questions", [Q_THREE, Q_FIVE, Q_HAND[:1]])
def test_build_messages_prompt_follows_questions(tmp_path, questions):
    path = tmp_path / "p.jpg"
    _write_jpeg(path, seed=3)
    msgs = vlm_answer.build_messages(str(path), questions)
    assert msgs[0]["content"][2]["text"] == _prompt(questions)
    assert msgs == _expected_messages(path, questions)


def test_build_messages_prompt_filled_in_one_pass(tmp_path):
    # proploop.fill does not search the inserted values again: "{b}" and "{questions}" inside a
    # question stay as written.
    questions = [{"question": "Does the sign read {b} or {questions}?", "answers": ["{b}", "neither"]},
                 {"question": "Is it raining?", "answers": ["yes", "no"]}]
    path = tmp_path / "p.jpg"
    _write_jpeg(path, seed=4)
    text = vlm_answer.build_messages(str(path), questions)[0]["content"][2]["text"]
    assert text == _prompt(questions)
    assert "Q1. Does the sign read {b} or {questions}? A) {b} B) neither\nQ2. Is it raining? A) yes B) no" in text
    assert text.startswith("Answer every question below for each of the 1 images above.")


def test_build_messages_data_uri_is_file_bytes_unchanged(tmp_path):
    # bytes after the JPEG's end marker, including CR LF, NUL and non-UTF-8 bytes: kept only
    # when the file is read as bytes and not re-encoded
    path = tmp_path / "p.jpg"
    data = _write_jpeg(path, seed=5, trailing=b"\r\n\x00\x1a\xff\xfe tail \r\n")
    url = vlm_answer.build_messages(str(path), Q_HAND)[0]["content"][1]["image_url"]["url"]
    assert url.startswith(URI_PREFIX)
    assert url == URI_PREFIX + base64.b64encode(data).decode("ascii")
    assert "\n" not in url
    assert base64.b64decode(url[len(URI_PREFIX):], validate=True) == data


def test_build_messages_data_uri_not_resized(tmp_path):
    # a larger photo is sent as stored, not shrunk
    path = tmp_path / "big.jpg"
    arr = np.random.default_rng(6).integers(0, 256, size=(700, 900, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(path), format="JPEG", quality=95)
    data = path.read_bytes()
    url = vlm_answer.build_messages(str(path), Q_HAND)[0]["content"][1]["image_url"]["url"]
    assert url == _data_uri(data)


def test_build_messages_jpeg_prefix_whatever_the_file_name(tmp_path):
    path = tmp_path / "photo_without_extension"
    data = _write_jpeg(path, seed=7)
    url = vlm_answer.build_messages(str(path), Q_HAND)[0]["content"][1]["image_url"]["url"]
    assert url == _data_uri(data)


def test_build_messages_different_files_different_uris(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    da, db = _write_jpeg(a, seed=8), _write_jpeg(b, seed=9)
    assert da != db
    ua = _uri_of(vlm_answer.build_messages(str(a), Q_HAND))
    ub = _uri_of(vlm_answer.build_messages(str(b), Q_HAND))
    assert (ua, ub) == (_data_uri(da), _data_uri(db))


# ---------------------------------------------------------------------------
# answer_photos: mixed case
#
# Photos in this order (ids deliberately not sorted), with Q_HAND:
#   p3: first OK_BC                       -> valid on the first pass, [1, 2]
#   p1: first BAD_PROSE, retry OK_A_UNK   -> valid on retry, [0, None]
#   p4: first BAD_LETTER, retry BAD_EMPTY -> never valid
#   p2: first OK_ALL_UNK                  -> valid on the first pass, [None, None]
#   p5: first BAD_ID, retry OK_FENCED     -> valid on retry, [1, 0]
# First pass: all five at 0.0, in photo order. Retry: p1, p4, p5 at 0.7, in photo order.
# ---------------------------------------------------------------------------

MIXED_IDS = ["p3", "p1", "p4", "p2", "p5"]
MIXED_REPLIES = {
    "p3": [OK_BC],
    "p1": [BAD_PROSE, OK_A_UNK],
    "p4": [BAD_LETTER, BAD_EMPTY],
    "p2": [OK_ALL_UNK],
    "p5": [BAD_ID, OK_FENCED],
}


def test_answer_photos_two_passes_temperatures_and_order(tmp_path):
    photos, gen, _ = _run(tmp_path, MIXED_IDS, MIXED_REPLIES)
    assert len(gen.calls) == 2
    (first, t1), (retry, t2) = gen.calls
    assert t1 == pytest.approx(0.0)
    assert t2 == pytest.approx(0.7)
    by_id = {p["id"]: p for p in photos}
    assert first == [_expected_messages(by_id[i]["path"], Q_HAND) for i in MIXED_IDS]
    assert retry == [_expected_messages(by_id[i]["path"], Q_HAND) for i in ["p1", "p4", "p5"]]


def test_answer_photos_records(tmp_path):
    _, _, out = _run(tmp_path, MIXED_IDS, MIXED_REPLIES)
    assert set(out) == set(MIXED_IDS)
    assert out["p3"] == _rec([1, 2], [OK_BC], True)
    assert out["p1"] == _rec([0, None], [BAD_PROSE, OK_A_UNK], True)
    assert out["p4"] == _rec(None, [BAD_LETTER, BAD_EMPTY], False)
    assert out["p2"] == _rec([None, None], [OK_ALL_UNK], True)
    assert out["p5"] == _rec([1, 0], [BAD_ID, OK_FENCED], True)
    for pid in MIXED_IDS:
        assert set(out[pid]) == {"answers", "texts", "valid", "model"}
        assert isinstance(out[pid]["valid"], bool)
        assert out[pid]["model"] == MODEL


def test_answer_photos_all_unknown_answers_still_valid(tmp_path):
    # a reply that parses is valid even when every answer is "?"
    _, gen, out = _run(tmp_path, ["only"], {"only": [OK_ALL_UNK]})
    assert out == {"only": _rec([None, None], [OK_ALL_UNK], True)}
    assert all(len(msgs) == 0 for msgs, _ in gen.calls[1:])


# ---------------------------------------------------------------------------
# answer_photos: single cases
# ---------------------------------------------------------------------------

def test_answer_photos_valid_only_on_retry(tmp_path):
    photos, gen, out = _run(tmp_path, ["x"], {"x": [BAD_SHORT, OK_BC]})
    assert out == {"x": _rec([1, 2], [BAD_SHORT, OK_BC], True)}
    assert out["x"]["valid"] is True
    assert len(gen.calls) == 2
    expected = [_expected_messages(photos[0]["path"], Q_HAND)]
    assert gen.calls[0][0] == expected
    assert gen.calls[0][1] == pytest.approx(0.0)
    assert gen.calls[1][0] == expected
    assert gen.calls[1][1] == pytest.approx(0.7)


def test_answer_photos_never_valid(tmp_path):
    photos, gen, out = _run(tmp_path, ["y"], {"y": [BAD_TWO, BAD_PROSE]})
    assert out == {"y": _rec(None, [BAD_TWO, BAD_PROSE], False)}
    assert out["y"]["answers"] is None
    assert out["y"]["valid"] is False
    # one retry pass only
    assert len(gen.calls) == 2
    assert gen.calls[0][1] == pytest.approx(0.0)
    assert gen.calls[1][1] == pytest.approx(0.7)
    assert [len(m) for m, _ in gen.calls] == [1, 1]


def test_answer_photos_all_valid_on_first_pass(tmp_path):
    ids = ["c", "a", "b"]
    replies = {"c": [OK_BC], "a": [OK_FENCED], "b": [OK_A_UNK]}
    photos, gen, out = _run(tmp_path, ids, replies)
    assert out == {"c": _rec([1, 2], [OK_BC], True),
                   "a": _rec([1, 0], [OK_FENCED], True),
                   "b": _rec([0, None], [OK_A_UNK], True)}
    assert 1 <= len(gen.calls) <= 2
    msgs, t = gen.calls[0]
    assert t == pytest.approx(0.0)
    assert msgs == [_expected_messages(p["path"], Q_HAND) for p in photos]
    # no photo is asked again
    assert all(len(m) == 0 for m, _ in gen.calls[1:])


def test_answer_photos_all_invalid_on_first_pass(tmp_path):
    ids = ["k2", "k1"]
    replies = {"k2": [BAD_EMPTY, BAD_ID], "k1": [BAD_LETTER, OK_ALL_UNK]}
    photos, gen, out = _run(tmp_path, ids, replies)
    assert out == {"k2": _rec(None, [BAD_EMPTY, BAD_ID], False),
                   "k1": _rec([None, None], [BAD_LETTER, OK_ALL_UNK], True)}
    expected = [_expected_messages(p["path"], Q_HAND) for p in photos]
    assert len(gen.calls) == 2
    assert gen.calls[0][0] == expected
    assert gen.calls[0][1] == pytest.approx(0.0)
    assert gen.calls[1][0] == expected
    assert gen.calls[1][1] == pytest.approx(0.7)


def test_answer_photos_parses_with_the_given_questions(tmp_path):
    reply = '{"images": [{"id": 1, "answers": ["A", "B", "C"]}]}'
    # three answers for two questions: does not parse, on either pass
    d2 = tmp_path / "two"
    d2.mkdir()
    _, gen2, out2 = _run(d2, ["z"], {"z": [reply, reply]}, questions=Q_HAND)
    assert out2 == {"z": _rec(None, [reply, reply], False)}
    assert len(gen2.calls) == 2
    # the same reply with three questions: parses on the first pass
    d3 = tmp_path / "three"
    d3.mkdir()
    photos3, gen3, out3 = _run(d3, ["z"], {"z": [reply]}, questions=Q_THREE)
    assert out3 == {"z": _rec([0, 1, 2], [reply], True)}
    assert gen3.calls[0][0] == [_expected_messages(photos3[0]["path"], Q_THREE)]
    assert all(len(m) == 0 for m, _ in gen3.calls[1:])


def test_answer_photos_model_stored_as_given(tmp_path):
    model = "/models/local-vlm-dir"
    _, _, out = _run(tmp_path, ["m1", "m2"], {"m1": [OK_BC], "m2": [BAD_PROSE, BAD_PROSE]}, model=model)
    assert out["m1"]["model"] == model
    assert out["m2"]["model"] == model


def test_answer_photos_replies_matched_by_position(tmp_path):
    # generate returns replies in the order of the message lists it was given; the records must
    # follow that order, not the photos' ids or paths
    ids = ["b", "a"]
    photos, _ = _photos(tmp_path, ids)
    seen = []

    def generate(message_lists, temperature):
        message_lists = list(message_lists)
        seen.append((len(message_lists), temperature))
        return [OK_BC, OK_A_UNK][:len(message_lists)] if not seen[1:] else ["not used"] * len(message_lists)

    out = vlm_answer.answer_photos(photos, Q_HAND, generate, MODEL)
    assert seen[0][0] == 2
    assert seen[0][1] == pytest.approx(0.0)
    assert all(n == 0 for n, _ in seen[1:])
    assert out["b"] == _rec([1, 2], [OK_BC], True)
    assert out["a"] == _rec([0, None], [OK_A_UNK], True)


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def test_selftest_prints_ok(capsys):
    vlm_answer.selftest()
    assert "selftest ok" in capsys.readouterr().out
