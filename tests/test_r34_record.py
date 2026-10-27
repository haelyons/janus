"""Black-box tests for tools/r34_record.py, written from spec.md only.

No pretrained model, GPU or network: a random two-layer OLMo-2 runs in bf16 on CPU.
Expected ids and differences are written out independently; expected activations
and attention rows come from a separate model forward, never from record helpers.

The stimuli are records with sid/text, as in the CLI contract. The production
hidden width 11008 becomes config.intermediate_size for this tiny model. The
unspecified JSON token-field name is left unconstrained.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import r34_record  # noqa: E402


READOUTS = ("suffix", "text")
ACTIVATIONS = ("hidden", "pre", "mid", "post")
TEXT_IDS = {"alpha": [4, 7, 11], "x": [13], "": []}
SUFFIX_IDS = {"\n": [20], "\nEND": [21, 22], "": []}
SENTENCES = [
    {"sid": "long", "text": "alpha", "group": "first", "meta": {"rank": 3}},
    {"sid": "short", "text": "x", "group": "second", "meta": {"rank": 1}},
    {"sid": "empty", "text": "", "group": "third", "meta": {"rank": 2}},
]


class TinyTokenizer:
    """BOS plus table-based ids; joint tokenization deliberately differs."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        pieces = {**TEXT_IDS, **SUFFIX_IDS,
                  "alpha\n": [30], "x\n": [31],
                  "alpha\nEND": [32], "x\nEND": [33]}
        return {"input_ids": ([1] if add_special_tokens else []) + list(pieces[text])}

    def convert_ids_to_tokens(self, ids):
        return [f"tok_{int(i)}" for i in ids]


@pytest.mark.parametrize("text,suffix,want,n_text", [
    ("alpha", None, [1, 4, 7, 11, 20], 4),
    ("x", None, [1, 13, 20], 2),
    ("alpha", "\nEND", [1, 4, 7, 11, 21, 22], 4),
    ("", "\nEND", [1, 21, 22], 1),
    ("alpha", "", [1, 4, 7, 11], 4),
])
def test_build_ids_separate_tokenization_bos_and_text_length(text, suffix, want, n_text):
    tokenizer = TinyTokenizer()
    if suffix is None:
        ids, got_n_text = r34_record.build_ids(tokenizer, text)
    else:
        ids, got_n_text = r34_record.build_ids(tokenizer, text, suffix=suffix)
    np.testing.assert_array_equal(ids, want)
    assert got_n_text == n_text
    assert tokenizer.calls == [(text, True), ("\n" if suffix is None else suffix, False)]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_decompose_signed_writes_preserve_all_axes(dtype):
    pre = np.array([[[1, -2, 4], [8, 3, -6]],
                    [[-5, 0, 2], [7, -9, 1]]], dtype=dtype)
    mid = np.array([[[4, -7, 4], [6, 9, -5]],
                    [[-9, 2, -1], [7, -8, 6]]], dtype=dtype)
    post = np.array([[[2, -6, 10], [9, 5, -5]],
                     [[-8, -3, 1], [3, -8, 4]]], dtype=dtype)
    want_attn = np.array([[[3, -5, 0], [-2, 6, 1]],
                          [[-4, 2, -3], [0, 1, 5]]], dtype=dtype)
    want_mlp = np.array([[[-2, 1, 6], [3, -4, 0]],
                         [[1, -5, 2], [-4, 0, -2]]], dtype=dtype)
    attn_write, mlp_write = r34_record.decompose(pre, mid, post)
    assert attn_write.shape == mlp_write.shape == pre.shape
    np.testing.assert_array_equal(attn_write, want_attn)
    np.testing.assert_array_equal(mlp_write, want_mlp)
    np.testing.assert_array_equal(pre + attn_write + mlp_write, post)


@pytest.fixture(scope="module")
def tiny_model():
    import torch
    from transformers import Olmo2Config, Olmo2ForCausalLM

    config = Olmo2Config(vocab_size=64, hidden_size=16, intermediate_size=40,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, max_position_embeddings=64,
                        attention_dropout=0.0, use_cache=False,
                        attn_implementation="eager")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(734)
        model = Olmo2ForCausalLM(config)
    return model.to(device="cpu", dtype=torch.bfloat16).eval()


def _numpy(tensor):
    return tensor.detach().float().cpu().numpy().copy()


def _reference(model, ids):
    """Independent forward, retaining raw last-layer output before final norm."""
    import torch

    n_layers = model.config.num_hidden_layers
    captured = {key: [None] * n_layers for key in ("mid", "hidden", "post")}
    handles = []

    def input_hook(key, layer):
        def capture(module, args):
            captured[key][layer] = _numpy(args[0][0])
        return capture

    def output_hook(layer):
        def capture(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            captured["post"][layer] = _numpy(value[0])
        return capture

    try:
        for layer, block in enumerate(model.model.layers):
            handles.append(block.mlp.register_forward_pre_hook(input_hook("mid", layer)))
            handles.append(block.mlp.down_proj.register_forward_pre_hook(input_hook("hidden", layer)))
            handles.append(block.register_forward_hook(output_hook(layer)))
        input_ids = torch.tensor([ids], dtype=torch.long, device="cpu")
        with torch.no_grad():
            embedding = _numpy(model.get_input_embeddings()(input_ids)[0])
            output = model(input_ids=input_ids, output_attentions=True,
                           output_hidden_states=True, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    states = [_numpy(state[0]) for state in output.hidden_states]
    return {"pre": np.stack(states[:-1]),
            **{key: np.stack(values) for key, values in captured.items()},
            "states": states, "embedding": embedding,
            "attn": np.stack([_numpy(weights[0]) for weights in output.attentions])}


@pytest.fixture(scope="module", params=[None, "\nEND"], ids=["default-suffix", "two-token-suffix"])
def recorded(tiny_model, request):
    import torch

    model = tiny_model
    suffix = "\n" if request.param is None else request.param
    ids = [[1] + TEXT_IDS[row["text"]] + SUFFIX_IDS[suffix] for row in SENTENCES]
    n_text = [1 + len(TEXT_IDS[row["text"]]) for row in SENTENCES]
    expected = [_reference(model, sentence_ids) for sentence_ids in ids]
    calls = []

    def observe(module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            input_ids = args[0]
        calls.append({"ids": _numpy(input_ids), "grad": torch.is_grad_enabled(),
                      "dtype": next(module.parameters()).dtype,
                      "device": input_ids.device.type,
                      "attentions": kwargs.get("output_attentions", module.config.output_attentions),
                      "hidden_states": kwargs.get("output_hidden_states", module.config.output_hidden_states)})

    handle = model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        tokenizer = TinyTokenizer()
        if request.param is None:
            rec = r34_record.record(model, tokenizer, SENTENCES, device="cpu")
        else:
            rec = r34_record.record(model, tokenizer, SENTENCES, suffix=suffix, device="cpu")
    finally:
        handle.remove()
    return {"rec": rec, "ids": ids, "n_text": n_text, "expected": expected,
            "calls": calls, "config": model.config}


def test_record_shapes_dtypes_ids_and_sentence_order(recorded):
    rec, config = recorded["rec"], recorded["config"]
    assert isinstance(rec, dict)
    assert len(rec["ids"]) == len(SENTENCES)
    for got, want in zip(rec["ids"], recorded["ids"]):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(rec["n_text"], recorded["n_text"])
    for readout in READOUTS:
        for key in ACTIVATIONS:
            value = rec[f"{key}_{readout}"]
            width = config.intermediate_size if key == "hidden" else config.hidden_size
            assert isinstance(value, np.ndarray)
            assert value.dtype == np.float32
            assert value.shape == (len(SENTENCES), config.num_hidden_layers, width)


def test_record_one_unpadded_bf16_cpu_forward_per_sentence_without_gradients(recorded):
    import torch

    calls = recorded["calls"]
    assert len(calls) == len(SENTENCES)
    for call, ids in zip(calls, recorded["ids"]):
        np.testing.assert_array_equal(call["ids"], [ids])
        assert not call["grad"]
        assert call["dtype"] == torch.bfloat16
        assert call["device"] == "cpu"
        assert call["attentions"]
    assert calls[0]["hidden_states"]


@pytest.mark.parametrize("readout", READOUTS)
def test_record_pre_matches_embeddings_and_model_hidden_states(recorded, readout):
    rec = recorded["rec"]
    for i, reference in enumerate(recorded["expected"]):
        position = len(recorded["ids"][i]) - 1 if readout == "suffix" else recorded["n_text"][i] - 1
        np.testing.assert_allclose(rec[f"pre_{readout}"][i], reference["pre"][:, position],
                                   rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(rec[f"pre_{readout}"][i, 0], reference["embedding"][position],
                                   rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("readout", READOUTS)
def test_record_mid_raw_post_and_down_projection_inputs(recorded, readout):
    rec = recorded["rec"]
    for i, reference in enumerate(recorded["expected"]):
        position = len(recorded["ids"][i]) - 1 if readout == "suffix" else recorded["n_text"][i] - 1
        for key in ("mid", "post", "hidden"):
            np.testing.assert_allclose(rec[f"{key}_{readout}"][i], reference[key][:, position],
                                       rtol=1e-5, atol=1e-6, err_msg=f"sentence {i}: {key}_{readout}")
        # hidden_states[-1] has final model normalization, so it is not the last raw post.
        for layer in range(recorded["config"].num_hidden_layers - 1):
            np.testing.assert_allclose(rec[f"post_{readout}"][i, layer],
                                       reference["states"][layer + 1][position], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(rec[f"post_{readout}"][:, :-1], rec[f"pre_{readout}"][:, 1:],
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("readout", READOUTS)
def test_record_attention_rows_match_model_and_have_zero_future_weight(recorded, readout):
    rows, config = recorded["rec"][f"attn_{readout}"], recorded["config"]
    assert len(rows) == len(SENTENCES)
    for i, (weights, reference) in enumerate(zip(rows, recorded["expected"])):
        length = len(recorded["ids"][i])
        position = length - 1 if readout == "suffix" else recorded["n_text"][i] - 1
        assert isinstance(weights, np.ndarray)
        assert weights.dtype == np.float32
        assert weights.shape == (config.num_hidden_layers, config.num_attention_heads, length)
        np.testing.assert_allclose(weights, reference["attn"][:, :, position, :], rtol=1e-5, atol=1e-6)
        assert np.all(weights >= 0)
        np.testing.assert_array_equal(weights[..., position + 1:], 0)
        # Eager bf16 attention rounds probabilities before conversion to float32.
        np.testing.assert_allclose(weights.sum(axis=-1), 1, rtol=0, atol=1e-2)


@pytest.mark.parametrize("state_index", [0, 1], ids=["pre-zero", "nonfinal-post"])
def test_record_asserts_first_sentence_hidden_state_identity_at_nonreadout_position(tiny_model, state_index):
    """Corrupt only position zero in the returned model reference, not layer data."""
    import torch

    def changed(states):
        states = list(states)
        states[state_index] = states[state_index].clone()
        states[state_index][0, 0, 0] += 16
        return tuple(states)

    def corrupt(module, args, output):
        states = getattr(output, "hidden_states", None)
        if states is not None:
            output.hidden_states = changed(states)
        elif isinstance(output, tuple):
            # Also permit an implementation that requests return_dict=False.
            values = list(output)
            for i, value in enumerate(values):
                if (isinstance(value, tuple) and len(value) == module.config.num_hidden_layers + 1
                        and all(torch.is_tensor(t) and t.ndim == 3 for t in value)):
                    values[i] = changed(value)
                    return tuple(values)
        return output

    handle = tiny_model.register_forward_hook(corrupt)
    try:
        with pytest.raises(AssertionError):
            r34_record.record(tiny_model, TinyTokenizer(), SENTENCES[:1], suffix="\nEND", device="cpu")
    finally:
        handle.remove()


def _contains_tokens(value, tokens):
    """Require token values without assigning the unspecified JSON field a name."""
    if isinstance(value, list):
        return value == tokens or any(_contains_tokens(item, tokens) for item in value)
    if isinstance(value, dict):
        return any(_contains_tokens(item, tokens) for item in value.values())
    return False


def test_save_arrays_ragged_attention_ids_and_original_sentence_records(recorded, tmp_path):
    out = tmp_path
    rec = recorded["rec"]
    r34_record.save(out, rec, SENTENCES)
    with np.load(out / "record.npz", allow_pickle=True) as saved:
        for readout in READOUTS:
            for key in ACTIVATIONS:
                name = f"{key}_{readout}"
                assert saved[name].dtype == np.float32
                np.testing.assert_array_equal(saved[name], rec[name])
            name = f"attn_{readout}"
            assert saved[name].dtype == object
            assert len(saved[name]) == len(SENTENCES)
            for got, want in zip(saved[name], rec[name]):
                np.testing.assert_array_equal(np.asarray(got, dtype=np.float32), want)
        assert saved["ids"].dtype == object
        assert len(saved["ids"]) == len(SENTENCES)
        for got, want in zip(saved["ids"], recorded["ids"]):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(saved["n_text"], recorded["n_text"])
    with (out / "sentences.json").open(encoding="utf-8") as stream:
        sentences = json.load(stream)
    assert isinstance(sentences, list)
    assert len(sentences) == len(SENTENCES)
    for got, original, ids in zip(sentences, SENTENCES, recorded["ids"]):
        for key, value in original.items():
            assert got[key] == value
        assert _contains_tokens(got, [f"tok_{i}" for i in ids])


def test_import_without_torch_or_transformers():
    code = ("import sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['transformers'] = None\n"
            "import r34_record\n")
    result = subprocess.run([sys.executable, "-c", code, str(ROOT / "tools")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_selftest_prints_ok(capsys):
    r34_record.selftest()
    assert "selftest ok" in capsys.readouterr().out


def test_main_selftest_flag(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["r34_record.py", "--selftest"])
    try:
        r34_record.main()
    except SystemExit as error:
        assert error.code in (None, 0), error.code
    assert "selftest ok" in capsys.readouterr().out
