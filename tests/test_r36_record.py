"""Independent acceptance test. Run with R36_RECORD=/path/to/r36_record.py."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

import numpy as np
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

ATOL, RTOL = 1e-7, 0.03
FAILURES = 0


def check(name, ok, detail=""):
    global FAILURES
    FAILURES += not bool(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""), flush=True)
    return bool(ok)


def near(actual, expected, atol=ATOL):
    return np.isclose(actual, expected, atol=atol, rtol=RTOL, equal_nan=True)


def array_check(name, actual, expected, dtype, values=True, atol=ATOL):
    shape_ok = check(name + " shape", actual.shape == expected.shape)
    check(name + " dtype", actual.dtype.kind in dtype if isinstance(dtype, str) else actual.dtype == dtype)
    if shape_ok and values:
        check(name + " values", np.all(near(actual, expected, atol)))
    return shape_ok


def fixture(root):
    torch.manual_seed(36)
    config = Olmo2Config(vocab_size=300, hidden_size=64, intermediate_size=96,
                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                        max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model_dir = root / "model"
    Olmo2ForCausalLM(config).save_pretrained(model_dir)
    rng = np.random.default_rng(36045)
    lengths = rng.integers(2, 41, 45)
    lengths[:3] = [2, 3, 40]
    docs, valid = [], []
    for d, length in enumerate(lengths):
        ids = rng.integers(3, 300, length, dtype=np.int32)
        ids[0] = 1
        mask = np.arange(length) > 0
        if d % 5 == 0 and length > 3:
            special = rng.integers(1, length - 1)
            ids[special], mask[special] = 2, False
        docs.append(ids)
        valid.append(mask)
    np.savez(root / "pool.npz", ids=np.concatenate(docs),
             off=np.r_[np.int64(0), np.cumsum(lengths, dtype=np.int64)], valid=np.concatenate(valid))
    special_doc = next(d for d, mask in enumerate(valid) if (~mask[1:]).any())
    special_pos = np.flatnonzero(~valid[special_doc])[1]
    triples = np.array([[2, 44, lengths[44] - 1], [0, 0, 0], [1, special_doc, special_pos],
                        [0, 12, lengths[12] - 1], [2, 17, 1], [1, special_doc, special_pos],
                        [2, 8, 0]], dtype=np.int64)
    np.save(root / "triples.npy", triples)
    return model_dir, docs, valid, triples


def reference(model_dir, docs, valid, triples):
    model = Olmo2ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16,
                                            local_files_only=True).eval()
    layers, width = model.config.num_hidden_layers, model.config.intermediate_size
    neurons, directions, names = [], [], []
    for layer in range(layers):
        draw = np.random.default_rng(zlib.crc32(f"r36:test/repo:L{layer}".encode())).choice(width, 32, replace=False)
        direction = np.random.default_rng(zlib.crc32(f"r36:dir:test/repo:L{layer}".encode())).standard_normal((width, 8))
        direction /= np.linalg.norm(direction, axis=0)
        neurons.append(draw)
        directions.append(direction.astype(np.float32))
        names.extend([f"L{layer}:n{i}" for i in draw] + [f"L{layer}:d{i}" for i in range(8)])
    down, mlp, handles = [None] * layers, [None] * layers, []

    def capture(target, layer):
        def hook(module, inputs):
            target[layer] = inputs[0][0].float().numpy().copy()
        return hook

    for layer, block in enumerate(model.model.layers):
        handles.append(block.mlp.down_proj.register_forward_pre_hook(capture(down, layer)))
        handles.append(block.mlp.register_forward_pre_hook(capture(mlp, layer)))
    activations, residuals, unit_values = [], [], []
    with torch.inference_mode():
        for ids in docs:
            model(torch.tensor(ids, dtype=torch.long)[None], use_cache=False)
            a = np.stack(down, axis=1)  # [position, layer, neuron]
            activations.append(a)
            residuals.append(np.stack(mlp, axis=1))
            unit_values.append(np.concatenate([np.concatenate((a[:, l, neurons[l]], a[:, l] @ directions[l]), axis=1)
                                               for l in range(layers)], axis=1))
    for handle in handles:
        handle.remove()
    tokens = np.concatenate([a[mask] for a, mask in zip(activations, valid)])
    record = {"tok_mean": tokens.mean(axis=0, dtype=np.float64).astype(np.float32),
              "tok_std": tokens.std(axis=0, dtype=np.float64).astype(np.float32), "units": np.array(names)}
    extrema = {}
    for prefix, operation, arg_operation, sign in [("max", np.max, np.argmax, -1), ("min", np.min, np.argmin, 1)]:
        values = np.stack([operation(a[m], axis=0) for a, m in zip(activations, valid)])
        positions = np.stack([np.flatnonzero(m)[arg_operation(a[m], axis=0)] for a, m in zip(activations, valid)])
        extrema[prefix] = values
        rank = np.argsort(sign * values, axis=0)[:32]
        side = "top" if prefix == "max" else "bot"
        record[side + "_val"] = np.take_along_axis(values, rank, axis=0).transpose(1, 0, 2).astype(np.float16)
        record[side + "_doc"] = rank.transpose(1, 0, 2).astype(np.int16)
        record[side + "_pos"] = np.take_along_axis(positions, rank, axis=0).transpose(1, 0, 2).astype(np.int16)
        record["u_" + prefix] = np.stack([operation(a[m], axis=0) for a, m in zip(unit_values, valid)])
        record["u_arg" + prefix] = np.stack([np.flatnonzero(m)[arg_operation(a[m], axis=0)] for a, m in zip(unit_values, valid)])
    record["u_tok"] = np.concatenate([np.where(m[:, None], a, np.nan) for a, m in zip(unit_values[:1000], valid[:1000])]).astype(np.float16)
    residual_tokens = np.concatenate([a[m] for a, m in zip(residuals[:200], valid[:200])])
    reps = {"reps": np.stack([residuals[d][p, l] for l, d, p in triples]).astype(np.float16),
            "mean": residual_tokens.mean(axis=0, dtype=np.float64).astype(np.float32),
            "std": residual_tokens.std(axis=0, dtype=np.float64).astype(np.float32)}
    return record, reps, activations, unit_values, extrema


def validate_record(out, expected, activations, unit_values, extrema, valid, model_dir):
    required = set(expected) | {"meta"}
    if not check("record fields", required <= out.keys(), str(sorted(required - out.keys()))):
        return
    shapes = {}
    # Scale absolute error by each neuron's/unit's RMS, including near-zero activations.
    neuron_atol = ATOL + 0.04 * np.hypot(expected["tok_mean"], expected["tok_std"])
    unit_atol = ATOL + 0.04 * np.sqrt(np.mean(np.concatenate(unit_values) ** 2, axis=0))
    for name, ref in expected.items():
        dtype = "US" if name == "units" else "iu" if name.startswith("u_arg") else ref.dtype
        # The spec does not fix the integer width of u_argmax/u_argmin.
        compare = name not in {"units", "top_doc", "top_pos", "bot_doc", "bot_pos", "u_argmax", "u_argmin"}
        atol = unit_atol if name.startswith("u_") else neuron_atol[:, None, :] if name.startswith(("top", "bot")) else neuron_atol
        shapes[name] = array_check("record " + name, out[name], ref, dtype, compare, atol)
    if shapes["units"]:
        check("record unit names and draw", np.array_equal(out["units"].astype(str), expected["units"]))
    meta = out["meta"]
    if check("record meta scalar string", meta.shape == () and meta.dtype.kind in "US"):
        metadata = json.loads(meta.item())
        check("record meta fields", isinstance(metadata, dict) and {"repo", "revision", "units_repo", "n_docs"} <= metadata.keys())
        for key, value in {"repo": str(model_dir), "units_repo": "test/repo", "n_docs": len(valid)}.items():
            check("record meta " + key, metadata.get(key) == value)
        # No --revision is supplied; the spec does not define the local default.
        check("record meta revision", "revision" in metadata and (metadata["revision"] is None or isinstance(metadata["revision"], str)))
    if shapes["u_tok"]:
        check("record u_tok NaN placement", np.array_equal(np.isnan(out["u_tok"]), np.isnan(expected["u_tok"])))
    for prefix, operation in [("max", np.argmax), ("min", np.argmin)]:
        name = "u_arg" + prefix
        if not shapes[name] or out[name].dtype.kind not in "iu":
            continue
        ok = True
        for d, a in enumerate(unit_values):
            pos = out[name][d].astype(np.int64)
            if np.any((pos < 0) | (pos >= len(a))):
                ok = False
                continue
            candidate = a[pos, np.arange(a.shape[1])]
            ok &= bool(valid[d][pos].all() and near(candidate, expected["u_" + prefix][d], unit_atol).all())
            directions = np.arange(a.shape[1]) % 40 >= 32
            exact_tie = candidate[directions] == expected["u_" + prefix][d, directions]
            ok &= bool((~exact_tie | (pos[directions] == expected[name][d, directions])).all())
            # Stored neuron values are exactly representable in float16; enforce first ties there.
            if shapes["u_tok"]:
                start = sum(map(len, valid[:d]))
                stored = out["u_tok"][start:start + len(a)]
                columns = np.array([i for i in range(a.shape[1]) if i % 40 < 32])
                first = np.flatnonzero(valid[d])[operation(stored[valid[d]][:, columns], axis=0)]
                ok &= bool(np.array_equal(pos[columns], first))
        check("record " + name + " valid extrema and first ties", ok)
    for side, prefix, sign in [("top", "max", -1), ("bot", "min", 1)]:
        if not all(shapes[side + suffix] for suffix in ["_val", "_doc", "_pos"]):
            continue
        ok = True
        for layer in range(len(activations[0][0])):
            for neuron in range(activations[0].shape[2]):
                atol = neuron_atol[layer, neuron]
                ds, ps = out[side + "_doc"][layer, :, neuron], out[side + "_pos"][layer, :, neuron]
                values = out[side + "_val"][layer, :, neuron].astype(np.float32)
                if len(set(ds.tolist())) != 32 or any(d < 0 or d >= len(valid) or p < 0 or p >= len(valid[d]) or not valid[d][p] for d, p in zip(ds, ps)):
                    ok = False
                    continue
                source = extrema[prefix][:, layer, neuron]
                selected = source[ds]
                at_pos = np.array([activations[d][p, layer, neuron] for d, p in zip(ds, ps)])
                ok &= bool(near(values, selected, atol).all() and near(at_pos, selected, atol).all())
                ok &= bool(((sign * np.diff(values) >= 0) | near(values[:-1], values[1:], atol)).all())
                # Compare sets of (document, position), ignoring order among tied ranks.
                pairs = set(zip(ds.tolist(), ps.tolist()))
                for d, p in zip(expected[side + "_doc"][layer, :, neuron], expected[side + "_pos"][layer, :, neuron]):
                    if (int(d), int(p)) not in pairs:
                        alternatives = [q for q, r in pairs if (q == d and near(activations[q][r, layer, neuron], source[d], atol)) or
                                        (d not in ds and near(source[q], source[d], atol))]
                        ok &= bool(alternatives)
                omitted = np.delete(source, ds)
                ok &= bool(((sign * omitted >= np.max(sign * selected)) | near(omitted, selected[np.argmax(sign * selected)], atol)).all())
        check("record " + side + " ordered extrema and tied pair sets", ok)


def run_tool(tool, command, root, model_dir):
    output = root / (command + ".npz")
    args = [sys.executable, "-I", tool, command, "--model", str(model_dir)]
    args += ["--units-repo", "test/repo"] if command == "record" else ["--triples", str(root / "triples.npy")]
    args += ["--pool", str(root / "pool.npz"), "--out", str(output), "--batch", "4", "--device", "cpu"]
    result = subprocess.run(args, capture_output=True, text=True, timeout=180)
    if not check(command + " subprocess", result.returncode == 0, "" if result.returncode == 0 else (result.stderr or result.stdout)[-1500:].replace("\n", " | ")):
        return None
    with np.load(output, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def main():
    if not check("R36_RECORD supplied", bool(os.environ.get("R36_RECORD"))):
        return 1
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="r36-test-", dir=Path(__file__).resolve().parent) as directory:
        root = Path(directory)
        model_dir, docs, valid, triples = fixture(root)
        expected, reps, activations, unit_values, extrema = reference(model_dir, docs, valid, triples)
        for command in ["record", "reps"]:
            try:
                out = run_tool(os.environ["R36_RECORD"], command, root, model_dir)
                if out is None:
                    continue
                if command == "record":
                    validate_record(out, expected, activations, unit_values, extrema, valid, model_dir)
                elif check("reps fields", reps.keys() <= out.keys()):
                    for name, value in reps.items():
                        atol = ATOL + 0.04 * np.hypot(reps["mean"], reps["std"])
                        array_check("reps " + name, out[name], value, value.dtype, atol=atol[triples[:, 0]] if name == "reps" else atol)
            except Exception as exc:
                check(command + " validation", False, f"{type(exc).__name__}: {exc}")
    return int(FAILURES > 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        check("test execution", False, f"{type(exc).__name__}: {exc}")
        sys.exit(1)
