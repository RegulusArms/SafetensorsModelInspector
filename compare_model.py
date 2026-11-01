#!/usr/bin/env python3
"""
Safetensors Model Comparer
Diffs a fine-tuned / merged .safetensors model against the base model it came from,
tensor by tensor, and summarizes what changed per component.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from inspect_model import read_safetensors_header, _tensor_component_bucket, format_params


# ---------------------------------------------------------------------------
# Tensor loading
# ---------------------------------------------------------------------------

NUMPY_DTYPES = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": np.uint16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8,
    "BOOL": np.bool_,
}

# Elements per chunk when reducing large tensors (keeps float64 temporaries ~64 MB)
CHUNK_ELEMENTS = 8_000_000

# A tensor whose direction is unchanged (cosine >= this) but whose magnitude moved by more
# than RESCALE_RATIO is reported as "rescaled" rather than trained.
RESCALE_COSINE = 0.9999
RESCALE_RATIO = 1.5


class _TensorFile:
    """Memory-mapped safetensors file giving flat float arrays per tensor."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.metadata, self.tensor_info, self.file_size = read_safetensors_header(filepath)
        with open(filepath, "rb") as f:
            header_size = int.from_bytes(f.read(8), "little")
        self.data_start = 8 + header_size
        self.mm = np.memmap(filepath, dtype=np.uint8, mode="r")

    def supported(self, key: str) -> bool:
        return self.tensor_info[key]["dtype"] in NUMPY_DTYPES

    def flat(self, key: str) -> np.ndarray:
        info = self.tensor_info[key]
        start, end = info["data_offsets"]
        raw = self.mm[self.data_start + start: self.data_start + end]
        arr = raw.view(NUMPY_DTYPES[info["dtype"]])
        if info["dtype"] == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr


def _tensor_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Reduce two flat tensors to diff / norm / dot sums, skipping non-finite elements."""
    diff2 = a2 = b2 = dot = 0.0
    nonfinite_a = nonfinite_b = 0
    for i in range(0, a.size, CHUNK_ELEMENTS):
        x = a[i:i + CHUNK_ELEMENTS].astype(np.float64)
        y = b[i:i + CHUNK_ELEMENTS].astype(np.float64)
        fx, fy = np.isfinite(x), np.isfinite(y)
        nonfinite_a += int(x.size - fx.sum())
        nonfinite_b += int(y.size - fy.sum())
        if not (fx.all() and fy.all()):
            ok = fx & fy
            x, y = x[ok], y[ok]
        d = x - y
        diff2 += float(d @ d)
        a2 += float(x @ x)
        b2 += float(y @ y)
        dot += float(x @ y)
    return {
        "diff2": diff2, "a2": a2, "b2": b2, "dot": dot,
        "nonfinite_a": nonfinite_a, "nonfinite_b": nonfinite_b,
    }


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

_BUCKET_LABELS = {
    "unet": "UNet",
    "transformer": "Transformer",
    "vae": "VAE",
    "text_encoder": "Text Encoder",
    "text_encoder_2": "Text Encoder 2",
}


def compare_group(key: str) -> str:
    """Map a tensor key to a fine-grained group label for the comparison table."""
    parts = key.split(".")
    if key.startswith("model.diffusion_model."):
        rest = parts[2:]
        if rest and rest[0] in ("input_blocks", "output_blocks") and len(rest) > 1:
            return f"UNet {rest[0]}.{rest[1]}"
        if rest and rest[0] in ("double_blocks", "single_blocks", "joint_blocks",
                                "transformer_blocks", "blocks") and len(rest) > 1:
            return f"Diffusion {rest[0]}"
        return f"UNet {rest[0]}" if rest else "UNet"
    if key.startswith("conditioner.embedders.") and len(parts) > 2:
        return f"Text Encoder {int(parts[2]) + 1}" if parts[2].isdigit() else "Text Encoder"
    bucket = _tensor_component_bucket(key)
    if bucket:
        return _BUCKET_LABELS.get(bucket, bucket)
    return parts[0]


def _group_sort_key(name: str):
    head, _, tail = name.rpartition(".")
    if tail.isdigit():
        return (head, int(tail))
    return (name, -1)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _rel(diff2: float, base2: float) -> float | None:
    return (diff2 / base2) ** 0.5 if base2 > 0 else None


def _cosine(dot: float, a2: float, b2: float) -> float | None:
    return dot / (a2 ** 0.5 * b2 ** 0.5) if a2 > 0 and b2 > 0 else None


def _group_verdict(g: dict) -> str:
    if g["identical"] == g["tensors"]:
        return "unchanged"
    if g["rescaled"] and (g["min_cosine"] or 0) >= RESCALE_COSINE:
        return "rescaled copy (not trained)"
    if g["rescaled"]:
        return "different variant (rescaled + modified)"
    cos = g["cosine"]
    if cos is not None and cos < 0.9:
        return "replaced / heavily changed"
    if g["identical"] > 0:
        return "partially changed"
    return "changed"


def compare_models(tuned_path: str, base_path: str, top_n: int = 15, progress_cb=None) -> dict:
    """
    Compare a tuned model against its base model.

    progress_cb(done, total) is called after each tensor, if given.
    Returns a JSON-serializable dict.
    """
    tuned = _TensorFile(tuned_path)
    base = _TensorFile(base_path)

    keys_t, keys_b = set(tuned.tensor_info), set(base.tensor_info)
    common = sorted(keys_t & keys_b)

    groups = defaultdict(lambda: {
        "tensors": 0, "identical": 0, "rescaled": 0, "params": 0, "min_cosine": None,
        "diff2": 0.0, "a2": 0.0, "b2": 0.0, "dot": 0.0,
        "nonfinite_tuned": 0, "nonfinite_base": 0,
    })
    shape_mismatches, dtype_changes, unsupported = [], defaultdict(int), []
    nonfinite_tensors, rescaled_tensors, per_tensor = [], [], []

    for i, key in enumerate(common):
        ti, bi = tuned.tensor_info[key], base.tensor_info[key]
        group = compare_group(key)
        if ti["shape"] != bi["shape"]:
            shape_mismatches.append({"key": key, "tuned": ti["shape"], "base": bi["shape"]})
        elif not (tuned.supported(key) and base.supported(key)):
            unsupported.append(key)
        else:
            if ti["dtype"] != bi["dtype"]:
                dtype_changes[(group, bi["dtype"], ti["dtype"])] += 1
            a, b = tuned.flat(key), base.flat(key)
            s = _tensor_stats(a, b)
            g = groups[group]
            g["tensors"] += 1
            g["params"] += int(a.size)
            g["identical"] += int(s["diff2"] == 0.0 and s["nonfinite_a"] == s["nonfinite_b"])
            g["nonfinite_tuned"] += s["nonfinite_a"]
            g["nonfinite_base"] += s["nonfinite_b"]
            if s["nonfinite_a"] or s["nonfinite_b"]:
                nonfinite_tensors.append({
                    "key": key, "tuned": s["nonfinite_a"], "base": s["nonfinite_b"],
                })
            cos = _cosine(s["dot"], s["a2"], s["b2"])
            ratio = (s["a2"] / s["b2"]) ** 0.5 if s["b2"] > 0 else None
            if cos is not None and ratio is not None and cos >= RESCALE_COSINE and \
                    (ratio > RESCALE_RATIO or ratio < 1 / RESCALE_RATIO):
                # Rescaled tensors would swamp the change stats, so they are reported separately
                g["rescaled"] += 1
                rescaled_tensors.append({"key": key, "norm_ratio": ratio})
            else:
                for f in ("diff2", "a2", "b2", "dot"):
                    g[f] += s[f]
                rel = _rel(s["diff2"], s["b2"])
                if rel is not None and rel > 0:
                    per_tensor.append({"key": key, "rel_change": rel, "cosine": cos})
            if cos is not None:
                g["min_cosine"] = cos if g["min_cosine"] is None else min(g["min_cosine"], cos)
        if progress_cb:
            progress_cb(i + 1, len(common))

    group_rows = []
    tot = {"diff2": 0.0, "b2": 0.0}
    for name in sorted(groups, key=_group_sort_key):
        g = groups[name]
        tot["diff2"] += g["diff2"]
        tot["b2"] += g["b2"]
        row = {
            "group": name,
            "tensors": g["tensors"],
            "identical": g["identical"],
            "rescaled": g["rescaled"],
            "params": g["params"],
            "rel_change": _rel(g["diff2"], g["b2"]),
            "cosine": _cosine(g["dot"], g["a2"], g["b2"]),
            "min_cosine": g["min_cosine"],
            "nonfinite_tuned": g["nonfinite_tuned"],
            "nonfinite_base": g["nonfinite_base"],
        }
        row["verdict"] = _group_verdict(row)
        group_rows.append(row)

    per_tensor.sort(key=lambda r: r["rel_change"], reverse=True)
    rescaled_tensors.sort(key=lambda r: abs(np.log(r["norm_ratio"])), reverse=True)

    return {
        "tuned": str(Path(tuned_path).resolve()),
        "base": str(Path(base_path).resolve()),
        "tuned_tensor_count": len(keys_t),
        "base_tensor_count": len(keys_b),
        "keys_only_in_tuned": sorted(keys_t - keys_b),
        "keys_only_in_base": sorted(keys_b - keys_t),
        "shape_mismatches": shape_mismatches,
        "unsupported_dtype_keys": unsupported,
        "dtype_changes": [
            {"group": g, "base_dtype": bd, "tuned_dtype": td, "count": n}
            for (g, bd, td), n in sorted(dtype_changes.items())
        ],
        "groups": group_rows,
        "overall_rel_change": _rel(tot["diff2"], tot["b2"]),
        "nonfinite_tensors": nonfinite_tensors,
        "rescaled_tensor_count": len(rescaled_tensors),
        "rescaled_tensors": rescaled_tensors[:top_n],
        "most_changed_tensors": per_tensor[:top_n],
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt(v, spec=".4%"):
    return "-" if v is None else format(v, spec)


def format_compare_report(result: dict) -> str:
    lines = []
    sep = "=" * 92
    lines += [
        sep,
        "  MODEL COMPARISON",
        sep,
        f"  Tuned:  {result['tuned']}  ({result['tuned_tensor_count']} tensors)",
        f"  Base:   {result['base']}  ({result['base_tensor_count']} tensors)",
        "",
        "  rel change = ||tuned - base|| / ||base||   (0% = identical, ~100% = unrelated weights)",
        "  Rescaled tensors and NaN/Inf values are excluded from rel change and cosine.",
        "",
    ]

    header = (f"  {'group':26} {'tensors':>7} {'identical':>9} {'rescaled':>8} {'params':>9} "
              f"{'rel change':>11} {'cosine':>8}  verdict")
    lines += [header, "  " + "-" * (len(header) - 2)]
    for g in result["groups"]:
        lines.append(
            f"  {g['group']:26} {g['tensors']:7} {g['identical']:9} {g['rescaled']:8} {format_params(g['params']):>9} "
            f"{_fmt(g['rel_change'], '.3%'):>11} {_fmt(g['cosine'], '.4f'):>8}  {g['verdict']}"
        )
    lines += ["", f"  Overall relative change: {_fmt(result['overall_rel_change'])}", ""]

    if result["keys_only_in_tuned"] or result["keys_only_in_base"]:
        lines.append("  Key differences:")
        for k in result["keys_only_in_tuned"]:
            lines.append(f"    + only in tuned: {k}")
        for k in result["keys_only_in_base"]:
            lines.append(f"    - only in base:  {k}")
        lines.append("")
    if result["shape_mismatches"]:
        lines.append("  Shape mismatches (not compared):")
        for m in result["shape_mismatches"]:
            lines.append(f"    {m['key']}  tuned {m['tuned']} vs base {m['base']}")
        lines.append("")
    if result["unsupported_dtype_keys"]:
        lines.append(f"  Skipped {len(result['unsupported_dtype_keys'])} tensor(s) with unsupported dtypes (e.g. FP8)")
        lines.append("")
    if result["dtype_changes"]:
        lines.append("  Dtype changes (base -> tuned):")
        for d in result["dtype_changes"]:
            lines.append(f"    {d['group']:26} {d['base_dtype']} -> {d['tuned_dtype']}  ({d['count']} tensors)")
        lines.append("")
    if result["nonfinite_tensors"]:
        lines.append("  WARNING: tensors containing NaN/Inf values (excluded from the math above):")
        for n in result["nonfinite_tensors"]:
            lines.append(f"    {n['key']}  tuned: {n['tuned']:,}  base: {n['base']:,}")
        lines.append("")
    if result["rescaled_tensors"]:
        lines.append(f"  Rescaled tensors: {result['rescaled_tensor_count']} "
                     f"(same direction, magnitude changed by >{RESCALE_RATIO}x; typical of a swapped VAE variant)")
        for r in result["rescaled_tensors"]:
            lines.append(f"    x{r['norm_ratio']:<8.3f} {r['key']}")
        lines.append("")
    if result["most_changed_tensors"]:
        lines.append("  Most-changed tensors:")
        for r in result["most_changed_tensors"]:
            lines.append(f"    {_fmt(r['rel_change'], '9.3%')}  cos {_fmt(r['cosine'], '.4f')}  {r['key']}")
        lines.append("")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="compare_model.py",
        description="Compare a fine-tuned or merged .safetensors model against its base model.",
    )
    parser.add_argument("tuned", help="The fine-tuned / merged model")
    parser.add_argument("base", help="The base model it was derived from")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--top", type=int, default=15, help="How many top tensors to list (default 15)")
    parser.add_argument("--quiet", action="store_true", help="Do not print progress to stderr")
    args = parser.parse_args(argv)

    for p in (args.tuned, args.base):
        if not Path(p).is_file():
            print(f"[ERROR] Not a file: {p}", file=sys.stderr)
            return 1

    def progress(done, total):
        if sys.stderr.isatty() and (done == total or done % 50 == 0):
            print(f"\r  Comparing tensors... {done}/{total}", end="", file=sys.stderr, flush=True)
            if done == total:
                print(file=sys.stderr)

    try:
        result = compare_models(args.tuned, args.base, top_n=args.top,
                                progress_cb=None if args.quiet else progress)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_compare_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
