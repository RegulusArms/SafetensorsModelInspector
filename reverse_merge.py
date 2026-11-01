#!/usr/bin/env python3
"""
Safetensors Merge Reverse Engineer
Works out which models were merged to create a target model, and how.

Two independent sources of evidence:
  1. Merge recipe metadata (A1111 / Forge checkpoint merger writes `sd_merge_recipe` and a nested
     `sd_merge_models` history, which describes the full merge tree, including nested merges).
  2. Weight analysis against a pool of candidate models: the target's weights are fitted as a
     linear combination of candidates (least squares on sampled tensor slices), per component and
     per UNet block. Candidates that are themselves merges of other candidates are detected as
     intermediate steps, which reveals nested merges like (a+b)+(c+d).

Note: nested weighted sums collapse into one flat linear combination, so without intermediate
models in the pool the nesting cannot be uniquely recovered from weights; plausible trees are
ranked by how "round" their merge ratios are.
"""

import argparse
import hashlib
import itertools
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

from compare_model import _TensorFile, compare_group, _group_sort_key
from inspect_model import _iter_safetensors_paths


# Contiguous elements sampled from each tensor (same position in every model)
SAMPLE_WINDOW = 2048
# Fitting limits
MAX_SOURCES = 6
MIN_IMPROVEMENT = 0.85      # each added source must cut the residual to below 85% of the previous
MIN_WEIGHT = 0.02           # weights smaller than this are pruned
MAX_ABS_WEIGHT = 1.25       # larger weights are treated as implausible (collinearity artefacts)
WEIGHT_SUM_RANGE = (0.8, 1.2)
BLOCK_WEIGHT_TOLERANCE = 0.05
IDENTICAL_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Part 1: merge recipe metadata
# ---------------------------------------------------------------------------

def a1111_legacy_hash(filepath: str) -> str:
    """A1111's short 'legacy' model hash (sha256 of 64 KB at offset 1 MB)."""
    with open(filepath, "rb") as f:
        f.seek(0x100000)
        return hashlib.sha256(f.read(0x10000)).hexdigest()[:8]


def _short_name(name: str) -> str:
    return Path(name).stem if name else name


def _load_json(raw):
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return None


UNRECORDED = "<third model not recorded>"


def _recipe_node(name: str, recipe: dict | None, models: dict, legacy: str | None, seen: set) -> dict:
    node = {"name": name, "legacy_hash": legacy, "recipe": None, "inputs": []}
    if not recipe or recipe.get("type") not in (None, "webui"):
        if recipe:
            node["recipe"] = {"method": f"unknown recipe type {recipe.get('type')!r}", "raw": recipe}
        return node
    method = recipe.get("interp_method")
    node["recipe"] = {
        "method": method,
        "multiplier": recipe.get("multiplier"),
        "bake_in_vae": recipe.get("bake_in_vae") if recipe.get("bake_in_vae") not in (None, "None") else None,
        "discard_weights": recipe.get("discard_weights") or None,
    }
    for role in ("primary", "secondary", "tertiary"):
        h = recipe.get(f"{role}_model_hash")
        if not h:
            continue
        info = models.get(h) or {}
        child_name = _short_name(info.get("name")) or f"<unknown {h[:10]}>"
        if h in seen:
            node["inputs"].append({"name": child_name, "legacy_hash": info.get("legacy_hash"),
                                   "recipe": None, "inputs": [], "cycle": True})
            continue
        node["inputs"].append(_recipe_node(child_name, info.get("sd_merge_recipe"), models,
                                           info.get("legacy_hash"), seen | {h}))
    return node


def _flatten_recipe(node: dict) -> dict[str, float]:
    """Leaf contributions of a recipe tree, as {model name: coefficient}."""
    r, ins = node.get("recipe"), node.get("inputs") or []
    if not r or not ins or str(r.get("method", "")).startswith("unknown"):
        return {node["name"]: 1.0}
    flat = [_flatten_recipe(c) for c in ins]
    m = float(r.get("multiplier") or 0)
    out: dict[str, float] = {}

    def add(d, k):
        for name, v in d.items():
            out[name] = out.get(name, 0.0) + k * v

    method = r.get("method")
    if method == "Weighted sum" and len(flat) >= 2:
        add(flat[0], 1 - m); add(flat[1], m)
    elif method == "Add difference" and len(flat) >= 3:
        add(flat[0], 1.0); add(flat[1], m); add(flat[2], -m)
    elif method == "Add difference" and len(flat) == 2:
        add(flat[0], 1.0); add(flat[1], m); add({UNRECORDED: 1.0}, -m)
    else:  # "No interpolation" and anything else: primary model passes through
        add(flat[0], 1.0)
    return {k: v for k, v in out.items() if abs(v) > 1e-9}


def _recipe_formula(node: dict) -> str:
    r, ins = node.get("recipe"), node.get("inputs") or []
    names = [c["name"] for c in ins]
    if not r or not ins:
        return ""
    m = r.get("multiplier")
    method = r.get("method")
    if method == "Weighted sum" and len(names) >= 2:
        f = f"{1 - m:g} x {names[0]}  +  {m:g} x {names[1]}"
    elif method == "Add difference" and len(names) >= 3:
        f = f"{names[0]}  +  {m:g} x ({names[1]} - {names[2]})"
    elif method == "Add difference" and len(names) == 2:
        f = f"{names[0]}  +  {m:g} x ({names[1]} - {UNRECORDED})"
    elif method == "No interpolation":
        f = f"copy of {names[0]}"
    else:
        f = f"{method} of {', '.join(names)}"
    extras = []
    if r.get("bake_in_vae"):
        extras.append(f"VAE baked in: {r['bake_in_vae']}")
    if r.get("discard_weights"):
        extras.append(f"discarded weights: {r['discard_weights']}")
    return f + (f"   [{'; '.join(extras)}]" if extras else "")


def analyze_merge_metadata(metadata: dict, root_name: str) -> dict:
    """Extract merge history from safetensors metadata."""
    out = {"recipe_tree": None, "flattened": None, "merged_from": None, "other_hints": {}}
    recipe = _load_json(metadata.get("sd_merge_recipe"))
    models = _load_json(metadata.get("sd_merge_models")) or {}
    if recipe:
        name = recipe.get("custom_name") or root_name
        tree = _recipe_node(name, recipe, models, None, set())
        out["recipe_tree"] = tree
        out["flattened"] = _flatten_recipe(tree)
    if metadata.get("modelspec.merged_from"):
        out["merged_from"] = [s.strip() for s in str(metadata["modelspec.merged_from"]).split(",") if s.strip()]
    for k in ("modelspec.title", "modelspec.author", "modelspec.description", "ss_sd_model_name",
              "ss_base_model_version", "Creator"):
        if metadata.get(k):
            out["other_hints"][k] = str(metadata[k])[:300]
    if any(k in metadata for k in ("prompt", "workflow")):
        out["other_hints"]["comfyui"] = "ComfyUI workflow embedded (not parsed)"
    return out


# ---------------------------------------------------------------------------
# Part 2: weight analysis
# ---------------------------------------------------------------------------

def _component(group: str) -> str:
    if group.startswith(("UNet", "Diffusion")):
        return "UNet"
    return group


def _window(key: str, n: int) -> tuple[int, int]:
    if n <= SAMPLE_WINDOW:
        return 0, n
    off = zlib.crc32(key.encode()) % (n - SAMPLE_WINDOW)
    return off, off + SAMPLE_WINDOW


def _sample_models(paths: list[str], progress_cb=None):
    """
    Sample the same tensor slices from every model.
    Returns ({component: matrix rows x models}, {component: [(block name, row_start, row_end)]},
    used paths, skipped paths).
    Each tensor's rows are normalized by the target's RMS so every tensor counts equally.
    progress_cb(done, total, message) is called while reading.
    """
    files = [_TensorFile(p) for p in paths]
    target = files[0]
    keys = []
    for key, info in target.tensor_info.items():
        if not target.supported(key) or target.numel(key) < 2:
            continue
        keys.append(key)

    # Drop candidates whose shapes don't match the target
    usable = [0]
    skipped = []
    for i, f in enumerate(files[1:], start=1):
        common = [k for k in keys if k in f.tensor_info and f.supported(k)
                  and f.tensor_info[k]["shape"] == target.tensor_info[k]["shape"]]
        if len(common) >= 0.9 * len(keys):
            usable.append(i)
        else:
            skipped.append(paths[i])
    files = [files[i] for i in usable]
    keys = [k for k in keys if all(k in f.tensor_info and f.tensor_info[k]["shape"] ==
                                   target.tensor_info[k]["shape"] and f.supported(k) for f in files)]

    windows = {k: _window(k, target.numel(k)) for k in keys}
    cols = {k: np.empty((b - a, len(files)), dtype=np.float64) for k, (a, b) in windows.items()}
    total = len(keys) * len(files)
    done = 0
    for j, f in enumerate(files):  # model-major order: sequential reads per file
        label = f"Sampling tensors from {Path(f.filepath).name} (model {j + 1} of {len(files)})"
        for k in keys:
            a, b = windows[k]
            cols[k][:, j] = f.flat(k, a, b).astype(np.float64)
            done += 1
            if progress_cb and (done % 100 == 0 or done == total):
                progress_cb(done, total, label)

    per_group: dict[str, list[np.ndarray]] = {}
    for k in keys:
        m = cols[k]
        m = m[np.isfinite(m).all(axis=1)]
        if not m.size:
            continue
        rms = float(np.sqrt(np.mean(m[:, 0] ** 2)))
        if rms == 0:
            rms = float(np.sqrt(np.mean(m ** 2))) or 1.0
        per_group.setdefault(compare_group(k), []).append(m / rms)
    comps: dict[str, list[np.ndarray]] = {}
    blocks: dict[str, list[tuple[str, int, int]]] = {}
    for g in sorted(per_group, key=_group_sort_key):
        m = np.vstack(per_group[g])
        c = _component(g)
        start = sum(len(x) for x in comps.get(c, []))
        comps.setdefault(c, []).append(m)
        blocks.setdefault(c, []).append((g, start, start + len(m)))
    components = {c: np.vstack(v) for c, v in comps.items()}
    return components, blocks, [paths[i] for i in usable], skipped


class _Gram:
    """
    Least squares over columns of a tall sample matrix M (rows x models), done on its Gram
    matrices M^T M (whole and per block). Every fit then costs a tiny models x models solve
    instead of a pass over millions of rows.
    """

    def __init__(self, M: np.ndarray, blocks: list[tuple[str, int, int]]):
        self.block_names = [name for name, _, _ in blocks]
        self.blocks = [M[a:b].T @ M[a:b] for _, a, b in blocks]
        self.G = sum(self.blocks)
        n = M.shape[1]
        # Exact pairwise distances (the Gram form loses precision for near-identical models)
        self.D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                self.D[i, j] = self.D[j, i] = float(np.linalg.norm(M[:, i] - M[:, j]))

    @staticmethod
    def _solve(G: np.ndarray, t: int, cols: list[int]):
        A = G[np.ix_(cols, cols)]
        c = G[cols, t]
        w = np.linalg.lstsq(A, c, rcond=None)[0]
        r2 = G[t, t] - 2 * w @ c + w @ A @ w
        return w, max(float(r2), 0.0) ** 0.5

    def lstsq(self, t: int, cols: list[int]):
        """Weights and residual norm for column t ~ columns cols."""
        return self._solve(self.G, t, list(cols))

    def block_lstsq(self, t: int, cols: list[int]):
        """Per-block (weights, residual norm) for column t ~ columns cols."""
        return [self._solve(B, t, list(cols)) for B in self.blocks]

    def norm(self, t: int) -> float:
        return float(self.G[t, t]) ** 0.5


def _plausible(w: np.ndarray) -> bool:
    return bool(np.all(np.abs(w) <= MAX_ABS_WEIGHT) and
                WEIGHT_SUM_RANGE[0] <= w.sum() <= WEIGHT_SUM_RANGE[1])


def fit_sources(gram: _Gram, target: int, allowed: list[int], max_sources: int = MAX_SOURCES,
                use_blocks: bool = False) -> dict:
    """
    Explain column `target` as a linear combination of columns in `allowed`,
    choosing sources greedily (forward stepwise least squares).

    With use_blocks, sources are scored by fitting each block separately, so block-weighted
    merges are found even when the average ratio matches some other model.
    """
    def score(cols):
        w, rn = gram.lstsq(target, cols)
        if use_blocks and len(cols) > 1:
            rn = sum(r ** 2 for _, r in gram.block_lstsq(target, cols)) ** 0.5
        return w, rn

    ynorm = gram.norm(target) or 1.0
    dists = {j: float(gram.D[target, j]) for j in allowed}
    if not dists:
        return {"sources": [], "weights": [], "nearest": None, "unexplained": 1.0}
    nearest = min(dists, key=dists.get)
    d_near = dists[nearest]
    if d_near / ynorm < IDENTICAL_TOLERANCE:
        return {"sources": [nearest], "weights": [1.0], "nearest": nearest, "nearest_rel_dist": 0.0,
                "residual_rel": 0.0, "unexplained": 0.0, "identical": True}

    chosen: list[int] = []
    best_w = np.array([])
    res = ynorm

    def best_addition(additions):
        best = None
        for add in additions:
            cols = chosen + list(add)
            w, rn = score(cols)
            if (len(cols) == 1 or _plausible(w)) and (best is None or rn < best[1]):
                best = (list(add), rn, w)
        return best

    while len(chosen) < max_sources:
        rest = [j for j in allowed if j not in chosen]
        best = best_addition((j,) for j in rest)
        if chosen and (best is None or best[1] > res * MIN_IMPROVEMENT) and len(chosen) + 2 <= max_sources:
            # Add-difference merges (A + m(B - C)) only show up when B and C are added together
            best = best_addition(itertools.combinations(rest, 2))
        if best is None or (chosen and best[1] > res * MIN_IMPROVEMENT):
            break
        chosen += best[0]
        res, best_w = best[1], best[2]

    # Prune negligible weights and refit
    while len(chosen) > 1:
        small = [i for i, w in enumerate(best_w) if abs(w) < MIN_WEIGHT]
        if not small:
            break
        del chosen[small[0]]
        best_w, res = score(chosen)

    # Related candidates are collinear, so several mixes can fit equally well: prefer the
    # explanation with the fewest sources, then all-positive weights, then the lowest residual.
    if len(chosen) > 2:
        tolerance = res * 1.25 + 0.01 * d_near
        options = []
        for r in range(1, len(chosen) + 1):
            for subset in itertools.combinations(chosen, r):
                w, rn = score(list(subset))
                if rn <= tolerance and (r == 1 or _plausible(w)):
                    options.append((r, bool(np.any(w < 0)), rn, list(subset), w))
        if options:
            _, _, res, chosen, best_w = min(options, key=lambda o: o[:3])

    order = np.argsort(-np.abs(best_w))
    return {
        "sources": [chosen[i] for i in order],
        "weights": [float(best_w[i]) for i in order],
        "nearest": nearest,
        "nearest_rel_dist": d_near / ynorm,
        "residual_rel": res / ynorm,
        # Fraction of the target's distance from its nearest candidate that the fit leaves unexplained
        "unexplained": res / d_near if d_near else 0.0,
        "identical": False,
    }


def _quality(unexplained: float) -> str:
    if unexplained < 0.05:
        return "fully explained"
    if unexplained < 0.3:
        return "mostly explained (some extra changes, e.g. training, LoRA, or fp16 rounding)"
    if unexplained < 0.7:
        return "partly explained (significant changes not from these candidates)"
    return "not explained (target differs in ways the candidates can't account for)"


def _all_trees(leaves: tuple):
    """All full binary trees over the given leaves (unordered children)."""
    if len(leaves) == 1:
        yield leaves[0]
        return
    first, rest = leaves[0], leaves[1:]
    for r in range(0, len(rest)):
        for combo in itertools.combinations(rest, r):
            left = (first,) + combo
            right = tuple(x for x in rest if x not in combo)
            if not right:
                continue
            for lt in _all_trees(left):
                for rt in _all_trees(right):
                    yield (lt, rt)


def _tree_weight(t, w: dict) -> float:
    return w[t] if not isinstance(t, tuple) else _tree_weight(t[0], w) + _tree_weight(t[1], w)


def _tree_ratios(t, w: dict, out: list):
    if not isinstance(t, tuple):
        return
    wl, wr = _tree_weight(t[0], w), _tree_weight(t[1], w)
    out.append(wr / (wl + wr))
    _tree_ratios(t[0], w, out)
    _tree_ratios(t[1], w, out)


def _tree_str(t, w: dict, names: dict) -> str:
    if not isinstance(t, tuple):
        return names[t]
    wl, wr = _tree_weight(t[0], w), _tree_weight(t[1], w)
    m = wr / (wl + wr)
    return f"({_tree_str(t[0], w, names)} <{1 - m:.2f}|{m:.2f}> {_tree_str(t[1], w, names)})"


def rank_merge_trees(weights: dict, names: dict, top: int = 3) -> list[dict]:
    """
    For positive weights, list the merge trees whose pairwise ratios are 'roundest'.
    Every tree reproduces the weights exactly; only the ratio roundness differs.
    """
    leaves = tuple(weights)
    if len(leaves) < 3 or len(leaves) > 6 or any(v <= 0 for v in weights.values()):
        return []
    total = sum(weights.values())
    w = {k: v / total for k, v in weights.items()}
    scored = []
    for t in _all_trees(leaves):
        ratios = []
        _tree_ratios(t, w, ratios)
        roughness = sum(min(abs(r - k * 0.05) for k in range(21)) for r in ratios)
        off_half = sum(abs(r - 0.5) for r in ratios)
        scored.append((roughness, off_half, t, ratios))
    scored.sort(key=lambda s: (round(s[0], 3), s[1]))
    return [{"tree": _tree_str(t, w, names), "ratios": [round(r, 3) for r in ratios],
             "roughness": round(rough, 4)} for rough, _, t, ratios in scored[:top]]


def _describe_weights(weights: list[float], names: list[str]) -> str:
    pos = [(n, w) for n, w in zip(names, weights) if w > 0]
    neg = [(n, w) for n, w in zip(names, weights) if w < 0]
    if not neg:
        return "weighted sum"
    if len(neg) == 1 and pos:
        c, m = neg[0][0], -neg[0][1]
        # Add difference A + m(B - C): B is the positive source whose weight is closest to m
        b = min((p for p in pos), key=lambda p: abs(p[1] - m))[0]
        return f"add difference: ... + {m:.3f} x ({b} - {c})"
    return "includes subtracted models (add-difference style merge)"


def analyze_merge_weights(target_path: str, candidate_paths: list[str], max_sources: int = MAX_SOURCES,
                          progress_cb=None) -> dict:
    """progress_cb(done, total, message) reports overall progress (total is fixed at 1000)."""
    def report(fraction: float, message: str):
        if progress_cb:
            progress_cb(min(int(fraction * 1000), 1000), 1000, message)

    paths = [target_path] + [p for p in candidate_paths
                             if os.path.realpath(p) != os.path.realpath(target_path)]
    # Reading dominates the run time: sampling takes the first 85% of the progress bar
    comps, blocks, used, skipped = _sample_models(
        paths, lambda d, t, m: report(0.85 * d / t, m))
    report(0.86, "Preparing least-squares fits...")
    grams = {c: _Gram(comps[c], blocks[c]) for c in comps}
    names = {i: Path(p).stem for i, p in enumerate(used)}
    cand = list(range(1, len(used)))
    result = {"candidates": [names[i] for i in cand],
              "skipped_incompatible": [Path(p).stem for p in skipped],
              "components": {}, "unet_blocks": None, "intermediates": [], "tree_hypotheses": []}
    if not cand:
        return result

    comp_order = sorted(comps, key=lambda c: (c != "UNet", c))
    for ci, comp in enumerate(comp_order):
        report(0.88 + 0.06 * ci / len(comp_order), f"Fitting {comp} against {len(cand)} candidates...")
        fit = fit_sources(grams[comp], 0, cand, max_sources, use_blocks=comp == "UNet")
        src_names = [names[s] for s in fit["sources"]]
        single = len(fit["sources"]) == 1 and not fit.get("identical")
        quality = ("identical copy" if fit.get("identical") else
                   f"not a merge of the candidates; closest relative is {names[fit['nearest']]} "
                   f"({fit['nearest_rel_dist']:.2%} relative distance) - one was likely made from the other "
                   f"by training, LoRA or merging with models not in the pool"
                   if single else _quality(fit["unexplained"]))
        result["components"][comp] = {
            "sources": src_names,
            "weights": fit["weights"],
            "weight_sum": sum(fit["weights"]),
            "nearest": names.get(fit["nearest"]),
            "nearest_rel_dist": fit.get("nearest_rel_dist"),
            "residual_rel": fit.get("residual_rel"),
            "unexplained": fit["unexplained"],
            "identical": fit.get("identical", False),
            "quality": quality,
            "description": _describe_weights(fit["weights"], src_names),
        }

    unet = grams.get("UNet")
    main = result["components"].get("UNet") or next(iter(result["components"].values()))
    main_idx = [i for n in main["sources"] for i, nm in names.items() if nm == n and i != 0]

    # Per-block weights for the main sources: detects block-weighted (MBW) merges
    if unet is not None and len(main_idx) >= 2:
        report(0.94, "Checking per-block weights (block-weighted merge?)...")
        rows, spread, res2 = [], 0.0, 0.0
        for g, (w, rn) in zip(unet.block_names, unet.block_lstsq(0, main_idx)):
            rows.append({"group": g, "weights": [float(x) for x in w]})
            res2 += rn ** 2
        for i in range(len(main_idx)):
            col = [r["weights"][i] for r in rows]
            spread = max(spread, max(col) - min(col))
        d_near = min(float(unet.D[0, j]) for j in cand)
        block_unexplained = res2 ** 0.5 / d_near if d_near else 0.0
        _, global_res = unet.lstsq(0, main_idx)
        global_unexplained = global_res / d_near if d_near else 0.0
        # Per-block weights scatter whenever the fit is poor, so only call it a block-weighted
        # merge if per-block ratios explain the target well and clearly beat a single ratio.
        block_weighted = (spread > BLOCK_WEIGHT_TOLERANCE * 2 and block_unexplained < 0.3 and
                          block_unexplained < 0.5 * global_unexplained)
        result["unet_blocks"] = {"sources": main["sources"], "rows": rows, "max_spread": spread,
                                 "block_weighted": block_weighted, "unexplained": block_unexplained,
                                 "inconclusive": block_unexplained >= 0.3}
        if block_weighted:
            main["quality"] = (f"{_quality(block_unexplained)} as a block-weighted merge "
                               f"(single-ratio fit leaves {global_unexplained:.1%} unexplained)")
            main["description"] = "block-weighted sum"
            main["unexplained"] = block_unexplained
        else:
            main["unexplained"] = global_unexplained
            main["quality"] = _quality(global_unexplained)

    # Nested merges. Linear relations run both ways (X = A/2 + B/2 also means A = 2X - B), so only
    # all-positive decompositions count as "X was made by merging these".
    def positive_merge(fit):
        return (len(fit["sources"]) >= 2 and fit["unexplained"] < 0.1 and
                all(w > 0 for w in fit["weights"]) and _plausible(np.array(fit["weights"])))

    if unet is not None and len(main_idx) >= 2:
        d_near = min(float(unet.D[0, j]) for j in cand)
        main_res = main["unexplained"] * d_near
        checks = [j for j in cand if j not in main_idx] + list(main_idx)
        # Intermediate steps: a candidate X made from some of the target's sources S, such that the
        # target is equally well explained as a merge of X with the remaining sources.
        for n_done, j in enumerate(checks[:len(checks) - len(main_idx)]):
            report(0.95 + 0.05 * n_done / len(checks), f"Looking for nested merges: is {names[j]} an intermediate step?")
            fit = fit_sources(unet, j, main_idx, max_sources)
            if not positive_merge(fit):
                continue
            outer = [j] + [s for s in main_idx if s not in fit["sources"]]
            if len(outer) < 2:
                continue  # X uses all of the target's sources, so "target = X" would just mean target == X
            w, rn = unet.lstsq(0, outer)
            if rn <= main_res * 1.25 + 0.01 * d_near and np.all(w > 0) and _plausible(w):
                result["intermediates"].append({
                    "model": names[j],
                    "sources": [names[s] for s in fit["sources"]],
                    "weights": fit["weights"],
                    "unexplained": fit["unexplained"],
                    "target_from": [names[s] for s in outer],
                    "target_weights": [float(x) for x in w],
                })
        # Ingredients of the target that are themselves merges of other candidates
        for n_done, s in enumerate(main_idx, start=len(checks) - len(main_idx)):
            report(0.95 + 0.05 * n_done / len(checks), f"Looking for nested merges: is {names[s]} itself a merge?")
            fit = fit_sources(unet, s, [j for j in cand if j != s], max_sources)
            if positive_merge(fit):
                result["intermediates"].append({
                    "model": names[s],
                    "sources": [names[x] for x in fit["sources"]],
                    "weights": fit["weights"],
                    "unexplained": fit["unexplained"],
                    "is_ingredient": True,
                })

    # Weights can't tell merge direction: a model derived FROM the target can pose as an ingredient
    target_mtime = os.path.getmtime(used[0])
    result["newer_sources"] = sorted({
        n for c in result["components"].values() for n in c["sources"]
        if os.path.getmtime(used[[k for k, v in names.items() if v == n][0]]) > target_mtime
    })

    report(1.0, "Ranking possible merge orders...")
    if main["weights"] and all(w > 0 for w in main["weights"]):
        result["tree_hypotheses"] = rank_merge_trees(
            dict(zip(range(len(main["sources"])), main["weights"])),
            dict(enumerate(main["sources"])),
        )
    return result


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def reverse_engineer(target_path: str, candidate_paths: list[str] | None = None,
                     max_sources: int = MAX_SOURCES, progress_cb=None) -> dict:
    target = _TensorFile(target_path)
    candidate_paths = [p for p in candidate_paths or []
                       if os.path.realpath(p) != os.path.realpath(target_path)]
    if progress_cb:
        progress_cb(0, 1000, "Reading merge recipe metadata...")
    meta = analyze_merge_metadata(target.metadata, Path(target_path).stem)

    # Match recipe ingredients to local files by A1111 legacy hash, falling back to file name
    local, local_names = {}, {}
    for p in candidate_paths:
        local_names[Path(p).stem.lower()] = Path(p).name
        try:
            local[a1111_legacy_hash(p)] = Path(p).name
        except OSError:
            pass

    found, missing = set(), set()

    def annotate(node, is_root=False):
        hit = local.get(node.get("legacy_hash")) or local_names.get(node["name"].lower())
        if hit:
            node["local_file"] = hit
        if not is_root and node["name"] != UNRECORDED:
            (found if hit else missing).add(node["name"])
        if hit and not is_root:
            return  # a local copy of this step makes its own ingredients unnecessary
        for c in node.get("inputs") or []:
            annotate(c)
    if meta["recipe_tree"]:
        annotate(meta["recipe_tree"], is_root=True)
        meta["ingredients_found_locally"] = sorted(found)
        meta["ingredients_missing_locally"] = sorted(missing - found)

    # A candidate whose own recipe history contains the target was made FROM it: exclude it
    target_hash = a1111_legacy_hash(target_path)
    target_names = {Path(target_path).stem.lower()}
    if meta["recipe_tree"]:
        target_names.add(meta["recipe_tree"]["name"].lower())
    descendants = []
    if progress_cb and candidate_paths:
        progress_cb(0, 1000, f"Checking {len(candidate_paths)} candidate models...")
    for p in list(candidate_paths):
        try:
            cmeta = _TensorFile(p).metadata
        except Exception:
            continue
        history = _load_json(cmeta.get("sd_merge_models")) or {}
        if any(info.get("legacy_hash") == target_hash or
               _short_name(info.get("name", "")).lower() in target_names for info in history.values()):
            descendants.append(Path(p).stem)
            candidate_paths.remove(p)

    weights = None
    if candidate_paths:
        weights = analyze_merge_weights(target_path, candidate_paths, max_sources, progress_cb)
        weights["descendants_excluded"] = descendants

    return {
        "target": str(Path(target_path).resolve()),
        "metadata": meta,
        "weights": weights,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _render_recipe(node: dict, lines: list, indent: int = 2):
    pad = " " * indent
    label = node["name"]
    if node.get("local_file"):
        label += f"  (found locally: {node['local_file']})"
    formula = _recipe_formula(node)
    if formula:
        lines.append(f"{pad}{label} = {formula}")
    else:
        tag = " (cycle)" if node.get("cycle") else ("  [original / no recipe recorded]" if not node.get("recipe") else "")
        lines.append(f"{pad}{label}{tag}")
    for c in node.get("inputs") or []:
        if c.get("recipe") and c.get("inputs") or c.get("local_file") or c.get("cycle"):
            _render_recipe(c, lines, indent + 4)


def format_reverse_report(result: dict) -> str:
    sep = "=" * 92
    lines = [sep, "  MERGE REVERSE ENGINEERING", sep, f"  Target: {result['target']}", ""]

    meta = result["metadata"]
    lines.append("  1. Merge recipe metadata")
    lines.append("  " + "-" * 40)
    if meta["recipe_tree"]:
        lines.append("  Recorded merge tree (from sd_merge_recipe / sd_merge_models):")
        lines.append("")
        _render_recipe(meta["recipe_tree"], lines, 4)
        lines.append("")
        lines.append("  Flattened contribution of original models:")
        flat = sorted(meta["flattened"].items(), key=lambda kv: -abs(kv[1]))
        for name, v in flat:
            lines.append(f"    {v:+8.4f}  {name}")
        lines.append(f"    (sum {sum(meta['flattened'].values()):.4f}; negative = subtracted by add-difference)")
    else:
        lines.append("  No merge recipe recorded in the file.")
    if meta["recipe_tree"] and "ingredients_found_locally" in meta:
        lines.append("")
        lines.append(f"  Recorded ingredients found among candidates: "
                     f"{', '.join(meta['ingredients_found_locally']) or 'none'}")
        if meta["ingredients_missing_locally"]:
            lines.append(f"  Not available locally ({len(meta['ingredients_missing_locally'])}): "
                         f"{', '.join(meta['ingredients_missing_locally'])}")
            lines.append("  The weight analysis below can only identify ingredients that are present, so")
            lines.append("  missing ones show up as unexplained changes.")
    if meta["merged_from"]:
        lines.append(f"  modelspec.merged_from: {', '.join(meta['merged_from'])}")
    for k, v in meta["other_hints"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    w = result["weights"]
    lines.append("  2. Weight analysis against candidate models")
    lines.append("  " + "-" * 40)
    if w is None:
        lines.append("  No candidate models given. Add the models you suspect were merged (or a folder of")
        lines.append("  them) to identify ingredients from the weights themselves.")
        lines.append(sep)
        return "\n".join(lines)
    lines.append(f"  Candidates ({len(w['candidates'])}): {', '.join(w['candidates']) or '-'}")
    if w.get("descendants_excluded"):
        lines.append(f"  Excluded (their recorded recipe shows they were made from the target): "
                     f"{', '.join(w['descendants_excluded'])}")
    if w["skipped_incompatible"]:
        lines.append(f"  Skipped (different architecture): {', '.join(w['skipped_incompatible'])}")
    if not w["candidates"]:
        lines.append(sep)
        return "\n".join(lines)
    lines.append("")
    for comp, c in w["components"].items():
        lines.append(f"  {comp}: {c['quality']}")
        if c["identical"]:
            lines.append(f"    identical to {c['sources'][0]}")
        else:
            for n, x in zip(c["sources"], c["weights"]):
                lines.append(f"    {x:+8.4f}  {n}")
            lines.append(f"    weight sum {c['weight_sum']:.4f} | {c['description']} | "
                         f"unexplained {c['unexplained']:.1%} of the distance to nearest candidate "
                         f"({c['nearest']})")
        lines.append("")

    b = w["unet_blocks"]
    if b:
        verdict = ("YES - weights vary by block (block-weighted / MBW merge)" if b["block_weighted"]
                   else "inconclusive - overall fit too poor to judge" if b.get("inconclusive")
                   else "no - same ratio in every block")
        lines.append(f"  Per-block UNet weights: {verdict}  (max spread {b['max_spread']:.3f})")
        lines.append("    " + f"{'block':26}" + "".join(f"{n[:14]:>16}" for n in b["sources"]))
        for r in b["rows"]:
            lines.append("    " + f"{r['group']:26}" + "".join(f"{x:16.3f}" for x in r["weights"]))
        lines.append("")

    if w.get("newer_sources"):
        lines.append("  Caution: these sources have newer files than the target, so they may have been made")
        lines.append("  FROM the target rather than merged into it (weights alone can't tell direction):")
        lines.append(f"    {', '.join(w['newer_sources'])}")
        lines.append("")

    if w["intermediates"]:
        lines.append("  Nested merges found among candidates:")
        for it in w["intermediates"]:
            parts = " + ".join(f"{x:.3f} x {n}" for n, x in zip(it["sources"], it["weights"]))
            role = " (ingredient of target)" if it.get("is_ingredient") else " (intermediate step)"
            lines.append(f"    {it['model']} = {parts}{role}  [unexplained {it['unexplained']:.1%}]")
            if it.get("target_from"):
                outer = " + ".join(f"{x:.3f} x {n}" for n, x in zip(it["target_from"], it["target_weights"]))
                lines.append(f"      -> target = {outer}")
        lines.append("")

    if w["tree_hypotheses"]:
        lines.append("  Possible merge orders (all give identical weights; ranked by roundest ratios):")
        lines.append("    notation: (A <wA|wB> B) = wA x A + wB x B")
        for i, t in enumerate(w["tree_hypotheses"], 1):
            lines.append(f"    {i}. {t['tree']}")
        lines.append("    Nested weighted sums flatten to one linear mix, so the order is a best guess")
        lines.append("    unless an intermediate model is among the candidates (see above).")
        lines.append("")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="reverse_merge.py",
        description="Work out which models were merged to create a target .safetensors model.",
    )
    parser.add_argument("target", help="The merged model to analyze")
    parser.add_argument("candidates", nargs="*",
                        help="Candidate source models (files or folders). Without these, only "
                             "the merge recipe metadata is read.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recurse into candidate folders")
    parser.add_argument("--max-sources", type=int, default=MAX_SOURCES,
                        help=f"Maximum number of source models to fit (default {MAX_SOURCES})")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--quiet", action="store_true", help="Do not print progress to stderr")
    args = parser.parse_args(argv)

    if not Path(args.target).is_file():
        print(f"[ERROR] Not a file: {args.target}", file=sys.stderr)
        return 1
    candidates = _iter_safetensors_paths(args.candidates, args.recursive) if args.candidates else []

    def progress(done, total, message=""):
        if sys.stderr.isatty():
            print(f"\r\033[K  [{done / total:6.1%}] {message}", end="", file=sys.stderr, flush=True)
            if done == total:
                print(file=sys.stderr)

    try:
        result = reverse_engineer(args.target, candidates, args.max_sources,
                                  progress_cb=None if args.quiet else progress)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2) if args.json else format_reverse_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
