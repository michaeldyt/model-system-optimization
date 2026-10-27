#!/usr/bin/env python3
"""Compare CPU-loaded tensor snapshots against an explicit numerical contract.

Input: torch.save(Tensor) or torch.save(dict[str, Tensor]), dense ordinary dtypes.
No default tolerances; no model execution; no pickle fallback. Exit: 0 pass,
1 structural/numerical mismatch, 2 invalid input/configuration/metric overflow.
"""

import argparse
import json
import math
from pathlib import Path

import torch

SUPPORTED = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def load_tensors(path, *, allow_empty=False):
    values = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(values, torch.Tensor):
        values = {"output": values}
    if not isinstance(values, dict) or (not values and not allow_empty):
        raise ValueError(f"{path}: expected a nonempty tensor or string-to-tensor dictionary")
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"{path}: expected string keys and Tensor values")
        if value.layout != torch.strided or value.is_quantized or value.dtype not in SUPPORTED:
            raise ValueError(f"{path}/{key}: unsupported layout or dtype; export an explicit numerical view")
        if value.numel() == 0:
            raise ValueError(f"{path}/{key}: empty tensor cannot establish coverage")
    return values


def bits_equal(a, b):
    return torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def compare_field(a, b, mask, args):
    result = {
        "reference_shape": list(a.shape),
        "candidate_shape": list(b.shape),
        "reference_dtype": str(a.dtype),
        "candidate_dtype": str(b.dtype),
        "reference_stride": list(a.stride()),
        "candidate_stride": list(b.stride()),
    }
    if a.shape != b.shape or a.dtype != b.dtype:
        return {**result, "passed": False, "reason": "shape or dtype mismatch"}
    if mask is not None:
        if mask.dtype != torch.bool or mask.shape != a.shape or not mask.any().item():
            raise ValueError("Each mask must be boolean, same shape as its field, and select at least one element")
        aa, bb = a[mask], b[mask]
    else:
        aa, bb = a.reshape(-1), b.reshape(-1)
    result.update(elements=a.numel(), checked_elements=aa.numel(), coverage=aa.numel() / a.numel())
    # Nonfinite output is always reported, including outside a numerical mask.
    nonfinite = int((~torch.isfinite(a)).sum()) + int((~torch.isfinite(b)).sum())
    result["nonfinite_values"] = nonfinite
    if nonfinite:
        return {**result, "passed": False, "reason": "nonfinite value in output"}
    exact = bits_equal(aa, bb)
    result["bit_identical"] = exact
    if not aa.is_floating_point():
        return {**result, "passed": exact, "comparison": "exact discrete values"}
    reference, candidate = aa.double(), bb.double()
    delta = candidate - reference
    max_abs = delta.abs().max().item()
    rmse = delta.square().mean().sqrt().item()
    ref_norm, delta_norm = reference.norm().item(), delta.norm().item()
    if not all(math.isfinite(v) for v in (max_abs, rmse, ref_norm, delta_norm)):
        raise ValueError("FP64 metric overflow; rescale a diagnostic copy or use a problem-specific stable metric")
    relative = delta_norm / ref_norm if ref_norm else (0.0 if delta_norm == 0 else None)
    if relative is not None and not math.isfinite(relative):
        raise ValueError("Relative L2 overflow; define a suitable absolute-error contract")
    result.update(max_abs=max_abs, rmse=rmse, reference_l2=ref_norm, relative_l2=relative)
    if args.mode == "bitwise":
        return {**result, "passed": exact}
    failed = delta.abs() > args.atol + args.rtol * reference.abs()
    relative_ok = args.relative_l2_max is None or (relative is not None and relative <= args.relative_l2_max)
    result.update(elementwise_failed_fraction=failed.double().mean().item(), relative_l2_passed=relative_ok)
    if relative is None:
        result["relative_l2_note"] = "Nonzero difference against an all-zero reference; relative error undefined"
    return {**result, "passed": not failed.any().item() and relative_ok}


def run(args):
    reference, candidate = load_tensors(args.reference), load_tensors(args.candidate)
    if reference.keys() != candidate.keys():
        return {
            "passed": False,
            "reason": "field sets differ",
            "missing": sorted(reference.keys() - candidate.keys()),
            "extra": sorted(candidate.keys() - reference.keys()),
        }
    masks = load_tensors(args.mask_file, allow_empty=True) if args.mask_file else {}
    if masks.keys() - reference.keys():
        raise ValueError("Mask has unknown output fields")
    fields = {key: compare_field(reference[key], candidate[key], masks.get(key), args) for key in sorted(reference)}
    return {
        "passed": all(row["passed"] for row in fields.values()),
        "mode": args.mode,
        "atol": args.atol,
        "rtol": args.rtol,
        "relative_l2_max": args.relative_l2_max,
        "masked_fields": sorted(masks),
        "fields": fields,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--mode", choices=("bitwise", "tolerance"), required=True)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--relative-l2-max", type=float)
    parser.add_argument("--mask-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.resolve() in {
        p.resolve() for p in (args.reference, args.candidate, args.mask_file) if p is not None
    }:
        parser.error("Report output must not overwrite a tensor input file")
    limits = (args.atol, args.rtol, args.relative_l2_max)
    if any(v is not None and (not math.isfinite(v) or v < 0) for v in limits):
        parser.error("Tolerances must be finite and nonnegative")
    if args.mode == "tolerance" and (args.atol is None or args.rtol is None):
        parser.error("Tolerance mode requires both --atol and --rtol")
    if args.mode == "bitwise" and any(v is not None for v in limits):
        parser.error("Bitwise mode does not use numerical tolerances")
    try:
        result = run(args)
        code = 0 if result["passed"] else 1
    except Exception as error:
        result, code = {"passed": False, "error": f"{type(error).__name__}: {error}"}, 2
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
