# Measurement and Correctness

## Define what must be preserved

| Requirement | Suitable evidence | Limitation |
| --- | --- | --- |
| Bitwise identity | Matching shape and dtype, byte comparison over relevant outputs | Applies to tested inputs and environments |
| Numerical agreement | Agreed absolute/relative tolerances and worst-case errors | Does not establish task quality |
| Task-quality preservation | Representative held-out evaluation or closed-loop tasks | Depends on the evaluation distribution |

Choose metrics appropriate to the consumed output and its units. Useful measures include maximum absolute error, RMSE, relative L2 error, and the fraction of elements exceeding tolerance. Cosine similarity alone misses changes in magnitude. Near-zero reference norms require absolute-error analysis; adding an arbitrary epsilon should not determine acceptance. Evaluate valid regions separately from padding, and handle nonfinite values according to the interface semantics.

Elementwise tolerance, `|candidate - reference| <= atol + rtol * |reference|`, differs from an aggregate relative L2 bound. For stochastic inference, control random inputs when possible; otherwise agree on a distributional evaluation rather than assuming samplewise equivalence.

## Preserve a reproducible reference

Record enough information to reconstruct the experiment: code and weight versions, actual execution settings, shapes and layouts, precision policies, random and mutable state, software versions, device configuration, and measurement boundaries. Existing artifact identifiers may be sufficient; exhaustive hashing is not required for every trial.

Repeated baseline runs should cover sources of variability relevant to the change, such as process restart, compilation, cache state, or request order. Fixed seeds do not fix reduction order or algorithm selection. Document reference changes instead of regenerating expected outputs to hide a regression.

Floating-point arithmetic is not associative. Batching, slicing, layout changes, and higher-precision intermediates can change outputs without changing the mathematical expression. See [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html).

## Diagnose discrepancies

Use captured model inputs and real weights for local comparisons, with representative boundary cases. Synthetic inputs extend coverage but do not establish task quality.

Find the earliest divergent intermediate and compare implementations using identical inputs at that boundary. Comparing already-divergent activations confounds the diagnosis. Preserve the original shape and layout when constructing a reference; a smaller submatrix may select a different algorithm. Higher-precision results can help explain errors but constitute a separate reference.

Extend validation to stateful execution, concurrency, or service integration when affected. Instrumentation can alter compilation and timing, so separate diagnostic runs from performance measurements.

## Measure the intended outcome

Distinguish device execution time, model wall-clock time, service latency, client round-trip time, and delivery to the consumer. State whether queueing is included. Synchronize asynchronous work at the appropriate measurement boundary without inserting barriers that destroy production overlap. Timestamps from independent device or host clocks are not directly comparable.

Warm up representative execution paths until stable. Report initialization separately when permitted, but account for compilation or capture triggered by new inputs during normal operation. Use paired, interleaved, or counterbalanced trials; separate processes may be needed when compiled state cannot be switched reliably.

Match mutable state, cache conditions, and shared-device load. Confirm speedups without detailed profiling enabled. Kernel microbenchmarks should reflect realistic operand reuse; repeated access to a small resident matrix can overstate application performance.

Report sample counts, distributions, and variation across trials. Small samples do not support reliable extreme quantiles. Account for correlation when estimating uncertainty. Include failed, rejected, and timed-out requests in service results; completed requests alone can conceal overload.

## Optional tensor comparison utility

`scripts/compare_tensors.py` loads a nonempty tensor or `dict[str, Tensor]` saved with `torch.save`, on CPU with `weights_only=True`. It supports BF16, requires matching fields/shapes/dtypes, and reports strides without requiring equality.

```bash
python scripts/compare_tensors.py ref.pt candidate.pt --mode bitwise
python scripts/compare_tensors.py ref.pt candidate.pt --mode tolerance \
  --atol "$ATOL" --rtol "$RTOL" --relative-l2-max "$REL_L2" \
  --mask-file valid_regions.pt --output comparison.json
```

Tolerances are explicit. Optional masks are same-shape Boolean tensors keyed by output; unspecified fields are checked in full. The utility expects finite outputs and rejects NaN/Inf even outside masks. Use a domain-specific check for legitimate infinities, such as masked attention logits. A zero reference with nonzero error has undefined relative L2 and fails an explicit relative-L2 limit. Discrete values are compared exactly; bitwise mode distinguishes signed zero. Exit codes are 0 for success, 1 for a mismatch, and 2 for invalid input or configuration. Different per-output tolerances require separate comparisons or an adapted harness.
