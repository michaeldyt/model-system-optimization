---
name: model-system-optimization
description: Profile and optimize AI inference systems while preserving agreed numerical accuracy and task quality. Use for GPU performance analysis, compilation, CUDA Graphs, kernels, caching, data pipelines, and concurrent serving.
---

# System-Level Optimization for AI Models

Prioritize changes that reduce the measured bottleneck, evaluate one main hypothesis at a time, and validate correctness before accepting a performance improvement. This skill focuses on inference; training additionally requires gradient, optimizer-state, and convergence evaluation.

## Establish the baseline and acceptance criteria

Inspect the active execution path, existing tests, and relevant history. Distinguish deployed behavior from optional experiments and outdated documentation. Reuse the project's evaluation tools where appropriate.

Confirm any requirements not already agreed with the user:

- **Performance:** target metric, representative inputs and load, resource budget, importance of cold starts, and the minimum useful improvement.
- **Accuracy:** bitwise identity, explicit numerical tolerances, or an acceptable change in task quality; specify the outputs, valid regions, and evaluation data. Clarify whether changes to precision or model semantics are allowed.

Propose reasonable criteria when the user needs guidance. Continue baseline measurements and profiling while clarification is pending, without assuming that an unspecified accuracy loss is acceptable.

Preserve the original reference and record the code, weights, execution settings, inputs, random state, and software/hardware environment needed to reproduce it. Use repeated baseline runs (A/A comparisons) to characterize numerical variability and timing noise. Keep the original accuracy reference distinct from the latest accepted performance baseline; explain any change of reference.

## Profile and prioritize

Start with representative requests and identify the critical path. Separate computation, data movement, host overhead, synchronization, and queueing without double-counting overlapping work.

For the dominant GPU operations, assess whether performance is **compute-bound or memory-bound** using arithmetic intensity, achieved throughput, memory bandwidth, and relevant stall metrics. Distinguish bandwidth limits from memory latency; consider launch overhead, dependencies, and resource contention when neither explanation fits. Report supporting evidence and uncertainty rather than inferring the bottleneck from average GPU utilization.

Prioritize the largest recoverable contribution to the target metric. Use an approximate end-to-end benefit and implementation cost to select experiments; adapt the threshold to the user's objective and measurement noise. Increase profiling detail only where it can change the decision.

## Iterate with controlled experiments

For each iteration, state the hypothesis, expected benefit, numerical or state-related risks, and evaluation method.

- Make one major, attributable change. Necessary supporting edits may belong to the same experiment; explain coupled changes and use ablations where practical.
- Verify that the candidate path actually executes. Testing only the fallback does not validate a new kernel, batch path, or cache policy.
- Check correctness at the relevant levels: operator, complete model, stateful sequence, or service. Validate the complete model whenever arithmetic changes, even if an isolated operator matches.
- If correctness fails, stop adding changes and locate the earliest divergence under identical inputs and state. Distinguish pre-existing variability from effects introduced by the candidate.
- After correctness passes, compare performance under matched conditions, preferably with paired or counterbalanced trials. Include relevant memory use, compilation overhead, and tail latency.
- Accept a change only when it meets the agreed criteria and the benefit is reproducible. Recheck combinations of accepted changes and retain brief records of unsuccessful experiments.

Choose implementation techniques and test scope according to the evidence. Routine reversible iterations do not require renewed approval; changes to agreed accuracy, semantics, or resource limits require a new agreement.

## References

Read only the material relevant to the current bottleneck. These references describe options and constraints, not a mandatory optimization sequence.

| Topic | Reference |
| --- | --- |
| Accuracy criteria, baseline variability, benchmarking | [Measurement and correctness](references/measurement-and-correctness.md) |
| Compute and memory bottlenecks, roofline analysis, priorities | [Profiling and prioritization](references/profiling-and-priorities.md) |
| Graph breaks, recompilation, execution capture | [Compilation and CUDA Graphs](references/compile-and-graphs.md) |
| GEMM algorithms, tiling, fusion, rounding | [Kernel optimization](references/kernel-optimization.md) |
| Reuse, preprocessing, asynchronous transfers | [Caching and data pipelines](references/cache-and-pipeline.md) |
| Batching, capacity, deadlines, concurrent workloads | [Serving and batching](references/serving-and-batching.md) |

The optional `scripts/compare_tensors.py` utility compares tensor snapshots using bitwise checks or explicit tolerances. It supports numerical regression testing, not task-quality or service-state evaluation.

## Conclude the investigation

Stop when the target is met, remaining benefits are below the agreed value or measurement resolution, or further work requires missing evidence or revised requirements. Report retained changes, reproducible configurations, correctness coverage, performance distributions, and unresolved limitations. Distinguish measurements from estimates. Check version-specific APIs and hardware assumptions before applying any technique from the references.
