# System-Level Optimization for AI Models

A reusable skill for coding agents working on inference performance. It combines profiling, controlled experiments, and correctness checks to improve latency, throughput, or memory use while preserving agreed numerical accuracy and task quality.

## Workflow

1. **Define the objective.** Agree on the workload, performance target, and accuracy criteria.
2. **Establish a baseline.** Preserve the original reference and characterize numerical variability and timing noise.
3. **Identify the bottleneck.** Analyze the critical path and distinguish compute, memory, launch, synchronization, and queueing limits.
4. **Evaluate one main change.** Verify that the candidate executes, check correctness, then measure performance under comparable conditions. Investigate discrepancies before adding further changes.
5. **Retain demonstrated improvements.** Record results, reassess the bottleneck, and repeat while further work is worthwhile.

Methods, tolerances, and minimum useful speedups depend on the task. The references provide technical considerations rather than a fixed sequence of optimizations.

## Contents

| File or directory | Purpose |
| --- | --- |
| [SKILL.md](SKILL.md) | Agent instructions and the main optimization workflow |
| [references/](references/) | Measurement, profiling, compilation, CUDA Graphs, kernels, caching, data pipelines, and concurrent serving |
| [scripts/compare_tensors.py](scripts/compare_tensors.py) | Optional tensor comparison with bitwise checks, explicit tolerances, and valid-region masks |
| [agents/openai.yaml](agents/openai.yaml) | Skill display metadata and default prompt |

## Example request

```text
Use $model-system-optimization to reduce inference latency.
Establish a baseline, confirm accuracy requirements, and prioritize
the largest measured bottleneck. Validate each change before continuing.
```

The scope is primarily inference. Training also requires evaluation of gradients, optimizer state, and convergence. Tensor comparisons support numerical regression testing; they do not replace task-quality evaluation.

