# Compilation and CUDA Graphs

## Distinguish the mechanisms

| Mechanism | Meaning | Useful evidence |
| --- | --- | --- |
| Graph break | Tracing is interrupted; execution is partitioned around an unsupported operation | Graph-break logs and the execution timeline |
| Recompilation | Existing guards fail and another specialization is generated | Recompilation logs and changing input or state properties |
| CUDA Graph capture | GPU operations and their dependencies are recorded for replay | Capture/replay activity, memory use, and the device timeline |

`fullgraph=True` rejects graph breaks within the traced region. It does not imply a single kernel, one specialization for every shape, or CUDA Graph replay. Relevant diagnostics include `TORCH_LOGS="graph_breaks,recompiles"`; enable more detail only as needed. Consult the version-appropriate [graph-break](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html) and [recompilation](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.recompilation.html) documentation.

## Choose compilation boundaries

Target repeated tensor computation with substantial dispatch or fusion opportunities. Keep decoding, logging, I/O, and request management outside when practical. Compare narrower regions with larger stable regions according to compilation cost, memory use, and runtime benefit; neither whole-model nor per-operator compilation is universally best.

Common causes of avoidable overhead include:

- **Tensor-to-Python conversion and data-dependent control flow.** Keep values on the device when possible. `torch.cond` may support required branching under its constraints; `torch.where` is not a short-circuit replacement because its arguments are generally evaluated first.
- **Frequently changing guards.** Inspect shapes, strides, dtypes, devices, Python constants, and container structure. Represent a changing scalar as a tensor only when its role permits it. Reuse compiled callables.
- **Variable input shapes.** Choose specialization, bucketing, or dynamic shapes according to the workload. Padding requires correct masks and output/state handling. Dynamic shapes may reduce recompilation without improving kernel performance.
- **Intentional eager regions.** Place unsupported work outside compiled regions or allow explicit partitioning. Disabling compilation inside an enclosing `fullgraph=True` region does not satisfy that requirement. Suppressing errors does not demonstrate successful compilation.

Use a small reproduction when the cause is unclear; avoid accumulating global compiler flags without evidence. See [PyTorch compiler troubleshooting](https://github.com/pytorch/pytorch/blob/main/docs/source/user_guide/torch_compiler/torch.compiler_troubleshooting.md).

## Preserve numerical behavior and integration

Compilation, layout changes, and opaque custom operators can alter fusion, reductions, and GEMM algorithm selection elsewhere in the model. Unchanged input dtype or an identical isolated kernel does not establish numerical agreement for the complete model. When necessary, inspect intermediate casts, fused multiply-add behavior, TF32, and reduction precision.

Custom operators need correct schemas, mutation/aliasing declarations, and fake/meta behavior. Registration checks such as `opcheck` do not replace numerical tests. Autograd support requires separate validation when relevant. Prefer public interfaces; private compiler hooks or fixed algorithm configurations need explicit environment constraints and a fallback. See [PyTorch custom operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html).

## Manage CUDA Graph lifetimes

CUDA Graphs reduce repeated launch overhead; they do not reduce matrix-multiplication work. Evaluate them when execution structure is sufficiently stable and dispatch matters. Include capture, memory pools, workspace, and input/output copying in the cost assessment.

Conventional replay requires stable addresses and compatible execution structure. Coordinate capture with other CUDA work and avoid unsupported host synchronization. Buffer ownership must prevent concurrent overwrite: graph outputs retained across replays need independent storage or a lifetime guarantee. Neither `detach()` nor an already-contiguous `contiguous()` call creates a copy.

Select concurrency and stream management appropriate to the runtime rather than assuming a single-threaded design. Verify actual replay and recapture behavior, especially with changing shapes, live outputs, random-number generation, and framework-managed graph pools. See [CUDA Graph semantics](https://docs.pytorch.org/docs/stable/notes/cuda#cuda-graphs) and [CUDA Graph Trees](https://github.com/pytorch/pytorch/blob/main/docs/source/user_guide/torch_compiler/torch.compiler_cudagraph_trees.md).
