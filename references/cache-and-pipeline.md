# Caching and Data Pipelines

## Establish when reuse is valid

Cache a result only when its relevant dependencies remain unchanged. Account for inputs, conditioning, timestep, masks, positions, parameter or adapter versions, precision, and execution mode as applicable. A cache key or a well-defined lifetime can enforce these invariants; not every variable needs to appear in every key.

Potential candidates include invariant conditioning, embeddings, masks, static weight transformations, and genuinely causal prefixes. Identical prompts do not guarantee reusable text key/value tensors in bidirectional multimodal attention. A causal attention mask does not establish causality for preprocessing, convolution, or normalization.

Cache entries need valid storage ownership, capacity limits, and invalidation rules. Views into reusable scratch space or graph outputs may be overwritten. `detach()` and `contiguous()` do not necessarily provide independent storage. Keep request-specific random, sampling, and recurrent state separate from shared immutable data.

## Treat approximate reuse as a quality trade-off

Similarity thresholds are heuristics, not bounds on future error. Shadow evaluation can compare recomputed and reused results while measuring error and coverage; its extra work is not a deployment speedup. Select thresholds on calibration data and evaluate the actual policy on independent data.

Forced reuse can expose potential savings and failure modes, but it does not bound errors under other reuse patterns. Zero cache hits validate only the fallback path. Preserve state updates and coordinate reuse decisions across cooperating branches when required.

## Reduce redundant preparation and transfer

Remove unused fields, repeated serialization, unnecessary copies, or repeated preprocessing when consumer semantics permit. Incremental transfer requires unambiguous state/version handling across retries and resets.

Preprocessing changes must preserve the required values and interpretation, including interpolation, normalization, masks, and temporal alignment. Layout changes can select different downstream kernels even when tensor values match. Measure the full producer-to-consumer path, including conversions.

Move preparation off the critical path when it can overlap useful computation. Choose threads, processes, or vectorization according to the workload; control queue capacity, memory use, and CPU oversubscription. Avoid making the compute stage wait unnecessarily for preparation. Cancellation, errors, and request ordering remain part of the design. Random-number assignment should not inadvertently depend on worker completion order.

## Evaluate transfer and compute overlap

Asynchronous copies require appropriate memory, stream dependencies, and storage lifetimes. `non_blocking=True` alone does not establish overlap. Pinned-memory allocation and copying can offset transfer savings, so include preparation costs. See [PyTorch pinned memory and nonblocking transfers](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).

The host source must remain valid and unmodified until a transfer completes; the device destination must not be overwritten until computation finishes reading it. Cross-stream use also requires correct allocator lifetime management. CPU consumers must wait for asynchronous device-to-host results to become valid.

Concurrent GPU stages can compete for compute and bandwidth. Use the timeline and end-to-end measurements to determine whether overlap reduces latency or merely slows both stages.
