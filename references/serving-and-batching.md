# Serving and Batching

Use this reference when service capacity, communication, batching, or queueing is within scope. A request for kernel optimization alone does not imply a scheduling redesign.

## Relate offered load to capacity

For a serial resource, approximate utilization as `rho = sum(lambda_i * E[S_i])`, where arrival rates and service demands use consistent units. Load above capacity cannot be sustained without growing queues, rejection, or throttling; utilization below one still does not guarantee latency deadlines. Analyze distinct resources separately in a pipeline.

GPUs cooperating on one request are not independent workers. Device imbalance, topology, communication, and synchronization determine effective capacity. Scheduling can reduce avoidable waiting but cannot remove required computation. Earlier prefetch may improve deadline margins while increasing load or changing input freshness.

## Transfer LLM techniques selectively

Small-batch autoregressive decoding and large dense multimodal forwards may have different arithmetic intensity and scaling. Test throughput `b/T(b)`, per-request latency, and memory at relevant batch sizes before assuming batching is beneficial.

Continuous batching requires independent request state and valid admission boundaries. Key/value or prefix reuse requires invariant dependencies. Chunked execution requires a valid decomposition that preserves state and attention semantics. Paged storage is most useful when variable-length allocation or fragmentation is material. Choose these methods based on the workload rather than analogy alone.

## Preserve request independence and numerical behavior

Each request retains its inputs, masks, positions, conditioning, random state, timestep, and sampling or recurrent history. Admit and retire requests at supported boundaries; preserve output routing and completion semantics. Group compatible requests or provide explicit fallback behavior.

Batching can change matrix shapes, reduction order, and selected algorithms. If strict agreement with independent execution is required, consider retaining sensitive operations at their original shapes or constraining algorithm selection. Measure the cost and avoid redefining the reference to claim equivalence.

Validate actual batch execution, mixed inputs, and relevant cancellation or reset behavior. Changing another request's content or arrival order should not corrupt a request's state. Tests that always fall back to independent execution do not establish batching correctness or throughput.

## Choose scheduling and evaluation methods to fit the service

Bound preparation and execution queues and define backpressure. Scheduling should account for readiness, request ordering, fairness, and any deadline objective. Calibrate service-time estimates for relevant shapes and load; distinguish capacity estimates from deadline guarantees.

Independent host monotonic clocks are not directly comparable. Relative budgets need accounting for local delay and transport uncertainty, or use a clock protocol with a known error bound. Retries should preserve the intended deadline. Thread scheduling does not provide arbitrary preemption of a running GPU kernel.

Use open-loop arrivals to expose overload and burst behavior, and realistic closed-loop clients to verify application behavior. One-time staggering may drift. Distinguish regular consumer activity from delivery of fresh results, and include control or housekeeping operations when they delay delivery.

Report aggregate and per-client latency, throughput, deadline misses, rejection, and queue behavior as relevant. Compare configurations under equivalent resources and input/output timing semantics; an aggregate mean can conceal starvation or blocking for individual clients.
