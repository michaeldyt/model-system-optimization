# Profiling and Prioritization

## Identify the critical path

Use a representative execution trace to distinguish device work, host overhead, transfers, synchronization, and queueing. Cumulative kernel time is not necessarily the request critical path: nested spans and concurrent branches must not be added indiscriminately.

A timeline profiler is usually sufficient for the first pass. Collect detailed hardware counters only for hotspots where they can distinguish plausible explanations. Profiling may serialize execution, replay kernels, alter cache state, or control clocks. On shared devices, restrict scope and avoid changing device settings without authorization; for Nsight Compute, check the local clock-control default and use `--clock-control none` when appropriate. Interpret results in light of these interventions. See the [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

## Distinguish compute and memory bottlenecks

For important GPU operations, relate actual shapes, precision, operation counts, memory traffic, and elapsed time. Compare achieved arithmetic throughput and bandwidth with sustainable rates under comparable conditions. Distinguish measured traffic from estimates, and use the same execution interval and memory-hierarchy level throughout a comparison.

| Evidence | Likely explanation | Candidate directions |
| --- | --- | --- |
| Arithmetic pipelines approach sustainable throughput | Compute throughput limit | Reduce redundant work or padding; improve algorithm selection |
| DRAM or L2 bandwidth approaches its sustainable limit | Bandwidth limit at that level | Reduce intermediate traffic, improve reuse or access patterns |
| Memory stalls with low achieved bandwidth | Latency, access-pattern, or concurrency limits | Examine coalescing, locality, outstanding requests, and occupancy |
| Device gaps and substantial host dispatch time | Launch or CPU overhead | Compilation, execution capture, fusion, fewer host synchronizations |
| Register spills or excessive synchronization | Resource pressure or execution dependencies | Adjust tiling, live ranges, or work partitioning |
| Growing queues with stable service time | Insufficient service capacity | Reassess load and resource allocation within scope |

Average GPU utilization and occupancy do not establish the bottleneck. Multiple limits may coexist, and neither high nor low bandwidth alone is decisive. Support the initial classification with counters or a controlled experiment; report uncertainty when evidence is incomplete. The [Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/) explains relevant distinctions.

## Use roofline analysis as a consistency check

For dense `A[M,K] @ B[K,N]`, count approximately `F = 2*M*N*K` operations. An idealized data volume is:

`Q_min = bytes(A)*M*K + bytes(B)*K*N + bytes(C)*M*N`

Here `bytes(X)` is bytes per element. Include additional reads, scales, padding, conversions, and intermediates when relevant. This estimate is not actual DRAM traffic: cache reuse may reduce it, while repeated loads may increase it.

For traffic Q at a specified memory level, arithmetic intensity is `F/Q`. A simple performance model is:

`T_est >= max(F/P, Q/BW)`

P and BW must correspond to the arithmetic precision, instruction type, memory level, and operating conditions. Sustainable measured rates provide a practical estimate; hardware peak rates provide optimistic bounds. Dense BF16 operations cannot be assessed against sparse FP8 throughput. A roofline estimate suggests a limiting resource but does not prove it.

## Rank opportunities by application benefit

For a serial region occupying fraction f of the critical path, a local speedup s gives an ideal overall speedup:

`S = 1 / ((1 - f) + f/s)`

Use this estimate to compare recoverable time with implementation cost and numerical risk. Account separately for overlap, queueing, and rare events that dominate tail latency. A faster kernel may move the bottleneck or increase contention, so confirm application-level performance rather than assuming local gains accumulate.
