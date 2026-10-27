# Kernel Optimization

## Optimize the actual workload

Characterize dimensions, layouts, alignment, precision, accumulation, operand reuse, and adjacent operations. Equal operation counts do not imply equal performance; large matrix multiplications and small-batch matrix-vector workloads often have different limits.

Evaluate supported libraries, backends, and algorithm choices before investing in a custom implementation. Algorithm selection must respect workspace, accumulation, and epilogue requirements. Cached choices are environment-dependent; a heuristic rank is not a portable algorithm identifier.

## Tiling, pipelining, and fusion

Choose parameters to address a measured limitation rather than maximizing occupancy or tile size in isolation.

| Choice | Potential benefit | Main trade-off |
| --- | --- | --- |
| Larger output tiles | Greater data reuse | Register/shared-memory demand and boundary padding |
| Deeper pipelines | Better latency hiding | Resource use and fewer resident thread blocks |
| Shorter register live ranges | Less register pressure | Possible extra instructions; hard register caps may cause spills |
| Split-K or Stream-K partitioning | More parallel work for some shapes | Reduction overhead and changed accumulation order |
| Layout changes or weight packing | Better access patterns and alignment | Conversion cost and additional storage |
| Fused projections or epilogues | Fewer launches and intermediate writes | Resource pressure and altered rounding |

For gated MLPs, combining gate/up projections and activation can eliminate large intermediates. Shared activation quantization is valid only if the original inputs, scale rules, and rounding are compatible. Include conversion and packing costs according to their actual reuse. See [CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) for the relevant implementation concepts.

## Preserve the reference arithmetic

Identify where the reference rounds, not just its input and output dtypes. If a gated activation rounds each projection to BF16, rounds SiLU to BF16, and then multiplies, keeping every intermediate in FP32 until the final cast changes the computation. It may be more accurate mathematically while differing from the deployed reference.

The same applies to bias/residual fusion, quantization-scale multiplication, integer-to-floating-point conversion, fused multiply-add contraction, and approximate transcendental functions. Preserve intermediate rounding when required by the accuracy criteria; otherwise validate the proposed change explicitly. Fusion can be slower when register or shared-memory pressure outweighs reduced traffic.

## Evaluate reduced precision separately

When permitted, isolate weight quantization, activation quantization, scale granularity, accumulation, and output rounding before combining them. Use real activations to examine clipping, underflow, channel outliers, and error propagation. Higher-precision treatment of sensitive channels or layers requires calibration and independent validation.

A dequantized high-precision GEMM helps diagnose quantization error but does not measure native low-precision kernel performance. Include online quantization, conversion, padding, and compensation costs. Multiple low-precision products may improve local accuracy while eliminating the speed advantage. Small operator errors can still amplify through the full model or iterative inference.

## Evaluate integration and application benefit

Test supported shapes, layouts, and boundary cases with actual model inputs. Define fallback behavior for unsupported cases and verify that benchmarks exercise the candidate. If model outputs diverge despite local agreement, inspect neighboring reductions, layouts, and compiler decisions.

Microbenchmarks should reflect realistic operand reuse and launch conditions. Report hot-cache measurements separately when they are unrepresentative of the application. Select promising configurations with limited exploration, then confirm them on independent inputs and complete requests. A local improvement without a reproducible application benefit need not be retained.
