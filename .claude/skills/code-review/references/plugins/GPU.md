# Intel GPU Plugin Review

Apply together with [Plugins](../plugins.md) for changes under
`src/plugins/intel_gpu/`, GPU kernels, kernel selection, GPU runtime, or Intel
GPU platform enablement. Report findings in the format of the core skill.
Primary dimensions are functional correctness and performance; architecture and
test quality matter only when they create a concrete risk.

Background docs: `src/plugins/intel_gpu/docs/` (`gpu_kernels.md`,
`gpu_kernel_synchronization.md`, `gpu_memory_formats.md`, `dynamic_shape/`,
`graph_optimization_passes.md`, `gpu_dual_runtime_dispatch.md`,
`memory_allocation_gpu_plugin.md`, `gpu_plugin_unit_test.md`).

## Method

- Review the final PR head, not an earlier commit or summary. Existing bot or
  human comments are leads; re-verify each against the final code.
- Trace the changed path end to end, for example: transformation or op
  conversion -> primitive -> impl registry/selection -> kernel selector params
  -> JIT constants and kernel arguments -> every kernel variant -> fallback or
  rejection -> tests.
- Build a small behavior matrix over the dimensions the PR touches: device
  generation or capability (XMX/systolic, iGPU/dGPU), runtime (OpenCL, Level
  Zero, SYCL, oneDNN, CM), static/dynamic shape, precision, layout, batch,
  prefill/decode or paged/non-paged variants, optimized/fallback/rejected
  paths. Mark a cell covered only with direct evidence (code or test). Name the
  guard that enforces each unsupported cell.
- Do not claim "all variants/sites/platforms are handled" unless each was
  inspected or one shared invariant proves it. State what was and was not
  inspected.
- Do not call a diff correct because it compiles or one config passes.

## Functional checks

**Kernel selection and JIT**
- Selector `Validate`/`GetSupportedKey`/`GetKernelsPriority` conditions match what
  the kernel actually supports (layouts, dtypes, padding, dynamic shape,
  fused ops, tail sizes). A new path must not capture shapes it cannot handle,
  and rejected cases must reach a working fallback.
- Every JIT constant or macro read in the `.cl`/`.cm` source is defined for
  every kernel variant compiled for the configuration. Host-side kernel
  argument order and count match the kernel signature, including dynamic-shape
  `shape_info` and fused-op inputs.
- Dispatch (`gws`/`lws`, subgroup size) covers tails and does not rely on
  alignment the selector does not enforce. Check work-item index math for
  overflow (use 64-bit where sizes can exceed 32 bits) and for out-of-bounds
  reads or writes.

**Layout, memory, shapes**
- Pitches, padding, offsets, and `dynamic_pad` are honored, not assumed dense
  `bfyx`. New reorders or format changes keep `can_be_optimized`, in-place
  and memory-reuse (`memory_pool`, buffer aliasing) assumptions valid.
- Dynamic shapes: output layout and `shape_info` are updated on shape change;
  cached impls, kernel params, and allocated sizes are re-derived; empty,
  scalar, and partially-known shapes work; shape-agnostic kernels do not use
  stale static values.
- Buffer size, lifetime, and USM vs buffer memory type are consistent with the
  kernel and runtime (`ocl`, `ze`, `sycl`).

**Quantized and packed data** (check every load/decode/dequantize site)
- Signed vs unsigned storage; zero-point convention and where it is applied;
  scale/zero-point group indexing and broadcast; nibble order and
  byte/pitch conversion; accumulation precision; tail lanes that could be
  stored out of range.
- Symmetric vs asymmetric, per-channel vs grouped, and dynamic quantization
  paths each keep their numerics.

**Synchronization and concurrency**
- Events, dependencies, barriers, and in-order vs out-of-order queue
  assumptions are preserved; no wait removed or added without reason.
- Shared or cached state (kernels cache, impl cache, compiled model, memory
  pool) mutated by multiple streams or infer requests is protected. Name the
  shared state, the concurrent access, and the missing synchronization; do not
  report a generic "possible race".
- OpenCL, Level Zero, and SYCL runtime paths stay semantically consistent
  when the change touches shared runtime interfaces (stream, event, memory,
  command list).

**Platform and capability**
- Device checks use capability queries (e.g. `supports_immad`, subgroup/
  extension support), not hard-coded generation lists, unless a workaround is
  genuinely generation-specific; workarounds carry a comment and a removal
  condition or tracking issue.
- oneDNN vs OpenCL implementation selection, `ocl_v2`/CM/SYCL registry order,
  and `ImplementationManager::validate` stay consistent with the new path.
- Model cache export/import: new serialized state or properties are saved,
  loaded, and versioned; behavior-affecting properties are included in cache
  keys.
- Errors use `OPENVINO_ASSERT`/`OPENVINO_THROW` with context and are not
  swallowed; build failure of a kernel has a defined fallback.

**Transformations** (also load [Transformations](../transformations.md))
- GPU-specific fusions and decompositions guard on element type, rank, dynamic
  shapes, and constant-ness; they keep tensor names, friendly names, and
  runtime info where downstream passes rely on them.

## Performance checks

Raise a performance finding only with an explained mechanism or provided
measurements. Look for:
- extra kernel launches, reorders, copies, conversions, host-device transfers,
  or blocking waits in the inference path;
- allocation or kernel/impl rebuild per inference, larger or longer-lived
  temporary buffers;
- lost fusion, in-place or optimized-out reorder behavior;
- selector changes that move existing workloads to a generic kernel;
- reduced parallelism, occupancy, vectorization, or memory coalescing
  (check `gws`/`lws`, subgroup size, SLM use);
- higher compile time, first-inference latency, or cache invalidation;
- a win on one GPU generation at the cost of another.

Keep measurement scopes separate: kernel time, end-to-end latency/throughput,
compile/load time, first inference, cache import, memory. A kernel speedup is
not an end-to-end speedup. For changed existing paths, expect a comparison with
the base commit on the affected workload and generation; for newly enabled
paths, absolute numbers are enough. If the mechanism is plausible but impact is
unknown, report a validation gap, not a regression.

## Tests

- Prefer GPU unit tests in `tests/unit/` (primitive, kernel, transformation,
  module tests) with a reference comparison; functional tests in
  `tests/functional/` for model-level behavior. A bug fix needs a test that
  fails without the fix.
- Tests must cover the new variants: dynamic and static shape, tail and
  boundary sizes, each precision/layout the selector admits, fallback or
  rejection, and both runtimes where the change is runtime-visible.
- Tests requiring XMX, a specific generation, Level Zero, or oneDNN must skip
  or be gated on devices without it, not fail.
- A missing test is a validation gap, not a defect, unless the code itself
  shows the failure. Do not request tests that duplicate existing coverage.

## Severity

- **Critical/High**: demonstrated reachable path on a supported configuration:
  OOB or corruption, wrong output, crash, missing synchronization, broken
  fallback, overflow in indexing/allocation, or a major hot-path regression
  with a shown mechanism.
- **Medium**: credible edge-case risk, unverified invariant, performance
  concern without measurements, or important coverage gap.
- **Low**: clarity or minor maintainability with no current impact.
- **Question**: intent or platform scope cannot be determined from the PR;
  explain why the answer matters. Do not use a question to disguise an
  unsupported allegation.

Do not downgrade evidence by inference from similarly structured code, and do
not report pre-existing issues unless the PR makes them reachable or worse.
Escalate architectural intent and product trade-offs to GPU owners. Do not
recommend approval or merge.
