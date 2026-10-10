"""Timing for jitted functions: compile once, then time steady-state calls.

Used by every block's performance number (forward, sampler, train step), so they are measured the same way.
FLOPs come from the caller (an analytic count): XLA's cost_analysis() on GPU counts a lax.scan body once
and did not scale with batch size in Block 1b, so it is not used for MFU.
"""
import statistics
import time

import jax

# Dense (no sparsity) bf16 tensor-core peak, FLOP/s, from the NVIDIA spec sheets.
BF16_PEAK = {"NVIDIA A40": 149.7e12, "NVIDIA A100-SXM4-80GB": 312e12, "NVIDIA H100 80GB HBM3": 989e12}


def measure(fn, *args, flops=None, iters=10):
    """fn: a jax.jit-wrapped function; flops: the work of one call, if known.
    Returns compile time, median step time, MFU and the executable's device memory
    (arguments + outputs + temporaries)."""
    t0 = time.perf_counter()
    compiled = fn.lower(*args).compile()
    compile_s = time.perf_counter() - t0

    jax.block_until_ready(compiled(*args))                    # warm-up
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        times.append(time.perf_counter() - t0)

    mem = compiled.memory_analysis()
    step_s = statistics.median(times)
    peak = BF16_PEAK.get(jax.devices()[0].device_kind)
    return {"compile_s": compile_s, "step_s": step_s, "flops": flops,
            "tflops_per_s": flops / step_s / 1e12 if flops else None,
            "mfu": flops / step_s / peak if flops and peak else None,
            "memory_gb": (mem.argument_size_in_bytes + mem.output_size_in_bytes + mem.temp_size_in_bytes) / 1e9}
