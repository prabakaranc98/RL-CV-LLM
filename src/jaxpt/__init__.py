"""jaxpt: our JAX post-training stack (Tunix is used for RL orchestration only).

Process-wide defaults, set before JAX creates its GPU client, so every script that imports jaxpt behaves the same:
  - no preallocation, so several processes or phases can share a GPU and memory_stats() means something;
  - a 95% allocator cap: JAX's default of 75% capped an 80 GB GPU at 59.4 GiB and caused X2's out-of-memory errors.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
