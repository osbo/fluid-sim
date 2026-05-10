"""
BenchmarkBaseline.py — Two-pass benchmark of the baseline LeafOnly model
(scale=8192, leaf=128, d_model=128, num_layers=3, use_highways=True, cos-Hutchinson loss).

Pass 1 (timing):   fine-grained per-component CUDA-event timings for all 20 test frames.
Pass 2 (metrics):  torch.profiler op/kernel tables + NVML sampling + (optional) Nsight Compute
                   (ncu) hardware-counter pass for tensor-core util, memory traffic, occupancy, etc.

Outputs (in --output-dir, default: results/baseline_benchmark/):
  timing_components.csv     wide rows=frames cols=components, μs (one fwd, mean over --repeat)
  timing_long.csv           long form: frame_idx, component, mean_ms, std_ms, n
  profiler_ops.csv          aggregated torch.profiler op table across the run
  profiler_kernels.csv      torch.profiler kernel-level table (CUDA only)
  profiler_memory.csv       per-frame allocator stats + peak memory
  gpu_nvml.csv              periodic NVML sample stream (sm%, mem%, MiB, power, temp, clocks)
  gpu_nvml_per_frame.csv    per-frame aggregates of the NVML stream
  ncu_kernels.csv           Nsight Compute kernel breakdown (if --ncu-frames set and ncu is available)
  ncu_section_<N>.txt       human-readable ncu section dump per frame

Usage:
  python3 BenchmarkBaseline.py                              # default: 20 frames, ncu on frame 0
  python3 BenchmarkBaseline.py --weights /path/to/v2_8192_d128_L3_hw.bytes
  python3 BenchmarkBaseline.py --frames 0 5 10              # subset
  python3 BenchmarkBaseline.py --skip-ncu                   # skip ncu pass
  python3 BenchmarkBaseline.py --no-profiler                # skip torch.profiler pass

Internal:
  python3 BenchmarkBaseline.py --ncu-fwd-only --frame N     # one-shot forward (called from under `ncu`)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from leafonly import (  # noqa: E402
    LeafOnlyNet,
    FluidGraphDataset,
    build_leaf_block_connectivity,
    default_attention_layout,
    load_leaf_only_weights,
    warmup_hmatrix_prolong_gpu,
)
from leafonly.checkpoint import (  # noqa: E402
    apply_leaf_only_runtime_from_checkpoint,
    leaf_only_arch_from_checkpoint,
)
import leafonly.config as lo_config  # noqa: E402
from leafonly.config import problem_padded_num_nodes  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# CUDA event timer collector
# ──────────────────────────────────────────────────────────────────────────────

class CudaTimerCollector:
    """Collects pairs of CUDA events. Resolves to elapsed ms after sync()."""
    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.events: dict[str, list[tuple[Any, Any]]] = {}

    def record(self, name: str):
        return _CudaCtx(self, name)

    def reset(self) -> None:
        self.events = {}

    def collect_ms(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        if self.is_cuda:
            torch.cuda.synchronize()
            for name, pairs in self.events.items():
                out[name] = [float(s.elapsed_time(e)) for (s, e) in pairs]
        else:
            for name, pairs in self.events.items():
                out[name] = [float((e - s) * 1000.0) for (s, e) in pairs]
        return out


class _CudaCtx:
    __slots__ = ("col", "name", "_s", "_e")
    def __init__(self, col: CudaTimerCollector, name: str):
        self.col = col
        self.name = name
    def __enter__(self):
        if self.col.is_cuda:
            self._s = torch.cuda.Event(enable_timing=True)
            self._e = torch.cuda.Event(enable_timing=True)
            self._s.record()
        else:
            self._s = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        if self.col.is_cuda:
            self._e.record()
            self.col.events.setdefault(self.name, []).append((self._s, self._e))
        else:
            self._e = time.perf_counter()
            self.col.events.setdefault(self.name, []).append((self._s, self._e))


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model_from_checkpoint(weights_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    apply_leaf_only_runtime_from_checkpoint(weights_path)
    arch = leaf_only_arch_from_checkpoint(weights_path)
    if arch is None:
        raise SystemExit(f"Could not read architecture from {weights_path}")
    use_hw = bool(int(arch.get("highway_ffn_mlp", 0)))
    ffn_cw = int(arch.get("ffn_concat_width", 3 if use_hw else 1))
    model = LeafOnlyNet(
        input_dim=arch["input_dim"],
        d_model=arch["d_model"],
        leaf_size=arch["leaf_size"],
        num_layers=arch["num_layers"],
        num_heads=arch["num_heads"],
        use_gcn=bool(arch["use_gcn"]),
        attention_layout=default_attention_layout(arch["leaf_size"]),
        off_diag_dense_attention=True,
        diag_dense_attention=True,
        use_highways=use_hw,
        ffn_concat_width=ffn_cw if use_hw else None,
    ).to(device)
    load_leaf_only_weights(model, weights_path)
    model.eval()
    return model, arch


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame input preparation
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_frame_inputs(batch: dict, device: torch.device, leaf_size: int) -> dict:
    x = batch["x"].unsqueeze(0).to(device)
    num_nodes_real = int(batch["num_nodes"])
    n_requested = problem_padded_num_nodes(num_nodes_real)
    n_pad = int(n_requested)
    ei, ev = batch["edge_index"], batch["edge_values"]
    em = (ei[0] < n_requested) & (ei[1] < n_requested)

    x_leaf = x[:, :n_requested, :].clone()
    if n_pad > n_requested:
        x_leaf = F.pad(x_leaf, (0, 0, 0, n_pad - n_requested), value=0.0)
    edge_index_leaf = ei[:, em].to(device)
    edge_values_leaf = ev[em].to(device)
    global_feat = batch.get("global_features")
    if global_feat is not None:
        global_feat = global_feat.to(device)
        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0)
    positions_leaf = x_leaf[0, :, :3]
    pre_connect = build_leaf_block_connectivity(
        edge_index_leaf,
        edge_values_leaf,
        positions_leaf,
        leaf_size,
        device,
        x_leaf.dtype,
        off_diag_dense_attention=True,
        diag_dense_attention=True,
    )
    pre_connect = tuple(
        t.contiguous() if isinstance(t, torch.Tensor) else t for t in pre_connect
    )
    return {
        "x_leaf": x_leaf,
        "edge_index_leaf": edge_index_leaf,
        "edge_values_leaf": edge_values_leaf,
        "global_feat": global_feat,
        "pre_connect": pre_connect,
        "n_requested": n_requested,
        "n_pad": n_pad,
        "num_nodes_real": num_nodes_real,
        "frame_path": batch.get("frame_path", ""),
    }


def _fwd(model, inputs: dict) -> torch.Tensor:
    return model(
        inputs["x_leaf"],
        edge_index=inputs["edge_index_leaf"],
        edge_values=inputs["edge_values_leaf"],
        global_features=inputs["global_feat"],
        precomputed_leaf_connectivity=inputs["pre_connect"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Component instrumentation (pass 1)
# ──────────────────────────────────────────────────────────────────────────────

# Helper-method names on LeafOnlyNet that we wrap by instance-attribute shadowing.
_LEAFONLY_HELPERS = [
    "_pool_full_leaf_to_diag_tokens",
    "_diag_tokens_to_full_leaf",
    "_build_token_highways",
    "_apply_transformer_stacks",
    "_get_leaf_blocks",
    "_get_jacobi_scale",
    "_build_off_strip",
    "_build_diag_mlp_hw",
    "_build_off_mlp_hw",
    "_mlp_global_broadcast",
]

# nn.Modules whose forward we time via forward hooks.
def _module_timer_paths(model) -> list[tuple[str, torch.nn.Module]]:
    """Return (label, module) pairs for top-level timed modules."""
    out: list[tuple[str, torch.nn.Module]] = [
        ("embed", model.embed),
        ("enc_input_proj", model.enc_input_proj),
        ("leaf_head_diag_uut", model.leaf_head),
        ("offdiag_head_U", model.off_diag_head_U),
        ("offdiag_head_V", model.off_diag_head_V),
        ("node_u", model.node_u),
        ("node_v", model.node_v),
        ("jacobi_gate", model.jacobi_gate),
    ]
    return out


def install_instrumentation(model, collector: CudaTimerCollector) -> list[Callable[[], None]]:
    """Wrap helper methods and register forward hooks. Returns un-install callbacks."""
    undos: list[Callable[[], None]] = []

    # 1. Helper instance methods on LeafOnlyNet.
    for attr in _LEAFONLY_HELPERS:
        if not hasattr(model, attr):
            continue
        orig = getattr(model, attr)

        def make_wrap(_orig, _name):
            def wrapped(*args, **kwargs):
                with collector.record(_name):
                    return _orig(*args, **kwargs)
            return wrapped

        wrapped = make_wrap(orig, attr)
        setattr(model, attr, wrapped)
        undos.append(lambda a=attr: object.__getattribute__(model, "__dict__").pop(a, None))

    # 2. TransformerBlock.forward_attn / forward_mlp per layer (both stacks).
    for stack_name, blocks in (("diag", model.blocks), ("off", model.off_diag_blocks)):
        for i, blk in enumerate(blocks):
            for meth in ("forward_attn", "forward_mlp"):
                if not hasattr(blk, meth):
                    continue
                orig = getattr(blk, meth)
                label = f"{stack_name}_L{i}_{meth.replace('forward_', '')}"

                def make_wrap(_orig, _label):
                    def wrapped(*args, **kwargs):
                        with collector.record(_label):
                            return _orig(*args, **kwargs)
                    return wrapped

                setattr(blk, meth, make_wrap(orig, label))
                undos.append(lambda obj=blk, m=meth: object.__getattribute__(obj, "__dict__").pop(m, None))

    # 3. nn.Module forward hooks. Use a stack per module id to tolerate reentrant calls.
    pending_starts: dict[int, list] = {}

    def make_pre(label):
        def pre(mod, inputs):
            if collector.is_cuda:
                s = torch.cuda.Event(enable_timing=True)
                s.record()
            else:
                s = time.perf_counter()
            pending_starts.setdefault(id(mod), []).append((label, s))
        return pre

    def post(mod, inputs, output):
        stk = pending_starts.get(id(mod))
        if not stk:
            return
        label, s = stk.pop()
        if collector.is_cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            collector.events.setdefault(label, []).append((s, e))
        else:
            e = time.perf_counter()
            collector.events.setdefault(label, []).append((s, e))

    for label, mod in _module_timer_paths(model):
        h1 = mod.register_forward_pre_hook(make_pre(label))
        h2 = mod.register_forward_hook(post)
        undos.append(h1.remove)
        undos.append(h2.remove)

    return undos


# ──────────────────────────────────────────────────────────────────────────────
# Pass 1: fine-grained component timing
# ──────────────────────────────────────────────────────────────────────────────

def pass1_component_timing(
    model,
    frame_inputs: list[dict],
    device: torch.device,
    warmup: int,
    repeat: int,
    out_dir: Path,
) -> Path:
    collector = CudaTimerCollector(device)
    install_instrumentation(model, collector)

    # Warmup on first frame so caches/cuDNN/Inductor (if any) settle before measurement.
    if frame_inputs:
        with torch.inference_mode():
            for _ in range(max(1, warmup)):
                _fwd(model, frame_inputs[0])
            if device.type == "cuda":
                torch.cuda.synchronize()

    rows_long: list[dict[str, Any]] = []
    component_keys: list[str] = []
    rows_wide: list[dict[str, Any]] = []

    with torch.inference_mode():
        for idx, inputs in enumerate(frame_inputs):
            # Quick warm on this frame's shape (avoid first-time cache misses biasing).
            _fwd(model, inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            collector.reset()
            # Whole-forward timer + per-component.
            for _ in range(repeat):
                with collector.record("forward_total"):
                    _fwd(model, inputs)
            times = collector.collect_ms()  # name -> list of ms (length = repeat)
            row = {
                "frame_idx": idx,
                "frame_path": inputs.get("frame_path", ""),
                "n_requested": inputs["n_requested"],
                "n_pad": inputs["n_pad"],
                "num_nodes_real": inputs["num_nodes_real"],
                "repeat": repeat,
            }
            for name, ms in times.items():
                if name not in component_keys:
                    component_keys.append(name)
                mean = float(statistics.fmean(ms)) if ms else float("nan")
                std = float(statistics.pstdev(ms)) if len(ms) > 1 else 0.0
                row[f"{name}_mean_ms"] = mean
                row[f"{name}_std_ms"] = std
                rows_long.append({
                    "frame_idx": idx,
                    "component": name,
                    "mean_ms": mean,
                    "std_ms": std,
                    "n": len(ms),
                })
            rows_wide.append(row)
            print(
                f"  [pass1] frame {idx:2d}/{len(frame_inputs)}  "
                f"fwd={row.get('forward_total_mean_ms', float('nan')):7.3f} ms",
                flush=True,
            )

    # Write wide CSV.
    wide_path = out_dir / "timing_components.csv"
    fieldnames = ["frame_idx", "frame_path", "n_requested", "n_pad", "num_nodes_real", "repeat"]
    for k in component_keys:
        fieldnames.append(f"{k}_mean_ms")
        fieldnames.append(f"{k}_std_ms")
    with wide_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_wide:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Write long CSV.
    long_path = out_dir / "timing_long.csv"
    with long_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_idx", "component", "mean_ms", "std_ms", "n"])
        w.writeheader()
        for r in rows_long:
            w.writerow(r)

    return wide_path


# ──────────────────────────────────────────────────────────────────────────────
# Pass 2a: torch.profiler op + kernel tables
# ──────────────────────────────────────────────────────────────────────────────

def pass2_torch_profiler(
    model,
    frame_inputs: list[dict],
    device: torch.device,
    repeat: int,
    out_dir: Path,
    nvml_sampler: Optional["NvmlSampler"],
) -> None:
    try:
        from torch.profiler import ProfilerActivity, profile, record_function
    except Exception as e:
        print(f"[pass2] torch.profiler unavailable: {e}", file=sys.stderr)
        return

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    mem_rows: list[dict[str, Any]] = []
    per_frame_nvml: list[dict[str, Any]] = []

    # One profiler over all frames (cheaper than per-frame setup); each frame in its own record_function.
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        # Brief warmup
        if frame_inputs:
            for _ in range(2):
                _fwd(model, frame_inputs[0])
            if device.type == "cuda":
                torch.cuda.synchronize()

        with profile(
            activities=activities,
            record_shapes=True,
            with_stack=False,
            with_flops=True,
            profile_memory=True,
        ) as prof:
            for idx, inputs in enumerate(frame_inputs):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                t_start_ns = time.perf_counter_ns()
                if nvml_sampler is not None:
                    nvml_sampler.mark(f"frame_{idx:02d}_start")
                with record_function(f"frame_{idx:02d}"):
                    for _ in range(repeat):
                        with record_function(f"frame_{idx:02d}_fwd"):
                            _fwd(model, inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if nvml_sampler is not None:
                    nvml_sampler.mark(f"frame_{idx:02d}_end")
                t_end_ns = time.perf_counter_ns()
                mem = {}
                if device.type == "cuda":
                    mem = {
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                        "current_allocated_bytes": int(torch.cuda.memory_allocated()),
                        "current_reserved_bytes": int(torch.cuda.memory_reserved()),
                    }
                mem_rows.append({
                    "frame_idx": idx,
                    "wall_ms": (t_end_ns - t_start_ns) / 1e6,
                    "repeat": repeat,
                    **mem,
                })
                print(f"  [pass2] frame {idx:2d}  wall={(t_end_ns - t_start_ns)/1e6:.2f} ms", flush=True)

    # Aggregate op table.
    ops_path = out_dir / "profiler_ops.csv"
    kernels_path = out_dir / "profiler_kernels.csv"
    try:
        ka = prof.key_averages(group_by_input_shape=False)
        with ops_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "key", "count",
                "cpu_time_us_total", "cpu_time_us_avg",
                "cuda_time_us_total", "cuda_time_us_avg",
                "self_cpu_time_us_total", "self_cuda_time_us_total",
                "flops",
                "cpu_memory_bytes", "cuda_memory_bytes",
            ])
            for evt in ka:
                w.writerow([
                    evt.key, int(evt.count),
                    float(evt.cpu_time_total), float(evt.cpu_time),
                    float(getattr(evt, "device_time_total", 0.0) or 0.0),
                    float(getattr(evt, "device_time", 0.0) or 0.0),
                    float(evt.self_cpu_time_total),
                    float(getattr(evt, "self_device_time_total", 0.0) or 0.0),
                    int(getattr(evt, "flops", 0) or 0),
                    int(getattr(evt, "cpu_memory_usage", 0) or 0),
                    int(getattr(evt, "device_memory_usage", 0) or 0),
                ])
    except Exception as e:
        print(f"[pass2] op table dump failed: {e}", file=sys.stderr)

    # Kernel-level table (CUDA).
    if device.type == "cuda":
        try:
            ka = prof.key_averages(group_by_input_shape=False)
            with kernels_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "kernel_or_op", "count",
                    "self_device_time_us", "device_time_us_total", "device_time_us_avg",
                    "input_shapes",
                ])
                for evt in ka:
                    sd = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
                    if sd <= 0:
                        continue
                    w.writerow([
                        evt.key, int(evt.count),
                        sd,
                        float(getattr(evt, "device_time_total", 0.0) or 0.0),
                        float(getattr(evt, "device_time", 0.0) or 0.0),
                        json.dumps(list(getattr(evt, "input_shapes", []) or [])),
                    ])
        except Exception as e:
            print(f"[pass2] kernel table dump failed: {e}", file=sys.stderr)

    # Memory CSV.
    with (out_dir / "profiler_memory.csv").open("w", newline="") as f:
        keys = ["frame_idx", "wall_ms", "repeat",
                "peak_allocated_bytes", "peak_reserved_bytes",
                "current_allocated_bytes", "current_reserved_bytes"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in mem_rows:
            w.writerow({k: r.get(k, "") for k in keys})

    # Chrome trace (full).
    try:
        prof.export_chrome_trace(str(out_dir / "torch_profiler_trace.json"))
    except Exception as e:
        print(f"[pass2] chrome trace export failed: {e}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# NVML sampler (pass 2b)
# ──────────────────────────────────────────────────────────────────────────────

class NvmlSampler:
    def __init__(self, period_ms: float = 10.0, gpu_index: int = 0):
        try:
            import pynvml  # type: ignore
            self._pynvml = pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self._available = True
        except Exception as e:
            print(f"[nvml] pynvml unavailable, skipping NVML sampling: {e}", file=sys.stderr)
            self._available = False
        self._period = max(0.001, period_ms / 1000.0)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: list[dict[str, Any]] = []
        self._marks: list[tuple[float, str]] = []

    @property
    def available(self) -> bool:
        return self._available

    def mark(self, tag: str) -> None:
        if not self._available:
            return
        self._marks.append((time.perf_counter(), tag))

    def _loop(self) -> None:
        pynvml = self._pynvml
        h = self._handle
        get_util = pynvml.nvmlDeviceGetUtilizationRates
        get_mem = pynvml.nvmlDeviceGetMemoryInfo
        get_pow = pynvml.nvmlDeviceGetPowerUsage
        get_temp = pynvml.nvmlDeviceGetTemperature
        TEMP_GPU = pynvml.NVML_TEMPERATURE_GPU
        get_clock = pynvml.nvmlDeviceGetClockInfo
        CLK_GR = pynvml.NVML_CLOCK_GRAPHICS
        CLK_MEM = pynvml.NVML_CLOCK_MEM
        CLK_SM = pynvml.NVML_CLOCK_SM
        # Optional: per-process util + PCIe throughput. These may fail on some drivers; guard each.
        try_pcie = hasattr(pynvml, "nvmlDeviceGetPcieThroughput")
        if try_pcie:
            PCIE_TX = pynvml.NVML_PCIE_UTIL_TX_BYTES
            PCIE_RX = pynvml.NVML_PCIE_UTIL_RX_BYTES
        while not self._stop.is_set():
            t = time.perf_counter()
            rec: dict[str, Any] = {"t": t}
            try:
                u = get_util(h)
                rec["gpu_util_pct"] = u.gpu
                rec["mem_util_pct"] = u.memory
            except Exception:
                pass
            try:
                m = get_mem(h)
                rec["mem_used_bytes"] = m.used
                rec["mem_total_bytes"] = m.total
            except Exception:
                pass
            try:
                rec["power_mw"] = get_pow(h)
            except Exception:
                pass
            try:
                rec["temp_c"] = get_temp(h, TEMP_GPU)
            except Exception:
                pass
            try:
                rec["clock_gr_mhz"] = get_clock(h, CLK_GR)
                rec["clock_mem_mhz"] = get_clock(h, CLK_MEM)
                rec["clock_sm_mhz"] = get_clock(h, CLK_SM)
            except Exception:
                pass
            if try_pcie:
                try:
                    rec["pcie_tx_kbps"] = pynvml.nvmlDeviceGetPcieThroughput(h, PCIE_TX)
                    rec["pcie_rx_kbps"] = pynvml.nvmlDeviceGetPcieThroughput(h, PCIE_RX)
                except Exception:
                    pass
            self._samples.append(rec)
            time.sleep(self._period)

    def start(self) -> None:
        if not self._available:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._available or self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass

    def write(self, out_dir: Path) -> None:
        if not self._available or not self._samples:
            return
        keys = sorted({k for r in self._samples for k in r.keys()})
        # Stable order with t first.
        keys = ["t"] + [k for k in keys if k != "t"]
        # Translate to relative time from first sample.
        t0 = self._samples[0]["t"]
        with (out_dir / "gpu_nvml.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t_rel_s"] + [k for k in keys if k != "t"])
            w.writeheader()
            for r in self._samples:
                rr = {"t_rel_s": r["t"] - t0}
                for k in keys:
                    if k == "t":
                        continue
                    rr[k] = r.get(k, "")
                w.writerow(rr)
        # Per-frame aggregates using marks.
        with (out_dir / "gpu_nvml_per_frame.csv").open("w", newline="") as f:
            cols = [
                "frame_idx",
                "gpu_util_pct_max", "gpu_util_pct_mean",
                "mem_util_pct_max", "mem_util_pct_mean",
                "power_mw_max", "power_mw_mean",
                "temp_c_max", "temp_c_mean",
                "clock_sm_mhz_max", "clock_sm_mhz_mean",
                "clock_mem_mhz_max", "clock_mem_mhz_mean",
                "pcie_tx_kbps_max", "pcie_rx_kbps_max",
                "n_samples",
            ]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            # Pair marks by suffix.
            starts: dict[int, float] = {}
            ends: dict[int, float] = {}
            for t_mark, tag in self._marks:
                if tag.startswith("frame_") and tag.endswith("_start"):
                    starts[int(tag.split("_")[1])] = t_mark
                elif tag.startswith("frame_") and tag.endswith("_end"):
                    ends[int(tag.split("_")[1])] = t_mark
            for idx in sorted(starts):
                if idx not in ends:
                    continue
                t_a, t_b = starts[idx], ends[idx]
                bucket = [r for r in self._samples if t_a <= r["t"] <= t_b]
                if not bucket:
                    continue
                def _stat(key, agg):
                    xs = [r[key] for r in bucket if key in r]
                    if not xs:
                        return ""
                    return float(agg(xs))
                w.writerow({
                    "frame_idx": idx,
                    "gpu_util_pct_max": _stat("gpu_util_pct", max),
                    "gpu_util_pct_mean": _stat("gpu_util_pct", statistics.fmean),
                    "mem_util_pct_max": _stat("mem_util_pct", max),
                    "mem_util_pct_mean": _stat("mem_util_pct", statistics.fmean),
                    "power_mw_max": _stat("power_mw", max),
                    "power_mw_mean": _stat("power_mw", statistics.fmean),
                    "temp_c_max": _stat("temp_c", max),
                    "temp_c_mean": _stat("temp_c", statistics.fmean),
                    "clock_sm_mhz_max": _stat("clock_sm_mhz", max),
                    "clock_sm_mhz_mean": _stat("clock_sm_mhz", statistics.fmean),
                    "clock_mem_mhz_max": _stat("clock_mem_mhz", max),
                    "clock_mem_mhz_mean": _stat("clock_mem_mhz", statistics.fmean),
                    "pcie_tx_kbps_max": _stat("pcie_tx_kbps", max),
                    "pcie_rx_kbps_max": _stat("pcie_rx_kbps", max),
                    "n_samples": len(bucket),
                })


# ──────────────────────────────────────────────────────────────────────────────
# Pass 2c: Nsight Compute hardware-counter pass
# ──────────────────────────────────────────────────────────────────────────────

# Sections to collect. Tensor-core utilization is in ComputeWorkloadAnalysis_Tensor /
# SpeedOfLight (sm__pipe_tensor_op_*). MemoryWorkloadAnalysis covers HBM/L2 traffic.
NCU_SECTIONS = [
    "SpeedOfLight",
    "SpeedOfLight_RooflineChart",
    "SpeedOfLight_HierarchicalTensorRooflineChart",
    "ComputeWorkloadAnalysis",
    "MemoryWorkloadAnalysis",
    "MemoryWorkloadAnalysis_Tables",
    "MemoryWorkloadAnalysis_Chart",
    "Occupancy",
    "InstructionStats",
    "LaunchStats",
    "WarpStateStats",
    "SchedulerStats",
    "SourceCounters",
    "Nvlink",
    "PmSampling",
]


def pass3_ncu(
    weights_path: Path,
    data_folder: Path,
    frames: list[int],
    out_dir: Path,
    extra_args: list[str],
) -> None:
    ncu = shutil.which("ncu")
    if ncu is None:
        # Common cluster location.
        for cand in ("/usr/local/cuda/bin/ncu", "/opt/nvidia/nsight-compute/ncu"):
            if Path(cand).exists():
                ncu = cand
                break
    if ncu is None:
        print("[ncu] ncu not on PATH; skipping. Load nsight-compute or pass --ncu-binary.",
              file=sys.stderr)
        return

    ncu_dir = out_dir / "ncu"
    ncu_dir.mkdir(parents=True, exist_ok=True)
    for idx in frames:
        target = ncu_dir / f"frame_{idx:02d}"
        section_args: list[str] = []
        for s in NCU_SECTIONS:
            section_args += ["--section", s]
        cmd = [
            ncu,
            "--target-processes", "application-only",
            "--replay-mode", "kernel",
            "--cache-control", "all",
            "--clock-control", "base",
            "--export", str(target),
            "--force-overwrite",
            "--print-summary", "per-kernel",
            "--csv",
            "--log-file", str(ncu_dir / f"frame_{idx:02d}_summary.csv"),
            *section_args,
            sys.executable, str(Path(__file__).resolve()),
            "--ncu-fwd-only",
            "--frame", str(idx),
            "--weights", str(weights_path),
            "--data", str(data_folder),
        ]
        cmd += extra_args
        print(f"[ncu] frame {idx}: {' '.join(cmd[:6])} … --export {target}", flush=True)
        try:
            r = subprocess.run(cmd, check=False)
            if r.returncode != 0:
                print(f"[ncu] frame {idx} returned {r.returncode}", file=sys.stderr)
        except Exception as e:
            print(f"[ncu] frame {idx} failed: {e}", file=sys.stderr)

        # Detailed section dump (human readable).
        detail_txt = ncu_dir / f"frame_{idx:02d}_details.txt"
        try:
            r = subprocess.run(
                [ncu, "--import", str(target) + ".ncu-rep",
                 "--print-details", "all", "--print-units", "base"],
                check=False, capture_output=True, text=True,
            )
            detail_txt.write_text(r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr else ""))
        except Exception as e:
            print(f"[ncu] frame {idx} details dump failed: {e}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _default_weights(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    cand = SCRIPT_DIR / "weights" / "v2_8192_d128_L3_hw.bytes"
    if cand.exists():
        return cand
    streamed = SCRIPT_DIR.parent / "StreamingAssets" / "leaf_only_weights_sim_8192.bytes"
    if streamed.exists():
        return streamed
    return cand  # will fail with a clear message later


def _default_data(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    return SCRIPT_DIR / "data" / "multiphase_v2_8192" / "test"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=None,
                    help="Baseline checkpoint .bytes (default: weights/v2_8192_d128_L3_hw.bytes).")
    ap.add_argument("--data", type=str, default=None,
                    help="Test-frame folder (default: data/multiphase_v2_8192/test).")
    ap.add_argument("--output-dir", type=str,
                    default=str(SCRIPT_DIR / "results" / "baseline_benchmark"),
                    help="Where to write CSVs.")
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                    help="Subset of frame indices (default: all up to 20).")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=20,
                    help="Per-frame forward repeats for pass 1 statistics.")
    ap.add_argument("--repeat-pass2", type=int, default=5,
                    help="Per-frame repeats inside torch.profiler.")
    ap.add_argument("--no-profiler", action="store_true",
                    help="Skip pass 2a (torch.profiler).")
    ap.add_argument("--no-nvml", action="store_true",
                    help="Skip NVML sampler.")
    ap.add_argument("--nvml-period-ms", type=float, default=10.0)
    ap.add_argument("--skip-ncu", action="store_true",
                    help="Skip Nsight Compute pass.")
    ap.add_argument("--ncu-frames", type=int, nargs="*", default=[0],
                    help="Frame indices to profile with ncu (default: [0]). Pass nothing to use all.")
    # Internal: forward-only for ncu replay.
    ap.add_argument("--ncu-fwd-only", action="store_true")
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    weights_path = _default_weights(args.weights)
    data_folder = _default_data(args.data)
    if not weights_path.exists():
        raise SystemExit(f"weights not found: {weights_path}")
    if not data_folder.exists():
        raise SystemExit(f"data folder not found: {data_folder}")

    device = _select_device()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model.
    model, arch = _build_model_from_checkpoint(weights_path, device)
    if device.type == "cuda":
        warmup_hmatrix_prolong_gpu(device)

    # --- ncu replay mode: single forward and exit -----------------------------
    if args.ncu_fwd_only:
        dataset = FluidGraphDataset([data_folder])
        if len(dataset) == 0:
            raise SystemExit(f"No frames found in {data_folder}")
        frame_idx = max(0, min(args.frame, len(dataset) - 1))
        batch = dataset[frame_idx]
        inputs = _prepare_frame_inputs(batch, device, leaf_size=int(arch["leaf_size"]))
        with torch.inference_mode():
            for _ in range(3):
                _fwd(model, inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            _fwd(model, inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
        return 0

    # --- regular benchmark ----------------------------------------------------
    dataset = FluidGraphDataset([data_folder])
    if len(dataset) == 0:
        raise SystemExit(f"No frames found in {data_folder}")
    all_indices = list(range(min(20, len(dataset))))
    if args.frames is not None and len(args.frames) > 0:
        frame_idxs = [i for i in args.frames if 0 <= i < len(dataset)]
    else:
        frame_idxs = all_indices

    print(f"Device: {device}")
    print(f"Weights: {weights_path}")
    print(f"Data: {data_folder}")
    print(f"Output: {out_dir}")
    print(f"Arch: d_model={arch['d_model']} leaf_size={arch['leaf_size']} L={arch['num_layers']} "
          f"hw={int(arch.get('highway_ffn_mlp', 0))} input_dim={arch['input_dim']}")
    print(f"Frames: {frame_idxs}")

    # Preload all batches → device tensors + connectivity.
    frame_inputs: list[dict] = []
    for fi in frame_idxs:
        batch = dataset[fi]
        inp = _prepare_frame_inputs(batch, device, leaf_size=int(arch["leaf_size"]))
        frame_inputs.append(inp)
    print(f"Loaded {len(frame_inputs)} frames "
          f"(N range {min(i['num_nodes_real'] for i in frame_inputs)}.."
          f"{max(i['num_nodes_real'] for i in frame_inputs)}).", flush=True)

    # ── Pass 1: component timing ────────────────────────────────────────────
    print("\n=== Pass 1: per-component CUDA timing ===", flush=True)
    pass1_component_timing(
        model=model,
        frame_inputs=frame_inputs,
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
        out_dir=out_dir,
    )

    # Rebuild a clean (un-instrumented) model for pass 2 so wrapper overhead
    # does not bias profiler counters.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, _ = _build_model_from_checkpoint(weights_path, device)

    # ── Pass 2: torch.profiler + NVML ──────────────────────────────────────
    sampler: Optional[NvmlSampler] = None
    if not args.no_nvml:
        sampler = NvmlSampler(period_ms=args.nvml_period_ms, gpu_index=0)
        if sampler.available:
            sampler.start()
        else:
            sampler = None

    if not args.no_profiler:
        print("\n=== Pass 2: torch.profiler ===", flush=True)
        pass2_torch_profiler(
            model=model,
            frame_inputs=frame_inputs,
            device=device,
            repeat=args.repeat_pass2,
            out_dir=out_dir,
            nvml_sampler=sampler,
        )

    if sampler is not None:
        sampler.stop()
        sampler.write(out_dir)

    # ── Pass 3: ncu hardware counters ──────────────────────────────────────
    if not args.skip_ncu:
        print("\n=== Pass 3: Nsight Compute ===", flush=True)
        ncu_frames = args.ncu_frames if args.ncu_frames is not None else frame_idxs
        pass3_ncu(
            weights_path=weights_path,
            data_folder=data_folder,
            frames=ncu_frames,
            out_dir=out_dir,
            extra_args=[],
        )

    print(f"\nDone. Results in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
