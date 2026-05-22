from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_TYPES = ("baseline", "async", "async_o", "gockpt", "gockpt_o")


@dataclass
class FinetuneBenchmarkRun:
    hook_type: str
    output_dir: str
    command: list[str]
    returncode: int
    wall_time_sec: float
    train_runtime_sec: float | None
    train_steps_per_sec: float | None
    checkpoint_count: int
    checkpoint_files: list[str]
    phase_summary: dict[str, Any]
    raw_foreground_summary: dict[str, Any]
    checkpoint_results: list[dict[str, Any]]
    ringbuffer_pressure_samples: list[dict[str, Any]]
    host_memory_samples_file: str | None
    host_memory_stream_file: str | None
    host_memory_sample_count: int
    host_memory_peak_process_tree_rss_gb: float | None
    host_memory_peak_system_used_gb: float | None
    host_memory_min_system_available_gb: float | None
    host_memory_peak_cgroup_current_gb: float | None
    host_memory_cgroup_limit_gb: float | None
    host_memory_cgroup_oom_events_delta: int | None
    host_memory_cgroup_oom_kill_events_delta: int | None
    host_memory_sampling_error: str | None
    power_samples_file: str | None
    power_sample_count: int
    power_avg_w: float | None
    power_peak_w: float | None
    power_energy_j: float | None
    power_sampling_error: str | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real Qwen finetune benchmark across checkpoint hooks."
    )
    parser.add_argument(
        "--hook-types",
        nargs="+",
        default=list(HOOK_TYPES),
        choices=HOOK_TYPES,
        help="Hook implementations to benchmark.",
    )
    parser.add_argument("--seq-len", type=int, default=512, choices=(256, 512))
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--overlap-steps", type=int, default=7)
    parser.add_argument(
        "--gockpt-inflight-packets",
        "--gockpt-reconstruction-queue-depth",
        dest="gockpt_reconstruction_queue_depth",
        type=int,
        default=None,
        help="Max in-flight GoCkpt reconstruction packets forwarded to finetune.py.",
    )
    parser.add_argument(
        "--gockpt-transfer-chunk-mb",
        type=float,
        default=0.0,
        help="GoCkpt GPU-to-CPU transfer chunk size in MiB. Use 0 for whole-tensor copies.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark/finetune_runs")
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=12.0,
        help="Fail before launching runs unless GPU 0 has at least this much free memory.",
    )
    parser.add_argument(
        "--save-final-model",
        action="store_true",
        help="Persist the final Hugging Face model for each run. Disabled by default to focus on checkpoint timing.",
    )
    parser.add_argument(
        "--power-sample-interval-sec",
        type=float,
        default=0.5,
        help="GPU power sampling interval in seconds. Set to 0 to disable power sampling.",
    )
    parser.add_argument(
        "--power-gpu-index",
        type=int,
        default=0,
        help="GPU index for nvidia-smi power sampling.",
    )
    parser.add_argument(
        "--host-memory-sample-interval-sec",
        type=float,
        default=0.5,
        help="Host memory sampling interval in seconds. Set to 0 to disable host memory sampling.",
    )
    args = parser.parse_args()
    if args.gockpt_transfer_chunk_mb < 0:
        raise ValueError("--gockpt-transfer-chunk-mb must be >= 0.")
    if args.power_sample_interval_sec < 0:
        raise ValueError("--power-sample-interval-sec must be >= 0.")
    if args.power_gpu_index < 0:
        raise ValueError("--power-gpu-index must be >= 0.")
    if args.host_memory_sample_interval_sec < 0:
        raise ValueError("--host-memory-sample-interval-sec must be >= 0.")
    return args


def get_gpu0_free_gb() -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    if completed.returncode != 0:
        return None

    lines = completed.stdout.strip().splitlines()
    if not lines:
        return None

    first_line = lines[0]
    return float(first_line) / 1024.0


def query_gpu_power_w(gpu_index: int) -> tuple[float | None, str | None]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None, "nvidia-smi not found"

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if stderr:
            return None, stderr
        return None, "nvidia-smi returned non-zero exit status"

    lines = completed.stdout.strip().splitlines()
    if not lines:
        return None, "nvidia-smi returned empty power output"

    raw = lines[0].strip()
    if not raw:
        return None, "nvidia-smi returned blank power sample"
    if raw == "[Not Supported]":
        return None, "power draw query is not supported on this GPU"
    try:
        return float(raw), None
    except ValueError:
        return None, f"unable to parse nvidia-smi power sample: {raw!r}"


def summarize_power_samples(
    samples: list[dict[str, float]],
) -> dict[str, float | int | None]:
    if not samples:
        return {
            "sample_count": 0,
            "duration_sec": None,
            "avg_power_w": None,
            "peak_power_w": None,
            "energy_j": None,
        }

    powers = [sample["power_w"] for sample in samples]
    energy_j = 0.0
    for previous, current in zip(samples, samples[1:]):
        dt = current["time_sec"] - previous["time_sec"]
        if dt <= 0:
            continue
        energy_j += 0.5 * (previous["power_w"] + current["power_w"]) * dt

    duration_sec = samples[-1]["time_sec"] - samples[0]["time_sec"]
    if duration_sec < 0:
        duration_sec = 0.0
    return {
        "sample_count": len(samples),
        "duration_sec": duration_sec,
        "avg_power_w": sum(powers) / len(powers),
        "peak_power_w": max(powers),
        "energy_j": energy_j,
    }


def sample_gpu_power_until_stopped(
    *,
    stop_event: threading.Event,
    gpu_index: int,
    interval_sec: float,
    samples: list[dict[str, float]],
    error_box: list[str],
) -> None:
    start = time.perf_counter()
    while True:
        power_w, error = query_gpu_power_w(gpu_index)
        elapsed = time.perf_counter() - start
        if power_w is not None:
            samples.append(
                {
                    "time_sec": elapsed,
                    "power_w": power_w,
                }
            )
        elif error and not error_box:
            error_box.append(error)
            break

        if stop_event.wait(interval_sec):
            break



def _kb_to_gb(kb: int | float) -> float:
    return float(kb) / (1024.0 * 1024.0)


def _bytes_to_gb(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0 * 1024.0)


def _read_key_value_ints(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        values[parts[0]] = int(parts[1])
                    except ValueError:
                        continue
    except OSError:
        return {}
    return values


def read_proc_meminfo_kb() -> dict[str, int]:
    meminfo: dict[str, int] = {}
    with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
        for line in handle:
            key, _, raw_value = line.partition(":")
            parts = raw_value.strip().split()
            if not parts:
                continue
            try:
                meminfo[key] = int(parts[0])
            except ValueError:
                continue
    return meminfo


def _read_process_rss_kb(pid: int) -> int:
    try:
        with (Path("/proc") / str(pid) / "status").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except (OSError, ValueError):
        return 0
    return 0


def _read_process_children(pid: int) -> list[int]:
    children_path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
    try:
        raw = children_path.read_text(encoding="utf-8").strip()
    except OSError:
        return []
    children: list[int] = []
    for item in raw.split():
        try:
            children.append(int(item))
        except ValueError:
            continue
    return children


def process_tree_pids(root_pid: int) -> list[int]:
    pids: list[int] = []
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        if not (Path("/proc") / str(pid)).exists():
            continue
        seen.add(pid)
        pids.append(pid)
        stack.extend(_read_process_children(pid))
    return pids


def _decode_mount_path(raw: str) -> Path:
    return Path(raw.replace("\\040", " "))


def _cgroup_mounts() -> tuple[Path | None, Path | None]:
    cgroup2_mount: Path | None = None
    memory_mount: Path | None = None
    try:
        with Path("/proc/self/mountinfo").open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if "-" not in parts:
                    continue
                separator = parts.index("-")
                if separator + 3 >= len(parts):
                    continue
                mount_point = _decode_mount_path(parts[4])
                fs_type = parts[separator + 1]
                super_options = parts[separator + 3].split(",")
                if fs_type == "cgroup2":
                    cgroup2_mount = mount_point
                elif fs_type == "cgroup" and "memory" in super_options:
                    memory_mount = mount_point
    except OSError:
        return None, None
    return cgroup2_mount, memory_mount


def resolve_memory_cgroup(pid: int) -> tuple[str, Path] | None:
    cgroup2_mount, memory_mount = _cgroup_mounts()
    try:
        lines = (Path("/proc") / str(pid) / "cgroup").read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError:
        return None

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, cgroup_path = parts
        relative = cgroup_path.lstrip("/")
        if hierarchy == "0" and controllers == "" and cgroup2_mount is not None:
            return "v2", cgroup2_mount / relative
        if memory_mount is not None and "memory" in controllers.split(","):
            return "v1", memory_mount / relative
    return None


def query_cgroup_memory(cgroup_memory: tuple[str, Path] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cgroup_memory_current_gb": None,
        "cgroup_memory_limit_gb": None,
        "cgroup_memory_used_percent": None,
        "cgroup_memory_oom_events": None,
        "cgroup_memory_oom_kill_events": None,
    }
    if cgroup_memory is None:
        return payload

    version, cgroup_dir = cgroup_memory
    try:
        if version == "v2":
            current = int((cgroup_dir / "memory.current").read_text().strip())
            raw_limit = (cgroup_dir / "memory.max").read_text().strip()
            limit = None if raw_limit == "max" else int(raw_limit)
            events = _read_key_value_ints(cgroup_dir / "memory.events")
            payload["cgroup_memory_current_gb"] = _bytes_to_gb(current)
            payload["cgroup_memory_limit_gb"] = (
                _bytes_to_gb(limit) if limit is not None else None
            )
            if limit:
                payload["cgroup_memory_used_percent"] = current / limit * 100.0
            payload["cgroup_memory_oom_events"] = events.get("oom")
            payload["cgroup_memory_oom_kill_events"] = events.get("oom_kill")
        else:
            current = int((cgroup_dir / "memory.usage_in_bytes").read_text().strip())
            raw_limit = int((cgroup_dir / "memory.limit_in_bytes").read_text().strip())
            # Very large v1 limits usually mean "unlimited".
            limit = None if raw_limit >= (1 << 60) else raw_limit
            payload["cgroup_memory_current_gb"] = _bytes_to_gb(current)
            payload["cgroup_memory_limit_gb"] = (
                _bytes_to_gb(limit) if limit is not None else None
            )
            if limit:
                payload["cgroup_memory_used_percent"] = current / limit * 100.0
            failcnt_path = cgroup_dir / "memory.failcnt"
            if failcnt_path.exists():
                payload["cgroup_memory_oom_events"] = int(
                    failcnt_path.read_text().strip()
                )
    except (OSError, ValueError):
        return payload
    return payload


def query_host_memory(
    root_pid: int,
    cgroup_memory: tuple[str, Path] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        meminfo = read_proc_meminfo_kb()
    except OSError as exc:
        return None, f"unable to read /proc/meminfo: {exc}"

    total_kb = meminfo.get("MemTotal")
    available_kb = meminfo.get("MemAvailable")
    if available_kb is None:
        available_kb = (
            meminfo.get("MemFree", 0)
            + meminfo.get("Buffers", 0)
            + meminfo.get("Cached", 0)
            + meminfo.get("SReclaimable", 0)
        )
    if total_kb is None or total_kb <= 0:
        return None, "/proc/meminfo did not include MemTotal"

    used_kb = max(0, total_kb - available_kb)
    pids = process_tree_pids(root_pid)
    process_tree_rss_kb = sum(_read_process_rss_kb(pid) for pid in pids)
    sample: dict[str, Any] = {
        "system_total_gb": _kb_to_gb(total_kb),
        "system_available_gb": _kb_to_gb(available_kb),
        "system_used_gb": _kb_to_gb(used_kb),
        "system_used_percent": used_kb / total_kb * 100.0,
        "process_tree_rss_gb": _kb_to_gb(process_tree_rss_kb),
        "process_tree_pid_count": len(pids),
        "process_tree_pids": pids,
    }
    sample.update(query_cgroup_memory(cgroup_memory))
    return sample, None


def _numeric_samples(samples: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [sample for sample in samples if isinstance(sample.get(key), (int, float))]


def _first_numeric(samples: list[dict[str, Any]], key: str) -> int | float | None:
    values = _numeric_samples(samples, key)
    if not values:
        return None
    return values[0][key]


def _last_numeric(samples: list[dict[str, Any]], key: str) -> int | float | None:
    values = _numeric_samples(samples, key)
    if not values:
        return None
    return values[-1][key]


def _max_numeric(samples: list[dict[str, Any]], key: str) -> int | float | None:
    values = _numeric_samples(samples, key)
    if not values:
        return None
    return max(float(sample[key]) for sample in values)


def _min_numeric(samples: list[dict[str, Any]], key: str) -> int | float | None:
    values = _numeric_samples(samples, key)
    if not values:
        return None
    return min(float(sample[key]) for sample in values)


def _time_for_extreme(
    samples: list[dict[str, Any]], key: str, *, find_max: bool
) -> float | None:
    values = _numeric_samples(samples, key)
    if not values:
        return None
    selected = max(values, key=lambda sample: float(sample[key])) if find_max else min(
        values, key=lambda sample: float(sample[key])
    )
    return float(selected.get("time_sec", 0.0))


def _counter_delta(samples: list[dict[str, Any]], key: str) -> int | None:
    first = _first_numeric(samples, key)
    last = _last_numeric(samples, key)
    if first is None or last is None:
        return None
    return max(0, int(last) - int(first))


def summarize_host_memory_samples(
    samples: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    if not samples:
        return {
            "sample_count": 0,
            "duration_sec": None,
            "peak_process_tree_rss_gb": None,
            "peak_process_tree_rss_time_sec": None,
            "peak_system_used_gb": None,
            "peak_system_used_time_sec": None,
            "min_system_available_gb": None,
            "min_system_available_time_sec": None,
            "peak_system_used_percent": None,
            "peak_cgroup_current_gb": None,
            "peak_cgroup_current_time_sec": None,
            "cgroup_limit_gb": None,
            "peak_cgroup_used_percent": None,
            "cgroup_oom_events_delta": None,
            "cgroup_oom_kill_events_delta": None,
            "last_process_tree_rss_gb": None,
            "last_system_available_gb": None,
            "last_cgroup_current_gb": None,
        }

    duration_sec = float(samples[-1].get("time_sec", 0.0)) - float(
        samples[0].get("time_sec", 0.0)
    )
    if duration_sec < 0:
        duration_sec = 0.0

    return {
        "sample_count": len(samples),
        "duration_sec": duration_sec,
        "peak_process_tree_rss_gb": _max_numeric(samples, "process_tree_rss_gb"),
        "peak_process_tree_rss_time_sec": _time_for_extreme(
            samples, "process_tree_rss_gb", find_max=True
        ),
        "peak_system_used_gb": _max_numeric(samples, "system_used_gb"),
        "peak_system_used_time_sec": _time_for_extreme(
            samples, "system_used_gb", find_max=True
        ),
        "min_system_available_gb": _min_numeric(samples, "system_available_gb"),
        "min_system_available_time_sec": _time_for_extreme(
            samples, "system_available_gb", find_max=False
        ),
        "peak_system_used_percent": _max_numeric(samples, "system_used_percent"),
        "peak_cgroup_current_gb": _max_numeric(samples, "cgroup_memory_current_gb"),
        "peak_cgroup_current_time_sec": _time_for_extreme(
            samples, "cgroup_memory_current_gb", find_max=True
        ),
        "cgroup_limit_gb": _first_numeric(samples, "cgroup_memory_limit_gb"),
        "peak_cgroup_used_percent": _max_numeric(
            samples, "cgroup_memory_used_percent"
        ),
        "cgroup_oom_events_delta": _counter_delta(
            samples, "cgroup_memory_oom_events"
        ),
        "cgroup_oom_kill_events_delta": _counter_delta(
            samples, "cgroup_memory_oom_kill_events"
        ),
        "last_process_tree_rss_gb": _last_numeric(samples, "process_tree_rss_gb"),
        "last_system_available_gb": _last_numeric(samples, "system_available_gb"),
        "last_cgroup_current_gb": _last_numeric(samples, "cgroup_memory_current_gb"),
    }


def record_host_memory_sample(
    samples: list[dict[str, Any]],
    sample: dict[str, Any],
    stream_path: Path | None,
) -> None:
    samples.append(sample)
    if stream_path is None:
        return
    try:
        with stream_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True))
            handle.write("\n")
    except OSError:
        return


def sample_host_memory_until_stopped(
    *,
    stop_event: threading.Event,
    root_pid: int,
    cgroup_memory: tuple[str, Path] | None,
    interval_sec: float,
    start_time: float,
    samples: list[dict[str, Any]],
    stream_path: Path | None,
    error_box: list[str],
) -> None:
    while True:
        sample, error = query_host_memory(root_pid, cgroup_memory)
        elapsed = time.perf_counter() - start_time
        if sample is not None:
            sample["time_sec"] = elapsed
            sample["event"] = "sample"
            record_host_memory_sample(samples, sample, stream_path)
        elif error and not error_box:
            error_box.append(error)
            break

        if stop_event.wait(interval_sec):
            break

def build_power_report(
    args: argparse.Namespace,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    power_runs: list[dict[str, Any]] = []
    for run in runs:
        run_dir = Path(run["output_dir"])
        samples_path = run_dir / "power_samples.json"
        samples_payload: dict[str, Any] = {}
        if samples_path.exists():
            with samples_path.open("r", encoding="utf-8") as handle:
                samples_payload = json.load(handle)

        power_runs.append(
            {
                "hook_type": run["hook_type"],
                "output_dir": run["output_dir"],
                "returncode": run["returncode"],
                "wall_time_sec": run["wall_time_sec"],
                "power_sampling_error": run.get("power_sampling_error"),
                "power_summary": samples_payload.get("summary", {}),
                "power_samples": samples_payload.get("samples", []),
            }
        )

    return {
        "config": {
            "hook_types": args.hook_types,
            "seq_len": args.seq_len,
            "max_steps": args.max_steps,
            "save_steps": args.save_steps,
            "overlap_steps": args.overlap_steps,
            "gockpt_reconstruction_queue_depth": args.gockpt_reconstruction_queue_depth,
            "gockpt_transfer_chunk_mb": args.gockpt_transfer_chunk_mb,
            "gradient_checkpointing": args.gradient_checkpointing,
            "power_sample_interval_sec": args.power_sample_interval_sec,
            "power_gpu_index": args.power_gpu_index,
        },
        "runs": power_runs,
    }


def build_host_memory_report(
    args: argparse.Namespace,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    memory_runs: list[dict[str, Any]] = []
    for run in runs:
        run_dir = Path(run["output_dir"])
        samples_path = run_dir / "host_memory_samples.json"
        samples_payload: dict[str, Any] = {}
        if samples_path.exists():
            with samples_path.open("r", encoding="utf-8") as handle:
                samples_payload = json.load(handle)

        memory_runs.append(
            {
                "hook_type": run["hook_type"],
                "output_dir": run["output_dir"],
                "returncode": run["returncode"],
                "wall_time_sec": run["wall_time_sec"],
                "host_memory_sampling_error": run.get("host_memory_sampling_error"),
                "host_memory_summary": samples_payload.get("summary", {}),
                "host_memory_samples": samples_payload.get("samples", []),
            }
        )

    return {
        "config": {
            "hook_types": args.hook_types,
            "seq_len": args.seq_len,
            "max_steps": args.max_steps,
            "save_steps": args.save_steps,
            "overlap_steps": args.overlap_steps,
            "gockpt_reconstruction_queue_depth": args.gockpt_reconstruction_queue_depth,
            "gockpt_transfer_chunk_mb": args.gockpt_transfer_chunk_mb,
            "gradient_checkpointing": args.gradient_checkpointing,
            "host_memory_sample_interval_sec": args.host_memory_sample_interval_sec,
        },
        "runs": memory_runs,
    }


def check_gpu_capacity(args: argparse.Namespace) -> None:
    if args.min_free_gb <= 0:
        return

    free_gb = get_gpu0_free_gb()
    if free_gb is None or free_gb >= args.min_free_gb:
        return

    raise RuntimeError(
        f"GPU 0 has only {free_gb:.2f} GiB free, below --min-free-gb "
        f"{args.min_free_gb:.2f}. Free the RTX 4090 or pass a lower threshold "
        "for a constrained smoke run."
    )


def build_command(args: argparse.Namespace, hook_type: str, run_dir: Path) -> list[str]:
    command = [
        str(args.python),
        str(REPO_ROOT / "finetune.py"),
        "--hook-type",
        hook_type,
        "--seq-len",
        str(args.seq_len),
        "--batch-size",
        "1",
        "--max-steps",
        str(args.max_steps),
        "--save-steps",
        str(args.save_steps),
        "--overlap-steps",
        str(args.overlap_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--logging-steps",
        str(args.logging_steps),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(run_dir),
        "--profile-phases",
    ]
    if args.gockpt_reconstruction_queue_depth is not None:
        command.extend(
            [
                "--gockpt-inflight-packets",
                str(args.gockpt_reconstruction_queue_depth),
            ]
        )
    command.extend(["--gockpt-transfer-chunk-mb", str(args.gockpt_transfer_chunk_mb)])
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    if not args.save_final_model:
        command.append("--skip-final-model-save")
    return command


def run_one(args: argparse.Namespace, hook_type: str) -> FinetuneBenchmarkRun:
    run_dir = args.output_dir / hook_type
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(args, hook_type, run_dir)

    log_path = run_dir / "stdout.log"
    start = time.perf_counter()
    power_samples: list[dict[str, float]] = []
    power_error_box: list[str] = []
    power_stop_event = threading.Event()
    power_thread: threading.Thread | None = None
    if args.power_sample_interval_sec > 0:
        power_thread = threading.Thread(
            target=sample_gpu_power_until_stopped,
            kwargs={
                "stop_event": power_stop_event,
                "gpu_index": args.power_gpu_index,
                "interval_sec": args.power_sample_interval_sec,
                "samples": power_samples,
                "error_box": power_error_box,
            },
            daemon=True,
            name=f"gpu-power-sampler-{hook_type}",
        )
        power_thread.start()

    host_memory_samples: list[dict[str, Any]] = []
    host_memory_stream_path = run_dir / "host_memory_samples.jsonl"
    if host_memory_stream_path.exists():
        host_memory_stream_path.unlink()
    host_memory_error_box: list[str] = []
    host_memory_stop_event = threading.Event()
    host_memory_thread: threading.Thread | None = None
    host_memory_start: float | None = None
    host_memory_cgroup: tuple[str, Path] | None = None

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if args.host_memory_sample_interval_sec > 0:
            host_memory_start = time.perf_counter()
            host_memory_cgroup = resolve_memory_cgroup(process.pid)
            initial_sample, initial_error = query_host_memory(
                process.pid, host_memory_cgroup
            )
            if initial_sample is not None:
                initial_sample["time_sec"] = 0.0
                initial_sample["event"] = "process_start"
                record_host_memory_sample(
                    host_memory_samples, initial_sample, host_memory_stream_path
                )
            elif initial_error and not host_memory_error_box:
                host_memory_error_box.append(initial_error)
            host_memory_thread = threading.Thread(
                target=sample_host_memory_until_stopped,
                kwargs={
                    "stop_event": host_memory_stop_event,
                    "root_pid": process.pid,
                    "cgroup_memory": host_memory_cgroup,
                    "interval_sec": args.host_memory_sample_interval_sec,
                    "start_time": host_memory_start,
                    "samples": host_memory_samples,
                    "stream_path": host_memory_stream_path,
                    "error_box": host_memory_error_box,
                },
                daemon=True,
                name=f"host-memory-sampler-{hook_type}",
            )
            host_memory_thread.start()
        returncode = process.wait()

    if host_memory_thread is not None:
        host_memory_stop_event.set()
        host_memory_thread.join()
        if host_memory_start is not None:
            final_sample, final_error = query_host_memory(
                process.pid, host_memory_cgroup
            )
            if final_sample is not None:
                final_sample["time_sec"] = time.perf_counter() - host_memory_start
                final_sample["event"] = "process_exit"
                record_host_memory_sample(
                    host_memory_samples, final_sample, host_memory_stream_path
                )
            elif final_error and not host_memory_error_box:
                host_memory_error_box.append(final_error)

    if power_thread is not None:
        power_stop_event.set()
        power_thread.join()

    wall_time = time.perf_counter() - start

    power_summary = summarize_power_samples(power_samples)
    power_samples_path = run_dir / "power_samples.json"
    with power_samples_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "hook_type": hook_type,
                "gpu_index": args.power_gpu_index,
                "sample_interval_sec": args.power_sample_interval_sec,
                "summary": power_summary,
                "error": power_error_box[0] if power_error_box else None,
                "samples": power_samples,
            },
            handle,
            indent=2,
        )

    host_memory_summary = summarize_host_memory_samples(host_memory_samples)
    host_memory_samples_path = run_dir / "host_memory_samples.json"
    with host_memory_samples_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "hook_type": hook_type,
                "sample_interval_sec": args.host_memory_sample_interval_sec,
                "stream_file": str(host_memory_stream_path),
                "summary": host_memory_summary,
                "error": host_memory_error_box[0] if host_memory_error_box else None,
                "samples": host_memory_samples,
            },
            handle,
            indent=2,
        )

    summary_path = run_dir / "run_summary.json"
    summary: dict[str, Any] = {}
    error = None
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    elif returncode != 0:
        error = f"run failed; see {log_path}"
    else:
        error = f"missing run summary at {summary_path}"

    checkpoint_files = summary.get("checkpoint_files", [])
    return FinetuneBenchmarkRun(
        hook_type=hook_type,
        output_dir=str(run_dir),
        command=command,
        returncode=returncode,
        wall_time_sec=wall_time,
        train_runtime_sec=summary.get("train_runtime_sec"),
        train_steps_per_sec=summary.get("train_steps_per_sec"),
        checkpoint_count=len(checkpoint_files),
        checkpoint_files=checkpoint_files,
        phase_summary=summary.get("phase_summary", {}),
        raw_foreground_summary=summary.get("raw_foreground_summary", {}),
        checkpoint_results=summary.get("checkpoint_results", []),
        ringbuffer_pressure_samples=summary.get("ringbuffer_pressure_samples", []),
        host_memory_samples_file=str(host_memory_samples_path),
        host_memory_stream_file=str(host_memory_stream_path),
        host_memory_sample_count=int(host_memory_summary["sample_count"]),
        host_memory_peak_process_tree_rss_gb=(
            float(host_memory_summary["peak_process_tree_rss_gb"])
            if host_memory_summary["peak_process_tree_rss_gb"] is not None
            else None
        ),
        host_memory_peak_system_used_gb=(
            float(host_memory_summary["peak_system_used_gb"])
            if host_memory_summary["peak_system_used_gb"] is not None
            else None
        ),
        host_memory_min_system_available_gb=(
            float(host_memory_summary["min_system_available_gb"])
            if host_memory_summary["min_system_available_gb"] is not None
            else None
        ),
        host_memory_peak_cgroup_current_gb=(
            float(host_memory_summary["peak_cgroup_current_gb"])
            if host_memory_summary["peak_cgroup_current_gb"] is not None
            else None
        ),
        host_memory_cgroup_limit_gb=(
            float(host_memory_summary["cgroup_limit_gb"])
            if host_memory_summary["cgroup_limit_gb"] is not None
            else None
        ),
        host_memory_cgroup_oom_events_delta=(
            int(host_memory_summary["cgroup_oom_events_delta"])
            if host_memory_summary["cgroup_oom_events_delta"] is not None
            else None
        ),
        host_memory_cgroup_oom_kill_events_delta=(
            int(host_memory_summary["cgroup_oom_kill_events_delta"])
            if host_memory_summary["cgroup_oom_kill_events_delta"] is not None
            else None
        ),
        host_memory_sampling_error=(
            host_memory_error_box[0] if host_memory_error_box else None
        ),
        power_samples_file=str(power_samples_path),
        power_sample_count=int(power_summary["sample_count"]),
        power_avg_w=(
            float(power_summary["avg_power_w"])
            if power_summary["avg_power_w"] is not None
            else None
        ),
        power_peak_w=(
            float(power_summary["peak_power_w"])
            if power_summary["peak_power_w"] is not None
            else None
        ),
        power_energy_j=(
            float(power_summary["energy_j"])
            if power_summary["energy_j"] is not None
            else None
        ),
        power_sampling_error=power_error_box[0] if power_error_box else None,
        error=error,
    )


def phase_total(run: dict[str, Any], phase: str) -> float | None:
    phase_data = run.get("phase_summary", {}).get(phase)
    if phase_data is None:
        return None
    return float(phase_data["total_sec"])


def build_comparison(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_hook = {run["hook_type"]: run for run in runs if run["returncode"] == 0}
    baseline = by_hook.get("baseline")
    comparisons: dict[str, Any] = {}

    for hook_type, run in by_hook.items():
        item: dict[str, Any] = {
            "train_steps_per_sec": run.get("train_steps_per_sec"),
            "wall_time_sec": run.get("wall_time_sec"),
            "checkpoint_count": run.get("checkpoint_count"),
            "save_checkpoint_total_sec": phase_total(run, "hook.save_checkpoint"),
            "forward_begin_total_sec": phase_total(run, "hook.forward_begin"),
            "backward_end_total_sec": phase_total(run, "hook.backward_end"),
            "update_begin_total_sec": phase_total(run, "hook.update_begin"),
            "update_end_total_sec": phase_total(run, "hook.update_end"),
        }
        if baseline is not None and hook_type != "baseline":
            base_sps = baseline.get("train_steps_per_sec")
            run_sps = run.get("train_steps_per_sec")
            if base_sps and run_sps:
                item["speedup_vs_baseline"] = run_sps / base_sps
            base_ckpt = phase_total(baseline, "hook.save_checkpoint")
            run_ckpt = phase_total(run, "hook.save_checkpoint")
            if base_ckpt and run_ckpt is not None:
                item["save_checkpoint_time_ratio_vs_baseline"] = run_ckpt / base_ckpt
        comparisons[hook_type] = item

    return comparisons


def main() -> None:
    args = parse_args()
    check_gpu_capacity(args)

    if args.output_dir.exists() and not args.keep_output:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = [asdict(run_one(args, hook_type)) for hook_type in args.hook_types]
    report = {
        "config": {
            "hook_types": args.hook_types,
            "seq_len": args.seq_len,
            "max_steps": args.max_steps,
            "save_steps": args.save_steps,
            "overlap_steps": args.overlap_steps,
            "gockpt_reconstruction_queue_depth": args.gockpt_reconstruction_queue_depth,
            "gockpt_transfer_chunk_mb": args.gockpt_transfer_chunk_mb,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "gradient_checkpointing": args.gradient_checkpointing,
            "save_final_model": args.save_final_model,
            "min_free_gb": args.min_free_gb,
            "power_sample_interval_sec": args.power_sample_interval_sec,
            "power_gpu_index": args.power_gpu_index,
            "host_memory_sample_interval_sec": args.host_memory_sample_interval_sec,
        },
        "runs": runs,
        "comparison": build_comparison(runs),
    }

    report_path = args.output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    power_report_path = args.output_dir / "power.json"
    with power_report_path.open("w", encoding="utf-8") as handle:
        json.dump(build_power_report(args, runs), handle, indent=2)

    host_memory_report_path = args.output_dir / "host_memory.json"
    with host_memory_report_path.open("w", encoding="utf-8") as handle:
        json.dump(build_host_memory_report(args, runs), handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"power report saved to {power_report_path}")
    print(f"host memory report saved to {host_memory_report_path}")
    if any(run["returncode"] != 0 for run in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
