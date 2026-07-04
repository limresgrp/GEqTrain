"""Benchmark GEqTrain model scaling on synthetic graphs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import yaml
from e3nn.o3 import Irreps
from omegaconf import OmegaConf

from geqtrain.data import AtomicDataDict
from geqtrain.data.dataloader import Collater
from geqtrain.model import model_from_config
from geqtrain.train.components.inference import prepare_chunked_input_data, run_inference
from geqtrain.train.utils import evaluate_end_chunking_condition
from geqtrain.utils import Config, load_config
from geqtrain.utils._global_options import apply_global_config
from geqtrain.utils.torch_geometric import Data


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-GPU scaling benchmark on synthetic graphs for a GEqTrain model. "
            "The benchmark records wall time, CUDA memory, OOM limits, and optional chunking behavior."
        )
    )
    parser.add_argument("config", help="Experiment YAML or model-only YAML to benchmark.")
    parser.add_argument("-d", "--device", default="cuda:0", help="Device to benchmark, default: cuda:0.")
    parser.add_argument("-o", "--output-dir", default="scaling_results", help="Directory for CSV/report/plots.")
    parser.add_argument("--node-counts", nargs="+", type=int, help="Explicit node counts to test.")
    parser.add_argument("--start-nodes", type=int, default=256, help="First node count when --node-counts is omitted.")
    parser.add_argument("--max-nodes", type=int, default=16384, help="Maximum generated node count.")
    parser.add_argument("--growth-factor", type=float, default=1.6, help="Geometric growth factor for node counts.")
    parser.add_argument("--avg-degree", type=int, default=32, help="Outgoing synthetic edges per node.")
    parser.add_argument(
        "--modes",
        choices=("full", "chunked", "both"),
        default="both",
        help="Benchmark full graph, chunked graph, or both.",
    )
    parser.add_argument(
        "--chunk-batch-max-atoms",
        nargs="+",
        type=int,
        default=[1000],
        help="batch_max_atoms values used for chunked inference.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup repeats per size/mode.")
    parser.add_argument("--repeats", type=int, default=3, help="Measured repeats per size/mode.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic inputs.")
    parser.add_argument("--stop-on-oom", action="store_true", help="Stop increasing full-graph size after first OOM.")
    parser.add_argument("--plot-format", default="png", help="Plot file format, default: png.")
    parser.add_argument(
        "--mixed-precision",
        choices=("off", "bf16", "fp16"),
        default="off",
        help="Override inference autocast for the benchmark.",
    )
    parser.add_argument("--default-dtype", choices=("float32", "float64"), help="Override torch default dtype.")
    parser.add_argument("--dry-run-model", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(args=args)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _repo_config_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[2] / "config" / Path(*parts)


def _strip_hydra_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    data = deepcopy(data)
    data.pop("defaults", None)
    return data


def _model_only_config(path: Path, avg_degree: int) -> Config:
    """Load a model YAML with enough synthetic defaults to resolve stack interpolations."""
    base = _read_yaml(_repo_config_path("base.yaml"))
    common = _read_yaml(_repo_config_path("model", "stack_blocks", "common.yaml"))
    model_cfg = _read_yaml(path)

    merged: Dict[str, Any] = {}
    for part in (base, common, model_cfg):
        merged = _deep_merge(merged, _strip_hydra_defaults(part))

    merged.setdefault("num_types", 21)
    merged.setdefault("type_names", [f"type_{i}" for i in range(int(merged["num_types"]))])
    merged.setdefault("avg_num_neighbors", int(avg_degree))
    merged.setdefault("node_attributes", {})
    merged.setdefault("eq_node_attributes", {})
    merged.setdefault("edge_attributes", {})
    merged.setdefault("eq_edge_attributes", {})
    merged.setdefault("graph_attributes", {})
    merged.setdefault("eq_graph_attributes", {})
    merged.setdefault("irreps_edge_sh", None)
    merged.setdefault("wandb", False)
    merged.setdefault("denormalize_inference_outputs", False)
    node_attrs = merged.get("node_attributes", {})
    if isinstance(node_attrs, dict) and isinstance(node_attrs.get(AtomicDataDict.NODE_TYPE_KEY), dict):
        node_type_cfg = node_attrs[AtomicDataDict.NODE_TYPE_KEY]
        node_type_cfg.setdefault("attribute_type", "categorical")
        node_type_cfg.setdefault("embedding_mode", "one_hot")
        node_type_cfg.setdefault("num_types", int(merged["num_types"]))

    resolved = OmegaConf.to_container(OmegaConf.create(merged), resolve=True)
    config = Config.from_dict(resolved)
    config._sync_stack_embedding_input_attrs()
    config.filepath = str(path)
    return config


def load_benchmark_config(config_path: str, avg_degree: int) -> Config:
    path = Path(config_path).expanduser().resolve()
    try:
        config = load_config(str(path))
    except Exception:
        raw = _read_yaml(path)
        if "model" not in raw:
            raise
        config = _model_only_config(path, avg_degree=avg_degree)

    config["avg_num_neighbors"] = int(config.get("avg_num_neighbors", avg_degree))
    config["denormalize_inference_outputs"] = False
    config["wandb"] = False
    config["ddp"] = False
    return config


def make_node_counts(args: argparse.Namespace) -> List[int]:
    if args.node_counts:
        return sorted(set(int(n) for n in args.node_counts if n > 0))

    values: List[int] = []
    n = int(args.start_nodes)
    while n <= int(args.max_nodes):
        values.append(n)
        n_next = int(math.ceil(n * float(args.growth_factor)))
        if n_next <= n:
            n_next = n + 1
        n = n_next
    return values


def _attr_dim(values: Dict[str, Any], *, equivariant: bool = False) -> int:
    if equivariant:
        return Irreps(values["irreps"]).dim
    return int(values.get("embedding_dimensionality", 1))


def _is_numerical(values: Dict[str, Any]) -> bool:
    return values.get("attribute_type", "categorical") == "numerical" and not any(
        key in values for key in ("min_value", "max_value", "bin_edges", "bins")
    )


def _categorical_num_types(name: str, values: Dict[str, Any], config: Config) -> int:
    if name == AtomicDataDict.NODE_TYPE_KEY:
        return int(config.get("num_types", values.get("num_types", 1)))
    return int(values.get("actual_num_types", values.get("num_types", config.get("num_types", 1))))


def _make_attr_tensor(
    name: str,
    values: Dict[str, Any],
    count: int,
    config: Config,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    if _is_numerical(values):
        return torch.randn((count, _attr_dim(values)), generator=generator, dtype=dtype)
    n_types = max(1, _categorical_num_types(name, values, config))
    return torch.randint(0, n_types, (count, 1), generator=generator, dtype=torch.long)


def _make_eq_attr_tensor(
    values: Dict[str, Any],
    count: int,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn((count, _attr_dim(values, equivariant=True)), generator=generator, dtype=dtype)


def make_synthetic_batch(config: Config, n_nodes: int, avg_degree: int, seed: int) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    dtype = torch.get_default_dtype()

    degree = max(1, int(avg_degree))
    n_edges = int(n_nodes) * degree
    src = torch.arange(n_nodes, dtype=torch.long).repeat_interleave(degree)
    dst = torch.randint(0, n_nodes, (n_edges,), generator=generator, dtype=torch.long)
    if n_nodes > 1:
        dst = torch.where(dst == src, (dst + 1) % n_nodes, dst)

    fields: Dict[str, torch.Tensor] = {
        AtomicDataDict.POSITIONS_KEY: torch.randn((n_nodes, 3), generator=generator, dtype=dtype),
        AtomicDataDict.EDGE_INDEX_KEY: torch.stack((src, dst), dim=0),
        AtomicDataDict.ENSEMBLE_INDEX_KEY: torch.tensor([0], dtype=torch.long),
    }

    for name, values in config.get("node_attributes", {}).items():
        fields[name] = _make_attr_tensor(name, values or {}, n_nodes, config, generator, dtype)
    for name, values in config.get("eq_node_attributes", {}).items():
        fields[name] = _make_eq_attr_tensor(values or {}, n_nodes, generator, dtype)
    for name, values in config.get("edge_attributes", {}).items():
        fields[name] = _make_attr_tensor(name, values or {}, n_edges, config, generator, dtype)
    for name, values in config.get("eq_edge_attributes", {}).items():
        fields[name] = _make_eq_attr_tensor(values or {}, n_edges, generator, dtype)
    for name, values in config.get("graph_attributes", {}).items():
        fields[name] = _make_attr_tensor(name, values or {}, 1, config, generator, dtype)
    for name, values in config.get("eq_graph_attributes", {}).items():
        fields[name] = _make_eq_attr_tensor(values or {}, 1, generator, dtype)

    if AtomicDataDict.NODE_TYPE_KEY not in fields:
        n_types = int(config.get("num_types", 1))
        fields[AtomicDataDict.NODE_TYPE_KEY] = torch.randint(
            0, max(1, n_types), (n_nodes, 1), generator=generator, dtype=torch.long
        )

    data = Data(**fields)
    return Collater().collate([data])


def cuda_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gb": props.total_memory / 1024**3,
                "gpu_capability": f"{props.major}.{props.minor}",
                "gpu_multi_processor_count": props.multi_processor_count,
            }
        )
        try:
            smi = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,power.limit",
                    "--format=csv,noheader,nounits",
                    f"--id={device.index if device.index is not None else 0}",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if smi.returncode == 0 and smi.stdout.strip():
                driver, power_limit = [x.strip() for x in smi.stdout.strip().split(",", maxsplit=1)]
                info["nvidia_driver"] = driver
                info["gpu_power_limit_w"] = power_limit
        except OSError:
            pass
    return info


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _memory_stats(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {"peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}
    return {
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _materialize_output(out: Dict[str, Any]) -> None:
    for value in out.values():
        if torch.is_tensor(value) and (torch.is_floating_point(value) or torch.is_complex(value)):
            _ = float(value.detach().abs().mean().cpu())


def _run_forward(model: torch.nn.Module, batch: Any, device: torch.device, config: Config) -> int:
    if config.get("chunking", False):
        return _run_forward_streamed_chunks(model, batch, device, config)

    out, _, _, _ = run_inference(
        model=model,
        data=batch,
        device=device,
        config=config,
        loss_fn=None,
        already_computed_nodes=None,
        is_train=False,
    )
    _materialize_output(out)
    del out
    return 1


def _run_forward_streamed_chunks(model: torch.nn.Module, batch: Any, device: torch.device, config: Config) -> int:
    already_computed_nodes = None
    chunk_config = deepcopy(config)
    chunk_config["chunking"] = False
    n_chunks = 0
    total_centers = int(len(batch[AtomicDataDict.EDGE_INDEX_KEY][0].unique()))
    while True:
        batch_chunk, chunk_center_nodes = prepare_chunked_input_data(
            batch=batch,
            already_computed_nodes=already_computed_nodes,
            batch_max_atoms=int(config.get("batch_max_atoms", 1000)),
            chunk_ignore_keys=config.get("chunk_ignore_keys", []),
        )
        if batch_chunk is None or len(chunk_center_nodes) == 0:
            break
        out, _, _, num_centers = run_inference(
            model=model,
            data=batch_chunk,
            device=device,
            config=chunk_config,
            loss_fn=None,
            already_computed_nodes=None,
            is_train=False,
        )
        _materialize_output(out)
        del out
        n_chunks += 1
        del num_centers
        already_computed_nodes = evaluate_end_chunking_condition(
            already_computed_nodes,
            chunk_center_nodes,
            total_centers,
        )
        if already_computed_nodes is None:
            break
    return n_chunks


def benchmark_case(
    model: torch.nn.Module,
    batch: Any,
    device: torch.device,
    base_config: Config,
    *,
    mode: str,
    chunk_size: Optional[int],
    warmup: int,
    repeats: int,
) -> Dict[str, Any]:
    config = deepcopy(base_config)
    config["chunking"] = mode == "chunked"
    if chunk_size is not None:
        config["batch_max_atoms"] = int(chunk_size)

    try:
        for _ in range(max(0, warmup)):
            _run_forward(model, batch, device, config)
            _sync(device)

        _reset_memory(device)
        times: List[float] = []
        chunks = 1
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            chunks = _run_forward(model, batch, device, config)
            _sync(device)
            times.append(time.perf_counter() - t0)

        memory = _memory_stats(device)
        return {
            "status": "ok",
            "seconds_mean": sum(times) / len(times),
            "seconds_min": min(times),
            "seconds_max": max(times),
            "chunks": chunks,
            **memory,
            "error": "",
        }
    except RuntimeError as exc:
        if _is_oom(exc):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return {
                "status": "oom",
                "seconds_mean": "",
                "seconds_min": "",
                "seconds_max": "",
                "chunks": "",
                "peak_allocated_gb": "",
                "peak_reserved_gb": "",
                "error": str(exc).splitlines()[0],
            }
        raise


def write_report(
    path: Path,
    info: Dict[str, Any],
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    *,
    model_parameters: int,
) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    oom_rows = [r for r in rows if r["status"] == "oom"]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("GEqTrain synthetic scaling benchmark\n")
        handle.write("=" * 40 + "\n\n")
        handle.write("Environment\n")
        for key, value in info.items():
            handle.write(f"{key}: {value}\n")
        handle.write("\nConfiguration\n")
        handle.write(f"config: {Path(args.config).expanduser().resolve()}\n")
        handle.write(f"device: {args.device}\n")
        handle.write(f"avg_degree: {args.avg_degree}\n")
        handle.write(f"warmup: {args.warmup}\n")
        handle.write(f"repeats: {args.repeats}\n")
        handle.write(f"model_parameters: {model_parameters:,}\n")
        handle.write("\nSummary\n")
        handle.write(f"successful_cases: {len(ok_rows)}\n")
        handle.write(f"oom_cases: {len(oom_rows)}\n")
        if ok_rows:
            largest = max(ok_rows, key=lambda r: int(r["nodes"]))
            handle.write(f"largest_success_nodes: {largest['nodes']} ({largest['mode']}, chunk={largest['chunk_size']})\n")
        if oom_rows:
            first = min(oom_rows, key=lambda r: int(r["nodes"]))
            handle.write(f"first_oom_nodes: {first['nodes']} ({first['mode']}, chunk={first['chunk_size']})\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "nodes",
        "edges",
        "mode",
        "chunk_size",
        "status",
        "seconds_mean",
        "seconds_min",
        "seconds_max",
        "chunks",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: List[Dict[str, Any]], output_dir: Path, plot_format: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        return

    def label(row: Dict[str, Any]) -> str:
        if row["mode"] == "full":
            return "full"
        return f"chunked ({row['chunk_size']} atoms)"

    for metric, ylabel, filename in (
        ("seconds_mean", "Mean inference time (s)", f"scaling_time.{plot_format}"),
        ("peak_reserved_gb", "Peak CUDA reserved memory (GB)", f"scaling_memory.{plot_format}"),
    ):
        plt.figure(figsize=(7, 4.5))
        labels = sorted(set(label(r) for r in ok_rows))
        for series in labels:
            series_rows = [r for r in ok_rows if label(r) == series]
            series_rows.sort(key=lambda r: int(r["nodes"]))
            x = [int(r["nodes"]) for r in series_rows]
            y = [float(r[metric]) for r in series_rows]
            plt.plot(x, y, marker="o", label=series)
        plt.xscale("log")
        all_y = [float(r[metric]) for r in ok_rows]
        if any(v > 0.0 for v in all_y):
            plt.yscale("log")
        plt.xlabel("Nodes")
        plt.ylabel(ylabel)
        plt.grid(True, which="both", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()


def main(args: Optional[Sequence[str]] = None) -> None:
    parsed = parse_args(args)
    output_dir = Path(parsed.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(parsed.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    config = load_benchmark_config(parsed.config, avg_degree=parsed.avg_degree)
    if parsed.default_dtype:
        config["default_dtype"] = parsed.default_dtype
    if parsed.mixed_precision == "off":
        config["mixed_precision"] = False
    else:
        config["mixed_precision"] = True
        config["mixed_precision_dtype"] = parsed.mixed_precision
    if parsed.dry_run_model:
        config["dry_run"] = True

    apply_global_config(config)
    model, _ = model_from_config(config=config, initialize=False, dataset=None, deploy=False)
    model.to(device)
    model.eval()
    model_parameters = sum(p.numel() for p in model.parameters())

    info = cuda_info(device)
    node_counts = make_node_counts(parsed)
    modes: List[tuple[str, Optional[int]]] = []
    if parsed.modes in ("full", "both"):
        modes.append(("full", None))
    if parsed.modes in ("chunked", "both"):
        for chunk_size in parsed.chunk_batch_max_atoms:
            modes.append(("chunked", int(chunk_size)))

    rows: List[Dict[str, Any]] = []
    full_oom = False
    for n_nodes in node_counts:
        batch = make_synthetic_batch(
            config=config,
            n_nodes=int(n_nodes),
            avg_degree=int(parsed.avg_degree),
            seed=int(parsed.seed) + int(n_nodes),
        )
        n_edges = int(batch[AtomicDataDict.EDGE_INDEX_KEY].shape[1])
        for mode, chunk_size in modes:
            if full_oom and parsed.stop_on_oom and mode == "full":
                continue
            result = benchmark_case(
                model=model,
                batch=batch,
                device=device,
                base_config=config,
                mode=mode,
                chunk_size=chunk_size,
                warmup=parsed.warmup,
                repeats=parsed.repeats,
            )
            row = {
                "nodes": int(n_nodes),
                "edges": n_edges,
                "mode": mode,
                "chunk_size": "" if chunk_size is None else int(chunk_size),
                **result,
            }
            rows.append(row)
            write_csv(output_dir / "scaling_results.csv", rows)
            if mode == "full" and result["status"] == "oom":
                full_oom = True
        del batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_dir / "scaling_results.csv", rows)
    write_report(
        output_dir / "scaling_report.txt",
        info,
        parsed,
        rows,
        model_parameters=model_parameters,
    )
    plot_results(rows, output_dir=output_dir, plot_format=parsed.plot_format)

    print(f"Wrote scaling benchmark results to {output_dir}")


if __name__ == "__main__":
    main()
