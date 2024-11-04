import gc
import json
import os
import platform
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
import torch.nn as nn
from tabulate import tabulate

# Set environment variables for deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    )
)
from grasp_ldm.models.modules.ext.pvcnn.pointnet2 import PointNet2SSG as PointNet2
from grasp_ldm.models.modules.ext.pvcnn.pvcnn_base import PVCNN, PVCNN2


class BenchmarkMetrics(NamedTuple):
    """Container for comprehensive benchmark measurements."""

    peak_memory_mb: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_samples_per_sec: float
    model_parameters_mb: float
    workspace_size_mb: float


@contextmanager
def benchmark_context():
    """Context manager for clean benchmark environment."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()


def measure_inference_time(
    model: nn.Module,
    input_data: torch.Tensor,
    num_iterations: int = 100,
    warmup_iterations: int = 20,
) -> List[float]:
    """Measure accurate inference time with proper warmup and synchronization."""
    times = []
    model.eval()

    # Warmup runs
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _ = model(input_data)
            torch.cuda.synchronize()

    # Timed runs
    with torch.inference_mode():
        for _ in range(num_iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = model(input_data)
            end_event.record()

            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))

    return times


def benchmark_model(
    model: nn.Module,
    batch_sizes: List[int],
    num_points: int,
    in_channels: int,
    num_iterations: int = 100,
) -> Dict[int, BenchmarkMetrics]:
    """Run comprehensive benchmark for given model across batch sizes."""
    results = {}

    for batch_size in batch_sizes:
        with benchmark_context():
            # Generate input data
            input_data = torch.randn(
                batch_size, in_channels, num_points, device="cuda", dtype=torch.float32
            )

            # Measure model parameters
            param_size = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / (1024 * 1024)

            # Memory measurement
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                _ = model(input_data)
                torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

            # Latency measurement
            times = measure_inference_time(model, input_data, num_iterations)
            times = np.array(times)

            # Calculate metrics
            avg_latency = np.mean(times)
            p95_latency = np.percentile(times, 95)
            p99_latency = np.percentile(times, 99)
            throughput = batch_size * 1000 / avg_latency  # samples/second

            workspace_size = (
                torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
            ) / (1024 * 1024)

            results[batch_size] = BenchmarkMetrics(
                peak_memory_mb=peak_memory,
                avg_latency_ms=avg_latency,
                p95_latency_ms=p95_latency,
                p99_latency_ms=p99_latency,
                throughput_samples_per_sec=throughput,
                model_parameters_mb=param_size,
                workspace_size_mb=workspace_size,
            )

    return results


def format_table_data(table_data, headers, fmt="pipe"):
    """Format table data ensuring all values are strings."""
    # Convert all values to strings
    formatted_data = [[str(cell) for cell in row] for row in table_data]
    return tabulate(formatted_data, headers=headers, tablefmt=fmt)


def export_benchmark_report(
    benchmark_results: Dict[str, Dict],
    output_dir: str = "benchmark_results",
    reports_dir: str = "reports",
    reference_model: str = None,
) -> None:
    """Generate and export comprehensive benchmark report with plots."""
    reports_dir = os.path.join(output_dir, reports_dir)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate plots first
    generate_comparison_plots(benchmark_results, output_dir)

    # Export data to CSV
    records = []
    for model_name, batch_results in benchmark_results.items():
        for batch_size, metrics in batch_results.items():
            record = {
                "model": model_name,
                "batch_size": batch_size,
                "avg_latency_ms": metrics.avg_latency_ms,
                "p95_latency_ms": metrics.p95_latency_ms,
                "p99_latency_ms": metrics.p99_latency_ms,
                "throughput_samples_per_sec": metrics.throughput_samples_per_sec,
                "peak_memory_mb": metrics.peak_memory_mb,
                "model_parameters_mb": metrics.model_parameters_mb,
            }
            records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(f"{reports_dir}/benchmark_data.csv", index=False)

    # Generate report content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_names = list(benchmark_results.keys())
    if reference_model is None:
        reference_model = model_names[0]

    report_parts = [
        "# Point Cloud Model Performance Comparison Report",
        f"Generated: {timestamp}",
        "",
        "## System Information",
        "",
        f"- Python: {platform.python_version()}",
        f"- PyTorch: {torch.__version__}",
        f"- CUDA: {torch.version.cuda}",
        f"- GPU: {torch.cuda.get_device_name()}",
        f"- Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB",
        "",
        "## Performance Visualization",
        "",
        "### Latency Analysis",
        "![Latency Comparison](../benchmark_results/avg_latency_ms_comparison.png)",
        "",
        "### Throughput Analysis",
        "![Throughput Comparison](../benchmark_results/throughput_samples_per_sec_comparison.png)",
        "",
        "### Memory Usage Analysis",
        "![Memory Usage](../benchmark_results/peak_memory_mb_comparison.png)",
        "",
        "### Model Comparisons",
        "![Latency Ratio](../benchmark_results/avg_latency_ms_heatmap.png)",
        "![Memory Ratio](../benchmark_results/peak_memory_mb_heatmap.png)",
        "",
        "## Detailed Performance Metrics",
        "",
    ]

    # Create detailed results table
    headers = [
        "Batch Size",
        "Model",
        "Avg Latency (ms)",
        "P95 Latency (ms)",
        "P99 Latency (ms)",
        "Throughput (samples/s)",
        "Peak Memory (MB)",
        "Relative Speedup",
        "Memory Ratio",
        "Parameters (M)",
    ]

    table_data = []
    batch_sizes = sorted(next(iter(benchmark_results.values())).keys())

    for batch_size in batch_sizes:
        ref_metrics = benchmark_results[reference_model][batch_size]
        for model_name, model_results in benchmark_results.items():
            metrics = model_results[batch_size]
            speedup = ref_metrics.avg_latency_ms / metrics.avg_latency_ms
            memory_ratio = metrics.peak_memory_mb / ref_metrics.peak_memory_mb

            row = [
                str(batch_size),  # Convert to string
                model_name,
                f"{metrics.avg_latency_ms:.2f}",
                f"{metrics.p95_latency_ms:.2f}",
                f"{metrics.p99_latency_ms:.2f}",
                f"{metrics.throughput_samples_per_sec:.1f}",
                f"{metrics.peak_memory_mb:.1f}",
                f"{speedup:.2f}x" if model_name != reference_model else "1.00x",
                f"{memory_ratio:.2f}x" if model_name != reference_model else "1.00x",
                f"{metrics.model_parameters_mb:.1f}",
            ]
            table_data.append(row)

    report_parts.extend(
        [
            format_table_data(table_data, headers),
            "",
            f"## Overall Comparison (relative to {reference_model})",
            "",
        ]
    )

    # Add model comparisons
    for model_name in model_names:
        if model_name == reference_model:
            continue

        avg_speedup = np.mean(
            [
                benchmark_results[reference_model][bs].avg_latency_ms
                / benchmark_results[model_name][bs].avg_latency_ms
                for bs in batch_sizes
            ]
        )

        avg_memory = np.mean(
            [
                benchmark_results[model_name][bs].peak_memory_mb
                / benchmark_results[reference_model][bs].peak_memory_mb
                for bs in batch_sizes
            ]
        )

        report_parts.extend(
            [
                f"### {model_name} vs {reference_model}",
                f"- Average Speedup: {avg_speedup:.2f}x",
                f"- Average Memory Ratio: {avg_memory:.2f}x",
                "",
                "#### Per-batch Size Analysis",
                "",
            ]
        )

        batch_metrics = [
            [
                str(bs),  # Convert batch size to string
                f"{benchmark_results[reference_model][bs].avg_latency_ms / benchmark_results[model_name][bs].avg_latency_ms:.2f}",
                f"{benchmark_results[model_name][bs].peak_memory_mb / benchmark_results[reference_model][bs].peak_memory_mb:.2f}",
            ]
            for bs in batch_sizes
        ]

        batch_headers = ["Batch Size", "Speedup", "Memory Ratio"]
        report_parts.append(format_table_data(batch_metrics, batch_headers))
        report_parts.append("")

    # Add recommendations
    report_parts.extend(["## Recommendations", ""])

    for model_name in model_names:
        if model_name == reference_model:
            continue

        avg_speedup = np.mean(
            [
                benchmark_results[reference_model][bs].avg_latency_ms
                / benchmark_results[model_name][bs].avg_latency_ms
                for bs in batch_sizes
            ]
        )

        avg_memory = np.mean(
            [
                benchmark_results[model_name][bs].peak_memory_mb
                / benchmark_results[reference_model][bs].peak_memory_mb
                for bs in batch_sizes
            ]
        )

        report_parts.extend([f"### {model_name} vs {reference_model}", ""])

        if avg_speedup > 1.1:
            report_parts.append(
                f"✓ {model_name} offers better performance ({avg_speedup:.2f}x faster)"
            )
        elif avg_speedup < 0.9:
            report_parts.append(
                f"✗ {model_name} is slower ({1/avg_speedup:.2f}x slower)"
            )
        else:
            report_parts.append("○ Performance is comparable (within 10%)")

        if avg_memory < 0.9:
            report_parts.append(
                f"✓ {model_name} is more memory efficient ({1/avg_memory:.2f}x less memory)"
            )
        elif avg_memory > 1.1:
            report_parts.append(
                f"✗ {model_name} uses more memory ({avg_memory:.2f}x more)"
            )
        else:
            report_parts.append("○ Memory usage is comparable (within 10%)")

        speedups = [
            benchmark_results[reference_model][bs].avg_latency_ms
            / benchmark_results[model_name][bs].avg_latency_ms
            for bs in batch_sizes
        ]
        best_batch_idx = np.argmax(speedups)
        report_parts.append(
            f"➤ Optimal batch size: {batch_sizes[best_batch_idx]} "
            f"(speedup: {speedups[best_batch_idx]:.2f}x)"
        )
        report_parts.append("")

    # Save report
    with open(f"{reports_dir}/benchmark_report.md", "w") as f:
        f.write("\n".join(report_parts))


def generate_comparison_plots(
    benchmark_results: Dict[str, Dict], output_dir: str = "benchmark_results"
):
    """Generate comparison plots for multiple models.

    Args:
        benchmark_results: Dictionary mapping model names to their benchmark results
        output_dir: Directory to save the plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_names = list(benchmark_results.keys())
    batch_sizes = sorted(next(iter(benchmark_results.values())).keys())
    metrics = ["avg_latency_ms", "throughput_samples_per_sec", "peak_memory_mb"]

    # Set style for better visualization
    plt.style.use("seaborn")
    plt.tick_params(axis="both", labelsize=16)

    for metric in metrics:
        plt.figure(figsize=(12, 7))

        for model_name in model_names:
            values = [
                getattr(benchmark_results[model_name][bs], metric) for bs in batch_sizes
            ]
            plt.plot(
                batch_sizes, values, "o-", label=model_name, linewidth=2, markersize=8
            )

        plt.xlabel("Batch Size", fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.title(
            f'{metric.replace("_", " ").title()} vs Batch Size', fontsize=14, pad=20
        )
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=10)
        plt.xscale("log", base=2)

        if "memory" in metric:
            plt.yscale("log", base=2)

        # Add value labels
        for model_name in model_names:
            values = [
                getattr(benchmark_results[model_name][bs], metric) for bs in batch_sizes
            ]
            # for x, y in zip(batch_sizes, values):
            #     plt.annotate(
            #         f"{y:.1f}",
            #         (x, y),
            #         textcoords="offset points",
            #         xytext=(0, 10),
            #         ha="center",
            #         fontsize=12,  # increase from 8
            #     )

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/{metric}_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # Create comparison heatmap
    reference_model = model_names[0]
    other_models = model_names[1:]

    if other_models:  # Only create heatmap if there are models to compare
        metrics_for_heatmap = ["avg_latency_ms", "peak_memory_mb"]

        for metric in metrics_for_heatmap:
            ratios = np.zeros((len(other_models), len(batch_sizes)))

            for i, model_name in enumerate(other_models):
                for j, bs in enumerate(batch_sizes):
                    ref_value = getattr(benchmark_results[reference_model][bs], metric)
                    model_value = getattr(benchmark_results[model_name][bs], metric)
                    ratios[i, j] = model_value / ref_value

            plt.figure(figsize=(12, len(other_models) * 1.5))
            sns.heatmap(
                ratios,
                annot=True,
                fmt=".2f",
                cmap="RdYlBu_r",
                xticklabels=batch_sizes,
                yticklabels=other_models,
                center=1.0,
                vmin=0.5,
                vmax=2.0,
                annot_kws={"size": 14},  # add this line
            )

            plt.xlabel("Batch Size", fontsize=16)  # increase from 12
            plt.ylabel(
                metric.replace("_", " ").title(), fontsize=16
            )  # increase from 12
            plt.title(
                f'{metric.replace("_", " ").title()} vs Batch Size', fontsize=20, pad=20
            )  # increase from 14
            plt.legend(fontsize=14)  # increase from 10
            plt.tight_layout()

            plt.savefig(
                f"{output_dir}/{metric}_heatmap.png", dpi=300, bbox_inches="tight"
            )
            plt.close()


def main():
    """Run the benchmark comparison."""
    batch_sizes = [1, 4, 16, 64, 256]
    num_points = 1024
    in_channels = 3
    scale_channels = 0.5
    scale_voxel_resolution = 0.5

    # Initialize models
    pvcnn = PVCNN(
        in_channels=in_channels,
        extra_feature_channels=0,
        scale_channels=scale_channels,
        scale_voxel_resolution=scale_voxel_resolution,
    ).cuda()
    pvcnn2 = PVCNN2(
        in_channels=in_channels,
        extra_feature_channels=0,
        width_multiplier=scale_channels,
        voxel_resolution_multiplier=scale_voxel_resolution,
    ).cuda()

    pointnet2 = PointNet2(
        # in_channels=in_channels,
        extra_feature_channels=0,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
    ).cuda()

    # Run benchmarks
    print("Running PointNet++ benchmark...")
    pointnet2 = benchmark_model(pointnet2, batch_sizes, num_points, in_channels)

    print("Running PVCNN benchmark...")
    pvcnn_results = benchmark_model(pvcnn, batch_sizes, num_points, in_channels)

    print("Running PVCNN2 benchmark...")
    pvcnn2_results = benchmark_model(pvcnn2, batch_sizes, num_points, in_channels)

    # Generate report and plots
    results_dict = {
        "PVCNN": pvcnn_results,
        "PVCNN2": pvcnn2_results,
        "PointNet2": pointnet2,
    }
    output_dir = "doc/pc_encoder_benchmark_results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report = export_benchmark_report(results_dict, output_dir=output_dir)
    generate_comparison_plots(results_dict)

    print(f"\nBenchmark complete. Report saved to {output_dir}")
    return report


if __name__ == "__main__":
    main()
