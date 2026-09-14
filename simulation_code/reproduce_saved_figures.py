"""Rebuild checked-in study figures from saved CSV outputs.

This script is the lightweight reproducibility entrypoint for the project.
It does not rerun the expensive simulation studies. Instead, it reads the
saved summary tables already committed to the repository and regenerates the
publication figures into a separate output root.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
from typing import Callable

import pandas as pd

import aws_gds_arm as weighted
import aws_gds_arm_adaptive as adaptive
import aws_gds_arm_adaptive_sampling as sampling
import aws_gds_arm_common as common
import mixed_gds_arm as core


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reproduced_from_saved_results"
SUPPORTED_STUDIES = ("baseline", "weighted", "adaptive", "sampling", "methodology")
VERSIONED_DISTRIBUTIONS = (
    "clarabel",
    "cvxpy",
    "highspy",
    "matplotlib",
    "mizani",
    "numpy",
    "osqp",
    "pandas",
    "plotnine",
    "scikit-learn",
    "scipy",
    "scs",
    "statsmodels",
    "tqdm",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required saved table was not found: {path}")
    return pd.read_csv(path)


def _save_plot(plot_obj: object, target: Path) -> Path:
    return core._save_plotnine_plot(plot_obj, target)


def _build_environment_snapshot() -> dict[str, object]:
    versions: dict[str, str] = {}
    for distribution_name in VERSIONED_DISTRIBUTIONS:
        versions[distribution_name] = importlib.metadata.version(distribution_name)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "direct_dependency_versions": versions,
    }


def _rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _baseline_inputs() -> list[Path]:
    tables_dir = PROJECT_ROOT / "study_outputs" / "tables"
    return [
        PROJECT_ROOT / "study_outputs" / "run_settings.json",
        tables_dir / "design_diagnostics.csv",
        tables_dir / "nint_summary_results.csv",
        tables_dir / "ntop_pkeep_aggregate.csv",
        tables_dir / "nrep_summary_results.csv",
    ]


def _generate_baseline(output_root: Path) -> tuple[list[Path], list[Path]]:
    tables_dir = PROJECT_ROOT / "study_outputs" / "tables"
    plots_dir = output_root / "study_outputs" / "plots"
    diagnostics = _read_csv(tables_dir / "design_diagnostics.csv")
    nint_summary = _read_csv(tables_dir / "nint_summary_results.csv")
    ntop_pkeep_aggregate = _read_csv(tables_dir / "ntop_pkeep_aggregate.csv")
    nrep_summary = _read_csv(tables_dir / "nrep_summary_results.csv")
    n_lookup = dict(zip(diagnostics["design_id"], diagnostics["n_runs"], strict=False))
    nint_plot_data = core.nint_curve_data(nint_summary, n_lookup=n_lookup)
    nrep_plot_data = core.boxplot_nrep_data(nrep_summary)

    outputs = [
        _save_plot(core.plot_score_vs_nint_plotnine(nint_plot_data), plots_dir / "score_vs_nint_plotnine.png"),
        _save_plot(
            core.plot_ntop_pkeep_scatter_plotnine(ntop_pkeep_aggregate, metric="score"),
            plots_dir / "ntop_pkeep_score_plotnine.png",
        ),
        _save_plot(
            core.plot_ntop_pkeep_scatter_plotnine(ntop_pkeep_aggregate, metric="power"),
            plots_dir / "ntop_pkeep_power_plotnine.png",
        ),
        _save_plot(
            core.plot_nrep_boxplot_plotnine(nrep_plot_data, metric="score"),
            plots_dir / "nrep_score_plotnine.png",
        ),
        _save_plot(
            core.plot_nrep_boxplot_plotnine(nrep_plot_data, metric="power"),
            plots_dir / "nrep_power_plotnine.png",
        ),
    ]
    return _baseline_inputs(), outputs


def _weighted_inputs() -> list[Path]:
    tables_dir = PROJECT_ROOT / "study_outputs_aws_weighted" / "weighted_tables"
    return [
        PROJECT_ROOT / "study_outputs_aws_weighted" / "weighted_run_settings.json",
        tables_dir / "weighted_ntop_pkeep_aggregate.csv",
        tables_dir / "weighted_variant_summary_results.csv",
    ]


def _generate_weighted(output_root: Path) -> tuple[list[Path], list[Path]]:
    tables_dir = PROJECT_ROOT / "study_outputs_aws_weighted" / "weighted_tables"
    plots_dir = output_root / "study_outputs_aws_weighted" / "weighted_plots"
    ntop_pkeep_aggregate = _read_csv(tables_dir / "weighted_ntop_pkeep_aggregate.csv")
    benchmark_summary = _read_csv(tables_dir / "weighted_variant_summary_results.csv")

    outputs = [
        _save_plot(
            common.plot_ntop_pkeep_scatter_plotnine(
                ntop_pkeep_aggregate,
                metric="score",
                title="Weighted-Only Mean vs SD for (ntop, pkeep): score",
                subtitle="Grid search over the weighted aggregation operating point.",
            ),
            plots_dir / "weighted_ntop_pkeep_score_plotnine.png",
        ),
        _save_plot(
            common.plot_ntop_pkeep_scatter_plotnine(
                ntop_pkeep_aggregate,
                metric="power",
                title="Weighted-Only Mean vs SD for (ntop, pkeep): power",
                subtitle="Grid search over the weighted aggregation operating point.",
            ),
            plots_dir / "weighted_ntop_pkeep_power_plotnine.png",
        ),
        _save_plot(
            weighted.plot_weighted_metric_by_scenario_plotnine(benchmark_summary, metric="score"),
            plots_dir / "weighted_score_by_scenario_plotnine.png",
        ),
        _save_plot(
            weighted.plot_weighted_metric_by_scenario_plotnine(benchmark_summary, metric="power"),
            plots_dir / "weighted_power_by_scenario_plotnine.png",
        ),
        _save_plot(
            weighted.plot_weighted_condition_scatter_plotnine(benchmark_summary, metric="score"),
            plots_dir / "weighted_score_scatter_plotnine.png",
        ),
        _save_plot(
            weighted.plot_weighted_runtime_by_scenario_plotnine(benchmark_summary),
            plots_dir / "weighted_runtime_by_scenario_plotnine.png",
        ),
    ]
    return _weighted_inputs(), outputs


def _adaptive_inputs() -> list[Path]:
    tables_dir = PROJECT_ROOT / "study_outputs_aws_adaptive" / "adaptive_tables"
    return [
        PROJECT_ROOT / "study_outputs_aws_adaptive" / "adaptive_run_settings.json",
        tables_dir / "adaptive_ntop_pkeep_aggregate.csv",
        tables_dir / "adaptive_variant_summary_results.csv",
    ]


def _generate_adaptive(output_root: Path) -> tuple[list[Path], list[Path]]:
    tables_dir = PROJECT_ROOT / "study_outputs_aws_adaptive" / "adaptive_tables"
    plots_dir = output_root / "study_outputs_aws_adaptive" / "adaptive_plots"
    ntop_pkeep_aggregate = _read_csv(tables_dir / "adaptive_ntop_pkeep_aggregate.csv")
    benchmark_summary = _read_csv(tables_dir / "adaptive_variant_summary_results.csv")

    outputs = [
        _save_plot(
            adaptive.plot_adaptive_ntop_pkeep_scatter_plotnine(ntop_pkeep_aggregate, metric="score"),
            plots_dir / "adaptive_ntop_pkeep_score_plotnine.png",
        ),
        _save_plot(
            adaptive.plot_adaptive_ntop_pkeep_scatter_plotnine(ntop_pkeep_aggregate, metric="power"),
            plots_dir / "adaptive_ntop_pkeep_power_plotnine.png",
        ),
        _save_plot(
            adaptive.plot_adaptive_metric_by_scenario_plotnine(benchmark_summary, metric="score"),
            plots_dir / "adaptive_score_by_scenario_plotnine.png",
        ),
        _save_plot(
            adaptive.plot_adaptive_metric_by_scenario_plotnine(benchmark_summary, metric="power"),
            plots_dir / "adaptive_power_by_scenario_plotnine.png",
        ),
        _save_plot(
            adaptive.plot_adaptive_condition_scatter_plotnine(benchmark_summary, metric="score"),
            plots_dir / "adaptive_score_scatter_plotnine.png",
        ),
        _save_plot(
            adaptive.plot_adaptive_runtime_by_scenario_plotnine(benchmark_summary),
            plots_dir / "adaptive_runtime_by_scenario_plotnine.png",
        ),
    ]
    return _adaptive_inputs(), outputs


def _sampling_inputs() -> list[Path]:
    tables_dir = PROJECT_ROOT / "study_outputs_aws_sampling" / "sampling_tables"
    return [
        PROJECT_ROOT / "study_outputs_aws_sampling" / "sampling_run_settings.json",
        tables_dir / "sampling_ntop_pkeep_aggregate.csv",
        tables_dir / "sampling_variant_summary_results.csv",
    ]


def _generate_sampling(output_root: Path) -> tuple[list[Path], list[Path]]:
    tables_dir = PROJECT_ROOT / "study_outputs_aws_sampling" / "sampling_tables"
    plots_dir = output_root / "study_outputs_aws_sampling" / "sampling_plots"
    ntop_pkeep_aggregate = _read_csv(tables_dir / "sampling_ntop_pkeep_aggregate.csv")
    benchmark_summary = _read_csv(tables_dir / "sampling_variant_summary_results.csv")

    outputs = [
        _save_plot(
            sampling.plot_sampling_ntop_pkeep_scatter_plotnine(ntop_pkeep_aggregate, metric="score"),
            plots_dir / "sampling_ntop_pkeep_score_plotnine.png",
        ),
        _save_plot(
            sampling.plot_sampling_ntop_pkeep_scatter_plotnine(ntop_pkeep_aggregate, metric="power"),
            plots_dir / "sampling_ntop_pkeep_power_plotnine.png",
        ),
        _save_plot(
            sampling.plot_sampling_metric_by_scenario_plotnine(benchmark_summary, metric="score"),
            plots_dir / "sampling_score_by_scenario_plotnine.png",
        ),
        _save_plot(
            sampling.plot_sampling_metric_by_scenario_plotnine(benchmark_summary, metric="power"),
            plots_dir / "sampling_power_by_scenario_plotnine.png",
        ),
        _save_plot(
            sampling.plot_sampling_condition_scatter_plotnine(benchmark_summary, metric="score"),
            plots_dir / "sampling_score_scatter_plotnine.png",
        ),
        _save_plot(
            sampling.plot_sampling_runtime_by_scenario_plotnine(benchmark_summary),
            plots_dir / "sampling_runtime_by_scenario_plotnine.png",
        ),
    ]
    return _sampling_inputs(), outputs


def _methodology_inputs() -> list[Path]:
    tables_dir = PROJECT_ROOT / "study_outputs_aws_methodology" / "methodology_tables"
    return [
        PROJECT_ROOT / "study_outputs_aws_methodology" / "methodology_run_settings.json",
        tables_dir / "methodology_benchmark_summary_results.csv",
        tables_dir / "methodology_overall_summary.csv",
        tables_dir / "methodology_delta_table.csv",
        tables_dir / "methodology_win_table.csv",
    ]


def _generate_methodology(output_root: Path) -> tuple[list[Path], list[Path]]:
    tables_dir = PROJECT_ROOT / "study_outputs_aws_methodology" / "methodology_tables"
    plots_dir = output_root / "study_outputs_aws_methodology" / "methodology_plots"
    benchmark_summary = _read_csv(tables_dir / "methodology_benchmark_summary_results.csv")
    overall_summary = _read_csv(tables_dir / "methodology_overall_summary.csv")
    delta_table = _read_csv(tables_dir / "methodology_delta_table.csv")
    win_table = _read_csv(tables_dir / "methodology_win_table.csv")

    outputs = [
        _save_plot(
            common.plot_metric_by_scenario_plotnine(
                benchmark_summary,
                metric="score",
                title="Methodology Comparison: Mean Score by Scenario",
                subtitle="All four methodology stages evaluated on the same benchmark conditions.",
            ),
            plots_dir / "methodology_score_by_scenario_plotnine.png",
        ),
        _save_plot(
            common.plot_metric_by_scenario_plotnine(
                benchmark_summary,
                metric="power",
                title="Methodology Comparison: Mean Power by Scenario",
                subtitle="All four methodology stages evaluated on the same benchmark conditions.",
            ),
            plots_dir / "methodology_power_by_scenario_plotnine.png",
        ),
        _save_plot(
            common.plot_runtime_by_scenario_plotnine(
                benchmark_summary,
                title="Methodology Comparison: Mean Runtime by Scenario",
                subtitle="Lower runtime is better; all methods use the same design and scenario suite.",
            ),
            plots_dir / "methodology_runtime_by_scenario_plotnine.png",
        ),
        _save_plot(
            common.plot_condition_scatter_plotnine(
                benchmark_summary,
                metric="score",
                title="Methodology Comparison: Score Stability by Condition",
                subtitle="Higher mean score and lower SD indicate more reliable performance.",
            ),
            plots_dir / "methodology_score_scatter_plotnine.png",
        ),
        _save_plot(
            common.plot_methodology_runtime_score_frontier_plotnine(benchmark_summary),
            plots_dir / "methodology_runtime_score_frontier_plotnine.png",
        ),
        _save_plot(
            common.plot_methodology_power_error_frontier_plotnine(benchmark_summary),
            plots_dir / "methodology_power_error_frontier_plotnine.png",
        ),
        _save_plot(
            common.plot_methodology_speedup_score_delta_plotnine(delta_table),
            plots_dir / "methodology_speedup_score_delta_plotnine.png",
        ),
        _save_plot(
            common.plot_methodology_metric_bars_plotnine(overall_summary),
            plots_dir / "methodology_metric_profile_plotnine.png",
        ),
        _save_plot(
            common.plot_methodology_win_heatmap_plotnine(
                win_table,
                metric="score_win_rate",
                title="Methodology Score Win Rates by Design",
                subtitle="Share of benchmark scenarios won on score within each design.",
            ),
            plots_dir / "methodology_score_win_heatmap_plotnine.png",
        ),
        _save_plot(
            common.plot_methodology_win_heatmap_plotnine(
                win_table,
                metric="runtime_win_rate",
                title="Methodology Runtime Win Rates by Design",
                subtitle="Share of benchmark scenarios won on runtime within each design.",
            ),
            plots_dir / "methodology_runtime_win_heatmap_plotnine.png",
        ),
        _save_plot(
            common.plot_methodology_win_heatmap_plotnine(
                win_table,
                metric="power_win_rate",
                title="Methodology Power Win Rates by Design",
                subtitle="Share of benchmark scenarios won on power within each design.",
            ),
            plots_dir / "methodology_power_win_heatmap_plotnine.png",
        ),
    ]
    return _methodology_inputs(), outputs


STUDY_GENERATORS: dict[str, Callable[[Path], tuple[list[Path], list[Path]]]] = {
    "baseline": _generate_baseline,
    "weighted": _generate_weighted,
    "adaptive": _generate_adaptive,
    "sampling": _generate_sampling,
    "methodology": _generate_methodology,
}


def _normalize_studies(raw_studies: list[str]) -> list[str]:
    requested = [study.strip().lower() for study in raw_studies]
    if not requested or "all" in requested:
        return list(SUPPORTED_STUDIES)
    invalid = sorted(set(requested) - set(SUPPORTED_STUDIES))
    if invalid:
        valid = ", ".join(["all", *SUPPORTED_STUDIES])
        raise ValueError(f"Unsupported study name(s): {invalid}. Valid options: {valid}.")
    return list(dict.fromkeys(requested))


def rebuild_saved_figures(*, studies: list[str], output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "project_root": str(PROJECT_ROOT),
        "output_root": str(output_root),
        "environment": _build_environment_snapshot(),
        "studies": {},
    }

    for study_name in studies:
        inputs, outputs = STUDY_GENERATORS[study_name](output_root)
        manifest["studies"][study_name] = {
            "input_files": [
                {"path": _rel(path), "sha256": _sha256(path)}
                for path in inputs
            ],
            "output_files": [
                {
                    "path": str(path.relative_to(output_root)),
                    "sha256": _sha256(path),
                }
                for path in outputs
            ],
        }

    manifest_path = output_root / "reproduction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the checked-in publication figures from the saved summary tables "
            "without rerunning the expensive simulations."
        )
    )
    parser.add_argument(
        "--study",
        nargs="+",
        default=["all"],
        help=(
            "Study bundles to rebuild. Choose from: all, baseline, weighted, adaptive, "
            "sampling, methodology."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Directory where regenerated figures should be written. The script mirrors the "
            "original study output layout below this root."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    studies = _normalize_studies(args.study)
    output_root = args.output_root.resolve()
    manifest = rebuild_saved_figures(studies=studies, output_root=output_root)

    print("Rebuilt study figures from saved results.")
    print(f"Output root: {output_root}")
    print(f"Studies: {', '.join(studies)}")
    print(f"Manifest: {manifest['manifest_path']}")
    for study_name in studies:
        output_files = manifest["studies"][study_name]["output_files"]
        print(f"- {study_name}: {len(output_files)} figure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
