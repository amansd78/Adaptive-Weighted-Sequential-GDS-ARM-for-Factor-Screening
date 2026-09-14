"""Shared benchmark utilities for AWS-GDS-ARM study modules.

This module centralizes:

- consistent generated-design and paper-scenario resolution
- a standard summary schema for benchmark replication results
- consistent labels and publication-style plot helpers
- cross-method comparison tables and figures
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import mixed_gds_arm as core


METHOD_LABELS: dict[str, str] = {
    "original_baseline": "Original GDS-ARM",
    "weighted_only": "Weighted Only",
    "adaptive_nrep_only": "Adaptive nrep Only",
    "adaptive_sampling_only": "Adaptive Sampling Only",
    "full_adaptive": "Full AWS-GDS-ARM",
}

METHOD_COLORS: dict[str, str] = {
    "original_baseline": core._PALETTE["brick"],
    "weighted_only": core._PALETTE["teal"],
    "adaptive_nrep_only": core._PALETTE["copper"],
    "adaptive_sampling_only": core._PALETTE["sea"],
    "full_adaptive": core._PALETTE["moss"],
}

_STANDARD_REPLICATION_DEFAULTS: dict[str, Any] = {
    "weighted_tau": np.nan,
    "nrep_used": np.nan,
    "stopping_rule": "fixed_nrep",
    "stop_reason": "fixed_nrep",
    "stopped_early": False,
    "n_batches_used": 1,
    "last_score_delta": np.nan,
    "last_set_jaccard": np.nan,
    "stable_batches_achieved": 0,
    "sampling_rule": "uniform_interaction_columns",
    "last_column_sampling_entropy": np.nan,
    "last_block_sampling_entropy": np.nan,
    "last_top_block_effect": "",
    "last_top_block_probability": np.nan,
}


@dataclass(slots=True)
class BenchmarkStudyContext:
    rng: np.random.Generator
    requested_design_ids: tuple[str, ...]
    designs: list[core.MixedLevelDesign]
    diagnostics: pd.DataFrame
    scenario_map: dict[str, list[dict[str, int]]]
    design_registry: dict[str, core.MixedLevelDesign]


def build_benchmark_context(
    *,
    random_state: int,
    simulation_design_ids: Sequence[str],
    scenario_ids_by_design: Mapping[str, Sequence[int | str]] | None,
    generated_design_coding: str,
    generated_design_candidate_pool: int,
    generated_design_restarts: int,
) -> BenchmarkStudyContext:
    rng = np.random.default_rng(random_state)
    requested_design_ids = tuple(dict.fromkeys(str(design_id) for design_id in simulation_design_ids))
    known_design_ids = set(core._GENERATED_DESIGN_SPECS)
    missing_designs = sorted(set(requested_design_ids) - known_design_ids)
    if missing_designs:
        raise ValueError(
            f"Unknown generated simulation_design_ids: {missing_designs}. "
            f"Known generated templates: {sorted(known_design_ids)}"
        )

    design_suite = core.generate_benchmark_design_suite(
        requested_design_ids,
        random_state=int(rng.integers(0, 2**31 - 1)),
        coding=generated_design_coding,
        candidate_pool_size=int(generated_design_candidate_pool),
        n_starts=int(generated_design_restarts),
    )
    by_id = design_suite["by_id"]
    designs = [by_id[design_id] for design_id in requested_design_ids]
    diagnostics = core.design_diagnostic_table(designs)
    scenario_map: dict[str, list[dict[str, int]]] = {}
    for design in designs:
        override_ids = None if scenario_ids_by_design is None else scenario_ids_by_design.get(design.design_id)
        scenario_map[design.design_id] = core.paper_scenarios_for_design(
            design,
            scenario_ids=override_ids,
        )
    return BenchmarkStudyContext(
        rng=rng,
        requested_design_ids=requested_design_ids,
        designs=designs,
        diagnostics=diagnostics,
        scenario_map=scenario_map,
        design_registry={design.design_id: design for design in designs},
    )


def build_benchmark_mask(
    summary_results: pd.DataFrame,
    *,
    benchmark_ntop: int,
    benchmark_pkeep: float,
    benchmark_tau: float,
) -> pd.Series:
    return (
        (summary_results["ntop"] == int(benchmark_ntop))
        & np.isclose(summary_results["pkeep"].to_numpy(dtype=float), float(benchmark_pkeep))
        & np.isclose(summary_results["weighted_tau"].to_numpy(dtype=float), float(benchmark_tau))
    )


def ensure_standard_replication_columns(replication_results: pd.DataFrame) -> pd.DataFrame:
    standardized = replication_results.copy()
    for column_name, default_value in _STANDARD_REPLICATION_DEFAULTS.items():
        if column_name not in standardized.columns:
            standardized[column_name] = default_value
    if "nrep_used" in standardized.columns:
        standardized["nrep_used"] = standardized["nrep_used"].fillna(standardized.get("nrep"))
    if "weighted_tau" in standardized.columns and "pkeep" in standardized.columns:
        standardized["weighted_tau"] = standardized["weighted_tau"].fillna(standardized["pkeep"])
    return standardized


def summarize_benchmark_results(replication_results: pd.DataFrame) -> pd.DataFrame:
    standardized = ensure_standard_replication_columns(replication_results)
    group_columns = [
        "design_id",
        "scenario_id",
        "variant",
        "nrep",
        "nint",
        "ntop",
        "pkeep",
        "weighted_tau",
        "group_stepwise",
        "paper_exact_stepwise",
        "paper_exact_baseline",
        "final_stepwise",
        "aggregation_method",
        "stopping_rule",
        "sampling_rule",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "mean_nrep_used": ("nrep_used", "mean"),
        "sd_nrep_used": ("nrep_used", "std"),
        "mean_n_batches_used": ("n_batches_used", "mean"),
        "stopped_early_rate": ("stopped_early", "mean"),
        "mean_power": ("power", "mean"),
        "sd_power": ("power", "std"),
        "mean_error": ("error", "mean"),
        "sd_error": ("error", "std"),
        "mean_score": ("score", "mean"),
        "sd_score": ("score", "std"),
        "mean_model_size": ("final_model_size", "mean"),
        "mean_BIC": ("final_BIC", "mean"),
        "mean_timing_seconds": ("timing_seconds", "mean"),
        "mean_last_score_delta": ("last_score_delta", "mean"),
        "mean_last_set_jaccard": ("last_set_jaccard", "mean"),
        "mean_last_column_sampling_entropy": ("last_column_sampling_entropy", "mean"),
        "mean_last_block_sampling_entropy": ("last_block_sampling_entropy", "mean"),
        "mean_last_top_block_probability": ("last_top_block_probability", "mean"),
        "n_datasets": ("dataset_rep", "count"),
    }

    summary_results = (
        standardized.groupby(group_columns, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .fillna(
            {
                "sd_nrep_used": 0.0,
                "sd_power": 0.0,
                "sd_error": 0.0,
                "sd_score": 0.0,
                "mean_n_batches_used": 1.0,
                "stopped_early_rate": 0.0,
                "mean_last_score_delta": np.nan,
                "mean_last_set_jaccard": np.nan,
                "mean_last_column_sampling_entropy": np.nan,
                "mean_last_block_sampling_entropy": np.nan,
                "mean_last_top_block_probability": np.nan,
            }
        )
        .sort_values(["design_id", "scenario_id", "variant"], kind="stable")
        .reset_index(drop=True)
    )
    return summary_results


def overall_summary(summary_results: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_results.groupby(["variant"], dropna=False)
        .agg(
            mean_nrep_used=("mean_nrep_used", "mean"),
            mean_n_batches_used=("mean_n_batches_used", "mean"),
            stopped_early_rate=("stopped_early_rate", "mean"),
            mean_score=("mean_score", "mean"),
            sd_score=("mean_score", "std"),
            mean_power=("mean_power", "mean"),
            mean_error=("mean_error", "mean"),
            mean_timing_seconds=("mean_timing_seconds", "mean"),
            mean_model_size=("mean_model_size", "mean"),
            n_conditions=("scenario_id", "count"),
        )
        .reset_index()
        .fillna({"sd_score": 0.0})
        .sort_values("variant", kind="stable")
        .reset_index(drop=True)
    )


def design_summary(summary_results: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_results.groupby(["design_id", "variant"], dropna=False)
        .agg(
            mean_nrep_used=("mean_nrep_used", "mean"),
            stopped_early_rate=("stopped_early_rate", "mean"),
            mean_score=("mean_score", "mean"),
            sd_score=("mean_score", "std"),
            mean_power=("mean_power", "mean"),
            mean_error=("mean_error", "mean"),
            mean_timing_seconds=("mean_timing_seconds", "mean"),
            mean_model_size=("mean_model_size", "mean"),
            n_conditions=("scenario_id", "count"),
        )
        .reset_index()
        .fillna({"sd_score": 0.0})
        .sort_values(["design_id", "variant"], kind="stable")
        .reset_index(drop=True)
    )


def condition_ranking(
    summary_results: pd.DataFrame,
    *,
    variant_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    ranked = prepare_plot_data(summary_results, variant_labels=variant_labels)
    columns = [
        "design_id",
        "design_label",
        "scenario_id",
        "scenario_label",
        "variant",
        "variant_label",
        "mean_nrep_used",
        "stopped_early_rate",
        "mean_score",
        "sd_score",
        "mean_power",
        "mean_error",
        "mean_model_size",
        "mean_timing_seconds",
        "n_datasets",
    ]
    return (
        ranked[columns]
        .sort_values(
            ["mean_score", "mean_power", "mean_timing_seconds"],
            ascending=[False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def variant_comparison(summary_results: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        summary_results.pivot_table(
            index=["design_id", "scenario_id"],
            columns="variant",
            values=[
                "mean_nrep_used",
                "stopped_early_rate",
                "mean_score",
                "mean_power",
                "mean_error",
                "mean_timing_seconds",
                "mean_model_size",
            ],
            aggfunc="first",
        )
        .sort_index(axis=1)
        .reset_index()
    )
    pivot.columns = [
        "_".join(str(level) for level in column if str(level))
        if isinstance(column, tuple)
        else str(column)
        for column in pivot.columns
    ]
    return pivot.sort_values(["design_id", "scenario_id"], kind="stable").reset_index(drop=True)


def prepare_plot_data(
    summary_results: pd.DataFrame,
    *,
    variant_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    plot_data = summary_results.copy()
    label_map = dict(METHOD_LABELS)
    if variant_labels is not None:
        label_map.update({str(key): str(value) for key, value in variant_labels.items()})
    if "design_id" in plot_data.columns:
        plot_data["design_label"] = plot_data["design_id"].map(core._paper_design_label)
    else:
        plot_data["design_label"] = "All designs"

    if "scenario_id" in plot_data.columns:
        plot_data["scenario_label"] = plot_data["scenario_id"].map(core._paper_scenario_label)
        plot_data["scenario_id_int"] = pd.to_numeric(plot_data["scenario_id"], errors="coerce")
    else:
        plot_data["scenario_label"] = "All scenarios"
        plot_data["scenario_id_int"] = np.nan

    if "variant" in plot_data.columns:
        plot_data["variant_label"] = plot_data["variant"].map(label_map).fillna(plot_data["variant"].astype(str))
    else:
        plot_data["variant_label"] = "Method"
    if "design_id" in plot_data.columns and "scenario_id" in plot_data.columns:
        plot_data["condition_label"] = (
            plot_data["design_label"].astype(str) + " | " + plot_data["scenario_label"].astype(str)
        )
    else:
        plot_data["condition_label"] = plot_data["variant_label"].astype(str)
    return plot_data


def build_method_color_map(variants: Sequence[str]) -> dict[str, str]:
    palette_cycle = [
        core._PALETTE["brick"],
        core._PALETTE["teal"],
        core._PALETTE["copper"],
        core._PALETTE["sea"],
        core._PALETTE["moss"],
        core._PALETTE["gold"],
    ]
    color_map: dict[str, str] = {}
    for index, variant in enumerate(dict.fromkeys(str(item) for item in variants)):
        color_map[variant] = METHOD_COLORS.get(variant, palette_cycle[index % len(palette_cycle)])
    return color_map


def _label_color_map(plot_data: pd.DataFrame) -> dict[str, str]:
    color_map = build_method_color_map(plot_data["variant"].astype(str).unique().tolist())
    return {
        str(variant_label): color_map.get(str(variant), core._PALETTE["teal"])
        for variant, variant_label in plot_data[["variant", "variant_label"]].drop_duplicates().itertuples(index=False)
    }


def build_methodology_delta_table(
    summary_results: pd.DataFrame,
    *,
    baseline_variant: str = "original_baseline",
) -> pd.DataFrame:
    indexed = summary_results.set_index(["design_id", "scenario_id", "variant"]).sort_index()
    rows: list[dict[str, Any]] = []
    for (design_id, scenario_id, variant), row in indexed.iterrows():
        if str(variant) == str(baseline_variant):
            continue
        baseline_key = (design_id, scenario_id, baseline_variant)
        if baseline_key not in indexed.index:
            continue
        baseline_row = indexed.loc[baseline_key]
        baseline_time = float(baseline_row["mean_timing_seconds"])
        variant_time = float(row["mean_timing_seconds"])
        rows.append(
            {
                "design_id": design_id,
                "scenario_id": scenario_id,
                "variant": str(variant),
                "baseline_variant": str(baseline_variant),
                "score_delta_vs_baseline": float(row["mean_score"] - baseline_row["mean_score"]),
                "power_delta_vs_baseline": float(row["mean_power"] - baseline_row["mean_power"]),
                "error_delta_vs_baseline": float(row["mean_error"] - baseline_row["mean_error"]),
                "runtime_delta_vs_baseline": float(variant_time - baseline_time),
                "runtime_ratio_vs_baseline": (
                    float(variant_time / baseline_time) if baseline_time > 0.0 else np.nan
                ),
                "runtime_speedup_vs_baseline": (
                    float(baseline_time / variant_time) if variant_time > 0.0 else np.nan
                ),
                "nrep_used_delta_vs_baseline": float(
                    row["mean_nrep_used"] - baseline_row["mean_nrep_used"]
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_methodology_win_table(summary_results: pd.DataFrame) -> pd.DataFrame:
    if summary_results.empty:
        return pd.DataFrame(
            columns=[
                "variant",
                "design_id",
                "score_wins",
                "runtime_wins",
                "power_wins",
                "n_conditions",
                "score_win_rate",
                "runtime_win_rate",
                "power_win_rate",
            ]
        )

    rows: list[dict[str, Any]] = []
    for design_id, design_df in summary_results.groupby("design_id", dropna=False):
        condition_count = int(design_df["scenario_id"].nunique())
        score_winners = (
            design_df.sort_values(
                ["scenario_id", "mean_score", "mean_power", "mean_timing_seconds"],
                ascending=[True, False, False, True],
                kind="stable",
            )
            .groupby("scenario_id", as_index=False)
            .head(1)["variant"]
            .value_counts()
        )
        runtime_winners = (
            design_df.sort_values(
                ["scenario_id", "mean_timing_seconds", "mean_score"],
                ascending=[True, True, False],
                kind="stable",
            )
            .groupby("scenario_id", as_index=False)
            .head(1)["variant"]
            .value_counts()
        )
        power_winners = (
            design_df.sort_values(
                ["scenario_id", "mean_power", "mean_error", "mean_timing_seconds"],
                ascending=[True, False, True, True],
                kind="stable",
            )
            .groupby("scenario_id", as_index=False)
            .head(1)["variant"]
            .value_counts()
        )
        variants = sorted(design_df["variant"].astype(str).unique().tolist())
        for variant in variants:
            rows.append(
                {
                    "variant": variant,
                    "design_id": design_id,
                    "score_wins": int(score_winners.get(variant, 0)),
                    "runtime_wins": int(runtime_winners.get(variant, 0)),
                    "power_wins": int(power_winners.get(variant, 0)),
                    "n_conditions": condition_count,
                    "score_win_rate": float(score_winners.get(variant, 0) / max(condition_count, 1)),
                    "runtime_win_rate": float(runtime_winners.get(variant, 0) / max(condition_count, 1)),
                    "power_win_rate": float(power_winners.get(variant, 0) / max(condition_count, 1)),
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(
        ["design_id", "variant"],
        kind="stable",
    ).reset_index(drop=True)


def _publication_theme(*, figure_size: tuple[float, float] = (11.5, 6.8), legend_position: str = "bottom"):
    from plotnine import element_blank, element_line, element_rect, element_text, theme

    return theme(
        figure_size=figure_size,
        plot_background=element_rect(fill=core._PALETTE["sand"], color=core._PALETTE["sand"]),
        panel_background=element_rect(fill=core._PALETTE["cream"], color=core._PALETTE["cream"]),
        panel_grid_minor=element_blank(),
        panel_grid_major=element_line(color="#d9d0c1", size=0.35),
        strip_background=element_rect(fill=core._PALETTE["sand"], color=core._PALETTE["sand"]),
        strip_text=element_text(color=core._PALETTE["ink"], weight="bold", size=10),
        legend_background=element_rect(fill=core._PALETTE["sand"], color=core._PALETTE["sand"]),
        legend_key=element_rect(fill=core._PALETTE["sand"], color=core._PALETTE["sand"]),
        axis_title=element_text(color=core._PALETTE["ink"], weight="bold", size=11),
        axis_text=element_text(color=core._PALETTE["ink"], size=9),
        plot_title=element_text(color=core._PALETTE["ink"], weight="bold", size=14),
        plot_subtitle=element_text(color=core._PALETTE["ink"], size=10),
        legend_title=element_text(color=core._PALETTE["ink"], weight="bold", size=10),
        legend_text=element_text(color=core._PALETTE["ink"], size=9),
        text=element_text(color=core._PALETTE["ink"]),
        legend_position=legend_position,
    )


def plot_ntop_pkeep_scatter_plotnine(
    data: pd.DataFrame,
    *,
    metric: str = "score",
    title: str,
    subtitle: str,
    variant_labels: Mapping[str, str] | None = None,
):
    from plotnine import (
        aes,
        facet_wrap,
        geom_point,
        ggplot,
        labs,
        scale_color_gradient,
        scale_size_continuous,
        theme_minimal,
    )

    mean_col = "mean_score" if metric == "score" else "mean_power"
    sd_col = "sd_score" if metric == "score" else "sd_power"
    plot_data = data.copy()
    if "variant" in plot_data.columns:
        plot_data = prepare_plot_data(plot_data, variant_labels=variant_labels)
    else:
        plot_data["variant_label"] = "Method"
    plot = (
        ggplot(plot_data, aes(sd_col, mean_col, color="pkeep", size="ntop"))
        + geom_point(alpha=0.88)
        + scale_color_gradient(low=core._PALETTE["teal"], high=core._PALETTE["gold"])
        + scale_size_continuous(range=(3, 10))
        + labs(
            title=title,
            subtitle=f"{subtitle} Color encodes pkeep and point size encodes ntop.",
            x=f"SD of mean {metric}",
            y=f"Mean {metric}",
            color="pkeep",
            size="ntop",
        )
        + theme_minimal()
        + _publication_theme()
    )
    if "variant" in plot_data.columns and plot_data["variant"].nunique() > 1:
        plot += facet_wrap("~variant_label")
    return plot


def plot_metric_by_scenario_plotnine(
    data: pd.DataFrame,
    *,
    metric: str = "score",
    title: str,
    subtitle: str,
    variant_labels: Mapping[str, str] | None = None,
):
    from plotnine import (
        aes,
        facet_wrap,
        geom_line,
        geom_point,
        ggplot,
        labs,
        scale_color_manual,
        theme_minimal,
    )

    metric_col = "mean_score" if metric == "score" else "mean_power"
    plot_data = prepare_plot_data(data, variant_labels=variant_labels)
    plot_data = plot_data.sort_values(["design_id", "variant", "scenario_id_int"], kind="stable")
    label_color_map = _label_color_map(plot_data)
    return (
        ggplot(plot_data, aes("scenario_id_int", metric_col, color="variant_label", group="variant_label"))
        + geom_line(size=1.15)
        + geom_point(size=2.6)
        + facet_wrap("~design_label", scales="free_x")
        + scale_color_manual(values=label_color_map)
        + labs(
            title=title,
            subtitle=subtitle,
            x="Scenario ID",
            y=f"Mean {metric}",
            color="Method",
        )
        + theme_minimal()
        + _publication_theme()
    )


def plot_condition_scatter_plotnine(
    data: pd.DataFrame,
    *,
    metric: str = "score",
    title: str,
    subtitle: str,
    variant_labels: Mapping[str, str] | None = None,
):
    from plotnine import (
        aes,
        facet_wrap,
        geom_point,
        ggplot,
        labs,
        scale_color_manual,
        scale_size_continuous,
        theme_minimal,
    )

    mean_col = "mean_score" if metric == "score" else "mean_power"
    sd_col = "sd_score" if metric == "score" else "sd_power"
    plot_data = prepare_plot_data(data, variant_labels=variant_labels)
    plot_data = plot_data.sort_values(["design_id", "scenario_id_int", "variant_label"], kind="stable")
    label_color_map = _label_color_map(plot_data)
    return (
        ggplot(plot_data, aes(sd_col, mean_col, color="variant_label", size="mean_model_size"))
        + geom_point(alpha=0.84)
        + facet_wrap("~design_label")
        + scale_color_manual(values=label_color_map)
        + scale_size_continuous(range=(3, 10))
        + labs(
            title=title,
            subtitle=f"{subtitle} Facets separate designs; color identifies method and point size shows mean model size.",
            x=f"SD of mean {metric}",
            y=f"Mean {metric}",
            color="Method",
            size="Mean model size",
        )
        + theme_minimal()
        + _publication_theme(figure_size=(12.8, 6.8))
    )


def plot_runtime_by_scenario_plotnine(
    data: pd.DataFrame,
    *,
    title: str,
    subtitle: str,
    variant_labels: Mapping[str, str] | None = None,
):
    from plotnine import (
        aes,
        facet_wrap,
        geom_line,
        geom_point,
        ggplot,
        labs,
        scale_color_manual,
        theme_minimal,
    )

    plot_data = prepare_plot_data(data, variant_labels=variant_labels)
    plot_data = plot_data.sort_values(["design_id", "variant", "scenario_id_int"], kind="stable")
    label_color_map = _label_color_map(plot_data)
    return (
        ggplot(plot_data, aes("scenario_id_int", "mean_timing_seconds", color="variant_label", group="variant_label"))
        + geom_line(size=1.15)
        + geom_point(size=2.6)
        + facet_wrap("~design_label", scales="free_x")
        + scale_color_manual(values=label_color_map)
        + labs(
            title=title,
            subtitle=subtitle,
            x="Scenario ID",
            y="Mean runtime (seconds)",
            color="Method",
        )
        + theme_minimal()
        + _publication_theme()
    )


def plot_methodology_runtime_score_frontier_plotnine(
    data: pd.DataFrame,
    *,
    title: str = "Runtime vs Score Frontier by Methodology",
    subtitle: str = "Each point is a design-scenario benchmark summary.",
    variant_labels: Mapping[str, str] | None = None,
):
    from plotnine import (
        aes,
        geom_point,
        ggplot,
        labs,
        scale_color_manual,
        scale_shape_discrete,
        theme_minimal,
    )

    plot_data = prepare_plot_data(data, variant_labels=variant_labels)
    label_color_map = _label_color_map(plot_data)
    return (
        ggplot(plot_data, aes("mean_timing_seconds", "mean_score", color="variant_label", shape="design_label"))
        + geom_point(size=3.2, alpha=0.88)
        + scale_color_manual(values=label_color_map)
        + scale_shape_discrete()
        + labs(
            title=title,
            subtitle=subtitle,
            x="Mean runtime (seconds)",
            y="Mean score",
            color="Method",
            shape="Design",
        )
        + theme_minimal()
        + _publication_theme()
    )


def plot_methodology_power_error_frontier_plotnine(
    data: pd.DataFrame,
    *,
    title: str = "Power-Error Frontier by Methodology",
    subtitle: str = "Upper-left behavior is better: higher power with lower error.",
    variant_labels: Mapping[str, str] | None = None,
):
    from plotnine import aes, geom_point, ggplot, labs, scale_color_manual, scale_shape_discrete, theme_minimal

    plot_data = prepare_plot_data(data, variant_labels=variant_labels)
    label_color_map = _label_color_map(plot_data)
    return (
        ggplot(plot_data, aes("mean_error", "mean_power", color="variant_label", shape="design_label"))
        + geom_point(size=3.2, alpha=0.88)
        + scale_color_manual(values=label_color_map)
        + scale_shape_discrete()
        + labs(
            title=title,
            subtitle=subtitle,
            x="Mean error",
            y="Mean power",
            color="Method",
            shape="Design",
        )
        + theme_minimal()
        + _publication_theme()
    )


def plot_methodology_speedup_score_delta_plotnine(
    delta_table: pd.DataFrame,
    *,
    title: str = "Runtime Speedup vs Score Change Relative to Original GDS-ARM",
    subtitle: str = "Values above 1 on x and above 0 on y improve runtime and score simultaneously.",
    variant_labels: Mapping[str, str] | None = None,
):
    from plotnine import (
        aes,
        geom_hline,
        geom_point,
        geom_vline,
        ggplot,
        labs,
        scale_color_manual,
        scale_shape_discrete,
        theme_minimal,
    )

    plot_data = prepare_plot_data(delta_table, variant_labels=variant_labels)
    label_color_map = _label_color_map(plot_data)
    return (
        ggplot(
            plot_data,
            aes(
                "runtime_speedup_vs_baseline",
                "score_delta_vs_baseline",
                color="variant_label",
                shape="design_label",
            ),
        )
        + geom_vline(xintercept=1.0, color=core._PALETTE["brick"], linetype="dashed", size=0.8)
        + geom_hline(yintercept=0.0, color=core._PALETTE["brick"], linetype="dashed", size=0.8)
        + geom_point(size=3.2, alpha=0.88)
        + scale_color_manual(values=label_color_map)
        + scale_shape_discrete()
        + labs(
            title=title,
            subtitle=subtitle,
            x="Runtime speedup vs baseline",
            y="Score delta vs baseline",
            color="Method",
            shape="Design",
        )
        + theme_minimal()
        + _publication_theme()
    )


def plot_methodology_metric_bars_plotnine(
    overall_summary_df: pd.DataFrame,
    *,
    title: str = "Methodology Profile Across Core Metrics",
    subtitle: str = "Facets preserve each metric's natural scale.",
    variant_labels: Mapping[str, str] | None = None,
):
    from plotnine import (
        aes,
        facet_wrap,
        geom_col,
        ggplot,
        labs,
        scale_color_manual,
        scale_fill_manual,
        theme_minimal,
    )

    plot_data = overall_summary_df.copy()
    plot_data["variant_label"] = plot_data["variant"].map({**METHOD_LABELS, **(variant_labels or {})}).fillna(
        plot_data["variant"].astype(str)
    )
    long_df = plot_data.melt(
        id_vars=["variant", "variant_label"],
        value_vars=[
            "mean_score",
            "mean_power",
            "mean_error",
            "mean_timing_seconds",
            "mean_model_size",
            "mean_nrep_used",
        ],
        var_name="metric",
        value_name="value",
    )
    metric_labels = {
        "mean_score": "Score",
        "mean_power": "Power",
        "mean_error": "Error",
        "mean_timing_seconds": "Runtime",
        "mean_model_size": "Model Size",
        "mean_nrep_used": "nrep Used",
    }
    long_df["metric_label"] = long_df["metric"].map(metric_labels).fillna(long_df["metric"])
    label_color_map = _label_color_map(plot_data)
    return (
        ggplot(long_df, aes("variant_label", "value", fill="variant_label", color="variant_label"))
        + geom_col(alpha=0.88, width=0.72)
        + facet_wrap("~metric_label", scales="free_y")
        + scale_fill_manual(values=label_color_map)
        + scale_color_manual(values=label_color_map)
        + labs(
            title=title,
            subtitle=subtitle,
            x="Method",
            y="Value",
            fill="Method",
            color="Method",
        )
        + theme_minimal()
        + _publication_theme(figure_size=(12.5, 7.2))
    )


def plot_methodology_win_heatmap_plotnine(
    win_table: pd.DataFrame,
    *,
    metric: str = "score_win_rate",
    title: str = "Method Win Rates by Design",
    subtitle: str = "Win rate is computed across benchmark scenarios within each design.",
    variant_labels: Mapping[str, str] | None = None,
):
    from plotnine import aes, geom_tile, ggplot, labs, scale_fill_gradient, theme_minimal

    plot_data = win_table.copy()
    plot_data["variant_label"] = plot_data["variant"].map({**METHOD_LABELS, **(variant_labels or {})}).fillna(
        plot_data["variant"].astype(str)
    )
    plot_data["design_label"] = plot_data["design_id"].map(core._paper_design_label)
    return (
        ggplot(plot_data, aes("design_label", "variant_label", fill=metric))
        + geom_tile(color=core._PALETTE["sand"], size=0.8)
        + scale_fill_gradient(low=core._PALETTE["cream"], high=core._PALETTE["teal"])
        + labs(
            title=title,
            subtitle=subtitle,
            x="Design",
            y="Method",
            fill=metric.replace("_", " "),
        )
        + theme_minimal()
        + _publication_theme(figure_size=(10.8, 5.8))
    )


__all__ = [
    "BenchmarkStudyContext",
    "METHOD_COLORS",
    "METHOD_LABELS",
    "build_benchmark_context",
    "build_benchmark_mask",
    "build_methodology_delta_table",
    "build_methodology_win_table",
    "condition_ranking",
    "design_summary",
    "ensure_standard_replication_columns",
    "overall_summary",
    "plot_condition_scatter_plotnine",
    "plot_methodology_metric_bars_plotnine",
    "plot_methodology_power_error_frontier_plotnine",
    "plot_methodology_runtime_score_frontier_plotnine",
    "plot_methodology_speedup_score_delta_plotnine",
    "plot_methodology_win_heatmap_plotnine",
    "plot_metric_by_scenario_plotnine",
    "plot_ntop_pkeep_scatter_plotnine",
    "plot_runtime_by_scenario_plotnine",
    "prepare_plot_data",
    "summarize_benchmark_results",
    "variant_comparison",
]
