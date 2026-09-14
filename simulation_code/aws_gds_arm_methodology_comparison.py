"""Unified methodology comparison for the four GDS-ARM study stages.

This study compares, on the same generated designs and paper scenarios:

- ``original_baseline``: original fixed-tuning GDS-ARM aggregation
- ``weighted_only``: BIC-weighted aggregation over fixed-``nrep`` repetitions
- ``adaptive_nrep_only``: weighted aggregation with adaptive sequential stopping
- ``adaptive_sampling_only``: weighted aggregation with adaptive interaction sampling

The module is designed as the final reporting layer for the current four-file
workflow so that tables, design/scenario settings, and publication figures stay
aligned across all methodologies.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import warnings
from typing import Any, Iterable, MutableMapping, Mapping, Sequence

import numpy as np
import pandas as pd

import aws_gds_arm as weighted
import aws_gds_arm_adaptive as adaptive
import aws_gds_arm_adaptive_sampling as sampling
import aws_gds_arm_common as common
import mixed_gds_arm as core


_CORE_DEFAULTS = core.StudyRunSettings()
_ALLOWED_VARIANTS = (
    "original_baseline",
    "weighted_only",
    "adaptive_nrep_only",
    "adaptive_sampling_only",
)
_METHODOLOGY_TABLES_DIRNAME = "methodology_tables"
_METHODOLOGY_PLOTS_DIRNAME = "methodology_plots"
_METHODOLOGY_SETTINGS_FILENAME = "methodology_run_settings.json"
_BASELINE_P_ENTER = 0.01
_BASELINE_P_REMOVE = 0.05


@dataclass(slots=True)
class MethodologyComparisonSettings:
    output_dir: str | os.PathLike[str] = "study_outputs_aws_methodology"
    run_name: str = "methodology_phase1"
    generated_design_coding: str = _CORE_DEFAULTS.generated_design_coding
    generated_design_candidate_pool: int = _CORE_DEFAULTS.generated_design_candidate_pool
    generated_design_restarts: int = _CORE_DEFAULTS.generated_design_restarts
    random_state: int = _CORE_DEFAULTS.random_state
    reps_per_condition: int = _CORE_DEFAULTS.reps_per_condition
    simulation_design_ids: tuple[str, ...] = _CORE_DEFAULTS.simulation_design_ids
    scenario_ids_by_design: dict[str, tuple[int | str, ...]] | None = None
    benchmark_nrep: int = _CORE_DEFAULTS.ntop_pkeep_nrep
    benchmark_nint: int | str = _CORE_DEFAULTS.ntop_pkeep_nint
    benchmark_ntop: int = _CORE_DEFAULTS.nrep_study_ntop
    benchmark_pkeep: float = _CORE_DEFAULTS.nrep_study_pkeep
    weighted_tau: float | None = None
    adaptive_max_nrep: int = _CORE_DEFAULTS.ntop_pkeep_nrep
    adaptive_batch_size: int = 50
    adaptive_min_nrep: int = max(_CORE_DEFAULTS.nrep_study_ntop, 100)
    adaptive_stability_epsilon: float = 0.01
    adaptive_patience: int = 2
    adaptive_require_set_stability: bool = False
    adaptive_min_jaccard: float = 0.95
    adaptive_set_tau: float | None = None
    sampling_batch_size: int = 50
    sampling_warmup_batches: int = 2
    sampling_adaptation_lambda: float = 0.7
    sampling_heredity_eta: float = 0.5
    sampling_block_floor: float = 1e-3
    group_stepwise: bool = False
    paper_exact_stepwise: bool = True
    paper_exact_baseline: bool = True
    stepwise_mode: str = "bidirectional"
    p_enter: float = _BASELINE_P_ENTER
    p_remove: float = _BASELINE_P_REMOVE
    final_stepwise: bool = True
    variants: tuple[str, ...] = _ALLOWED_VARIANTS
    n_jobs: int = -1
    write_replication_tables: bool = True
    write_plots: bool = True
    write_settings_file: bool = True
    show_progress: bool = True


def _settings_payload(settings: MethodologyComparisonSettings) -> dict[str, Any]:
    payload = asdict(settings)
    payload["output_dir"] = str(settings.output_dir)
    return payload


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_manifest(settings: MethodologyComparisonSettings) -> dict[str, Any]:
    manifest = _settings_payload(settings)
    manifest["methodology_source_sha256"] = _source_sha256(Path(__file__))
    manifest["weighted_source_sha256"] = _source_sha256(Path(weighted.__file__))
    manifest["adaptive_source_sha256"] = _source_sha256(Path(adaptive.__file__))
    manifest["sampling_source_sha256"] = _source_sha256(Path(sampling.__file__))
    manifest["core_source_sha256"] = core._source_sha256()
    manifest["cache_version"] = 1
    return json.loads(json.dumps(manifest, sort_keys=True))


def _resolve_weighted_tau(pkeep: float, weighted_tau_override: float | None) -> float:
    return float(pkeep if weighted_tau_override is None else weighted_tau_override)


def _validate_settings(settings: MethodologyComparisonSettings) -> None:
    if int(settings.reps_per_condition) <= 0:
        raise ValueError("reps_per_condition must be positive.")
    if int(settings.benchmark_nrep) <= 0:
        raise ValueError("benchmark_nrep must be positive.")
    if int(settings.adaptive_max_nrep) <= 0:
        raise ValueError("adaptive_max_nrep must be positive.")
    if int(settings.adaptive_batch_size) <= 0:
        raise ValueError("adaptive_batch_size must be positive.")
    if int(settings.adaptive_min_nrep) <= 0:
        raise ValueError("adaptive_min_nrep must be positive.")
    if int(settings.adaptive_patience) <= 0:
        raise ValueError("adaptive_patience must be positive.")
    if float(settings.adaptive_stability_epsilon) < 0.0:
        raise ValueError("adaptive_stability_epsilon must be non-negative.")
    if int(settings.sampling_batch_size) <= 0:
        raise ValueError("sampling_batch_size must be positive.")
    if int(settings.sampling_warmup_batches) < 0:
        raise ValueError("sampling_warmup_batches must be non-negative.")
    if int(settings.benchmark_nrep) < int(settings.benchmark_ntop):
        raise ValueError("benchmark_nrep must be at least benchmark_ntop.")
    if int(settings.adaptive_min_nrep) > int(settings.adaptive_max_nrep):
        raise ValueError("adaptive_min_nrep cannot exceed adaptive_max_nrep.")
    if not 0.0 <= float(settings.sampling_adaptation_lambda) <= 1.0:
        raise ValueError("sampling_adaptation_lambda must lie in [0, 1].")
    if not 0.0 <= float(settings.sampling_heredity_eta) <= 1.0:
        raise ValueError("sampling_heredity_eta must lie in [0, 1].")
    if float(settings.sampling_block_floor) < 0.0:
        raise ValueError("sampling_block_floor must be non-negative.")
    if not 0.0 <= float(settings.adaptive_min_jaccard) <= 1.0:
        raise ValueError("adaptive_min_jaccard must lie in [0, 1].")
    invalid_variants = sorted(set(settings.variants) - set(_ALLOWED_VARIANTS))
    if invalid_variants:
        raise ValueError(
            f"Unsupported variants: {invalid_variants}. Expected any of {_ALLOWED_VARIANTS}."
        )


def _count_threshold_feature_table(
    top_results: pd.DataFrame,
    metadata: core.MixedLevelDesign,
    *,
    pkeep: float,
    paper_exact_baseline: bool,
) -> tuple[pd.DataFrame, list[str], list[str], int]:
    n_top = int(len(top_results))
    threshold_count = int(np.ceil(float(pkeep) * max(n_top, 1)))
    effect_type_lookup = dict(
        zip(metadata.effect_table["effect_id"], metadata.effect_table["effect_type"], strict=False)
    )

    if paper_exact_baseline:
        column_counts: MutableMapping[str, int] = {}
        for selected_columns in top_results["selected_columns"]:
            for column_id in selected_columns:
                column_counts[column_id] = column_counts.get(column_id, 0) + 1

        rows = []
        for column_id in metadata.X_full.columns.tolist():
            count = int(column_counts.get(column_id, 0))
            rows.append(
                {
                    "feature_id": column_id,
                    "aggregation_unit": "column",
                    "count": count,
                    "proportion": count / n_top if n_top else 0.0,
                    "score": float(count),
                    "kept": bool(count >= threshold_count and n_top > 0),
                    "effect_type": metadata.column_to_metadata[column_id]["effect_type"],
                }
            )
        feature_table = pd.DataFrame(rows).sort_values(
            ["kept", "count", "feature_id"],
            ascending=[False, False, True],
        )
        aggregated_columns = feature_table.loc[feature_table["kept"], "feature_id"].tolist()
        aggregated_effects = sorted(
            metadata.column_table[
                metadata.column_table["column_id"].isin(aggregated_columns)
            ]["effect_id"].unique().tolist()
        )
    else:
        effect_counts: MutableMapping[str, int] = {}
        for selected_effects in top_results["selected_effects"]:
            for effect_id in selected_effects:
                effect_counts[effect_id] = effect_counts.get(effect_id, 0) + 1

        rows = []
        for effect_id in metadata.effect_ids:
            count = int(effect_counts.get(effect_id, 0))
            rows.append(
                {
                    "feature_id": effect_id,
                    "aggregation_unit": "effect",
                    "count": count,
                    "proportion": count / n_top if n_top else 0.0,
                    "score": float(count),
                    "kept": bool(count >= threshold_count and n_top > 0),
                    "effect_type": effect_type_lookup[effect_id],
                }
            )
        feature_table = pd.DataFrame(rows).sort_values(
            ["kept", "count", "feature_id"],
            ascending=[False, False, True],
        )
        aggregated_effects = feature_table.loc[feature_table["kept"], "feature_id"].tolist()
        aggregated_columns = metadata.columns_for_effects(aggregated_effects)

    return feature_table, aggregated_effects, aggregated_columns, threshold_count


def _build_original_baseline_result(
    repetition_results: pd.DataFrame,
    metadata: core.MixedLevelDesign,
    y: Sequence[float] | pd.Series | np.ndarray,
    *,
    ntop: int,
    pkeep: float,
    group_stepwise: bool,
    paper_exact_stepwise: bool,
    paper_exact_baseline: bool,
    final_stepwise: bool,
    stepwise_mode: str,
    p_enter: float,
    p_remove: float,
) -> dict[str, Any]:
    ranked_results = repetition_results.sort_values("bic", ascending=True, kind="stable").reset_index(drop=True)
    top_results = ranked_results.head(min(max(int(ntop), 0), len(ranked_results))).copy()
    feature_table, aggregated_effects, aggregated_columns, threshold_count = _count_threshold_feature_table(
        top_results,
        metadata,
        pkeep=float(pkeep),
        paper_exact_baseline=paper_exact_baseline,
    )

    (
        final_stepwise_result,
        final_selected_effects,
        final_selected_columns,
        final_selected_factors,
    ) = weighted._finalize_selection(
        metadata,
        y,
        aggregated_effects=aggregated_effects,
        aggregated_columns=aggregated_columns,
        group_stepwise=group_stepwise,
        paper_exact_stepwise=paper_exact_stepwise,
        final_stepwise=final_stepwise,
        stepwise_mode=stepwise_mode,
        p_enter=p_enter,
        p_remove=p_remove,
    )

    diagnostics = {
        "aggregation_method": "top_count_threshold",
        "variant": "original_baseline",
        "nrep_used": int(len(repetition_results)),
        "pkeep": float(pkeep),
        "threshold_count": int(threshold_count),
        "ntop_used": int(len(top_results)),
        "final_bic": float(final_stepwise_result.fit.bic),
        "final_rss": float(final_stepwise_result.fit.rss),
        "final_model_size": int(len(final_selected_columns)),
        "final_stepwise": bool(final_stepwise),
        "group_stepwise": bool(group_stepwise),
        "paper_exact_stepwise": bool(paper_exact_stepwise),
        "paper_exact_baseline": bool(paper_exact_baseline),
        "stepwise_mode": str(stepwise_mode),
        "p_enter": float(p_enter),
        "p_remove": float(p_remove),
    }
    return {
        "variant": "original_baseline",
        "top_results": top_results,
        "feature_table": feature_table,
        "aggregated_effects": aggregated_effects,
        "aggregated_columns": aggregated_columns,
        "final_stepwise_result": final_stepwise_result,
        "final_selected_effects": final_selected_effects,
        "final_selected_columns": final_selected_columns,
        "final_selected_factors": final_selected_factors,
        "diagnostics": diagnostics,
    }


def _iter_methodology_tasks(
    design_ids: Sequence[str],
    scenario_map: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reps_per_condition: int,
    task_seed: int,
    benchmark_nrep: int,
    benchmark_nint: int | str,
    benchmark_ntop: int,
    benchmark_pkeep: float,
    weighted_tau_override: float | None,
    group_stepwise: bool,
    paper_exact_stepwise: bool,
    paper_exact_baseline: bool,
    final_stepwise: bool,
    stepwise_mode: str,
    p_enter: float,
    p_remove: float,
    variants: Sequence[str],
    adaptive_max_nrep: int,
    adaptive_batch_size: int,
    adaptive_min_nrep: int,
    adaptive_stability_epsilon: float,
    adaptive_patience: int,
    adaptive_require_set_stability: bool,
    adaptive_min_jaccard: float,
    adaptive_set_tau: float,
    sampling_batch_size: int,
    sampling_warmup_batches: int,
    sampling_adaptation_lambda: float,
    sampling_heredity_eta: float,
    sampling_block_floor: float,
) -> Iterable[Mapping[str, Any]]:
    task_rng = np.random.default_rng(int(task_seed))
    task_id = 0
    for design_id in design_ids:
        for scenario in scenario_map[str(design_id)]:
            for dataset_rep in range(1, int(reps_per_condition) + 1):
                yield {
                    "task_id": int(task_id),
                    "design_id": str(design_id),
                    "scenario": dict(scenario),
                    "dataset_rep": int(dataset_rep),
                    "benchmark_nrep": int(benchmark_nrep),
                    "benchmark_nint": benchmark_nint,
                    "benchmark_ntop": int(benchmark_ntop),
                    "benchmark_pkeep": float(benchmark_pkeep),
                    "weighted_tau_override": (
                        None if weighted_tau_override is None else float(weighted_tau_override)
                    ),
                    "group_stepwise": bool(group_stepwise),
                    "paper_exact_stepwise": bool(paper_exact_stepwise),
                    "paper_exact_baseline": bool(paper_exact_baseline),
                    "final_stepwise": bool(final_stepwise),
                    "stepwise_mode": str(stepwise_mode),
                    "p_enter": float(p_enter),
                    "p_remove": float(p_remove),
                    "variants": tuple(str(variant) for variant in variants),
                    "seed": int(task_rng.integers(0, 2**31 - 1)),
                    "adaptive_max_nrep": int(adaptive_max_nrep),
                    "adaptive_batch_size": int(adaptive_batch_size),
                    "adaptive_min_nrep": int(adaptive_min_nrep),
                    "adaptive_stability_epsilon": float(adaptive_stability_epsilon),
                    "adaptive_patience": int(adaptive_patience),
                    "adaptive_require_set_stability": bool(adaptive_require_set_stability),
                    "adaptive_min_jaccard": float(adaptive_min_jaccard),
                    "adaptive_set_tau": float(adaptive_set_tau),
                    "sampling_batch_size": int(sampling_batch_size),
                    "sampling_warmup_batches": int(sampling_warmup_batches),
                    "sampling_adaptation_lambda": float(sampling_adaptation_lambda),
                    "sampling_heredity_eta": float(sampling_heredity_eta),
                    "sampling_block_floor": float(sampling_block_floor),
                }
                task_id += 1


def _run_methodology_replication(
    task: Mapping[str, Any],
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    design = core._get_simulation_design(str(task["design_id"]))
    scenario = dict(task["scenario"])
    dataset_rep = int(task["dataset_rep"])
    rng = np.random.default_rng(int(task["seed"]))
    scenario_id = core._scenario_id_string(scenario)

    dataset = core.generate_synthetic_dataset(
        design,
        scenario,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    current_delta_grid = (
        None
        if bool(task["paper_exact_baseline"])
        else core.derive_delta_grid(design.X_full, dataset["y"])
    )
    resolved_nint = core._resolve_relative_count(task["benchmark_nint"], design.n_runs)
    benchmark_tau = _resolve_weighted_tau(
        float(task["benchmark_pkeep"]),
        task["weighted_tau_override"],
    )
    shared_rep_seed_count = max(int(task["benchmark_nrep"]), int(task["adaptive_max_nrep"]))
    shared_rep_seeds = [
        int(rng.integers(0, 2**31 - 1))
        for _ in range(shared_rep_seed_count)
    ]

    path_results_by_variant: dict[str, dict[str, Any]] = {}

    if "original_baseline" in task["variants"] or "weighted_only" in task["variants"]:
        path_t0 = time.perf_counter()
        uniform_repetition_results = weighted._build_repetition_path(
            design,
            dataset["y"],
            nrep=int(task["benchmark_nrep"]),
            nint=resolved_nint,
            delta_grid=current_delta_grid,
            random_state=None,
            solver_order=None,
            paper_exact_baseline=bool(task["paper_exact_baseline"]),
            rep_seeds=shared_rep_seeds[: int(task["benchmark_nrep"])],
        )
        path_elapsed = time.perf_counter() - path_t0
        uniform_path_diagnostics = {
            "sampling_rule": "uniform_interaction_columns",
            "stopping_rule": "fixed_nrep",
            "stop_reason": "fixed_nrep",
            "nrep_used": int(len(uniform_repetition_results)),
            "stopped_early": False,
            "n_batches_used": 1,
            "last_score_delta": np.nan,
            "last_set_jaccard": np.nan,
            "stable_batches_achieved": 0,
            "last_column_sampling_entropy": 1.0 if design.X_int.shape[1] > 1 else 0.0,
            "last_block_sampling_entropy": 1.0
            if len(sampling._interaction_block_table(design)) > 1
            else 0.0,
            "last_top_block_effect": "",
            "last_top_block_probability": np.nan,
        }
        if "original_baseline" in task["variants"]:
            path_results_by_variant["original_baseline"] = {
                "repetition_results": uniform_repetition_results,
                "path_elapsed": float(path_elapsed),
                "path_diagnostics": uniform_path_diagnostics,
            }
        if "weighted_only" in task["variants"]:
            path_results_by_variant["weighted_only"] = {
                "repetition_results": uniform_repetition_results,
                "path_elapsed": float(path_elapsed),
                "path_diagnostics": uniform_path_diagnostics,
            }

    if "adaptive_nrep_only" in task["variants"]:
        path_t0 = time.perf_counter()
        repetition_results, _stability_history, path_diagnostics = adaptive._build_adaptive_repetition_path(
            design,
            dataset["y"],
            max_nrep=int(task["adaptive_max_nrep"]),
            batch_size=int(task["adaptive_batch_size"]),
            min_nrep=int(task["adaptive_min_nrep"]),
            stability_epsilon=float(task["adaptive_stability_epsilon"]),
            patience=int(task["adaptive_patience"]),
            stability_tau=float(task["adaptive_set_tau"]),
            require_set_stability=bool(task["adaptive_require_set_stability"]),
            min_jaccard=float(task["adaptive_min_jaccard"]),
            nint=resolved_nint,
            delta_grid=current_delta_grid,
            random_state=int(rng.integers(0, 2**31 - 1)),
            solver_order=None,
            paper_exact_baseline=bool(task["paper_exact_baseline"]),
            rep_seeds=shared_rep_seeds[: int(task["adaptive_max_nrep"])],
        )
        path_elapsed = time.perf_counter() - path_t0
        path_diagnostics = {
            **path_diagnostics,
            "sampling_rule": "uniform_interaction_columns",
            "last_column_sampling_entropy": 1.0 if design.X_int.shape[1] > 1 else 0.0,
            "last_block_sampling_entropy": 1.0
            if len(sampling._interaction_block_table(design)) > 1
            else 0.0,
            "last_top_block_effect": "",
            "last_top_block_probability": np.nan,
        }
        path_results_by_variant["adaptive_nrep_only"] = {
            "repetition_results": repetition_results,
            "path_elapsed": float(path_elapsed),
            "path_diagnostics": path_diagnostics,
        }

    if "adaptive_sampling_only" in task["variants"]:
        path_t0 = time.perf_counter()
        repetition_results, _sampling_history, path_diagnostics = sampling._build_adaptive_sampling_repetition_path(
            design,
            dataset["y"],
            nrep=int(task["benchmark_nrep"]),
            nint=resolved_nint,
            batch_size=int(task["sampling_batch_size"]),
            warmup_batches=int(task["sampling_warmup_batches"]),
            adaptation_lambda=float(task["sampling_adaptation_lambda"]),
            heredity_eta=float(task["sampling_heredity_eta"]),
            block_floor=float(task["sampling_block_floor"]),
            delta_grid=current_delta_grid,
            random_state=int(rng.integers(0, 2**31 - 1)),
            solver_order=None,
            paper_exact_baseline=bool(task["paper_exact_baseline"]),
            rep_seeds=shared_rep_seeds[: int(task["benchmark_nrep"])],
        )
        path_elapsed = time.perf_counter() - path_t0
        path_results_by_variant["adaptive_sampling_only"] = {
            "repetition_results": repetition_results,
            "path_elapsed": float(path_elapsed),
            "path_diagnostics": path_diagnostics,
        }

    replication_rows: list[dict[str, Any]] = []
    for variant_idx, variant in enumerate(task["variants"]):
        variant_path = path_results_by_variant[str(variant)]
        variant_t0 = time.perf_counter()
        if str(variant) == "original_baseline":
            variant_result = _build_original_baseline_result(
                variant_path["repetition_results"],
                design,
                dataset["y"],
                ntop=int(task["benchmark_ntop"]),
                pkeep=float(task["benchmark_pkeep"]),
                group_stepwise=bool(task["group_stepwise"]),
                paper_exact_stepwise=bool(task["paper_exact_stepwise"]),
                paper_exact_baseline=bool(task["paper_exact_baseline"]),
                final_stepwise=bool(task["final_stepwise"]),
                stepwise_mode=str(task["stepwise_mode"]),
                p_enter=float(task["p_enter"]),
                p_remove=float(task["p_remove"]),
            )
        elif str(variant) == "weighted_only":
            variant_result = weighted._build_variant_result(
                variant_path["repetition_results"],
                design,
                dataset["y"],
                variant=str(variant),
                ntop=int(task["benchmark_ntop"]),
                pkeep=float(task["benchmark_pkeep"]),
                weighted_tau=float(benchmark_tau),
                group_stepwise=bool(task["group_stepwise"]),
                paper_exact_stepwise=bool(task["paper_exact_stepwise"]),
                paper_exact_baseline=bool(task["paper_exact_baseline"]),
                final_stepwise=bool(task["final_stepwise"]),
                stepwise_mode=str(task["stepwise_mode"]),
                p_enter=float(task["p_enter"]),
                p_remove=float(task["p_remove"]),
            )
        elif str(variant) == "adaptive_nrep_only":
            variant_result = adaptive._build_variant_result(
                variant_path["repetition_results"],
                design,
                dataset["y"],
                variant=str(variant),
                ntop=int(task["benchmark_ntop"]),
                pkeep=float(task["benchmark_pkeep"]),
                weighted_tau=float(benchmark_tau),
                group_stepwise=bool(task["group_stepwise"]),
                paper_exact_stepwise=bool(task["paper_exact_stepwise"]),
                paper_exact_baseline=bool(task["paper_exact_baseline"]),
                final_stepwise=bool(task["final_stepwise"]),
                stepwise_mode=str(task["stepwise_mode"]),
                p_enter=float(task["p_enter"]),
                p_remove=float(task["p_remove"]),
            )
        elif str(variant) == "adaptive_sampling_only":
            variant_result = sampling._build_variant_result(
                variant_path["repetition_results"],
                design,
                dataset["y"],
                variant=str(variant),
                ntop=int(task["benchmark_ntop"]),
                pkeep=float(task["benchmark_pkeep"]),
                weighted_tau=float(benchmark_tau),
                group_stepwise=bool(task["group_stepwise"]),
                paper_exact_stepwise=bool(task["paper_exact_stepwise"]),
                paper_exact_baseline=bool(task["paper_exact_baseline"]),
                final_stepwise=bool(task["final_stepwise"]),
                stepwise_mode=str(task["stepwise_mode"]),
                p_enter=float(task["p_enter"]),
                p_remove=float(task["p_remove"]),
            )
        else:
            raise ValueError(f"Unknown methodology variant: {variant}")

        variant_elapsed = time.perf_counter() - variant_t0
        metrics = core._factor_metrics(
            selected_factors=variant_result["final_selected_factors"],
            true_important_factors=dataset["true_important_factors"],
            all_factors=design.factor_ids,
        )
        diagnostics = {
            **variant_result["diagnostics"],
            **variant_path["path_diagnostics"],
        }
        replication_rows.append(
            {
                "task_id": int(task["task_id"]),
                "variant_order": int(variant_idx),
                "design_id": design.design_id,
                "scenario_id": scenario_id,
                "dataset_rep": dataset_rep,
                "variant": str(variant),
                "nrep": (
                    int(task["benchmark_nrep"])
                    if str(variant) != "adaptive_nrep_only"
                    else int(task["adaptive_max_nrep"])
                ),
                "nrep_used": int(diagnostics["nrep_used"]),
                "nint": int(resolved_nint),
                "ntop": int(task["benchmark_ntop"]),
                "pkeep": float(task["benchmark_pkeep"]),
                "weighted_tau": float(benchmark_tau),
                "group_stepwise": bool(task["group_stepwise"]),
                "paper_exact_stepwise": bool(task["paper_exact_stepwise"]),
                "paper_exact_baseline": bool(task["paper_exact_baseline"]),
                "final_stepwise": bool(task["final_stepwise"]),
                "TP": metrics["TP"],
                "FP": metrics["FP"],
                "power": metrics["power"],
                "error": metrics["error"],
                "score": metrics["score"],
                "final_model_size": int(len(variant_result["final_selected_columns"])),
                "final_BIC": float(diagnostics["final_bic"]),
                "timing_seconds": float(variant_path["path_elapsed"] + variant_elapsed),
                "path_timing_seconds": float(variant_path["path_elapsed"]),
                "variant_timing_seconds": float(variant_elapsed),
                "aggregation_method": str(diagnostics["aggregation_method"]),
                "sampling_rule": str(diagnostics["sampling_rule"]),
                "stopping_rule": str(diagnostics["stopping_rule"]),
                "stop_reason": str(diagnostics["stop_reason"]),
                "stopped_early": bool(diagnostics["stopped_early"]),
                "n_batches_used": int(diagnostics["n_batches_used"]),
                "last_score_delta": float(diagnostics["last_score_delta"]),
                "last_set_jaccard": float(diagnostics["last_set_jaccard"]),
                "stable_batches_achieved": int(diagnostics["stable_batches_achieved"]),
                "last_column_sampling_entropy": float(diagnostics["last_column_sampling_entropy"]),
                "last_block_sampling_entropy": float(diagnostics["last_block_sampling_entropy"]),
                "last_top_block_effect": str(diagnostics["last_top_block_effect"]),
                "last_top_block_probability": float(diagnostics["last_top_block_probability"]),
                "selected_factors": tuple(variant_result["final_selected_factors"]),
                "selected_effects": tuple(variant_result["final_selected_effects"]),
                "selected_columns": tuple(variant_result["final_selected_columns"]),
            }
        )

    truth_row = {
        "task_id": int(task["task_id"]),
        "design_id": design.design_id,
        "scenario_id": scenario_id,
        "dataset_rep": dataset_rep,
        "true_important_factors": tuple(dataset["true_important_factors"]),
        "true_active_effects": tuple(dataset["true_active_effects"]),
        "true_active_columns": tuple(dataset["true_active_columns"]),
    }
    return int(task["task_id"]), replication_rows, truth_row


def _append_methodology_result(
    result: tuple[int, list[dict[str, Any]], dict[str, Any]],
    replication_items: list[dict[str, Any]],
    truth_items: list[dict[str, Any]],
) -> None:
    _task_id, replication_rows, truth_row = result
    replication_items.extend(replication_rows)
    truth_items.append(truth_row)


def _run_serial_tasks(
    task_iter: Iterable[Mapping[str, Any]],
    *,
    progress: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    replication_items: list[dict[str, Any]] = []
    truth_items: list[dict[str, Any]] = []
    for task in task_iter:
        _append_methodology_result(
            _run_methodology_replication(task),
            replication_items,
            truth_items,
        )
        progress.update(1)
    return replication_items, truth_items


def _run_parallel_tasks(
    task_iter: Iterable[Mapping[str, Any]],
    *,
    worker_count: int,
    design_registry: Mapping[str, core.MixedLevelDesign],
    progress: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    replication_items: list[dict[str, Any]] = []
    truth_items: list[dict[str, Any]] = []
    max_pending = max(int(worker_count) * 4, 1)
    pending: dict[Any, None] = {}
    mp_context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=int(worker_count),
        mp_context=mp_context,
        initializer=core._set_simulation_design_registry,
        initargs=(design_registry,),
    ) as executor:
        task_iter = iter(task_iter)

        while len(pending) < max_pending:
            try:
                task = next(task_iter)
            except StopIteration:
                break
            pending[executor.submit(_run_methodology_replication, task)] = None

        while pending:
            done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future, None)
                _append_methodology_result(
                    future.result(),
                    replication_items,
                    truth_items,
                )
                progress.update(1)
                try:
                    next_task = next(task_iter)
                except StopIteration:
                    continue
                pending[executor.submit(_run_methodology_replication, next_task)] = None

    return replication_items, truth_items


def run_methodology_comparison_study(
    settings: MethodologyComparisonSettings | None = None,
    /,
    **overrides: Any,
) -> dict[str, Any]:
    if settings is None:
        settings = MethodologyComparisonSettings(**overrides)
    elif overrides:
        settings = replace(settings, **overrides)

    _validate_settings(settings)
    benchmark_tau = _resolve_weighted_tau(float(settings.benchmark_pkeep), settings.weighted_tau)
    adaptive_stability_tau = _resolve_weighted_tau(
        float(settings.benchmark_pkeep),
        settings.weighted_tau if settings.adaptive_set_tau is None else settings.adaptive_set_tau,
    )
    output_path = Path(settings.output_dir)
    tables_dir = output_path / _METHODOLOGY_TABLES_DIRNAME
    plots_dir = output_path / _METHODOLOGY_PLOTS_DIRNAME
    manifest_path = output_path / _METHODOLOGY_SETTINGS_FILENAME
    manifest = _run_manifest(settings)

    tables_dir.mkdir(parents=True, exist_ok=True)
    if settings.write_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
    if settings.write_settings_file:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    context = common.build_benchmark_context(
        random_state=int(settings.random_state),
        simulation_design_ids=settings.simulation_design_ids,
        scenario_ids_by_design=settings.scenario_ids_by_design,
        generated_design_coding=str(settings.generated_design_coding),
        generated_design_candidate_pool=int(settings.generated_design_candidate_pool),
        generated_design_restarts=int(settings.generated_design_restarts),
    )
    rng = context.rng
    requested_design_ids = context.requested_design_ids
    designs = context.designs
    diagnostics = context.diagnostics
    scenario_map = context.scenario_map
    diagnostics.to_csv(tables_dir / "methodology_design_diagnostics.csv", index=False)

    progress_total = sum(
        len(scenario_map[design.design_id]) * int(settings.reps_per_condition)
        for design in designs
    )
    progress = core._make_progress(
        total=progress_total,
        desc="AWS methodology comparison",
        disable=not settings.show_progress,
        leave=True,
    )

    design_registry = context.design_registry
    core._set_simulation_design_registry(design_registry)
    task_seed = int(rng.integers(0, 2**31 - 1))
    worker_count = min(core._resolve_n_jobs(settings.n_jobs), max(progress_total, 1))

    def _task_iter():
        return _iter_methodology_tasks(
            requested_design_ids,
            scenario_map,
            reps_per_condition=int(settings.reps_per_condition),
            task_seed=task_seed,
            benchmark_nrep=int(settings.benchmark_nrep),
            benchmark_nint=settings.benchmark_nint,
            benchmark_ntop=int(settings.benchmark_ntop),
            benchmark_pkeep=float(settings.benchmark_pkeep),
            weighted_tau_override=settings.weighted_tau,
            group_stepwise=bool(settings.group_stepwise),
            paper_exact_stepwise=bool(settings.paper_exact_stepwise),
            paper_exact_baseline=bool(settings.paper_exact_baseline),
            final_stepwise=bool(settings.final_stepwise),
            stepwise_mode=str(settings.stepwise_mode),
            p_enter=float(settings.p_enter),
            p_remove=float(settings.p_remove),
            variants=settings.variants,
            adaptive_max_nrep=int(settings.adaptive_max_nrep),
            adaptive_batch_size=int(settings.adaptive_batch_size),
            adaptive_min_nrep=int(settings.adaptive_min_nrep),
            adaptive_stability_epsilon=float(settings.adaptive_stability_epsilon),
            adaptive_patience=int(settings.adaptive_patience),
            adaptive_require_set_stability=bool(settings.adaptive_require_set_stability),
            adaptive_min_jaccard=float(settings.adaptive_min_jaccard),
            adaptive_set_tau=float(adaptive_stability_tau),
            sampling_batch_size=int(settings.sampling_batch_size),
            sampling_warmup_batches=int(settings.sampling_warmup_batches),
            sampling_adaptation_lambda=float(settings.sampling_adaptation_lambda),
            sampling_heredity_eta=float(settings.sampling_heredity_eta),
            sampling_block_floor=float(settings.sampling_block_floor),
        )

    try:
        if worker_count <= 1 or progress_total <= 1:
            replication_items, truth_items = _run_serial_tasks(_task_iter(), progress=progress)
        else:
            try:
                replication_items, truth_items = _run_parallel_tasks(
                    _task_iter(),
                    worker_count=worker_count,
                    design_registry=design_registry,
                    progress=progress,
                )
            except (BrokenProcessPool, OSError, PermissionError) as exc:
                warnings.warn(
                    f"Parallel execution failed ({exc!r}); falling back to serial execution.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                progress.close()
                progress = core._make_progress(
                    total=progress_total,
                    desc="AWS methodology comparison",
                    disable=not settings.show_progress,
                    leave=True,
                )
                replication_items, truth_items = _run_serial_tasks(_task_iter(), progress=progress)
    finally:
        progress.close()

    replication_results = pd.DataFrame.from_records(replication_items)
    truth_records = pd.DataFrame.from_records(truth_items)
    if not replication_results.empty:
        replication_results = replication_results.sort_values(
            ["task_id", "variant_order"],
            kind="stable",
        ).drop(columns=["task_id", "variant_order"]).reset_index(drop=True)
    if not truth_records.empty:
        truth_records = truth_records.sort_values(["task_id"], kind="stable").drop(columns=["task_id"]).reset_index(drop=True)

    replication_results = common.ensure_standard_replication_columns(replication_results)
    summary_results = common.summarize_benchmark_results(replication_results)
    benchmark_mask = common.build_benchmark_mask(
        summary_results,
        benchmark_ntop=int(settings.benchmark_ntop),
        benchmark_pkeep=float(settings.benchmark_pkeep),
        benchmark_tau=float(benchmark_tau),
    )
    benchmark_summary_results = summary_results.loc[benchmark_mask].copy().reset_index(drop=True)
    overall_summary = common.overall_summary(benchmark_summary_results)
    design_summary = common.design_summary(benchmark_summary_results)
    condition_ranking = common.condition_ranking(benchmark_summary_results)
    comparison_table = common.variant_comparison(benchmark_summary_results)
    delta_table = common.build_methodology_delta_table(
        benchmark_summary_results,
        baseline_variant="original_baseline",
    )
    win_table = common.build_methodology_win_table(benchmark_summary_results)

    summary_results.to_csv(tables_dir / "methodology_summary_results.csv", index=False)
    benchmark_summary_results.to_csv(tables_dir / "methodology_benchmark_summary_results.csv", index=False)
    overall_summary.to_csv(tables_dir / "methodology_overall_summary.csv", index=False)
    design_summary.to_csv(tables_dir / "methodology_design_summary.csv", index=False)
    condition_ranking.to_csv(tables_dir / "methodology_condition_ranking.csv", index=False)
    comparison_table.to_csv(tables_dir / "methodology_condition_comparison.csv", index=False)
    delta_table.to_csv(tables_dir / "methodology_delta_table.csv", index=False)
    win_table.to_csv(tables_dir / "methodology_win_table.csv", index=False)
    truth_records.to_csv(tables_dir / "methodology_truth_records.csv", index=False)
    if settings.write_replication_tables:
        replication_results.to_csv(tables_dir / "methodology_replication_results.csv", index=False)

    if settings.write_plots:
        core._save_plotnine_plot(
            common.plot_metric_by_scenario_plotnine(
                benchmark_summary_results,
                metric="score",
                title="Methodology Comparison: Mean Score by Scenario",
                subtitle="All four methodology stages evaluated on the same benchmark conditions.",
            ),
            plots_dir / "methodology_score_by_scenario_plotnine.png",
        )
        core._save_plotnine_plot(
            common.plot_metric_by_scenario_plotnine(
                benchmark_summary_results,
                metric="power",
                title="Methodology Comparison: Mean Power by Scenario",
                subtitle="All four methodology stages evaluated on the same benchmark conditions.",
            ),
            plots_dir / "methodology_power_by_scenario_plotnine.png",
        )
        core._save_plotnine_plot(
            common.plot_runtime_by_scenario_plotnine(
                benchmark_summary_results,
                title="Methodology Comparison: Mean Runtime by Scenario",
                subtitle="Lower runtime is better; all methods use the same design and scenario suite.",
            ),
            plots_dir / "methodology_runtime_by_scenario_plotnine.png",
        )
        core._save_plotnine_plot(
            common.plot_condition_scatter_plotnine(
                benchmark_summary_results,
                metric="score",
                title="Methodology Comparison: Score Stability by Condition",
                subtitle="Higher mean score and lower SD indicate more reliable performance.",
            ),
            plots_dir / "methodology_score_scatter_plotnine.png",
        )
        core._save_plotnine_plot(
            common.plot_methodology_runtime_score_frontier_plotnine(benchmark_summary_results),
            plots_dir / "methodology_runtime_score_frontier_plotnine.png",
        )
        core._save_plotnine_plot(
            common.plot_methodology_power_error_frontier_plotnine(benchmark_summary_results),
            plots_dir / "methodology_power_error_frontier_plotnine.png",
        )
        if not delta_table.empty:
            core._save_plotnine_plot(
                common.plot_methodology_speedup_score_delta_plotnine(delta_table),
                plots_dir / "methodology_speedup_score_delta_plotnine.png",
            )
        core._save_plotnine_plot(
            common.plot_methodology_metric_bars_plotnine(overall_summary),
            plots_dir / "methodology_metric_profile_plotnine.png",
        )
        core._save_plotnine_plot(
            common.plot_methodology_win_heatmap_plotnine(
                win_table,
                metric="score_win_rate",
                title="Methodology Score Win Rates by Design",
                subtitle="Share of benchmark scenarios won on score within each design.",
            ),
            plots_dir / "methodology_score_win_heatmap_plotnine.png",
        )
        core._save_plotnine_plot(
            common.plot_methodology_win_heatmap_plotnine(
                win_table,
                metric="runtime_win_rate",
                title="Methodology Runtime Win Rates by Design",
                subtitle="Share of benchmark scenarios won on runtime within each design.",
            ),
            plots_dir / "methodology_runtime_win_heatmap_plotnine.png",
        )
        core._save_plotnine_plot(
            common.plot_methodology_win_heatmap_plotnine(
                win_table,
                metric="power_win_rate",
                title="Methodology Power Win Rates by Design",
                subtitle="Share of benchmark scenarios won on power within each design.",
            ),
            plots_dir / "methodology_power_win_heatmap_plotnine.png",
        )

    print("\nDesign diagnostics")
    print(diagnostics.to_string(index=False))
    print(f"\nRun name: {settings.run_name}")
    print(f"\nSaved tables to: {tables_dir}")
    if settings.write_plots:
        print(f"Saved plots to:  {plots_dir}")
    if settings.write_settings_file:
        print(f"Saved settings to: {manifest_path}")
    print("All methodology stages used the same generated design defaults and paper scenarios.")
    print(
        "Benchmark settings: "
        f"nrep={settings.benchmark_nrep}, nint={settings.benchmark_nint}, "
        f"ntop={settings.benchmark_ntop}, pkeep={settings.benchmark_pkeep}, tau={benchmark_tau}"
    )
    print(
        "Adaptive nrep settings: "
        f"max_nrep={settings.adaptive_max_nrep}, min_nrep={settings.adaptive_min_nrep}, "
        f"batch_size={settings.adaptive_batch_size}, epsilon={settings.adaptive_stability_epsilon}, "
        f"patience={settings.adaptive_patience}"
    )
    print(
        "Adaptive sampling settings: "
        f"batch_size={settings.sampling_batch_size}, warmup_batches={settings.sampling_warmup_batches}, "
        f"lambda={settings.sampling_adaptation_lambda}, eta={settings.sampling_heredity_eta}, "
        f"block_floor={settings.sampling_block_floor}"
    )
    print(f"Variants: {', '.join(settings.variants)}")
    print(f"Each condition used reps_per_condition={settings.reps_per_condition}.")
    print(f"Parallel workers: {worker_count}")

    return {
        "output_dir": output_path,
        "tables_dir": tables_dir,
        "plots_dir": plots_dir,
        "settings": settings,
        "design_diagnostics": diagnostics,
        "replication_results": replication_results,
        "truth_records": truth_records,
        "summary_results": summary_results,
        "benchmark_summary_results": benchmark_summary_results,
        "overall_summary": overall_summary,
        "design_summary": design_summary,
        "condition_ranking": condition_ranking,
        "comparison_table": comparison_table,
        "delta_table": delta_table,
        "win_table": win_table,
    }


__all__ = [
    "MethodologyComparisonSettings",
    "run_methodology_comparison_study",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the unified four-methodology AWS-GDS-ARM comparison on the shared benchmark design/scenario suite."
    )
    parser.add_argument(
        "--output-dir",
        default="study_outputs_aws_methodology",
        help="Directory where methodology comparison tables will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_CORE_DEFAULTS.random_state,
        help="Random seed for design generation and simulation.",
    )
    parser.add_argument(
        "--reps-per-condition",
        type=int,
        default=_CORE_DEFAULTS.reps_per_condition,
        help="Monte Carlo datasets per design-scenario condition.",
    )
    parser.add_argument(
        "--benchmark-nrep",
        type=int,
        default=_CORE_DEFAULTS.ntop_pkeep_nrep,
        help="Fixed-nrep benchmark used for the original and weighted methodologies.",
    )
    parser.add_argument(
        "--adaptive-max-nrep",
        type=int,
        default=_CORE_DEFAULTS.ntop_pkeep_nrep,
        help="Maximum repetition budget for the adaptive nrep methodology.",
    )
    parser.add_argument(
        "--adaptive-batch-size",
        type=int,
        default=50,
        help="Batch size used by the adaptive stopping rule.",
    )
    parser.add_argument(
        "--adaptive-min-nrep",
        type=int,
        default=max(_CORE_DEFAULTS.nrep_study_ntop, 100),
        help="Minimum repetitions before adaptive stopping can trigger.",
    )
    parser.add_argument(
        "--adaptive-epsilon",
        type=float,
        default=0.01,
        help="Score-stability tolerance for adaptive stopping.",
    )
    parser.add_argument(
        "--adaptive-patience",
        type=int,
        default=2,
        help="Number of consecutive stable batches required to stop.",
    )
    parser.add_argument(
        "--sampling-batch-size",
        type=int,
        default=50,
        help="Batch size used to update adaptive interaction probabilities.",
    )
    parser.add_argument(
        "--sampling-warmup-batches",
        type=int,
        default=2,
        help="Number of initial uniform batches before adaptive interaction sampling begins.",
    )
    parser.add_argument(
        "--sampling-lambda",
        type=float,
        default=0.7,
        help="Adaptation weight for interaction-block probabilities.",
    )
    parser.add_argument(
        "--sampling-eta",
        type=float,
        default=0.5,
        help="Weight on strong-heredity evidence inside adaptive interaction updates.",
    )
    parser.add_argument(
        "--sampling-block-floor",
        type=float,
        default=1e-3,
        help="Per-block exploration floor for adaptive interaction sampling.",
    )
    parser.add_argument(
        "--weighted-tau",
        type=float,
        default=None,
        help="Threshold for weighted aggregation score. Defaults to pkeep when omitted.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel worker processes. Use 0 or -1 for all CPUs.",
    )
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        choices=_ALLOWED_VARIANTS,
        help="Methodology variant to include.",
    )
    if any(arg.startswith("--f=") for arg in sys.argv[1:]) or "ipykernel_launcher" in Path(sys.argv[0]).name:
        print("Detected Jupyter kernel arguments; skipping CLI entry point.")
    else:
        args, _unknown = parser.parse_known_args()
        run_methodology_comparison_study(
            MethodologyComparisonSettings(
                output_dir=args.output_dir,
                random_state=args.seed,
                reps_per_condition=args.reps_per_condition,
                benchmark_nrep=args.benchmark_nrep,
                adaptive_max_nrep=args.adaptive_max_nrep,
                adaptive_batch_size=args.adaptive_batch_size,
                adaptive_min_nrep=args.adaptive_min_nrep,
                adaptive_stability_epsilon=args.adaptive_epsilon,
                adaptive_patience=args.adaptive_patience,
                sampling_batch_size=args.sampling_batch_size,
                sampling_warmup_batches=args.sampling_warmup_batches,
                sampling_adaptation_lambda=args.sampling_lambda,
                sampling_heredity_eta=args.sampling_eta,
                sampling_block_floor=args.sampling_block_floor,
                weighted_tau=args.weighted_tau,
                n_jobs=args.n_jobs,
                variants=tuple(args.variants or _ALLOWED_VARIANTS),
            )
        )
