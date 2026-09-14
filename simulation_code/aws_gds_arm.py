"""Phase-I AWS-GDS-ARM weighted-only tuning study.

This module keeps the generated-design defaults from ``mixed_gds_arm.py`` but
evaluates a weighted-only aggregation rule over the paper-style ``(ntop, pkeep)``
grid while holding the remaining settings fixed:

- ``nint = 2n``
- ``nrep = 1000``

It runs the weighted-only aggregation rule as a direct analogue of the baseline
aggregation study:

- ``weighted_only``: BIC-weighted inclusion scores over the top ``ntop`` models,
  thresholded at ``tau`` where ``tau = pkeep`` by default

The goal is to make the first extension easy to compare to the baseline plots
without modifying ``mixed_gds_arm.py``.
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
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

import aws_gds_arm_common as common
import mixed_gds_arm as core


_CORE_DEFAULTS = core.StudyRunSettings()
_ALLOWED_VARIANTS = ("weighted_only",)
_WEIGHTED_TABLES_DIRNAME = "weighted_tables"
_WEIGHTED_PLOTS_DIRNAME = "weighted_plots"
_WEIGHTED_SETTINGS_FILENAME = "weighted_run_settings.json"
_BASELINE_P_ENTER = 0.01
_BASELINE_P_REMOVE = 0.05


@dataclass(slots=True)
class WeightedComparisonSettings:
    output_dir: str | os.PathLike[str] = "study_outputs_aws_weighted"
    run_name: str = "weighted_only_phase1"
    generated_design_coding: str = _CORE_DEFAULTS.generated_design_coding
    generated_design_candidate_pool: int = _CORE_DEFAULTS.generated_design_candidate_pool
    generated_design_restarts: int = _CORE_DEFAULTS.generated_design_restarts
    random_state: int = _CORE_DEFAULTS.random_state
    reps_per_condition: int = _CORE_DEFAULTS.reps_per_condition
    simulation_design_ids: tuple[str, ...] = _CORE_DEFAULTS.simulation_design_ids
    scenario_ids_by_design: dict[str, tuple[int | str, ...]] | None = None
    ntop_values: tuple[int, ...] = _CORE_DEFAULTS.ntop_values
    pkeep_values: tuple[float, ...] = _CORE_DEFAULTS.pkeep_values
    benchmark_nrep: int = _CORE_DEFAULTS.ntop_pkeep_nrep
    benchmark_nint: int | str = _CORE_DEFAULTS.ntop_pkeep_nint
    benchmark_ntop: int = _CORE_DEFAULTS.nrep_study_ntop
    benchmark_pkeep: float = _CORE_DEFAULTS.nrep_study_pkeep
    weighted_tau: float | None = None
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


def _settings_payload(settings: WeightedComparisonSettings) -> dict[str, Any]:
    payload = asdict(settings)
    payload["output_dir"] = str(settings.output_dir)
    return payload


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_manifest(settings: WeightedComparisonSettings) -> dict[str, Any]:
    manifest = _settings_payload(settings)
    manifest["aws_source_sha256"] = _source_sha256(Path(__file__))
    manifest["core_source_sha256"] = core._source_sha256()
    manifest["cache_version"] = 1
    return json.loads(json.dumps(manifest, sort_keys=True))


def _resolve_weighted_tau(pkeep: float, weighted_tau_override: float | None) -> float:
    return float(pkeep if weighted_tau_override is None else weighted_tau_override)


def _validate_settings(settings: WeightedComparisonSettings) -> None:
    if int(settings.reps_per_condition) <= 0:
        raise ValueError("reps_per_condition must be positive.")
    if int(settings.benchmark_nrep) <= 0:
        raise ValueError("benchmark_nrep must be positive.")
    ntop_values = tuple(int(ntop) for ntop in settings.ntop_values)
    pkeep_values = tuple(float(pkeep) for pkeep in settings.pkeep_values)
    if not ntop_values:
        raise ValueError("ntop_values must contain at least one value.")
    if not pkeep_values:
        raise ValueError("pkeep_values must contain at least one value.")
    if int(settings.benchmark_ntop) not in set(ntop_values):
        raise ValueError("benchmark_ntop must be present in ntop_values.")
    if not any(np.isclose(float(settings.benchmark_pkeep), value) for value in pkeep_values):
        raise ValueError("benchmark_pkeep must be present in pkeep_values.")
    invalid_variants = sorted(set(settings.variants) - set(_ALLOWED_VARIANTS))
    if invalid_variants:
        raise ValueError(
            f"Unsupported variants: {invalid_variants}. Expected any of {_ALLOWED_VARIANTS}."
        )


def _build_repetition_path(
    design: core.MixedLevelDesign,
    y: Sequence[float] | pd.Series | np.ndarray,
    *,
    nrep: int,
    nint: int,
    delta_grid: Sequence[float] | None,
    random_state: int | np.random.Generator | None,
    solver_order: Sequence[str] | None,
    paper_exact_baseline: bool,
    rep_seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    rng = None if rep_seeds is not None else core._ensure_rng(random_state)
    y_centered = core.center_response(y)
    nint = int(min(max(int(nint), 0), design.X_int.shape[1]))
    repetition_rows: list[dict[str, Any]] = []
    seed_iter = iter(rep_seeds) if rep_seeds is not None else None

    for rep_idx in range(1, int(nrep) + 1):
        if seed_iter is not None:
            try:
                rep_seed = int(next(seed_iter))
            except StopIteration as exc:
                raise ValueError("rep_seeds must contain at least nrep seeds.") from exc
        else:
            assert rng is not None
            rep_seed = int(rng.integers(0, 2**31 - 1))
        step_i_result = core.run_GDS_step_i(
            design.X_main,
            design.X_int,
            y_centered,
            metadata=design,
            nint=nint,
            delta_grid=delta_grid,
            random_state=rep_seed,
            solver_order=solver_order,
            paper_exact_baseline=paper_exact_baseline,
        )
        gds_result = step_i_result.gds_result
        repetition_rows.append(
            {
                "rep": rep_idx,
                "sampled_interaction_columns": tuple(step_i_result.sampled_interaction_columns),
                "selected_columns": tuple(gds_result.best_selected_columns),
                "selected_effects": tuple(gds_result.best_selected_effects),
                "selected_factors": tuple(gds_result.best_selected_factors),
                "bic": gds_result.best_bic,
                "delta": gds_result.best_delta,
                "ols_coefficients": gds_result.best_ols_result.coefficients,
                "solver_status": gds_result.solver_status,
            }
        )

    return pd.DataFrame.from_records(repetition_rows)


def _bic_model_weights(repetition_results: pd.DataFrame) -> np.ndarray:
    if repetition_results.empty:
        return np.array([], dtype=float)

    bic_values = repetition_results["bic"].to_numpy(dtype=float)
    finite_mask = np.isfinite(bic_values)
    if not finite_mask.any():
        return np.full(len(bic_values), 1.0 / len(bic_values), dtype=float)

    weights = np.zeros(len(bic_values), dtype=float)
    min_bic = float(np.min(bic_values[finite_mask]))
    weights[finite_mask] = np.exp(-0.5 * (bic_values[finite_mask] - min_bic))
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(bic_values), 1.0 / len(bic_values), dtype=float)
    return weights / total


def _weighted_aggregate(
    repetition_results: pd.DataFrame,
    metadata: core.MixedLevelDesign,
    *,
    tau: float,
    paper_exact_baseline: bool,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    weights = _bic_model_weights(repetition_results)
    effect_type_lookup = dict(
        zip(metadata.effect_table["effect_id"], metadata.effect_table["effect_type"], strict=False)
    )

    if paper_exact_baseline:
        weighted_scores: MutableMapping[str, float] = {}
        inclusion_counts: MutableMapping[str, int] = {}
        for weight, selected_columns in zip(weights, repetition_results["selected_columns"], strict=False):
            for column_id in selected_columns:
                weighted_scores[column_id] = weighted_scores.get(column_id, 0.0) + float(weight)
                inclusion_counts[column_id] = inclusion_counts.get(column_id, 0) + 1

        rows = []
        n_rows = max(len(repetition_results), 1)
        for column_id in metadata.X_full.columns.tolist():
            count = int(inclusion_counts.get(column_id, 0))
            weighted_score = float(weighted_scores.get(column_id, 0.0))
            rows.append(
                {
                    "feature_id": column_id,
                    "aggregation_unit": "column",
                    "count": count,
                    "proportion": count / n_rows,
                    "score": weighted_score,
                    "weighted_score": weighted_score,
                    "kept": weighted_score >= float(tau),
                    "effect_type": metadata.column_to_metadata[column_id]["effect_type"],
                }
            )
        feature_table = pd.DataFrame(rows).sort_values(
            ["kept", "score", "count", "feature_id"],
            ascending=[False, False, False, True],
        )
        aggregated_columns = feature_table.loc[feature_table["kept"], "feature_id"].tolist()
        aggregated_effects = sorted(
            metadata.column_table[
                metadata.column_table["column_id"].isin(aggregated_columns)
            ]["effect_id"].unique().tolist()
        )
    else:
        weighted_scores = {}
        inclusion_counts = {}
        for weight, selected_effects in zip(weights, repetition_results["selected_effects"], strict=False):
            for effect_id in selected_effects:
                weighted_scores[effect_id] = weighted_scores.get(effect_id, 0.0) + float(weight)
                inclusion_counts[effect_id] = inclusion_counts.get(effect_id, 0) + 1

        rows = []
        n_rows = max(len(repetition_results), 1)
        for effect_id in metadata.effect_ids:
            count = int(inclusion_counts.get(effect_id, 0))
            weighted_score = float(weighted_scores.get(effect_id, 0.0))
            rows.append(
                {
                    "feature_id": effect_id,
                    "aggregation_unit": "effect",
                    "count": count,
                    "proportion": count / n_rows,
                    "score": weighted_score,
                    "weighted_score": weighted_score,
                    "kept": weighted_score >= float(tau),
                    "effect_type": effect_type_lookup[effect_id],
                }
            )
        feature_table = pd.DataFrame(rows).sort_values(
            ["kept", "score", "count", "feature_id"],
            ascending=[False, False, False, True],
        )
        aggregated_effects = feature_table.loc[feature_table["kept"], "feature_id"].tolist()
        aggregated_columns = metadata.columns_for_effects(aggregated_effects)

    diagnostics = {
        "aggregation_method": "bic_weighted",
        "nrep_used": int(len(repetition_results)),
        "tau": float(tau),
        "min_bic": float(repetition_results["bic"].min()) if not repetition_results.empty else np.nan,
    }
    return feature_table, aggregated_effects, aggregated_columns, diagnostics


def _finalize_selection(
    metadata: core.MixedLevelDesign,
    y: Sequence[float] | pd.Series | np.ndarray,
    *,
    aggregated_effects: Sequence[str],
    aggregated_columns: Sequence[str],
    group_stepwise: bool,
    paper_exact_stepwise: bool,
    final_stepwise: bool,
    stepwise_mode: str,
    p_enter: float,
    p_remove: float,
) -> tuple[core.StepwiseResult, list[str], list[str], list[str]]:
    y_centered = core.center_response(y)
    aggregated_effects = list(aggregated_effects)
    aggregated_columns = list(aggregated_columns)

    if final_stepwise:
        if paper_exact_stepwise:
            final_stepwise_result = core._run_paper_exact_stepwise(
                metadata.X_full,
                y_centered,
                start_columns=aggregated_columns,
                main_columns=metadata.X_main.columns.tolist(),
                p_enter=p_enter,
                p_remove=p_remove,
            )
            final_selected_effects = sorted(
                metadata.column_table[
                    metadata.column_table["column_id"].isin(final_stepwise_result.selected_columns)
                ]["effect_id"].unique().tolist()
            )
        else:
            group_to_columns, group_types = core._resolve_group_structure(
                metadata,
                group_stepwise=group_stepwise,
            )
            if group_stepwise:
                start_groups = aggregated_effects[:]
                remaining_main = [
                    effect_id
                    for effect_id in metadata.effect_table.loc[
                        metadata.effect_table["effect_type"] == "main",
                        "effect_id",
                    ].tolist()
                    if effect_id not in aggregated_effects
                ]
            else:
                start_groups = aggregated_columns[:]
                interaction_columns = set(metadata.X_int.columns)
                remaining_main = [
                    column_id
                    for column_id in metadata.X_main.columns.tolist()
                    if column_id not in start_groups and column_id not in interaction_columns
                ]

            final_stepwise_result = core._run_stepwise_selection(
                metadata.X_full,
                y_centered,
                group_to_columns=group_to_columns,
                group_types=group_types,
                start_groups=start_groups,
                add_candidate_groups=remaining_main,
                stepwise_mode=stepwise_mode,
                p_enter=p_enter,
                p_remove=p_remove,
            )
            if group_stepwise:
                final_selected_effects = final_stepwise_result.selected_groups[:]
            else:
                final_selected_effects = sorted(
                    metadata.column_table[
                        metadata.column_table["column_id"].isin(final_stepwise_result.selected_columns)
                    ]["effect_id"].unique().tolist()
                )
    else:
        final_columns = aggregated_columns[:]
        fit = (
            core._fit_ols(metadata.X_full[final_columns], y_centered, final_columns)
            if final_columns
            else core._fit_ols(pd.DataFrame(index=metadata.X_full.index), y_centered, [])
        )
        final_stepwise_result = core.StepwiseResult(
            selected_groups=aggregated_effects[:] if group_stepwise else final_columns[:],
            selected_columns=final_columns,
            coefficients=fit.coefficients,
            fit=fit,
            history=pd.DataFrame(columns=["action", "group", "p_value"]),
        )
        final_selected_effects = aggregated_effects[:]

    final_selected_columns = final_stepwise_result.selected_columns[:]
    final_selected_factors = metadata.factors_for_effects(final_selected_effects)
    return (
        final_stepwise_result,
        final_selected_effects,
        final_selected_columns,
        final_selected_factors,
    )


def _build_variant_result(
    repetition_results: pd.DataFrame,
    metadata: core.MixedLevelDesign,
    y: Sequence[float] | pd.Series | np.ndarray,
    *,
    variant: str,
    ntop: int,
    pkeep: float,
    weighted_tau: float,
    group_stepwise: bool,
    paper_exact_stepwise: bool,
    paper_exact_baseline: bool,
    final_stepwise: bool,
    stepwise_mode: str,
    p_enter: float,
    p_remove: float,
) -> dict[str, Any]:
    ranked_results = repetition_results.sort_values("bic", ascending=True, kind="stable").reset_index(drop=True)
    top_n = max(int(ntop), 0)
    top_results = ranked_results.head(min(top_n, len(ranked_results))).copy() if top_n > 0 else ranked_results.head(0).copy()

    if variant == "weighted_only":
        feature_table, aggregated_effects, aggregated_columns, aggregation_diagnostics = _weighted_aggregate(
            top_results,
            metadata,
            tau=weighted_tau,
            paper_exact_baseline=paper_exact_baseline,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    (
        final_stepwise_result,
        final_selected_effects,
        final_selected_columns,
        final_selected_factors,
    ) = _finalize_selection(
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
        **aggregation_diagnostics,
        "variant": variant,
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
        "variant": variant,
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


def _iter_weighted_comparison_tasks(
    design_ids: Sequence[str],
    scenario_map: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reps_per_condition: int,
    task_seed: int,
    benchmark_nrep: int,
    benchmark_nint: int | str,
    ntop_values: Sequence[int],
    pkeep_values: Sequence[float],
    weighted_tau_override: float | None,
    group_stepwise: bool,
    paper_exact_stepwise: bool,
    paper_exact_baseline: bool,
    final_stepwise: bool,
    stepwise_mode: str,
    p_enter: float,
    p_remove: float,
    variants: Sequence[str],
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
                    "ntop_values": tuple(int(ntop) for ntop in ntop_values),
                    "pkeep_values": tuple(float(pkeep) for pkeep in pkeep_values),
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
                }
                task_id += 1


def _run_weighted_comparison_replication(
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
    if bool(task["paper_exact_baseline"]):
        current_delta_grid = None
    else:
        current_delta_grid = core.derive_delta_grid(design.X_full, dataset["y"])

    resolved_nint = core._resolve_relative_count(task["benchmark_nint"], design.n_runs)
    path_t0 = time.perf_counter()
    repetition_results = _build_repetition_path(
        design,
        dataset["y"],
        nrep=int(task["benchmark_nrep"]),
        nint=resolved_nint,
        delta_grid=current_delta_grid,
        random_state=int(rng.integers(0, 2**31 - 1)),
        solver_order=None,
        paper_exact_baseline=bool(task["paper_exact_baseline"]),
    )
    path_elapsed = time.perf_counter() - path_t0

    replication_rows: list[dict[str, Any]] = []
    config_order = 0
    for ntop in task["ntop_values"]:
        for pkeep in task["pkeep_values"]:
            weighted_tau = _resolve_weighted_tau(
                float(pkeep),
                task["weighted_tau_override"],
            )
            for variant_idx, variant in enumerate(task["variants"]):
                variant_t0 = time.perf_counter()
                variant_result = _build_variant_result(
                    repetition_results,
                    design,
                    dataset["y"],
                    variant=str(variant),
                    ntop=int(ntop),
                    pkeep=float(pkeep),
                    weighted_tau=float(weighted_tau),
                    group_stepwise=bool(task["group_stepwise"]),
                    paper_exact_stepwise=bool(task["paper_exact_stepwise"]),
                    paper_exact_baseline=bool(task["paper_exact_baseline"]),
                    final_stepwise=bool(task["final_stepwise"]),
                    stepwise_mode=str(task["stepwise_mode"]),
                    p_enter=float(task["p_enter"]),
                    p_remove=float(task["p_remove"]),
                )
                variant_elapsed = time.perf_counter() - variant_t0
                metrics = core._factor_metrics(
                    selected_factors=variant_result["final_selected_factors"],
                    true_important_factors=dataset["true_important_factors"],
                    all_factors=design.factor_ids,
                )
                diagnostics = variant_result["diagnostics"]
                replication_rows.append(
                    {
                        "task_id": int(task["task_id"]),
                        "config_order": int(config_order),
                        "variant_order": int(variant_idx),
                        "design_id": design.design_id,
                        "scenario_id": scenario_id,
                        "dataset_rep": dataset_rep,
                        "variant": str(variant),
                        "nrep": int(task["benchmark_nrep"]),
                        "nrep_used": int(diagnostics["nrep_used"]),
                        "nint": int(resolved_nint),
                        "ntop": int(ntop),
                        "pkeep": float(pkeep),
                        "weighted_tau": float(weighted_tau) if variant == "weighted_only" else np.nan,
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
                        "timing_seconds": float(path_elapsed + variant_elapsed),
                        "path_timing_seconds": float(path_elapsed),
                        "variant_timing_seconds": float(variant_elapsed),
                        "aggregation_method": str(diagnostics["aggregation_method"]),
                        "selected_factors": tuple(variant_result["final_selected_factors"]),
                        "selected_effects": tuple(variant_result["final_selected_effects"]),
                        "selected_columns": tuple(variant_result["final_selected_columns"]),
                    }
                )
            config_order += 1

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


def _append_weighted_result(
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
        _append_weighted_result(
            _run_weighted_comparison_replication(task),
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
            pending[executor.submit(_run_weighted_comparison_replication, task)] = None

        while pending:
            done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future, None)
                _append_weighted_result(
                    future.result(),
                    replication_items,
                    truth_items,
                )
                progress.update(1)
                try:
                    next_task = next(task_iter)
                except StopIteration:
                    continue
                pending[executor.submit(_run_weighted_comparison_replication, next_task)] = None

    return replication_items, truth_items


def _summarize_results(replication_results: pd.DataFrame) -> pd.DataFrame:
    return (
        replication_results.groupby(
            [
                "design_id",
                "scenario_id",
                "variant",
                "nrep",
                "nrep_used",
                "nint",
                "ntop",
                "pkeep",
                "weighted_tau",
                "group_stepwise",
                "paper_exact_stepwise",
                "paper_exact_baseline",
                "final_stepwise",
                "aggregation_method",
            ],
            dropna=False,
        )
        .agg(
            mean_power=("power", "mean"),
            sd_power=("power", "std"),
            mean_error=("error", "mean"),
            sd_error=("error", "std"),
            mean_score=("score", "mean"),
            sd_score=("score", "std"),
            mean_model_size=("final_model_size", "mean"),
            mean_BIC=("final_BIC", "mean"),
            mean_timing_seconds=("timing_seconds", "mean"),
            n_datasets=("dataset_rep", "count"),
        )
        .reset_index()
        .fillna({"sd_power": 0.0, "sd_error": 0.0, "sd_score": 0.0})
        .sort_values(["design_id", "scenario_id", "variant"], kind="stable")
        .reset_index(drop=True)
    )


def _overall_summary(summary_results: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_results.groupby(["variant"], dropna=False)
        .agg(
            mean_score=("mean_score", "mean"),
            sd_score=("mean_score", "std"),
            mean_power=("mean_power", "mean"),
            mean_error=("mean_error", "mean"),
            mean_timing_seconds=("mean_timing_seconds", "mean"),
            n_conditions=("scenario_id", "count"),
        )
        .reset_index()
        .fillna({"sd_score": 0.0})
        .sort_values("variant", kind="stable")
        .reset_index(drop=True)
    )


def weighted_ntop_pkeep_aggregate(summary_results: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_results.groupby(["ntop", "pkeep"], dropna=False)
        .agg(
            mean_score=("mean_score", "mean"),
            sd_score=("mean_score", "std"),
            mean_power=("mean_power", "mean"),
            sd_power=("mean_power", "std"),
            n_conditions=("design_id", "count"),
        )
        .reset_index()
        .fillna({"sd_score": 0.0, "sd_power": 0.0})
        .sort_values(["pkeep", "ntop"], kind="stable")
        .reset_index(drop=True)
    )


def _weighted_plot_data(summary_results: pd.DataFrame) -> pd.DataFrame:
    plot_data = summary_results.copy()
    plot_data["design_label"] = plot_data["design_id"].map(core._paper_design_label)
    plot_data["scenario_label"] = plot_data["scenario_id"].map(core._paper_scenario_label)
    plot_data["scenario_id_int"] = pd.to_numeric(plot_data["scenario_id"], errors="coerce")
    plot_data["condition_label"] = (
        plot_data["design_label"].astype(str) + " | " + plot_data["scenario_label"].astype(str)
    )
    return plot_data


def weighted_design_summary(summary_results: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_results.groupby(["design_id", "variant"], dropna=False)
        .agg(
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


def weighted_condition_ranking(summary_results: pd.DataFrame) -> pd.DataFrame:
    ranked = _weighted_plot_data(summary_results)
    columns = [
        "design_id",
        "design_label",
        "scenario_id",
        "scenario_label",
        "variant",
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


def _variant_comparison(summary_results: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        summary_results.pivot_table(
            index=["design_id", "scenario_id"],
            columns="variant",
            values=["mean_score", "mean_power", "mean_error", "mean_timing_seconds"],
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


def plot_weighted_metric_by_scenario_plotnine(data: pd.DataFrame, metric: str = "score"):
    return common.plot_metric_by_scenario_plotnine(
        data,
        metric=metric,
        title=f"Weighted-Only Mean {metric.title()} by Scenario",
        subtitle="Fixed benchmark tuning carried over from the baseline.",
    )


def plot_weighted_condition_scatter_plotnine(data: pd.DataFrame, metric: str = "score"):
    return common.plot_condition_scatter_plotnine(
        data,
        metric=metric,
        title=f"Weighted-Only Mean vs SD: {metric}",
        subtitle="Condition-level stability at the benchmark operating point.",
    )


def plot_weighted_runtime_by_scenario_plotnine(data: pd.DataFrame):
    return common.plot_runtime_by_scenario_plotnine(
        data,
        title="Weighted-Only Mean Runtime by Scenario",
        subtitle="Fixed benchmark tuning carried over from the baseline.",
    )


def run_weighted_benchmark_study(
    settings: WeightedComparisonSettings | None = None,
    /,
    **overrides: Any,
) -> dict[str, Any]:
    if settings is None:
        settings = WeightedComparisonSettings(**overrides)
    elif overrides:
        settings = replace(settings, **overrides)

    _validate_settings(settings)
    benchmark_tau = _resolve_weighted_tau(float(settings.benchmark_pkeep), settings.weighted_tau)
    output_path = Path(settings.output_dir)
    tables_dir = output_path / _WEIGHTED_TABLES_DIRNAME
    plots_dir = output_path / _WEIGHTED_PLOTS_DIRNAME
    manifest_path = output_path / _WEIGHTED_SETTINGS_FILENAME
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
    diagnostics.to_csv(tables_dir / "weighted_design_diagnostics.csv", index=False)
    scenario_map = context.scenario_map

    progress_total = sum(
        len(scenario_map[design.design_id]) * int(settings.reps_per_condition)
        for design in designs
    )
    progress = core._make_progress(
        total=progress_total,
        desc="AWS benchmark",
        disable=not settings.show_progress,
        leave=True,
    )

    design_registry = context.design_registry
    core._set_simulation_design_registry(design_registry)
    task_seed = int(rng.integers(0, 2**31 - 1))
    worker_count = min(core._resolve_n_jobs(settings.n_jobs), max(progress_total, 1))

    def _task_iter():
        return _iter_weighted_comparison_tasks(
            requested_design_ids,
            scenario_map,
            reps_per_condition=int(settings.reps_per_condition),
            task_seed=task_seed,
            benchmark_nrep=int(settings.benchmark_nrep),
            benchmark_nint=settings.benchmark_nint,
            ntop_values=tuple(settings.ntop_values),
            pkeep_values=tuple(settings.pkeep_values),
            weighted_tau_override=settings.weighted_tau,
            group_stepwise=bool(settings.group_stepwise),
            paper_exact_stepwise=bool(settings.paper_exact_stepwise),
            paper_exact_baseline=bool(settings.paper_exact_baseline),
            final_stepwise=bool(settings.final_stepwise),
            stepwise_mode=str(settings.stepwise_mode),
            p_enter=float(settings.p_enter),
            p_remove=float(settings.p_remove),
            variants=settings.variants,
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
                    desc="AWS benchmark",
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
            ["task_id", "config_order", "variant_order"],
            kind="stable",
        ).drop(columns=["task_id", "config_order", "variant_order"]).reset_index(drop=True)
    if not truth_records.empty:
        truth_records = truth_records.sort_values(["task_id"], kind="stable").drop(columns=["task_id"]).reset_index(drop=True)

    summary_results = _summarize_results(replication_results)
    benchmark_mask = common.build_benchmark_mask(
        summary_results,
        benchmark_ntop=int(settings.benchmark_ntop),
        benchmark_pkeep=float(settings.benchmark_pkeep),
        benchmark_tau=float(benchmark_tau),
    )
    benchmark_summary_results = summary_results.loc[benchmark_mask].copy().reset_index(drop=True)
    ntop_pkeep_aggregate = weighted_ntop_pkeep_aggregate(summary_results)
    overall_summary = _overall_summary(benchmark_summary_results)
    design_summary = weighted_design_summary(benchmark_summary_results)
    condition_ranking = weighted_condition_ranking(benchmark_summary_results)
    comparison_table = _variant_comparison(benchmark_summary_results)

    summary_results.to_csv(tables_dir / "weighted_ntop_pkeep_summary_results.csv", index=False)
    benchmark_summary_results.to_csv(tables_dir / "weighted_variant_summary_results.csv", index=False)
    overall_summary.to_csv(tables_dir / "weighted_variant_overall_summary.csv", index=False)
    design_summary.to_csv(tables_dir / "weighted_design_summary.csv", index=False)
    condition_ranking.to_csv(tables_dir / "weighted_condition_ranking.csv", index=False)
    comparison_table.to_csv(tables_dir / "weighted_variant_condition_comparison.csv", index=False)
    ntop_pkeep_aggregate.to_csv(tables_dir / "weighted_ntop_pkeep_aggregate.csv", index=False)
    truth_records.to_csv(tables_dir / "weighted_truth_records.csv", index=False)
    if settings.write_replication_tables:
        replication_results.to_csv(tables_dir / "weighted_variant_replication_results.csv", index=False)
    if settings.write_plots:
        core._save_plotnine_plot(
            common.plot_ntop_pkeep_scatter_plotnine(
                ntop_pkeep_aggregate,
                metric="score",
                title="Weighted-Only Mean vs SD for (ntop, pkeep): score",
                subtitle="Grid search over the weighted aggregation operating point.",
            ),
            plots_dir / "weighted_ntop_pkeep_score_plotnine.png",
        )
        core._save_plotnine_plot(
            common.plot_ntop_pkeep_scatter_plotnine(
                ntop_pkeep_aggregate,
                metric="power",
                title="Weighted-Only Mean vs SD for (ntop, pkeep): power",
                subtitle="Grid search over the weighted aggregation operating point.",
            ),
            plots_dir / "weighted_ntop_pkeep_power_plotnine.png",
        )
        core._save_plotnine_plot(
            plot_weighted_metric_by_scenario_plotnine(benchmark_summary_results, metric="score"),
            plots_dir / "weighted_score_by_scenario_plotnine.png",
        )
        core._save_plotnine_plot(
            plot_weighted_metric_by_scenario_plotnine(benchmark_summary_results, metric="power"),
            plots_dir / "weighted_power_by_scenario_plotnine.png",
        )
        core._save_plotnine_plot(
            plot_weighted_condition_scatter_plotnine(benchmark_summary_results, metric="score"),
            plots_dir / "weighted_score_scatter_plotnine.png",
        )
        core._save_plotnine_plot(
            plot_weighted_runtime_by_scenario_plotnine(benchmark_summary_results),
            plots_dir / "weighted_runtime_by_scenario_plotnine.png",
        )

    print("\nDesign diagnostics")
    print(diagnostics.to_string(index=False))
    print(f"\nRun name: {settings.run_name}")
    print(f"\nSaved tables to: {tables_dir}")
    if settings.write_plots:
        print(f"Saved plots to:  {plots_dir}")
    if settings.write_settings_file:
        print(f"Saved settings to: {manifest_path}")
    print("Design defaults and RNG seed follow mixed_gds_arm.py unless overridden.")
    print(
        "Fixed non-grid tuning carried over from the baseline: "
        f"nrep={settings.benchmark_nrep}, nint={settings.benchmark_nint}"
    )
    print(
        "Weighted ntop/pkeep grid: "
        f"ntop={tuple(settings.ntop_values)}, pkeep={tuple(settings.pkeep_values)}"
    )
    print(
        "Benchmark scenario summaries use: "
        f"ntop={settings.benchmark_ntop}, pkeep={settings.benchmark_pkeep}, tau={benchmark_tau}"
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
        "ntop_pkeep_aggregate": ntop_pkeep_aggregate,
        "overall_summary": overall_summary,
        "design_summary": design_summary,
        "condition_ranking": condition_ranking,
        "comparison_table": comparison_table,
    }


__all__ = [
    "WeightedComparisonSettings",
    "weighted_ntop_pkeep_aggregate",
    "weighted_design_summary",
    "weighted_condition_ranking",
    "plot_weighted_metric_by_scenario_plotnine",
    "plot_weighted_condition_scatter_plotnine",
    "plot_weighted_runtime_by_scenario_plotnine",
    "run_weighted_benchmark_study",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the weighted-only AWS-GDS ntop/pkeep study on the same design/scenario settings used by the baseline."
    )
    parser.add_argument(
        "--output-dir",
        default="study_outputs_aws_weighted",
        help="Directory where comparison tables will be written.",
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
        help="Number of GDS-ARM repetitions inside each dataset replication.",
    )
    parser.add_argument(
        "--weighted-tau",
        type=float,
        default=None,
        help="Threshold for the weighted aggregation score. Defaults to pkeep when omitted.",
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
        help="Weighted variant to include.",
    )
    if any(arg.startswith("--f=") for arg in sys.argv[1:]) or "ipykernel_launcher" in Path(sys.argv[0]).name:
        print("Detected Jupyter kernel arguments; skipping CLI entry point.")
    else:
        args, _unknown = parser.parse_known_args()
        run_weighted_benchmark_study(
            WeightedComparisonSettings(
                output_dir=args.output_dir,
                random_state=args.seed,
                reps_per_condition=args.reps_per_condition,
                benchmark_nrep=args.benchmark_nrep,
                weighted_tau=args.weighted_tau,
                n_jobs=args.n_jobs,
                variants=tuple(args.variants or _ALLOWED_VARIANTS),
            )
        )
