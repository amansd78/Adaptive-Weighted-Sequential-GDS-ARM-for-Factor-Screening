"""Phase-III AWS-GDS-ARM adaptive interaction-sampling study.

This module keeps the generated-design defaults from ``mixed_gds_arm.py`` and
extends the weighted-only comparison in ``aws_gds_arm.py`` by adding an
adaptive interaction-block sampling rule while keeping ``nrep`` fixed.

The two currently supported variants are:

- ``weighted_only``: fixed-``nrep`` BIC-weighted inclusion scores over the top
  ``ntop`` models with uniformly sampled interaction columns
- ``adaptive_sampling_only``: the same weighted aggregation rule and fixed
  repetition budget, but interaction columns are sampled in batches from
  adaptive interaction-block probabilities after a uniform warm-up phase

This keeps adaptive interaction sampling isolated from adaptive sequential
stopping so the incremental value of interaction-search adaptation can be
compared directly against the weighted fixed-``nrep`` benchmark.
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import aws_gds_arm_common as common
import aws_gds_arm_adaptive as shared
import mixed_gds_arm as core


_CORE_DEFAULTS = core.StudyRunSettings()
_ALLOWED_VARIANTS = ("weighted_only", "adaptive_sampling_only")
_SAMPLING_TABLES_DIRNAME = "sampling_tables"
_SAMPLING_PLOTS_DIRNAME = "sampling_plots"
_SAMPLING_SETTINGS_FILENAME = "sampling_run_settings.json"
_BASELINE_P_ENTER = 0.01
_BASELINE_P_REMOVE = 0.05


@dataclass(slots=True)
class AdaptiveSamplingComparisonSettings:
    output_dir: str | os.PathLike[str] = "study_outputs_aws_sampling"
    run_name: str = "adaptive_sampling_phase1"
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
    write_sampling_history_table: bool = True
    write_plots: bool = True
    write_settings_file: bool = True
    show_progress: bool = True


def _settings_payload(settings: AdaptiveSamplingComparisonSettings) -> dict[str, Any]:
    payload = asdict(settings)
    payload["output_dir"] = str(settings.output_dir)
    return payload


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_manifest(settings: AdaptiveSamplingComparisonSettings) -> dict[str, Any]:
    manifest = _settings_payload(settings)
    manifest["sampling_source_sha256"] = _source_sha256(Path(__file__))
    manifest["weighted_source_sha256"] = _source_sha256(Path(shared.__file__))
    manifest["core_source_sha256"] = core._source_sha256()
    manifest["cache_version"] = 1
    return json.loads(json.dumps(manifest, sort_keys=True))


def _validate_settings(settings: AdaptiveSamplingComparisonSettings) -> None:
    if int(settings.reps_per_condition) <= 0:
        raise ValueError("reps_per_condition must be positive.")
    if int(settings.benchmark_nrep) <= 0:
        raise ValueError("benchmark_nrep must be positive.")
    if int(settings.sampling_batch_size) <= 0:
        raise ValueError("sampling_batch_size must be positive.")
    if int(settings.sampling_warmup_batches) < 0:
        raise ValueError("sampling_warmup_batches must be non-negative.")

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
    if int(settings.benchmark_nrep) < max(ntop_values):
        raise ValueError("benchmark_nrep must be at least max(ntop_values).")
    if not 0.0 <= float(settings.sampling_adaptation_lambda) <= 1.0:
        raise ValueError("sampling_adaptation_lambda must lie in [0, 1].")
    if not 0.0 <= float(settings.sampling_heredity_eta) <= 1.0:
        raise ValueError("sampling_heredity_eta must lie in [0, 1].")
    if float(settings.sampling_block_floor) < 0.0:
        raise ValueError("sampling_block_floor must be non-negative.")

    invalid_variants = sorted(set(settings.variants) - set(_ALLOWED_VARIANTS))
    if invalid_variants:
        raise ValueError(
            f"Unsupported variants: {invalid_variants}. Expected any of {_ALLOWED_VARIANTS}."
        )


def _interaction_block_table(metadata: core.MixedLevelDesign) -> pd.DataFrame:
    blocks = metadata.effect_table[metadata.effect_table["effect_type"] == "interaction"].copy()
    if blocks.empty:
        return pd.DataFrame(
            columns=["effect_id", "factor_i", "factor_j", "column_ids", "n_columns"]
        )
    blocks["factor_i"] = blocks["factor_ids"].map(lambda ids: str(ids[0]))
    blocks["factor_j"] = blocks["factor_ids"].map(lambda ids: str(ids[1]))
    return blocks[["effect_id", "factor_i", "factor_j", "column_ids", "n_columns"]].reset_index(
        drop=True
    )


def _weighted_effect_scores(
    repetition_results: pd.DataFrame,
    metadata: core.MixedLevelDesign,
) -> pd.Series:
    effect_ids = metadata.effect_ids
    if not effect_ids:
        return pd.Series(dtype=float)

    weights = shared._bic_model_weights(repetition_results)
    scores = {effect_id: 0.0 for effect_id in effect_ids}
    for weight, selected_effects in zip(weights, repetition_results["selected_effects"], strict=False):
        for effect_id in selected_effects:
            if effect_id in scores:
                scores[effect_id] += float(weight)
    return pd.Series(scores, dtype=float).reindex(effect_ids, fill_value=0.0)


def _normalized_entropy(probabilities: pd.Series) -> float:
    if probabilities.empty:
        return 0.0
    probs = probabilities.to_numpy(dtype=float)
    probs = probs[probs > 0.0]
    if probs.size <= 1:
        return 0.0
    entropy = float(-(probs * np.log(probs)).sum())
    max_entropy = float(np.log(probs.size))
    return entropy / max_entropy if max_entropy > 0.0 else 0.0


def _uniform_interaction_column_probabilities(metadata: core.MixedLevelDesign) -> pd.Series:
    columns = metadata.X_int.columns.tolist()
    if not columns:
        return pd.Series(dtype=float)
    probability = 1.0 / len(columns)
    return pd.Series(probability, index=columns, dtype=float)


def _block_probabilities_to_column_probabilities(
    metadata: core.MixedLevelDesign,
    block_probabilities: pd.Series,
) -> pd.Series:
    columns = metadata.X_int.columns.tolist()
    if not columns:
        return pd.Series(dtype=float)

    interaction_blocks = _interaction_block_table(metadata)
    column_probabilities = pd.Series(0.0, index=columns, dtype=float)
    for row in interaction_blocks.itertuples(index=False):
        block_mass = float(block_probabilities.get(str(row.effect_id), 0.0))
        block_columns = list(row.column_ids)
        if not block_columns:
            continue
        column_probabilities.loc[block_columns] = block_mass / len(block_columns)

    total = float(column_probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        return _uniform_interaction_column_probabilities(metadata)
    return column_probabilities / total


def _update_interaction_block_probabilities(
    metadata: core.MixedLevelDesign,
    effect_scores: pd.Series,
    *,
    adaptation_lambda: float,
    heredity_eta: float,
    block_floor: float,
) -> pd.Series:
    interaction_blocks = _interaction_block_table(metadata)
    if interaction_blocks.empty:
        return pd.Series(dtype=float)

    n_blocks = int(len(interaction_blocks))
    main_scores = {
        factor_id: float(effect_scores.get(metadata.factor_to_main_effect[factor_id], 0.0))
        for factor_id in metadata.factor_ids
        if factor_id in metadata.factor_to_main_effect
    }

    raw_scores: dict[str, float] = {}
    for row in interaction_blocks.itertuples(index=False):
        direct_evidence = float(effect_scores.get(str(row.effect_id), 0.0))
        heredity_evidence = (
            float(main_scores.get(str(row.factor_i), 0.0))
            * float(main_scores.get(str(row.factor_j), 0.0))
        )
        raw_scores[str(row.effect_id)] = (
            float(block_floor)
            + (1.0 - float(adaptation_lambda)) / n_blocks
            + float(adaptation_lambda)
            * ((1.0 - float(heredity_eta)) * direct_evidence + float(heredity_eta) * heredity_evidence)
        )

    probabilities = pd.Series(raw_scores, dtype=float).clip(lower=0.0)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        return pd.Series(1.0 / n_blocks, index=interaction_blocks["effect_id"], dtype=float)
    return probabilities / total


def _sample_interaction_columns(
    metadata: core.MixedLevelDesign,
    *,
    nint: int,
    column_probabilities: pd.Series,
    random_state: int | np.random.Generator | None,
) -> list[str]:
    rng = core._ensure_rng(random_state)
    columns = metadata.X_int.columns.to_numpy()
    nint = int(min(max(int(nint), 0), len(columns)))
    if nint <= 0:
        return []
    if nint >= len(columns):
        return columns.tolist()

    probabilities = column_probabilities.reindex(columns).astype(float).to_numpy()
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(len(columns), 1.0 / len(columns), dtype=float)
    else:
        probabilities = probabilities / total
    return rng.choice(columns, size=nint, replace=False, p=probabilities).tolist()


def _run_GDS_step_i_with_sampled_interactions(
    design: core.MixedLevelDesign,
    y: Sequence[float] | pd.Series | np.ndarray,
    *,
    sampled_interaction_columns: Sequence[str],
    delta_grid: Sequence[float] | None,
    random_state: int | np.random.Generator | None,
    solver_order: Sequence[str] | None,
    paper_exact_baseline: bool,
) -> core.GDSStepIResult:
    rng = core._ensure_rng(random_state)
    column_lookup = design.column_table.set_index("column_id", drop=False)
    main_columns = design.X_main.columns.tolist()
    sampled_columns = [str(column_id) for column_id in sampled_interaction_columns]
    rep_columns = main_columns + sampled_columns
    rep_X_full, rep_column_table = core._subset_gds_problem(design, column_lookup, rep_columns)
    y_centered = core.center_response(y)
    current_delta_grid = core._resolve_gds_delta_grid(
        rep_X_full,
        y_centered,
        delta_grid,
        paper_exact_baseline=paper_exact_baseline,
    )
    gds_result = core.run_GDS(
        rep_X_full,
        y_centered,
        effect_metadata=rep_column_table,
        delta_grid=current_delta_grid,
        random_state=int(rng.integers(0, 2**31 - 1)),
        design_id=f"{design.design_id}_adaptive_sampling_step_i",
        solver_order=solver_order,
        paper_exact_kmeans=paper_exact_baseline,
        full_column_count=design.X_full.shape[1],
    )
    return core.GDSStepIResult(
        design_id=f"{design.design_id}_adaptive_sampling_step_i",
        sampled_interaction_columns=sampled_columns,
        gds_result=gds_result,
    )


def _build_adaptive_sampling_repetition_path(
    design: core.MixedLevelDesign,
    y: Sequence[float] | pd.Series | np.ndarray,
    *,
    nrep: int,
    nint: int,
    batch_size: int,
    warmup_batches: int,
    adaptation_lambda: float,
    heredity_eta: float,
    block_floor: float,
    delta_grid: Sequence[float] | None,
    random_state: int | np.random.Generator | None,
    solver_order: Sequence[str] | None,
    paper_exact_baseline: bool,
    rep_seeds: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = core._ensure_rng(random_state)
    interaction_blocks = _interaction_block_table(design)
    interaction_block_probabilities = (
        pd.Series(1.0 / len(interaction_blocks), index=interaction_blocks["effect_id"], dtype=float)
        if not interaction_blocks.empty
        else pd.Series(dtype=float)
    )

    repetition_batches: list[pd.DataFrame] = []
    history_rows: list[dict[str, Any]] = []
    total_reps = 0
    batch_index = 0
    nrep = int(max(int(nrep), 0))

    while total_reps < nrep:
        batch_index += 1
        current_batch_size = min(int(batch_size), nrep - total_reps)
        use_uniform_sampling = batch_index <= int(warmup_batches) or interaction_block_probabilities.empty
        if use_uniform_sampling:
            column_probabilities = _uniform_interaction_column_probabilities(design)
        else:
            column_probabilities = _block_probabilities_to_column_probabilities(
                design,
                interaction_block_probabilities,
            )

        batch_rows: list[dict[str, Any]] = []
        for offset in range(current_batch_size):
            rep_seed = (
                int(rep_seeds[total_reps + offset])
                if rep_seeds is not None
                else int(rng.integers(0, 2**31 - 1))
            )
            rep_rng = np.random.default_rng(rep_seed)
            sampled_columns = _sample_interaction_columns(
                design,
                nint=nint,
                column_probabilities=column_probabilities,
                random_state=rep_rng,
            )
            step_i_result = _run_GDS_step_i_with_sampled_interactions(
                design,
                y,
                sampled_interaction_columns=sampled_columns,
                delta_grid=delta_grid,
                random_state=rep_rng,
                solver_order=solver_order,
                paper_exact_baseline=paper_exact_baseline,
            )
            gds_result = step_i_result.gds_result
            batch_rows.append(
                {
                    "rep": total_reps + offset + 1,
                    "batch_index": batch_index,
                    "sampling_mode": "uniform_warmup" if use_uniform_sampling else "adaptive",
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

        batch_results = pd.DataFrame.from_records(batch_rows)
        repetition_batches.append(batch_results)
        repetition_results = pd.concat(repetition_batches, ignore_index=True)
        total_reps = int(len(repetition_results))

        effect_scores = _weighted_effect_scores(repetition_results, design)
        if not interaction_block_probabilities.empty:
            interaction_block_probabilities = _update_interaction_block_probabilities(
                design,
                effect_scores,
                adaptation_lambda=float(adaptation_lambda),
                heredity_eta=float(heredity_eta),
                block_floor=float(block_floor),
            )

        main_effect_ids = design.effect_table.loc[
            design.effect_table["effect_type"] == "main",
            "effect_id",
        ].tolist()
        interaction_effect_ids = interaction_blocks["effect_id"].tolist()
        top_block_effect = ""
        top_block_probability = np.nan
        if not interaction_block_probabilities.empty:
            top_block_effect = str(interaction_block_probabilities.idxmax())
            top_block_probability = float(interaction_block_probabilities.max())

        history_rows.append(
            {
                "batch_index": batch_index,
                "batch_size": int(current_batch_size),
                "nrep_used": total_reps,
                "sampling_mode": "uniform_warmup" if use_uniform_sampling else "adaptive",
                "column_sampling_entropy": float(_normalized_entropy(column_probabilities)),
                "block_sampling_entropy": float(_normalized_entropy(interaction_block_probabilities)),
                "mean_main_score": float(effect_scores.reindex(main_effect_ids).mean()) if main_effect_ids else np.nan,
                "mean_interaction_score": float(effect_scores.reindex(interaction_effect_ids).mean())
                if interaction_effect_ids
                else np.nan,
                "top_block_effect": top_block_effect,
                "top_block_probability": top_block_probability,
            }
        )

    repetition_results = (
        pd.concat(repetition_batches, ignore_index=True)
        if repetition_batches
        else pd.DataFrame.from_records([])
    )
    sampling_history = pd.DataFrame.from_records(history_rows)
    last_row = history_rows[-1] if history_rows else {}
    diagnostics = {
        "sampling_rule": "adaptive_interaction_blocks",
        "stopping_rule": "fixed_nrep",
        "stop_reason": "fixed_nrep",
        "n_batches_used": int(batch_index),
        "nrep_used": int(len(repetition_results)),
        "stopped_early": False,
        "warmup_batches": int(warmup_batches),
        "sampling_batch_size": int(batch_size),
        "sampling_adaptation_lambda": float(adaptation_lambda),
        "sampling_heredity_eta": float(heredity_eta),
        "sampling_block_floor": float(block_floor),
        "last_score_delta": np.nan,
        "last_set_jaccard": np.nan,
        "stable_batches_achieved": 0,
        "last_column_sampling_entropy": float(last_row.get("column_sampling_entropy", np.nan)),
        "last_block_sampling_entropy": float(last_row.get("block_sampling_entropy", np.nan)),
        "last_top_block_effect": str(last_row.get("top_block_effect", "")),
        "last_top_block_probability": float(last_row.get("top_block_probability", np.nan)),
    }
    return repetition_results, sampling_history, diagnostics


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
    ranked_results = repetition_results.sort_values("bic", ascending=True, kind="stable").reset_index(
        drop=True
    )
    top_n = max(int(ntop), 0)
    top_results = (
        ranked_results.head(min(top_n, len(ranked_results))).copy()
        if top_n > 0
        else ranked_results.head(0).copy()
    )

    if variant not in _ALLOWED_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    feature_table, aggregated_effects, aggregated_columns, aggregation_diagnostics = shared._weighted_aggregate(
        top_results,
        metadata,
        tau=weighted_tau,
        paper_exact_baseline=paper_exact_baseline,
    )

    (
        final_stepwise_result,
        final_selected_effects,
        final_selected_columns,
        final_selected_factors,
    ) = shared._finalize_selection(
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


def _iter_sampling_comparison_tasks(
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
                    "sampling_batch_size": int(sampling_batch_size),
                    "sampling_warmup_batches": int(sampling_warmup_batches),
                    "sampling_adaptation_lambda": float(sampling_adaptation_lambda),
                    "sampling_heredity_eta": float(sampling_heredity_eta),
                    "sampling_block_floor": float(sampling_block_floor),
                }
                task_id += 1


def _run_sampling_comparison_replication(
    task: Mapping[str, Any],
) -> tuple[int, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
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
    shared_rep_seeds = [
        int(rng.integers(0, 2**31 - 1))
        for _ in range(int(task["benchmark_nrep"]))
    ]

    path_results_by_variant: dict[str, dict[str, Any]] = {}
    if "weighted_only" in task["variants"]:
        path_t0 = time.perf_counter()
        repetition_results = shared._build_repetition_path(
            design,
            dataset["y"],
            nrep=int(task["benchmark_nrep"]),
            nint=resolved_nint,
            delta_grid=current_delta_grid,
            random_state=None,
            solver_order=None,
            paper_exact_baseline=bool(task["paper_exact_baseline"]),
            rep_seeds=shared_rep_seeds,
        )
        path_elapsed = time.perf_counter() - path_t0
        path_results_by_variant["weighted_only"] = {
            "repetition_results": repetition_results,
            "path_elapsed": float(path_elapsed),
            "path_diagnostics": {
                "sampling_rule": "uniform_interaction_columns",
                "stopping_rule": "fixed_nrep",
                "stop_reason": "fixed_nrep",
                "nrep_used": int(len(repetition_results)),
                "stopped_early": False,
                "n_batches_used": 1,
                "last_score_delta": np.nan,
                "last_set_jaccard": np.nan,
                "stable_batches_achieved": 0,
                "last_column_sampling_entropy": 1.0 if design.X_int.shape[1] > 1 else 0.0,
                "last_block_sampling_entropy": 1.0
                if len(_interaction_block_table(design)) > 1
                else 0.0,
                "last_top_block_effect": "",
                "last_top_block_probability": np.nan,
            },
            "sampling_history": pd.DataFrame(),
        }

    if "adaptive_sampling_only" in task["variants"]:
        path_t0 = time.perf_counter()
        repetition_results, sampling_history, path_diagnostics = _build_adaptive_sampling_repetition_path(
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
            rep_seeds=shared_rep_seeds,
        )
        path_elapsed = time.perf_counter() - path_t0
        path_results_by_variant["adaptive_sampling_only"] = {
            "repetition_results": repetition_results,
            "path_elapsed": float(path_elapsed),
            "path_diagnostics": path_diagnostics,
            "sampling_history": sampling_history,
        }

    replication_rows: list[dict[str, Any]] = []
    sampling_history_rows: list[dict[str, Any]] = []
    config_order = 0
    for ntop in task["ntop_values"]:
        for pkeep in task["pkeep_values"]:
            weighted_tau = shared._resolve_weighted_tau(
                float(pkeep),
                task["weighted_tau_override"],
            )
            for variant_idx, variant in enumerate(task["variants"]):
                variant_path = path_results_by_variant[str(variant)]
                variant_t0 = time.perf_counter()
                variant_result = _build_variant_result(
                    variant_path["repetition_results"],
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
                diagnostics = {
                    **variant_result["diagnostics"],
                    **variant_path["path_diagnostics"],
                }
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
                        "weighted_tau": float(weighted_tau),
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
                        "last_column_sampling_entropy": float(
                            diagnostics["last_column_sampling_entropy"]
                        ),
                        "last_block_sampling_entropy": float(
                            diagnostics["last_block_sampling_entropy"]
                        ),
                        "last_top_block_effect": str(diagnostics["last_top_block_effect"]),
                        "last_top_block_probability": float(
                            diagnostics["last_top_block_probability"]
                        ),
                        "selected_factors": tuple(variant_result["final_selected_factors"]),
                        "selected_effects": tuple(variant_result["final_selected_effects"]),
                        "selected_columns": tuple(variant_result["final_selected_columns"]),
                    }
                )

                if str(variant) == "adaptive_sampling_only" and ntop == task["ntop_values"][0] and pkeep == task["pkeep_values"][0]:
                    history_df = variant_path["sampling_history"]
                    if not history_df.empty:
                        history_rows = history_df.assign(
                            task_id=int(task["task_id"]),
                            design_id=design.design_id,
                            scenario_id=scenario_id,
                            dataset_rep=dataset_rep,
                            variant=str(variant),
                            nint=int(resolved_nint),
                        )
                        sampling_history_rows.extend(history_rows.to_dict("records"))
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
    return int(task["task_id"]), replication_rows, truth_row, sampling_history_rows


def _append_sampling_result(
    result: tuple[int, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]],
    replication_items: list[dict[str, Any]],
    truth_items: list[dict[str, Any]],
    sampling_history_items: list[dict[str, Any]],
) -> None:
    _task_id, replication_rows, truth_row, sampling_history_rows = result
    replication_items.extend(replication_rows)
    truth_items.append(truth_row)
    sampling_history_items.extend(sampling_history_rows)


def _run_serial_tasks(
    task_iter: Iterable[Mapping[str, Any]],
    *,
    progress: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    replication_items: list[dict[str, Any]] = []
    truth_items: list[dict[str, Any]] = []
    sampling_history_items: list[dict[str, Any]] = []
    for task in task_iter:
        _append_sampling_result(
            _run_sampling_comparison_replication(task),
            replication_items,
            truth_items,
            sampling_history_items,
        )
        progress.update(1)
    return replication_items, truth_items, sampling_history_items


def _run_parallel_tasks(
    task_iter: Iterable[Mapping[str, Any]],
    *,
    worker_count: int,
    design_registry: Mapping[str, core.MixedLevelDesign],
    progress: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    replication_items: list[dict[str, Any]] = []
    truth_items: list[dict[str, Any]] = []
    sampling_history_items: list[dict[str, Any]] = []
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
            pending[executor.submit(_run_sampling_comparison_replication, task)] = None

        while pending:
            done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future, None)
                _append_sampling_result(
                    future.result(),
                    replication_items,
                    truth_items,
                    sampling_history_items,
                )
                progress.update(1)
                try:
                    next_task = next(task_iter)
                except StopIteration:
                    continue
                pending[executor.submit(_run_sampling_comparison_replication, next_task)] = None

    return replication_items, truth_items, sampling_history_items


def _sampling_plot_data(summary_results: pd.DataFrame) -> pd.DataFrame:
    plot_data = summary_results.copy()
    plot_data["design_label"] = plot_data["design_id"].map(core._paper_design_label)
    plot_data["scenario_label"] = plot_data["scenario_id"].map(core._paper_scenario_label)
    plot_data["scenario_id_int"] = pd.to_numeric(plot_data["scenario_id"], errors="coerce")
    plot_data["variant_label"] = plot_data["variant"].map(
        {
            "weighted_only": "Weighted Only",
            "adaptive_sampling_only": "Adaptive sampling Only",
        }
    ).fillna(plot_data["variant"].astype(str))
    plot_data["condition_label"] = (
        plot_data["design_label"].astype(str) + " | " + plot_data["scenario_label"].astype(str)
    )
    return plot_data


def _sampling_condition_ranking(summary_results: pd.DataFrame) -> pd.DataFrame:
    ranked = _sampling_plot_data(summary_results)
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


def plot_sampling_ntop_pkeep_scatter_plotnine(data: pd.DataFrame, metric: str = "score"):
    return common.plot_ntop_pkeep_scatter_plotnine(
        data,
        metric=metric,
        title=f"AWS-GDS-ARM Mean vs SD for (ntop, pkeep): {metric}",
        subtitle="Weighted-only benchmark versus adaptive interaction sampling.",
    )


def plot_sampling_metric_by_scenario_plotnine(data: pd.DataFrame, metric: str = "score"):
    return common.plot_metric_by_scenario_plotnine(
        data,
        metric=metric,
        title=f"AWS-GDS-ARM Mean {metric.title()} by Scenario",
        subtitle="Weighted-only benchmark versus adaptive interaction sampling.",
    )


def plot_sampling_condition_scatter_plotnine(data: pd.DataFrame, metric: str = "score"):
    return common.plot_condition_scatter_plotnine(
        data,
        metric=metric,
        title=f"AWS-GDS-ARM Mean vs SD: {metric}",
        subtitle="Condition-level benchmark comparison for adaptive interaction sampling.",
    )


def plot_sampling_runtime_by_scenario_plotnine(data: pd.DataFrame):
    return common.plot_runtime_by_scenario_plotnine(
        data,
        title="AWS-GDS-ARM Mean Runtime by Scenario",
        subtitle="Weighted-only benchmark versus adaptive interaction sampling.",
    )


def run_adaptive_sampling_benchmark_study(
    settings: AdaptiveSamplingComparisonSettings | None = None,
    /,
    **overrides: Any,
) -> dict[str, Any]:
    if settings is None:
        settings = AdaptiveSamplingComparisonSettings(**overrides)
    elif overrides:
        settings = replace(settings, **overrides)

    _validate_settings(settings)
    benchmark_tau = shared._resolve_weighted_tau(float(settings.benchmark_pkeep), settings.weighted_tau)
    output_path = Path(settings.output_dir)
    tables_dir = output_path / _SAMPLING_TABLES_DIRNAME
    plots_dir = output_path / _SAMPLING_PLOTS_DIRNAME
    manifest_path = output_path / _SAMPLING_SETTINGS_FILENAME
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
    diagnostics.to_csv(tables_dir / "sampling_design_diagnostics.csv", index=False)
    scenario_map = context.scenario_map

    progress_total = sum(
        len(scenario_map[design.design_id]) * int(settings.reps_per_condition)
        for design in designs
    )
    progress = core._make_progress(
        total=progress_total,
        desc="AWS sampling benchmark",
        disable=not settings.show_progress,
        leave=True,
    )

    design_registry = context.design_registry
    core._set_simulation_design_registry(design_registry)
    task_seed = int(rng.integers(0, 2**31 - 1))
    worker_count = min(core._resolve_n_jobs(settings.n_jobs), max(progress_total, 1))

    def _task_iter():
        return _iter_sampling_comparison_tasks(
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
            sampling_batch_size=int(settings.sampling_batch_size),
            sampling_warmup_batches=int(settings.sampling_warmup_batches),
            sampling_adaptation_lambda=float(settings.sampling_adaptation_lambda),
            sampling_heredity_eta=float(settings.sampling_heredity_eta),
            sampling_block_floor=float(settings.sampling_block_floor),
        )

    try:
        if worker_count <= 1 or progress_total <= 1:
            replication_items, truth_items, sampling_history_items = _run_serial_tasks(
                _task_iter(),
                progress=progress,
            )
        else:
            try:
                replication_items, truth_items, sampling_history_items = _run_parallel_tasks(
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
                    desc="AWS sampling benchmark",
                    disable=not settings.show_progress,
                    leave=True,
                )
                replication_items, truth_items, sampling_history_items = _run_serial_tasks(
                    _task_iter(),
                    progress=progress,
                )
    finally:
        progress.close()

    replication_results = pd.DataFrame.from_records(replication_items)
    truth_records = pd.DataFrame.from_records(truth_items)
    sampling_history = pd.DataFrame.from_records(sampling_history_items)
    if not replication_results.empty:
        replication_results = replication_results.sort_values(
            ["task_id", "config_order", "variant_order"],
            kind="stable",
        ).drop(columns=["task_id", "config_order", "variant_order"]).reset_index(drop=True)
    if not truth_records.empty:
        truth_records = truth_records.sort_values(["task_id"], kind="stable").drop(
            columns=["task_id"]
        ).reset_index(drop=True)
    if not sampling_history.empty:
        sampling_history = sampling_history.sort_values(
            ["task_id", "batch_index"],
            kind="stable",
        ).drop(columns=["task_id"]).reset_index(drop=True)

    summary_results = shared._summarize_results(replication_results)
    benchmark_mask = common.build_benchmark_mask(
        summary_results,
        benchmark_ntop=int(settings.benchmark_ntop),
        benchmark_pkeep=float(settings.benchmark_pkeep),
        benchmark_tau=float(benchmark_tau),
    )
    benchmark_summary_results = summary_results.loc[benchmark_mask].copy().reset_index(drop=True)
    ntop_pkeep_aggregate = shared.adaptive_ntop_pkeep_aggregate(summary_results)
    overall_summary = shared._overall_summary(benchmark_summary_results)
    design_summary = shared.adaptive_design_summary(benchmark_summary_results)
    condition_ranking = _sampling_condition_ranking(benchmark_summary_results)
    comparison_table = shared._variant_comparison(benchmark_summary_results)

    summary_results.to_csv(tables_dir / "sampling_ntop_pkeep_summary_results.csv", index=False)
    benchmark_summary_results.to_csv(tables_dir / "sampling_variant_summary_results.csv", index=False)
    overall_summary.to_csv(tables_dir / "sampling_variant_overall_summary.csv", index=False)
    design_summary.to_csv(tables_dir / "sampling_design_summary.csv", index=False)
    condition_ranking.to_csv(tables_dir / "sampling_condition_ranking.csv", index=False)
    comparison_table.to_csv(tables_dir / "sampling_variant_condition_comparison.csv", index=False)
    ntop_pkeep_aggregate.to_csv(tables_dir / "sampling_ntop_pkeep_aggregate.csv", index=False)
    truth_records.to_csv(tables_dir / "sampling_truth_records.csv", index=False)
    if settings.write_replication_tables:
        replication_results.to_csv(tables_dir / "sampling_variant_replication_results.csv", index=False)
    if settings.write_sampling_history_table and not sampling_history.empty:
        sampling_history.to_csv(tables_dir / "sampling_history.csv", index=False)
    if settings.write_plots:
        core._save_plotnine_plot(
            plot_sampling_ntop_pkeep_scatter_plotnine(ntop_pkeep_aggregate, metric="score"),
            plots_dir / "sampling_ntop_pkeep_score_plotnine.png",
        )
        core._save_plotnine_plot(
            plot_sampling_ntop_pkeep_scatter_plotnine(ntop_pkeep_aggregate, metric="power"),
            plots_dir / "sampling_ntop_pkeep_power_plotnine.png",
        )
        core._save_plotnine_plot(
            plot_sampling_metric_by_scenario_plotnine(benchmark_summary_results, metric="score"),
            plots_dir / "sampling_score_by_scenario_plotnine.png",
        )
        core._save_plotnine_plot(
            plot_sampling_metric_by_scenario_plotnine(benchmark_summary_results, metric="power"),
            plots_dir / "sampling_power_by_scenario_plotnine.png",
        )
        core._save_plotnine_plot(
            plot_sampling_condition_scatter_plotnine(benchmark_summary_results, metric="score"),
            plots_dir / "sampling_score_scatter_plotnine.png",
        )
        core._save_plotnine_plot(
            plot_sampling_runtime_by_scenario_plotnine(benchmark_summary_results),
            plots_dir / "sampling_runtime_by_scenario_plotnine.png",
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
        "Fixed weighted benchmark tuning carried over from the baseline: "
        f"nrep={settings.benchmark_nrep}, nint={settings.benchmark_nint}"
    )
    print(
        "AWS ntop/pkeep grid: "
        f"ntop={tuple(settings.ntop_values)}, pkeep={tuple(settings.pkeep_values)}"
    )
    print(
        "Benchmark scenario summaries use: "
        f"ntop={settings.benchmark_ntop}, pkeep={settings.benchmark_pkeep}, tau={benchmark_tau}"
    )
    print(
        "Adaptive sampling settings: "
        f"batch_size={settings.sampling_batch_size}, "
        f"warmup_batches={settings.sampling_warmup_batches}, "
        f"lambda={settings.sampling_adaptation_lambda}, "
        f"eta={settings.sampling_heredity_eta}, "
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
        "sampling_history": sampling_history,
        "summary_results": summary_results,
        "benchmark_summary_results": benchmark_summary_results,
        "ntop_pkeep_aggregate": ntop_pkeep_aggregate,
        "overall_summary": overall_summary,
        "design_summary": design_summary,
        "condition_ranking": condition_ranking,
        "comparison_table": comparison_table,
    }


__all__ = [
    "AdaptiveSamplingComparisonSettings",
    "plot_sampling_ntop_pkeep_scatter_plotnine",
    "plot_sampling_metric_by_scenario_plotnine",
    "plot_sampling_condition_scatter_plotnine",
    "plot_sampling_runtime_by_scenario_plotnine",
    "run_adaptive_sampling_benchmark_study",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the adaptive interaction-sampling AWS-GDS-ARM benchmark on the same design/scenario settings used by the baseline."
    )
    parser.add_argument(
        "--output-dir",
        default="study_outputs_aws_sampling",
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
        help="Fixed-nrep weighted benchmark used for all variants.",
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
        help="Number of initial uniform-sampling batches before adaptation begins.",
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
        help="Weight on strong-heredity evidence inside the adaptive interaction update.",
    )
    parser.add_argument(
        "--sampling-block-floor",
        type=float,
        default=1e-3,
        help="Per-block exploration floor in the adaptive interaction update.",
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
        help="Variant to include.",
    )
    if any(arg.startswith("--f=") for arg in sys.argv[1:]) or "ipykernel_launcher" in Path(sys.argv[0]).name:
        print("Detected Jupyter kernel arguments; skipping CLI entry point.")
    else:
        args, _unknown = parser.parse_known_args()
        run_adaptive_sampling_benchmark_study(
            AdaptiveSamplingComparisonSettings(
                output_dir=args.output_dir,
                random_state=args.seed,
                reps_per_condition=args.reps_per_condition,
                benchmark_nrep=args.benchmark_nrep,
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
