"""Phase-II AWS-GDS-ARM adaptive-nrep study.

This module keeps the generated-design defaults from ``mixed_gds_arm.py`` and
extends the weighted-only comparison in ``aws_gds_arm.py`` by adding an adaptive
sequential stopping rule for ``nrep``.

The two currently supported variants are:

- ``weighted_only``: fixed-``nrep`` BIC-weighted inclusion scores over the top
  ``ntop`` models, thresholded at ``tau`` where ``tau = pkeep`` by default
- ``adaptive_nrep_only``: the same weighted aggregation rule, but the
  repetition path is built in batches and stops early once the weighted
  evidence scores stabilize

This keeps adaptive stopping isolated from adaptive interaction sampling so the
incremental value of sequential stopping can be compared directly against the
weighted fixed-``nrep`` benchmark.
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
_ALLOWED_VARIANTS = ("weighted_only", "adaptive_nrep_only")
_ADAPTIVE_TABLES_DIRNAME = "adaptive_tables"
_ADAPTIVE_PLOTS_DIRNAME = "adaptive_plots"
_ADAPTIVE_SETTINGS_FILENAME = "adaptive_run_settings.json"
_BASELINE_P_ENTER = 0.01
_BASELINE_P_REMOVE = 0.05


@dataclass(slots=True)
class AdaptiveComparisonSettings:
    output_dir: str | os.PathLike[str] = "study_outputs_aws_adaptive"
    run_name: str = "adaptive_nrep_phase1"
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
    adaptive_max_nrep: int = _CORE_DEFAULTS.ntop_pkeep_nrep
    adaptive_batch_size: int = 50
    adaptive_min_nrep: int = max(_CORE_DEFAULTS.nrep_study_ntop, 100)
    adaptive_stability_epsilon: float = 0.01
    adaptive_patience: int = 2
    adaptive_require_set_stability: bool = False
    adaptive_min_jaccard: float = 0.95
    adaptive_set_tau: float | None = None
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


def _settings_payload(settings: AdaptiveComparisonSettings) -> dict[str, Any]:
    payload = asdict(settings)
    payload["output_dir"] = str(settings.output_dir)
    return payload


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_manifest(settings: AdaptiveComparisonSettings) -> dict[str, Any]:
    manifest = _settings_payload(settings)
    manifest["aws_source_sha256"] = _source_sha256(Path(__file__))
    manifest["core_source_sha256"] = core._source_sha256()
    manifest["cache_version"] = 1
    return json.loads(json.dumps(manifest, sort_keys=True))


def _resolve_weighted_tau(pkeep: float, weighted_tau_override: float | None) -> float:
    return float(pkeep if weighted_tau_override is None else weighted_tau_override)


def _validate_settings(settings: AdaptiveComparisonSettings) -> None:
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
    if int(settings.adaptive_min_nrep) > int(settings.adaptive_max_nrep):
        raise ValueError("adaptive_min_nrep cannot exceed adaptive_max_nrep.")
    if int(settings.adaptive_min_nrep) < max(ntop_values):
        raise ValueError("adaptive_min_nrep must be at least max(ntop_values).")
    if settings.adaptive_min_jaccard < 0.0 or settings.adaptive_min_jaccard > 1.0:
        raise ValueError("adaptive_min_jaccard must lie in [0, 1].")
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


def _weighted_feature_table(
    repetition_results: pd.DataFrame,
    metadata: core.MixedLevelDesign,
    *,
    paper_exact_baseline: bool,
) -> pd.DataFrame:
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

        n_rows = max(len(repetition_results), 1)
        rows = []
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
                    "effect_type": metadata.column_to_metadata[column_id]["effect_type"],
                }
            )
        return pd.DataFrame(rows)

    weighted_scores = {}
    inclusion_counts = {}
    for weight, selected_effects in zip(weights, repetition_results["selected_effects"], strict=False):
        for effect_id in selected_effects:
            weighted_scores[effect_id] = weighted_scores.get(effect_id, 0.0) + float(weight)
            inclusion_counts[effect_id] = inclusion_counts.get(effect_id, 0) + 1

    n_rows = max(len(repetition_results), 1)
    rows = []
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
                "effect_type": effect_type_lookup[effect_id],
            }
        )
    return pd.DataFrame(rows)


def _weighted_score_snapshot(
    repetition_results: pd.DataFrame,
    metadata: core.MixedLevelDesign,
    *,
    tau: float,
    paper_exact_baseline: bool,
) -> tuple[pd.Series, frozenset[str]]:
    feature_table = _weighted_feature_table(
        repetition_results,
        metadata,
        paper_exact_baseline=paper_exact_baseline,
    )
    scores = feature_table.set_index("feature_id")["weighted_score"].astype(float)
    kept = frozenset(
        feature_table.loc[feature_table["weighted_score"] >= float(tau), "feature_id"].astype(str).tolist()
    )
    return scores, kept


def _jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right) / len(union))


def _build_adaptive_repetition_path(
    design: core.MixedLevelDesign,
    y: Sequence[float] | pd.Series | np.ndarray,
    *,
    max_nrep: int,
    batch_size: int,
    min_nrep: int,
    stability_epsilon: float,
    patience: int,
    stability_tau: float,
    require_set_stability: bool,
    min_jaccard: float,
    nint: int,
    delta_grid: Sequence[float] | None,
    random_state: int | np.random.Generator | None,
    solver_order: Sequence[str] | None,
    paper_exact_baseline: bool,
    rep_seeds: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = core._ensure_rng(random_state)
    repetition_batches: list[pd.DataFrame] = []
    history_rows: list[dict[str, Any]] = []
    previous_scores: pd.Series | None = None
    previous_selected: frozenset[str] | None = None
    stable_batches = 0
    total_reps = 0
    batch_index = 0
    stop_reason = "max_nrep_reached"

    while total_reps < int(max_nrep):
        current_batch_size = min(int(batch_size), int(max_nrep) - total_reps)
        batch_results = _build_repetition_path(
            design,
            y,
            nrep=current_batch_size,
            nint=nint,
            delta_grid=delta_grid,
            random_state=int(rng.integers(0, 2**31 - 1)),
            solver_order=solver_order,
            paper_exact_baseline=paper_exact_baseline,
            rep_seeds=(
                None
                if rep_seeds is None
                else rep_seeds[total_reps : total_reps + current_batch_size]
            ),
        )
        if not batch_results.empty:
            batch_results = batch_results.copy()
            batch_results["rep"] = np.arange(total_reps + 1, total_reps + len(batch_results) + 1, dtype=int)
            repetition_batches.append(batch_results)
            repetition_results = pd.concat(repetition_batches, ignore_index=True)
        else:
            repetition_results = pd.DataFrame()

        total_reps = int(len(repetition_results))
        batch_index += 1
        current_scores, current_selected = _weighted_score_snapshot(
            repetition_results,
            metadata=design,
            tau=stability_tau,
            paper_exact_baseline=paper_exact_baseline,
        )

        if previous_scores is None:
            score_delta = np.nan
            set_jaccard = np.nan
            score_stable = False
            set_stable = not require_set_stability
            stable_batches = 0
        else:
            score_delta = float((current_scores - previous_scores).abs().max())
            score_stable = score_delta < float(stability_epsilon)
            set_jaccard = float(_jaccard_similarity(current_selected, previous_selected or frozenset()))
            set_stable = (set_jaccard >= float(min_jaccard)) if require_set_stability else True
            stable_batches = stable_batches + 1 if score_stable and set_stable else 0

        eligible_to_stop = total_reps >= int(min_nrep)
        stop_now = eligible_to_stop and stable_batches >= int(patience)
        history_rows.append(
            {
                "batch_index": batch_index,
                "batch_size": int(current_batch_size),
                "nrep_used": total_reps,
                "score_delta": score_delta,
                "set_jaccard": set_jaccard,
                "score_stable": bool(score_stable),
                "set_stable": bool(set_stable),
                "stable_batches": int(stable_batches),
                "eligible_to_stop": bool(eligible_to_stop),
                "stop_now": bool(stop_now),
            }
        )
        previous_scores = current_scores
        previous_selected = current_selected
        if stop_now:
            stop_reason = "stability_reached"
            break

    repetition_results = (
        pd.concat(repetition_batches, ignore_index=True)
        if repetition_batches
        else pd.DataFrame.from_records([])
    )
    stability_history = pd.DataFrame.from_records(history_rows)
    last_row = history_rows[-1] if history_rows else {}
    diagnostics = {
        "stopping_rule": "adaptive_stability",
        "stop_reason": stop_reason,
        "n_batches_used": int(batch_index),
        "nrep_used": int(len(repetition_results)),
        "stopped_early": bool(len(repetition_results) < int(max_nrep)),
        "stability_tau": float(stability_tau),
        "stability_epsilon": float(stability_epsilon),
        "patience": int(patience),
        "require_set_stability": bool(require_set_stability),
        "min_jaccard": float(min_jaccard),
        "last_score_delta": float(last_row.get("score_delta", np.nan)),
        "last_set_jaccard": float(last_row.get("set_jaccard", np.nan)),
        "stable_batches_achieved": int(last_row.get("stable_batches", 0)),
    }
    return repetition_results, stability_history, diagnostics


def _weighted_aggregate(
    repetition_results: pd.DataFrame,
    metadata: core.MixedLevelDesign,
    *,
    tau: float,
    paper_exact_baseline: bool,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    feature_table = _weighted_feature_table(
        repetition_results,
        metadata,
        paper_exact_baseline=paper_exact_baseline,
    )
    feature_table = feature_table.assign(
        kept=feature_table["weighted_score"].to_numpy(dtype=float) >= float(tau)
    ).sort_values(
        ["kept", "score", "count", "feature_id"],
        ascending=[False, False, False, True],
    )

    if paper_exact_baseline:
        aggregated_columns = feature_table.loc[feature_table["kept"], "feature_id"].tolist()
        aggregated_effects = sorted(
            metadata.column_table[
                metadata.column_table["column_id"].isin(aggregated_columns)
            ]["effect_id"].unique().tolist()
        )
    else:
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

    if variant not in _ALLOWED_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    feature_table, aggregated_effects, aggregated_columns, aggregation_diagnostics = _weighted_aggregate(
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


def _iter_adaptive_comparison_tasks(
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
    adaptive_max_nrep: int,
    adaptive_batch_size: int,
    adaptive_min_nrep: int,
    adaptive_stability_epsilon: float,
    adaptive_patience: int,
    adaptive_require_set_stability: bool,
    adaptive_min_jaccard: float,
    adaptive_set_tau: float,
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
                    "adaptive_max_nrep": int(adaptive_max_nrep),
                    "adaptive_batch_size": int(adaptive_batch_size),
                    "adaptive_min_nrep": int(adaptive_min_nrep),
                    "adaptive_stability_epsilon": float(adaptive_stability_epsilon),
                    "adaptive_patience": int(adaptive_patience),
                    "adaptive_require_set_stability": bool(adaptive_require_set_stability),
                    "adaptive_min_jaccard": float(adaptive_min_jaccard),
                    "adaptive_set_tau": float(adaptive_set_tau),
                }
                task_id += 1


def _run_adaptive_comparison_replication(
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
    shared_rep_seed_count = max(int(task["benchmark_nrep"]), int(task["adaptive_max_nrep"]))
    shared_rep_seeds = [
        int(rng.integers(0, 2**31 - 1))
        for _ in range(shared_rep_seed_count)
    ]

    path_results_by_variant: dict[str, dict[str, Any]] = {}
    if "weighted_only" in task["variants"]:
        path_t0 = time.perf_counter()
        repetition_results = _build_repetition_path(
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
        path_results_by_variant["weighted_only"] = {
            "repetition_results": repetition_results,
            "path_elapsed": float(path_elapsed),
            "path_diagnostics": {
                "stopping_rule": "fixed_nrep",
                "stop_reason": "fixed_nrep",
                "nrep_used": int(len(repetition_results)),
                "stopped_early": False,
                "n_batches_used": 1,
                "last_score_delta": np.nan,
                "last_set_jaccard": np.nan,
                "stable_batches_achieved": 0,
            },
        }

    if "adaptive_nrep_only" in task["variants"]:
        path_t0 = time.perf_counter()
        repetition_results, stability_history, path_diagnostics = _build_adaptive_repetition_path(
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
        path_results_by_variant["adaptive_nrep_only"] = {
            "repetition_results": repetition_results,
            "path_elapsed": float(path_elapsed),
            "path_diagnostics": path_diagnostics,
            "stability_history": stability_history,
        }

    replication_rows: list[dict[str, Any]] = []
    config_order = 0
    for ntop in task["ntop_values"]:
        for pkeep in task["pkeep_values"]:
            weighted_tau = _resolve_weighted_tau(
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
                        "nrep": (
                            int(task["benchmark_nrep"])
                            if str(variant) == "weighted_only"
                            else int(task["adaptive_max_nrep"])
                        ),
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
                        "stopping_rule": str(diagnostics["stopping_rule"]),
                        "stop_reason": str(diagnostics["stop_reason"]),
                        "stopped_early": bool(diagnostics["stopped_early"]),
                        "n_batches_used": int(diagnostics["n_batches_used"]),
                        "last_score_delta": float(diagnostics["last_score_delta"]),
                        "last_set_jaccard": float(diagnostics["last_set_jaccard"]),
                        "stable_batches_achieved": int(diagnostics["stable_batches_achieved"]),
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


def _append_adaptive_result(
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
        _append_adaptive_result(
            _run_adaptive_comparison_replication(task),
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
            pending[executor.submit(_run_adaptive_comparison_replication, task)] = None

        while pending:
            done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future, None)
                _append_adaptive_result(
                    future.result(),
                    replication_items,
                    truth_items,
                )
                progress.update(1)
                try:
                    next_task = next(task_iter)
                except StopIteration:
                    continue
                pending[executor.submit(_run_adaptive_comparison_replication, next_task)] = None

    return replication_items, truth_items


def _summarize_results(replication_results: pd.DataFrame) -> pd.DataFrame:
    return (
        replication_results.groupby(
            [
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
            ],
            dropna=False,
        )
        .agg(
            mean_nrep_used=("nrep_used", "mean"),
            sd_nrep_used=("nrep_used", "std"),
            mean_n_batches_used=("n_batches_used", "mean"),
            stopped_early_rate=("stopped_early", "mean"),
            mean_power=("power", "mean"),
            sd_power=("power", "std"),
            mean_error=("error", "mean"),
            sd_error=("error", "std"),
            mean_score=("score", "mean"),
            sd_score=("score", "std"),
            mean_model_size=("final_model_size", "mean"),
            mean_BIC=("final_BIC", "mean"),
            mean_timing_seconds=("timing_seconds", "mean"),
            mean_last_score_delta=("last_score_delta", "mean"),
            mean_last_set_jaccard=("last_set_jaccard", "mean"),
            n_datasets=("dataset_rep", "count"),
        )
        .reset_index()
        .fillna(
            {
                "sd_nrep_used": 0.0,
                "sd_power": 0.0,
                "sd_error": 0.0,
                "sd_score": 0.0,
                "mean_last_score_delta": np.nan,
                "mean_last_set_jaccard": np.nan,
            }
        )
        .sort_values(["design_id", "scenario_id", "variant"], kind="stable")
        .reset_index(drop=True)
    )


def _overall_summary(summary_results: pd.DataFrame) -> pd.DataFrame:
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
            n_conditions=("scenario_id", "count"),
        )
        .reset_index()
        .fillna({"sd_score": 0.0})
        .sort_values("variant", kind="stable")
        .reset_index(drop=True)
    )


def adaptive_ntop_pkeep_aggregate(summary_results: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_results.groupby(["variant", "ntop", "pkeep"], dropna=False)
        .agg(
            mean_nrep_used=("mean_nrep_used", "mean"),
            mean_score=("mean_score", "mean"),
            sd_score=("mean_score", "std"),
            mean_power=("mean_power", "mean"),
            sd_power=("mean_power", "std"),
            n_conditions=("design_id", "count"),
        )
        .reset_index()
        .fillna({"sd_score": 0.0, "sd_power": 0.0})
        .sort_values(["variant", "pkeep", "ntop"], kind="stable")
        .reset_index(drop=True)
    )


def _adaptive_plot_data(summary_results: pd.DataFrame) -> pd.DataFrame:
    plot_data = summary_results.copy()
    plot_data["design_label"] = plot_data["design_id"].map(core._paper_design_label)
    plot_data["scenario_label"] = plot_data["scenario_id"].map(core._paper_scenario_label)
    plot_data["scenario_id_int"] = pd.to_numeric(plot_data["scenario_id"], errors="coerce")
    plot_data["variant_label"] = plot_data["variant"].map(
        {
            "weighted_only": "Weighted Only",
            "adaptive_nrep_only": "Adaptive nrep Only",
        }
    ).fillna(plot_data["variant"].astype(str))
    plot_data["condition_label"] = (
        plot_data["design_label"].astype(str) + " | " + plot_data["scenario_label"].astype(str)
    )
    return plot_data


def adaptive_design_summary(summary_results: pd.DataFrame) -> pd.DataFrame:
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


def adaptive_condition_ranking(summary_results: pd.DataFrame) -> pd.DataFrame:
    ranked = _adaptive_plot_data(summary_results)
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


def _variant_comparison(summary_results: pd.DataFrame) -> pd.DataFrame:
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


def plot_adaptive_ntop_pkeep_scatter_plotnine(data: pd.DataFrame, metric: str = "score"):
    return common.plot_ntop_pkeep_scatter_plotnine(
        data,
        metric=metric,
        title=f"AWS-GDS-ARM Mean vs SD for (ntop, pkeep): {metric}",
        subtitle="Weighted-only benchmark versus adaptive nrep stopping.",
    )


def plot_adaptive_metric_by_scenario_plotnine(data: pd.DataFrame, metric: str = "score"):
    return common.plot_metric_by_scenario_plotnine(
        data,
        metric=metric,
        title=f"AWS-GDS-ARM Mean {metric.title()} by Scenario",
        subtitle="Weighted-only benchmark versus adaptive nrep stopping.",
    )


def plot_adaptive_condition_scatter_plotnine(data: pd.DataFrame, metric: str = "score"):
    return common.plot_condition_scatter_plotnine(
        data,
        metric=metric,
        title=f"AWS-GDS-ARM Mean vs SD: {metric}",
        subtitle="Condition-level benchmark comparison for adaptive nrep stopping.",
    )


def plot_adaptive_runtime_by_scenario_plotnine(data: pd.DataFrame):
    return common.plot_runtime_by_scenario_plotnine(
        data,
        title="AWS-GDS-ARM Mean Runtime by Scenario",
        subtitle="Weighted-only benchmark versus adaptive nrep stopping.",
    )


def run_adaptive_benchmark_study(
    settings: AdaptiveComparisonSettings | None = None,
    /,
    **overrides: Any,
) -> dict[str, Any]:
    if settings is None:
        settings = AdaptiveComparisonSettings(**overrides)
    elif overrides:
        settings = replace(settings, **overrides)

    _validate_settings(settings)
    benchmark_tau = _resolve_weighted_tau(float(settings.benchmark_pkeep), settings.weighted_tau)
    adaptive_stability_tau = _resolve_weighted_tau(
        float(settings.benchmark_pkeep),
        settings.weighted_tau if settings.adaptive_set_tau is None else settings.adaptive_set_tau,
    )
    output_path = Path(settings.output_dir)
    tables_dir = output_path / _ADAPTIVE_TABLES_DIRNAME
    plots_dir = output_path / _ADAPTIVE_PLOTS_DIRNAME
    manifest_path = output_path / _ADAPTIVE_SETTINGS_FILENAME
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
    diagnostics.to_csv(tables_dir / "adaptive_design_diagnostics.csv", index=False)
    scenario_map = context.scenario_map

    progress_total = sum(
        len(scenario_map[design.design_id]) * int(settings.reps_per_condition)
        for design in designs
    )
    progress = core._make_progress(
        total=progress_total,
        desc="AWS adaptive benchmark",
        disable=not settings.show_progress,
        leave=True,
    )

    design_registry = context.design_registry
    core._set_simulation_design_registry(design_registry)
    task_seed = int(rng.integers(0, 2**31 - 1))
    worker_count = min(core._resolve_n_jobs(settings.n_jobs), max(progress_total, 1))

    def _task_iter():
        return _iter_adaptive_comparison_tasks(
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
            adaptive_max_nrep=int(settings.adaptive_max_nrep),
            adaptive_batch_size=int(settings.adaptive_batch_size),
            adaptive_min_nrep=int(settings.adaptive_min_nrep),
            adaptive_stability_epsilon=float(settings.adaptive_stability_epsilon),
            adaptive_patience=int(settings.adaptive_patience),
            adaptive_require_set_stability=bool(settings.adaptive_require_set_stability),
            adaptive_min_jaccard=float(settings.adaptive_min_jaccard),
            adaptive_set_tau=float(adaptive_stability_tau),
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
                    desc="AWS adaptive benchmark",
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
    ntop_pkeep_aggregate = adaptive_ntop_pkeep_aggregate(summary_results)
    overall_summary = _overall_summary(benchmark_summary_results)
    design_summary = adaptive_design_summary(benchmark_summary_results)
    condition_ranking = adaptive_condition_ranking(benchmark_summary_results)
    comparison_table = _variant_comparison(benchmark_summary_results)

    summary_results.to_csv(tables_dir / "adaptive_ntop_pkeep_summary_results.csv", index=False)
    benchmark_summary_results.to_csv(tables_dir / "adaptive_variant_summary_results.csv", index=False)
    overall_summary.to_csv(tables_dir / "adaptive_variant_overall_summary.csv", index=False)
    design_summary.to_csv(tables_dir / "adaptive_design_summary.csv", index=False)
    condition_ranking.to_csv(tables_dir / "adaptive_condition_ranking.csv", index=False)
    comparison_table.to_csv(tables_dir / "adaptive_variant_condition_comparison.csv", index=False)
    ntop_pkeep_aggregate.to_csv(tables_dir / "adaptive_ntop_pkeep_aggregate.csv", index=False)
    truth_records.to_csv(tables_dir / "adaptive_truth_records.csv", index=False)
    if settings.write_replication_tables:
        replication_results.to_csv(tables_dir / "adaptive_variant_replication_results.csv", index=False)
    if settings.write_plots:
        core._save_plotnine_plot(
            plot_adaptive_ntop_pkeep_scatter_plotnine(ntop_pkeep_aggregate, metric="score"),
            plots_dir / "adaptive_ntop_pkeep_score_plotnine.png",
        )
        core._save_plotnine_plot(
            plot_adaptive_ntop_pkeep_scatter_plotnine(ntop_pkeep_aggregate, metric="power"),
            plots_dir / "adaptive_ntop_pkeep_power_plotnine.png",
        )
        core._save_plotnine_plot(
            plot_adaptive_metric_by_scenario_plotnine(benchmark_summary_results, metric="score"),
            plots_dir / "adaptive_score_by_scenario_plotnine.png",
        )
        core._save_plotnine_plot(
            plot_adaptive_metric_by_scenario_plotnine(benchmark_summary_results, metric="power"),
            plots_dir / "adaptive_power_by_scenario_plotnine.png",
        )
        core._save_plotnine_plot(
            plot_adaptive_condition_scatter_plotnine(benchmark_summary_results, metric="score"),
            plots_dir / "adaptive_score_scatter_plotnine.png",
        )
        core._save_plotnine_plot(
            plot_adaptive_runtime_by_scenario_plotnine(benchmark_summary_results),
            plots_dir / "adaptive_runtime_by_scenario_plotnine.png",
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
        "Adaptive stopping settings: "
        f"max_nrep={settings.adaptive_max_nrep}, min_nrep={settings.adaptive_min_nrep}, "
        f"batch_size={settings.adaptive_batch_size}, epsilon={settings.adaptive_stability_epsilon}, "
        f"patience={settings.adaptive_patience}, set_tau={adaptive_stability_tau}, "
        f"require_set_stability={settings.adaptive_require_set_stability}, "
        f"min_jaccard={settings.adaptive_min_jaccard}"
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
    "AdaptiveComparisonSettings",
    "adaptive_ntop_pkeep_aggregate",
    "adaptive_design_summary",
    "adaptive_condition_ranking",
    "plot_adaptive_ntop_pkeep_scatter_plotnine",
    "plot_adaptive_metric_by_scenario_plotnine",
    "plot_adaptive_condition_scatter_plotnine",
    "plot_adaptive_runtime_by_scenario_plotnine",
    "run_adaptive_benchmark_study",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the adaptive-nrep AWS-GDS-ARM benchmark on the same design/scenario settings used by the baseline."
    )
    parser.add_argument(
        "--output-dir",
        default="study_outputs_aws_adaptive",
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
        help="Fixed-nrep weighted benchmark for the weighted_only variant.",
    )
    parser.add_argument(
        "--adaptive-max-nrep",
        type=int,
        default=_CORE_DEFAULTS.ntop_pkeep_nrep,
        help="Maximum repetition budget for the adaptive_nrep_only variant.",
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
        "--adaptive-require-set-stability",
        action="store_true",
        help="Require Jaccard stability of the selected weighted set as well as score stability.",
    )
    parser.add_argument(
        "--adaptive-min-jaccard",
        type=float,
        default=0.95,
        help="Minimum Jaccard similarity when set stability is enabled.",
    )
    parser.add_argument(
        "--adaptive-set-tau",
        type=float,
        default=None,
        help="Threshold used for the adaptive set-stability check. Defaults to the benchmark weighted tau.",
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
        run_adaptive_benchmark_study(
            AdaptiveComparisonSettings(
                output_dir=args.output_dir,
                random_state=args.seed,
                reps_per_condition=args.reps_per_condition,
                benchmark_nrep=args.benchmark_nrep,
                adaptive_max_nrep=args.adaptive_max_nrep,
                adaptive_batch_size=args.adaptive_batch_size,
                adaptive_min_nrep=args.adaptive_min_nrep,
                adaptive_stability_epsilon=args.adaptive_epsilon,
                adaptive_patience=args.adaptive_patience,
                adaptive_require_set_stability=args.adaptive_require_set_stability,
                adaptive_min_jaccard=args.adaptive_min_jaccard,
                adaptive_set_tau=args.adaptive_set_tau,
                weighted_tau=args.weighted_tau,
                n_jobs=args.n_jobs,
                variants=tuple(args.variants or _ALLOWED_VARIANTS),
            )
        )
