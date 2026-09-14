"""Mixed-level GDS and GDS-ARM implementation for supersaturated designs.

This module implements:

1. Orthogonal polynomial coding for mixed-level factors.
2. Explicit metadata for factors, grouped effects, and individual columns.
3. The Gauss-Dantzig Selector (GDS) with k-means thresholding and OLS refit.
4. The GDS-ARM algorithm with aggregation over random interaction submodels.
5. Heuristic generation of benchmark mixed-level supersaturated designs.
6. A simulation framework for factor-screening studies.
7. Plotnine plotting helpers for the tuning-study graphics.

The implementation follows the mixed-level extension described by:

Zhang, F., Singh, R., & Stufken, J. (2026).
An Extension of the GDS-ARM Algorithm for Factor Screening in Mixed-Level
Supersaturated Designs.
Journal of Statistical Theory and Practice, 20, 41.
https://doi.org/10.1007/s42519-026-00542-x
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, replace
import gzip
import hashlib
import json
from itertools import combinations, product
from math import ceil
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re
import sys
import time
import warnings
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")

from sklearn.cluster import KMeans

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None


_CONTRAST_LABELS = [
    "linear",
    "quadratic",
    "cubic",
    "quartic",
    "quintic",
    "sextic",
    "septic",
    "octic",
    "nonic",
]

_PALETTE = {
    "ink": "#1F2937",
    "slate": "#40536A",
    "sand": "#F3E9DC",
    "cream": "#FBF7F2",
    "copper": "#C67A3D",
    "gold": "#E3A008",
    "teal": "#1F8A70",
    "sea": "#147A8A",
    "brick": "#B44C43",
    "moss": "#6C8A3B",
    "mist": "#D9E3EA",
}

_PAPER_SCENARIO_CATALOG: dict[int, dict[str, int]] = {
    1: {"mmain": 2, "mint": 0, "mimp": 2},
    2: {"mmain": 2, "mint": 1, "mimp": 2},
    3: {"mmain": 1, "mint": 1, "mimp": 2},
    4: {"mmain": 3, "mint": 0, "mimp": 3},
    5: {"mmain": 3, "mint": 2, "mimp": 3},
    6: {"mmain": 2, "mint": 2, "mimp": 3},
    7: {"mmain": 4, "mint": 0, "mimp": 4},
    8: {"mmain": 4, "mint": 2, "mimp": 4},
    9: {"mmain": 3, "mint": 1, "mimp": 4},
    10: {"mmain": 3, "mint": 2, "mimp": 4},
    11: {"mmain": 2, "mint": 2, "mimp": 4},
    12: {"mmain": 5, "mint": 0, "mimp": 5},
    13: {"mmain": 5, "mint": 2, "mimp": 5},
    14: {"mmain": 4, "mint": 1, "mimp": 5},
    15: {"mmain": 8, "mint": 0, "mimp": 8},
}

_PAPER_DESIGN_SCENARIO_MAP: dict[str, list[int]] = {
    "design1_n21m8": [1, 2, 3, 4, 5, 6],
    "design2_n27m20": [7, 8, 9, 10, 11, 15],
    "design5_n8m12": [1, 2, 3, 12, 13, 14],
}

_DEFAULT_SIMULATION_DESIGN_IDS = tuple(_PAPER_DESIGN_SCENARIO_MAP)
_DEFAULT_NINT_DESIGN_IDS = ("design1_n21m8", "design2_n27m20")
_DEFAULT_NINT_MULTIPLIERS_BY_DESIGN: dict[str, tuple[float, ...]] = {
    "design1_n21m8": (0.5, 1.0, 2.0, 3.0),
    "design2_n27m20": (0.5, 1.0, 2.0, 4.0, 6.0),
}
_DEFAULT_NTOP_VALUES = (20, 30, 40, 60, 80, 100)
_DEFAULT_PKEEP_VALUES = (0.1, 0.25, 0.4, 0.55, 0.7)
_DEFAULT_NREP_VALUES = (100, 500, 1000, 4000)
_GENERATED_DESIGN_SPECS: dict[str, dict[str, Any]] = {
    "design1_n21m8": {"n_runs": 21, "q_values": (2, 2, 2, 2, 3, 3, 3, 3)},
    "design2_n27m20": {"n_runs": 27, "q_values": (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3)},
    "design5_n8m12": {"n_runs": 8, "q_values": (2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4)},
}


@dataclass(slots=True)
class MixedLevelDesign:
    """Container for a mixed-level model matrix and explicit metadata."""

    design_id: str
    factor_data: pd.DataFrame | None
    X_main: pd.DataFrame
    X_int: pd.DataFrame
    X_full: pd.DataFrame
    factor_table: pd.DataFrame
    effect_table: pd.DataFrame
    column_table: pd.DataFrame
    effect_to_columns: dict[str, list[str]]
    factor_to_main_effect: dict[str, str]
    interaction_to_pair: dict[str, tuple[str, str]]
    column_to_metadata: dict[str, dict[str, Any]]
    standardization: dict[str, Any]

    @property
    def n_runs(self) -> int:
        return int(self.X_full.shape[0])

    @property
    def ncol_main(self) -> int:
        return int(self.X_main.shape[1])

    @property
    def r_ratio(self) -> float:
        return float(self.ncol_main / self.n_runs)

    @property
    def factor_ids(self) -> list[str]:
        return self.factor_table["factor_id"].tolist()

    @property
    def effect_ids(self) -> list[str]:
        return self.effect_table["effect_id"].tolist()

    @property
    def column_ids(self) -> list[str]:
        return self.column_table["column_id"].tolist()

    def columns_for_effects(self, effect_ids: Iterable[str]) -> list[str]:
        columns: list[str] = []
        for effect_id in effect_ids:
            columns.extend(self.effect_to_columns.get(effect_id, []))
        return columns

    def factors_for_effects(self, effect_ids: Iterable[str]) -> list[str]:
        selected: set[str] = set()
        effect_set = set(effect_ids)
        matched = self.effect_table[self.effect_table["effect_id"].isin(effect_set)]
        for factor_ids in matched["factor_ids"]:
            selected.update(factor_ids)
        return sorted(selected)

    def subset_by_columns(self, column_ids: Sequence[str], design_id: str | None = None) -> "MixedLevelDesign":
        column_ids = list(dict.fromkeys(column_ids))
        missing = sorted(set(column_ids) - set(self.column_ids))
        if missing:
            raise KeyError(f"Unknown columns requested: {missing}")

        subset_column_table = self.column_table[self.column_table["column_id"].isin(column_ids)].copy()
        effect_ids = subset_column_table["effect_id"].unique().tolist()
        subset_effect_table = self.effect_table[self.effect_table["effect_id"].isin(effect_ids)].copy()
        factor_ids = sorted({fid for ids in subset_effect_table["factor_ids"] for fid in ids})
        subset_factor_table = self.factor_table[self.factor_table["factor_id"].isin(factor_ids)].copy()

        X_main_cols = [cid for cid in column_ids if cid in self.X_main.columns]
        X_int_cols = [cid for cid in column_ids if cid in self.X_int.columns]
        X_main = self.X_main.loc[:, X_main_cols].copy()
        X_int = self.X_int.loc[:, X_int_cols].copy()
        X_full = pd.concat([X_main, X_int], axis=1)

        effect_to_columns = _build_effect_to_columns(subset_column_table)
        factor_to_main_effect = _build_factor_to_main_effect(subset_factor_table)
        interaction_to_pair = _build_interaction_to_pair(subset_effect_table)
        column_to_metadata = _build_column_to_metadata(subset_column_table)

        return MixedLevelDesign(
            design_id=design_id or f"{self.design_id}_subset",
            factor_data=self.factor_data,
            X_main=X_main,
            X_int=X_int,
            X_full=X_full,
            factor_table=subset_factor_table.reset_index(drop=True),
            effect_table=subset_effect_table.reset_index(drop=True),
            column_table=subset_column_table.reset_index(drop=True),
            effect_to_columns=effect_to_columns,
            factor_to_main_effect=factor_to_main_effect,
            interaction_to_pair=interaction_to_pair,
            column_to_metadata=column_to_metadata,
            standardization=self.standardization.copy(),
        )


@dataclass(slots=True)
class OLSRefitResult:
    selected_columns: list[str]
    coefficients: pd.Series
    fitted: np.ndarray
    residuals: np.ndarray
    rss: float
    bic: float
    aic: float
    r_squared: float
    rank: int
    df_resid: int


@dataclass(slots=True)
class GDSResult:
    design_id: str | None
    delta_grid: np.ndarray
    path_results: pd.DataFrame
    best_delta: float
    best_beta: pd.Series
    best_selected_columns: list[str]
    best_selected_effects: list[str]
    best_selected_factors: list[str]
    best_ols_result: OLSRefitResult
    best_bic: float
    solver_status: str


@dataclass(slots=True)
class GDSStepIResult:
    design_id: str | None
    sampled_interaction_columns: list[str]
    gds_result: GDSResult


@dataclass(slots=True)
class StepwiseResult:
    selected_groups: list[str]
    selected_columns: list[str]
    coefficients: pd.Series
    fit: OLSRefitResult
    history: pd.DataFrame


@dataclass(slots=True)
class GDSARMResult:
    design_id: str | None
    repetition_results: pd.DataFrame
    top_results: pd.DataFrame
    effect_frequency_table: pd.DataFrame
    aggregated_effects: list[str]
    final_stepwise_result: StepwiseResult
    final_selected_effects: list[str]
    final_selected_columns: list[str]
    final_selected_factors: list[str]
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class SimulationStudyResult:
    replication_results: pd.DataFrame
    summary_results: pd.DataFrame
    design_diagnostics: pd.DataFrame
    truth_records: pd.DataFrame
    tuning_grid: pd.DataFrame
    arm_artifacts: dict[str, GDSARMResult] | None = None


@dataclass(slots=True)
class StudyRunSettings:
    output_dir: str | os.PathLike[str] = "study_outputs"
    run_name: str = "generated_baseline"
    generated_design_coding: str = "paper_exact"
    generated_design_candidate_pool: int = 64
    generated_design_restarts: int = 8
    random_state: int = 2026
    reps_per_condition: int = 200
    studies: tuple[str, ...] = ("nint", "ntop_pkeep", "nrep")
    simulation_design_ids: tuple[str, ...] = _DEFAULT_SIMULATION_DESIGN_IDS
    nint_design_ids: tuple[str, ...] = _DEFAULT_NINT_DESIGN_IDS
    scenario_ids_by_design: dict[str, tuple[int | str, ...]] | None = None
    nint_multipliers_by_design: dict[str, tuple[float, ...]] | None = None
    ntop_values: tuple[int, ...] = _DEFAULT_NTOP_VALUES
    pkeep_values: tuple[float, ...] = _DEFAULT_PKEEP_VALUES
    nrep_values: tuple[int, ...] = _DEFAULT_NREP_VALUES
    ntop_pkeep_nrep: int = 1000
    ntop_pkeep_nint: int | str = "2n"
    nrep_study_ntop: int = 20
    nrep_study_pkeep: float = 0.25
    nrep_study_nint: int | str = "2n"
    n_jobs: int = -1
    task_shards: int | None = None
    task_shard_index: int | None = None
    reuse_saved_results: bool = True
    write_cache_file: bool = True
    force_rerun: bool = False
    write_replication_tables: bool = True
    write_plots: bool = True
    write_settings_file: bool = True
    show_progress: bool = True


_SIMULATION_WORKER_DESIGNS: dict[str, MixedLevelDesign] = {}

_SIMULATION_REPLICATION_COLUMNS = [
    "design_id",
    "scenario_id",
    "dataset_rep",
    "tuning_id",
    "analysis_stage",
    "nrep",
    "nint",
    "ntop",
    "pkeep",
    "group_stepwise",
    "paper_exact_stepwise",
    "paper_exact_baseline",
    "final_stepwise",
    "TP",
    "FP",
    "power",
    "error",
    "score",
    "sampled_interaction_columns",
    "selected_factors",
    "selected_effects",
    "selected_columns",
    "final_model_size",
    "BIC",
    "timing_seconds",
    "artifact_key",
]

_SIMULATION_TRUTH_COLUMNS = [
    "design_id",
    "scenario_id",
    "dataset_rep",
    "tuning_id",
    "analysis_stage",
    "true_important_factors",
    "true_active_effects",
    "true_active_columns",
]

_SIMULATION_SUMMARY_COLUMNS = [
    "design_id",
    "scenario_id",
    "tuning_id",
    "analysis_stage",
    "nrep",
    "nint",
    "ntop",
    "pkeep",
    "group_stepwise",
    "paper_exact_stepwise",
    "paper_exact_baseline",
    "final_stepwise",
    "mean_power",
    "sd_power",
    "mean_error",
    "sd_error",
    "mean_score",
    "sd_score",
    "mean_model_size",
    "mean_BIC",
    "mean_timing_seconds",
]


def _ensure_rng(random_state: int | np.random.Generator | None = None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


class _SimpleProgress:
    """Lightweight fallback when tqdm is unavailable."""

    def __init__(self, total: int | None, desc: str = "", disable: bool = False) -> None:
        self.total = total
        self.desc = desc
        self.disable = disable
        self.count = 0
        self.started_at = time.perf_counter()
        self._last_fraction = -1
        if not self.disable:
            if self.total:
                print(f"{self.desc}: 0/{self.total}")
            else:
                print(f"{self.desc}: started")

    def update(self, n: int = 1) -> None:
        if self.disable:
            return
        self.count += n
        if self.total:
            fraction = int((10 * self.count) / self.total)
            if fraction > self._last_fraction or self.count == self.total:
                elapsed = time.perf_counter() - self.started_at
                print(f"{self.desc}: {self.count}/{self.total} ({elapsed:0.1f}s)")
                self._last_fraction = fraction
        else:
            elapsed = time.perf_counter() - self.started_at
            print(f"{self.desc}: {self.count} ({elapsed:0.1f}s)")

    def close(self) -> None:
        if self.disable:
            return
        elapsed = time.perf_counter() - self.started_at
        if self.total:
            print(f"{self.desc}: done in {elapsed:0.1f}s")
        else:
            print(f"{self.desc}: finished in {elapsed:0.1f}s")


def _make_progress(total: int | None, desc: str, *, disable: bool = False, leave: bool = True):
    if _tqdm is not None:
        return _tqdm(total=total, desc=desc, disable=disable, leave=leave)
    return _SimpleProgress(total=total, desc=desc, disable=disable)


def _as_dataframe(data: pd.DataFrame | Mapping[str, Sequence[Any]] | np.ndarray) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data).copy()


def _normalize_factor_order(q_values: Mapping[str, int], factor_columns: Sequence[str]) -> dict[str, int]:
    q_normalized: dict[str, int] = {}
    for factor_id in factor_columns:
        if factor_id not in q_values:
            raise KeyError(f"Missing q_i for factor '{factor_id}'.")
        q_normalized[str(factor_id)] = int(q_values[factor_id])
    return q_normalized


def _contrast_labels(n_cols: int) -> list[str]:
    labels = list(_CONTRAST_LABELS[:n_cols])
    if n_cols > len(labels):
        for degree in range(len(labels) + 1, n_cols + 1):
            labels.append(f"degree_{degree}")
    return labels


def orthogonal_polynomial_contrasts(q_levels: int) -> pd.DataFrame:
    """Return q x (q - 1) orthonormal polynomial contrasts for q levels."""

    if q_levels < 2:
        raise ValueError("A factor must have at least 2 levels.")

    scores = np.arange(1, q_levels + 1, dtype=float)
    centered = scores - scores.mean()
    # Include the constant column in the orthogonalization so the returned
    # columns are true contrasts, i.e. orthogonal to the intercept as well as
    # orthogonal to each other.
    vandermonde = np.column_stack([np.ones(q_levels)] + [centered ** degree for degree in range(1, q_levels)])
    q_matrix, _ = np.linalg.qr(vandermonde)
    contrast_matrix = q_matrix[:, 1:]

    for degree_idx in range(contrast_matrix.shape[1]):
        target = centered ** (degree_idx + 1)
        if float(np.dot(contrast_matrix[:, degree_idx], target)) < 0:
            contrast_matrix[:, degree_idx] *= -1.0

    return pd.DataFrame(contrast_matrix, columns=_contrast_labels(q_levels - 1))


def paper_exact_polynomial_contrasts(q_levels: int) -> pd.DataFrame:
    """Return the supplement's exact polynomial coding for low-order factors.

    The paper's R code uses named low-order polynomial columns rather than an
    arbitrary orthonormal basis. Because the final stepwise search is
    column-wise, reproducing those specific columns matters for baseline
    alignment.
    """

    if q_levels == 2:
        data = np.array([[-1.0], [1.0]])
        columns = [""]
    elif q_levels == 3:
        data = np.array(
            [
                [-1.0, 1.0],
                [0.0, -2.0],
                [1.0, 1.0],
            ]
        )
        columns = ["l", "q"]
    elif q_levels == 4:
        data = np.array(
            [
                [-3.0, 1.0, -1.0],
                [-1.0, -1.0, 3.0],
                [1.0, -1.0, -3.0],
                [3.0, 1.0, 1.0],
            ]
        )
        columns = ["l", "q", "c"]
    else:
        # Fall back to the exact generic orthogonal basis outside the q <= 4
        # setting used in the paper's main tuning study.
        generic = orthogonal_polynomial_contrasts(q_levels)
        generic.columns = [f"d{idx}" for idx in range(1, q_levels)]
        return generic

    return pd.DataFrame(data, columns=columns)


def _resolve_factor_levels(
    series: pd.Series,
    q_levels: int,
    explicit_levels: Sequence[Any] | None = None,
    *,
    allow_inference: bool = False,
) -> list[Any]:
    if explicit_levels is not None:
        levels = list(explicit_levels)
    elif q_levels > 2 and not allow_inference:
        raise ValueError(
            f"Factor '{series.name}' has q_i={q_levels}. Explicit ordered factor_levels are required for mixed-level factors."
        )
    else:
        levels = sorted(pd.unique(series).tolist())
    if len(levels) != q_levels:
        raise ValueError(
            f"Factor '{series.name}' expected {q_levels} levels but resolved {len(levels)} levels: {levels}"
        )
    missing = sorted(set(pd.unique(series)) - set(levels))
    if missing:
        raise ValueError(f"Factor '{series.name}' has values not covered by the level list: {missing}")
    return levels


def _build_factor_to_main_effect(factor_table: pd.DataFrame) -> dict[str, str]:
    return dict(zip(factor_table["factor_id"], factor_table["main_effect_id"], strict=False))


def _build_interaction_to_pair(effect_table: pd.DataFrame) -> dict[str, tuple[str, str]]:
    subset = effect_table[effect_table["effect_type"] == "interaction"]
    mapping: dict[str, tuple[str, str]] = {}
    for row in subset.itertuples(index=False):
        mapping[row.effect_id] = tuple(row.factor_ids)
    return mapping


def _build_effect_to_columns(column_table: pd.DataFrame) -> dict[str, list[str]]:
    grouped = column_table.groupby("effect_id", sort=False)["column_id"]
    return {effect_id: columns.tolist() for effect_id, columns in grouped}


def _build_column_to_metadata(column_table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        row.column_id: {
            "column_id": row.column_id,
            "effect_type": row.effect_type,
            "factor_ids": row.factor_ids,
            "effect_id": row.effect_id,
            "contrast_name": row.contrast_name,
            "component_contrast_names": row.component_contrast_names,
        }
        for row in column_table.itertuples(index=False)
    }


def _prune_metadata_tables(
    factor_table: pd.DataFrame,
    effect_table: pd.DataFrame,
    column_table: pd.DataFrame,
    dropped_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not dropped_columns:
        return factor_table, effect_table, column_table

    dropped = set(dropped_columns)
    column_table = column_table[~column_table["column_id"].isin(dropped)].reset_index(drop=True)

    effect_rows: list[dict[str, Any]] = []
    for row in effect_table.itertuples(index=False):
        kept_columns = [column_id for column_id in row.column_ids if column_id not in dropped]
        if not kept_columns:
            continue
        effect_rows.append(
            {
                "effect_id": row.effect_id,
                "effect_type": row.effect_type,
                "factor_ids": row.factor_ids,
                "column_ids": tuple(kept_columns),
                "n_columns": len(kept_columns),
            }
        )

    effect_table = pd.DataFrame.from_records(effect_rows)
    kept_factor_ids = sorted({factor_id for factor_ids in effect_table["factor_ids"] for factor_id in factor_ids})
    factor_table = factor_table[factor_table["factor_id"].isin(kept_factor_ids)].reset_index(drop=True)
    return factor_table, effect_table, column_table


def _standardize_matrix(
    X: pd.DataFrame,
    target_norm: float | None = None,
    *,
    drop_zero_columns: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    if X.empty:
        return X.copy(), pd.Series(dtype=float), pd.Series(dtype=float), []

    column_means = X.mean(axis=0)
    centered = X - column_means
    centered_values = centered.to_numpy(dtype=float, copy=False)
    norms = pd.Series(np.linalg.norm(centered_values, axis=0), index=centered.columns, dtype=float)
    zero_columns = norms[norms <= 1e-12].index.tolist()
    if zero_columns:
        if not drop_zero_columns:
            raise ValueError(f"Zero-length columns encountered after centering: {zero_columns}")
        centered = centered.drop(columns=zero_columns)
        norms = norms.drop(index=zero_columns)

    if target_norm is None:
        # Match R's scale(..., center=TRUE, scale=TRUE), which uses sample SD.
        target_norm = math.sqrt(max(centered.shape[0] - 1, 1))
    if centered.empty:
        scaled = centered.copy()
    else:
        scaled_values = centered.to_numpy(dtype=float, copy=False) / norms.to_numpy(dtype=float)
        scaled = pd.DataFrame(
            scaled_values * float(target_norm),
            index=centered.index,
            columns=centered.columns,
        )
    return scaled, column_means, norms, zero_columns


def center_response(y: Sequence[float] | pd.Series | np.ndarray) -> pd.Series:
    y_series = pd.Series(np.asarray(y, dtype=float).reshape(-1), name="y")
    return y_series - float(y_series.mean())


def _mixed_main_column_names(factor_id: str, contrast_names: Sequence[str], _q_i: int) -> list[str]:
    return [f"{factor_id}__{contrast_name}" for contrast_name in contrast_names]


def _paper_exact_main_column_names(factor_id: str, contrast_names: Sequence[str], q_i: int) -> list[str]:
    if q_i == 2:
        return [factor_id]
    return [f"{factor_id}_{contrast_name}" for contrast_name in contrast_names]


def _mixed_interaction_column_id(
    factor_i: str,
    factor_j: str,
    _col_i: str,
    _col_j: str,
    contrast_i: str,
    contrast_j: str,
) -> str:
    return f"{factor_i}__{contrast_i}__x__{factor_j}__{contrast_j}"


def _paper_exact_interaction_column_id(
    _factor_i: str,
    _factor_j: str,
    col_i: str,
    col_j: str,
    _contrast_i: str,
    _contrast_j: str,
) -> str:
    return f"{col_i}:{col_j}"


def _build_design(
    factors: pd.DataFrame | Mapping[str, Sequence[Any]] | np.ndarray,
    *,
    q_values: Mapping[str, int] | None,
    factor_levels: Mapping[str, Sequence[Any]] | None,
    allow_level_inference: bool,
    standardize: bool,
    target_norm: float | None,
    design_id: str,
    contrast_builder: Callable[[int], pd.DataFrame],
    main_column_namer: Callable[[str, Sequence[str], int], list[str]],
    interaction_column_namer: Callable[[str, str, str, str, str, str], str],
    coding: str | None = None,
) -> MixedLevelDesign:
    factor_df = _as_dataframe(factors)
    factor_df.columns = [str(col) for col in factor_df.columns]

    if q_values is None:
        q_values = {factor_id: int(factor_df[factor_id].nunique()) for factor_id in factor_df.columns}
    q_values = _normalize_factor_order(q_values, factor_df.columns)
    factor_levels = {} if factor_levels is None else {str(k): list(v) for k, v in factor_levels.items()}

    factor_records: list[dict[str, Any]] = []
    effect_records: list[dict[str, Any]] = []
    column_records: list[dict[str, Any]] = []
    main_blocks: dict[str, pd.DataFrame] = {}
    main_contrast_lookup: dict[str, str] = {}

    for factor_id in factor_df.columns:
        q_i = q_values[factor_id]
        levels = _resolve_factor_levels(
            factor_df[factor_id],
            q_i,
            factor_levels.get(factor_id),
            allow_inference=allow_level_inference,
        )
        contrast_table = contrast_builder(q_i)
        contrast_names = [
            (str(name) if str(name) else "l") if coding == "paper_exact" else str(name)
            for name in contrast_table.columns
        ]
        level_to_index = {level: idx for idx, level in enumerate(levels)}
        row_index = factor_df[factor_id].map(level_to_index)
        if row_index.isna().any():
            raise ValueError(f"Factor '{factor_id}' contains unknown levels.")

        block = contrast_table.iloc[row_index.to_numpy(dtype=int)].reset_index(drop=True)
        block.columns = main_column_namer(factor_id, contrast_names, q_i)
        effect_id = f"main::{factor_id}"

        factor_records.append(
            {
                "factor_id": factor_id,
                "q_i": q_i,
                "levels": tuple(levels),
                "main_effect_id": effect_id,
            }
        )
        effect_records.append(
            {
                "effect_id": effect_id,
                "effect_type": "main",
                "factor_ids": (factor_id,),
                "column_ids": tuple(block.columns.tolist()),
                "n_columns": int(block.shape[1]),
            }
        )

        for column_id, contrast_name in zip(block.columns, contrast_names, strict=False):
            column_records.append(
                {
                    "column_id": column_id,
                    "effect_type": "main",
                    "factor_ids": (factor_id,),
                    "effect_id": effect_id,
                    "contrast_name": contrast_name,
                    "component_contrast_names": (contrast_name,),
                }
            )
            main_contrast_lookup[column_id] = contrast_name
        main_blocks[factor_id] = block

    X_main = pd.concat([main_blocks[factor_id] for factor_id in factor_df.columns], axis=1)
    interaction_blocks: list[pd.DataFrame] = []

    for factor_i, factor_j in combinations(factor_df.columns, 2):
        block_i = main_blocks[factor_i]
        block_j = main_blocks[factor_j]
        effect_id = f"interaction::{factor_i}::{factor_j}"
        interaction_dict: dict[str, np.ndarray] = {}
        effect_columns: list[str] = []

        for col_i, col_j in product(block_i.columns, block_j.columns):
            contrast_i = main_contrast_lookup[col_i]
            contrast_j = main_contrast_lookup[col_j]
            column_id = interaction_column_namer(factor_i, factor_j, col_i, col_j, contrast_i, contrast_j)
            interaction_dict[column_id] = block_i[col_i].to_numpy(dtype=float) * block_j[col_j].to_numpy(dtype=float)
            effect_columns.append(column_id)
            column_records.append(
                {
                    "column_id": column_id,
                    "effect_type": "interaction",
                    "factor_ids": (factor_i, factor_j),
                    "effect_id": effect_id,
                    "contrast_name": None,
                    "component_contrast_names": (contrast_i, contrast_j),
                }
            )

        effect_records.append(
            {
                "effect_id": effect_id,
                "effect_type": "interaction",
                "factor_ids": (factor_i, factor_j),
                "column_ids": tuple(effect_columns),
                "n_columns": len(effect_columns),
            }
        )
        interaction_blocks.append(pd.DataFrame(interaction_dict))

    X_int = pd.concat(interaction_blocks, axis=1) if interaction_blocks else pd.DataFrame(index=factor_df.index)
    factor_table = pd.DataFrame.from_records(factor_records)
    effect_table = pd.DataFrame.from_records(effect_records)
    column_table = pd.DataFrame.from_records(column_records)

    if standardize:
        target_norm = float(target_norm or math.sqrt(max(len(factor_df) - 1, 1)))
        X_main, main_means, main_norms, dropped_main = _standardize_matrix(
            X_main,
            target_norm=target_norm,
            drop_zero_columns=True,
        )
        X_int, int_means, int_norms, dropped_int = _standardize_matrix(
            X_int,
            target_norm=target_norm,
            drop_zero_columns=True,
        )
        factor_table, effect_table, column_table = _prune_metadata_tables(
            factor_table,
            effect_table,
            column_table,
            dropped_main + dropped_int,
        )
        standardization = {
            "target_norm": target_norm,
            "scale_style": "r_sample_sd",
            "main_means": main_means.to_dict(),
            "main_norms": main_norms.to_dict(),
            "int_means": int_means.to_dict(),
            "int_norms": int_norms.to_dict(),
            "dropped_zero_columns": dropped_main + dropped_int,
        }
    else:
        standardization = {"target_norm": None, "scale_style": None}

    if coding is not None:
        standardization["coding"] = coding

    X_full = pd.concat([X_main, X_int], axis=1)
    return MixedLevelDesign(
        design_id=design_id,
        factor_data=factor_df,
        X_main=X_main,
        X_int=X_int,
        X_full=X_full,
        factor_table=factor_table,
        effect_table=effect_table,
        column_table=column_table,
        effect_to_columns=_build_effect_to_columns(column_table),
        factor_to_main_effect=_build_factor_to_main_effect(factor_table),
        interaction_to_pair=_build_interaction_to_pair(effect_table),
        column_to_metadata=_build_column_to_metadata(column_table),
        standardization=standardization,
    )


def build_mixed_level_design(
    factors: pd.DataFrame | Mapping[str, Sequence[Any]] | np.ndarray,
    q_values: Mapping[str, int] | None = None,
    factor_levels: Mapping[str, Sequence[Any]] | None = None,
    allow_level_inference: bool = False,
    standardize: bool = True,
    target_norm: float | None = None,
    design_id: str = "mixed_level_design",
) -> MixedLevelDesign:
    """Construct the mixed-level main-effect and interaction model matrices."""

    return _build_design(
        factors,
        q_values=q_values,
        factor_levels=factor_levels,
        allow_level_inference=allow_level_inference,
        standardize=standardize,
        target_norm=target_norm,
        design_id=design_id,
        contrast_builder=orthogonal_polynomial_contrasts,
        main_column_namer=_mixed_main_column_names,
        interaction_column_namer=_mixed_interaction_column_id,
    )


def build_paper_exact_design(
    factors: pd.DataFrame | Mapping[str, Sequence[Any]] | np.ndarray,
    q_values: Mapping[str, int] | None = None,
    factor_levels: Mapping[str, Sequence[Any]] | None = None,
    allow_level_inference: bool = False,
    standardize: bool = True,
    target_norm: float | None = None,
    design_id: str = "paper_exact_design",
) -> MixedLevelDesign:
    """Construct the paper-exact model matrix used by the supplement R code."""

    return _build_design(
        factors,
        q_values=q_values,
        factor_levels=factor_levels,
        allow_level_inference=allow_level_inference,
        standardize=standardize,
        target_norm=target_norm,
        design_id=design_id,
        contrast_builder=paper_exact_polynomial_contrasts,
        main_column_namer=_paper_exact_main_column_names,
        interaction_column_namer=_paper_exact_interaction_column_id,
        coding="paper_exact",
    )


def derive_delta_grid(
    X: pd.DataFrame | np.ndarray,
    y: Sequence[float] | pd.Series | np.ndarray,
    lower_fraction: float = 0.05,
    upper_fraction: float = 0.85,
    n_values: int = 12,
) -> np.ndarray:
    """Construct a configurable delta grid relative to max |X^T y|."""

    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(center_response(y), dtype=float)
    max_corr = float(np.max(np.abs(X_mat.T @ y_vec))) if X_mat.size else 0.0
    if max_corr <= 1e-12:
        return np.array([0.0])
    fractions = np.linspace(lower_fraction, upper_fraction, n_values)
    return np.unique(np.round(fractions * max_corr, decimals=12))


def paper_exact_delta_grid(
    X: pd.DataFrame | np.ndarray,
    y: Sequence[float] | pd.Series | np.ndarray,
    lambda_n: int = 11,
) -> np.ndarray:
    """Return the delta grid used in the supplement R implementation.

    The R code forms `seq(0, delta0, length.out=lambda.n+1)`, drops zero, and
    then evaluates only the first `lambda.n - 1` positive values, which is
    equivalent to `{delta0/11, ..., 10*delta0/11}` when `lambda_n = 11`.
    """

    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(center_response(y), dtype=float)
    max_corr = float(np.max(np.abs(X_mat.T @ y_vec))) if X_mat.size else 0.0
    if max_corr <= 1e-12:
        return np.array([0.0])
    return np.linspace(max_corr / lambda_n, max_corr * (lambda_n - 1) / lambda_n, lambda_n - 1)


def _fit_ols(X: pd.DataFrame | np.ndarray, y: Sequence[float] | pd.Series | np.ndarray, selected_columns: Sequence[str]) -> OLSRefitResult:
    y_vec = np.asarray(y, dtype=float).reshape(-1)
    X_mat = np.asarray(X, dtype=float)
    n = y_vec.shape[0]
    selected_columns = list(selected_columns)

    if X_mat.size == 0 or not selected_columns:
        residuals = y_vec.copy()
        rss = float(np.sum(residuals**2))
        rss = max(rss, 1e-12)
        bic = float(n * np.log(rss / n))
        aic = float(n * np.log(rss / n))
        coefficients = pd.Series(dtype=float)
        fitted = np.zeros_like(y_vec)
        return OLSRefitResult(
            selected_columns=[],
            coefficients=coefficients,
            fitted=fitted,
            residuals=residuals,
            rss=rss,
            bic=bic,
            aic=aic,
            r_squared=0.0,
            rank=0,
            df_resid=n,
        )

    beta_hat, _, rank, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
    fitted = X_mat @ beta_hat
    residuals = y_vec - fitted
    rss = float(np.sum(residuals**2))
    rss = max(rss, 1e-12)
    k = len(selected_columns)
    bic = float(n * np.log(rss / n) + k * np.log(n))
    aic = float(n * np.log(rss / n) + 2 * k)
    total_ss = float(np.sum(y_vec**2))
    r_squared = float(1.0 - rss / total_ss) if total_ss > 1e-12 else 0.0
    coefficients = pd.Series(beta_hat, index=selected_columns, dtype=float)
    df_resid = max(n - int(rank), 0)
    return OLSRefitResult(
        selected_columns=selected_columns,
        coefficients=coefficients,
        fitted=fitted,
        residuals=residuals,
        rss=rss,
        bic=bic,
        aic=aic,
        r_squared=r_squared,
        rank=int(rank),
        df_resid=int(df_resid),
    )


def _nested_model_pvalue(
    y: np.ndarray,
    X_reduced: np.ndarray,
    X_full: np.ndarray,
) -> float:
    n = y.shape[0]
    reduced_rank = int(np.linalg.matrix_rank(X_reduced)) if X_reduced.size else 0
    full_rank = int(np.linalg.matrix_rank(X_full)) if X_full.size else 0

    if full_rank <= reduced_rank:
        return 1.0

    rss_reduced = _fit_ols(X_reduced, y, [f"x{i}" for i in range(X_reduced.shape[1])]).rss if X_reduced.size else float(np.sum(y**2))
    rss_full = _fit_ols(X_full, y, [f"x{i}" for i in range(X_full.shape[1])]).rss if X_full.size else float(np.sum(y**2))

    df_num = full_rank - reduced_rank
    df_den = n - full_rank
    if df_num <= 0 or df_den <= 0:
        return 1.0

    improvement = max(rss_reduced - rss_full, 0.0)
    if improvement <= 1e-14:
        return 1.0

    mse_full = rss_full / df_den
    if mse_full <= 1e-14:
        return 0.0

    f_stat = (improvement / df_num) / mse_full
    return float(stats.f.sf(f_stat, df_num, df_den))


def _ols_coefficient_pvalues(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
) -> pd.Series:
    if X.empty:
        return pd.Series(dtype=float)

    X_df = X.copy()
    X_df.columns = [str(col) for col in X_df.columns]
    y_vec = np.asarray(y, dtype=float).reshape(-1)
    n = y_vec.shape[0]
    X_mat = np.column_stack([np.ones(n), X_df.to_numpy(dtype=float)])
    beta_hat, _, rank, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
    fitted = X_mat @ beta_hat
    residuals = y_vec - fitted
    df_resid = n - int(rank)
    if df_resid <= 0:
        return pd.Series(1.0, index=X_df.columns, dtype=float)

    rss = float(np.sum(residuals**2))
    mse = rss / max(df_resid, 1)
    cov = np.linalg.pinv(X_mat.T @ X_mat) * mse
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    slope_beta = beta_hat[1:]
    slope_se = se[1:]

    p_values = np.ones(X_df.shape[1], dtype=float)
    estimable = slope_se > 1e-14
    if np.any(estimable):
        t_stats = slope_beta[estimable] / slope_se[estimable]
        p_values[estimable] = 2.0 * stats.t.sf(np.abs(t_stats), df_resid)
    if np.any(~estimable):
        p_values[~estimable] = np.where(np.abs(slope_beta[~estimable]) > 1e-14, 0.0, 1.0)
    return pd.Series(p_values, index=X_df.columns, dtype=float)


def _select_active_indices_from_beta(beta: np.ndarray, random_state: int | None = None) -> np.ndarray:
    abs_beta = np.abs(np.asarray(beta, dtype=float).reshape(-1))
    p = abs_beta.shape[0]
    if p == 0:
        return np.array([], dtype=int)
    if p == 1:
        return np.array([0], dtype=int) if abs_beta[0] > 1e-10 else np.array([], dtype=int)
    if np.allclose(abs_beta, 0.0):
        return np.array([], dtype=int)
    if np.allclose(abs_beta, abs_beta[0]):
        return np.arange(p, dtype=int)

    kmeans = KMeans(n_clusters=2, n_init=25, random_state=random_state)
    labels = kmeans.fit_predict(abs_beta.reshape(-1, 1))
    means = np.array([abs_beta[labels == cluster].mean() for cluster in range(2)], dtype=float)
    active_cluster = int(np.argmax(means))
    return np.flatnonzero(labels == active_cluster)


def _select_active_columns_paper_exact(
    beta: np.ndarray,
    column_ids: Sequence[str],
    full_column_count: int,
    random_state: int | None = None,
) -> list[str]:
    """Replicate the supplement's k-means screening rule as closely as possible."""

    beta_series = pd.Series(np.asarray(beta, dtype=float).reshape(-1), index=list(column_ids), dtype=float)
    nonzero = beta_series[np.abs(beta_series) > 1e-12]
    if nonzero.empty:
        return []

    ranked = nonzero.reindex(nonzero.abs().sort_values(ascending=False).index)
    if len(ranked) == 1:
        return ranked.index.tolist()

    padded = np.zeros(int(full_column_count), dtype=float)
    padded[: len(ranked)] = ranked.abs().to_numpy(dtype=float)
    if np.allclose(padded, padded[0]):
        return ranked.index.tolist()

    kmeans = KMeans(n_clusters=2, n_init=25, random_state=random_state)
    labels = kmeans.fit_predict(padded.reshape(-1, 1))
    cluster_size = int(np.sum(labels == labels[0]))
    keep_n = min(cluster_size, len(ranked))
    return ranked.index[:keep_n].tolist()


def _available_cvxpy_solvers() -> list[tuple[str, str]]:
    order: list[tuple[str, str]] = []
    for name in ("HIGHS", "CLARABEL", "SCS"):
        solver = getattr(cp, name, None)
        if solver is not None:
            order.append((name, solver))
    return order


def solve_dantzig_selector_path(
    X: pd.DataFrame | np.ndarray,
    y: Sequence[float] | pd.Series | np.ndarray,
    delta_grid: Sequence[float],
    solver_order: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Solve the Dantzig selector over a path of delta values."""

    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(center_response(y), dtype=float)
    p = X_mat.shape[1]

    if p == 0:
        return [{"delta": float(delta), "beta": np.zeros(0), "status": "empty"} for delta in delta_grid]

    gram = X_mat.T @ X_mat
    score = X_mat.T @ y_vec

    beta = cp.Variable(p)
    t = cp.Variable(p, nonneg=True)
    delta_param = cp.Parameter(nonneg=True)
    constraints = [
        score - gram @ beta <= delta_param,
        -score + gram @ beta <= delta_param,
        beta <= t,
        -beta <= t,
    ]
    problem = cp.Problem(cp.Minimize(cp.sum(t)), constraints)

    available = _available_cvxpy_solvers()
    if solver_order is not None:
        rank_map = {name: idx for idx, name in enumerate(solver_order)}
        available.sort(key=lambda item: rank_map.get(item[0], len(rank_map)))

    results: list[dict[str, Any]] = []
    for delta in delta_grid:
        delta_param.value = float(max(delta, 0.0))
        status = "failed"
        solution: np.ndarray | None = None
        last_error: Exception | None = None

        for solver_name, solver in available:
            try:
                problem.solve(solver=solver, warm_start=True, verbose=False)
            except Exception as exc:  # pragma: no cover - solver availability is environment-specific
                last_error = exc
                continue
            if problem.status in {"optimal", "optimal_inaccurate"} and beta.value is not None:
                status = f"{solver_name}:{problem.status}"
                solution = np.asarray(beta.value, dtype=float).reshape(-1)
                break

        if solution is None:
            warnings.warn(
                f"Dantzig selector failed for delta={delta!r}. Last solver error: {last_error}",
                RuntimeWarning,
                stacklevel=2,
            )
            solution = np.full(p, np.nan, dtype=float)

        results.append({"delta": float(delta), "beta": solution, "status": status})

    return results


def run_GDS(
    X: pd.DataFrame | np.ndarray,
    y: Sequence[float] | pd.Series | np.ndarray,
    effect_metadata: pd.DataFrame,
    delta_grid: Sequence[float],
    random_state: int | None = None,
    design_id: str | None = None,
    solver_order: Sequence[str] | None = None,
    paper_exact_kmeans: bool = False,
    full_column_count: int | None = None,
) -> GDSResult:
    """Run the Gauss-Dantzig Selector on one model matrix."""

    X_df = _as_dataframe(X)
    X_df.columns = [str(col) for col in X_df.columns]
    y_centered = center_response(y)

    required_columns = {"column_id", "effect_id", "factor_ids"}
    if not required_columns.issubset(effect_metadata.columns):
        missing = required_columns - set(effect_metadata.columns)
        raise KeyError(f"effect_metadata is missing required columns: {sorted(missing)}")

    column_table = effect_metadata.copy()
    column_table["column_id"] = column_table["column_id"].astype(str)
    column_table = column_table[column_table["column_id"].isin(X_df.columns)].copy()
    column_table = column_table.set_index("column_id", drop=False).loc[X_df.columns].reset_index(drop=True)

    path = solve_dantzig_selector_path(X_df, y_centered, delta_grid=delta_grid, solver_order=solver_order)
    rows: list[dict[str, Any]] = []

    for entry in path:
        beta = entry["beta"]
        status = entry["status"]
        if np.isnan(beta).any():
            rows.append(
                {
                    "delta": entry["delta"],
                    "beta_hat": beta,
                    "status": status,
                    "selected_columns": tuple(),
                    "selected_effects": tuple(),
                    "selected_factors": tuple(),
                    "ols_coefficients": pd.Series(dtype=float),
                    "rss": np.inf,
                    "bic": np.inf,
                    "n_selected_columns": 0,
                }
            )
            continue

        if paper_exact_kmeans:
            selected_columns = _select_active_columns_paper_exact(
                beta,
                column_ids=X_df.columns.tolist(),
                full_column_count=int(full_column_count or X_df.shape[1]),
                random_state=random_state,
            )
        else:
            active_idx = _select_active_indices_from_beta(beta, random_state=random_state)
            selected_columns = X_df.columns[active_idx].tolist()
        selected_effects = sorted(column_table[column_table["column_id"].isin(selected_columns)]["effect_id"].unique().tolist())
        selected_factors = sorted(
            {
                factor_id
                for factor_ids in column_table[column_table["column_id"].isin(selected_columns)]["factor_ids"]
                for factor_id in factor_ids
            }
        )
        X_selected = X_df[selected_columns] if selected_columns else pd.DataFrame(index=X_df.index)
        ols_result = _fit_ols(X_selected, y_centered, selected_columns)

        rows.append(
            {
                "delta": entry["delta"],
                "beta_hat": beta,
                "status": status,
                "selected_columns": tuple(selected_columns),
                "selected_effects": tuple(selected_effects),
                "selected_factors": tuple(selected_factors),
                "ols_coefficients": ols_result.coefficients,
                "rss": ols_result.rss,
                "bic": ols_result.bic,
                "n_selected_columns": len(selected_columns),
            }
        )

    path_results = pd.DataFrame(rows)
    best_idx = int(path_results["bic"].astype(float).idxmin())
    best_row = path_results.loc[best_idx]
    best_beta = pd.Series(best_row["beta_hat"], index=X_df.columns, dtype=float)
    best_selected_columns = list(best_row["selected_columns"])
    best_selected_effects = list(best_row["selected_effects"])
    best_selected_factors = list(best_row["selected_factors"])
    best_ols_result = _fit_ols(X_df[best_selected_columns], y_centered, best_selected_columns) if best_selected_columns else _fit_ols(pd.DataFrame(index=X_df.index), y_centered, [])

    return GDSResult(
        design_id=design_id,
        delta_grid=np.asarray(delta_grid, dtype=float),
        path_results=path_results,
        best_delta=float(best_row["delta"]),
        best_beta=best_beta,
        best_selected_columns=best_selected_columns,
        best_selected_effects=best_selected_effects,
        best_selected_factors=best_selected_factors,
        best_ols_result=best_ols_result,
        best_bic=float(best_row["bic"]),
        solver_status=str(best_row["status"]),
    )


def _coerce_arm_matrices(
    X_main: pd.DataFrame | np.ndarray,
    X_int: pd.DataFrame | np.ndarray,
    metadata: MixedLevelDesign,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_main_df = _as_dataframe(X_main)
    X_int_df = _as_dataframe(X_int)
    if not isinstance(X_main, pd.DataFrame):
        X_main_df.columns = metadata.X_main.columns[: X_main_df.shape[1]]
    else:
        X_main_df.columns = [str(col) for col in X_main_df.columns]
    if not isinstance(X_int, pd.DataFrame):
        X_int_df.columns = metadata.X_int.columns[: X_int_df.shape[1]]
    else:
        X_int_df.columns = [str(col) for col in X_int_df.columns]

    if not set(X_main_df.columns).issubset(metadata.X_main.columns):
        raise ValueError("X_main columns must be a subset of metadata.X_main columns.")
    if not set(X_int_df.columns).issubset(metadata.X_int.columns):
        raise ValueError("X_int columns must be a subset of metadata.X_int columns.")
    return X_main_df, X_int_df


def _resolve_gds_delta_grid(
    X: pd.DataFrame,
    y: pd.Series,
    delta_grid: Sequence[float] | np.ndarray | None,
    *,
    paper_exact_baseline: bool,
) -> np.ndarray:
    if delta_grid is None:
        if paper_exact_baseline:
            return paper_exact_delta_grid(X, y)
        raise ValueError("delta_grid must be provided unless paper_exact_baseline=True.")
    return np.asarray(delta_grid, dtype=float)


def _subset_gds_problem(
    metadata: MixedLevelDesign,
    column_lookup: pd.DataFrame,
    rep_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep_columns = list(rep_columns)
    return (
        metadata.X_full.loc[:, rep_columns].copy(),
        column_lookup.loc[rep_columns].reset_index(drop=True).copy(),
    )


def run_GDS_step_i(
    X_main: pd.DataFrame | np.ndarray,
    X_int: pd.DataFrame | np.ndarray,
    y: Sequence[float] | pd.Series | np.ndarray,
    metadata: MixedLevelDesign,
    nint: int,
    delta_grid: Sequence[float] | None,
    random_state: int | np.random.Generator | None = None,
    solver_order: Sequence[str] | None = None,
    paper_exact_baseline: bool = False,
) -> GDSStepIResult:
    """Run the core GDS-ARM step (i): one random-interaction GDS submodel."""

    rng = _ensure_rng(random_state)
    X_main_df, X_int_df = _coerce_arm_matrices(X_main, X_int, metadata)
    main_columns = X_main_df.columns.tolist()
    interaction_columns = X_int_df.columns.to_numpy()
    column_lookup = metadata.column_table.set_index("column_id", drop=False)
    nint = int(min(max(nint, 0), X_int_df.shape[1]))
    sampled_interactions = (
        rng.choice(interaction_columns, size=nint, replace=False).tolist() if nint > 0 else []
    )
    rep_columns = main_columns + sampled_interactions
    rep_X_full, rep_column_table = _subset_gds_problem(metadata, column_lookup, rep_columns)
    y_centered = center_response(y)
    current_delta_grid = _resolve_gds_delta_grid(
        rep_X_full,
        y_centered,
        delta_grid,
        paper_exact_baseline=paper_exact_baseline,
    )
    gds_result = run_GDS(
        rep_X_full,
        y_centered,
        effect_metadata=rep_column_table,
        delta_grid=current_delta_grid,
        random_state=int(rng.integers(0, 2**31 - 1)),
        design_id=f"{metadata.design_id}_step_i",
        solver_order=solver_order,
        paper_exact_kmeans=paper_exact_baseline,
        full_column_count=metadata.X_full.shape[1],
    )
    return GDSStepIResult(
        design_id=f"{metadata.design_id}_step_i",
        sampled_interaction_columns=sampled_interactions,
        gds_result=gds_result,
    )


def _resolve_group_structure(
    design: MixedLevelDesign,
    group_stepwise: bool,
) -> tuple[dict[str, list[str]], pd.Series]:
    effect_types = pd.Series(
        dict(zip(design.effect_table["effect_id"], design.effect_table["effect_type"], strict=False))
    )
    if group_stepwise:
        group_to_columns = {effect_id: columns[:] for effect_id, columns in design.effect_to_columns.items()}
        return group_to_columns, effect_types

    group_to_columns = {column_id: [column_id] for column_id in design.column_ids}
    effect_rows = []
    for column_id in design.column_ids:
        effect_rows.append((column_id, design.column_to_metadata[column_id]["effect_type"]))
    return group_to_columns, pd.Series(dict(effect_rows))


def _run_stepwise_selection(
    X: pd.DataFrame,
    y: pd.Series,
    group_to_columns: Mapping[str, list[str]],
    group_types: pd.Series,
    start_groups: Sequence[str],
    add_candidate_groups: Sequence[str],
    stepwise_mode: str = "bidirectional",
    p_enter: float = 0.15,
    p_remove: float = 0.15,
) -> StepwiseResult:
    stepwise_mode = stepwise_mode.lower()
    if stepwise_mode not in {"forward", "backward", "bidirectional"}:
        raise ValueError("stepwise_mode must be one of {'forward', 'backward', 'bidirectional'}.")

    current = list(dict.fromkeys(start_groups))
    all_add_candidates = set(add_candidate_groups)
    add_pool = [group for group in dict.fromkeys(add_candidate_groups) if group not in current]
    history: list[dict[str, Any]] = []
    y_vec = y.to_numpy(dtype=float)

    def current_columns(groups: Sequence[str]) -> list[str]:
        columns: list[str] = []
        for group in groups:
            columns.extend(group_to_columns[group])
        return list(dict.fromkeys(columns))

    changed = True
    while changed:
        changed = False

        if stepwise_mode in {"forward", "bidirectional"} and add_pool:
            base_columns = current_columns(current)
            X_reduced = X[base_columns].to_numpy(dtype=float) if base_columns else np.empty((len(y), 0))
            best_group = None
            best_p = np.inf

            for group in add_pool:
                candidate_columns = current_columns(current + [group])
                X_full = X[candidate_columns].to_numpy(dtype=float)
                p_value = _nested_model_pvalue(y_vec, X_reduced, X_full)
                if p_value < best_p:
                    best_p = p_value
                    best_group = group

            if best_group is not None and best_p <= p_enter:
                current.append(best_group)
                add_pool = [group for group in add_pool if group != best_group]
                history.append({"action": "add", "group": best_group, "p_value": float(best_p)})
                changed = True

        if stepwise_mode in {"backward", "bidirectional"} and current:
            while True:
                full_columns = current_columns(current)
                X_full = X[full_columns].to_numpy(dtype=float) if full_columns else np.empty((len(y), 0))
                worst_group = None
                worst_p = -np.inf

                for group in current:
                    reduced_groups = [candidate for candidate in current if candidate != group]
                    reduced_columns = current_columns(reduced_groups)
                    X_reduced = X[reduced_columns].to_numpy(dtype=float) if reduced_columns else np.empty((len(y), 0))
                    p_value = _nested_model_pvalue(y_vec, X_reduced, X_full)
                    if p_value > worst_p:
                        worst_p = p_value
                        worst_group = group

                if worst_group is None or worst_p <= p_remove:
                    break

                current = [group for group in current if group != worst_group]
                if worst_group in all_add_candidates and group_types.get(worst_group, "") == "main":
                    add_pool.append(worst_group)
                    add_pool = list(dict.fromkeys(add_pool))
                history.append({"action": "remove", "group": worst_group, "p_value": float(worst_p)})
                changed = True

                if stepwise_mode == "backward":
                    continue
                break

    selected_columns = current_columns(current)
    fit = _fit_ols(X[selected_columns], y, selected_columns) if selected_columns else _fit_ols(pd.DataFrame(index=X.index), y, [])
    history_df = pd.DataFrame(history, columns=["action", "group", "p_value"])
    return StepwiseResult(
        selected_groups=current,
        selected_columns=selected_columns,
        coefficients=fit.coefficients,
        fit=fit,
        history=history_df,
    )


def _run_paper_exact_stepwise(
    X: pd.DataFrame,
    y: pd.Series,
    start_columns: Sequence[str],
    main_columns: Sequence[str],
    p_enter: float = 0.01,
    p_remove: float = 0.05,
    max_loops: int = 30,
) -> StepwiseResult:
    """Replicate the supplement's final column-wise stepwise refinement."""

    X_df = X.copy()
    X_df.columns = [str(col) for col in X_df.columns]
    current = [column for column in dict.fromkeys(start_columns) if column in X_df.columns]
    xadd = [column for column in dict.fromkeys(list(current) + list(main_columns)) if column in X_df.columns]
    remaining = [column for column in xadd if column not in current]
    history: list[dict[str, Any]] = []
    model_finished = False
    deadloop = 0

    while len(current) + 2 < len(y) and not model_finished:
        deadloop += 1
        if deadloop > max_loops:
            break

        move_forward = False
        move_backward = False

        if remaining:
            criteria = []
            for candidate in remaining:
                candidate_columns = current + [candidate]
                p_values = _ols_coefficient_pvalues(X_df[candidate_columns], y)
                criteria.append(float(p_values.loc[candidate]))

            best_idx = int(np.argmin(criteria))
            best_p = float(criteria[best_idx])
            if best_p <= p_enter:
                added = remaining.pop(best_idx)
                current.append(added)
                history.append({"action": "add", "group": added, "p_value": best_p})
                move_forward = True

        if current:
            p_values = _ols_coefficient_pvalues(X_df[current], y)
            if not p_values.empty:
                worst_column = str(p_values.idxmax())
                worst_p = float(p_values.loc[worst_column])
            else:
                worst_column = ""
                worst_p = 0.0
        else:
            worst_column = ""
            worst_p = 0.0

        if current and worst_p > p_remove and worst_p <= 1.0:
            current = [column for column in current if column != worst_column]
            if worst_column and worst_column not in remaining:
                remaining.append(worst_column)
            history.append({"action": "remove", "group": worst_column, "p_value": worst_p})
            move_backward = True

        if len(current) == 0 or len(remaining) == 0 or len(current) >= len(y) or (not move_forward and not move_backward):
            model_finished = True

    fit = _fit_ols(X_df[current], y, current) if current else _fit_ols(pd.DataFrame(index=X_df.index), y, [])
    history_df = pd.DataFrame(history, columns=["action", "group", "p_value"])
    return StepwiseResult(
        selected_groups=current[:],
        selected_columns=current[:],
        coefficients=fit.coefficients,
        fit=fit,
        history=history_df,
    )


def run_GDS_ARM(
    X_main: pd.DataFrame | np.ndarray,
    X_int: pd.DataFrame | np.ndarray,
    y: Sequence[float] | pd.Series | np.ndarray,
    metadata: MixedLevelDesign,
    nrep: int,
    nint: int,
    ntop: int,
    pkeep: float,
    delta_grid: Sequence[float] | None,
    stepwise_mode: str = "bidirectional",
    p_enter: float = 0.15,
    p_remove: float = 0.15,
    group_stepwise: bool = True,
    paper_exact_stepwise: bool = False,
    paper_exact_baseline: bool = False,
    random_state: int | np.random.Generator | None = None,
    final_stepwise: bool = True,
    solver_order: Sequence[str] | None = None,
    show_progress: bool = False,
) -> GDSARMResult:
    """Run GDS-ARM on a centered response."""

    rng = _ensure_rng(random_state)
    X_main_df, X_int_df = _coerce_arm_matrices(X_main, X_int, metadata)
    y_centered = center_response(y)
    fixed_delta_grid = None if delta_grid is None else np.asarray(delta_grid, dtype=float)
    main_columns = X_main_df.columns.tolist()
    interaction_columns = X_int_df.columns.to_numpy()
    column_lookup = metadata.column_table.set_index("column_id", drop=False)

    repetition_rows: list[dict[str, Any]] = []
    nint = int(min(max(nint, 0), X_int_df.shape[1]))
    rep_progress = _make_progress(
        total=int(nrep),
        desc=f"GDS-ARM reps [{metadata.design_id}]",
        disable=not show_progress,
        leave=False,
    )

    try:
        for rep_idx in range(1, int(nrep) + 1):
            sampled_interactions = (
                rng.choice(interaction_columns, size=nint, replace=False).tolist() if nint > 0 else []
            )
            rep_columns = main_columns + sampled_interactions
            rep_X_full, rep_column_table = _subset_gds_problem(metadata, column_lookup, rep_columns)
            current_delta_grid = _resolve_gds_delta_grid(
                rep_X_full,
                y_centered,
                fixed_delta_grid,
                paper_exact_baseline=paper_exact_baseline,
            )
            gds_result = run_GDS(
                rep_X_full,
                y_centered,
                effect_metadata=rep_column_table,
                delta_grid=current_delta_grid,
                random_state=int(rng.integers(0, 2**31 - 1)),
                design_id=f"{metadata.design_id}_rep_{rep_idx}",
                solver_order=solver_order,
                paper_exact_kmeans=paper_exact_baseline,
                full_column_count=metadata.X_full.shape[1],
            )
            repetition_rows.append(
                {
                    "rep": rep_idx,
                    "sampled_interaction_columns": tuple(sampled_interactions),
                    "selected_columns": tuple(gds_result.best_selected_columns),
                    "selected_effects": tuple(gds_result.best_selected_effects),
                    "selected_factors": tuple(gds_result.best_selected_factors),
                    "bic": gds_result.best_bic,
                    "delta": gds_result.best_delta,
                    "ols_coefficients": gds_result.best_ols_result.coefficients,
                    "solver_status": gds_result.solver_status,
                }
            )
            rep_progress.update(1)
    finally:
        rep_progress.close()

    repetition_results = pd.DataFrame(repetition_rows).sort_values("bic", ascending=True, kind="stable").reset_index(drop=True)
    top_results = repetition_results.head(min(max(int(ntop), 0), len(repetition_results))).copy()
    (
        effect_frequency_table,
        aggregated_effects,
        aggregated_columns,
        threshold_count,
    ) = _baseline_frequency_table(
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
    ) = _finalize_selection_from_aggregates(
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
        "nrep": int(nrep),
        "nint": int(nint),
        "ntop": int(ntop),
        "pkeep": float(pkeep),
        "threshold_count": threshold_count,
        "aggregation_unit": "column" if paper_exact_baseline else "effect",
        "group_stepwise": bool(group_stepwise),
        "paper_exact_stepwise": bool(paper_exact_stepwise),
        "paper_exact_baseline": bool(paper_exact_baseline),
        "stepwise_mode": stepwise_mode,
        "p_enter": float(p_enter),
        "p_remove": float(p_remove),
        "final_bic": final_stepwise_result.fit.bic,
        "final_rss": final_stepwise_result.fit.rss,
    }

    return GDSARMResult(
        design_id=metadata.design_id,
        repetition_results=repetition_results,
        top_results=top_results,
        effect_frequency_table=effect_frequency_table,
        aggregated_effects=aggregated_effects,
        final_stepwise_result=final_stepwise_result,
        final_selected_effects=final_selected_effects,
        final_selected_columns=final_selected_columns,
        final_selected_factors=final_selected_factors,
        diagnostics=diagnostics,
    )


def _build_baseline_repetition_path(
    design: MixedLevelDesign,
    y: Sequence[float] | pd.Series | np.ndarray,
    *,
    nrep: int,
    nint: int,
    delta_grid: Sequence[float] | None,
    random_state: int | np.random.Generator | None,
    solver_order: Sequence[str] | None,
    paper_exact_baseline: bool,
) -> pd.DataFrame:
    """Build a reusable baseline repetition path for one dataset."""

    rng = _ensure_rng(random_state)
    nint = int(min(max(int(nint), 0), design.X_int.shape[1]))
    repetition_rows: list[dict[str, Any]] = []
    cumulative_elapsed = 0.0

    for rep_idx in range(1, int(nrep) + 1):
        rep_t0 = time.perf_counter()
        step_i_result = run_GDS_step_i(
            design.X_main,
            design.X_int,
            y,
            metadata=design,
            nint=nint,
            delta_grid=delta_grid,
            random_state=int(rng.integers(0, 2**31 - 1)),
            solver_order=solver_order,
            paper_exact_baseline=paper_exact_baseline,
        )
        rep_elapsed = time.perf_counter() - rep_t0
        cumulative_elapsed += rep_elapsed
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
                "rep_timing_seconds": rep_elapsed,
                "cumulative_timing_seconds": cumulative_elapsed,
            }
        )

    return pd.DataFrame.from_records(repetition_rows)


def _baseline_frequency_table(
    top_results: pd.DataFrame,
    metadata: MixedLevelDesign,
    *,
    pkeep: float,
    paper_exact_baseline: bool,
) -> tuple[pd.DataFrame, list[str], list[str], int]:
    """Apply the original count-threshold aggregation rule to a top-model set."""

    n_top = len(top_results)
    threshold_count = int(ceil(float(pkeep) * max(n_top, 1)))
    effect_type_lookup = dict(
        zip(metadata.effect_table["effect_id"], metadata.effect_table["effect_type"], strict=False)
    )

    if paper_exact_baseline:
        column_counts: MutableMapping[str, int] = {}
        for selected_columns in top_results["selected_columns"]:
            for column_id in selected_columns:
                column_counts[column_id] = column_counts.get(column_id, 0) + 1

        frequency_rows = []
        for column_id in metadata.X_full.columns.tolist():
            count = int(column_counts.get(column_id, 0))
            frequency_rows.append(
                {
                    "feature_id": column_id,
                    "aggregation_unit": "column",
                    "count": count,
                    "proportion": count / n_top if n_top else 0.0,
                    "kept": count >= threshold_count and n_top > 0,
                    "effect_type": metadata.column_to_metadata[column_id]["effect_type"],
                }
            )
        feature_table = pd.DataFrame(frequency_rows).sort_values(
            ["kept", "count", "feature_id"], ascending=[False, False, True]
        )
        aggregated_effects = sorted(
            metadata.column_table[
                metadata.column_table["column_id"].isin(
                    feature_table.loc[feature_table["kept"], "feature_id"].tolist()
                )
            ]["effect_id"].unique().tolist()
        )
        aggregated_columns = feature_table.loc[feature_table["kept"], "feature_id"].tolist()
    else:
        effect_counts: MutableMapping[str, int] = {}
        for selected_effects in top_results["selected_effects"]:
            for effect_id in selected_effects:
                effect_counts[effect_id] = effect_counts.get(effect_id, 0) + 1

        frequency_rows = []
        for effect_id in metadata.effect_ids:
            count = int(effect_counts.get(effect_id, 0))
            frequency_rows.append(
                {
                    "feature_id": effect_id,
                    "aggregation_unit": "effect",
                    "count": count,
                    "proportion": count / n_top if n_top else 0.0,
                    "kept": count >= threshold_count and n_top > 0,
                    "effect_type": effect_type_lookup[effect_id],
                }
            )
        feature_table = pd.DataFrame(frequency_rows).sort_values(
            ["kept", "count", "feature_id"], ascending=[False, False, True]
        )
        aggregated_effects = feature_table.loc[feature_table["kept"], "feature_id"].tolist()
        aggregated_columns = metadata.columns_for_effects(aggregated_effects)

    return feature_table, aggregated_effects, aggregated_columns, threshold_count


def _finalize_selection_from_aggregates(
    metadata: MixedLevelDesign,
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
) -> tuple[StepwiseResult, list[str], list[str], list[str]]:
    """Apply the post-aggregation selection cleanup used by GDS-ARM."""

    y_centered = center_response(y)
    aggregated_effects = list(aggregated_effects)
    aggregated_columns = list(aggregated_columns)

    if final_stepwise:
        if paper_exact_stepwise:
            final_stepwise_result = _run_paper_exact_stepwise(
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
            group_to_columns, group_types = _resolve_group_structure(metadata, group_stepwise=group_stepwise)
            if group_stepwise:
                start_groups = aggregated_effects[:]
                remaining_main = [
                    effect_id
                    for effect_id in metadata.effect_table.loc[metadata.effect_table["effect_type"] == "main", "effect_id"].tolist()
                    if effect_id not in aggregated_effects
                ]
            else:
                start_groups = aggregated_columns
                interaction_columns = set(metadata.X_int.columns)
                remaining_main = [
                    column_id
                    for column_id in metadata.X_main.columns.tolist()
                    if column_id not in start_groups and column_id not in interaction_columns
                ]

            final_stepwise_result = _run_stepwise_selection(
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
                final_selected_effects = final_stepwise_result.selected_groups
            else:
                final_selected_effects = sorted(
                    metadata.column_table[
                        metadata.column_table["column_id"].isin(final_stepwise_result.selected_columns)
                    ]["effect_id"].unique().tolist()
                )
    else:
        final_columns = aggregated_columns
        fit = _fit_ols(metadata.X_full[final_columns], y_centered, final_columns) if final_columns else _fit_ols(pd.DataFrame(index=metadata.X_full.index), y_centered, [])
        final_stepwise_result = StepwiseResult(
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


def _build_original_baseline_variant_result(
    repetition_results: pd.DataFrame,
    metadata: MixedLevelDesign,
    y: Sequence[float] | pd.Series | np.ndarray,
    *,
    nrep: int,
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
    """Build the original baseline result from a reusable repetition path."""

    prefix_len = min(max(int(nrep), 0), len(repetition_results))
    prefix_results = repetition_results.head(prefix_len).copy()
    ranked_results = prefix_results.sort_values("bic", ascending=True, kind="stable").reset_index(drop=True)
    top_results = ranked_results.head(min(max(int(ntop), 0), len(ranked_results))).copy()
    feature_table, aggregated_effects, aggregated_columns, threshold_count = _baseline_frequency_table(
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
    ) = _finalize_selection_from_aggregates(
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
        "nrep_used": int(prefix_len),
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


def design_diagnostic_table(designs: Sequence[MixedLevelDesign]) -> pd.DataFrame:
    rows = []
    for design in designs:
        r = design.r_ratio
        if r > 1.7:
            note = "Potentially poor choice for GDS-ARM"
        elif r > 1.5:
            note = "Usable but not preferred"
        else:
            note = "Preferred range"
        rows.append(
            {
                "design_id": design.design_id,
                "n_runs": design.n_runs,
                "m_factors": len(design.factor_ids),
                "ncolmain": design.ncol_main,
                "r": r,
                "flag_r_gt_1_7": bool(r > 1.7),
                "note": note,
            }
        )
    return pd.DataFrame(rows).sort_values(["r", "design_id"]).reset_index(drop=True)


def _resolve_design_builder(coding: str) -> Callable[..., MixedLevelDesign]:
    normalized = str(coding).strip().lower()
    if normalized == "paper_exact":
        return build_paper_exact_design
    if normalized in {"mixed_level", "mixed"}:
        return build_mixed_level_design
    raise ValueError("generated_design_coding must be one of {'paper_exact', 'mixed_level'}.")


def _balanced_level_indices(
    n_runs: int,
    q_levels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = np.full(q_levels, n_runs // q_levels, dtype=int)
    remainder = n_runs % q_levels
    if remainder:
        counts[rng.permutation(q_levels)[:remainder]] += 1

    values = np.concatenate(
        [np.full(count, level_idx, dtype=int) for level_idx, count in enumerate(counts, start=0) if count > 0]
    )
    rng.shuffle(values)
    return values


def _factor_block_for_generation(
    level_indices: np.ndarray,
    q_levels: int,
    *,
    coding: str,
) -> np.ndarray:
    if q_levels < 2:
        raise ValueError("Each generated factor must have at least 2 levels.")

    normalized_coding = str(coding).strip().lower()
    contrast_builder = paper_exact_polynomial_contrasts if normalized_coding == "paper_exact" else orthogonal_polynomial_contrasts
    contrast_table = contrast_builder(q_levels)
    block = contrast_table.iloc[np.asarray(level_indices, dtype=int)].reset_index(drop=True)
    standardized, _, _, dropped = _standardize_matrix(
        block,
        target_norm=math.sqrt(max(len(level_indices) - 1, 1)),
        drop_zero_columns=True,
    )
    if dropped:
        raise ValueError(f"Generated factor block dropped zero-length columns unexpectedly: {dropped}")
    return standardized.to_numpy(dtype=float, copy=False)


def _main_effect_alias_score(X_main: np.ndarray, n_runs: int) -> tuple[float, float]:
    if X_main.size == 0 or X_main.shape[1] <= 1:
        return 0.0, 0.0

    gram = np.abs((X_main.T @ X_main) / float(n_runs))
    off_diag = gram[~np.eye(gram.shape[0], dtype=bool)]
    if off_diag.size == 0:
        return 0.0, 0.0
    return float(off_diag.max()), float(off_diag.mean())


def generate_mixed_level_factor_data(
    n_runs: int,
    q_values: Mapping[str, int] | Sequence[int],
    random_state: int | np.random.Generator | None = None,
    *,
    factor_ids: Sequence[str] | None = None,
    coding: str = "paper_exact",
    candidate_pool_size: int = 64,
    n_starts: int = 8,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, list[int]], dict[str, float]]:
    """Generate a mixed-level SSD proxy with balanced levels and low main-effect aliasing."""

    rng = _ensure_rng(random_state)
    if isinstance(q_values, Mapping):
        resolved_factor_ids = [str(factor_id) for factor_id in q_values]
        q_map = {str(factor_id): int(q_values[factor_id]) for factor_id in resolved_factor_ids}
    else:
        q_sequence = [int(value) for value in q_values]
        if factor_ids is None:
            resolved_factor_ids = [f"X{idx + 1}" for idx in range(len(q_sequence))]
        else:
            resolved_factor_ids = [str(factor_id) for factor_id in factor_ids]
        if len(resolved_factor_ids) != len(q_sequence):
            raise ValueError("factor_ids and q_values must have the same length.")
        q_map = dict(zip(resolved_factor_ids, q_sequence, strict=False))

    if not q_map:
        raise ValueError("q_values must contain at least one factor.")

    normalized_coding = str(coding).strip().lower()
    _resolve_design_builder(normalized_coding)
    candidate_pool_size = max(int(candidate_pool_size), 1)
    n_starts = max(int(n_starts), 1)

    best_assignments: dict[str, np.ndarray] | None = None
    best_score = (np.inf, np.inf)

    for _ in range(n_starts):
        factor_order = rng.permutation(resolved_factor_ids).tolist()
        selected_matrix = np.empty((int(n_runs), 0), dtype=float)
        current_assignments: dict[str, np.ndarray] = {}

        for factor_id in factor_order:
            q_i = int(q_map[factor_id])
            best_factor_levels: np.ndarray | None = None
            best_factor_block: np.ndarray | None = None
            best_factor_score = (np.inf, np.inf)

            for _candidate_idx in range(candidate_pool_size):
                level_indices = _balanced_level_indices(int(n_runs), q_i, rng)
                block = _factor_block_for_generation(level_indices, q_i, coding=normalized_coding)
                if selected_matrix.shape[1]:
                    cross = np.abs((selected_matrix.T @ block) / float(n_runs))
                    candidate_score = (float(cross.max()), float(cross.mean()))
                else:
                    candidate_score = (0.0, 0.0)

                if candidate_score < best_factor_score:
                    best_factor_levels = level_indices.copy()
                    best_factor_block = block.copy()
                    best_factor_score = candidate_score
                    if candidate_score[0] <= 1e-12:
                        break

            if best_factor_levels is None or best_factor_block is None:
                raise RuntimeError(f"Failed to generate a candidate block for factor '{factor_id}'.")

            current_assignments[factor_id] = best_factor_levels
            selected_matrix = (
                best_factor_block
                if selected_matrix.shape[1] == 0
                else np.column_stack([selected_matrix, best_factor_block])
            )

        overall_score = _main_effect_alias_score(selected_matrix, int(n_runs))
        if overall_score < best_score or best_assignments is None:
            best_assignments = {factor_id: values.copy() for factor_id, values in current_assignments.items()}
            best_score = overall_score

    if best_assignments is None:
        raise RuntimeError("Design generation failed to produce any candidate design.")

    factor_df = pd.DataFrame(
        {factor_id: best_assignments[factor_id].astype(int).tolist() for factor_id in resolved_factor_ids}
    )
    factor_levels = {factor_id: list(range(int(q_map[factor_id]))) for factor_id in resolved_factor_ids}
    generation_diagnostics = {
        "candidate_pool_size": float(candidate_pool_size),
        "n_starts": float(n_starts),
        "max_abs_main_corr": float(best_score[0]),
        "mean_abs_main_corr": float(best_score[1]),
    }
    return factor_df, q_map, factor_levels, generation_diagnostics


def generate_benchmark_design(
    design_id: str,
    random_state: int | np.random.Generator | None = None,
    *,
    coding: str = "paper_exact",
    candidate_pool_size: int = 64,
    n_starts: int = 8,
) -> MixedLevelDesign:
    """Generate one benchmark design matching the paper's n/m/q template but not its exact rows."""

    if design_id not in _GENERATED_DESIGN_SPECS:
        raise KeyError(
            f"Unknown generated design id '{design_id}'. "
            f"Known generated templates: {sorted(_GENERATED_DESIGN_SPECS)}"
        )

    spec = _GENERATED_DESIGN_SPECS[design_id]
    factor_ids = [f"X{idx + 1}" for idx in range(len(spec["q_values"]))]
    factor_df, q_map, factor_levels, _generation_diagnostics = generate_mixed_level_factor_data(
        int(spec["n_runs"]),
        spec["q_values"],
        random_state=random_state,
        factor_ids=factor_ids,
        coding=coding,
        candidate_pool_size=candidate_pool_size,
        n_starts=n_starts,
    )

    builder = _resolve_design_builder(coding)
    design = builder(
        factor_df,
        q_values=q_map,
        factor_levels=factor_levels,
        allow_level_inference=False,
        design_id=design_id,
    )
    return design


def generate_benchmark_design_suite(
    design_ids: Sequence[str] | None = None,
    *,
    random_state: int | np.random.Generator | None = None,
    coding: str = "paper_exact",
    candidate_pool_size: int = 64,
    n_starts: int = 8,
) -> dict[str, Any]:
    """Generate a suite of benchmark designs with the paper's dimension templates."""

    rng = _ensure_rng(random_state)
    if design_ids is None:
        design_ids = tuple(_GENERATED_DESIGN_SPECS)

    all_designs = [
        generate_benchmark_design(
            str(design_id),
            random_state=int(rng.integers(0, 2**31 - 1)),
            coding=coding,
            candidate_pool_size=candidate_pool_size,
            n_starts=n_starts,
        )
        for design_id in design_ids
    ]
    by_id = {design.design_id: design for design in all_designs}
    return {
        "all_designs": all_designs,
        "by_id": by_id,
    }


def _select_interaction_effects_for_scenario(
    design: MixedLevelDesign,
    active_main_effects: list[str],
    mint: int,
    mimp: int,
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    factor_ids = design.factor_ids
    active_main_factors = [effect_id.split("::", 1)[1] for effect_id in active_main_effects]
    active_main_set = set(active_main_factors)

    if mint == 0:
        if mimp != len(active_main_set):
            raise ValueError("When mint=0, mimp must equal the number of active main effects.")
        return [], sorted(active_main_set)

    if mimp < len(active_main_set):
        raise ValueError("mimp cannot be smaller than the number of active main effects.")

    additional_needed = mimp - len(active_main_set)
    if additional_needed > mint:
        raise ValueError(
            "Weak heredity cannot introduce more new important factors than the number of active interactions."
        )

    remaining_factors = sorted(set(factor_ids) - active_main_set)
    if additional_needed > len(remaining_factors):
        raise ValueError("Not enough remaining factors to satisfy mimp.")

    extra_factors = rng.choice(np.array(remaining_factors), size=additional_needed, replace=False).tolist() if additional_needed else []
    important_factor_pool = sorted(active_main_set | set(extra_factors))
    interaction_table = design.effect_table[design.effect_table["effect_type"] == "interaction"].copy()
    interaction_table["factor_i"] = interaction_table["factor_ids"].map(lambda ids: ids[0])
    interaction_table["factor_j"] = interaction_table["factor_ids"].map(lambda ids: ids[1])

    mandatory: list[str] = []
    used_effects: set[str] = set()
    for factor_id in extra_factors:
        candidates = interaction_table[
            (
                (interaction_table["factor_i"] == factor_id)
                & (interaction_table["factor_j"].isin(active_main_set))
            )
            | (
                (interaction_table["factor_j"] == factor_id)
                & (interaction_table["factor_i"].isin(active_main_set))
            )
        ]["effect_id"].tolist()
        if not candidates:
            raise ValueError(f"No hereditary interaction is available to introduce factor '{factor_id}'.")
        available_candidates = [effect_id for effect_id in candidates if effect_id not in used_effects]
        choice_pool = available_candidates or candidates
        selected = str(rng.choice(np.array(choice_pool)))
        mandatory.append(selected)
        used_effects.add(selected)

    candidate_effects = interaction_table[
        interaction_table["factor_i"].isin(important_factor_pool)
        & interaction_table["factor_j"].isin(important_factor_pool)
        & (
            interaction_table["factor_i"].isin(active_main_set)
            | interaction_table["factor_j"].isin(active_main_set)
        )
    ]["effect_id"].tolist()
    candidate_effects = [effect_id for effect_id in candidate_effects if effect_id not in used_effects]

    remaining_needed = mint - len(mandatory)
    if remaining_needed > len(candidate_effects):
        raise ValueError("Not enough interaction effects are available for the requested scenario.")

    additional_effects = (
        rng.choice(np.array(candidate_effects), size=remaining_needed, replace=False).tolist() if remaining_needed > 0 else []
    )
    selected_effects = mandatory + additional_effects
    important_factors = design.factors_for_effects(active_main_effects + selected_effects)
    if len(important_factors) != mimp:
        raise ValueError(
            f"Scenario construction failed: expected mimp={mimp}, obtained {len(important_factors)} important factors."
        )
    return selected_effects, important_factors


def _select_interaction_effects_paper_exact(
    design: MixedLevelDesign,
    important_factors: Sequence[str],
    active_main_factors: Sequence[str],
    mint: int,
    rng: np.random.Generator,
) -> list[str]:
    interaction_table = design.effect_table[design.effect_table["effect_type"] == "interaction"].copy()
    interaction_table["factor_i"] = interaction_table["factor_ids"].map(lambda ids: ids[0])
    interaction_table["factor_j"] = interaction_table["factor_ids"].map(lambda ids: ids[1])

    important_set = set(important_factors)
    active_set = set(active_main_factors)
    inactive_important = [factor_id for factor_id in important_factors if factor_id not in active_set]

    mandatory: list[str] = []
    used_effects: set[str] = set()
    for factor_id in inactive_important:
        candidates = interaction_table[
            (
                (interaction_table["factor_i"] == factor_id)
                & (interaction_table["factor_j"].isin(active_set))
            )
            | (
                (interaction_table["factor_j"] == factor_id)
                & (interaction_table["factor_i"].isin(active_set))
            )
        ]["effect_id"].tolist()
        if not candidates:
            raise ValueError(f"No hereditary interaction is available to introduce factor '{factor_id}'.")
        selected = str(rng.choice(np.array(candidates)))
        mandatory.append(selected)
        used_effects.add(selected)

    candidate_effects = interaction_table[
        interaction_table["factor_i"].isin(important_set)
        & interaction_table["factor_j"].isin(important_set)
    ]["effect_id"].tolist()
    candidate_effects = [effect_id for effect_id in candidate_effects if effect_id not in used_effects]

    remaining_needed = mint - len(mandatory)
    if remaining_needed < 0:
        raise ValueError("mint is too small to cover all non-main important factors.")
    if remaining_needed > len(candidate_effects):
        raise ValueError("Not enough interaction effects are available for the requested exact scenario.")

    additional_effects = (
        rng.choice(np.array(candidate_effects), size=remaining_needed, replace=False).tolist() if remaining_needed > 0 else []
    )
    return mandatory + additional_effects


def _sample_active_columns(
    columns: Sequence[str],
    rng: np.random.Generator,
    *,
    paper_exact: bool = False,
    binomialp: float = 0.5,
) -> list[str]:
    if not columns:
        return []
    if paper_exact:
        subset_size = 1 + int(rng.binomial(len(columns) - 1, binomialp)) if len(columns) > 1 else 1
    else:
        subset_size = int(rng.integers(1, len(columns) + 1))
    return rng.choice(np.array(columns), size=subset_size, replace=False).tolist()


def generate_synthetic_dataset(
    design: MixedLevelDesign,
    scenario: Mapping[str, Any],
    random_state: int | np.random.Generator | None = None,
) -> dict[str, Any]:
    """Generate one synthetic response according to the paper's hierarchy rules."""

    rng = _ensure_rng(random_state)
    m_factors = len(design.factor_ids)
    mmain = _resolve_relative_factor_count(scenario["mmain"], m_factors)
    mint = _resolve_relative_factor_count(scenario["mint"], m_factors)
    mimp = _resolve_relative_factor_count(scenario["mimp"], m_factors)
    main_effects = design.effect_table.loc[design.effect_table["effect_type"] == "main", "effect_id"].tolist()
    paper_exact = bool(scenario.get("paper_exact", False))
    binomialp = float(scenario.get("binomialp", 0.5))

    if mmain > len(main_effects):
        raise ValueError("mmain exceeds the number of available main effects.")

    if paper_exact:
        important_factors_ordered = rng.choice(np.array(design.factor_ids), size=mimp, replace=False).tolist() if mimp > 0 else []
        active_main_factors = important_factors_ordered[:mmain]
        active_main_effects = [design.factor_to_main_effect[factor_id] for factor_id in active_main_factors]
        active_interaction_effects = _select_interaction_effects_paper_exact(
            design=design,
            important_factors=important_factors_ordered,
            active_main_factors=active_main_factors,
            mint=mint,
            rng=rng,
        )
        important_factors = sorted(set(important_factors_ordered))
    else:
        active_main_effects = rng.choice(np.array(main_effects), size=mmain, replace=False).tolist() if mmain > 0 else []
        active_interaction_effects, important_factors = _select_interaction_effects_for_scenario(
            design=design,
            active_main_effects=active_main_effects,
            mint=mint,
            mimp=mimp,
            rng=rng,
        )

    active_effects = active_main_effects + active_interaction_effects
    true_active_columns: list[str] = []
    beta = pd.Series(0.0, index=design.X_full.columns, dtype=float)

    for effect_id in active_effects:
        columns = design.effect_to_columns[effect_id]
        subset = _sample_active_columns(columns, rng, paper_exact=paper_exact, binomialp=binomialp)
        subset_size = len(subset)
        true_active_columns.extend(subset)
        magnitudes = rng.normal(loc=5.0, scale=1.0, size=subset_size)
        signs = rng.choice(np.array([-1.0, 1.0]), size=subset_size, replace=True)
        beta.loc[subset] = magnitudes * signs

    epsilon = rng.normal(loc=0.0, scale=1.0, size=design.n_runs)
    y = design.X_full.to_numpy(dtype=float) @ beta.to_numpy(dtype=float) + epsilon
    y_centered = center_response(y)

    return {
        "y": y_centered,
        "beta": beta,
        "true_active_columns": sorted(set(true_active_columns)),
        "true_active_effects": sorted(active_effects),
        "true_important_factors": sorted(important_factors),
    }


def _coerce_tuning_grid(
    tuning_grid: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | pd.DataFrame,
    design: MixedLevelDesign | None = None,
) -> pd.DataFrame:
    if isinstance(tuning_grid, pd.DataFrame):
        grid = tuning_grid.copy()
    elif isinstance(tuning_grid, Mapping):
        keys = list(tuning_grid.keys())
        values = [list(tuning_grid[key]) for key in keys]
        rows = [dict(zip(keys, combo, strict=False)) for combo in product(*values)]
        grid = pd.DataFrame(rows)
    else:
        grid = pd.DataFrame(list(tuning_grid))

    if grid.empty:
        raise ValueError("tuning_grid must contain at least one configuration.")

    if "nint" in grid.columns and design is not None:
        grid["nint"] = grid["nint"].map(lambda value: _resolve_relative_count(value, design.n_runs))

    defaults = {
        "ntop": 20,
        "pkeep": 0.25,
        "nrep": 1000,
        "group_stepwise": True,
        "paper_exact_stepwise": False,
        "paper_exact_baseline": False,
        "stepwise_mode": "bidirectional",
        "p_enter": 0.15,
        "p_remove": 0.15,
        "final_stepwise": True,
        "analysis_stage": "gds_arm",
    }
    for key, value in defaults.items():
        if key not in grid.columns:
            grid[key] = value

    grid["analysis_stage"] = grid["analysis_stage"].astype(str).str.lower()
    invalid_analysis_stage = sorted(set(grid["analysis_stage"]) - {"gds_arm", "step_i"})
    if invalid_analysis_stage:
        raise ValueError(
            "analysis_stage must be one of {'gds_arm', 'step_i'}; "
            f"received {invalid_analysis_stage}"
        )

    if "tuning_id" not in grid.columns:
        grid["tuning_id"] = [
            f"cfg_{idx + 1}:nint={row.nint},ntop={row.ntop},pkeep={row.pkeep},nrep={row.nrep}"
            for idx, row in enumerate(grid.itertuples(index=False))
        ]

    return grid.reset_index(drop=True)


def _resolve_relative_factor_count(value: Any, m_factors: int) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        resolved = int(round(value * m_factors))
        return 1 if value > 0 and resolved == 0 else resolved
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*m\s*", value)
        if match:
            numeric = float(match.group(1))
            resolved = int(round(numeric * m_factors))
            return 1 if numeric > 0 and resolved == 0 else resolved
        return int(value)
    raise TypeError(f"Unsupported factor-count specification: {value!r}")


def _resolve_relative_count(value: Any, n_runs: int) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return int(round(value * n_runs))
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*n\s*", value)
        if match:
            return int(round(float(match.group(1)) * n_runs))
        return int(value)
    raise TypeError(f"Unsupported count specification: {value!r}")


def _factor_metrics(
    selected_factors: Sequence[str],
    true_important_factors: Sequence[str],
    all_factors: Sequence[str],
) -> dict[str, float]:
    selected_set = set(selected_factors)
    truth_set = set(true_important_factors)
    all_set = set(all_factors)
    tp = len(selected_set & truth_set)
    fp = len(selected_set - truth_set)
    mimp = len(truth_set)
    denom_neg = max(len(all_set) - mimp, 1)
    power = tp / max(mimp, 1)
    error = fp / denom_neg
    return {
        "TP": float(tp),
        "FP": float(fp),
        "power": float(power),
        "error": float(error),
        "score": float(power - error),
    }


def _read_positive_int_env(var_names: Sequence[str]) -> int | None:
    for var_name in var_names:
        raw_value = os.getenv(var_name)
        if raw_value is None:
            continue
        try:
            value = int(str(raw_value).strip())
        except ValueError:
            continue
        if value > 0:
            return value
    return None


def _available_cpu_count() -> int:
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except (AttributeError, NotImplementedError, OSError):
        pass

    env_cpu_count = _read_positive_int_env(
        (
            "SLURM_CPUS_PER_TASK",
            "PBS_NP",
            "NSLOTS",
        )
    )
    if env_cpu_count is not None:
        return env_cpu_count
    return max(os.cpu_count() or 1, 1)


def _resolve_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None:
        return 1
    if int(n_jobs) <= 0:
        return _available_cpu_count()
    return int(n_jobs)


def _scenario_id_string(scenario: Mapping[str, Any]) -> str:
    return str(
        scenario.get(
            "scenario_id",
            scenario.get("name", f"mmain={scenario['mmain']}_mint={scenario['mint']}_mimp={scenario['mimp']}"),
        )
    )


def _set_simulation_design_registry(designs: Mapping[str, MixedLevelDesign]) -> None:
    global _SIMULATION_WORKER_DESIGNS
    _SIMULATION_WORKER_DESIGNS = {str(design_id): design for design_id, design in designs.items()}


def _get_simulation_design(design_id: str) -> MixedLevelDesign:
    design = _SIMULATION_WORKER_DESIGNS.get(str(design_id))
    if design is None:
        raise KeyError(f"Simulation worker could not resolve design '{design_id}'.")
    return design


def _slurm_array_sharding() -> tuple[int | None, int | None]:
    task_count = _read_positive_int_env(("SLURM_ARRAY_TASK_COUNT",))
    task_id_raw = os.getenv("SLURM_ARRAY_TASK_ID")
    if task_count is None or task_id_raw is None:
        return None, None
    try:
        task_id = int(str(task_id_raw).strip())
    except ValueError:
        return None, None

    task_min_raw = os.getenv("SLURM_ARRAY_TASK_MIN")
    if task_min_raw is not None:
        try:
            task_id -= int(str(task_min_raw).strip())
        except ValueError:
            pass
    return task_count, task_id


def _resolve_task_sharding(
    task_shards: int | None,
    task_shard_index: int | None,
) -> tuple[int, int]:
    if task_shards is None and task_shard_index is None:
        slurm_shards, slurm_index = _slurm_array_sharding()
        if slurm_shards is None or slurm_index is None:
            return 1, 0
        task_shards = slurm_shards
        task_shard_index = slurm_index
    elif task_shards is None or task_shard_index is None:
        raise ValueError("task_shards and task_shard_index must either both be set or both be omitted.")

    resolved_task_shards = int(task_shards)
    resolved_task_shard_index = int(task_shard_index)
    if resolved_task_shards <= 0:
        raise ValueError("task_shards must be positive.")
    if not 0 <= resolved_task_shard_index < resolved_task_shards:
        raise ValueError(
            f"task_shard_index must lie in [0, task_shards). Received "
            f"{resolved_task_shard_index} for task_shards={resolved_task_shards}."
        )
    return resolved_task_shards, resolved_task_shard_index


def _selected_task_count(total_tasks: int, task_shards: int, task_shard_index: int) -> int:
    total_tasks = int(total_tasks)
    task_shards = int(task_shards)
    task_shard_index = int(task_shard_index)
    if total_tasks <= 0 or task_shard_index >= total_tasks:
        return 0
    return 1 + (total_tasks - 1 - task_shard_index) // task_shards


def _iter_task_shard(
    task_iter: Iterable[Mapping[str, Any]],
    *,
    task_shards: int,
    task_shard_index: int,
):
    if int(task_shards) <= 1:
        yield from task_iter
        return

    for task in task_iter:
        if int(task["task_id"]) % int(task_shards) == int(task_shard_index):
            yield task


def _iter_simulation_replication_tasks(
    design_work_items: Sequence[dict[str, Any]],
    scenario_list: Sequence[Mapping[str, Any]],
    *,
    reps_per_condition: int,
    store_arm_results: bool,
    solver_order: Sequence[str] | None,
    task_seed: int,
):
    task_rng = np.random.default_rng(int(task_seed))
    task_id = 0
    solver_order_tuple = tuple(solver_order) if solver_order is not None else None

    for work_item in design_work_items:
        design_id = str(work_item["design_id"])
        grid_records = work_item["grid_records"]
        design_delta_grid = work_item["design_delta_grid"]
        for scenario in scenario_list:
            for tuning in grid_records:
                for dataset_rep in range(1, int(reps_per_condition) + 1):
                    yield {
                        "task_id": task_id,
                        "design_id": design_id,
                        "scenario": dict(scenario),
                        "tuning": dict(tuning),
                        "dataset_rep": int(dataset_rep),
                        "design_delta_grid": design_delta_grid,
                        "store_arm_results": bool(store_arm_results),
                        "solver_order": solver_order_tuple,
                        "seed": int(task_rng.integers(0, 2**31 - 1)),
                    }
                    task_id += 1


def _run_simulation_replication(
    task: Mapping[str, Any],
) -> tuple[int, dict[str, Any], dict[str, Any], tuple[str, GDSARMResult] | None]:
    design = _get_simulation_design(str(task["design_id"]))
    scenario = task["scenario"]
    tuning = task["tuning"]
    dataset_rep = int(task["dataset_rep"])
    design_delta_grid = task["design_delta_grid"]
    store_arm_results = bool(task["store_arm_results"])
    solver_order = task["solver_order"]
    rng = np.random.default_rng(int(task["seed"]))

    scenario_id = _scenario_id_string(scenario)
    analysis_stage = str(tuning["analysis_stage"]).lower()
    dataset = generate_synthetic_dataset(design, scenario, random_state=int(rng.integers(0, 2**31 - 1)))
    if bool(tuning["paper_exact_baseline"]) and design_delta_grid is None:
        current_delta_grid = None
    else:
        current_delta_grid = (
            derive_delta_grid(design.X_full, dataset["y"]) if design_delta_grid is None else design_delta_grid
        )

    t0 = time.perf_counter()
    sampled_interaction_columns: tuple[str, ...] = tuple()
    artifact_item: tuple[str, GDSARMResult] | None = None

    if analysis_stage == "step_i":
        step_i_result = run_GDS_step_i(
            design.X_main,
            design.X_int,
            dataset["y"],
            metadata=design,
            nint=int(tuning["nint"]),
            delta_grid=current_delta_grid,
            random_state=int(rng.integers(0, 2**31 - 1)),
            solver_order=solver_order,
            paper_exact_baseline=bool(tuning["paper_exact_baseline"]),
        )
        selected_factors = step_i_result.gds_result.best_selected_factors
        selected_effects = step_i_result.gds_result.best_selected_effects
        selected_columns = step_i_result.gds_result.best_selected_columns
        bic_value = step_i_result.gds_result.best_bic
        sampled_interaction_columns = tuple(step_i_result.sampled_interaction_columns)
        artifact_key = None
    else:
        arm_result = run_GDS_ARM(
            design.X_main,
            design.X_int,
            dataset["y"],
            metadata=design,
            nrep=int(tuning["nrep"]),
            nint=int(tuning["nint"]),
            ntop=int(tuning["ntop"]),
            pkeep=float(tuning["pkeep"]),
            delta_grid=current_delta_grid,
            stepwise_mode=str(tuning["stepwise_mode"]),
            p_enter=float(tuning["p_enter"]),
            p_remove=float(tuning["p_remove"]),
            group_stepwise=bool(tuning["group_stepwise"]),
            paper_exact_stepwise=bool(tuning["paper_exact_stepwise"]),
            paper_exact_baseline=bool(tuning["paper_exact_baseline"]),
            random_state=int(rng.integers(0, 2**31 - 1)),
            final_stepwise=bool(tuning["final_stepwise"]),
            solver_order=solver_order,
            show_progress=False,
        )
        selected_factors = arm_result.final_selected_factors
        selected_effects = arm_result.final_selected_effects
        selected_columns = arm_result.final_selected_columns
        bic_value = arm_result.diagnostics["final_bic"]
        artifact_key = f"{design.design_id}|{scenario_id}|{tuning['tuning_id']}|rep={dataset_rep}"
        if store_arm_results:
            artifact_item = (artifact_key, arm_result)
    elapsed = time.perf_counter() - t0

    metrics = _factor_metrics(
        selected_factors=selected_factors,
        true_important_factors=dataset["true_important_factors"],
        all_factors=design.factor_ids,
    )

    replication_row = {
        "design_id": design.design_id,
        "scenario_id": scenario_id,
        "dataset_rep": dataset_rep,
        "tuning_id": tuning["tuning_id"],
        "analysis_stage": analysis_stage,
        "nrep": int(tuning["nrep"]),
        "nint": int(tuning["nint"]),
        "ntop": int(tuning["ntop"]),
        "pkeep": float(tuning["pkeep"]),
        "group_stepwise": bool(tuning["group_stepwise"]),
        "paper_exact_stepwise": bool(tuning["paper_exact_stepwise"]),
        "paper_exact_baseline": bool(tuning["paper_exact_baseline"]),
        "final_stepwise": bool(tuning["final_stepwise"]),
        "TP": metrics["TP"],
        "FP": metrics["FP"],
        "power": metrics["power"],
        "error": metrics["error"],
        "score": metrics["score"],
        "sampled_interaction_columns": sampled_interaction_columns,
        "selected_factors": tuple(selected_factors),
        "selected_effects": tuple(selected_effects),
        "selected_columns": tuple(selected_columns),
        "final_model_size": len(selected_columns),
        "BIC": bic_value,
        "timing_seconds": elapsed,
        "artifact_key": artifact_key if (store_arm_results and analysis_stage == "gds_arm") else None,
    }
    truth_row = {
        "design_id": design.design_id,
        "scenario_id": scenario_id,
        "dataset_rep": dataset_rep,
        "tuning_id": tuning["tuning_id"],
        "analysis_stage": analysis_stage,
        "true_important_factors": tuple(dataset["true_important_factors"]),
        "true_active_effects": tuple(dataset["true_active_effects"]),
        "true_active_columns": tuple(dataset["true_active_columns"]),
    }
    return int(task["task_id"]), replication_row, truth_row, artifact_item


def _append_simulation_result(
    result: tuple[int, dict[str, Any], dict[str, Any], tuple[str, GDSARMResult] | None],
    replication_items: list[tuple[int, dict[str, Any]]],
    truth_items: list[tuple[int, dict[str, Any]]],
    arm_artifacts: dict[str, GDSARMResult] | None,
) -> None:
    task_id, replication_row, truth_row, artifact_item = result
    replication_items.append((task_id, replication_row))
    truth_items.append((task_id, truth_row))
    if arm_artifacts is not None and artifact_item is not None:
        artifact_key, artifact = artifact_item
        arm_artifacts[artifact_key] = artifact


def _run_serial_simulation_tasks(
    task_iter: Iterable[Mapping[str, Any]],
    *,
    simulation_progress: Any,
    arm_artifacts: dict[str, GDSARMResult] | None,
) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
    replication_items: list[tuple[int, dict[str, Any]]] = []
    truth_items: list[tuple[int, dict[str, Any]]] = []
    for task in task_iter:
        _append_simulation_result(
            _run_simulation_replication(task),
            replication_items,
            truth_items,
            arm_artifacts,
        )
        simulation_progress.update(1)
    return replication_items, truth_items


def _run_parallel_simulation_tasks(
    task_iter: Iterable[Mapping[str, Any]],
    *,
    worker_count: int,
    design_registry: Mapping[str, MixedLevelDesign],
    simulation_progress: Any,
    arm_artifacts: dict[str, GDSARMResult] | None,
) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
    replication_items: list[tuple[int, dict[str, Any]]] = []
    truth_items: list[tuple[int, dict[str, Any]]] = []
    max_pending = max(int(worker_count) * 4, 1)
    pending: dict[Any, None] = {}
    mp_context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=int(worker_count),
        mp_context=mp_context,
        initializer=_set_simulation_design_registry,
        initargs=(design_registry,),
    ) as executor:
        task_iter = iter(task_iter)

        while len(pending) < max_pending:
            try:
                task = next(task_iter)
            except StopIteration:
                break
            pending[executor.submit(_run_simulation_replication, task)] = None

        while pending:
            done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future, None)
                _append_simulation_result(
                    future.result(),
                    replication_items,
                    truth_items,
                    arm_artifacts,
                )
                simulation_progress.update(1)
                try:
                    next_task = next(task_iter)
                except StopIteration:
                    continue
                pending[executor.submit(_run_simulation_replication, next_task)] = None

    return replication_items, truth_items


def _iter_reused_baseline_tasks(
    design_work_items: Sequence[dict[str, Any]],
    *,
    reps_per_condition: int,
    task_seed: int,
):
    task_rng = np.random.default_rng(int(task_seed))
    task_id = 0

    for work_item in design_work_items:
        design_id = str(work_item["design_id"])
        grid_records = work_item["grid_records"]
        design_delta_grid = work_item["design_delta_grid"]
        for scenario in work_item["scenarios"]:
            for dataset_rep in range(1, int(reps_per_condition) + 1):
                yield {
                    "task_id": int(task_id),
                    "design_id": design_id,
                    "scenario": dict(scenario),
                    "grid_records": [dict(record) for record in grid_records],
                    "design_delta_grid": design_delta_grid,
                    "dataset_rep": int(dataset_rep),
                    "seed": int(task_rng.integers(0, 2**31 - 1)),
                }
                task_id += 1


def _run_reused_baseline_replication(
    task: Mapping[str, Any],
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    design = _get_simulation_design(str(task["design_id"]))
    scenario = dict(task["scenario"])
    grid_records = [dict(record) for record in task["grid_records"]]
    design_delta_grid = task["design_delta_grid"]
    dataset_rep = int(task["dataset_rep"])
    rng = np.random.default_rng(int(task["seed"]))

    scenario_id = _scenario_id_string(scenario)
    dataset = generate_synthetic_dataset(
        design,
        scenario,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    paper_exact_baseline = all(bool(record["paper_exact_baseline"]) for record in grid_records)
    if paper_exact_baseline and design_delta_grid is None:
        current_delta_grid = None
    else:
        current_delta_grid = (
            derive_delta_grid(design.X_full, dataset["y"]) if design_delta_grid is None else design_delta_grid
        )

    path_cache: dict[int, pd.DataFrame] = {}
    for resolved_nint in sorted({int(record["nint"]) for record in grid_records}):
        max_nrep = max(int(record["nrep"]) for record in grid_records if int(record["nint"]) == resolved_nint)
        path_cache[resolved_nint] = _build_baseline_repetition_path(
            design,
            dataset["y"],
            nrep=max_nrep,
            nint=resolved_nint,
            delta_grid=current_delta_grid,
            random_state=int(rng.integers(0, 2**31 - 1)),
            solver_order=None,
            paper_exact_baseline=paper_exact_baseline,
        )

    replication_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    for tuning in grid_records:
        truth_rows.append(
            {
                "design_id": design.design_id,
                "scenario_id": scenario_id,
                "dataset_rep": dataset_rep,
                "tuning_id": tuning["tuning_id"],
                "analysis_stage": str(tuning["analysis_stage"]).lower(),
                "true_important_factors": tuple(dataset["true_important_factors"]),
                "true_active_effects": tuple(dataset["true_active_effects"]),
                "true_active_columns": tuple(dataset["true_active_columns"]),
            }
        )

        resolved_nint = int(tuning["nint"])
        repetition_results = path_cache[resolved_nint]
        used_nrep = min(max(int(tuning["nrep"]), 0), len(repetition_results))
        path_elapsed = (
            float(repetition_results.iloc[used_nrep - 1]["cumulative_timing_seconds"])
            if used_nrep > 0
            else 0.0
        )
        variant_t0 = time.perf_counter()
        baseline_result = _build_original_baseline_variant_result(
            repetition_results,
            design,
            dataset["y"],
            nrep=int(tuning["nrep"]),
            ntop=int(tuning["ntop"]),
            pkeep=float(tuning["pkeep"]),
            group_stepwise=bool(tuning["group_stepwise"]),
            paper_exact_stepwise=bool(tuning["paper_exact_stepwise"]),
            paper_exact_baseline=bool(tuning["paper_exact_baseline"]),
            final_stepwise=bool(tuning["final_stepwise"]),
            stepwise_mode=str(tuning["stepwise_mode"]),
            p_enter=float(tuning["p_enter"]),
            p_remove=float(tuning["p_remove"]),
        )
        variant_elapsed = time.perf_counter() - variant_t0
        diagnostics = baseline_result["diagnostics"]
        metrics = _factor_metrics(
            selected_factors=baseline_result["final_selected_factors"],
            true_important_factors=dataset["true_important_factors"],
            all_factors=design.factor_ids,
        )
        replication_rows.append(
            {
                "design_id": design.design_id,
                "scenario_id": scenario_id,
                "dataset_rep": dataset_rep,
                "tuning_id": tuning["tuning_id"],
                "analysis_stage": str(tuning["analysis_stage"]).lower(),
                "nrep": int(tuning["nrep"]),
                "nint": resolved_nint,
                "ntop": int(tuning["ntop"]),
                "pkeep": float(tuning["pkeep"]),
                "group_stepwise": bool(tuning["group_stepwise"]),
                "paper_exact_stepwise": bool(tuning["paper_exact_stepwise"]),
                "paper_exact_baseline": bool(tuning["paper_exact_baseline"]),
                "final_stepwise": bool(tuning["final_stepwise"]),
                "TP": metrics["TP"],
                "FP": metrics["FP"],
                "power": metrics["power"],
                "error": metrics["error"],
                "score": metrics["score"],
                "sampled_interaction_columns": tuple(),
                "selected_factors": tuple(baseline_result["final_selected_factors"]),
                "selected_effects": tuple(baseline_result["final_selected_effects"]),
                "selected_columns": tuple(baseline_result["final_selected_columns"]),
                "final_model_size": int(len(baseline_result["final_selected_columns"])),
                "BIC": float(diagnostics["final_bic"]),
                "timing_seconds": float(path_elapsed + variant_elapsed),
                "artifact_key": None,
            }
        )

    return int(task["task_id"]), replication_rows, truth_rows


def _append_reused_baseline_result(
    result: tuple[int, list[dict[str, Any]], list[dict[str, Any]]],
    replication_items: list[tuple[int, dict[str, Any]]],
    truth_items: list[tuple[int, dict[str, Any]]],
) -> None:
    task_id, replication_rows, truth_rows = result
    replication_items.extend((task_id, row) for row in replication_rows)
    truth_items.extend((task_id, row) for row in truth_rows)


def _run_serial_reused_baseline_tasks(
    task_iter: Iterable[Mapping[str, Any]],
    *,
    simulation_progress: Any,
) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
    replication_items: list[tuple[int, dict[str, Any]]] = []
    truth_items: list[tuple[int, dict[str, Any]]] = []
    for task in task_iter:
        _append_reused_baseline_result(
            _run_reused_baseline_replication(task),
            replication_items,
            truth_items,
        )
        simulation_progress.update(1)
    return replication_items, truth_items


def _run_parallel_reused_baseline_tasks(
    task_iter: Iterable[Mapping[str, Any]],
    *,
    worker_count: int,
    design_registry: Mapping[str, MixedLevelDesign],
    simulation_progress: Any,
) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
    replication_items: list[tuple[int, dict[str, Any]]] = []
    truth_items: list[tuple[int, dict[str, Any]]] = []
    max_pending = max(int(worker_count) * 4, 1)
    pending: dict[Any, None] = {}
    mp_context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=int(worker_count),
        mp_context=mp_context,
        initializer=_set_simulation_design_registry,
        initargs=(design_registry,),
    ) as executor:
        task_iter = iter(task_iter)

        while len(pending) < max_pending:
            try:
                task = next(task_iter)
            except StopIteration:
                break
            pending[executor.submit(_run_reused_baseline_replication, task)] = None

        while pending:
            done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future, None)
                _append_reused_baseline_result(
                    future.result(),
                    replication_items,
                    truth_items,
                )
                simulation_progress.update(1)
                try:
                    next_task = next(task_iter)
                except StopIteration:
                    continue
                pending[executor.submit(_run_reused_baseline_replication, next_task)] = None

    return replication_items, truth_items


def _assemble_simulation_study_result(
    *,
    replication_items: list[tuple[int, dict[str, Any]]],
    truth_items: list[tuple[int, dict[str, Any]]],
    tuning_frames: list[pd.DataFrame],
    design_list: Sequence[MixedLevelDesign],
    arm_artifacts: dict[str, GDSARMResult] | None,
) -> SimulationStudyResult:
    if replication_items:
        replication_results = pd.DataFrame(
            row for _, row in sorted(replication_items, key=lambda item: item[0])
        )
    else:
        replication_results = pd.DataFrame(columns=_SIMULATION_REPLICATION_COLUMNS)

    if truth_items:
        truth_records = pd.DataFrame(
            row for _, row in sorted(truth_items, key=lambda item: item[0])
        )
    else:
        truth_records = pd.DataFrame(columns=_SIMULATION_TRUTH_COLUMNS)

    tuning_grid_df = (
        pd.concat(tuning_frames, axis=0, ignore_index=True)
        if tuning_frames
        else pd.DataFrame()
    )

    if replication_results.empty:
        summary_results = pd.DataFrame(columns=_SIMULATION_SUMMARY_COLUMNS)
    else:
        summary_results = (
            replication_results.groupby(
                [
                    "design_id",
                    "scenario_id",
                    "tuning_id",
                    "analysis_stage",
                    "nrep",
                    "nint",
                    "ntop",
                    "pkeep",
                    "group_stepwise",
                    "paper_exact_stepwise",
                    "paper_exact_baseline",
                    "final_stepwise",
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
                mean_BIC=("BIC", "mean"),
                mean_timing_seconds=("timing_seconds", "mean"),
            )
            .reset_index()
            .fillna({"sd_power": 0.0, "sd_error": 0.0, "sd_score": 0.0})
        )

    return SimulationStudyResult(
        replication_results=replication_results,
        summary_results=summary_results,
        design_diagnostics=design_diagnostic_table(design_list),
        truth_records=truth_records,
        tuning_grid=tuning_grid_df,
        arm_artifacts=arm_artifacts,
    )


def _run_design_wise_reused_baseline_simulation(
    design_list: Sequence[MixedLevelDesign],
    scenario_builder: Callable[[MixedLevelDesign], Sequence[Mapping[str, Any]]],
    tuning_grid_builder: Callable[[MixedLevelDesign], Sequence[Mapping[str, Any]] | pd.DataFrame],
    *,
    reps_per_condition: int = 200,
    delta_grid: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
    random_state: int | np.random.Generator | None = None,
    show_progress: bool = True,
    n_jobs: int = -1,
    task_shards: int | None = None,
    task_shard_index: int | None = None,
) -> SimulationStudyResult:
    """Run paper baseline studies with one reusable repetition path per dataset."""

    rng = _ensure_rng(random_state)
    tuning_frames: list[pd.DataFrame] = []
    progress_total = 0
    design_work_items: list[dict[str, Any]] = []
    design_registry = {design.design_id: design for design in design_list}
    resolved_task_shards, resolved_task_shard_index = _resolve_task_sharding(task_shards, task_shard_index)

    for design in design_list:
        scenarios = list(scenario_builder(design))
        grid_df = _coerce_tuning_grid(tuning_grid_builder(design), design=design)
        if set(grid_df["analysis_stage"].astype(str).str.lower()) != {"gds_arm"}:
            raise ValueError("Reusable baseline simulation only supports gds_arm tuning grids.")
        progress_total += len(scenarios) * int(reps_per_condition)
        tuning_frames.append(grid_df.assign(design_id=design.design_id))
        if delta_grid is None:
            design_delta_grid = None
        elif isinstance(delta_grid, Mapping):
            design_delta_grid = np.asarray(delta_grid[design.design_id], dtype=float)
        else:
            design_delta_grid = np.asarray(delta_grid, dtype=float)
        design_work_items.append(
            {
                "design_id": design.design_id,
                "grid_records": grid_df.to_dict(orient="records"),
                "design_delta_grid": design_delta_grid,
                "scenarios": scenarios,
            }
        )

    selected_progress_total = _selected_task_count(
        progress_total,
        task_shards=resolved_task_shards,
        task_shard_index=resolved_task_shard_index,
    )
    simulation_progress = _make_progress(
        total=selected_progress_total,
        desc="Simulation study",
        disable=not show_progress,
        leave=True,
    )

    replication_items: list[tuple[int, dict[str, Any]]] = []
    truth_items: list[tuple[int, dict[str, Any]]] = []
    try:
        task_seed = int(rng.integers(0, 2**31 - 1))
        worker_count = min(_resolve_n_jobs(n_jobs), max(selected_progress_total, 1))
        _set_simulation_design_registry(design_registry)

        def _task_iter():
            return _iter_task_shard(
                _iter_reused_baseline_tasks(
                    design_work_items,
                    reps_per_condition=int(reps_per_condition),
                    task_seed=task_seed,
                ),
                task_shards=resolved_task_shards,
                task_shard_index=resolved_task_shard_index,
            )

        if selected_progress_total <= 0:
            replication_items = []
            truth_items = []
        elif worker_count <= 1 or selected_progress_total <= 1:
            replication_items, truth_items = _run_serial_reused_baseline_tasks(
                _task_iter(),
                simulation_progress=simulation_progress,
            )
        else:
            try:
                replication_items, truth_items = _run_parallel_reused_baseline_tasks(
                    _task_iter(),
                    worker_count=worker_count,
                    design_registry=design_registry,
                    simulation_progress=simulation_progress,
                )
            except (BrokenProcessPool, OSError, PermissionError) as exc:
                warnings.warn(
                    f"Parallel simulation execution failed ({exc!r}); falling back to serial execution.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                simulation_progress.close()
                simulation_progress = _make_progress(
                    total=selected_progress_total,
                    desc="Simulation study",
                    disable=not show_progress,
                    leave=True,
                )
                replication_items, truth_items = _run_serial_reused_baseline_tasks(
                    _task_iter(),
                    simulation_progress=simulation_progress,
                )
    finally:
        simulation_progress.close()

    return _assemble_simulation_study_result(
        replication_items=replication_items,
        truth_items=truth_items,
        tuning_frames=tuning_frames,
        design_list=design_list,
        arm_artifacts=None,
    )


def run_simulation_study(
    design_list: Sequence[MixedLevelDesign],
    scenario_list: Sequence[Mapping[str, Any]],
    tuning_grid: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | pd.DataFrame,
    reps_per_condition: int = 200,
    delta_grid: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
    random_state: int | np.random.Generator | None = None,
    store_arm_results: bool = False,
    solver_order: Sequence[str] | None = None,
    show_progress: bool = True,
    n_jobs: int = -1,
    task_shards: int | None = None,
    task_shard_index: int | None = None,
) -> SimulationStudyResult:
    """Run the paper-style simulation study over designs, scenarios, and tuning settings."""

    rng = _ensure_rng(random_state)
    arm_artifacts: dict[str, GDSARMResult] | None = {} if store_arm_results else None
    tuning_frames: list[pd.DataFrame] = []
    progress_total = 0
    design_work_items: list[dict[str, Any]] = []
    design_registry = {design.design_id: design for design in design_list}
    resolved_task_shards, resolved_task_shard_index = _resolve_task_sharding(task_shards, task_shard_index)

    for design in design_list:
        grid_df = _coerce_tuning_grid(tuning_grid, design=design)
        progress_total += len(scenario_list) * len(grid_df) * int(reps_per_condition)
        grid_records = grid_df.to_dict(orient="records")
        tuning_frames.append(grid_df.assign(design_id=design.design_id))
        if delta_grid is None:
            design_delta_grid = None
        elif isinstance(delta_grid, Mapping):
            design_delta_grid = np.asarray(delta_grid[design.design_id], dtype=float)
        else:
            design_delta_grid = np.asarray(delta_grid, dtype=float)

        design_work_items.append(
            {
                "design_id": design.design_id,
                "grid_records": grid_records,
                "design_delta_grid": design_delta_grid,
            }
        )
    selected_progress_total = _selected_task_count(
        progress_total,
        task_shards=resolved_task_shards,
        task_shard_index=resolved_task_shard_index,
    )
    simulation_progress = _make_progress(
        total=selected_progress_total,
        desc="Simulation study",
        disable=not show_progress,
        leave=True,
    )

    replication_items: list[tuple[int, dict[str, Any]]] = []
    truth_items: list[tuple[int, dict[str, Any]]] = []
    try:
        task_seed = int(rng.integers(0, 2**31 - 1))
        worker_count = min(_resolve_n_jobs(n_jobs), max(selected_progress_total, 1))
        _set_simulation_design_registry(design_registry)

        def _task_iter():
            return _iter_task_shard(
                _iter_simulation_replication_tasks(
                    design_work_items,
                    scenario_list,
                    reps_per_condition=int(reps_per_condition),
                    store_arm_results=bool(store_arm_results),
                    solver_order=solver_order,
                    task_seed=task_seed,
                ),
                task_shards=resolved_task_shards,
                task_shard_index=resolved_task_shard_index,
            )

        if selected_progress_total <= 0:
            replication_items = []
            truth_items = []
        elif worker_count <= 1 or selected_progress_total <= 1:
            replication_items, truth_items = _run_serial_simulation_tasks(
                _task_iter(),
                simulation_progress=simulation_progress,
                arm_artifacts=arm_artifacts,
            )
        else:
            try:
                replication_items, truth_items = _run_parallel_simulation_tasks(
                    _task_iter(),
                    worker_count=worker_count,
                    design_registry=design_registry,
                    simulation_progress=simulation_progress,
                    arm_artifacts=arm_artifacts,
                )
            except (BrokenProcessPool, OSError, PermissionError) as exc:
                warnings.warn(
                    f"Parallel simulation execution failed ({exc!r}); falling back to serial execution.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                if arm_artifacts is not None:
                    arm_artifacts.clear()
                simulation_progress.close()
                simulation_progress = _make_progress(
                    total=selected_progress_total,
                    desc="Simulation study",
                    disable=not show_progress,
                    leave=True,
                )
                replication_items, truth_items = _run_serial_simulation_tasks(
                    _task_iter(),
                    simulation_progress=simulation_progress,
                    arm_artifacts=arm_artifacts,
                )
    finally:
        simulation_progress.close()

    return _assemble_simulation_study_result(
        replication_items=replication_items,
        truth_items=truth_items,
        tuning_frames=tuning_frames,
        design_list=design_list,
        arm_artifacts=arm_artifacts,
    )


def aggregate_ntop_pkeep_performance(summary_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate condition-level summaries into the paper's mean-vs-sd scatter inputs."""

    return (
        summary_results.groupby(["ntop", "pkeep"], dropna=False)
        .agg(
            mean_score=("mean_score", "mean"),
            sd_score=("mean_score", "std"),
            mean_power=("mean_power", "mean"),
            sd_power=("mean_power", "std"),
            n_conditions=("tuning_id", "count"),
        )
        .reset_index()
        .fillna({"sd_score": 0.0, "sd_power": 0.0})
        .sort_values(["pkeep", "ntop"])
    )


def boxplot_nrep_data(summary_results: pd.DataFrame) -> pd.DataFrame:
    """Return condition-level summary values used for nrep boxplots."""

    return summary_results[["design_id", "scenario_id", "nrep", "mean_score", "mean_power"]].copy()


def nint_curve_data(summary_results: pd.DataFrame, n_lookup: Mapping[str, int] | None = None) -> pd.DataFrame:
    data = summary_results.copy()
    if n_lookup is None:
        if "n_runs" in data.columns:
            n_lookup = data.groupby("design_id")["n_runs"].first().to_dict()
        else:
            raise ValueError("n_lookup must be supplied unless summary_results already contains an 'n_runs' column.")
    data["n_runs"] = data["design_id"].map(n_lookup)
    if data["n_runs"].isna().any():
        raise ValueError("n_lookup must provide n_runs for every design_id in summary_results.")
    data["nint_ratio"] = data["nint"] / data["n_runs"]
    data["scenario_label"] = data["scenario_id"].map(_paper_scenario_label)
    data["design_label"] = data["design_id"].map(_paper_design_label)
    return data


def _paper_scenario_label(value: Any) -> str:
    try:
        scenario_int = int(value)
    except (TypeError, ValueError):
        return str(value)
    scenario = _PAPER_SCENARIO_CATALOG.get(scenario_int)
    if scenario is None:
        return str(value)
    return f"{scenario_int} ({scenario['mmain']}, {scenario['mint']})"


def _paper_design_label(value: Any) -> str:
    match = re.fullmatch(r"design(\d+)_.*", str(value))
    if match:
        return f"Design {int(match.group(1))}"
    return str(value)


def plot_score_vs_nint_plotnine(data: pd.DataFrame, design_id: str | None = None):
    from plotnine import (
        aes,
        annotate,
        element_blank,
        element_rect,
        element_text,
        geom_line,
        geom_point,
        geom_vline,
        ggplot,
        labs,
        scale_color_manual,
        theme,
        theme_minimal,
    )

    palette = [_PALETTE["teal"], _PALETTE["copper"], _PALETTE["sea"], _PALETTE["moss"], _PALETTE["brick"]]
    plot_data = data.copy()
    if design_id is not None:
        plot_data = plot_data[plot_data["design_id"] == design_id].copy()
    if plot_data.empty:
        raise ValueError("No rows are available for the requested nint plot.")
    plot_data["design_scenario"] = plot_data["design_id"].astype(str) + " | " + plot_data["scenario_label"].astype(str)
    return (
        ggplot(plot_data, aes("nint_ratio", "mean_score", color="design_label", group="design_scenario"))
        + geom_line(size=1.2)
        + geom_point(size=2.8)
        + geom_vline(xintercept=2.0, linetype="dashed", color=_PALETTE["brick"], size=0.8)
        + annotate(
            "text",
            x=2.06,
            y=float(plot_data["mean_score"].max()),
            label="recommended 2n",
            color=_PALETTE["brick"],
            ha="left",
        )
        + scale_color_manual(values=palette)
        + labs(
            title="Mean Score of GDS Step (i) vs Interaction Budget",
            subtitle="Paper-style nint study",
            x="nint / n",
            y="Mean score (power - error)",
            color="Design",
        )
        + theme_minimal()
        + theme(
            figure_size=(11, 6.5),
            plot_background=element_rect(fill=_PALETTE["sand"], color=_PALETTE["sand"]),
            panel_background=element_rect(fill=_PALETTE["cream"], color=_PALETTE["cream"]),
            panel_grid_minor=element_blank(),
            legend_background=element_rect(fill=_PALETTE["sand"], color=_PALETTE["sand"]),
            text=element_text(color=_PALETTE["ink"]),
        )
    )


def plot_ntop_pkeep_scatter_plotnine(data: pd.DataFrame, metric: str = "score"):
    from plotnine import (
        aes,
        element_blank,
        element_rect,
        element_text,
        geom_point,
        ggplot,
        labs,
        scale_color_gradient,
        scale_size_continuous,
        theme,
        theme_minimal,
    )

    mean_col = "mean_score" if metric == "score" else "mean_power"
    sd_col = "sd_score" if metric == "score" else "sd_power"
    plot_data = data.copy()
    return (
        ggplot(plot_data, aes(sd_col, mean_col, color="pkeep", size="ntop"))
        + geom_point(alpha=0.88)
        + scale_color_gradient(low=_PALETTE["teal"], high=_PALETTE["gold"])
        + scale_size_continuous(range=(3, 10))
        + labs(
            title=f"Mean vs SD for (ntop, pkeep): {metric}",
            subtitle="Color encodes pkeep and point size encodes ntop.",
            x=f"SD of mean {metric}",
            y=f"Mean {metric}",
            color="pkeep",
            size="ntop",
        )
        + theme_minimal()
        + theme(
            figure_size=(10.5, 6.5),
            plot_background=element_rect(fill=_PALETTE["sand"], color=_PALETTE["sand"]),
            panel_background=element_rect(fill=_PALETTE["cream"], color=_PALETTE["cream"]),
            panel_grid_minor=element_blank(),
            text=element_text(color=_PALETTE["ink"]),
        )
    )


def plot_nrep_boxplot_plotnine(data: pd.DataFrame, metric: str = "score"):
    from plotnine import (
        aes,
        element_blank,
        element_rect,
        element_text,
        geom_boxplot,
        geom_point,
        ggplot,
        labs,
        stat_summary,
        theme,
        theme_minimal,
    )

    metric_col = "mean_score" if metric == "score" else "mean_power"
    return (
        ggplot(data, aes("factor(nrep)", metric_col, fill="factor(nrep)"))
        + geom_boxplot(alpha=0.8)
        + stat_summary(fun_y=np.mean, geom="point", color=_PALETTE["brick"], size=3.2)
        + labs(
            title=f"{metric.title()} Stability vs nrep",
            x="nrep",
            y=f"Condition-level mean {metric}",
        )
        + theme_minimal()
        + theme(
            figure_size=(10.5, 6.5),
            plot_background=element_rect(fill=_PALETTE["sand"], color=_PALETTE["sand"]),
            panel_background=element_rect(fill=_PALETTE["cream"], color=_PALETTE["cream"]),
            panel_grid_minor=element_blank(),
            text=element_text(color=_PALETTE["ink"]),
            legend_position="none",
        )
    )


def paper_scenario(scenario_id: int | str) -> dict[str, int]:
    """Return an exact scenario definition from the paper supplement when available."""

    scenario_int = int(scenario_id)
    if scenario_int not in _PAPER_SCENARIO_CATALOG:
        raise KeyError(
            f"Scenario {scenario_int} is not available in the exact supplement-backed catalog. "
            "The supplement files in this workspace identify scenarios "
            f"{sorted(_PAPER_SCENARIO_CATALOG)}."
        )
    scenario = _PAPER_SCENARIO_CATALOG[scenario_int].copy()
    scenario["scenario_id"] = scenario_int
    scenario["paper_exact"] = True
    scenario["binomialp"] = 0.5
    return scenario


def paper_scenarios_for_design(
    design: MixedLevelDesign,
    scenario_ids: Sequence[int | str] | None = None,
) -> list[dict[str, int]]:
    """Return the exact supplement-backed scenarios known for a given design."""

    if scenario_ids is None:
        scenario_ids = _PAPER_DESIGN_SCENARIO_MAP.get(design.design_id)
        if scenario_ids is None:
            raise ValueError(
                f"No exact paper scenario mapping is available for design '{design.design_id}'. "
                f"Known design mappings: {sorted(_PAPER_DESIGN_SCENARIO_MAP)}"
            )
    return [paper_scenario(scenario_id) for scenario_id in scenario_ids]


def _paper_baseline_tuning(
    tuning_id: str,
    *,
    nint: int | str,
    ntop: int,
    pkeep: float,
    nrep: int,
    analysis_stage: str = "gds_arm",
    final_stepwise: bool = True,
    paper_exact_stepwise: bool = True,
) -> dict[str, Any]:
    return {
        "tuning_id": tuning_id,
        "nint": nint,
        "ntop": ntop,
        "pkeep": pkeep,
        "nrep": nrep,
        "analysis_stage": analysis_stage,
        "final_stepwise": final_stepwise,
        "group_stepwise": False,
        "paper_exact_stepwise": paper_exact_stepwise,
        "paper_exact_baseline": True,
        "p_enter": 0.01,
        "p_remove": 0.05,
    }


def paper_nint_tuning_grid(
    design: MixedLevelDesign,
    multipliers: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    if multipliers is None:
        multipliers = _DEFAULT_NINT_MULTIPLIERS_BY_DESIGN.get(design.design_id)
    if multipliers is None:
        raise ValueError(
            f"No default nint multiplier grid is available for design '{design.design_id}'. "
            f"Known design mappings: {sorted(_DEFAULT_NINT_MULTIPLIERS_BY_DESIGN)}"
        )
    return [
        _paper_baseline_tuning(
            f"nint={mult}n",
            nint=f"{mult}n",
            ntop=1,
            pkeep=1.0,
            nrep=1,
            analysis_stage="step_i",
            final_stepwise=False,
            paper_exact_stepwise=False,
        )
        for mult in multipliers
    ]


def paper_ntop_pkeep_tuning_grid(
    ntop_values: Sequence[int] = _DEFAULT_NTOP_VALUES,
    pkeep_values: Sequence[float] = _DEFAULT_PKEEP_VALUES,
    *,
    nrep: int = 1000,
    nint: int | str = "2n",
) -> list[dict[str, Any]]:
    return [
        _paper_baseline_tuning(
            f"ntop={ntop},pkeep={pkeep}",
            nint=nint,
            ntop=ntop,
            pkeep=pkeep,
            nrep=nrep,
        )
        for ntop in ntop_values
        for pkeep in pkeep_values
    ]


def paper_nrep_tuning_grid(
    nrep_values: Sequence[int] = _DEFAULT_NREP_VALUES,
    *,
    ntop: int = 20,
    pkeep: float = 0.25,
    nint: int | str = "2n",
) -> list[dict[str, Any]]:
    return [
        _paper_baseline_tuning(
            f"nrep={nrep}",
            nint=nint,
            ntop=ntop,
            pkeep=pkeep,
            nrep=nrep,
        )
        for nrep in nrep_values
    ]


def _merge_simulation_results(
    results: Sequence[SimulationStudyResult],
    design_list: Sequence[MixedLevelDesign],
) -> SimulationStudyResult:
    arm_artifacts: dict[str, GDSARMResult] | None = {}
    for result in results:
        if result.arm_artifacts:
            arm_artifacts.update(result.arm_artifacts)
    if not arm_artifacts:
        arm_artifacts = None

    return SimulationStudyResult(
        replication_results=pd.concat([result.replication_results for result in results], ignore_index=True),
        summary_results=pd.concat([result.summary_results for result in results], ignore_index=True),
        design_diagnostics=design_diagnostic_table(design_list),
        truth_records=pd.concat([result.truth_records for result in results], ignore_index=True),
        tuning_grid=pd.concat([result.tuning_grid for result in results], ignore_index=True),
        arm_artifacts=arm_artifacts,
    )


def _run_design_wise_simulation(
    design_list: Sequence[MixedLevelDesign],
    scenario_builder: Callable[[MixedLevelDesign], Sequence[Mapping[str, Any]]],
    tuning_grid_builder: Callable[[MixedLevelDesign], Sequence[Mapping[str, Any]] | pd.DataFrame],
    *,
    reps_per_condition: int = 200,
    delta_grid: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
    random_state: int | np.random.Generator | None = None,
    store_arm_results: bool = False,
    solver_order: Sequence[str] | None = None,
    show_progress: bool = True,
    n_jobs: int = -1,
    task_shards: int | None = None,
    task_shard_index: int | None = None,
) -> SimulationStudyResult:
    rng = _ensure_rng(random_state)
    study_results: list[SimulationStudyResult] = []

    for design in design_list:
        scenarios = scenario_builder(design)
        tuning_grid = tuning_grid_builder(design)
        if isinstance(delta_grid, Mapping):
            design_delta_grid = delta_grid.get(design.design_id)
        else:
            design_delta_grid = delta_grid

        study_results.append(
            run_simulation_study(
                design_list=[design],
                scenario_list=scenarios,
                tuning_grid=tuning_grid,
                reps_per_condition=reps_per_condition,
                delta_grid=design_delta_grid,
                random_state=int(rng.integers(0, 2**31 - 1)),
                store_arm_results=store_arm_results,
                solver_order=solver_order,
                show_progress=show_progress,
                n_jobs=n_jobs,
                task_shards=task_shards,
                task_shard_index=task_shard_index,
            )
        )

    return _merge_simulation_results(study_results, design_list)


def _save_plotnine_plot(plot_obj: Any, target: Path, *, width: float = 11, height: float = 6.5, dpi: int = 200) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    plot_obj.save(filename=str(target), width=width, height=height, dpi=dpi, verbose=False)
    return target


def _write_study_outputs(
    *,
    study_key: str,
    study_result: SimulationStudyResult,
    summary_df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    results: dict[str, Any],
    summary_filename: str,
    write_replication_tables: bool,
    replication_filename: str | None = None,
    extra_tables: Sequence[tuple[pd.DataFrame, str]] = (),
    plot_jobs: Sequence[tuple[Any, str]] = (),
) -> None:
    summary_df.to_csv(tables_dir / summary_filename, index=False)
    if write_replication_tables and replication_filename is not None:
        study_result.replication_results.to_csv(tables_dir / replication_filename, index=False)
    for table_df, filename in extra_tables:
        table_df.to_csv(tables_dir / filename, index=False)
    for plot_obj, filename in plot_jobs:
        _save_plotnine_plot(plot_obj, plots_dir / filename)

    results[f"{study_key}_study"] = study_result
    results[f"{study_key}_summary"] = summary_df


def _source_sha256(path: Path | None = None) -> str:
    if path is not None:
        source_path = Path(path)
    else:
        module_file = globals().get("__file__")
        if module_file is not None:
            source_path = Path(module_file)
        else:
            source_path = Path.cwd() / "mixed_gds_arm.py"
            if not source_path.exists():
                raise RuntimeError(
                    "_source_sha256() could not resolve the source file in this notebook/runtime. "
                    "Run the notebook from the project directory or pass an explicit path."
                )
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


def _settings_payload(settings: StudyRunSettings) -> dict[str, Any]:
    payload = asdict(settings)
    payload["output_dir"] = str(settings.output_dir)
    return payload


def _run_manifest(settings: StudyRunSettings) -> dict[str, Any]:
    manifest = _settings_payload(settings)
    manifest["source_sha256"] = _source_sha256()
    manifest["cache_version"] = 1
    return json.loads(json.dumps(manifest, sort_keys=True))


def _cache_file_path(output_path: Path) -> Path:
    return output_path / "study_results.pkl.gz"


def _load_study_cache(cache_path: Path) -> dict[str, Any]:
    with gzip.open(cache_path, "rb") as handle:
        return pickle.load(handle)


def _save_study_cache(cache_path: Path, results: dict[str, Any]) -> None:
    with gzip.open(cache_path, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_study_results(
    output_dir: str | os.PathLike[str],
    *,
    require_current_source: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    manifest_path = output_path / "run_settings.json"
    cache_path = _cache_file_path(output_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Study manifest not found: {manifest_path}")
    if not cache_path.exists():
        raise FileNotFoundError(f"Study cache not found: {cache_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_current_source and manifest.get("source_sha256") != _source_sha256():
        raise ValueError(
            "Saved results were produced by a different version of mixed_gds_arm.py. "
            "Set require_current_source=False to load them anyway."
        )
    return _load_study_cache(cache_path)


def run_study(
    settings: StudyRunSettings | None = None,
    /,
    **overrides: Any,
) -> dict[str, Any]:
    """Run the configurable mixed-level GDS-ARM study specified by settings."""

    if settings is None:
        settings = StudyRunSettings(**overrides)
    elif overrides:
        settings = replace(settings, **overrides)
    resolved_task_shards, resolved_task_shard_index = _resolve_task_sharding(
        settings.task_shards,
        settings.task_shard_index,
    )
    settings = replace(
        settings,
        task_shards=resolved_task_shards,
        task_shard_index=resolved_task_shard_index,
    )

    output_path = Path(settings.output_dir)
    tables_dir = output_path / "tables"
    plots_dir = output_path / "plots"
    manifest = _run_manifest(settings)
    manifest_path = output_path / "run_settings.json"
    cache_path = _cache_file_path(output_path)

    if settings.reuse_saved_results and not settings.force_rerun and manifest_path.exists() and cache_path.exists():
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved_manifest == manifest:
            print(f"Loaded cached study results from: {cache_path}")
            return _load_study_cache(cache_path)

    tables_dir.mkdir(parents=True, exist_ok=True)
    if settings.write_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    if settings.write_settings_file or settings.write_cache_file or settings.reuse_saved_results:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    rng = np.random.default_rng(settings.random_state)
    selected_studies = tuple(dict.fromkeys(settings.studies))
    invalid_studies = sorted(set(selected_studies) - {"nint", "ntop_pkeep", "nrep"})
    if invalid_studies:
        raise ValueError(f"Unsupported study names: {invalid_studies}. Expected any of ('nint', 'ntop_pkeep', 'nrep').")

    simulation_design_ids = tuple(settings.simulation_design_ids)
    nint_design_ids = tuple(settings.nint_design_ids)
    requested_design_ids = tuple(dict.fromkeys([*simulation_design_ids, *nint_design_ids]))
    known_design_ids = set(_GENERATED_DESIGN_SPECS)
    missing_simulation_designs = sorted(set(simulation_design_ids) - known_design_ids)
    missing_nint_designs = sorted(set(nint_design_ids) - known_design_ids)
    if missing_simulation_designs:
        raise ValueError(
            f"Unknown generated simulation_design_ids: {missing_simulation_designs}. "
            f"Known generated templates: {sorted(known_design_ids)}"
        )
    if missing_nint_designs:
        raise ValueError(
            f"Unknown generated nint_design_ids: {missing_nint_designs}. "
            f"Known generated templates: {sorted(known_design_ids)}"
        )
    design_suite = generate_benchmark_design_suite(
        requested_design_ids,
        random_state=int(rng.integers(0, 2**31 - 1)),
        coding=settings.generated_design_coding,
        candidate_pool_size=settings.generated_design_candidate_pool,
        n_starts=settings.generated_design_restarts,
    )

    by_id = design_suite["by_id"]
    missing_simulation_designs = sorted(set(simulation_design_ids) - set(by_id))
    missing_nint_designs = sorted(set(nint_design_ids) - set(by_id))
    if missing_simulation_designs:
        raise ValueError(f"Unknown simulation_design_ids: {missing_simulation_designs}")
    if missing_nint_designs:
        raise ValueError(f"Unknown nint_design_ids: {missing_nint_designs}")

    simulation_designs = [by_id[design_id] for design_id in simulation_design_ids]
    nint_designs = [by_id[design_id] for design_id in nint_design_ids]
    if any(study in selected_studies for study in ("ntop_pkeep", "nrep")) and not simulation_designs:
        raise ValueError("At least one simulation design is required for the selected studies.")
    if "nint" in selected_studies and not nint_designs:
        raise ValueError("At least one nint design is required when running the nint study.")

    diagnostic_designs: list[MixedLevelDesign] = []
    seen_design_ids: set[str] = set()
    for design in [*simulation_designs, *nint_designs]:
        if design.design_id not in seen_design_ids:
            diagnostic_designs.append(design)
            seen_design_ids.add(design.design_id)

    diagnostics = design_diagnostic_table(diagnostic_designs)
    diagnostics.to_csv(tables_dir / "design_diagnostics.csv", index=False)

    def _scenario_builder(design: MixedLevelDesign) -> list[dict[str, int]]:
        override_ids = None if settings.scenario_ids_by_design is None else settings.scenario_ids_by_design.get(design.design_id)
        return paper_scenarios_for_design(design, scenario_ids=override_ids)

    results: dict[str, Any] = {
        "output_dir": output_path,
        "tables_dir": tables_dir,
        "plots_dir": plots_dir,
        "settings": settings,
        "design_diagnostics": diagnostics,
        "studies": selected_studies,
    }

    if "nint" in selected_studies:
        nint_study = _run_design_wise_simulation(
            nint_designs,
            _scenario_builder,
            lambda design: paper_nint_tuning_grid(
                design,
                multipliers=None if settings.nint_multipliers_by_design is None else settings.nint_multipliers_by_design.get(design.design_id),
            ),
            reps_per_condition=settings.reps_per_condition,
            delta_grid=None,
            random_state=int(rng.integers(0, 2**31 - 1)),
            store_arm_results=False,
            show_progress=settings.show_progress,
            n_jobs=settings.n_jobs,
            task_shards=settings.task_shards,
            task_shard_index=settings.task_shard_index,
        )
        nint_summary_df = nint_study.summary_results.copy()
        n_lookup = {design.design_id: design.n_runs for design in nint_designs}
        nint_plot_data = nint_curve_data(nint_summary_df, n_lookup=n_lookup)
        _write_study_outputs(
            study_key="nint",
            study_result=nint_study,
            summary_df=nint_summary_df,
            tables_dir=tables_dir,
            plots_dir=plots_dir,
            results=results,
            summary_filename="nint_summary_results.csv",
            write_replication_tables=settings.write_replication_tables,
            replication_filename="nint_replication_results.csv",
            plot_jobs=[
                (plot_score_vs_nint_plotnine(nint_plot_data), "score_vs_nint_plotnine.png"),
            ] if settings.write_plots else [],
        )

    if "ntop_pkeep" in selected_studies:
        ntop_pkeep_study = _run_design_wise_reused_baseline_simulation(
            simulation_designs,
            _scenario_builder,
            lambda _design: paper_ntop_pkeep_tuning_grid(
                ntop_values=settings.ntop_values,
                pkeep_values=settings.pkeep_values,
                nrep=settings.ntop_pkeep_nrep,
                nint=settings.ntop_pkeep_nint,
            ),
            reps_per_condition=settings.reps_per_condition,
            delta_grid=None,
            random_state=int(rng.integers(0, 2**31 - 1)),
            show_progress=settings.show_progress,
            n_jobs=settings.n_jobs,
            task_shards=settings.task_shards,
            task_shard_index=settings.task_shard_index,
        )
        ntop_pkeep_summary_df = ntop_pkeep_study.summary_results.copy()
        ntop_pkeep_plot_data = aggregate_ntop_pkeep_performance(ntop_pkeep_summary_df)
        _write_study_outputs(
            study_key="ntop_pkeep",
            study_result=ntop_pkeep_study,
            summary_df=ntop_pkeep_summary_df,
            tables_dir=tables_dir,
            plots_dir=plots_dir,
            results=results,
            summary_filename="ntop_pkeep_summary_results.csv",
            write_replication_tables=settings.write_replication_tables,
            replication_filename="ntop_pkeep_replication_results.csv",
            extra_tables=[(ntop_pkeep_plot_data, "ntop_pkeep_aggregate.csv")],
            plot_jobs=[
                (plot_ntop_pkeep_scatter_plotnine(ntop_pkeep_plot_data, metric="score"), "ntop_pkeep_score_plotnine.png"),
                (plot_ntop_pkeep_scatter_plotnine(ntop_pkeep_plot_data, metric="power"), "ntop_pkeep_power_plotnine.png"),
            ] if settings.write_plots else [],
        )

    if "nrep" in selected_studies:
        nrep_study = _run_design_wise_reused_baseline_simulation(
            simulation_designs,
            _scenario_builder,
            lambda _design: paper_nrep_tuning_grid(
                nrep_values=settings.nrep_values,
                ntop=settings.nrep_study_ntop,
                pkeep=settings.nrep_study_pkeep,
                nint=settings.nrep_study_nint,
            ),
            reps_per_condition=settings.reps_per_condition,
            delta_grid=None,
            random_state=int(rng.integers(0, 2**31 - 1)),
            show_progress=settings.show_progress,
            n_jobs=settings.n_jobs,
            task_shards=settings.task_shards,
            task_shard_index=settings.task_shard_index,
        )
        nrep_summary_df = nrep_study.summary_results.copy()
        nrep_plot_data = boxplot_nrep_data(nrep_summary_df)
        _write_study_outputs(
            study_key="nrep",
            study_result=nrep_study,
            summary_df=nrep_summary_df,
            tables_dir=tables_dir,
            plots_dir=plots_dir,
            results=results,
            summary_filename="nrep_summary_results.csv",
            write_replication_tables=settings.write_replication_tables,
            replication_filename="nrep_replication_results.csv",
            plot_jobs=[
                (plot_nrep_boxplot_plotnine(nrep_plot_data, metric="score"), "nrep_score_plotnine.png"),
                (plot_nrep_boxplot_plotnine(nrep_plot_data, metric="power"), "nrep_power_plotnine.png"),
            ] if settings.write_plots else [],
        )

    print("\nDesign diagnostics")
    print(diagnostics.to_string(index=False))
    print(f"\nRun name: {settings.run_name}")
    print(f"\nSaved tables to: {tables_dir}")
    if settings.write_plots:
        print(f"Saved plots to:  {plots_dir}")
    if settings.write_settings_file:
        print(f"Saved settings to: {output_path / 'run_settings.json'}")
    print("The script uses generated mixed-level designs matched to the paper's n, m, and q_i templates.")
    print(f"Generated design coding: {settings.generated_design_coding}")
    print(f"Studies run: {', '.join(selected_studies) if selected_studies else 'none'}")
    print(f"Each tuning condition used reps_per_condition={settings.reps_per_condition}.")
    print(f"Parallel workers: { _resolve_n_jobs(settings.n_jobs) }")
    if int(settings.task_shards) > 1:
        print(
            f"Task shard: {int(settings.task_shard_index) + 1}/{int(settings.task_shards)} "
            f"(zero-based index {int(settings.task_shard_index)})."
        )

    if settings.write_cache_file:
        _save_study_cache(cache_path, results)

    return results


__all__ = [
    "MixedLevelDesign",
    "OLSRefitResult",
    "GDSResult",
    "GDSStepIResult",
    "StepwiseResult",
    "GDSARMResult",
    "SimulationStudyResult",
    "StudyRunSettings",
    "orthogonal_polynomial_contrasts",
    "paper_exact_polynomial_contrasts",
    "center_response",
    "build_mixed_level_design",
    "build_paper_exact_design",
    "derive_delta_grid",
    "paper_exact_delta_grid",
    "run_GDS",
    "run_GDS_step_i",
    "run_GDS_ARM",
    "design_diagnostic_table",
    "generate_mixed_level_factor_data",
    "generate_benchmark_design",
    "generate_benchmark_design_suite",
    "generate_synthetic_dataset",
    "run_simulation_study",
    "aggregate_ntop_pkeep_performance",
    "boxplot_nrep_data",
    "nint_curve_data",
    "plot_score_vs_nint_plotnine",
    "plot_ntop_pkeep_scatter_plotnine",
    "plot_nrep_boxplot_plotnine",
    "load_study_results",
    "paper_scenario",
    "paper_scenarios_for_design",
    "paper_nint_tuning_grid",
    "paper_ntop_pkeep_tuning_grid",
    "paper_nrep_tuning_grid",
    "run_study",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the configurable mixed-level GDS-ARM study and save results.")
    parser.add_argument("--output-dir", default="study_outputs", help="Directory where tables and graphs will be written.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for the analysis run.")
    parser.add_argument(
        "--reps-per-condition",
        type=int,
        default=StudyRunSettings().reps_per_condition,
        help="Monte Carlo repetitions per design-scenario-tuning condition.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel worker processes for simulation conditions. Use 0 or -1 for all CPUs.",
    )
    parser.add_argument(
        "--task-shards",
        type=int,
        default=None,
        help="Split the total Monte Carlo task stream across this many shard jobs. Defaults to the active SLURM array size when available.",
    )
    parser.add_argument(
        "--task-shard-index",
        type=int,
        default=None,
        help="Zero-based shard index for this run. Defaults to the active SLURM array index when available.",
    )
    if any(arg.startswith("--f=") for arg in sys.argv[1:]) or "ipykernel_launcher" in Path(sys.argv[0]).name:
        print("Detected Jupyter kernel arguments; skipping CLI entry point.")
    else:
        args, _unknown = parser.parse_known_args()
        run_study(
            StudyRunSettings(
                output_dir=args.output_dir,
                random_state=args.seed,
                reps_per_condition=args.reps_per_condition,
                n_jobs=args.n_jobs,
                task_shards=args.task_shards,
                task_shard_index=args.task_shard_index,
            )
        )
