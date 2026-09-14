# Adaptive Weighted Sequential GDS-ARM for Factor Screening

Reproducible Python implementation and simulation framework for **Adaptive Weighted Sequential Gauss–Dantzig Selector Aggregation over Random Models (AWS-GDS-ARM)** in mixed-level supersaturated designs.

The project starts from the mixed-level GDS-ARM screening procedure and studies three extensions:

1. **BIC-weighted aggregation** of the best random-submodel fits.
2. **Adaptive sequential stopping** when the weighted inclusion evidence stabilizes.
3. **Adaptive interaction sampling** that redirects search effort toward promising interaction blocks.

The repository contains the simulation engine, method-specific benchmark drivers, saved baseline results, adaptive-run settings, and tools for rebuilding publication figures after the corresponding summary tables have been generated.

> **Main empirical conclusion:** the original mixed-level GDS-ARM baseline gives the best overall power–error score in the common benchmark. BIC weighting increases power but also increases false selections. Adaptive stopping is the most useful extension because it reduces the mean repetition budget and runtime by about 63% while preserving most of the weighted method's selection behavior.

## Contents

- [Research objective](#research-objective)
- [Methods](#methods)
- [Results at a glance](#results-at-a-glance)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick validation](#quick-validation)
- [Using the saved baseline results](#using-the-saved-baseline-results)
- [Running the simulation studies](#running-the-simulation-studies)
- [Rebuilding figures](#rebuilding-figures)
- [Configuration and command-line options](#configuration-and-command-line-options)
- [Output files](#output-files)
- [Reproducibility notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [References and citation](#references-and-citation)

## Research objective

Supersaturated designs contain more candidate effects than experimental runs. The full regression model is therefore not estimable without a sparsity assumption, and the central task becomes **factor screening**: finding a small set of active factors while limiting false selections.

The problem is harder for mixed-level experiments. A factor with $q_i$ levels contributes $q_i-1$ contrast columns, while an interaction between factors $i$ and $j$ contributes $(q_i-1)(q_j-1)$ columns. The interaction-expanded design matrix can grow far beyond the available run size.

GDS-ARM addresses this problem by repeatedly fitting Gauss–Dantzig selector models to smaller random submodels. Each submodel contains all main-effect columns and a sampled subset of interaction columns. The best fitted models are then aggregated into a final factor-screening decision.

This project asks which parts of that workflow benefit from adaptivity:

- Should better-fitting models receive more aggregation weight?
- Can the repetition path stop once the inclusion evidence is stable?
- Can accumulated evidence guide which interaction blocks are sampled next?

## Methods

### 1. Original mixed-level GDS-ARM

For each simulated dataset, the baseline procedure:

1. Builds an orthogonal-polynomial representation of the mixed-level factors.
2. Includes every main-effect column in each random submodel.
3. Samples `nint` interaction columns uniformly.
4. Solves a Dantzig-selector path.
5. Uses two-cluster coefficient thresholding to identify an active set.
6. Refits the active variables by ordinary least squares.
7. Ranks candidate models by post-refit BIC.
8. Retains the best `ntop` models from `nrep` repetitions.
9. Keeps columns appearing in at least a `pkeep` fraction of those models.
10. Applies final bidirectional stepwise regression and maps selected columns back to effects and physical factors.

The four primary tuning controls are:

| Parameter | Meaning |
| --- | --- |
| `nint` | Number of interaction columns sampled in a random submodel |
| `ntop` | Number of best-BIC models retained for aggregation |
| `pkeep` | Minimum inclusion-frequency threshold |
| `nrep` | Number of random-submodel repetitions |

### 2. BIC-weighted aggregation

The weighted variant replaces equal model counts with normalized BIC evidence weights. For the top $T$ models,

$$
w_t = \frac{\exp\{-\tfrac12(\mathrm{BIC}_t-\mathrm{BIC}_{\min})\}}
{\sum_{s=1}^{T}\exp\{-\tfrac12(\mathrm{BIC}_s-\mathrm{BIC}_{\min})\}},
\qquad
\pi_c = \sum_{t=1}^{T} w_t\,\mathbf{1}\{c\in B_t\}.
$$

A column is retained when its weighted inclusion score $\pi_c$ reaches the threshold `weighted_tau`. When that option is omitted, the implementation uses `pkeep` as the threshold.

### 3. Adaptive sequential stopping

The adaptive-stopping variant processes repetitions in batches and monitors the maximum change in the weighted evidence vector,

$$
\Delta_b = \lVert \boldsymbol{\pi}^{(b)}-\boldsymbol{\pi}^{(b-1)}\rVert_\infty.
$$

Stopping becomes eligible after `adaptive_min_nrep` repetitions. It occurs when the change remains below `adaptive_epsilon` for `adaptive_patience` consecutive batches, unless `adaptive_max_nrep` is reached first. An optional Jaccard condition can also require stability of the selected set.

The reported configuration uses:

- batch size: `50`
- minimum repetitions: `100`
- maximum repetitions: `1000`
- score tolerance: `0.01`
- patience: `2` consecutive stable batches

### 4. Adaptive interaction sampling

The adaptive-sampling variant keeps a fixed repetition budget but changes the interaction proposal distribution. After uniform warm-up batches, it combines:

- accumulated evidence for an interaction block;
- heredity evidence from its two parent main effects;
- a uniform exploration component; and
- a positive probability floor that prevents permanent exclusion.

For interaction block $e=(a,b)$, the unnormalized proposal weight is

$$\widetilde q_e = q_{\min} + \frac{1-\lambda}{|\mathcal E_{\mathrm{int}}|} + \lambda\{(1-\eta)v_e + \eta u_a u_b\}.$$

The reported configuration uses `sampling_lambda = 0.7`, `sampling_eta = 0.5`, two warm-up batches, and `sampling_block_floor = 0.001`.

## Results at a glance

The common benchmark uses three generated mixed-level design templates, six sparse scenarios per design, and 200 Monte Carlo datasets per condition. Fixed-budget methods use `nint = 2n`, `ntop = 20`, `pkeep = 0.25`, and `nrep = 1000`.

The evaluation metrics are

$$
\mathrm{Power}=\frac{|\widehat T\cap T|}{|T|},\qquad
\mathrm{Error}=\frac{|\widehat T\setminus T|}{m-|T|},\qquad
\mathrm{Score}=\mathrm{Power}-\mathrm{Error}.
$$

| Method | Score | Power | Error | Mean runtime (s) | Mean model size | Mean repetitions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original GDS-ARM | **0.601** | 0.836 | **0.235** | 216.22 | **6.04** | 1000.00 |
| BIC-weighted aggregation | 0.566 | **0.872** | 0.307 | 216.21 | 7.43 | 1000.00 |
| Adaptive stopping | 0.561 | 0.864 | 0.303 | **79.93** | 7.27 | **368.81** |
| Adaptive interaction sampling | 0.552 | 0.865 | 0.312 | 220.77 | 7.28 | 1000.00 |

Interpretation:

- The **original baseline** provides the strongest overall balance of power and false-selection control.
- **Weighted aggregation** has the highest power, but its larger false-selection rate lowers the combined score.
- **Adaptive stopping** is the clear computational winner and uses approximately 63% fewer repetitions on average.
- **Adaptive sampling** produces some condition-specific gains but does not improve the overall benchmark.

Runtime values depend on hardware and process configuration. The relative comparison is more portable than the absolute number of seconds.

## Repository structure

```text
.
├── README.md
├── simulation_code/
│   ├── mixed_gds_arm.py
│   ├── aws_gds_arm.py
│   ├── aws_gds_arm_adaptive.py
│   ├── aws_gds_arm_adaptive_sampling.py
│   ├── aws_gds_arm_methodology_comparison.py
│   ├── aws_gds_arm_common.py
│   ├── reproduce_saved_figures.py
│   ├── requirements.txt
│   ├── environment_snapshot.json
│   └── Makefile
├── study_outputs/
│   ├── run_settings.json
│   └── study_results.pkl.gz
└── study_outputs_aws_adaptive/
    └── adaptive_run_settings.json
```

### Code modules

| Path | Purpose |
| --- | --- |
| [`simulation_code/mixed_gds_arm.py`](./simulation_code/mixed_gds_arm.py) | Mixed-level design construction, GDS/GDS-ARM implementation, baseline tuning studies, plots, caching, and task sharding |
| [`simulation_code/aws_gds_arm.py`](./simulation_code/aws_gds_arm.py) | BIC-weighted aggregation benchmark |
| [`simulation_code/aws_gds_arm_adaptive.py`](./simulation_code/aws_gds_arm_adaptive.py) | Weighted aggregation with adaptive repetition stopping |
| [`simulation_code/aws_gds_arm_adaptive_sampling.py`](./simulation_code/aws_gds_arm_adaptive_sampling.py) | Weighted aggregation with adaptive interaction-block sampling |
| [`simulation_code/aws_gds_arm_methodology_comparison.py`](./simulation_code/aws_gds_arm_methodology_comparison.py) | Unified four-method comparison under shared designs, scenarios, and tuning settings |
| [`simulation_code/aws_gds_arm_common.py`](./simulation_code/aws_gds_arm_common.py) | Shared schemas, summaries, labels, and plotting utilities |
| [`simulation_code/reproduce_saved_figures.py`](./simulation_code/reproduce_saved_figures.py) | Rebuilds figures from previously generated CSV summary tables without rerunning simulations |

### Included saved artifacts

| Path | Included content | Status |
| --- | --- | --- |
| [`study_outputs/run_settings.json`](./study_outputs/run_settings.json) | Canonical baseline configuration and source hash | Included |
| `study_outputs/study_results.pkl.gz` | Compressed baseline Python result cache | Included |
| [`study_outputs_aws_adaptive/adaptive_run_settings.json`](./study_outputs_aws_adaptive/adaptive_run_settings.json) | Canonical adaptive-stopping configuration and source hashes | Included |

The minimal repository snapshot does **not** include every CSV table and PNG directory produced by all five drivers. In particular, the adaptive folder currently contains its settings manifest but not the full adaptive result tables. The scripts generate those files during a fresh run. See [Rebuilding figures](#rebuilding-figures) before using `--study all`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/amansd78/Adaptive-Weighted-Sequential-GDS-ARM-for-Factor-Screening.git
cd Adaptive-Weighted-Sequential-GDS-ARM-for-Factor-Screening
```

### 2. Create an isolated environment

The exact verified package versions are pinned in [`simulation_code/requirements.txt`](./simulation_code/requirements.txt). The recorded reference environment is CPython 3.14.2 on macOS ARM64; see [`simulation_code/environment_snapshot.json`](./simulation_code/environment_snapshot.json).

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r simulation_code/requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r simulation_code\requirements.txt

# The current code defaults Matplotlib's cache to /tmp on Unix.
# Set a writable Windows path before importing or running the modules.
$env:MPLCONFIGDIR = Join-Path $PWD ".matplotlib"
```

The principal dependencies are NumPy, pandas, SciPy, scikit-learn, statsmodels, CVXPY, Matplotlib, Plotnine, and tqdm. CVXPY-compatible solvers pinned by the project include Clarabel, HiGHS, OSQP, and SCS.

### Make-based setup

On macOS or Linux with GNU Make:

```bash
cd simulation_code
make venv
make install
```

The bundled Makefile assumes the Unix virtual-environment path `.venv/bin/python`. Use the direct PowerShell commands above on Windows.

## Quick validation

The following checks load argument parsers but do not launch a simulation:

```bash
python simulation_code/mixed_gds_arm.py --help
python simulation_code/aws_gds_arm_adaptive.py --help
python simulation_code/aws_gds_arm_methodology_comparison.py --help
```

To check Python syntax across the code bundle:

```bash
python -m compileall simulation_code
```

There is currently no separate automated test suite. The primary validation is successful execution of the benchmark drivers and agreement of generated manifests, tables, and figures with the selected configuration.

## Using the saved baseline results

The baseline cache can be loaded through the project API. Run this from the repository root after activating the environment:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("simulation_code").resolve()))

from mixed_gds_arm import load_study_results

results = load_study_results("study_outputs")

print(results["studies"])
print(results["design_diagnostics"])
print(results["nint"].summary_results.head())
print(results["ntop_pkeep"].summary_results.head())
print(results["nrep"].summary_results.head())
```

The cache was serialized with Python `pickle` and compressed with gzip. Keep these points in mind:

- Only unpickle artifacts obtained from a trusted source.
- Python and library-version differences can affect pickle compatibility.
- `load_study_results()` checks the stored source hash by default.
- If the source has been intentionally modified, `require_current_source=False` bypasses the hash check, but the loaded results should then be treated as historical rather than regenerated by the current code.

The cache is useful for programmatic analysis. It is not a substitute for the CSV directories expected by the standalone figure-reproduction script.

## Running the simulation studies

Full runs are computationally expensive. The default benchmark contains many design, scenario, tuning, and Monte Carlo combinations, and every dataset contains an inner random-submodel repetition loop. Start with `--n-jobs 1` when validating an environment, then increase parallelism deliberately.

Run commands from the repository root unless otherwise noted.

### Baseline tuning studies

```bash
python simulation_code/mixed_gds_arm.py \
  --output-dir reruns/study_outputs \
  --seed 2026 \
  --reps-per-condition 200 \
  --n-jobs -1
```

The baseline driver runs the `nint`, `(ntop, pkeep)`, and `nrep` studies. A new output directory is recommended when you want a true rerun rather than reuse of the included cache.

### Weighted-only benchmark

```bash
python simulation_code/aws_gds_arm.py \
  --output-dir reruns/study_outputs_aws_weighted \
  --seed 2026 \
  --reps-per-condition 200 \
  --benchmark-nrep 1000 \
  --n-jobs -1
```

### Adaptive-stopping benchmark

```bash
python simulation_code/aws_gds_arm_adaptive.py \
  --output-dir reruns/study_outputs_aws_adaptive \
  --seed 2026 \
  --reps-per-condition 200 \
  --benchmark-nrep 1000 \
  --adaptive-max-nrep 1000 \
  --adaptive-min-nrep 100 \
  --adaptive-batch-size 50 \
  --adaptive-epsilon 0.01 \
  --adaptive-patience 2 \
  --n-jobs -1
```

Add `--adaptive-require-set-stability` to require selected-set stability as well as evidence-score stability.

### Adaptive interaction-sampling benchmark

```bash
python simulation_code/aws_gds_arm_adaptive_sampling.py \
  --output-dir reruns/study_outputs_aws_sampling \
  --seed 2026 \
  --reps-per-condition 200 \
  --benchmark-nrep 1000 \
  --sampling-batch-size 50 \
  --sampling-warmup-batches 2 \
  --sampling-lambda 0.7 \
  --sampling-eta 0.5 \
  --sampling-block-floor 0.001 \
  --n-jobs -1
```

### Unified four-method comparison

```bash
python simulation_code/aws_gds_arm_methodology_comparison.py \
  --output-dir reruns/study_outputs_aws_methodology \
  --seed 2026 \
  --reps-per-condition 200 \
  --benchmark-nrep 1000 \
  --adaptive-max-nrep 1000 \
  --n-jobs -1
```

This is the preferred driver for a matched comparison because all four methods share the same generated designs, sparse scenarios, tuning constants, and evaluation definitions.

### Selecting individual variants

The extension drivers accept repeated `--variant` options. For example:

```bash
python simulation_code/aws_gds_arm_methodology_comparison.py \
  --output-dir reruns/methodology_selected \
  --variant original_baseline \
  --variant adaptive_nrep_only
```

Valid methodology labels are:

- `original_baseline`
- `weighted_only`
- `adaptive_nrep_only`
- `adaptive_sampling_only`

### Parallelism and task sharding

- `--n-jobs 1` runs serially and is easiest to debug.
- `--n-jobs 0` or `--n-jobs -1` uses all detected CPUs.
- Positive values request that many worker processes.
- The baseline driver also supports `--task-shards` and `--task-shard-index` for distributed task streams.
- When SLURM array variables are present, the baseline can infer shard count and index automatically.

Each shard should write to its own output directory. The repository does not currently provide a standalone command that combines independently written shard directories into one final bundle.

## Rebuilding figures

[`simulation_code/reproduce_saved_figures.py`](./simulation_code/reproduce_saved_figures.py) rebuilds plots from saved CSV summary tables. It does **not** rerun the Monte Carlo simulations.

The complete figure set contains:

| Study | Figures |
| --- | ---: |
| Baseline tuning | 5 |
| Weighted aggregation | 6 |
| Adaptive stopping | 6 |
| Adaptive interaction sampling | 6 |
| Unified methodology comparison | 11 |
| **Total** | **34** |

The reproducer searches for the five standard output bundles beside the script. To create that layout, enter `simulation_code/` and run each driver with its default output directory:

```bash
cd simulation_code

python mixed_gds_arm.py
python aws_gds_arm.py
python aws_gds_arm_adaptive.py
python aws_gds_arm_adaptive_sampling.py
python aws_gds_arm_methodology_comparison.py

python reproduce_saved_figures.py --study all
```

Regenerated figures are written to:

```text
simulation_code/reproduced_from_saved_results/
```

The directory also contains `reproduction_manifest.json`, which records input paths, SHA-256 hashes, output hashes, Python information, and installed package versions.

To rebuild selected bundles:

```bash
python simulation_code/reproduce_saved_figures.py --study baseline
python simulation_code/reproduce_saved_figures.py --study adaptive methodology
```

To choose another destination:

```bash
python simulation_code/reproduce_saved_figures.py \
  --study all \
  --output-root /path/to/reproduced_figures
```

> A minimal clone containing only `study_results.pkl.gz` and the settings manifests does not yet satisfy the reproducer's CSV inputs. Generate the corresponding `tables/`, `weighted_tables/`, `adaptive_tables/`, `sampling_tables/`, and `methodology_tables/` directories first.

## Configuration and command-line options

Use `--help` on any driver for its complete interface. The most common controls are:

| Option | Available in | Description |
| --- | --- | --- |
| `--output-dir` | All study drivers | Destination for settings, tables, plots, and optional caches |
| `--seed` | All study drivers | Random seed for design generation and simulation |
| `--reps-per-condition` | All study drivers | Monte Carlo datasets per design–scenario–tuning condition |
| `--n-jobs` | All study drivers | Local worker-process count |
| `--benchmark-nrep` | Extension drivers | Fixed inner repetition budget |
| `--weighted-tau` | Extension drivers | Weighted inclusion threshold; defaults to `pkeep` |
| `--variant` | Extension drivers | Restricts the run to one or more supported variants |
| `--adaptive-*` | Adaptive and methodology drivers | Sequential-stopping controls |
| `--sampling-*` | Sampling and methodology drivers | Adaptive proposal controls |
| `--task-shards` | Baseline driver | Number of distributed task partitions |
| `--task-shard-index` | Baseline driver | Zero-based partition assigned to the current process |

The checked-in baseline and adaptive JSON manifests are the authoritative record of the reported run configuration. They include seeds, design identifiers, tuning grids, process settings, write options, and source hashes.

## Output files

Fresh runs create method-specific directories:

| Driver | Default directory | Tables directory | Plots directory |
| --- | --- | --- | --- |
| `mixed_gds_arm.py` | `study_outputs/` | `tables/` | `plots/` |
| `aws_gds_arm.py` | `study_outputs_aws_weighted/` | `weighted_tables/` | `weighted_plots/` |
| `aws_gds_arm_adaptive.py` | `study_outputs_aws_adaptive/` | `adaptive_tables/` | `adaptive_plots/` |
| `aws_gds_arm_adaptive_sampling.py` | `study_outputs_aws_sampling/` | `sampling_tables/` | `sampling_plots/` |
| `aws_gds_arm_methodology_comparison.py` | `study_outputs_aws_methodology/` | `methodology_tables/` | `methodology_plots/` |

Depending on the driver and settings, outputs include:

- a JSON run manifest;
- design diagnostics;
- condition-level and overall summary tables;
- tuning-grid aggregates;
- factor truth records;
- optional replication-level tables;
- adaptive stopping histories or sampling diagnostics;
- publication-style PNG figures; and
- for the baseline, an optional gzip-compressed pickle cache.

## Reproducibility notes

### Fixed implementation controls

The included baseline manifest records:

- random seed `2026`;
- `paper_exact` mixed-level coding;
- 64 candidate generated designs per template;
- 8 random-start design-search restarts;
- 200 Monte Carlo datasets per condition;
- baseline `ntop` grid: `20, 30, 40, 60, 80, 100`;
- baseline `pkeep` grid: `0.10, 0.25, 0.40, 0.55, 0.70`; and
- baseline `nrep` grid: `100, 500, 1000, 4000`.

### Scope of numerical reproduction

The benchmark recreates the published mixed-level coding and algorithmic workflow on generated design templates matched to the reference design dimensions. It does not reuse the exact rows of every original supplement design. Results should therefore be interpreted as a controlled, template-matched numerical benchmark rather than a byte-for-byte reproduction of every published number.

### Cross-platform variation

- Random seeds and run manifests provide the primary computational provenance.
- Absolute runtimes vary with CPU, process count, solver build, and operating system.
- Matplotlib and Plotnine can produce slightly different PNG bytes across platforms even when the plotted data agree.
- Solver tolerances and numerical linear algebra libraries can create small floating-point differences.
- For comparisons, preserve the same seed, design configuration, simulation count, and package environment across methods.

## Troubleshooting

### `PermissionError` involving `/tmp/mplconfig` on Windows

Set a writable Matplotlib cache before starting Python:

```powershell
$env:MPLCONFIGDIR = Join-Path $PWD ".matplotlib"
```

### `ModuleNotFoundError`

Activate the intended virtual environment and reinstall the pinned dependencies:

```bash
python -m pip install -r simulation_code/requirements.txt
```

### CVXPY solver problems

Check the available solvers:

```bash
python -c "import cvxpy as cp; print(cp.installed_solvers())"
```

Then confirm that the pinned solver packages installed successfully. A clean virtual environment is preferable to mixing this project with unrelated scientific-Python packages.

### Figure reproducer reports missing CSV files

The reproducer consumes saved summary CSVs, not the pickle cache. Run the relevant study driver first and place its standard output folder beside `reproduce_saved_figures.py`, or copy a complete output bundle into that location.

### A full run appears slow

This is expected. Each Monte Carlo condition includes repeated sparse optimization, model ranking, and post-selection fitting. Validate the environment serially, then use `--n-jobs` or baseline task sharding on appropriate hardware. Avoid comparing runtime results across different process counts or machines.

### Saved cache fails the source-hash check

The cache was produced by a different byte-level version of `mixed_gds_arm.py`. Use the committed source associated with the manifest for strict reproduction. Only disable the check when intentionally analyzing a historical cache with modified code.

## References and citation

The baseline implementation follows the mixed-level extension described in:

> Zhang, F., Singh, R., and Stufken, J. (2026). “An Extension of the GDS-ARM Algorithm for Factor Screening in Mixed-Level Supersaturated Designs.” *Journal of Statistical Theory and Practice*, 20, 41. [https://doi.org/10.1007/s42519-026-00542-x](https://doi.org/10.1007/s42519-026-00542-x)

Additional methodological references:

- Candès, E. and Tao, T. (2007). “The Dantzig Selector: Statistical Estimation When $p$ Is Much Larger Than $n$.” *The Annals of Statistics*, 35(6), 2313–2351.
- Phoa, F. K., Pan, Y.-H., and Xu, H. (2009). “Analysis of Supersaturated Designs via the Dantzig Selector.” *Journal of Statistical Planning and Inference*, 139(7), 2362–2372.
- Singh, R. and Stufken, J. (2024). “Factor Selection in Screening Experiments by Aggregation over Random Models.” *Computational Statistics & Data Analysis*, 194, 107940.

If you use this repository, cite both the reference method and this implementation:

```text
Singh, A. (2026). Adaptive Weighted Sequential GDS-ARM for Factor Screening
[Computer software]. GitHub.
https://github.com/amansd78/Adaptive-Weighted-Sequential-GDS-ARM-for-Factor-Screening
```

## Questions and contributions

Questions, bug reports, and reproducibility notes are welcome through the repository's [GitHub Issues](https://github.com/amansd78/Adaptive-Weighted-Sequential-GDS-ARM-for-Factor-Screening/issues) page. When reporting a computational issue, include:

- operating system and Python version;
- the command used;
- the relevant JSON run manifest;
- installed CVXPY solvers;
- the complete traceback; and
- whether the run was serial, multiprocessing, or sharded.

## License

No software license is currently included in the repository. Until a license is added, reuse, modification, and redistribution are not automatically granted. Add a `LICENSE` file before presenting the project as open-source software.
