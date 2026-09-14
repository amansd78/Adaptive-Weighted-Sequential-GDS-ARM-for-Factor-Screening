# AWS-GDS-ARM Reproducibility Package

This repository contains the code, saved study outputs, manuscript sources, and plotting workflow for the adaptive extensions of mixed-level GDS-ARM developed for the STAT 778 project.

The repository is organized so that a reader can do either of the following:

1. Rebuild the submitted figure set directly from the saved result tables already included in the repo.
2. Rerun the full simulation studies from scratch with the same default settings used to create the saved outputs.

The first path is the recommended one for a professor or reviewer who only needs to reproduce the submitted graphs.

## Quick Start

If you only want to regenerate the figures from the saved results, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python reproduce_saved_figures.py --study all
```

This writes a fresh reproduction bundle to:

```text
reproduced_from_saved_results/
```

That folder mirrors the original output structure and also includes:

```text
reproduced_from_saved_results/reproduction_manifest.json
```

The manifest records:

- the exact input tables used for each figure bundle
- SHA256 hashes for those inputs
- SHA256 hashes for the regenerated PNG files
- the Python version and package versions used for the reproduction run

## One-Command Option

If you prefer `make` targets:

```bash
make venv
make install
make reproduce-figures
```

Available convenience targets:

- `make reproduce-baseline`
- `make reproduce-weighted`
- `make reproduce-adaptive`
- `make reproduce-sampling`
- `make reproduce-methodology`

## What Is Already Saved

The repository already includes the saved study outputs used to create the submission figures.

| Study bundle | Saved tables | Saved plots | Default output folder |
| --- | --- | --- | --- |
| Baseline mixed-level GDS-ARM | `study_outputs/tables/` | `study_outputs/plots/` | `study_outputs/` |
| Weighted-only AWS-GDS-ARM | `study_outputs_aws_weighted/weighted_tables/` | `study_outputs_aws_weighted/weighted_plots/` | `study_outputs_aws_weighted/` |
| Adaptive stopping benchmark | `study_outputs_aws_adaptive/adaptive_tables/` | `study_outputs_aws_adaptive/adaptive_plots/` | `study_outputs_aws_adaptive/` |
| Adaptive sampling benchmark | `study_outputs_aws_sampling/sampling_tables/` | `study_outputs_aws_sampling/sampling_plots/` | `study_outputs_aws_sampling/` |
| Unified methodology comparison | `study_outputs_aws_methodology/methodology_tables/` | `study_outputs_aws_methodology/methodology_plots/` | `study_outputs_aws_methodology/` |

The checked-in JSON settings files next to those folders are the canonical record of the run configuration used to generate the saved results.

## Figure Reproduction Workflow

The script [`reproduce_saved_figures.py`](./reproduce_saved_figures.py) is the main reproducibility entrypoint for this submission.

It does not rerun the expensive simulation studies. Instead, it reads the already-saved CSV summaries and rebuilds the figures in a clean output directory.

### Rebuild Everything

```bash
python reproduce_saved_figures.py --study all
```

### Rebuild Only One Study Bundle

```bash
python reproduce_saved_figures.py --study baseline
python reproduce_saved_figures.py --study weighted
python reproduce_saved_figures.py --study adaptive
python reproduce_saved_figures.py --study sampling
python reproduce_saved_figures.py --study methodology
```

### Write to a Custom Output Directory

```bash
python reproduce_saved_figures.py --study all --output-root /path/to/output_dir
```

### Figures Produced by the Reproducer

| Study | Number of regenerated figures |
| --- | ---: |
| Baseline | 5 |
| Weighted-only | 6 |
| Adaptive stopping | 6 |
| Adaptive sampling | 6 |
| Methodology comparison | 11 |

On the verified environment, rebuilding all saved-result figures completed successfully and generated 34 PNG files plus the reproduction manifest.

## Exact Environment Used for Verification

The reproduction workflow in this repository was verified with:

- Python `3.14.2`
- CPython on `macOS arm64`
- pinned package versions listed in [`requirements.txt`](./requirements.txt)
- a checked-in environment snapshot in [`environment_snapshot.json`](./environment_snapshot.json)

If you want the closest match to the verified setup, use the versions pinned in `requirements.txt` and compare against `environment_snapshot.json`.

Note on exact PNG hashes:

- the saved CSV inputs, settings files, and plotting code path are fixed by this repository
- the regenerated PNG bytes may still differ slightly across platforms or plotting-library builds because `matplotlib` and `plotnine` can render text and metadata differently
- for provenance, the strongest reproducibility checks in this repo are the input-table hashes, the run-settings files, and the environment snapshot

## Full Study Reruns

If you want to rerun the underlying simulations from scratch instead of regenerating figures from saved CSV outputs, use the study drivers below.

### Baseline GDS-ARM Studies

```bash
python mixed_gds_arm.py
```

This runs the three baseline studies:

- `nint` study
- `ntop/pkeep` study
- `nrep` study

### Weighted-Only AWS-GDS-ARM

```bash
python aws_gds_arm.py
```

### Adaptive Stopping Benchmark

```bash
python aws_gds_arm_adaptive.py
```

### Adaptive Sampling Benchmark

```bash
python aws_gds_arm_adaptive_sampling.py
```

### Unified Methodology Comparison

```bash
python aws_gds_arm_methodology_comparison.py
```

### Notes on Full Reruns

- Full reruns are much more expensive than the saved-results figure rebuild.
- The default study settings are stored in the corresponding `*_run_settings.json` files.
- The benchmark uses generated designs matched to the paper-style structure, not a byte-for-byte replay of every original supplement artifact.

## Repository Map

| Path | Purpose |
| --- | --- |
| [`mixed_gds_arm.py`](./mixed_gds_arm.py) | Baseline mixed-level GDS-ARM simulation driver and plotting functions |
| [`aws_gds_arm.py`](./aws_gds_arm.py) | Weighted-only extension study |
| [`aws_gds_arm_adaptive.py`](./aws_gds_arm_adaptive.py) | Adaptive stopping study |
| [`aws_gds_arm_adaptive_sampling.py`](./aws_gds_arm_adaptive_sampling.py) | Adaptive interaction-sampling study |
| [`aws_gds_arm_methodology_comparison.py`](./aws_gds_arm_methodology_comparison.py) | Unified four-method comparison |
| [`aws_gds_arm_common.py`](./aws_gds_arm_common.py) | Shared summary and plotting utilities |
| [`reproduce_saved_figures.py`](./reproduce_saved_figures.py) | Lightweight figure reproducer from saved CSV tables |
| [`Documents/aws_gds_arm_paper.tex`](./Documents/aws_gds_arm_paper.tex) | Manuscript source |
| [`paper_supplement/`](./paper_supplement/) | Original paper supplement materials kept with the project |

## Recommended Submission Note

If this repository is being shared with a professor or reviewer, the easiest path is:

1. Create a fresh virtual environment.
2. Install `requirements.txt`.
3. Run `python reproduce_saved_figures.py --study all`.
4. Inspect `reproduced_from_saved_results/reproduction_manifest.json` for the exact input/output hashes and environment metadata.

That route avoids the runtime cost of rerunning the simulations while still reproducing the submitted figures from the saved study tables included in the repository.
