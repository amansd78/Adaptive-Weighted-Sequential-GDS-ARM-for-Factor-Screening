PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: venv install reproduce-figures reproduce-baseline reproduce-weighted reproduce-adaptive reproduce-sampling reproduce-methodology rerun-baseline rerun-weighted rerun-adaptive rerun-sampling rerun-methodology

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

reproduce-figures:
	$(VENV_PYTHON) reproduce_saved_figures.py --study all

reproduce-baseline:
	$(VENV_PYTHON) reproduce_saved_figures.py --study baseline

reproduce-weighted:
	$(VENV_PYTHON) reproduce_saved_figures.py --study weighted

reproduce-adaptive:
	$(VENV_PYTHON) reproduce_saved_figures.py --study adaptive

reproduce-sampling:
	$(VENV_PYTHON) reproduce_saved_figures.py --study sampling

reproduce-methodology:
	$(VENV_PYTHON) reproduce_saved_figures.py --study methodology

rerun-baseline:
	$(VENV_PYTHON) mixed_gds_arm.py

rerun-weighted:
	$(VENV_PYTHON) aws_gds_arm.py

rerun-adaptive:
	$(VENV_PYTHON) aws_gds_arm_adaptive.py

rerun-sampling:
	$(VENV_PYTHON) aws_gds_arm_adaptive_sampling.py

rerun-methodology:
	$(VENV_PYTHON) aws_gds_arm_methodology_comparison.py
