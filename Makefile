# Real Estate Value Analyzer — Makefile
# Run `make help` to see available targets.

# Use the venv's Python if it exists, otherwise system python3
VENV_PY := .venv/bin/python
PYTHON := $(shell [ -x $(VENV_PY) ] && echo $(VENV_PY) || echo python3)

PYTHONPATH := src

.PHONY: help install data pipeline model test all clean

help:
	@echo "Available targets:"
	@echo "  make install   - Create venv and install dependencies"
	@echo "  make data      - Download all raw datasets (Assessment, Zillow, Census)"
	@echo "  make pipeline  - Clean + enrich data -> data/processed/boston_properties_enriched.csv"
	@echo "  make model     - Train models + generate figures -> outputs/checkpoint2/"
	@echo "  make test      - Run pytest test suite"
	@echo "  make all       - Run the full project end-to-end (install -> data -> pipeline -> model)"
	@echo "  make clean     - Remove generated outputs (keeps raw data)"

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Installed. To use the venv interactively, run: source .venv/bin/activate"

data:
	$(PYTHON) scripts/download_datasets.py
	$(PYTHON) scripts/fetch_api_data.py
	@echo ""
	@echo "If the Boston Property Assessment CSV failed (403), download manually from:"
	@echo "  https://data.boston.gov/dataset/property-assessment"
	@echo "Save it to: data/raw/boston_property_assessment_fy2026.csv"

pipeline:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_pipeline.py

model:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_model.py

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

all: install data pipeline model
	@echo ""
	@echo "Full project run complete. See outputs/checkpoint2/ for results."

clean:
	rm -rf outputs/checkpoint2/figures
	rm -f outputs/checkpoint2/metrics.json
	rm -f outputs/checkpoint2/residuals.csv
	rm -f outputs/checkpoint2/run_summary.json
	rm -rf data/processed
	@echo "Cleaned generated outputs. Raw data preserved."