# Real Estate Value Analyzer  
**CS 506 Final Project Proposal**

---

## 1. Project Description & Motivation
The residential real estate market exhibits significant information asymmetry: buyers and investors often struggle to determine whether a listing is priced reasonably relative to comparable properties. While existing platforms provide automated estimates, these systems are proprietary and often fail to incorporate granular qualitative signals such as property condition or hyper-local market context.

This project aims to build an end-to-end data science pipeline that estimates **expected market prices for residential properties based on observable features**, and then analyzes **pricing deviations relative to comparable homes**. Rather than asserting intrinsic or “true” value, the project focuses on identifying **relative mispricing** by comparing a listing’s price to the model’s expectation given similar properties.

The final product will be an interactive visualization that highlights properties priced unusually high or low relative to their peers, enabling exploratory analysis of potential market inefficiencies.

---

## 2. Project Goals
The primary goal of this project is to practice the full data science lifecycle—from data collection and cleaning to modeling, evaluation, and visualization—while answering a concrete applied question:

**Can we identify listings that are priced unusually high or low relative to comparable homes by quantifying both structured and unstructured listing data?**

### Specific, Measurable Goals
- **Data Collection:**  
  Assemble a dataset of 5,000+ residential property listings from a major metropolitan area (e.g., Greater Boston), including structured attributes and unstructured text descriptions.

- **Feature Engineering:**  
  Create at least three novel feature groups:
  - **Condition Indicators:** NLP-derived signals from listing descriptions (e.g., “needs work” vs. “recently renovated”).
  - **Layout Proxies:** Ratios such as square-feet-per-room to approximate layout efficiency.
  - **Neighborhood Context:** Local price statistics derived from nearby comparable properties.

- **Modeling:**  
  Train a regression model to estimate expected listing price, using a baseline linear model and a more expressive tree-based model.

- **Evaluation:**  
  Assess performance using held-out data and residual analysis rather than absolute price accuracy alone.

- **Visualization:**  
  Build an interactive map that allows users to explore properties by pricing deviation (predicted price minus listing price).

---

## 3. Data Collection Plan

### Data Sources
Data will be collected from publicly available real estate listings and open property datasets for a selected metropolitan area. When possible, structured open datasets and documented APIs will be preferred to ensure reproducibility and compliance.

To address proposal feedback, we are now explicitly tracking candidate sources:
- [Zillow Research Data](https://www.zillow.com/research/data/)
- [Redfin Data Center](https://www.redfin.com/news/data-center/)
- [MassGIS Data Portal](https://www.mass.gov/orgs/massgis-bureau-of-geographic-information)

### Collection Method
- Python-based data ingestion pipeline  
- HTML parsing for publicly accessible listing pages or structured datasets  
- Data collected at a low request rate and in compliance with site policies  
- Raw data stored locally for reproducibility  

### Fields Collected
- **Target Variable:** Listing price  
- **Structured Features:** Address (or approximate location), ZIP code, bedrooms, bathrooms, square footage, lot size, year built, HOA fees, property type  
- **Unstructured Features:** Property description text  

---

## 4. Data Cleaning & Feature Extraction

### Data Cleaning
- **Missing Values:**  
  Impute missing numeric attributes (e.g., year built, lot size) using K-nearest-neighbor imputation based on geographic or feature similarity.
- **Outliers:**  
  Remove obvious data errors (e.g., zero price) and extreme outliers that do not reflect typical residential properties.
- **Standardization:**  
  Normalize numeric features where appropriate.

### Feature Engineering
- **NLP Condition Signals:**  
  Extract binary or weighted indicators from listing descriptions using keyword matching or TF-IDF techniques.
- **Geospatial Comparables:**  
  Compute average price-per-square-foot of nearby properties to capture local market effects.
- **Volatility Metrics:**  
  Measure neighborhood-level price variability to approximate market stability.

Checkpoint 1 feature justification (implemented):
- `sqft`, `bedrooms`, `bathrooms`, `year_built`: core structural drivers of home price.
- `sqft_per_room`: layout-efficiency proxy to separate similarly sized homes with different room counts.
- `is_renovated_signal`, `needs_work_signal`: lightweight NLP condition indicators to capture quality not visible in numeric fields.
- `home_age`: interpretable transformation of `year_built` for non-linear age effects.

---

## 5. Modeling Plan
The task will be framed as a **regression problem**.

- **Baseline Model:** Linear Regression with ElasticNet regularization  
- **Primary Model:** Tree-based regression (e.g., Random Forest or Gradient Boosting)  
- **Validation:** K-Fold Cross-Validation (k = 5)  
- **Interpretation:** Residuals (predicted price − listing price) will be used to analyze relative mispricing rather than absolute predictions.

---

## 6. Visualization Plan
- **Interactive Map:**  
  Plot properties geographically and color points by pricing deviation (underpriced vs. overpriced relative to model expectations).
- **Feature Importance Plot:**  
  Visualize which features contribute most to predicted price.
- **Actual vs. Predicted Scatter Plot:**  
  Assess overall model fit and residual distribution.

---

## 7. Testing Strategy
- **Unit Tests:**  
  Validate feature engineering functions (e.g., layout ratio calculations).
- **Pipeline Tests:**  
  Ensure the full modeling pipeline can accept a single property’s features and return a numeric prediction.
- **Data Sanity Checks:**  
  Enforce constraints such as positive square footage and price.

A GitHub Actions workflow is included to run tests automatically.

---

## 8. Project Timeline (8 Weeks)
- **Week 1:** Repository setup, environment configuration, pipeline scaffolding  
- **Week 2:** Data collection and raw data storage  
- **Week 3 (Check-In 1):** Initial cleaning, EDA, baseline modeling artifacts  
- **Week 4:** Feature engineering (NLP and geospatial features)  
- **Week 5:** Model training and baseline comparisons  
- **Week 6 (Check-In 2):** Model evaluation and error analysis  
- **Week 7:** Visualization development and documentation  
- **Week 8:** Final report and presentation recording  

---

## 9. Reproducibility (Checkpoint 1)

Environment:
- Python 3.10+

Setup:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run checkpoint pipeline:
```bash
PYTHONPATH=src python scripts/run_checkpoint1.py \
  --input data/sample_listings.csv \
  --output outputs/checkpoint1
```

Run tests:
```bash
PYTHONPATH=src pytest -q
```

Minimal outputs for Check-In 1 are written to `outputs/checkpoint1/` (metrics + preliminary plots).

Checkpoint 1 preliminary interpretation (sample data run):
- Linear baseline: MAE = 15,756, RMSE = 25,438, R² = 0.992.
- Random forest: MAE = 36,985, RMSE = 53,182, R² = 0.966.
- At this stage, these numbers are from a small synthetic-style sample (`data/sample_listings.csv`), so they indicate that the pipeline is working, not final model quality.
- Next step is to evaluate on larger real collected data and analyze residuals by location/price segment to identify where the model fails.

---

## 10. Checkpoint 1 Completion Checklist

- [x] Data source candidates are explicitly linked (Zillow, Redfin, MassGIS).
- [x] Initial dataset is available for reproducible progress (`data/sample_listings.csv`).
- [x] Data cleaning and feature extraction pipeline is implemented (`src/real_estate_tracker/data_processing.py`).
- [x] Baseline and tree-based modeling are implemented (`src/real_estate_tracker/modeling.py`).
- [x] Preliminary visualizations are generated (`outputs/checkpoint1/figures/`).
- [x] Preliminary metrics are generated (`outputs/checkpoint1/metrics.json`).
- [x] Tests are included (`tests/`) and automated with GitHub Actions (`.github/workflows/ci.yml`).

---

## 11. Checkpoint Demo Runbook

From repository root:

```bash
source .venv/bin/activate
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 scripts/run_checkpoint1.py --input data/sample_listings.csv --output outputs/checkpoint1
```

Open generated visuals (macOS):
```bash
open outputs/checkpoint1/figures/*.png
```

Open summary files:
```bash
open outputs/checkpoint1/metrics.json
open outputs/checkpoint1/run_summary.json
```
