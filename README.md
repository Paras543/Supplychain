# SupplyChainIQ

**Explainable Shipment Delay Risk Intelligence System**

An end-to-end machine learning system that predicts shipment delay risk and frames it as a decision-support tool for logistics managers — not just a binary classifier, but a scored, explainable risk report.

> Instead of asking *"Will this shipment be delayed?"*, SupplyChainIQ asks *"How risky is this shipment, why is it risky, and what should be done about it?"*

---

## Overview

SupplyChainIQ is built on the [DataCo Smart Supply Chain dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (180,519 real shipment records) and covers the full ML lifecycle: EDA, leakage-safe feature engineering, multi-model benchmarking, hyperparameter tuning, a production-style `src/` pipeline, a FastAPI inference service, and a Streamlit demo frontend.

The project was built with a deliberate emphasis on **methodological honesty** — leakage was actively hunted down and removed, feature-drop decisions were backed by EDA evidence rather than intuition, and a hyperparameter tuning attempt that *didn't* improve the model is documented rather than hidden.

---

## System Architecture

```
Raw CSV
   |
   v
+------------------+     +------------------------+     +------------------+     +----------------------+
| Data Ingestion    |---->| Data Transformation     |---->| Model Trainer     |---->| Model Evaluation       |
| (leakage/PII      |     | (calendar features,      |     | (XGBoost,          |     | (classification        |
| drop, split)       |     | encoding, scaling)        |     | default params)     |     | report, confusion      |
+------------------+     +------------------------+     +------------------+     | matrix)                |
                                                                                    +----------------------+
                                                                                            |
                                              +---------------------------------------------+
                                              v
                              +--------------------------+       +---------------------+
                              | Prediction Pipeline        |----->| FastAPI / Streamlit   |
                              | (loads saved artifacts,     |      | UI                    |
                              | scores a single row)         |      +---------------------+
                              +--------------------------+
```

---

## Results

Six classifiers were trained with default hyperparameters and evaluated on train **and** test splits to explicitly surface overfitting, not just raw test performance.

| Model | Test F1 | Test ROC-AUC | Train/Test Gap | Verdict |
|---|---|---|---|---|
| Random Forest | 0.752 | 0.850 | 0.248 | Severely overfit |
| **CatBoost** | **0.729** | **0.841** | 0.039 | Best generalization |
| **XGBoost** | 0.722 | 0.820 | 0.035 | **Final model** |
| LightGBM | 0.687 | 0.809 | 0.013 | Underfitting slightly |
| HistGradientBoosting | 0.686 | 0.809 | 0.013 | Underfitting slightly |
| Logistic Regression | 0.669 | 0.767 | 0.005 | Baseline / honest floor |

**Final model:** default-hyperparameter XGBoost — **Test F1 = 0.722, Test ROC-AUC = 0.820, Test Accuracy ≈ 0.74**

A `RandomizedSearchCV` tuning pass was run on XGBoost and is documented rather than cherry-picked:
- A wide, unconstrained search improved accuracy to ~0.77 but **increased the overfitting gap**.
- A heavily-regularized search eliminated the gap (0.0037) but **underfit**, scoring below the default model.
- **Decision:** the untuned default model was kept, since neither tuning direction beat its combination of performance and generalization.

---

## Key EDA Findings

- **Target distribution:** 54.7% Late / 45.2% On-time — mild imbalance, addressed with stratified splits and F1/ROC-AUC/PR-AUC over raw accuracy.
- **Strongest signal:** `Days for shipment (scheduled)` — on-time shipments cluster at a 4-day schedule; late shipments spread widely with a lower median.
- **Leakage identified and removed:** `Delivery Status`, `Days for shipping (real)`, and `shipping date (DateOrders)` are only knowable *after* a shipment's outcome — all three were dropped before modeling.
- **Counterintuitive finding:** "First Class" shipping has the *highest* historical late rate (95.5%) of any shipping mode, higher than Standard Class (38.0%) — likely a symptom of unrealistically tight scheduling windows on premium tiers, kept as a documented finding rather than smoothed over.

---

## Repository Structure

```
supply-chain-iq/
|-- artifacts/                  # trained model + fitted encoders/scaler (.pkl)
|-- data/
|   |-- raw/                    # source CSV
|   |-- interim/                # cleaned, pre-split data
|   `-- processed/              # final train/test feature sets
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_feature_engineering.ipynb
|   |-- 03_model_training.ipynb
|   `-- 04_model_evaluation.ipynb
|-- src/
|   |-- exception.py            # custom exception with file/line traceback detail
|   |-- logger.py               # timestamped per-run logging
|   |-- components/
|   |   |-- data_ingestion.py
|   |   |-- data_transformation.py
|   |   |-- model_trainer.py
|   |   `-- model_evaluation.py
|   `-- pipeline/
|       |-- training_pipeline.py
|       `-- prediction_pipeline.py
|-- app/
|   |-- main.py                 # FastAPI app
|   `-- schemas.py               # Pydantic request/response models
|-- streamlit_app.py             # interactive demo frontend
|-- logs/                        # runtime logs
`-- requirements.txt
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & Modeling | pandas, NumPy, scikit-learn, XGBoost, LightGBM, CatBoost |
| Pipeline / Engineering | Custom `CustomException` + `logging` framework, dataclass-based configs |
| Serving | FastAPI, Pydantic |
| Demo Frontend | Streamlit |
| Artifact Persistence | joblib |

---

## Getting Started

### 1. Clone and install
```bash
git clone https://github.com/<your-username>/supply-chain-iq.git
cd supply-chain-iq
pip install -r requirements.txt
```

### 2. Add the dataset
Download the [DataCo Smart Supply Chain Dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) from Kaggle and place it at:
```
data/raw/DataCoSupplyChainDataset.csv
```

### 3. Run the full training pipeline
```bash
python -m src.pipeline.training_pipeline
```
This runs ingestion, transformation, training, and evaluation end to end, and saves the trained model and preprocessing artifacts to `artifacts/`.

### 4. Test a single prediction
```bash
python -m src.pipeline.prediction_pipeline
```

### 5. Launch the API
```bash
uvicorn app.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

### 6. Launch the demo frontend
```bash
streamlit run streamlit_app.py
```

---

## Engineering Practices

- **Leakage-first mindset:** every feature was evaluated for whether it would be known *before* a shipment ships, not just whether it correlates with the target.
- **Train-only statistics:** all encoders, frequency maps, and the scaler are fit exclusively on the training split and applied (never refit) to test data.
- **Overfitting as a first-class metric:** every model is judged on its train/test gap, not just its raw test score — this is what disqualified Random Forest despite a strong headline number.
- **Modular components:** each pipeline stage reads from and writes to disk independently, so any single stage can be re-run and debugged in isolation.
- **Consistent error handling:** every component raises a `CustomException` carrying the exact file and line number of the failure, and every stage logs to a timestamped run-specific log file.

---

## Limitations & Future Work

- [ ] Merge the validated `shipping_mode_late_rate` feature (and add `region_late_rate`, `category_late_rate`, `is_domestic`) into the model — the most promising lever for improving past the current ceiling.
- [ ] Resolve redundant price columns (`Sales`, `Order Item Total`, `Product Price`, `Order Item Product Price`) flagged by the correlation heatmap.
- [ ] Add SHAP-based explainability to power the "Top Risk Factors" component of the risk report.
- [ ] Calibrate output probabilities (Platt scaling / isotonic regression).
- [ ] Containerize with Docker and add CI/CD.
- [ ] Build a monitoring dashboard for data/model drift.

---

## License

This project is for educational and portfolio purposes, built on the publicly available DataCo Smart Supply Chain dataset.
