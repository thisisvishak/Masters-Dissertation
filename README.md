# Risk-Calibrated and Explainable Credit Card Default Prediction
**Using Monotonic LightGBM with Isotonic Calibration and Split Conformal Prediction**  
**MSc Dissertation Repository**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Method Summary](#method-summary)
- [Engineered Features](#engineered-features)
- [Model Suite](#model-suite)
- [Reproducible Environment](#reproducible-environment)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [End-to-End Pipeline (CLI)](#end-to-end-pipeline-cli)
- [Key Results (Test Set)](#key-results-test-set)
- [Business Policy](#business-policy)
- [Explainability](#explainability)
- [Fairness & Uncertainty](#fairness--uncertainty)
- [Saved Artifacts](#saved-artifacts)
- [Citations (Harvard style)](#citations-harvard-style)
- [License](#license)

---

## Project Overview
This repository implements an end-to-end, production-ready credit risk workflow on the **Default of Credit Card Clients (Taiwan)** dataset.  
It delivers:
- A **monotonic LightGBM** classifier calibrated with **isotonic regression**;
- **Split conformal prediction** for per-case uncertainty bands;
- **SHAP-based** global and local explanations;
- A **three-band decision policy** (approve / review / decline) optimised by profit curves;
- A packaged **predictor class** and versioned artifacts for deployment.

---

## Dataset
- **Source:** Kaggle – *Default of Credit Card Clients* (30,000 customers, Taiwan).
- **Target:** `default.payment.next.month` (1 = default next month).
- **Horizon:** 6 months of bills (`BILL_AMT1..6`), payments (`PAY_AMT1..6`), and repayment status (`PAY_0, PAY_2..6`) + demographics and `LIMIT_BAL`.

> Download the CSV into `data/` as `UCI_Credit_Card.csv`.

---

## Method Summary
1. **Data audit & cleaning** (types, label rate, outliers; no missing values in source).  
2. **Feature engineering** (utilisation, payment ratios, delinquency flags/trends, aggregates, demographic groupings).  
3. **Preprocessing** with `ColumnTransformer`:  
   - `StandardScaler` for numeric;  
   - `OneHotEncoder(drop="first")` for categorical;  
   - class imbalance via **class weights**.  
4. **Model comparison** across 8 algorithms with common preprocessing pipeline.  
5. **Champion**: monotonic **LightGBM** trained with early stopping; **isotonic calibration** on validation; **split conformal** on calibrated scores.  
6. **Evaluation**: Accuracy, ROC AUC, PR AUC, **KS**, Brier, lift/gain, calibration curves, approval–default trade-off, **profit curve**.  
7. **Explainability**: SHAP global bars, beeswarm, water­fall (local), targeted dependence (“story” groups).  
8. **Packaging**: `CreditRiskPredictor` class saved with `joblib` for downstream scoring.

---

## Engineered Features
- `UTILIZATION_1..6 = BILL_AMTi / LIMIT_BAL`  
- `PAY_RATIO_1..6 = PAY_AMTi / BILL_AMTi` (safe 0 when bill==0)  
- `EVER_60_PLUS` from repayment codes (`PAY_* >= 2`)  
- `AVG_PAY_STATUS` mean of `PAY_0, PAY_2..6`  
- `PAY_TREND = PAY_6 - PAY_0`  
- `FULL_PAYMENT_MONTHS` count where `PAY_AMTi >= BILL_AMTi > 0`  
- `TOTAL_PAY_RATIO = sum(PAY_AMT) / (sum(BILL_AMT)+1)`  
- `AGE_GROUP` bins; `EDUCATION_GROUP` merged rare categories

---

## Model Suite
- Logistic Regression (class-weighted)
- Decision Tree (tuned, `max_depth`, `min_samples_split`)
- Random Forest (tuned, class-weighted)
- XGBoost (tuned, `scale_pos_weight`)
- LightGBM (tuned, monotonic constraints, `scale_pos_weight`)
- SVM (RBF, class-weighted, probability=True)
- KNN (scaled)
- MLP (2 hidden layers, early stopping)

---

## Reproducible Environment
# Conda
conda create -n creditrisk python=3.10 -y
conda activate creditrisk
pip install -r requirements.txt
requirements.txt

pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
shap
joblib

# Project Structure

.
├── data/
│   └── UCI_Credit_Card.csv
├── notebooks/
│   └── 01_end_to_end_credit_risk.ipynb
├── src/
│   ├── features.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── explain.py
│   └── predict.py
├── artifacts/
│   ├── credit_risk_predictor.pkl
│   └── feature_list.json
├── figures/
│   └── (generated plots)
├── requirements.txt
└── README.md


# Quick Start

# 1) Place dataset
mkdir -p data artifacts figures
cp /path/to/UCI_Credit_Card.csv data/

# 2) Create environment & install
conda create -n creditrisk python=3.10 -y
conda activate creditrisk
pip install -r requirements.txt

# 3) Run notebook (recommended for full plots)
jupyter lab notebooks/01_end_to_end_credit_risk.ipynb
End-to-End Pipeline (CLI)
bash
Copy code
python - <<'PY'
import pandas as pd, numpy as np, joblib, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

# 4) Load
df = pd.read_csv("data/UCI_Credit_Card.csv")

# 5) Feature engineering (abridged; mirror notebook for full set)
data = df.rename(columns={"default.payment.next.month":"TARGET"})
for i in range(1,7):
    data[f"UTILIZATION_{i}"] = data[f"BILL_AMT{i}"]/ (data["LIMIT_BAL"].replace(0, np.nan))
    data[f"UTILIZATION_{i}"] = data[f"UTILIZATION_{i}"].fillna(0.0)
    data[f"PAY_RATIO_{i}"] = np.where(data[f"BILL_AMT{i}"]>0, data[f"PAY_AMT{i}"]/data[f"BILL_AMT{i}"], 0.0)

pay_cols = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
data["EVER_60_PLUS"] = (data[pay_cols] >= 2).any(axis=1).astype(int)
data["AVG_PAY_STATUS"] = data[pay_cols].mean(axis=1)
data["PAY_TREND"] = data["PAY_6"] - data["PAY_0"]
data["FULL_PAYMENT_MONTHS"] = sum((data[f"PAY_AMT{i}"]>=data[f"BILL_AMT{i}"])&(data[f"BILL_AMT{i}"]>0) for i in range(1,7))
data["TOTAL_PAY_RATIO"] = data[[f"PAY_AMT{i}" for i in range(1,7)]].sum(axis=1) / (data[[f"BILL_AMT{i}" for i in range(1,7)]].sum(axis=1) + 1)

bins = [20,30,40,50,60,80]
labels = ["20-30","30-40","40-50","50-60","60+"]
data["AGE_GROUP"] = pd.cut(data["AGE"], bins=bins, labels=labels, include_lowest=True)
data["EDUCATION_GROUP"] = data["EDUCATION"].replace({0:4,5:4,6:4})

# 6) Train/Val/Test split
y = data["TARGET"]
X = data.drop(columns=["ID","TARGET"], errors="ignore")
X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.30, stratify=y, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp,y_temp,test_size=0.50, stratify=y_temp, random_state=42)

# 7) Preprocessing
categorical_cols = [c for c in ["SEX","MARRIAGE","AGE_GROUP","EDUCATION_GROUP"] if c in X.columns]
numerical_cols   = [c for c in X.columns if c not in categorical_cols]
ct = ColumnTransformer(
    [("num", StandardScaler(), numerical_cols),
     ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)]
)

# 8) Fit LightGBM with monotonic constraints (basic illustration)
mono = []
for c in X.columns:
    if "UTILIZATION" in c or c in pay_cols: mono.append(1)
    elif "PAY_RATIO" in c or "TOTAL_PAY_RATIO" in c: mono.append(-1)
    else: mono.append(0)

train_processed = ct.fit_transform(X_train)
val_processed   = ct.transform(X_val)
test_processed  = ct.transform(X_test)

lgb_train = lgb.Dataset(train_processed, label=y_train)
lgb_val   = lgb.Dataset(val_processed,   label=y_val)

params = dict(objective="binary",
              metric="auc",
              learning_rate=0.05,
              num_leaves=64,
              max_depth=10,
              min_data_in_leaf=50,
              feature_fraction=0.8,
              bagging_fraction=0.8,
              bagging_freq=5,
              scale_pos_weight=2,
              monotone_constraints=mono,
              verbose=-1,
              seed=42)

model = lgb.train(params, lgb_train, valid_sets=[lgb_val],
                  num_boost_round=1000,
                  callbacks=[lgb.early_stopping(50, verbose=False)])

# 9) Calibration (isotonic)
val_raw = model.predict(val_processed)
iso = IsotonicRegression(out_of_bounds="clip").fit(val_raw, y_val)

# 10) Evaluate on test
test_raw = model.predict(test_processed)
test_cal = iso.transform(test_raw)

acc   = accuracy_score(y_test, (test_cal>0.5).astype(int))
auc   = roc_auc_score(y_test, test_cal)
brier = brier_score_loss(y_test, test_cal)

print(f"Test Accuracy: {acc:.3f} | ROC AUC: {auc:.3f} | Brier: {brier:.3f}")

# 11) Package predictor
class CreditRiskPredictor:
    def __init__(self, preprocessor, model, calibrator, qhat):
        self.preprocessor = preprocessor
        self.model = model
        self.calibrator = calibrator
        self.qhat = qhat
    def predict(self, Xdf):
        Z = self.preprocessor.transform(Xdf)
        raw = self.model.predict(Z)
        cal = self.calibrator.transform(raw)
        import numpy as np, pandas as pd
        low  = np.clip(cal - self.qhat, 0, 1)
        high = np.clip(cal + self.qhat, 0, 1)
        band = pd.cut(cal, bins=[0,0.33,0.66,1], labels=["Low","Medium","High"])
        return pd.DataFrame({"raw":raw,"calibrated":cal,"low":low,"high":high,"band":band})

# 12) Split conformal on calibrated scores (validation residuals)
residuals = np.abs(y_val - iso.transform(val_raw))
qhat = np.quantile(residuals, 0.90)  # 90% coverage

predictor = CreditRiskPredictor(ct, model, iso, qhat)
joblib.dump(predictor, "artifacts/credit_risk_predictor.pkl")

# 13) Save feature list
with open("artifacts/feature_list.json","w") as f:
    json.dump({"categorical":categorical_cols, "numerical":numerical_cols}, f, indent=2)

# 14) Key Results (Test Set)
LightGBM (monotonic + isotonic): ROC AUC ≈ 0.780, Brier ≈ 0.135

Comparative accuracy (Accuracy | ROC AUC):
Logistic Regression (0.739 | 0.742), Decision Tree (0.768 | 0.764),
Random Forest (0.800 | 0.780), XGBoost (0.805 | 0.769),
LightGBM (0.804 | 0.770–0.780), SVM (0.753 | 0.753),
KNN (0.812 | 0.736), MLP (0.768 | 0.685).

KS curves & lift/gain: strong early concentration; top 20% captures ~60% bads; top-decile lift > 3.

Profit curve: broad optimum around thresholds 0.22–0.26, enabling ~70–75% approvals with manageable bad rate among approvals (~11–13% in our test setting).

# 15) Business Policy
A three-band strategy derived from the calibrated probabilities:

Approve below lower threshold;

Manual Review in a narrow middle band (income verification / bureau refresh);

Decline or Limit-Reduce above upper threshold.
Thresholds can be tuned monthly on the profit curve under guardrails for minimum approval rate and maximum accepted bad rate.

# Explainability
Global drivers (SHAP): EVER_60_PLUS, recent PAY_0, utilisation ratios, payment-to-bill ratios, and AVG_PAY_STATUS.

Local: water­fall/force plots support adverse-action reason codes.

Dependence “story” views: risk increases sharply when utilisation approaches limit, especially with recent arrears.

# 16) Fairness & Uncertainty
Segment TPR/FPR by sex broadly similar; small-support groups (education/marriage) require monitoring.

Split conformal intervals provide per-case uncertainty; tight for very low risk, wider near decision boundary.

# 17) Saved Artifacts
artifacts/credit_risk_predictor.pkl — packaged predictor (preprocessing + model + calibrator + conformal qhat)

artifacts/feature_list.json — exact feature mapping used at train time

figures/ — ROC/PR, KS, lift/gain, calibration, approval–default, profit curves; SHAP plots.

# 18) Citations (Harvard style)
Balasubramanian, V., Ho, S. & Vovk, V. (2014).
Bequé, A. & Lessmann, S. (2017).
Brown, I. & Mues, C. (2012).
Chen, C., Giudici, P., Liu, Y. & Raffinetti, E. (2024).
de Lange, R. et al. (2022).
Finlay, S. (2010).
García García, J. & Rigobon, R. (2024).
Jiang, B. et al. (2023).
Ke, G. et al. (2017).
Kozodoi, N., Lessmann, S. et al. (2024).
Lessmann, S., Baesens, B., Seow, H. & Thomas, L. (2015).
Lundberg, S. & Lee, S.-I. (2017).
Mariscala, J. et al. (2024).
Moscato, V. et al. (2021).
Niculescu-Mizil, A. & Caruana, R. (2005).
Shen, F., Zhao, X. & Kou, G. (2020).
Shen, F. et al. (2022).
Verbraken, T., Bravo, C., Weber, R. & Baesens, B. (2014).
Wang, X. et al. (2022).
Yeh, I.-C. & Lien, C. (2009).

# License
This repository is released under the MIT License. See LICENSE for details.



