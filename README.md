# Masters-Dissertation
Risk-Calibrated & Explainable Credit Card Default Prediction

Monotonic LightGBM + Isotonic Calibration + Split Conformal Prediction

End-to-end reproducible pipeline on the Taiwan Default of Credit Card Clients dataset: data prep, feature engineering, model training, calibration, explainability (SHAP), business decisioning (profit & approval/default trade-offs), fairness checks, and packaging for scoring.

1) Project goals

Predict next-month default risk at customer level.

Explain why (global + local) and quantify uncertainty per case.

Convert scores into approve/review/decline actions using profit and risk guardrails.

Deliver a reproducible Python pipeline suitable for validation and monitoring.

2) Repository structure
.
├── README.md
├── environment.yml                # or requirements.txt
├── data/
│   ├── raw/                       # Kaggle dataset (not committed)
│   └── interim/                   # cached/processed artifacts
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training_compare.ipynb
│   ├── 04_calibration_conformal.ipynb
│   ├── 05_explainability_shap.ipynb
│   ├── 06_business_curves_policy.ipynb
│   └── 07_fairness_monitoring.ipynb
├── src/
│   ├── data_prep.py               # load/clean/split/encode/scale
│   ├── features.py                # engineered features
│   ├── models.py                  # model zoo + training utilities
│   ├── calibration.py             # isotonic / platt + reliability plots
│   ├── conformal.py               # split conformal intervals
│   ├── explain.py                 # SHAP helpers
│   ├── business.py                # profit, lift, gain, KS, policy tools
│   ├── fairness.py                # segment TPR/FPR, PSI helpers
│   └── predictor.py               # packaged predictor class
├── outputs/
│   ├── figures/                   # charts (ROC/PR/KS, SHAP, profit, etc.)
│   ├── tables/                    # CSV summaries
│   └── models/
│       ├── credit_risk_predictor.pkl
│       └── feature_list.json
└── Makefile                       # shortcuts to run pipeline


Keep raw data out of version control; place under data/raw/.

3) Data

Source: Kaggle – Default of Credit Card Clients (Taiwan), 30,000 rows

Target: default.payment.next.month (1 = default next month)

Time window: 6 months of bills & payments, plus demographics & limit

Place the CSV (usually UCI_Credit_Card.csv) into data/raw/.

4) Environment
Option A: Conda
conda env create -f environment.yml
conda activate credit-risk

Option B: pip
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt


Key libraries: pandas, numpy, scikit-learn, lightgbm, xgboost, shap, matplotlib, seaborn, joblib.

5) One-shot quick start

Run the full pipeline end-to-end and write artifacts to outputs/:

make all


Typical Makefile targets:

all: prep features train calibrate explain business

prep:
	python -m src.data_prep --input data/raw/UCI_Credit_Card.csv --out data/interim/clean.parquet

features:
	python -m src.features --input data/interim/clean.parquet --out data/interim/features.parquet

train:
	python -m src.models --input data/interim/features.parquet --out outputs/models

calibrate:
	python -m src.calibration --in-model outputs/models/lgb.pkl --valid data/interim/valid.parquet --out outputs/models

explain:
	python -m src.explain --in-model outputs/models/credit_risk_predictor.pkl --test data/interim/test.parquet --figdir outputs/figures

business:
	python -m src.business --pred outputs/tables/test_predictions.csv --figdir outputs/figures --tabledir outputs/tables


Prefer notebooks for exploratory work and the Makefile/scripts for reproducible runs.

6) Pipeline overview
6.1 Data preparation

Type fixes, label inspection, outlier winsorisation (bills & payments).

Train/validation/test split: 70/15/15 (stratified).

Class imbalance handled via class weights (default) and optional SMOTE for non-tree models.

6.2 Feature engineering

Utilisation: UTILIZATION_[1..6] = BILL_AMTi / LIMIT_BAL

Payment ratio: PAY_RATIO_[1..6] = PAY_AMTi / BILL_AMTi (safe division)

Delinquency: EVER_60_PLUS, AVG_PAY_STATUS, PAY_TREND

Payment behaviour: FULL_PAYMENT_MONTHS, TOTAL_PAY_RATIO

Demographic grouping: AGE_GROUP, EDUCATION_GROUP

6.3 Preprocessing

ColumnTransformer: StandardScaler for numeric; OneHotEncoder for categorical.

Unified sklearn Pipeline for each model.

6.4 Model suite

Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, SVM (RBF), KNN, MLP.

Tuning with compact grids; stratified CV; class_weight="balanced" where applicable.

6.5 Calibration & uncertainty

Isotonic Regression on validation set to calibrate probabilities.

Split Conformal Prediction to produce per-case risk intervals (e.g., 90% coverage).

6.6 Evaluation

ROC AUC, PR AUC, KS, accuracy, classification report.

Calibration curve & Brier score.

Lift/Gain, Profit curve, Approval vs Default trade-off.

Fairness slices: TPR/FPR by sex, education, marriage; PSI.

6.7 Explainability

SHAP global (bar + beeswarm), local (waterfall/force), dependence scatter.

Reason codes exported per applicant (top positive contributors).

6.8 Packaging

CreditRiskPredictor class:

.predict(X) → raw score, calibrated probability, conformal lower/upper, risk band.

Model bundle: credit_risk_predictor.pkl stored under outputs/models/.

7) Reproducible examples (snippets)
7.1 Train the champion (monotonic LightGBM)
import lightgbm as lgb
from joblib import dump
# build monotone_constraints list aligned to feature order
params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=64,
    max_depth=10,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    scale_pos_weight=2,
    monotone_constraints=monotone_constraints,
    random_state=42,
)
lgb_model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=1000,
                      callbacks=[lgb.early_stopping(50)])
dump(lgb_model, "outputs/models/lgb.pkl")

7.2 Calibrate + package predictor
from sklearn.isotonic import IsotonicRegression
from joblib import dump, load
from src.predictor import CreditRiskPredictor

lgb_model = load("outputs/models/lgb.pkl")
val_raw = lgb_model.predict(X_val)
iso = IsotonicRegression(out_of_bounds="clip").fit(val_raw, y_val)

residuals = abs(y_val - iso.transform(val_raw))
qhat = np.quantile(residuals, 0.9)

predictor = CreditRiskPredictor(lgb_model, iso, qhat)
dump(predictor, "outputs/models/credit_risk_predictor.pkl")

7.3 Score new data
from joblib import load
predictor = load("outputs/models/credit_risk_predictor.pkl")
scores = predictor.predict(X_new)  # columns aligned to training feature list

8) Key figures generated

ROC & PR curves for all models

KS curves and KS table

Accuracy + ROC AUC combined chart

Calibration plots (pre/post isotonic) + Brier score

Gain & Lift charts

Profit curve and optimal threshold with guardrails

Approval vs Default trade-off (operating thresholds)

SHAP: global bar, beeswarm, dependence, local waterfall/force

Risk-band distributions (e.g., utilisation by band)

All saved to outputs/figures/.

9) Decision policy (example)

Two thresholds on calibrated probability:

Approve: p < t_low

Manual review: t_low ≤ p < t_high

Decline/limit-reduce: p ≥ t_high

Choose (t_low, t_high) on the profit curve, constrained by a minimum approval rate and maximum bad rate among approvals.

10) Fairness & monitoring

Report TPR/FPR by segment (with minimum-support rules).

Track PSI monthly for top features and the score.

Refresh calibration quarterly; retrain on stability/calibration triggers.

11) Results summary (from the included run)

Strong learners (LightGBM/XGBoost/Random Forest) outperform linear and single-tree baselines on ranking metrics.

Calibrated monotonic LightGBM: AUROC ~0.78, Brier ~0.135 on test; well-behaved reliability.

Gain/Lift: top 20% captures ~60% of defaulters; top-decile lift > 3.

SHAP: repayment status, EVER_60_PLUS, utilisation, and payment-to-bill ratios dominate risk; explanations align with domain sense.

12) Reuse in other contexts

Swap in any tabular credit dataset with similar behavioural history.

Keep the feature contracts and monotonic signs consistent for safe reuse.

Regenerate thresholds using the local profit/cost matrix.

13) License

Specify your chosen license here (e.g., MIT). Add LICENSE to the repo.

14) How to cite

If this repository is used in academic or commercial work, please cite the project and the original dataset provider (Taiwan Default of Credit Card Clients).

15) Acknowledgements

Dataset: Default of Credit Card Clients (Taiwan).

Libraries: scikit-learn, lightgbm, xgboost, shap.
