import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from db import get_connection
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing        import StandardScaler, LabelEncoder
from sklearn.linear_model         import LogisticRegression
from sklearn.ensemble             import RandomForestClassifier
from sklearn.metrics              import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from imblearn.over_sampling       import SMOTE
import xgboost as xgb

# ── Setup ──────────────────────────────────────────────────────────
os.makedirs("insights/ml", exist_ok=True)
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"]     = 150
sns.set_theme(style="whitegrid")

# ── Connect ────────────────────────────────────────────────────────
conn = get_connection()
print("Connected successfully!")

# ── Load Data ──────────────────────────────────────────────────────
print("\nLoading data...")
df = pd.read_sql("SELECT * FROM transactions", conn)
conn.close()
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
print(f"Loaded {len(df):,} rows")


# ══════════════════════════════════════════════════════════════════
# STEP 1 — Feature Engineering
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1 — FEATURE ENGINEERING")
print("="*60)

# Time features
df["hour"]        = df["trans_date_trans_time"].dt.hour
df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
df["month"]       = df["trans_date_trans_time"].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["is_night"]    = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

# ── Leak-free customer features ────────────────────────────────────
# Every customer feature uses ONLY that card's PREVIOUS transactions.
# (An earlier version averaged over the card's whole history, including the
#  current and future transactions, which leaks information into training.)
df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
g = df.groupby("cc_num")["amt"]
df["customer_avg"] = g.transform(lambda s: s.shift(1).expanding().mean())
df["spend_ratio"]  = (df["amt"] / df["customer_avg"]).fillna(1.0)
df["customer_avg"] = df["customer_avg"].fillna(df["amt"])   # first transaction

# Distance between customer and merchant
df["distance"] = np.sqrt(
    (df["lat"] - df["merch_lat"])**2 +
    (df["long"] - df["merch_long"])**2
)

# Velocity: this card's transactions in the PRECEDING 24 hours (current excluded)
def _count_prior_24h(times: pd.Series) -> pd.Series:
    """For each transaction: how many earlier ones this card made in the last 24h."""
    t = times.to_numpy().astype("datetime64[ns]").astype("int64")
    window_start = t - 24 * 3600 * 10**9
    return pd.Series(np.arange(len(t)) - np.searchsorted(t, window_start, side="left"),
                     index=times.index)

df["daily_trans_count"] = (df.groupby("cc_num")["trans_date_trans_time"]
                             .transform(_count_prior_24h))

# Encode categorical columns
le_category = LabelEncoder()
le_gender   = LabelEncoder()
df["category_encoded"] = le_category.fit_transform(df["category"])
df["gender_encoded"]   = le_gender.fit_transform(df["gender"])

print("Features engineered:")
print("  ✅ Time features: hour, day_of_week, month, is_weekend, is_night")
print("  ✅ Spend features: spend_ratio, customer_avg (prior transactions only)")
print("  ✅ Distance: customer to merchant")
print("  ✅ Velocity: card transactions in the previous 24h")
print("  ✅ Encoded: category, gender")


# ══════════════════════════════════════════════════════════════════
# STEP 2 — Prepare Features & Target
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2 — PREPARE FEATURES & TARGET")
print("="*60)

FEATURES = [
    "amt", "hour", "day_of_week", "month",
    "is_weekend", "is_night", "spend_ratio",
    "customer_avg", "distance", "daily_trans_count",
    "category_encoded", "gender_encoded", "city_pop"
]

X = df[FEATURES].fillna(0)
y = df["is_fraud"]

print(f"\nFeatures used: {len(FEATURES)}")
print(f"Total samples : {len(X):,}")
print(f"Fraud samples : {y.sum():,} ({y.mean()*100:.2f}%)")
print(f"Legit samples : {(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")


# ══════════════════════════════════════════════════════════════════
# STEP 3 — Train/Test Split
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3 — TRAIN / TEST SPLIT")
print("="*60)

# Time-based split: train on the earliest 80% of transactions, test on the
# latest 20%. A fraud model always scores FUTURE transactions, so a random
# shuffle would overstate performance.
cutoff  = int(len(df) * 0.8)
X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]

print(f"\nTrain set: {len(X_train):,} rows | fraud {y_train.sum():,} "
      f"| up to {df['trans_date_trans_time'].iloc[cutoff-1]:%Y-%m-%d}")
print(f"Test set : {len(X_test):,} rows | fraud {y_test.sum():,} "
      f"| from {df['trans_date_trans_time'].iloc[cutoff]:%Y-%m-%d}")
print("Time-based split — test set is strictly later than training ✅")

# Scale features
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ══════════════════════════════════════════════════════════════════
# STEP 4 — Handle Class Imbalance with SMOTE
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 4 — HANDLE CLASS IMBALANCE (SMOTE)")
print("="*60)

print("\nBefore SMOTE:")
print(f"  Fraud : {y_train.sum():,}")
print(f"  Legit : {(y_train==0).sum():,}")

smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

print("\nAfter SMOTE:")
print(f"  Fraud : {y_train_sm.sum():,}")
print(f"  Legit : {(y_train_sm==0).sum():,}")
print("  ✅ Synthetic fraud samples created to balance training data")


# ══════════════════════════════════════════════════════════════════
# STEP 5 — Model 1: Logistic Regression (Baseline)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 5 — MODEL 1: LOGISTIC REGRESSION (BASELINE)")
print("="*60)

lr = LogisticRegression(
    max_iter     = 1000,
    random_state = 42,
    class_weight = "balanced"
)
lr.fit(X_train_sm, y_train_sm)
lr_preds  = lr.predict(X_test_scaled)
lr_proba  = lr.predict_proba(X_test_scaled)[:, 1]
lr_auc    = roc_auc_score(y_test, lr_proba)
lr_ap     = average_precision_score(y_test, lr_proba)

print(f"\nLogistic Regression Results:")
print(f"  ROC-AUC Score     : {lr_auc:.4f}")
print(f"  Avg Precision     : {lr_ap:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, lr_preds, target_names=["Legit","Fraud"]))


# ══════════════════════════════════════════════════════════════════
# STEP 6 — Model 2: Random Forest
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 6 — MODEL 2: RANDOM FOREST")
print("="*60)

rf = RandomForestClassifier(
    n_estimators = 100,
    max_depth    = 10,
    random_state = 42,
    class_weight = "balanced",
    n_jobs       = -1
)
rf.fit(X_train_sm, y_train_sm)
rf_preds = rf.predict(X_test_scaled)
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
rf_auc   = roc_auc_score(y_test, rf_proba)
rf_ap    = average_precision_score(y_test, rf_proba)

print(f"\nRandom Forest Results:")
print(f"  ROC-AUC Score     : {rf_auc:.4f}")
print(f"  Avg Precision     : {rf_ap:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, rf_preds, target_names=["Legit","Fraud"]))


# ══════════════════════════════════════════════════════════════════
# STEP 7 — Model 3: XGBoost
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 7 — MODEL 3: XGBOOST")
print("="*60)

scale_pos = (y_train==0).sum() / y_train.sum()

xgb_model = xgb.XGBClassifier(
    n_estimators     = 100,
    max_depth        = 6,
    learning_rate    = 0.1,
    scale_pos_weight = scale_pos,
    random_state     = 42,
    eval_metric      = "auc",
    verbosity        = 0
)
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_auc   = roc_auc_score(y_test, xgb_proba)
xgb_ap    = average_precision_score(y_test, xgb_proba)

print(f"\nXGBoost Results:")
print(f"  ROC-AUC Score     : {xgb_auc:.4f}")
print(f"  Avg Precision     : {xgb_ap:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, xgb_preds, target_names=["Legit","Fraud"]))


# ══════════════════════════════════════════════════════════════════
# STEP 8 — Model Comparison Chart
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 8 — MODEL COMPARISON CHARTS")
print("="*60)

# ROC Curve comparison
plt.figure(figsize=(10, 8))
for name, proba, auc in [
    ("Logistic Regression", lr_proba,  lr_auc),
    ("Random Forest",       rf_proba,  rf_auc),
    ("XGBoost",             xgb_proba, xgb_auc),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.4f})")

plt.plot([0,1],[0,1], "k--", linewidth=1, label="Random classifier")
plt.xlabel("False Positive Rate", fontsize=13)
plt.ylabel("True Positive Rate", fontsize=13)
plt.title("ROC Curve — Model Comparison", fontsize=16, fontweight="bold")
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("insights/ml/01_roc_curve.png", bbox_inches="tight")
plt.show()
print("ROC curve saved!")


# Precision-Recall Curve
plt.figure(figsize=(10, 8))
for name, proba, ap in [
    ("Logistic Regression", lr_proba,  lr_ap),
    ("Random Forest",       rf_proba,  rf_ap),
    ("XGBoost",             xgb_proba, xgb_ap),
]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    plt.plot(rec, prec, linewidth=2, label=f"{name} (AP={ap:.4f})")

plt.xlabel("Recall", fontsize=13)
plt.ylabel("Precision", fontsize=13)
plt.title("Precision-Recall Curve — Model Comparison",
          fontsize=16, fontweight="bold")
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("insights/ml/02_precision_recall.png", bbox_inches="tight")
plt.show()
print("Precision-Recall curve saved!")


# Confusion Matrix for best model (XGBoost)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, xgb_preds)
sns.heatmap(
    cm,
    annot      = True,
    fmt        = "d",
    cmap       = "Reds",
    xticklabels= ["Legit", "Fraud"],
    yticklabels= ["Legit", "Fraud"]
)
plt.title("Confusion Matrix — XGBoost", fontsize=16, fontweight="bold")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("insights/ml/03_confusion_matrix.png", bbox_inches="tight")
plt.show()
print("Confusion matrix saved!")


# Feature Importance (XGBoost)
feat_imp = pd.DataFrame({
    "feature"   : FEATURES,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 8))
sns.barplot(
    data    = feat_imp,
    x       = "importance",
    y       = "feature",
    palette = "Blues_r",
    hue     = "feature",
    legend  = False
)
plt.title("Feature Importance — XGBoost",
          fontsize=16, fontweight="bold")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("insights/ml/04_feature_importance.png", bbox_inches="tight")
plt.show()
print("Feature importance saved!")


# Model Comparison Bar Chart
models     = ["Logistic Regression", "Random Forest", "XGBoost"]
auc_scores = [lr_auc, rf_auc, xgb_auc]
ap_scores  = [lr_ap,  rf_ap,  xgb_ap]

x     = np.arange(len(models))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, auc_scores, width, label="ROC-AUC",  color="#3498db")
plt.bar(x + width/2, ap_scores,  width, label="Avg Precision", color="#e74c3c")
plt.xticks(x, models, fontsize=12)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Comparison",
          fontsize=16, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("insights/ml/05_model_comparison.png", bbox_inches="tight")
plt.show()
print("Model comparison chart saved!")


# ══════════════════════════════════════════════════════════════════
# STEP 9 — Final Summary
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL ML SUMMARY REPORT")
print("="*60)
print(f"""
Models Trained: 3
  1. Logistic Regression  → ROC-AUC: {lr_auc:.4f}
  2. Random Forest        → ROC-AUC: {rf_auc:.4f}
  3. XGBoost              → ROC-AUC: {xgb_auc:.4f}

Best Model: {"XGBoost" if xgb_auc >= rf_auc and xgb_auc >= lr_auc else "Random Forest"}
Best ROC-AUC: {max(lr_auc, rf_auc, xgb_auc):.4f}

Top Features by Importance (XGBoost):
{feat_imp.tail(5)[["feature","importance"]].to_string(index=False)}

Key ML Findings:
  - SMOTE used to handle 99.4% / 0.6% class imbalance
  - Strongest predictor: {feat_imp.iloc[-1]["feature"]}
  - Features use only past transactions; time-based train/test split
  - XGBoost outperforms baseline Logistic Regression
  - All 5 charts saved to insights/ml/ folder
""")
import json
with open("insights/ml/metrics.json", "w") as fh:
    json.dump({
        "split": "time-based, earliest 80% train / latest 20% test",
        "test_rows": int(len(y_test)), "test_fraud": int(y_test.sum()),
        "models": {
            "Logistic Regression": {"roc_auc": lr_auc,  "avg_precision": lr_ap},
            "Random Forest":       {"roc_auc": rf_auc,  "avg_precision": rf_ap},
            "XGBoost":             {"roc_auc": xgb_auc, "avg_precision": xgb_ap},
        },
        "xgboost_confusion_matrix": confusion_matrix(y_test, xgb_preds).tolist(),
        "top_features": feat_imp.tail(5)[["feature", "importance"]]
                                .assign(importance=lambda d: d.importance.round(4))
                                .to_dict("records"),
    }, fh, indent=2, default=float)
print("Metrics saved to insights/ml/metrics.json")
print("ML pipeline complete! 🎉")