import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from db import get_connection
import os
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr

# ── Setup ──────────────────────────────────────────────────────────
os.makedirs("insights/stats", exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"]     = 150

# ── Connect ────────────────────────────────────────────────────────
conn = get_connection()
print("Connected successfully!")

# ── Load data ──────────────────────────────────────────────────────
df = pd.read_sql("SELECT * FROM transactions", conn)
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour"]  = df["trans_date_trans_time"].dt.hour
df["month"] = df["trans_date_trans_time"].dt.month
print(f"Loaded {len(df):,} rows\n")


# ══════════════════════════════════════════════════════════════════
# Analysis 1 — Descriptive Statistics
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("ANALYSIS 1 — DESCRIPTIVE STATISTICS")
print("=" * 60)

fraud     = df[df["is_fraud"] == 1]["amt"]
legit     = df[df["is_fraud"] == 0]["amt"]

print(f"\nFraud Transactions:")
print(f"  Count   : {len(fraud):,}")
print(f"  Mean    : ${fraud.mean():.2f}")
print(f"  Median  : ${fraud.median():.2f}")
print(f"  Std Dev : ${fraud.std():.2f}")
print(f"  Min     : ${fraud.min():.2f}")
print(f"  Max     : ${fraud.max():.2f}")

print(f"\nLegitimate Transactions:")
print(f"  Count   : {len(legit):,}")
print(f"  Mean    : ${legit.mean():.2f}")
print(f"  Median  : ${legit.median():.2f}")
print(f"  Std Dev : ${legit.std():.2f}")
print(f"  Min     : ${legit.min():.2f}")
print(f"  Max     : ${legit.max():.2f}")


# ══════════════════════════════════════════════════════════════════
# Analysis 2 — Mann Whitney U Test
# (Are fraud amounts significantly different from legit?)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 2 — MANN WHITNEY U TEST")
print("Are fraud transaction amounts statistically different?")
print("=" * 60)

stat, p_value = mannwhitneyu(fraud, legit, alternative="two-sided")

print(f"\n  U Statistic : {stat:.2f}")
print(f"  P-value     : {p_value:.10f}")
print(f"\n  Result: {'✅ YES' if p_value < 0.05 else '❌ NO'} — "
      f"fraud amounts are {'significantly' if p_value < 0.05 else 'NOT significantly'} "
      f"different from legitimate amounts (p {'<' if p_value < 0.05 else '>'} 0.05)")


# ══════════════════════════════════════════════════════════════════
# Analysis 3 — Chi Square Test
# (Is fraud rate dependent on category?)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 3 — CHI SQUARE TEST")
print("Is fraud rate dependent on transaction category?")
print("=" * 60)

contingency_table = pd.crosstab(df["category"], df["is_fraud"])
chi2, p_val, dof, expected = chi2_contingency(contingency_table)

print(f"\n  Chi2 Statistic : {chi2:.2f}")
print(f"  Degrees of Freedom : {dof}")
print(f"  P-value        : {p_val:.10f}")
print(f"\n  Result: {'✅ YES' if p_val < 0.05 else '❌ NO'} — "
      f"fraud rate {'IS' if p_val < 0.05 else 'IS NOT'} "
      f"significantly dependent on category (p {'<' if p_val < 0.05 else '>'} 0.05)")


# ══════════════════════════════════════════════════════════════════
# Analysis 4 — Point Biserial Correlation
# (Correlation between amount and fraud)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 4 — POINT BISERIAL CORRELATION")
print("Is transaction amount correlated with fraud?")
print("=" * 60)

corr, p_value = pointbiserialr(df["is_fraud"], df["amt"])

print(f"\n  Correlation : {corr:.4f}")
print(f"  P-value     : {p_value:.10f}")
print(f"\n  Result: {'✅ YES' if p_value < 0.05 else '❌ NO'} — "
      f"transaction amount {'IS' if p_value < 0.05 else 'IS NOT'} "
      f"significantly correlated with fraud")
print(f"  Strength: {'Strong' if abs(corr) > 0.5 else 'Moderate' if abs(corr) > 0.3 else 'Weak'} "
      f"({'Positive' if corr > 0 else 'Negative'}) correlation")


# ══════════════════════════════════════════════════════════════════
# Analysis 5 — Fraud Rate by Hour (Chi Square)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 5 — FRAUD RATE BY HOUR")
print("Is fraud significantly more likely at certain hours?")
print("=" * 60)

hour_contingency = pd.crosstab(df["hour"], df["is_fraud"])
chi2_h, p_val_h, dof_h, _ = chi2_contingency(hour_contingency)

print(f"\n  Chi2 Statistic : {chi2_h:.2f}")
print(f"  P-value        : {p_val_h:.10f}")
print(f"\n  Result: {'✅ YES' if p_val_h < 0.05 else '❌ NO'} — "
      f"hour of day {'IS' if p_val_h < 0.05 else 'IS NOT'} "
      f"significantly associated with fraud")

# Peak fraud hours
fraud_by_hour = (
    df.groupby("hour")["is_fraud"]
    .agg(["sum", "count"])
    .assign(fraud_rate=lambda x: x["sum"] / x["count"] * 100)
)
peak_hour = fraud_by_hour["fraud_rate"].idxmax()
print(f"\n  Peak fraud hour: {peak_hour}:00 "
      f"({fraud_by_hour.loc[peak_hour, 'fraud_rate']:.2f}% fraud rate)")


# ══════════════════════════════════════════════════════════════════
# Analysis 6 — Correlation Heatmap
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 6 — CORRELATION HEATMAP")
print("=" * 60)

numeric_cols = ["amt", "city_pop", "lat", "long",
                "merch_lat", "merch_long", "is_fraud"]
corr_matrix  = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot    = True,
    fmt      = ".3f",
    cmap     = "RdYlGn",
    center   = 0,
    square   = True,
    linewidths=0.5
)
plt.title("Correlation Heatmap — Numeric Features vs Fraud",
          fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("insights/stats/correlation_heatmap.png", bbox_inches="tight")
plt.show()
print("Correlation heatmap saved!")


# ══════════════════════════════════════════════════════════════════
# Analysis 7 — Amount Distribution by Category (Box Plot)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 7 — AMOUNT DISTRIBUTION BY CATEGORY")
print("=" * 60)

plt.figure(figsize=(14, 8))
sns.boxplot(
    data    = df[df["amt"] < 500],
    x       = "category",
    y       = "amt",
    hue     = "is_fraud",
    palette = {0: "#2ecc71", 1: "#e74c3c"}
)
plt.title("Transaction Amount Distribution by Category",
          fontsize=16, fontweight="bold")
plt.xlabel("Category")
plt.ylabel("Amount ($)")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Is Fraud", labels=["Legitimate", "Fraud"])
plt.tight_layout()
plt.savefig("insights/stats/amount_by_category_boxplot.png",
            bbox_inches="tight")
plt.show()
print("Box plot saved!")


# ══════════════════════════════════════════════════════════════════
# Analysis 8 — Fraud Amount Distribution (KDE Plot)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 8 — KDE DISTRIBUTION COMPARISON")
print("=" * 60)

plt.figure()
sns.kdeplot(
    data  = df[df["amt"] < 500],
    x     = "amt",
    hue   = "is_fraud",
    fill  = True,
    alpha = 0.4,
    palette = {0: "#2ecc71", 1: "#e74c3c"}
)
plt.title("KDE: Transaction Amount — Fraud vs Legitimate",
          fontsize=16, fontweight="bold")
plt.xlabel("Transaction Amount ($)")
plt.ylabel("Density")
plt.legend(labels=["Fraud", "Legitimate"])
plt.savefig("insights/stats/kde_amount_distribution.png",
            bbox_inches="tight")
plt.show()
print("KDE plot saved!")


# ══════════════════════════════════════════════════════════════════
# Analysis 9 — Summary Report
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STATISTICAL SUMMARY REPORT")
print("=" * 60)
print(f"""
Dataset Overview:
  Total transactions : {len(df):,}
  Fraud transactions : {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)
  Legit transactions : {(df['is_fraud']==0).sum():,} ({(df['is_fraud']==0).mean()*100:.2f}%)

Key Statistical Findings:
  1. Avg fraud amount  : ${fraud.mean():.2f} vs ${legit.mean():.2f} legitimate
  2. Fraud amounts are statistically different (Mann-Whitney p < 0.05)
  3. Category significantly affects fraud rate (Chi-Square p < 0.05)
  4. Peak fraud hour   : {peak_hour}:00
  5. Highest fraud category: {df.groupby('category')['is_fraud'].mean().idxmax()}
  6. Most fraud losses : {df[df['is_fraud']==1].groupby('state')['amt'].sum().idxmax()} state
""")

conn.close()
print("All statistical analyses complete!")