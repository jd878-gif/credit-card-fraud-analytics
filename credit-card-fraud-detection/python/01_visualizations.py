import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import os

os.makedirs("Images", exist_ok=True)

# ── Connect to PostgreSQL ──────────────────────────────────────────
conn = psycopg2.connect(
    host     = "localhost",
    database = "credit_card_db",
    user     = "postgres",
    password = "REDACTED",
    port     = "5433"           # ← changed from 5432 to 5433
)

print("Connected to database successfully!")

# ── Load data ──────────────────────────────────────────────────────
df = pd.read_sql("SELECT * FROM transactions", conn)
print(f"Loaded {len(df):,} rows")

# ── Set visual style ───────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"]     = 150


# ══════════════════════════════════════════════════════════════════
# Chart 1 — Fraud vs Legitimate Transactions (Pie Chart)
# ══════════════════════════════════════════════════════════════════
fraud_counts = df["is_fraud"].value_counts()
labels       = ["Legitimate", "Fraud"]
colors       = ["#2ecc71", "#e74c3c"]

plt.figure()
plt.pie(
    fraud_counts,
    labels    = labels,
    colors    = colors,
    autopct   = "%1.1f%%",
    startangle= 90,
    explode   = (0, 0.1)
)
plt.title("Fraud vs Legitimate Transactions", fontsize=16, fontweight="bold")
plt.savefig("images/01_fraud_vs_legit.png", bbox_inches="tight")
plt.show()
print("Chart 1 saved!")


# ══════════════════════════════════════════════════════════════════
# Chart 2 — Total Spending by Category (Bar Chart)
# ══════════════════════════════════════════════════════════════════
category_spend = (
    df.groupby("category")["amt"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure()
sns.barplot(
    data    = category_spend,
    x       = "amt",
    y       = "category",
    palette = "Blues_r"
)
plt.title("Total Spending by Category", fontsize=16, fontweight="bold")
plt.xlabel("Total Amount ($)")
plt.ylabel("Category")
plt.savefig("images/02_spending_by_category.png", bbox_inches="tight")
plt.show()
print("Chart 2 saved!")


# ══════════════════════════════════════════════════════════════════
# Chart 3 — Fraud Rate by Category (Bar Chart)
# ══════════════════════════════════════════════════════════════════
fraud_by_cat = (
    df.groupby("category")
    .agg(
        total = ("is_fraud", "count"),
        fraud = ("is_fraud", "sum")
    )
    .assign(fraud_rate=lambda x: x["fraud"] / x["total"] * 100)
    .sort_values("fraud_rate", ascending=False)
    .reset_index()
)

plt.figure()
sns.barplot(
    data    = fraud_by_cat,
    x       = "fraud_rate",
    y       = "category",
    palette = "Reds_r"
)
plt.title("Fraud Rate by Category (%)", fontsize=16, fontweight="bold")
plt.xlabel("Fraud Rate (%)")
plt.ylabel("Category")
plt.savefig("images/03_fraud_rate_by_category.png", bbox_inches="tight")
plt.show()
print("Chart 3 saved!")


# ══════════════════════════════════════════════════════════════════
# Chart 4 — Transaction Amount Distribution (Histogram)
# ══════════════════════════════════════════════════════════════════
# Chart 4 — Transaction Amount Distribution (Histogram)
plt.figure()
df_chart4 = df[df["amt"] < 500].copy()
df_chart4["is_fraud"] = df_chart4["is_fraud"].astype(str)
sns.histplot(
    data    = df_chart4,
    x       = "amt",
    hue     = "is_fraud",
    bins    = 50,
    palette = {"0": "#2ecc71", "1": "#e74c3c"},
    alpha   = 0.7
)
plt.title("Transaction Amount Distribution: Fraud vs Legitimate",
          fontsize=16, fontweight="bold")
plt.xlabel("Transaction Amount ($)")
plt.ylabel("Count")
plt.legend(labels=["Legitimate", "Fraud"])
plt.savefig("images/04_amount_distribution.png", bbox_inches="tight")
plt.show()
print("Chart 4 saved!")


# ══════════════════════════════════════════════════════════════════
# Chart 5 — Fraud by Hour of Day (Line Chart)
# ══════════════════════════════════════════════════════════════════
df["hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour

fraud_by_hour = (
    df.groupby("hour")
    .agg(
        total = ("is_fraud", "count"),
        fraud = ("is_fraud", "sum")
    )
    .assign(fraud_rate=lambda x: x["fraud"] / x["total"] * 100)
    .reset_index()
)

plt.figure()
sns.lineplot(
    data   = fraud_by_hour,
    x      = "hour",
    y      = "fraud_rate",
    marker = "o",
    color  = "#e74c3c",
    linewidth=2.5
)
plt.title("Fraud Rate by Hour of Day", fontsize=16, fontweight="bold")
plt.xlabel("Hour of Day (0 = Midnight)")
plt.ylabel("Fraud Rate (%)")
plt.xticks(range(0, 24))
plt.savefig("images/05_fraud_by_hour.png", bbox_inches="tight")
plt.show()
print("Chart 5 saved!")


# ══════════════════════════════════════════════════════════════════
# Chart 6 — Monthly Spending Trend (Line Chart)
# ══════════════════════════════════════════════════════════════════
df["month"] = pd.to_datetime(df["trans_date_trans_time"]).dt.to_period("M")

monthly = (
    df.groupby("month")["amt"]
    .sum()
    .reset_index()
)
monthly["month"] = monthly["month"].astype(str)

plt.figure()
sns.lineplot(
    data      = monthly,
    x         = "month",
    y         = "amt",
    marker    = "o",
    color     = "#3498db",
    linewidth = 2.5
)
plt.title("Monthly Spending Trend", fontsize=16, fontweight="bold")
plt.xlabel("Month")
plt.ylabel("Total Amount ($)")
plt.xticks(rotation=45)
plt.savefig("images/06_monthly_trend.png", bbox_inches="tight")
plt.show()
print("Chart 6 saved!")


# ══════════════════════════════════════════════════════════════════
# Chart 7 — Fraud Amount by Gender (Box Plot)
# ══════════════════════════════════════════════════════════════════
fraud_only = df[df["is_fraud"] == 1]

plt.figure()
sns.boxplot(
    data    = fraud_only[fraud_only["amt"] < 1000],
    x       = "gender",
    y       = "amt",
    palette = {"M": "#3498db", "F": "#e91e8c"}
)
plt.title("Fraud Transaction Amounts by Gender",
          fontsize=16, fontweight="bold")
plt.xlabel("Gender")
plt.ylabel("Transaction Amount ($)")
plt.savefig("images/07_fraud_by_gender.png", bbox_inches="tight")
plt.show()
print("Chart 7 saved!")


# ══════════════════════════════════════════════════════════════════
# Chart 8 — Top 10 States by Fraud Amount (Bar Chart)
# ══════════════════════════════════════════════════════════════════
state_fraud = (
    df[df["is_fraud"] == 1]
    .groupby("state")["amt"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plt.figure()
sns.barplot(
    data    = state_fraud,
    x       = "amt",
    y       = "state",
    palette = "OrRd_r"
)
plt.title("Top 10 States by Fraud Amount Lost",
          fontsize=16, fontweight="bold")
plt.xlabel("Total Fraud Amount ($)")
plt.ylabel("State")
plt.savefig("images/08_fraud_by_state.png", bbox_inches="tight")
plt.show()
print("Chart 8 saved!")


# ── Close connection ───────────────────────────────────────────────
conn.close()
print("\nAll 8 charts saved to /images folder!")