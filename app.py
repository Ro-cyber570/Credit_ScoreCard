import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Risk Scorecard", layout="wide")
st.title("🏦 Credit Risk Scorecard & Model Validation")
st.caption("Upload your loan dataset to train a PD model, generate a scorecard, and run model validation.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    test_size    = st.slider("Test Set Size", 0.1, 0.4, 0.3, 0.05)
    pdo          = st.number_input("PDO (Points to Double Odds)", value=20)
    base_score   = st.number_input("Base Score", value=600)
    base_odds    = st.number_input("Base Odds", value=50)
    n_bins       = st.slider("WoE Bins", 5, 20, 10)
    run_button   = st.button("▶ Run Analysis", type="primary", use_container_width=True)

# ── Features ──────────────────────────────────────────────────────────────────
TARGET   = "SeriousDlqin2yrs"
FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
]

# ── Helper functions ──────────────────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df["MonthlyIncome"]    = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].median())
    return df


def calculate_woe_iv(dataset, feature, target_col, q=10):
    temp = dataset[[feature, target_col]].copy()
    try:
        temp["bin"] = pd.qcut(temp[feature], q=q, duplicates="drop")
    except Exception:
        temp["bin"] = pd.cut(temp[feature], bins=5)

    grouped = temp.groupby("bin", observed=False)[target_col].agg(["count", "sum"])
    grouped.rename(columns={"count": "Total", "sum": "Bad"}, inplace=True)
    grouped["Good"] = grouped["Total"] - grouped["Bad"]

    total_goods = grouped["Good"].sum()
    total_bads  = grouped["Bad"].sum()

    grouped["Dist_Good"] = grouped["Good"] / total_goods
    grouped["Dist_Bad"]  = grouped["Bad"]  / total_bads
    grouped["WoE"]       = np.log((grouped["Dist_Good"] + 1e-4) / (grouped["Dist_Bad"] + 1e-4))
    grouped["IV"]        = (grouped["Dist_Good"] - grouped["Dist_Bad"]) * grouped["WoE"]

    return grouped.reset_index()


def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.linspace(0, 1000, buckets + 1)
    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual,   bins=breakpoints)[0] / len(actual)
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)
    return float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))


def psi_label(psi):
    if psi < 0.1:
        return "🟢 Stable"
    elif psi < 0.25:
        return "🟡 Minor Drift"
    else:
        return "🔴 Significant Drift"


def iv_label(iv):
    if iv < 0.02:   return "Useless"
    elif iv < 0.1:  return "Weak"
    elif iv < 0.3:  return "Medium"
    else:           return "Strong"


# ── File Upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload `cs-training.csv`", type="csv")

if uploaded is None:
    st.info("👆 Upload your dataset to get started. Expected columns: `SeriousDlqin2yrs`, `RevolvingUtilizationOfUnsecuredLines`, `age`, `DebtRatio`, `MonthlyIncome`, `NumberOfOpenCreditLinesAndLoans`.")
    st.stop()

df_raw = pd.read_csv(uploaded)
df     = preprocess(df_raw)

st.success(f"Dataset loaded — {len(df):,} rows, {df[TARGET].mean()*100:.1f}% default rate")

with st.expander("Preview Data"):
    st.dataframe(df[FEATURES + [TARGET]].head(10), use_container_width=True)

# ── Run on button click ───────────────────────────────────────────────────────
if not run_button:
    st.info("Configure settings in the sidebar and click **▶ Run Analysis**.")
    st.stop()

# ── Split ─────────────────────────────────────────────────────────────────────
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

train_full = pd.concat([X_train, y_train], axis=1)

# ── Section 1: WoE / IV ───────────────────────────────────────────────────────
st.subheader("1️⃣ Feature Selection — Weight of Evidence & Information Value")

iv_summary = {}
woe_maps   = {}

for feat in FEATURES:
    woe_df = calculate_woe_iv(train_full, feat, TARGET, q=n_bins)
    woe_maps[feat]   = woe_df
    iv_summary[feat] = woe_df["IV"].sum()

iv_df = (
    pd.DataFrame.from_dict(iv_summary, orient="index", columns=["IV"])
    .sort_values("IV", ascending=False)
    .reset_index()
    .rename(columns={"index": "Feature"})
)
iv_df["Predictive Power"] = iv_df["IV"].apply(iv_label)

col1, col2 = st.columns([1, 1])
with col1:
    st.dataframe(iv_df.style.format({"IV": "{:.4f}"}), use_container_width=True)

with col2:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(iv_df["Feature"], iv_df["IV"], color="#2563eb")
    ax.axvline(0.1, color="orange", linestyle="--", label="Weak threshold (0.1)")
    ax.axvline(0.3, color="red",    linestyle="--", label="Strong threshold (0.3)")
    ax.set_xlabel("Information Value")
    ax.set_title("IV by Feature")
    ax.legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with st.expander("🔍 WoE Bin Detail (click to expand)"):
    selected_feat = st.selectbox("Select feature", FEATURES)
    st.dataframe(
        woe_maps[selected_feat].style.format({
            "WoE": "{:.4f}", "IV": "{:.4f}",
            "Dist_Good": "{:.4f}", "Dist_Bad": "{:.4f}"
        }),
        use_container_width=True
    )

# ── Section 2: Model Training ─────────────────────────────────────────────────
st.subheader("2️⃣ Model Training — Logistic Regression PD Model")

with st.spinner("Training model…"):
    model  = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    probs  = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, probs)

# ── Section 3: Scorecard Scaling ──────────────────────────────────────────────
factor    = pdo / np.log(2)
offset    = base_score - (factor * np.log(base_odds))
log_odds  = model.decision_function(X_test)
scores    = offset - (factor * log_odds)

train_log_odds = model.decision_function(X_train)
train_scores   = offset - (factor * train_log_odds)

st.subheader("3️⃣ Scorecard — Score Distribution")

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(scores[y_test == 0], bins=50, color="green", alpha=0.6, label="Good", density=True)
ax1.hist(scores[y_test == 1], bins=50, color="red",   alpha=0.6, label="Bad",  density=True)
ax1.set_title("Credit Score Distribution (Test Set)")
ax1.set_xlabel("Score")
ax1.legend()

fpr, tpr, _ = roc_curve(y_test, probs)
ax2.plot(fpr, tpr, color="#2563eb", label=f"AUC = {auc:.3f}")
ax2.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax2.set_title("ROC Curve")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()

plt.tight_layout()
st.pyplot(fig2)
plt.close()

# ── Section 4: Validation Dashboard ──────────────────────────────────────────
st.subheader("4️⃣ Model Validation — Risk Metrics")

gini      = 2 * auc - 1
ks_stat   = max(tpr - fpr)
psi_value = calculate_psi(train_scores, scores)

m1, m2, m3, m4 = st.columns(4)
m1.metric("AUC",   f"{auc:.4f}")
m2.metric("Gini",  f"{gini:.4f}")
m3.metric("KS Stat", f"{ks_stat:.4f}")
m4.metric("PSI",   f"{psi_value:.4f}", delta=psi_label(psi_value), delta_color="off")

# ── Section 5: Model Risk Findings ───────────────────────────────────────────
st.subheader("5️⃣ Model Risk Findings (SR 11-7 Style)")

findings = []

if gini >= 0.3:
    findings.append(("✅ Discriminatory Power", f"Gini of **{gini:.3f}** indicates acceptable model discrimination (threshold: 0.30)."))
else:
    findings.append(("⚠️ Discriminatory Power", f"Gini of **{gini:.3f}** is below the acceptable threshold of 0.30. Model may lack sufficient discrimination."))

if psi_value < 0.1:
    findings.append(("✅ Population Stability", f"PSI of **{psi_value:.4f}** — model is stable. No recalibration required."))
elif psi_value < 0.25:
    findings.append(("⚠️ Population Stability", f"PSI of **{psi_value:.4f}** — minor drift detected. Monitor closely and consider recalibration."))
else:
    findings.append(("🚨 Population Stability", f"PSI of **{psi_value:.4f}** — significant drift detected. **Immediate recalibration recommended** per SR 11-7 guidelines."))

weak_iv = iv_df[iv_df["IV"] < 0.02]["Feature"].tolist()
if weak_iv:
    findings.append(("⚠️ Feature Relevance", f"Features with IV < 0.02 (useless): **{', '.join(weak_iv)}**. Consider removing from production model."))
else:
    findings.append(("✅ Feature Relevance", "All features pass minimum IV threshold (>0.02)."))

for title, body in findings:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(body)

st.caption("This dashboard is for analytical purposes. All model risk findings should be reviewed by a qualified Model Risk Management team.")