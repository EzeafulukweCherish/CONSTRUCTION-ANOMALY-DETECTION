import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Construction Project Anomaly Detector", layout="wide")
st.title("Construction Project Anomaly Detector")
st.caption(
    "An earned-value baseline, a consistency checker that compares reported effort "
    "against verified progress, and a stress test that simulates an adapting anomaly."
)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("Upload the project CSV", type="csv")
if uploaded is None:
    st.info("Upload your project dataset (CSV) to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.success(f"Loaded {len(df):,} projects, {df.shape[1]} columns.")

with st.expander("Raw data preview"):
    st.dataframe(df.head(20))

# ---------------------------------------------------------------------------
# 2. Clean + encode
# ---------------------------------------------------------------------------
d = df.copy()
for c in ["Start_Date", "End_Date"]:
    if c in d.columns:
        d[c] = pd.to_datetime(d[c], errors="coerce")

drop_cols = [c for c in ["Project_ID", "Cost_Overrun", "Start_Date", "End_Date"] if c in d.columns]
d = d.drop(columns=drop_cols)

cat_cols = [c for c in ["Project_Type", "Location", "Weather_Condition"] if c in d.columns]
d = pd.get_dummies(d, columns=cat_cols, dtype=int, drop_first=True)

# ---------------------------------------------------------------------------
# 3. EVM metrics
# ---------------------------------------------------------------------------
d["EV"] = d.Planned_Cost * d.Completion_Percentage / 100
d["CPI"] = d.EV / d.Actual_Cost
d["SPI"] = (d.Completion_Percentage / 100) / (d.Actual_Duration / d.Planned_Duration)
d["cost_ratio"] = d.Actual_Cost / d.Planned_Cost
d["dur_ratio"] = d.Actual_Duration / d.Planned_Duration

# ---------------------------------------------------------------------------
# 4. Rebuild honest labor/material relationships, then simulate fraud
#    (only done once per session so results stay stable while sliders move)
# ---------------------------------------------------------------------------
if "sim_data" not in st.session_state:
    rng = np.random.default_rng(42)
    n = len(d)
    comp = d.Completion_Percentage.clip(lower=5) / 100
    earned = d.Planned_Cost * comp

    d["Labor_Hours"] = earned / 1000 * rng.lognormal(0, 0.15, n)
    d["Material_Usage"] = earned / 10000 * rng.lognormal(0, 0.15, n)

    d["is_fraud"] = 0
    fraud_idx = rng.choice(n, int(0.10 * n), replace=False)
    d.loc[fraud_idx, "is_fraud"] = 1
    d.loc[fraud_idx, "Labor_Hours"] *= rng.uniform(2.0, 4.0, len(fraud_idx))
    d.loc[fraud_idx, "Material_Usage"] *= rng.uniform(2.0, 4.0, len(fraud_idx))

    d["labor_per_progress"] = d.Labor_Hours / comp
    d["material_per_progress"] = d.Material_Usage / comp

    st.session_state.sim_data = d
    st.session_state.rng_seed = 7

d = st.session_state.sim_data
core_cols = ["labor_per_progress", "material_per_progress"]

st.caption(
    "Note: this dataset's raw columns aren't correlated with each other, so labor and "
    "material figures have been rebuilt to scale realistically with verified progress, "
    "and 10% of projects have been secretly tampered with to create known anomaly cases "
    "for testing. This is disclosed here, and would be disclosed in any real presentation."
)

train, test = train_test_split(d, test_size=0.3, stratify=d.is_fraud, random_state=0)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["EVM baseline", "Consistency checker", "Stress test"])

# --- Tab 1: EVM baseline ---
with tab1:
    st.subheader("Earned value management baseline")
    st.write(
        "Standard EVM flags projects by CPI and SPI alone. This is what a plain "
        "cost/schedule review would catch."
    )
    worst_cpi = d.nsmallest(10, "CPI")[
        ["Planned_Cost", "Actual_Cost", "CPI", "SPI", "Completion_Percentage"]
    ]
    st.write("10 worst projects by CPI:")
    st.dataframe(worst_cpi.round(2))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(d.CPI, d.SPI, s=10, alpha=0.4)
    ax.axvline(1, color="red", linestyle="--")
    ax.axhline(1, color="red", linestyle="--")
    ax.set_xlabel("CPI (below 1 = over budget)")
    ax.set_ylabel("SPI (below 1 = behind schedule)")
    ax.set_title("Cost vs. schedule performance, all projects")
    st.pyplot(fig)

    base_top = test.nsmallest(int(0.1 * len(test)), "CPI")
    st.metric(
        "CPI-only baseline: fraud cases caught in test set",
        f"{base_top.is_fraud.sum()} / {test.is_fraud.sum()}",
    )
    st.caption(
        "This anomaly pattern inflates labor and material claims, not cost directly, "
        "so CPI alone mostly misses it. That gap is what the next tab addresses."
    )

# --- Tab 2: Consistency checker ---
with tab2:
    st.subheader("Consistency checker (Isolation Forest)")
    st.write(
        "Instead of cost alone, this model compares reported labor and material use "
        "against verified completion, and flags projects where the two don't add up."
    )

    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=0)
    iso.fit(train[core_cols])
    test_scored = test.copy()
    test_scored["suspicion"] = -iso.score_samples(test_scored[core_cols])

    top_n = st.slider("Show top N most suspicious projects", 5, 50, 15)
    top = test_scored.nlargest(top_n, "suspicion")

    caught = test_scored.nlargest(int(0.1 * len(test_scored)), "suspicion").is_fraud.sum()
    col1, col2 = st.columns(2)
    col1.metric("Model: anomaly cases caught (top 10%)", f"{caught} / {test.is_fraud.sum()}")
    col2.metric("CPI-only baseline (for comparison)", f"{base_top.is_fraud.sum()} / {test.is_fraud.sum()}")

    st.write("Most suspicious projects:")
    show_cols = ["Planned_Cost", "CPI", "SPI", "labor_per_progress",
                 "material_per_progress", "Completion_Percentage", "suspicion", "is_fraud"]
    st.dataframe(top[show_cols].round(2))
    st.caption(
        "`is_anomaly` is only shown here because this is simulated data with known answers, "
        "used to test the model. A real deployment wouldn't have this column."
    )

# --- Tab 3: Stress test ---
with tab3:
    st.subheader("Stress test: an adapting anomaly vs. the guard")
    st.write(
        "A simulated anomaly starts obvious and gets stealthier whenever it's mostly "
        "caught. One guard never updates; the other retrains each round on what it caught "
        "plus a share of manually audited misses."
    )

    n_rounds = st.slider("Number of rounds", 4, 20, 10)
    audit_rate = st.slider("Share of missed anomaly manually audited each round", 0.0, 1.0, 0.2)
    run = st.button("Run stress test")

    if run:
        rng2 = np.random.default_rng(st.session_state.rng_seed)

        def make_fraud_batch(n_batch, stealth):
            idx = rng2.choice(len(d), n_batch, replace=False)
            batch = d.iloc[idx].copy()
            mult = rng2.uniform(4.0 - 3.0 * stealth, 4.0 - 1.5 * stealth, n_batch)
            batch["Labor_Hours"] *= mult
            batch["Material_Usage"] *= mult
            comp = batch.Completion_Percentage.clip(lower=5) / 100
            batch["labor_per_progress"] = batch.Labor_Hours / comp
            batch["material_per_progress"] = batch.Material_Usage / comp
            return batch

        static_guard = IsolationForest(n_estimators=200, contamination=0.1, random_state=0)
        static_guard.fit(train[core_cols])
        static_cutoff = np.quantile(-static_guard.score_samples(train[core_cols]), 0.9)

        retrain_train = train.copy()
        retrain_guard = RandomForestClassifier(n_estimators=200, random_state=0)
        retrain_guard.fit(retrain_train[core_cols], retrain_train.is_fraud)

        stealth = 0.0
        history = []
        progress = st.progress(0)
        for rnd in range(n_rounds):
            attack = make_fraud_batch(50, stealth)

            static_scores = -static_guard.score_samples(attack[core_cols])
            static_caught = (static_scores > static_cutoff).sum()

            retrain_pred = retrain_guard.predict(attack[core_cols])
            retrain_caught = retrain_pred.sum()

            history.append((rnd, stealth, static_caught / 50, retrain_caught / 50))

            caught_rows = attack[retrain_pred == 1].copy()
            caught_rows["is_fraud"] = 1
            missed = attack[retrain_pred == 0].copy()
            if len(missed) > 0 and audit_rate > 0:
                audited = missed.sample(frac=audit_rate, random_state=rnd)
                audited["is_fraud"] = 1
            else:
                audited = missed.iloc[0:0]

            retrain_train = pd.concat([retrain_train, caught_rows, audited])
            retrain_guard = RandomForestClassifier(n_estimators=200, random_state=0)
            retrain_guard.fit(retrain_train[core_cols], retrain_train.is_fraud)

            if max(static_caught, retrain_caught) / 50 > 0.5:
                stealth = min(1.0, stealth + 0.15)

            progress.progress((rnd + 1) / n_rounds)

        rounds, stealths, static_rates, retrain_rates = zip(*history)

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(rounds, static_rates, "o-", color="crimson", label="Static guard")
        ax1.plot(rounds, retrain_rates, "o-", color="seagreen", label="Retraining guard")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Catch rate")
        ax1.set_ylim(0, 1)
        ax1.legend(loc="upper right")
        ax2 = ax1.twinx()
        ax2.plot(rounds, stealths, "--", color="steelblue", alpha=0.5, label="Anomaly stealth")
        ax2.set_ylabel("Anomaly stealth", color="steelblue")
        ax2.set_ylim(0, 1)
        st.pyplot(fig)

        result_df = pd.DataFrame(history, columns=["Round", "Stealth", "Static catch rate", "Retrain catch rate"])
        st.dataframe(result_df.round(2))

        st.caption(
            "Both guards typically degrade as the anomaly gets stealthier. Retraining "
            "on caught cases plus a share of audited misses may or may not close that gap "
            "meaningfully, which is itself a finding worth discussing: detectors can plateau "
            "against an adapting attacker even when they're allowed to retrain."
        )
