
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, pearsonr, spearmanr
import statsmodels.api as sm
import io, zipfile
from matplotlib.backends.backend_pdf import PdfPages

st.set_page_config(page_title="Bosherston Lily Ponds Water Chemistry", layout="wide")

# ---------------- Header ----------------
st.title("Bosherston Lily Ponds Water Chemistry Assesment")
st.markdown(
    "##### Prepared by **Powell A. Ndlovu** — Swansea University, **Chevening Scholar 2025–2026**"
)

# ---------------- Data ----------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    for col in ["Temperature", "pH", "Conductivity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df = load_data("pond_data.csv")

st.sidebar.header("Controls")
ponds = sorted(df["Ponds"].dropna().unique().tolist())
default_ponds = ponds
sel_ponds = st.sidebar.multiselect("Select ponds/sites", ponds, default=ponds)
sel_vars = st.sidebar.multiselect("Variables", ["Temperature", "pH", "Conductivity"], default=["Temperature","pH","Conductivity"])

df = df[df["Ponds"].isin(sel_ponds)].copy()
df = df.dropna(subset=sel_vars)

# Collect figures for export
_figs = []
_monthly_cache = None
_yearly_cache = None

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5, tab7 = st.tabs(["Overview", "Seasonal (Monthly)", "Long-term (Yearly)", "Statistics", "Correlations", "Inflow vs Western Arm"])

with tab1:
    st.markdown("**Dataset preview**")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("**Counts by site**")
    st.dataframe(df.groupby("Ponds").size().rename("n").reset_index())

with tab2:
    st.subheader("Monthly climatology (Jan–Dec) by site")
    # Compute monthly means
    monthly = (df.groupby(["Month", "Ponds"], as_index=False)
                 .mean(numeric_only=True)
                 .sort_values("Month"))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for var in sel_vars:
        fig, ax = plt.subplots(figsize=(9,4.5))
        for p in sel_ponds:
            sub = monthly[monthly["Ponds"] == p].set_index("Month").reindex(range(1,13)).reset_index()
            ax.plot(sub["Month"], sub[var], marker="o", label=p)
        ax.set_xticks(range(1,13), month_labels, rotation=0)
        ax.set_xlabel("Month")
        ax.set_ylabel(var if var!="Conductivity" else "Conductivity (µS/cm)")
        ax.set_title(f"Monthly {var}")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        _figs.append(fig)

with tab3:
    st.subheader("Yearly averages by site")
    yearly = df.groupby(["Year","Ponds"]).mean(numeric_only=True).reset_index()
    for var in sel_vars:
        fig, ax = plt.subplots(figsize=(9,4.5))
        for p in sel_ponds:
            sub = yearly[yearly["Ponds"] == p]
            ax.plot(sub["Year"], sub[var], marker="o", label=p)
        ax.set_xlabel("Year")
        ax.set_ylabel(var if var!="Conductivity" else "Conductivity (µS/cm)")
        ax.set_title(f"Long-term {var}")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        _figs.append(fig)

with tab4:
    st.subheader("Kruskal–Wallis tests across sites")
    results = []
    for var in sel_vars:
        groups = [df[df["Ponds"]==p][var].dropna() for p in sel_ponds]
        if len(groups) >= 2 and all(len(g)>0 for g in groups):
            stat, p = kruskal(*groups)
            results.append({"Variable": var, "H": stat, "p-value": p})
    st.dataframe(pd.DataFrame(results), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Box & whisker plots")
    for var in sel_vars:
        fig, ax = plt.subplots(figsize=(9,4.5))
        data = [df[df["Ponds"]==p][var].dropna().values for p in sel_ponds]
        ax.boxplot(data, labels=sel_ponds, showfliers=True)
        ax.set_title(f"{var} distribution by site")
        ax.set_ylabel(var if var!="Conductivity" else "Conductivity (µS/cm)")
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        st.pyplot(fig, use_container_width=True)
        _figs.append(fig)

with tab5:
    st.subheader("Scatter plots with regression lines")
    pairs = [("Temperature", "Conductivity"), ("Temperature", "pH"), ("pH","Conductivity")]
    corr_rows = []
    for x,y in pairs:
        sub = df[[x,y]].dropna()
        # Scatter + OLS line
        X = sm.add_constant(sub[x])
        model = sm.OLS(sub[y], X).fit()
        slope = model.params[x]
        intercept = model.params["const"]
        r2 = model.rsquared
        
        # Correlations
        pear_r, pear_p = pearsonr(sub[x], sub[y])
        spear_r, spear_p = spearmanr(sub[x], sub[y])
        corr_rows.append({"Relationship": f"{y} vs {x}",
                          "Pearson_r": pear_r, "Pearson_p": pear_p,
                          "Spearman_r": spear_r, "Spearman_p": spear_p,
                          "OLS_slope": slope, "OLS_intercept": intercept, "R2": r2})
        # Plot
        fig, ax = plt.subplots(figsize=(7,4))
        ax.scatter(sub[x], sub[y], s=12, alpha=0.6)
        xs = np.linspace(sub[x].min(), sub[x].max(), 100)
        ax.plot(xs, slope*xs + intercept, linewidth=2)
        ax.set_xlabel(x if x!="Conductivity" else "Conductivity (µS/cm)")
        ax.set_ylabel(y if y!="Conductivity" else "Conductivity (µS/cm)")
        ax.set_title(f"{y} vs {x}  (R²={r2:.2f})")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        _figs.append(fig)
    
    st.markdown("**Correlation & regression summary**")
    st.dataframe(pd.DataFrame(corr_rows), use_container_width=True)

tab6, = st.tabs(["Downloads"])
with tab6:
    st.subheader("Download filtered data and figures")
    # Prepare CSV bytes
    csv_all = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download filtered CSV", data=csv_all, file_name="filtered_pond_data.csv", mime="text/csv")
    
    # Monthly and Yearly summaries for current selection
    monthly = (df.groupby(["Month", "Ponds"], as_index=False)
                 .mean(numeric_only=True)
                 .sort_values("Month"))
    yearly = df.groupby(["Year", "Ponds"]).mean(numeric_only=True).reset_index()
    st.download_button("⬇️ Download monthly means (CSV)", data=monthly.to_csv(index=False).encode('utf-8'), file_name="monthly_means.csv", mime="text/csv")
    st.download_button("⬇️ Download yearly means (CSV)", data=yearly.to_csv(index=False).encode('utf-8'), file_name="yearly_means.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Download all figures")
    # Create a multi-page PDF of all figures
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        for f in _figs:
            pdf.savefig(f, bbox_inches="tight")
    pdf_buffer.seek(0)
    st.download_button("📄 Download multi-page PDF of figures", data=pdf_buffer, file_name="bosherston_figures.pdf", mime="application/pdf")

    # Create a ZIP of individual PNGs
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, f in enumerate(_figs, start=1):
            img_bytes = io.BytesIO()
            f.savefig(img_bytes, format="png", dpi=200, bbox_inches="tight")
            img_bytes.seek(0)
            zf.writestr(f"figure_{i:02d}.png", img_bytes.read())
    zip_buffer.seek(0)
    st.download_button("🗂️ Download all figures as PNG (ZIP)", data=zip_buffer, file_name="bosherston_figures_png.zip", mime="application/zip")


with tab7:
    st.subheader("Western Arm vs Western Arm Spring Inflow")
    st.markdown("This tab aligns **Western_Arm_Spring_Inflow** and **Western_Arm** by Year+Month to test whether inflow explains Western Arm chemistry.")

    # Filter two sites
    inflow_df = df[df["Ponds"] == "Western_Arm_Spring_Inflow"].copy()
    arm_df = df[df["Ponds"] == "Western_Arm"].copy()

    # Monthly alignment by Year+Month
    inflow_m = inflow_df.groupby(["Year","Month"], as_index=False).mean(numeric_only=True)
    arm_m = arm_df.groupby(["Year","Month"], as_index=False).mean(numeric_only=True)
    merged = pd.merge(inflow_m, arm_m, on=["Year","Month"], suffixes=("_Inflow","_Arm"))
    st.markdown("**Aligned (Year+Month) sample size:** {}".format(len(merged)))

    # Show quick preview
    st.dataframe(merged.head(10), use_container_width=True)

    # Mann-Whitney tests on raw distributions (two-sided)
    from scipy.stats import mannwhitneyu, pearsonr
    from statsmodels.stats.multitest import multipletests
    results_rows = []
    for var in ["Temperature","pH","Conductivity"]:
        g1 = inflow_df[var].dropna()
        g2 = arm_df[var].dropna()
        stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
        results_rows.append({"Variable": var, "MannWhitneyU": stat, "p_value": p})
    results_df = pd.DataFrame(results_rows)
    results_df["p_value_holm"] = multipletests(results_df["p_value"], method="holm")[1]
    st.markdown("**Mann–Whitney U tests (raw distributions):**")
    st.dataframe(results_df, use_container_width=True)

    # Scatter with OLS per variable (Arm vs Inflow)
    import numpy as np
    import statsmodels.api as sm
    def scat_ols(xcol, ycol, xlabel, ylabel):
        sub = merged[[xcol, ycol]].dropna()
        if len(sub) < 3:
            st.info("Not enough paired data for {}".format(ylabel))
            return None
        X = sm.add_constant(sub[xcol])
        model = sm.OLS(sub[ycol], X).fit()
        slope = model.params[xcol]; intercept = model.params["const"]; r2 = model.rsquared
        r, p = pearsonr(sub[xcol], sub[ycol])
        fig, ax = plt.subplots(figsize=(7,4.2))
        ax.scatter(sub[xcol], sub[ycol], s=18, alpha=0.7)
        xs = np.linspace(sub[xcol].min(), sub[xcol].max(), 100)
        ax.plot(xs, slope*xs + intercept, linewidth=2)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs {xlabel}  (R²={r2:.2f}, r={r:.2f}, p={p:.3g})")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        _figs.append(fig)
        return {"Relationship": f"{ylabel} ~ {xlabel}", "slope": slope, "intercept": intercept, "R2": r2, "r": r, "p": p}

    st.markdown("---")
    st.markdown("**Regression (Western Arm as response, Inflow as predictor)**")
    rows = []
    rows.append(scat_ols("Temperature_Inflow","Temperature_Arm","Inflow Temperature (°C)","Western Arm Temperature (°C)"))
    rows.append(scat_ols("pH_Inflow","pH_Arm","Inflow pH","Western Arm pH"))
    rows.append(scat_ols("Conductivity_Inflow","Conductivity_Arm","Inflow Conductivity (µS/cm)","Western Arm Conductivity (µS/cm)"))
    rows = [r for r in rows if r is not None]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Deltas over time (Arm - Inflow), yearly means for stability
    st.markdown("---")
    st.subheader("Differences over time (Western Arm minus Inflow)")
    if len(merged) >= 3:
        diffs = merged.copy()
        diffs["dTemp"] = diffs["Temperature_Arm"] - diffs["Temperature_Inflow"]
        diffs["dpH"] = diffs["pH_Arm"] - diffs["pH_Inflow"]
        diffs["dCond"] = diffs["Conductivity_Arm"] - diffs["Conductivity_Inflow"]
        ymean = diffs.groupby("Year")[["dTemp","dpH","dCond"]].mean().reset_index()
        for col,label in [("dTemp","Δ Temperature (°C)"),("dpH","Δ pH"),("dCond","Δ Conductivity (µS/cm)") ]:
            fig, ax = plt.subplots(figsize=(8,3.6))
            ax.plot(ymean["Year"], ymean[col], marker="o")
            ax.axhline(0, linewidth=1)
            ax.set_title(f"Yearly mean {label} (Western Arm - Inflow)")
            ax.set_xlabel("Year"); ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            _figs.append(fig)

    # Downloads for this tab
    st.markdown("---")
    st.subheader("Download comparison data")
    st.download_button("⬇️ Download paired (Year+Month) CSV", data=merged.to_csv(index=False).encode("utf-8"), file_name="inflow_arm_paired.csv", mime="text/csv")

st.markdown("---")
st.caption("App renders plots with Matplotlib (no seaborn). Data: pond_data.csv")
