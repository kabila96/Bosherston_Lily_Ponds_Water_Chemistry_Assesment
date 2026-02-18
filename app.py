
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kruskal, pearsonr, spearmanr, mannwhitneyu
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import io, zipfile, textwrap
from matplotlib.backends.backend_pdf import PdfPages

# Figure numbering helper (for captions like Figure 1, Figure 2, ...)
FIG_NO = 0

def fig_caption(text: str) -> None:
    """Add an incrementing figure caption (Figure 1, Figure 2, ...)."""
    global FIG_NO
    FIG_NO += 1
    st.markdown(
        f"<div class='figcap'><b>Figure {FIG_NO}.</b> {text}</div>",
        unsafe_allow_html=True,
    )


# ============================
# App config
# ============================
st.set_page_config(
    page_title="Bosherston Lily Ponds | Water Chemistry Analysis",
    layout="wide",
    page_icon="💧"
)

st.markdown(
    """
<style>
.figcap{
  font-size: 0.92rem;
  opacity: 0.85;
  margin-top: -0.25rem;
  margin-bottom: 1.0rem;
}
.kpi-box{
  border: 1px solid rgba(49,51,63,0.2);
  border-radius: 12px;
  padding: 12px 14px;
}
.small-note{opacity:0.85; font-size:0.92rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ============================
# Branding header
# ============================
st.title("💧 Bosherston Lily Ponds — Water Chemistry Analysis")
st.markdown(
    """
**Powell Andile Ndlovu | Environmental Data and GIS Analyst**  
Chevening Scholar | MSc GIS & Climate Change | Swansea University  

This **Portfolio** explores **pH**, **water temperature**, and **electrical conductivity** across monitoring sites
within the Bosherston Lily Ponds/Bosherston Lakes system (Pembrokeshire, Wales).  
"""
)

# ============================
# Context + citations (DOI-only, verifiable)
# ============================

with st.expander("📍 Study area context (brief) + references", expanded=True):
    st.markdown(
        """
**Where is Bosherston?**  
Bosherston (Pembrokeshire, Wales) sits within the Stackpole Estate and the Pembrokeshire Coast National Park. The **Bosherston Lakes / Lily Ponds** are a linked chain of shallow lakes within limestone valleys, created historically by a series of estate dams (National Trust, 2025; Natural Resources Wales, 2025).

**Why the site matters (environmental context):**  
The lake system is widely recognised for conservation value and has been studied in relation to **hydrology and water balance in limestone terrain** and **long‑term water‑quality pressures linked to catchment inputs** (Vale & Holman, 2009; Rees et al., 1991; Husband & Cassidy, 2009).

**Scope of this project (important):**  
This project reports **only on the measured indicators in the dataset**: **pH**, **water temperature**, and **electrical conductivity (EC)**.  
No additional chemical, biological, or ecological variables are inferred beyond what is measured.


A short reference list (with DOI where available) is provided at the end of the report/download.
"""
    )


# ============================
# Data
# ============================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Clean numeric columns we actually analyze
    for col in ["Temperature", "pH", "Conductivity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Standardize common columns
    for col in ["Year", "Month"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df

df_raw = load_data("pond_data.csv")

required_cols = {"Ponds", "Year", "Month", "Temperature", "pH", "Conductivity"}
missing = required_cols - set(df_raw.columns)
if missing:
    st.error(f"Your CSV is missing required columns: {sorted(missing)}")
    st.stop()

# ============================
# Sidebar controls
# ============================
st.sidebar.header("Controls")
ponds = sorted(df_raw["Ponds"].dropna().unique().tolist())
sel_ponds = st.sidebar.multiselect("Select ponds/sites", ponds, default=ponds)

sel_vars = st.sidebar.multiselect(
    "Variables",
    ["Temperature", "pH", "Conductivity"],
    default=["Temperature", "pH", "Conductivity"]
)

min_year = int(df_raw["Year"].dropna().min())
max_year = int(df_raw["Year"].dropna().max())
year_range = st.sidebar.slider("Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))

st.sidebar.subheader("Downloads")
report_path = Path(__file__).parent / "Water_Chemistry_Report_Powell_Ndlovu.pdf"
if report_path.exists():
    report_bytes = report_path.read_bytes()
    st.sidebar.download_button(
        label="📄 Download full report (PDF)",
        data=report_bytes,
        file_name="Water_Chemistry_Report_Powell_Ndlovu.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
else:
    st.sidebar.info("To enable the full report download, place 'Water_Chemistry_Report_Powell_Ndlovu.pdf' in the same folder as this app.")


df = df_raw.copy()
df = df[df["Ponds"].isin(sel_ponds)]
df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]
df = df.dropna(subset=sel_vars)

# ============================
# Helpers
# ============================
def fmt_num(x, nd=2):
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NA"

def describe_var(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "std": float(s.std(ddof=1)) if s.size > 1 else np.nan
    }

def make_text_page(title: str, paragraphs: list[str]) -> plt.Figure:
    # A4-ish portrait canvas
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
    ax.axis("off")
    ax.text(0, 1.0, title, va="top", ha="left", fontsize=18, fontweight="bold")
    y = 0.95
    for p in paragraphs:
        wrapped = "\n".join(textwrap.wrap(p, width=105))
        ax.text(0, y, wrapped, va="top", ha="left", fontsize=11)
        y -= 0.06 + 0.012 * wrapped.count("\n")
        if y < 0.08:
            break
    return fig

# Collect figures for export
_figs: list[plt.Figure] = []

# ============================
# Executive findings (computed)
# ============================
st.subheader("Executive snapshot (based on your current filters)")
colA, colB, colC = st.columns(3)

with colA:
    st.metric("Records (n)", int(len(df)))
with colB:
    st.metric("Sites selected", int(len(sel_ponds)))
with colC:
    st.metric("Years", f"{year_range[0]}–{year_range[1]}")

# Compact stats table
rows = []
for v in sel_vars:
    d = describe_var(df[v])
    if d.get("n", 0) == 0:
        continue
    rows.append({
        "Variable": v,
        "n": d["n"],
        "Mean": d["mean"],
        "Median": d["median"],
        "Min": d["min"],
        "Max": d["max"],
        "Std": d["std"]
    })
summary_tbl = pd.DataFrame(rows)
if not summary_tbl.empty:
    st.dataframe(
        summary_tbl.style.format(
            {"Mean":"{:.2f}", "Median":"{:.2f}", "Min":"{:.2f}", "Max":"{:.2f}", "Std":"{:.2f}"}
        ),
        use_container_width=True
    )

fig_caption(
    "These summaries describe only the measured variables (pH, temperature, conductivity). "
    "They are intended to support transparent interpretation for environmental monitoring and interpretation."
)

st.markdown("---")

# ============================
# Tabs
# ============================
tabs = st.tabs([
    "Overview",
    "Seasonal (Monthly)",
    "Long‑term (Yearly)",
    "Between‑site statistics",
    "Correlations (paired variables)",
    "Inflow vs Western Arm",
    "project discussion",
    "Downloads (CSV, figures, executive PDF)"
])

# --------------------
# Tab 1: Overview
# --------------------
with tabs[0]:
    st.markdown("### Dataset preview")
    st.dataframe(df.head(30), use_container_width=True)

    st.markdown("### Sampling density by site")
    counts = df.groupby("Ponds").size().rename("n").reset_index().sort_values("n", ascending=False)
    st.dataframe(counts, use_container_width=True)
    fig_caption("Caption: Shows where monitoring effort is highest (important when comparing sites).")

# --------------------
# Tab 2: Monthly
# --------------------
with tabs[1]:
    st.markdown("### Monthly climatology (Jan–Dec) by site")
    monthly = (df.groupby(["Month", "Ponds"], as_index=False)
                 .mean(numeric_only=True)
                 .sort_values("Month"))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    for var in sel_vars:
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for p in sel_ponds:
            sub = (monthly[monthly["Ponds"] == p]
                   .set_index("Month")
                   .reindex(range(1,13))
                   .reset_index())
            ax.plot(sub["Month"], sub[var], marker="o", label=p)
        ax.set_xticks(range(1,13), month_labels)
        ax.set_xlabel("Month")
        ax.set_ylabel(var if var != "Conductivity" else "Conductivity (µS/cm)")
        ax.set_title(f"Monthly pattern: {var}")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        _figs.append(fig)

        # Data-driven caption
        m_all = monthly.groupby("Month")[var].mean(numeric_only=True)
        if len(m_all.dropna()) >= 3:
            peak_m = int(m_all.idxmax())
            low_m = int(m_all.idxmin())
            fig_caption(
                f"Caption: {var} peaks around **{month_labels[peak_m-1]}** and is lowest around **{month_labels[low_m-1]}** "
                f"(means across selected sites). Seasonal patterns help interpret natural variability versus site differences."
            )
        else:
            fig_caption("Caption: Monthly pattern by site for the selected variable.")

# --------------------
# Tab 3: Yearly
# --------------------
with tabs[2]:
    st.markdown("### Yearly averages by site")
    yearly = df.groupby(["Year", "Ponds"]).mean(numeric_only=True).reset_index()

    for var in sel_vars:
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for p in sel_ponds:
            sub = yearly[yearly["Ponds"] == p].sort_values("Year")
            ax.plot(sub["Year"], sub[var], marker="o", label=p)
        ax.set_xlabel("Year")
        ax.set_ylabel(var if var != "Conductivity" else "Conductivity (µS/cm)")
        ax.set_title(f"Long‑term trend: {var}")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        _figs.append(fig)

        # Caption with slope sign from pooled OLS (quick indicator, not causal)
        sub_all = yearly[["Year", var]].dropna()
        if len(sub_all) >= 5:
            X = sm.add_constant(sub_all["Year"].astype(float))
            model = sm.OLS(sub_all[var].astype(float), X).fit()
            slope = float(model.params["Year"])
            direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            fig_caption(
                f"Caption: Pooled yearly means suggest a **{direction}** tendency for {var} (OLS slope ≈ {slope:.3g} per year). "
                "This is a descriptive indicator and should be interpreted alongside sampling density and site context."
            )
        else:
            fig_caption("Caption: Yearly means by site for the selected variable.")

# --------------------
# Tab 4: Stats + boxplots
# --------------------
with tabs[3]:
    st.markdown("### Kruskal–Wallis tests across sites (non‑parametric)")
    results = []
    for var in sel_vars:
        groups = [df[df["Ponds"] == p][var].dropna() for p in sel_ponds]
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            stat, pval = kruskal(*groups)
            results.append({"Variable": var, "H": stat, "p-value": pval})
    res_df = pd.DataFrame(results)
    st.dataframe(res_df, use_container_width=True)
    fig_caption(
        "Caption: Kruskal–Wallis tests whether site medians differ. A small p-value suggests at least one site differs, "
        "but does not tell which—use boxplots to understand distributions."
    )

    st.markdown("---")
    st.markdown("### Box & whisker plots (distribution by site)")
    for var in sel_vars:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        data = [df[df["Ponds"] == p][var].dropna().values for p in sel_ponds]
        ax.boxplot(data, labels=sel_ponds, showfliers=True)
        ax.set_title(f"{var} distribution by site")
        ax.set_ylabel(var if var != "Conductivity" else "Conductivity (µS/cm)")
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        st.pyplot(fig, use_container_width=True)
        _figs.append(fig)

        # Caption: show top/bottom median sites
        med = df.groupby("Ponds")[var].median(numeric_only=True).dropna()
        if len(med) >= 2:
            hi = med.idxmax(); lo = med.idxmin()
            fig_caption(
                f"Caption: Median {var} is highest at **{hi}** ({fmt_num(med[hi])}) and lowest at **{lo}** ({fmt_num(med[lo])}) "
                "for the current filter window."
            )
        else:
            fig_caption("Caption: Distribution of the selected variable across sites.")

# --------------------
# Tab 5: correlations
# --------------------
with tabs[4]:
    st.markdown("### Scatter plots with regression lines (pooled across selected sites)")
    pairs = [("Temperature", "Conductivity"), ("Temperature", "pH"), ("pH", "Conductivity")]
    corr_rows = []

    for x, y in pairs:
        if (x not in sel_vars) or (y not in sel_vars):
            continue

        sub = df[[x, y]].dropna()
        if len(sub) < 8:
            st.info(f"Not enough paired observations for {y} vs {x}.")
            continue

        # OLS line (descriptive)
        X = sm.add_constant(sub[x].astype(float))
        model = sm.OLS(sub[y].astype(float), X).fit()
        slope = float(model.params[x])
        intercept = float(model.params["const"])
        r2 = float(model.rsquared)

        # Correlations
        pear_r, pear_p = pearsonr(sub[x], sub[y])
        spear_r, spear_p = spearmanr(sub[x], sub[y])

        corr_rows.append({
            "Relationship": f"{y} vs {x}",
            "Pearson_r": pear_r, "Pearson_p": pear_p,
            "Spearman_r": spear_r, "Spearman_p": spear_p,
            "OLS_slope": slope, "OLS_intercept": intercept, "R2": r2,
            "n": int(len(sub))
        })

        # Plot
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        ax.scatter(sub[x], sub[y], s=14, alpha=0.65)
        xs = np.linspace(sub[x].min(), sub[x].max(), 100)
        ax.plot(xs, slope * xs + intercept, linewidth=2)
        ax.set_xlabel(x if x != "Conductivity" else "Conductivity (µS/cm)")
        ax.set_ylabel(y if y != "Conductivity" else "Conductivity (µS/cm)")
        ax.set_title(f"{y} vs {x}  (R²={r2:.2f})")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        _figs.append(fig)

        direction = "positive" if pear_r > 0 else "negative" if pear_r < 0 else "near‑zero"
        fig_caption(
            f"Caption: The pooled association between {x} and {y} is **{direction}** "
            f"(Pearson r={pear_r:.2f}, p={pear_p:.3g}, n={len(sub)}). "
            "Correlation here is descriptive and may be influenced by seasonality and site mixing."
        )

    if corr_rows:
        st.markdown("**Correlation & regression summary (pooled)**")
        st.dataframe(pd.DataFrame(corr_rows), use_container_width=True)

# --------------------
# Tab 6: Inflow vs Western Arm
# --------------------
with tabs[5]:
    st.markdown("### Western Arm vs Western Arm Spring Inflow")
    st.markdown(
        "This section aligns **Western_Arm_Spring_Inflow** and **Western_Arm** by **Year + Month** to examine "
        "whether inflow variability is reflected downstream in the Western Arm (for pH, temperature, conductivity)."
    )

    inflow_df = df[df["Ponds"] == "Western_Arm_Spring_Inflow"].copy()
    arm_df = df[df["Ponds"] == "Western_Arm"].copy()

    if inflow_df.empty or arm_df.empty:
        st.info("To use this tab, ensure both **Western_Arm_Spring_Inflow** and **Western_Arm** are included in your selected sites.")
    else:
        inflow_m = inflow_df.groupby(["Year", "Month"], as_index=False).mean(numeric_only=True)
        arm_m = arm_df.groupby(["Year", "Month"], as_index=False).mean(numeric_only=True)
        merged = pd.merge(inflow_m, arm_m, on=["Year", "Month"], suffixes=("_Inflow", "_Arm"))
        st.markdown(f"**Aligned paired months (n):** {len(merged)}")
        st.dataframe(merged.head(12), use_container_width=True)

        # Mann-Whitney on raw distributions
        rows = []
        for var in ["Temperature", "pH", "Conductivity"]:
            g1 = inflow_df[var].dropna()
            g2 = arm_df[var].dropna()
            if len(g1) > 0 and len(g2) > 0:
                stat, pval = mannwhitneyu(g1, g2, alternative="two-sided")
                rows.append({"Variable": var, "MannWhitneyU": stat, "p_value": pval})
        results_df = pd.DataFrame(rows)
        if not results_df.empty:
            results_df["p_value_holm"] = multipletests(results_df["p_value"], method="holm")[1]
            st.markdown("**Mann–Whitney U tests (raw distributions, Holm-adjusted p-values shown):**")
            st.dataframe(results_df, use_container_width=True)
            fig_caption("Caption: Tests whether distributions differ between inflow and arm; it does not establish causality.")

        # Regression plots (Arm ~ Inflow)
        def scat_ols(xcol, ycol, xlabel, ylabel):
            sub = merged[[xcol, ycol]].dropna()
            if len(sub) < 6:
                st.info(f"Not enough paired data for {ylabel}.")
                return None
            X = sm.add_constant(sub[xcol].astype(float))
            model = sm.OLS(sub[ycol].astype(float), X).fit()
            slope = float(model.params[xcol])
            intercept = float(model.params["const"])
            r2 = float(model.rsquared)
            r, p = pearsonr(sub[xcol], sub[ycol])

            fig, ax = plt.subplots(figsize=(7.6, 4.3))
            ax.scatter(sub[xcol], sub[ycol], s=18, alpha=0.7)
            xs = np.linspace(sub[xcol].min(), sub[xcol].max(), 100)
            ax.plot(xs, slope * xs + intercept, linewidth=2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs {xlabel}  (R²={r2:.2f}, r={r:.2f}, p={p:.3g})")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            _figs.append(fig)

            fig_caption(
                f"Caption: Paired-month relationship shows r={r:.2f} (p={p:.3g}). "
                "A stronger relationship suggests the inflow signal is more directly reflected in the Western Arm."
            )
            return {"Relationship": f"{ylabel} ~ {xlabel}", "slope": slope, "intercept": intercept, "R2": r2, "r": r, "p": p, "n": int(len(sub))}

        st.markdown("---")
        st.markdown("**Regression (Western Arm as response, Inflow as predictor)**")
        rows = []
        rows.append(scat_ols("Temperature_Inflow", "Temperature_Arm", "Inflow Temperature (°C)", "Western Arm Temperature (°C)"))
        rows.append(scat_ols("pH_Inflow", "pH_Arm", "Inflow pH", "Western Arm pH"))
        rows.append(scat_ols("Conductivity_Inflow", "Conductivity_Arm", "Inflow Conductivity (µS/cm)", "Western Arm Conductivity (µS/cm)"))
        rows = [r for r in rows if r is not None]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Deltas over time (Arm - Inflow), yearly means
        st.markdown("---")
        st.markdown("### Differences over time (Western Arm minus Inflow)")
        if len(merged) >= 6:
            diffs = merged.copy()
            diffs["dTemp"] = diffs["Temperature_Arm"] - diffs["Temperature_Inflow"]
            diffs["dpH"] = diffs["pH_Arm"] - diffs["pH_Inflow"]
            diffs["dCond"] = diffs["Conductivity_Arm"] - diffs["Conductivity_Inflow"]
            ymean = diffs.groupby("Year")[["dTemp", "dpH", "dCond"]].mean().reset_index()

            for col, label in [("dTemp", "Δ Temperature (°C)"), ("dpH", "Δ pH"), ("dCond", "Δ Conductivity (µS/cm)")]:
                fig, ax = plt.subplots(figsize=(8.6, 3.8))
                ax.plot(ymean["Year"], ymean[col], marker="o")
                ax.axhline(0, linewidth=1)
                ax.set_title(f"Yearly mean {label} (Western Arm − Inflow)")
                ax.set_xlabel("Year")
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig, use_container_width=True)
                _figs.append(fig)

                fig_caption(
                    f"Caption: Values above 0 mean the Western Arm is higher than the inflow for {label}. "
                    "Persistent offsets may reflect in‑system processes (mixing, residence time) or additional inputs."
                )

        st.markdown("---")
        st.markdown("**Download paired comparison data**")
        st.download_button(
            "⬇️ Download paired (Year+Month) CSV",
            data=merged.to_csv(index=False).encode("utf-8"),
            file_name="inflow_arm_paired.csv",
            mime="text/csv"
        )

# --------------------
# Tab 7: Discussion + solutions (within scope)
# --------------------
with tabs[6]:
    st.markdown("## Discussion (project-ready, within measured variables)")
    st.markdown(
        """
This section translates the plots into concise, defensible environmental monitoring interpretations.
It **only** discusses patterns in *pH, temperature, and conductivity*—and avoids claims about unmeasured drivers.
"""
    )

    # Automatically derive a few “findings” statements
    findings = []
    # Seasonal amplitude by variable (pooled)
    monthly_all = df.groupby("Month")[sel_vars].mean(numeric_only=True)
    for var in sel_vars:
        s = monthly_all[var].dropna()
        if len(s) >= 3:
            amp = float(s.max() - s.min())
            findings.append(f"**Seasonality:** {var} shows a pooled monthly amplitude of ~{fmt_num(amp)} across the selected sites.")
    # Site spread by variable
    for var in sel_vars:
        med = df.groupby("Ponds")[var].median(numeric_only=True).dropna()
        if len(med) >= 2:
            spread = float(med.max() - med.min())
            findings.append(f"**Spatial variability:** Median {var} differs by ~{fmt_num(spread)} between the highest and lowest median sites.")

    if findings:
        st.markdown("### Key findings (auto-generated from the current filter)")
        for f in findings[:6]:
            st.markdown(f"- {f}")

    st.markdown("### What these indicators can and cannot tell us")
    st.markdown(
        """
- **Temperature:** Helps interpret seasonal habitat conditions, mixing, and potential stress periods (e.g., warm months).  
- **pH:** Indicates acid–base conditions; sustained shifts may affect chemical availability and organism tolerance.  
- **Conductivity:** Reflects ionic strength; changes can indicate different water sources, dilution/concentration effects, or catchment inputs.  

**Limitations:** Without additional parameters (e.g., nutrients, dissolved oxygen, alkalinity, chlorophyll‑a),
interpretation must remain **indicator‑based** and cautious.
"""
    )

    st.markdown("### Practical solutions / next steps")
    st.markdown(
        """
**Monitoring upgrades (low‑cost, high value):**
- Add **routine QA/QC**: field duplicates, calibration logs for pH and conductivity, and temperature probe verification.
- Record **metadata** consistently: sampling time, weather, recent rainfall, and site notes (helps interpret anomalies).
- Adopt a **fixed monthly sampling window** to reduce seasonal sampling bias.

**Targeted investigations (only if permitted by the project scope):**
- If conductivity changes are a concern, add **alkalinity/hardness** and **major ions** to identify the likely source of ionic shifts.
- For periods with warmer temperatures, consider measuring **dissolved oxygen** to better contextualize ecological risk.

**Communication & management:**
- Produce a short **quarterly dashboard** (like this app) for site managers: highlight outliers, seasonal peaks, and site comparisons.
- Use the **inflow vs Western Arm** evidence to prioritise where interventions or additional monitoring would be most informative.

These actions are designed to be defensible from the current dataset while aligning with good environmental monitoring practice.
"""
    )

# --------------------
# Tab 8: Downloads
# --------------------
with tabs[7]:
    st.markdown("## Downloads")

    # Filtered CSV
    csv_all = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered dataset (CSV)",
        data=csv_all,
        file_name="filtered_pond_data.csv",
        mime="text/csv"
    )

    # Monthly/Yearly summaries
    monthly = (df.groupby(["Month", "Ponds"], as_index=False)
                 .mean(numeric_only=True)
                 .sort_values("Month"))
    yearly = df.groupby(["Year", "Ponds"]).mean(numeric_only=True).reset_index()
    st.download_button("⬇️ Download monthly means (CSV)", data=monthly.to_csv(index=False).encode("utf-8"), file_name="monthly_means.csv", mime="text/csv")
    st.download_button("⬇️ Download yearly means (CSV)", data=yearly.to_csv(index=False).encode("utf-8"), file_name="yearly_means.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("### Download figures")

    # Multipage PDF of figures
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        for f in _figs:
            pdf.savefig(f, bbox_inches="tight")
    pdf_buffer.seek(0)
    st.download_button("📄 Download multi-page PDF of figures", data=pdf_buffer, file_name="bosherston_figures.pdf", mime="application/pdf")

    # ZIP of PNGs
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, f in enumerate(_figs, start=1):
            img_bytes = io.BytesIO()
            f.savefig(img_bytes, format="png", dpi=200, bbox_inches="tight")
            img_bytes.seek(0)
            zf.writestr(f"figure_{i:02d}.png", img_bytes.read())
    zip_buffer.seek(0)
    st.download_button("🗂️ Download figures as PNG (ZIP)", data=zip_buffer, file_name="bosherston_figures_png.zip", mime="application/zip")

    st.markdown("---")
    st.markdown("### Executive PDF report (project-ready)")
    fig_caption("Includes study context, methods (brief), key statistics, interpretations within scope, and DOI references.")

    def build_executive_report() -> bytes:
        buf = io.BytesIO()

        # Prepare a few tables for the report
        site_stats = []
        for v in ["Temperature", "pH", "Conductivity"]:
            if v not in sel_vars:
                continue
            med = df.groupby("Ponds")[v].median(numeric_only=True).dropna()
            if len(med) == 0:
                continue
            site_stats.append(pd.DataFrame({"Ponds": med.index, f"Median {v}": med.values}).sort_values(f"Median {v}", ascending=False))

        # Build pages
        with PdfPages(buf) as pdf:
            # Title/summary page
            paragraphs = [
                "Prepared by Powell Andile Ndlovu (Chevening Scholar; MSc GIS & Climate Change, Swansea University).",
                f"Study period (filtered): {year_range[0]}–{year_range[1]}. Sites included: {len(sel_ponds)}. Records (n): {len(df)}.",
                "Scope: This report summarizes patterns in pH, temperature, and conductivity from the supplied dataset. "
                "It does not infer nutrients, pollutants, or biological status.",
                "Study area context: Bosherston Lily Ponds/Bosherston Lakes are a shallow, regulated lake system in Pembrokeshire, Wales; "
                "hydrological regulation and seepage pathways are documented in peer-reviewed work (doi:10.3997/1873-0604.2009042).",
                "Key patterns (high-level): seasonality (especially temperature), spatial differences between monitoring sites, and "
                "descriptive associations between variables (correlation/regression).",
            ]
            fig0 = make_text_page("Executive Water Chemistry Summary — Bosherston Lily Ponds", paragraphs)
            pdf.savefig(fig0, bbox_inches="tight"); plt.close(fig0)

            # Add a stats table page
            if not summary_tbl.empty:
                fig1 = plt.figure(figsize=(8.27, 11.69))
                ax = fig1.add_axes([0.06, 0.06, 0.88, 0.88])
                ax.axis("off")
                ax.text(0, 1.0, "Key descriptive statistics (filtered data)", va="top", ha="left",
                        fontsize=16, fontweight="bold")
                # Render table
                tdf = summary_tbl.copy()
                for c in ["Mean","Median","Min","Max","Std"]:
                    if c in tdf.columns:
                        tdf[c] = tdf[c].map(lambda x: fmt_num(x, 2))
                cell_text = tdf.values.tolist()
                col_labels = tdf.columns.tolist()
                table = ax.table(cellText=cell_text, colLabels=col_labels, loc="upper left", cellLoc="left")
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.0, 1.25)
                ax.text(0, 0.08, "Note: Descriptive statistics only for measured variables.", fontsize=10)
                pdf.savefig(fig1, bbox_inches="tight"); plt.close(fig1)

            # Add a site medians page
            if site_stats:
                fig2 = plt.figure(figsize=(8.27, 11.69))
                ax = fig2.add_axes([0.06, 0.06, 0.88, 0.88])
                ax.axis("off")
                ax.text(0, 1.0, "Site medians (ranked)", va="top", ha="left",
                        fontsize=16, fontweight="bold")
                y = 0.93
                for stbl in site_stats:
                    ax.text(0, y, stbl.columns[1], fontsize=12, fontweight="bold", va="top")
                    y -= 0.03
                    show = stbl.head(10).copy()
                    show[stbl.columns[1]] = show[stbl.columns[1]].map(lambda x: fmt_num(x, 2))
                    table = ax.table(cellText=show.values.tolist(), colLabels=show.columns.tolist(),
                                     loc="upper left", cellLoc="left", bbox=[0, y-0.28, 0.9, 0.26])
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    y -= 0.32
                    if y < 0.25:
                        break
                pdf.savefig(fig2, bbox_inches="tight"); plt.close(fig2)

            # Add figures (if any)
            for f in _figs[:20]:  # keep report concise
                pdf.savefig(f, bbox_inches="tight")

            # References page (DOI only)
            ref_paras = [
                "References",
                "Husband, C.R. & Cassidy, N.J. (2009). The geophysical investigation of lake water seepage in the regulated environment of the Bosherston Lily Ponds, South Wales, UK. Near Surface Geophysics. doi:10.3997/1873-0604.2009042",
                "Husband, C.R. & Cassidy, N.J. (2009). Historical, dam-related pathways at Bosherston Lily Ponds. Near Surface Geophysics. doi:10.3997/1873-0604.2009044",
                "Rees, A.W.G., Hinton, G.C.F., Johnson, F.G. & O’Sullivan, P.E. (1991). The sediment column as a record of trophic status: examples from Bosherston Lakes, SW Wales. Hydrobiologia, 214, 171–180. doi:10.1007/BF00050947",
                "Vale, M. & Holman, I.P. (2009). Understanding the hydrological functioning of a shallow lake system within a coastal karstic aquifer in Wales, UK. Journal of Hydrology, 376, 285–294. doi:10.1016/j.jhydrol.2009.07.041",
                "de Sousa, D.N.R., Mozeto, A.A., Carneiro, R.L. & Fadini, P.S. (2014). Electrical conductivity as a marker of surface freshwater contamination by wastewater. Science of the Total Environment, 484, 19–26. doi:10.1016/j.scitotenv.2014.02.135",
                "Ndlovu, P.A. (2026). Bosherston Lily Ponds water chemistry analysis (pH, temperature, conductivity) — project report and reproducible code. Swansea University (unpublished)."
            ]
            figR = make_text_page("References", ref_paras)
            pdf.savefig(figR, bbox_inches="tight"); plt.close(figR)

        buf.seek(0)
        return buf.read()

    rep_bytes = build_executive_report()
    st.download_button(
        "📄 Download Executive PDF Report",
        data=rep_bytes,
        file_name="Bosherston_Lily_Ponds_Executive_Report_Powell_Ndlovu.pdf",
        mime="application/pdf"
    )

st.markdown("---")
fig_caption("Built with Streamlit + Matplotlib. Dataset: pond_data.csv (pH, Temperature, Conductivity).")
