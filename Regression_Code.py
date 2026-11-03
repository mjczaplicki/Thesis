from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import chi2

# ----------------------------- SETTINGS -----------------------------
IN = Path(r"C:\Users\LENOVO\Downloads\SPX_VIX_Tbill_matched_by_date_extended.csv")
OUT = IN.with_name("Results.xlsx")
GRAPH_DIR = OUT.parent / "Graphs"
HALF_LIVES, DAYS, SIG = [20, 40, 60, 80, 100, 120], 252, 4
def rnd(x):
    return float(f"{x:.{SIG}g}") if pd.notna(x) and np.isfinite(x) and x != 0 else x

# ----------------------------- LOAD DATA -----------------------------
df = pd.read_csv(IN, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
if df.Date.duplicated().any():
    raise ValueError("Duplicate dates found.")
for c in ["SPX", "VIX", "Tbill"]:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
df["r"] = df.SPX.pct_change()
rf = df.Tbill / 100
df["rf"] = (1 + rf)  **  (1 / DAYS) - 1
df["ERP0"] = (1.05 + rf)  **  (1 / DAYS) - 1
df["y"] = df.r.shift(-1) - df.rf
df["M"] = df.VIX
df["target_date"] = df.Date.shift(-1)
def earnings_season(dates):
    md = dates.dt.month * 100 + dates.dt.day
    return (((md >= 115) & (md <= 215)) | ((md >= 415) & (md <= 515)) |
            ((md >= 715) & (md <= 815)) | ((md >= 1015) & (md <= 1115))).fillna(False)
df["Earnings_Season"] = earnings_season(df.target_date)

# ----------------------------- SIGNALS -----------------------------
# Construct signals on the entire data.
for h in HALF_LIVES:
    phi = 1 - 2  **  (-1 / h)
    x = df.r.ewm(alpha=phi, adjust=False, min_periods=1).mean() - df.ERP0
    df[f"x_raw_{h}"] = x
    for side in ["+", "-"]:
        z = x.clip(lower=0) if side == "+" else x.clip(upper=0)
        df[f"x{side}{h}"] = z
        df[f"x{side}3_{h}"] = z  **  3
        df[f"Mx{side}_{h}"] = df.M * z
        df[f"Mx{side}3_{h}"] = df.M * z  **  3
# Common estimation sample: VIX observed 4y later.
est = df.loc[df.VIX.notna() & df.y.notna()].copy()
if est.empty:
    raise ValueError("No usable estimation observations.")
print(f"Signal start: {df.Date.min()} | Estimation start: {est.Date.min()} | "
      f"Estimation end: {est.Date.max()} | N={len(est)}")

# ----------------------------- MODEL DEFINITIONS -----------------------------
base = ["x+", "x+3", "Mx+", "Mx+3", "x-", "x-3", "Mx-", "Mx-3"]
# Refined cubic model: drop x+3 and x-3.
base11 = [v for v in base if v not in ("x+3", "x-3")]
# Two-sided cubic model without any VIX interactions.
base_noninteracted = ["x+", "x+3", "x-", "x-3"]
tests_noninteracted = {
    "Positive": ["x+", "x+3"],
    "Negative": ["x-", "x-3"],
    "Linear": ["x+", "x-"],
    "Cubic": ["x+3", "x-3"],
    "Complete model": base_noninteracted,
}
tests1 = {
    "Positive": base[:4],
    "Negative": base[4:],
    "Linear": ["x+", "x-", "Mx+", "Mx-"],
    "Cubic": ["x+3", "x-3", "Mx+3", "Mx-3"],
    "Standalone cubic": ["x+3", "x-3"],
    "VIX-interacted cubic": ["Mx+3", "Mx-3"],
    "VIX-interacted": ["Mx+", "Mx+3", "Mx-", "Mx-3"],
    "Non-interacted": ["x+", "x+3", "x-", "x-3"],
}
tests11 = {
    "Positive": ["x+", "Mx+", "Mx+3"],
    "Negative": ["x-", "Mx-", "Mx-3"],
    "Linear": ["x+", "x-", "Mx+", "Mx-"],
    "Cubic": ["Mx+3", "Mx-3"],
    "VIX-interacted": ["Mx+", "Mx+3", "Mx-", "Mx-3"],
    "Non-interacted": ["x+", "x-"],
}
sheets = [
    "General_Info", "X_Statistics", "VIX_Statistics",
    "VIX_Standalone_BP_F", "VIX_Standalone", "Linear_BP_F", "Linear",
    "Linear_Interacted", "Cubic_BP_F", "Cubic", "Cubic_Wald_Tests",
    "Cubic_Refined", "Cubic_Refined_Wald_Tests",
    "Cubic_Refined_Turning_Points", "Cubic_Refined_Earnings",
    "Cubic_Refined_Nonseason",
    "Cubic_Noninteracted_BP_F", "Cubic_Noninteracted",
    "Cubic_Noninteracted_Wald_Tests"
]
out = {s: [] for s in sheets}
walds = {"Cubic": {}, "Cubic refined": {}, "Cubic non-interacted": {}}
sample_sizes = {}
hac_lag_counts = {}

# ----------------------------- ESTIMATION HELPER -----------------------------
def estimate(data, variables, h, sheet, bp_sheet=None, tests=None, model=None):
    d = data[["y"] + variables].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) <= len(variables) + 1:
        raise ValueError(f"Insufficient observations: {sheet}, half-life={h}")
    X = sm.add_constant(d[variables], has_constant="add")
    ols = sm.OLS(d.y, X).fit()
    hac_lags = round(len(d) ** 0.25)
    res = ols.get_robustcov_results(cov_type="HAC", maxlags=hac_lags)
    params, pvals = pd.Series(res.params, index=X.columns), pd.Series(res.pvalues, index=X.columns)
    intercept = params["const"]
    sample_sizes[(model or sheet, h)] = len(d)
    hac_lag_counts[(model or sheet, h)] = hac_lags
    row = {"Half-life": h, "N": len(d), "HAC lags": hac_lags} if h is not None else {"N": len(d), "HAC lags": hac_lags}
    row["ERP"] = rnd(intercept)
    row["ERP (HAC p)"] = rnd(pvals["const"])
    for v in variables:
        row[v], row[f"{v} (HAC p)"] = rnd(params[v]), rnd(pvals[v])
    row.update({
        "Annualised ERP": rnd((1 + intercept)  **  DAYS - 1) if intercept > -1 else np.nan,
        "R2": rnd(ols.rsquared), "Adj-R2": rnd(ols.rsquared_adj)
    })
    out[sheet].append(row)
    if bp_sheet:
        _, _, f, p = het_breuschpagan(ols.resid, X)
        out[bp_sheet].append({"Half-life": h, "N": len(d), "HAC lags": hac_lags, "BP F": rnd(f), "BP F p-value": rnd(p)})
    if tests:
        names = list(X.columns)
        for label, terms in tests.items():
            R = np.zeros((len(terms), len(names)))
            for i, term in enumerate(terms):
                R[i, names.index(term)] = 1
            # HAC-robust Wald test with rank-adjusted degrees of freedom.
            # Effective df = rank of R[Cov(HAC)]R'.
            cov_hac = np.asarray(res.cov_params(), dtype=float)
            restriction_cov = R @ cov_hac @ R.T
            restriction_cov = (restriction_cov + restriction_cov.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(restriction_cov)

            scale = np.max(np.abs(eigenvalues)) if eigenvalues.size else 0.0
            tolerance = max(restriction_cov.shape) * np.finfo(float).eps * scale
            keep = eigenvalues > tolerance
            effective_df = int(np.count_nonzero(keep))

            restriction = R @ np.asarray(res.params, dtype=float)
            if effective_df == 0:
                wald_stat = np.nan
                wald_pvalue = np.nan
            else:
                projected = eigenvectors[:, keep].T @ restriction
                wald_stat = float(np.sum((projected ** 2) / eigenvalues[keep]))
                wald_pvalue = float(chi2.sf(wald_stat, effective_df))

            if effective_df < len(terms):
                print(
                    f"Rank-adjusted Wald test: {model}, half-life={h}, "
                    f"test={label}, requested restrictions={len(terms)}, "
                    f"effective df={effective_df}"
                )

            walds[model].setdefault(h, {})[label] = (
                rnd(wald_stat), rnd(wald_pvalue), len(terms), effective_df
            )
    return d

# ----------------------------- DESCRIPTIVE STATISTICS -----------------------------
for h in HALF_LIVES:
    x = est[f"x_raw_{h}"].dropna()
    s = x.describe(percentiles=[.05, .25, .50, .75, .95])
    out["X_Statistics"].append({
        "Half-life": h, "N": len(x), "Mean": rnd(s["mean"]), "Std": rnd(s["std"]),
        "Min": rnd(s["min"]), "5%": rnd(s["5%"]), "25%": rnd(s["25%"]),
        "Median": rnd(s["50%"]), "75%": rnd(s["75%"]), "95%": rnd(s["95%"]),
        "Max": rnd(s["max"])
    })
vix = est.VIX.dropna()
vs = vix.describe(percentiles=[.05, .25, .50, .75, .95])
out["VIX_Statistics"].append({
    "N": len(vix), "Mean": rnd(vs["mean"]), "Std": rnd(vs["std"]),
    "Min": rnd(vs["min"]), "5%": rnd(vs["5%"]), "25%": rnd(vs["25%"]),
    "Median": rnd(vs["50%"]), "75%": rnd(vs["75%"]), "95%": rnd(vs["95%"]),
    "Max": rnd(vs["max"])
})

# ----------------------------- REGRESSIONS -----------------------------
estimate(est, ["VIX"], None, "VIX_Standalone", "VIX_Standalone_BP_F")
for h in HALF_LIVES:
    rename = {
        f"x+{h}": "x+", f"x+3_{h}": "x+3", f"Mx+_{h}": "Mx+", f"Mx+3_{h}": "Mx+3",
        f"x-{h}": "x-", f"x-3_{h}": "x-3", f"Mx-_{h}": "Mx-", f"Mx-3_{h}": "Mx-3"
    }
    d = est.rename(columns=rename).copy()
    d["x"], d["xM"] = d[f"x_raw_{h}"], d[f"x_raw_{h}"] * d.M
    estimate(d, ["x"], h, "Linear", "Linear_BP_F")
    estimate(d, ["x", "xM"], h, "Linear_Interacted")
    estimate(d, base, h, "Cubic", "Cubic_BP_F", tests1, "Cubic")
    # Two-sided cubic response with no VIX interactions.
    estimate(
        d, base_noninteracted, h,
        "Cubic_Noninteracted", "Cubic_Noninteracted_BP_F",
        tests_noninteracted, "Cubic non-interacted"
    )

    full_refined = estimate(d, base11, h, "Cubic_Refined", tests=tests11, model="Cubic refined")
    N = len(full_refined)
    earnings_refined = estimate(d[d.Earnings_Season], base11, h, "Cubic_Refined_Earnings")
    N_earnings = len(earnings_refined)
    nonseason_refined = estimate(d[~d.Earnings_Season], base11, h, "Cubic_Refined_Nonseason")
    N_nonseason = len(nonseason_refined)
    print(f"Half-life={h}: N={N}, N_earnings={N_earnings}, "
          f"N_nonseason={N_nonseason}; HAC lags="
          f"{round(N ** 0.25)}, {round(N_earnings ** 0.25)}, "
          f"{round(N_nonseason ** 0.25)}")
for model, sheet, tests in [
    ("Cubic", "Cubic_Wald_Tests", tests1),
    ("Cubic refined", "Cubic_Refined_Wald_Tests", tests11),
    ("Cubic non-interacted", "Cubic_Noninteracted_Wald_Tests", tests_noninteracted),
] :
    for h in HALF_LIVES:
        row = {"Half-life": h}
        for label in tests:
            stat, p, requested_df, effective_df = walds[model][h][label]
            row[f"{label} Wald chi2"] = stat
            row[f"{label} requested restrictions"] = requested_df
            row[f"{label} effective df"] = effective_df
            row[f"{label} p-value"] = p
        out[sheet].append(row)

# ----------------------------- DISTRIBUTION GRAPHS -----------------------------
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
VIX_LEVELS = {
    "25th percentile": (float(vs["25%"]), "green"),
    "Median": (float(vs["50%"]), "yellow"),
    "75th percentile": (float(vs["75%"]), "orange"),
    "95th percentile": (float(vs["95%"]), "red")
}
for h in HALF_LIVES:
    fig, ax = plt.subplots(figsize=(9, 5))
    x_dist = est[f"x_raw_{h}"].dropna()
    x_mean, x_median = x_dist.mean(), x_dist.median()
    ax.hist(x_dist, bins=60, edgecolor="black", linewidth=.4)
    ax.axvline(x_mean, color="blue", linestyle="--", linewidth=1.8,
               label=f"Mean: {x_mean:.5g}")
    ax.axvline(x_median, color="red", linestyle="-.", linewidth=1.8,
               label=f"Median: {x_median:.5g}")
    ax.set(title=f"Empirical distribution of x — {h}-day half-life",
           xlabel="Trailing-return signal (x)", ylabel="Frequency")
    ax.grid(True, alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / f"Empirical_x_distribution_{h}day.png", dpi=300)
    plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(vix, bins=60, edgecolor="black", linewidth=.4)
for label, (value, color) in VIX_LEVELS.items():
    ax.axvline(value, color=color, linewidth=1.8, label=f"{label}: {value:.2f}")
ax.set(title="Empirical distribution of VIX", xlabel="VIX", ylabel="Frequency")
ax.grid(True, alpha=.2)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(GRAPH_DIR / "Empirical_VIX_distribution.png", dpi=300)
plt.close(fig)

# ----------------------------- RESPONSE CURVES / ZERO-CROSSINGS -----------------------------
turning = []
for sample_name, flag, sheet, folder_name in [
    ("Full sample", None, "Cubic_Refined", "Full_Sample"),
    ("Earnings season", True, "Cubic_Refined_Earnings", "Earnings_Season"),
    ("Non-earnings season", False, "Cubic_Refined_Nonseason", "NonEarnings_Season")
]:
    folder = GRAPH_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    coef_by_h = {int(r["Half-life"]): r for r in out[sheet]}

    for h in HALF_LIVES:
        sample = est if flag is None else est[est.Earnings_Season == flag]
        sample = sample[["y", f"x_raw_{h}", "VIX"] +
                        [f"x+{h}", f"x+3_{h}", f"Mx+_{h}", f"Mx+3_{h}",
                         f"x-{h}", f"x-3_{h}", f"Mx-_{h}", f"Mx-3_{h}"]]
        sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
        if sample.empty:
            continue

        xobs = sample[f"x_raw_{h}"]
        coef = coef_by_h[h]
        fig, ax = plt.subplots(figsize=(10, 6.5))

        for direction, linestyle in [("Upside", "-"), ("Downside", "--")]:
            xmin, xmax = ((max(0.0, xobs.min()), xobs.max()) if direction == "Upside"
                          else (xobs.min(), min(0.0, xobs.max())))
            if xmax <= xmin:
                continue
            grid = np.linspace(xmin, xmax, 300)

            for label, (m, color) in VIX_LEVELS.items():
                if direction == "Upside":
                    a = coef["x+"] + coef["Mx+"] * m
                    c = coef.get("x+3", 0.0) + coef["Mx+3"] * m
                else:
                    a = coef["x-"] + coef["Mx-"] * m
                    c = coef.get("x-3", 0.0) + coef["Mx-3"] * m

                # Plot estimated b(x,M)
                ax.plot(grid, a * grid + c * grid ** 3, color=color,
                        linestyle=linestyle, linewidth=2.2,
                        label=f"{direction} — {label} (VIX={m:.2f})")

                # Zero-crossings, restricted to the observed range.
                if np.isclose(a, 0) and np.isclose(c, 0):
                    roots = []  
                elif np.isclose(c, 0):
                    roots = [0.0]
                else:
                    roots = np.roots([c, 0, a, 0])
                valid = sorted(float(np.real(z)) for z in roots
                               if abs(float(np.imag(z))) < 1e-8
                               and abs(float(np.real(z))) > 1e-10  # Exclude the x=0 root.
                               and (np.real(z) > 0 if direction == "Upside"
                                    else np.real(z) < 0)
                               and xmin - 1e-10 <= np.real(z) <= xmax + 1e-10)
                for root in valid:
                    turning.append({
                        "Sample": sample_name, "Direction": direction, "Half-life": h,
                        "VIX scenario": label, "VIX": rnd(m),
                        "Zero-crossing x": rnd(root),
                        "Empirical x percentile (%)": rnd(100 * (xobs <= root).mean()),
                        "Observed x minimum": rnd(xmin), "Observed x maximum": rnd(xmax)
                    })

        ax.axhline(0, color="black", linewidth=.8)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(title=f"{sample_name} — Upside and Downside — {h}-day half-life",
               xlabel="Trailing-return signal (x)", ylabel="Signal contribution b(x,VIX)")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(True, alpha=.25)
        ax.legend(frameon=False, ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / f"Upside_Downside_{h}day.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

out["Cubic_Refined_Turning_Points"] = turning

# ----------------------------- EXPORT -----------------------------
out["General_Info"].append({
    "Input": str(IN), "Rows in extended input": len(df),
    "Signal construction start": df.Date.min(), "Estimation start": est.Date.min(),
    "Estimation end": est.Date.max(), "Estimation observations": len(est),
    "Annual ERP benchmark": .05,
    "Earnings-season windows": "Jan 15-Feb 15; Apr 15-May 15; Jul 15-Aug 15; Oct 15-Nov 15",
    "Season classification": "Based on target return date (t+1); y_t=r_(t+1)-rf_t",
    "HAC lag rule": "round(N^0.25), calculated separately for each regression using its complete-case N",
    "Half-lives": str(HALF_LIVES),
    "Estimation sample": "VIX observed and y non-missing; common to all regressions",
    "Signal construction": "Full extended dataset, including pre-VIX observations",
    "Cubic specification": "Original: all four cubic terms. Refined: drops x+3 and x-3; retains Mx+3 and Mx-3. Non-interacted: x+, x+3, x-, x-3 only; no VIX terms.",
    "Cubic Wald tests": "All Wald tests use HAC covariance, including the original, refined, and non-interacted cubic models. For each test and half-life, effective df is the numerical rank of R @ Cov(HAC) @ R.T; p-values use the corresponding chi-square distribution. Both requested restrictions and effective df are reported.",
    "Rounding": f"{SIG} significant digits"
})
with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
    for sheet in sheets:
        pd.DataFrame(out[sheet]).to_excel(writer, sheet_name=sheet, index=False)
print(f"Saved workbook: {OUT}")
print(f"Saved graphs: {GRAPH_DIR}")
