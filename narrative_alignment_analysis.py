"""
narrative_alignment_analysis.py
================================
Temporal narrative alignment analysis for:
  "Tracing Narrative Alignment Over Time Across Media Ecosystems:
   A Multilingual Computational Approach to Russian and Colombian Coverage"

This script reproduces the weekly narrative frequency and sentiment
time-series analysis reported in the paper, including lagged cross-series
correlations, Granger causality tests, and Benjamini-Hochberg FDR correction.

INPUT
-----
A CSV file with one row per article, containing at minimum:
    - date_adj       : publication date (parseable by pd.to_datetime)
    - country        : "Colombia" or "Russia"
    - Sentiment_adj  : integer sentiment score on [-2, -1, 0, 1, 2] scale
    - Macro          : list-valued column of macro-narrative labels
                       (stored as a Python list literal string, or already
                        a list if loaded from a pickle / parquet)
    - NER_Trigraph   : list-valued column of country trigraph codes
                       e.g. ["USA", "UKR", "RUS"]

  If you are loading from MongoDB, see the --mongo flag and the
  MONGODB NOTES section at the bottom of this file.

OUTPUT
------
A CSV file (default: weekly_narrative_results_bh.csv) with one row per
(year, trigraph, macro-narrative, metric) combination, including:
    - OLS trend statistics for Colombia and Russia series separately
    - Best-lag Pearson correlation between the two series
    - Granger causality test results (LR and F-test) at lags 1–4
    - Benjamini-Hochberg corrected p-values for all tests

USAGE
-----
  # From a pre-processed CSV:
  python narrative_alignment_analysis.py --input articles.csv

  # Specifying output path and analysis years:
  python narrative_alignment_analysis.py \
      --input articles.csv \
      --output results/my_results.csv \
      --years 2022 2023

  # Limiting to specific macro-narratives and trigraphs:
  python narrative_alignment_analysis.py \
      --input articles.csv \
      --macros "security and conflict" "diplomacy" "economy" \
      --trigraphs USA UKR RUS

REQUIREMENTS
------------
  pandas >= 1.5
  numpy
  scipy
  statsmodels >= 0.14

MONGODB NOTES
-------------
  The original analysis drew data from a local MongoDB instance.
  To replicate from MongoDB, export your collection to CSV first:

    mongoexport \
        --uri "mongodb://127.0.0.1:27017" \
        --db NMR \
        --collection RUS-COL \
        --type csv \
        --fields date,published_date,country,Sentiment,Macro,NER_Trigraph \
        --out articles_raw.csv

  Then pre-process the Sentiment column (map 1->-2, 2->-1, 3->0, 4->1, 5->2)
  and parse the Macro / NER_Trigraph list fields before passing to this script.
  See the load_data() function below for the expected format.
"""

import argparse
import ast
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import InfeasibleTestError

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


# =============================================================================
# Data loading
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Load and minimally validate the article-level input CSV.

    List-valued columns (Macro, NER_Trigraph) may be stored as Python
    literal strings (e.g. "['USA', 'RUS']") and are parsed automatically.
    """
    df = pd.read_csv(path)

    required = {"date_adj", "country", "Sentiment_adj", "Macro", "NER_Trigraph"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}\n"
            "See the INPUT section of this file's docstring for details."
        )

    df["date_adj"] = pd.to_datetime(df["date_adj"], errors="coerce")
    df = df[df["date_adj"].notna()].copy()

    # Parse list columns if stored as strings
    for col in ("Macro", "NER_Trigraph"):
        if df[col].dtype == object:
            df[col] = df[col].apply(_parse_list_field)

    df["Sentiment_adj"] = pd.to_numeric(df["Sentiment_adj"], errors="coerce")
    df = df[df["country"].isin(["Colombia", "Russia"])].copy()
    df = df[(df["date_adj"] >= "2013-01-01") & (df["date_adj"] < "2024-01-01")].copy()

    print(f"Loaded {len(df):,} articles from {path}")
    return df


def _parse_list_field(val):
    """Convert a stringified list to an actual list; leave lists unchanged."""
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else [result]
    except (ValueError, SyntaxError):
        return [str(val)]


# =============================================================================
# Weekly aggregation helpers
# =============================================================================

def safe_minmax(series: pd.Series) -> pd.Series:
    """Min-max scale to [0, 1]. Returns zeros if variance is zero."""
    s = pd.to_numeric(series, errors="coerce")
    smin, smax = s.min(), s.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - smin) / (smax - smin)


def safe_signed_scale(series: pd.Series) -> pd.Series:
    """Scale to [-1, 1] by max absolute value. Returns zeros if constant."""
    s = pd.to_numeric(series, errors="coerce")
    denom = s.abs().max()
    if pd.isna(denom) or denom == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return s / denom


def build_weekly_agg_for_macro_tri(
    dfT: pd.DataFrame,
    macro: str,
    tri: str
) -> pd.DataFrame:
    """
    Build weekly aggregated metrics for one macro-narrative + trigraph pair.

    Returns a dataframe with columns:
        date_adj, Country, Frequency, Sentiment_Agg, Sentiment_Mean,
        Log_Frequency, Log_Sentiment
    """
    dfTm = dfT.copy().explode("Macro")
    dfTm = dfTm[dfTm["Macro"] == macro].copy()

    dfT2 = dfTm.copy().explode("NER_Trigraph")
    dfT2 = dfT2[dfT2["NER_Trigraph"] == tri].copy()

    if dfT2.empty:
        return pd.DataFrame()

    weekly_freq = (
        dfT2.groupby([pd.Grouper(key="date_adj", freq="W"), "country"])["Sentiment_adj"]
        .count()
        .reset_index(name="Frequency")
    )
    sent_agg = (
        dfT2.groupby([pd.Grouper(key="date_adj", freq="W"), "country"])["Sentiment_adj"]
        .sum()
        .reset_index(name="Sentiment_Agg")
    )
    sent_mean = (
        dfT2.groupby([pd.Grouper(key="date_adj", freq="W"), "country"])["Sentiment_adj"]
        .mean()
        .reset_index(name="Sentiment_Mean")
    )

    weekly_agg = (
        weekly_freq
        .merge(sent_agg, on=["date_adj", "country"], how="outer")
        .merge(sent_mean, on=["date_adj", "country"], how="outer")
        .rename(columns={"country": "Country"})
        .sort_values(["Country", "date_adj"])
        .reset_index(drop=True)
    )

    out_parts = []
    for country_name, sub in weekly_agg.groupby("Country", sort=False):
        sub = sub.copy()
        sub["Log_Frequency"] = safe_minmax(sub["Frequency"])
        sub["Log_Sentiment"] = safe_signed_scale(sub["Sentiment_Agg"])
        out_parts.append(sub)

    return pd.concat(out_parts, ignore_index=True)


# =============================================================================
# Statistical analysis helpers
# =============================================================================

def fit_time_trend(series: pd.Series) -> dict:
    """OLS trend of a weekly series against sequential week index."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3 or s.nunique() <= 1:
        return dict(n_obs=len(s), slope=np.nan, intercept=np.nan,
                    r_squared=np.nan, p_value=np.nan, std_err=np.nan)
    x = np.arange(len(s), dtype=float)
    model = sm.OLS(s.values, sm.add_constant(x)).fit()
    return dict(
        n_obs=int(model.nobs),
        slope=float(model.params[1]),
        intercept=float(model.params[0]),
        r_squared=float(model.rsquared),
        p_value=float(model.pvalues[1]),
        std_err=float(model.bse[1]),
    )


def align_metric_pair(
    df_col: pd.DataFrame,
    df_rus: pd.DataFrame,
    metric: str
) -> pd.DataFrame:
    """Align Colombia and Russia weekly series on overlapping dates."""
    col = df_col[["date_adj", metric]].rename(columns={metric: "COL"}).copy()
    rus = df_rus[["date_adj", metric]].rename(columns={metric: "RUS"}).copy()
    return (
        col.merge(rus, on="date_adj", how="inner")
        .sort_values("date_adj")
        .dropna(subset=["COL", "RUS"])
        .reset_index(drop=True)
    )


def best_shifted_correlation(
    aligned: pd.DataFrame,
    max_shift: int = 4
) -> dict:
    """
    Find the lag in [-max_shift, max_shift] weeks that maximises the absolute
    Pearson correlation between the Russia and Colombia series.

    Positive lag: Russia leads Colombia.
    Negative lag: Colombia leads Russia.
    """
    if aligned.empty or len(aligned) < 4:
        return dict(best_lag=np.nan, best_r=np.nan, best_p=np.nan,
                    n_overlap=0, shift_skipped_reason="too_few_observations")

    best = dict(best_lag=np.nan, best_r=np.nan, best_p=np.nan,
                n_overlap=0, shift_skipped_reason="no_valid_lag")

    for lag in range(-max_shift, max_shift + 1):
        tmp = aligned.copy()
        if lag > 0:
            tmp["RUS_shifted"] = tmp["RUS"].shift(lag)
            pair = tmp.dropna(subset=["COL", "RUS_shifted"])
            x, y = pair["RUS_shifted"], pair["COL"]
        elif lag < 0:
            tmp["COL_shifted"] = tmp["COL"].shift(abs(lag))
            pair = tmp.dropna(subset=["COL_shifted", "RUS"])
            x, y = pair["RUS"], pair["COL_shifted"]
        else:
            pair = tmp.dropna(subset=["COL", "RUS"])
            x, y = pair["RUS"], pair["COL"]

        if len(pair) < 4:
            continue
        if x.nunique() <= 1 or y.nunique() <= 1:
            continue
        if np.isclose(x.std(ddof=1), 0.0) or np.isclose(y.std(ddof=1), 0.0):
            continue

        r, p = pearsonr(x, y)
        if pd.isna(best["best_r"]) or abs(r) > abs(best["best_r"]):
            best = dict(best_lag=lag, best_r=float(r), best_p=float(p),
                        n_overlap=int(len(pair)), shift_skipped_reason=None)

    return best


def run_granger_tests(
    aligned: pd.DataFrame,
    maxlag: int = 4,
    verbose: bool = False
) -> dict:
    """
    Granger causality tests for RUS -> COL direction.
    statsmodels convention: data matrix is [target, predictor].
    """
    out = dict(granger_tested=False, granger_skipped_reason=None,
               granger_n_obs=0, granger_maxlag_used=np.nan,
               granger_best_lag=np.nan, granger_best_lr_p=np.nan,
               granger_best_ftest_p=np.nan,
               granger_all_lr_p={}, granger_all_ftest_p={})

    if aligned.empty:
        out["granger_skipped_reason"] = "empty_aligned"
        return out

    gc_df = aligned[["COL", "RUS"]].dropna().copy()
    n_obs = len(gc_df)
    out["granger_n_obs"] = int(n_obs)

    if n_obs < 8:
        out["granger_skipped_reason"] = "too_few_observations"
        return out

    for col_name, reason in [("COL", "constant_COL"), ("RUS", "constant_RUS")]:
        if gc_df[col_name].nunique(dropna=True) <= 1:
            out["granger_skipped_reason"] = reason
            return out
        if np.isclose(gc_df[col_name].std(ddof=1), 0.0):
            out["granger_skipped_reason"] = f"near_{reason}"
            return out

    maxlag_used = min(maxlag, max(1, (n_obs // 3) - 1))
    out["granger_maxlag_used"] = int(maxlag_used)

    if maxlag_used < 1:
        out["granger_skipped_reason"] = "invalid_maxlag"
        return out

    try:
        results = grangercausalitytests(gc_df, maxlag=maxlag_used, verbose=verbose)
    except (InfeasibleTestError, ValueError) as e:
        out["granger_skipped_reason"] = f"test_error: {e}"
        return out

    lr_ps = {lag: float(res[0]["lrtest"][1]) for lag, res in results.items()}
    ftest_ps = {lag: float(res[0]["ssr_ftest"][1]) for lag, res in results.items()}
    best_lag = min(lr_ps, key=lr_ps.get)

    out.update(granger_tested=True, granger_skipped_reason=None,
               granger_best_lag=int(best_lag),
               granger_best_lr_p=float(lr_ps[best_lag]),
               granger_best_ftest_p=float(ftest_ps[best_lag]),
               granger_all_lr_p=lr_ps, granger_all_ftest_p=ftest_ps)
    return out


def analyze_metric_pair(
    df_col_year: pd.DataFrame,
    df_rus_year: pd.DataFrame,
    metric: str,
    year: int,
    tri: str,
    macro: str,
    max_ccf_lag: int = 4,
    max_granger_lag: int = 4,
    verbose: bool = False
) -> dict:
    """Analyze one (year, tri, macro, metric) combination."""
    col_trend = fit_time_trend(df_col_year[metric])
    rus_trend = fit_time_trend(df_rus_year[metric])
    aligned = align_metric_pair(df_col_year, df_rus_year, metric)
    shifted_corr = best_shifted_correlation(aligned, max_shift=max_ccf_lag)
    granger = run_granger_tests(aligned, maxlag=max_granger_lag, verbose=verbose)

    return {
        "year": year, "tri": tri, "macro": macro, "metric": metric,
        "col_n_obs": col_trend["n_obs"],
        "col_slope": col_trend["slope"],
        "col_intercept": col_trend["intercept"],
        "col_r_squared": col_trend["r_squared"],
        "col_trend_p": col_trend["p_value"],
        "col_trend_se": col_trend["std_err"],
        "rus_n_obs": rus_trend["n_obs"],
        "rus_slope": rus_trend["slope"],
        "rus_intercept": rus_trend["intercept"],
        "rus_r_squared": rus_trend["r_squared"],
        "rus_trend_p": rus_trend["p_value"],
        "rus_trend_se": rus_trend["std_err"],
        "aligned_n_obs": int(len(aligned)),
        "best_shift_lag": shifted_corr["best_lag"],
        "best_shift_r": shifted_corr["best_r"],
        "best_shift_p": shifted_corr["best_p"],
        "best_shift_n_overlap": shifted_corr["n_overlap"],
        "shift_skipped_reason": shifted_corr.get("shift_skipped_reason"),
        "granger_tested": granger["granger_tested"],
        "granger_skipped_reason": granger.get("granger_skipped_reason"),
        "granger_n_obs": granger["granger_n_obs"],
        "granger_maxlag_used": granger["granger_maxlag_used"],
        "granger_best_lag": granger["granger_best_lag"],
        "granger_best_lr_p": granger["granger_best_lr_p"],
        "granger_best_ftest_p": granger["granger_best_ftest_p"],
        "granger_all_lr_p": granger["granger_all_lr_p"],
        "granger_all_ftest_p": granger["granger_all_ftest_p"],
    }


def apply_bh_correction(
    results_df: pd.DataFrame,
    p_col: str,
    prefix: str
) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to one p-value column."""
    out = results_df.copy()
    valid = out[p_col].notna()
    out[f"{prefix}_bh_reject"] = False
    out[f"{prefix}_bh_p"] = np.nan
    if valid.sum() == 0:
        return out
    reject, p_adj, _, _ = multipletests(
        out.loc[valid, p_col].values, alpha=0.05, method="fdr_bh"
    )
    out.loc[valid, f"{prefix}_bh_reject"] = reject
    out.loc[valid, f"{prefix}_bh_p"] = p_adj
    return out


# =============================================================================
# Main analysis pipeline
# =============================================================================

def metric_analysis(
    dfT: pd.DataFrame,
    macros: list,
    trigraph_list: list = None,
    years=range(2022, 2024),
    metric_types: list = None,
    max_ccf_lag: int = 4,
    max_granger_lag: int = 4,
    verbose_granger: bool = False,
) -> pd.DataFrame:
    """
    Run the full analysis over all macros × trigraphs × years × metrics.

    Parameters
    ----------
    dfT : pd.DataFrame
        Article-level dataframe as returned by load_data().
    macros : list of str
        Macro-narrative labels to include.
    trigraph_list : list of str
        ISO trigraph codes for geopolitical targets (default: USA, UKR, RUS).
    years : iterable of int
        Calendar years to analyse (default: 2022, 2023).
    metric_types : list of str
        Metrics to compute. Defaults to all five available metrics.
    max_ccf_lag : int
        Maximum lag (weeks) for shifted correlation search.
    max_granger_lag : int
        Maximum lag for Granger causality tests.
    verbose_granger : bool
        Whether to print statsmodels Granger output.

    Returns
    -------
    pd.DataFrame with one row per (year, tri, macro, metric) combination,
    including BH-corrected p-values.
    """
    if trigraph_list is None:
        trigraph_list = ["USA", "UKR", "RUS"]
    if metric_types is None:
        metric_types = [
            "Frequency", "Sentiment_Agg", "Sentiment_Mean",
            "Log_Frequency", "Log_Sentiment",
        ]

    all_rows = []
    total = len(macros) * len(trigraph_list)
    done = 0

    for macro in macros:
        for tri in trigraph_list:
            done += 1
            print(f"  [{done}/{total}] macro={macro}, tri={tri}")
            weekly_agg = build_weekly_agg_for_macro_tri(dfT, macro, tri)
            if weekly_agg.empty:
                continue

            df_col = weekly_agg[weekly_agg["Country"] == "Colombia"].copy()
            df_rus = weekly_agg[weekly_agg["Country"] == "Russia"].copy()
            if df_col.empty or df_rus.empty:
                continue

            for year in years:
                start = pd.Timestamp(f"{year}-01-01")
                end = pd.Timestamp(f"{year + 1}-01-01")
                df_col_y = df_col[(df_col["date_adj"] >= start) & (df_col["date_adj"] < end)].copy()
                df_rus_y = df_rus[(df_rus["date_adj"] >= start) & (df_rus["date_adj"] < end)].copy()
                if df_col_y.empty or df_rus_y.empty:
                    continue

                for metric in metric_types:
                    df_col_m = df_col_y.dropna(subset=[metric]).copy()
                    df_rus_m = df_rus_y.dropna(subset=[metric]).copy()
                    if df_col_m.empty or df_rus_m.empty:
                        continue
                    row = analyze_metric_pair(
                        df_col_m, df_rus_m, metric, year, tri, macro,
                        max_ccf_lag, max_granger_lag, verbose_granger
                    )
                    all_rows.append(row)

    results_df = pd.DataFrame(all_rows)
    if results_df.empty:
        print("Warning: no results produced. Check your input data and parameters.")
        return results_df

    # BH correction across four p-value columns
    for p_col, prefix in [
        ("best_shift_p",      "shift_corr"),
        ("granger_best_lr_p", "granger_lr"),
        ("col_trend_p",       "col_trend"),
        ("rus_trend_p",       "rus_trend"),
    ]:
        results_df = apply_bh_correction(results_df, p_col, prefix)

    return results_df


# =============================================================================
# CLI entry point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Temporal narrative alignment analysis (Granger + BH correction).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to article-level CSV (see INPUT section in docstring)."
    )
    parser.add_argument(
        "--output", default="weekly_narrative_results_bh.csv",
        help="Output CSV path (default: weekly_narrative_results_bh.csv)."
    )
    parser.add_argument(
        "--macros", nargs="+",
        default=[
            "diplomacy", "economy",
            "security and conflict", "politics and society"
        ],
        help="Macro-narrative labels to include."
    )
    parser.add_argument(
        "--trigraphs", nargs="+", default=["USA", "UKR", "RUS"],
        help="Geopolitical target trigraphs."
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2022, 2023],
        help="Calendar years to analyse."
    )
    parser.add_argument(
        "--max-ccf-lag", type=int, default=4,
        help="Max lag (weeks) for shifted correlation (default: 4)."
    )
    parser.add_argument(
        "--max-granger-lag", type=int, default=4,
        help="Max lag for Granger tests (default: 4)."
    )
    parser.add_argument(
        "--verbose-granger", action="store_true",
        help="Print full statsmodels Granger output."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading data from: {args.input}")
    dfT = load_data(args.input)

    print(f"\nRunning analysis:")
    print(f"  Macros     : {args.macros}")
    print(f"  Trigraphs  : {args.trigraphs}")
    print(f"  Years      : {args.years}")
    print(f"  Max CCF lag: {args.max_ccf_lag}")
    print(f"  Max Granger lag: {args.max_granger_lag}")
    print()

    results = metric_analysis(
        dfT=dfT,
        macros=args.macros,
        trigraph_list=args.trigraphs,
        years=args.years,
        max_ccf_lag=args.max_ccf_lag,
        max_granger_lag=args.max_granger_lag,
        verbose_granger=args.verbose_granger,
    )

    if not results.empty:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        print(f"\nResults written to: {out_path}")
        print(f"  Total rows: {len(results)}")
        sig_raw = ((results["granger_best_lr_p"] < 0.05) |
                   (results["best_shift_p"] < 0.05)).sum()
        sig_bh = (results["granger_lr_bh_reject"] |
                  results["shift_corr_bh_reject"]).sum()
        print(f"  Significant (pre-FDR, p<.05): {sig_raw}/{len(results)}")
        print(f"  Significant (post-BH):         {sig_bh}/{len(results)}")
    else:
        print("No results to write.")
        sys.exit(1)


if __name__ == "__main__":
    main()
