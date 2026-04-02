import pandas as pd
from IPython.display import display
from src.constants import *

# True = glob pattern must match exactly one file
# False = the newest matching file is used
REQUIRE_UNIQUE_MATCH = False

SUMMARY_METRIC_COLUMNS = [
    "cv_mae_mean",
    "cv_mae_std",
    "cv_mse_mean",
    "cv_mse_std",
    "cv_rmse_mean",
    "cv_rmse_std",
    "cv_r2_mean",
    "cv_r2_std",
]

LOWER_IS_BETTER_METRICS = {"cv_mae_mean", "cv_mae_std", "cv_mse_mean", "cv_mse_std", "cv_rmse_mean", "cv_rmse_std"}
HIGHER_IS_BETTER_METRICS = {"cv_r2_mean", "cv_r2_std"}

def resolve_result_file(file_spec, results_dir=OUTPUT_DIR, require_unique_match=REQUIRE_UNIQUE_MATCH):
    file_path = Path(file_spec)

    candidates = []

    if file_path.exists():
        candidates = [file_path]
    else:
        candidate = results_dir / file_path.name
        if candidate.exists():
            candidates = [candidate]
        else:
            candidate = Path(file_spec)
            if candidate.exists():
                candidates = [candidate]

    if not candidates:
        raise FileNotFoundError(f"No file matched: {file_spec}")

    if len(candidates) > 1 and require_unique_match:
        raise ValueError(f"Multiple files matched {file_spec}: {[p.name for p in candidates]}")

    return candidates[0]

def load_result_csv(file_spec, experiment_name=None, results_dir=OUTPUT_DIR):
    path = Path(file_spec)

    if not path.exists():
        path = resolve_result_file(file_spec, results_dir=results_dir)

    df = pd.read_csv(path)

    missing = [col for col in ["backbone", "fusion", "regressor"] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    if experiment_name is not None and "experiment" not in df.columns:
        df = df.copy()
        df["experiment"] = experiment_name

    return df

def metric_sort_direction(metric):
    if metric in LOWER_IS_BETTER_METRICS:
        return True
    if metric in HIGHER_IS_BETTER_METRICS:
        return False
    raise ValueError(f"Unknown metric direction for: {metric}")

def sort_results(df, metric="cv_mae_mean", ascending=None, tie_breakers=None):
    if ascending is None:
        ascending = metric_sort_direction(metric)

    if tie_breakers is None:
        if ascending:
            tie_breakers = ["cv_rmse_mean", "cv_r2_mean"]
        else:
            tie_breakers = ["cv_mae_mean", "cv_rmse_mean"]

    sort_cols = [metric] + tie_breakers
    available_sort_cols = [c for c in sort_cols if c in df.columns]

    tmp = []
    for c in available_sort_cols:
        if c == metric:
            tmp.append(ascending)
        else:
            tmp.append(metric_sort_direction(c))

    return df.sort_values(available_sort_cols, ascending=tmp).reset_index(drop=True)

def filter_results(
    df,
    experiments=None,
    protocols=None,
    views=None,
    backbones=None,
    fusions=None,
    regressors=None,
):
    out = df.copy()

    if experiments is not None:
        out = out[out["experiment"].isin(experiments)]
    if backbones is not None:
        out = out[out["backbone"].isin(backbones)]
    if fusions is not None:
        out = out[out["fusion"].isin(fusions)]
    if regressors is not None:
        out = out[out["regressor"].isin(regressors)]

    return out.reset_index(drop=True)

def show_top(df, n=10, metric="cv_mae_mean", columns=None):
    ranked = sort_results(df, metric=metric).head(n)

    if columns is None:
        columns = [
            "experiment_name",
            "protocol",
            "view",
            "backbone",
            "fusion",
            "regressor",
            "cv_mae_mean",
            "cv_mse_mean",
            "cv_rmse_mean",
            "cv_r2_mean",
            "source_file",
        ]

    available = [c for c in columns if c in ranked.columns]
    display(ranked[available])

def summarize_by(df, group_col, metric="cv_mae_mean", top_k=10):
    top_df = sort_results(df, metric=metric).head(top_k)
    top_counts = top_df[group_col].value_counts()

    summary = (
        df.groupby(group_col)
        .agg(
            best_mae=("cv_mae_mean", "min"),
            mean_mae=("cv_mae_mean", "mean"),
            median_mae=("cv_mae_mean", "median"),
            best_rmse=("cv_rmse_mean", "min"),
            mean_rmse=("cv_rmse_mean", "mean"),
            best_r2=("cv_r2_mean", "max"),
            mean_r2=("cv_r2_mean", "mean"),
        )
        .reset_index()
    )

    summary[f"top_{top_k}_count"] = summary[group_col].map(top_counts).fillna(0).astype(int)
    summary = summary.sort_values(["best_mae", "mean_mae", "best_r2"], ascending=[True, True, False]).reset_index(drop=True)
    return summary