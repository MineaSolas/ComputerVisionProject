from src.result_analysis_helpers import *
from src.training_and_evaluation import *


def plot_group_summary(df, group_col, metric="cv_mae_mean", agg="best", title=None):
    agg_map = {
        "best": "min" if metric in LOWER_IS_BETTER_METRICS else "max",
        "mean": "mean",
        "median": "median",
    }
    if agg not in agg_map:
        raise ValueError(f"Unsupported agg: {agg}")

    summary = df.groupby(group_col)[metric].agg(agg_map[agg]).sort_values(ascending=metric_sort_direction(metric))

    plt.figure(figsize=(8, max(4, 0.45 * len(summary))))
    plt.barh(summary.index.astype(str), summary.values)
    plt.xlabel(f"{agg} {metric}")
    plt.ylabel(group_col)
    plt.title(title or f"{group_col} comparison ({agg} {metric})")
    plt.tight_layout()
    plt.show()

def plot_pairwise_heatmap(df, row_col, col_col, metric="cv_mae_mean", agg="min", title=None):
    agg_fn = {"min": "min", "max": "max", "mean": "mean", "median": "median"}[agg]
    table = pd.pivot_table(
        df,
        index=row_col,
        columns=col_col,
        values=metric,
        aggfunc=agg_fn,
    )

    fig, ax = plt.subplots(figsize=(1.2 * max(4, len(table.columns)), 0.8 * max(4, len(table.index))))
    im = ax.imshow(table.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(table.index)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iloc[i, j]
            text = "" if pd.isna(value) else f"{value:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    ax.set_title(title or f"{metric} heatmap ({agg} by {row_col} x {col_col})")
    fig.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    plt.show()

def plot_best_oof_predictions(results_df, oof_df, experiment_name, top_n=5):
    best_rows = results_df.nsmallest(top_n, "cv_mae_mean")
    print(f"Experiment: {experiment_name} (top {top_n} models)")

    for _, row in best_rows.iterrows():
        mask = (
            (oof_df["backbone"] == row["backbone"]) &
            (oof_df["fusion"] == row["fusion"]) &
            (oof_df["regressor"] == row["regressor"])
        )
        subset = oof_df[mask]

        if subset.empty:
            continue

        y_true = subset["y_true"].to_numpy(dtype=float)
        y_pred = subset["y_pred"].to_numpy(dtype=float)
        title = f"{row['backbone']} | {row['fusion']} | {row['regressor']}"
        make_oof_plot(y_true, y_pred, title_prefix=title)