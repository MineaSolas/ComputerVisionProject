import seaborn as sns
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

def plot_best_oof_predictions(results_df, oof_df, experiment_name, top_n=5, color="black"):
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
        make_oof_plot(y_true, y_pred, title_prefix=title, color=color)

def plot_combined_best_methods(results_dict, metric="MAE"):
    plot_data = []

    # Iterate through all experiments to find the best method for each
    for experiment_key, experiment_data in results_dict.items():
        best_method = None
        best_mean_score = float('inf')
        best_method_scores = []

        for method, folds in experiment_data.items():
            scores = [fold[metric] for fold in folds]
            mean_score = np.mean(scores)

            if mean_score < best_mean_score:
                best_mean_score = mean_score
                best_method = method
                best_method_scores = scores

        clean_exp_name = experiment_key.split('__')[-1].replace('_', ' ').title()
        x_label = f"{clean_exp_name}\n({best_method.capitalize()})"

        for score in best_method_scores:
            plot_data.append({
                "Experiment": x_label,
                metric: score
            })

    # Convert to DataFrame for easy Seaborn plotting
    df = pd.DataFrame(plot_data)
    results = df.groupby('Experiment')['MAE'].agg(['mean', 'std'])
    print(results.to_string())
    print(f"Mean {metric}: {results['mean'].mean():.2f} +/- {results['std'].mean():.2f}")

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    palette = ["#90C2E7", "#4E8098", "#E63946"]  # Light blue, Dark blue, Red/Orange

    # Plot the boxplot
    # Change whis=(0, 100) to force whiskers to the 0th and 100th percentiles (min and max)
    ax = sns.boxplot(x="Experiment", y=metric, data=df,
                     width=0.4, palette=palette, fliersize=0, whis=(0, 100))

    # Add the individual fold points on top
    sns.stripplot(x="Experiment", y=metric, data=df,
                  color="black", size=6, jitter=False, ax=ax, alpha=0.7)

    plt.ylabel(f"{metric} (Liters)", fontsize=12)
    plt.xlabel("Camera View Setup", fontsize=12)
    plt.xticks(fontsize=11)

    y_min, y_max = df[metric].min(), df[metric].max()
    padding = (y_max - y_min) * 0.15
    plt.ylim(y_min - padding, y_max + padding)

    plt.tight_layout()

    # plt.savefig("combined_view_comparison.png", dpi=300)
    plt.show()