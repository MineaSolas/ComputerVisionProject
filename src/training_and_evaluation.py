import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV

# Generic nested evaluation
def run_grid_search_cv(
    X, y,
    model_configs,
    outer_splitter,
    inner_groups,
    outer_name="CV",
    inner_splits=5,
    fold_label_fn=None,
):
    results = {}
    oof_predictions = {}

    for model_name, (pipeline, param_grid) in model_configs.items():
        fold_records = []
        oof_pred = np.full(len(y), np.nan, dtype=float)

        print(f"\n{'='*70}\n{outer_name} | {model_name}\n{'='*70}")

        for fold_idx, (train_idx, test_idx) in enumerate(outer_splitter, start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            g_train = inner_groups[train_idx]

            if param_grid:
                n_inner = min(inner_splits, len(np.unique(g_train)))
                inner_cv = GroupKFold(n_splits=n_inner)

                gs = GridSearchCV(
                    estimator=clone(pipeline),
                    param_grid=param_grid,
                    cv=inner_cv,
                    scoring="neg_mean_absolute_error",
                    refit=True,
                    n_jobs=-1,
                )
                gs.fit(X_train, y_train, groups=g_train)
                final_model = gs.best_estimator_
                best_params = gs.best_params_
                inner_mae = -gs.best_score_
            else:
                final_model = clone(pipeline)
                final_model.fit(X_train, y_train)
                best_params = {}
                inner_mae = np.nan

            y_pred = final_model.predict(X_test)
            oof_pred[test_idx] = y_pred

            record = {
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "inner_MAE": inner_mae,
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R2": r2_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else np.nan,
                "best_params": best_params,
            }

            if fold_label_fn is not None:
                record.update(fold_label_fn(train_idx, test_idx, fold_idx))

            fold_records.append(record)

            extra = ""
            if "test_volume" in record:
                extra = f" | test_volume={record['test_volume']}"
            print(
                f"Fold {fold_idx}{extra}: "
                f"MAE={record['MAE']:.3f}, RMSE={record['RMSE']:.3f}, "
                f"R2={record['R2']}, best={best_params}"
            )

        results[model_name] = fold_records
        oof_predictions[model_name] = oof_pred

    return results, oof_predictions

def run_nested_cv(X, y, groups, model_configs, outer_splits, inner_splits):
    outer_cv = GroupKFold(n_splits=outer_splits)
    outer_splitter = list(outer_cv.split(X, y, groups))

    return run_grid_search_cv(
        X=X,
        y=y,
        model_configs=model_configs,
        outer_splitter=outer_splitter,
        inner_groups=groups,
        outer_name="GroupKFold",
        inner_splits=inner_splits,
    )

def summarise_nested_results(nested_results, backbone_name, fusion_name):
    rows = []
    for model_name, folds in nested_results.items():
        rows.append({
            "backbone": backbone_name,
            "fusion": fusion_name,
            "regressor": model_name,
            "cv_mae_mean": np.mean([f["MAE"] for f in folds]),
            "cv_mae_std": np.std([f["MAE"] for f in folds]),
            "cv_mse_mean": np.mean([f["MSE"] for f in folds]),
            "cv_mse_stc": np.std([f["MSE"] for f in folds]),
            "cv_rmse_mean": np.mean([f["RMSE"] for f in folds]),
            "cv_rmse_std": np.std([f["RMSE"] for f in folds]),
            "cv_r2_mean": np.mean([f["R2"] for f in folds]),
            "cv_r2_std": np.std([f["R2"] for f in folds]),
        })
    return pd.DataFrame(rows).sort_values("cv_mae_mean").reset_index(drop=True)

def make_oof_plot(y_true, y_pred, title_prefix=""):
    valid = ~np.isnan(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_max = np.max(y_pred[valid])
    pred_limit = max(100, int(x_max * 1.1))

    axes[0].scatter(y_true[valid], y_pred[valid], alpha=0.8)
    axes[0].plot([0, 100], [0, 100], "--")
    axes[0].set_xlabel("True volume")
    axes[0].set_ylabel("Predicted volume")
    axes[0].set_title(f"{title_prefix} | Out-of-fold True vs Predicted")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, pred_limit)

    counts, _, _ = axes[1].hist(y_pred[valid], bins=(round(0.2 * pred_limit)), range=(0, pred_limit))
    hist_height = max(36, int(np.max(counts) * 1.1))
    axes[1].set_xlabel("Predicted volume")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{title_prefix} | Prediction Distribution")
    axes[1].set_xlim(0, pred_limit)
    axes[1].set_ylim(0, hist_height)

    plt.tight_layout()
    plt.show()

def make_lovo_splits(volume_groups):
    unique_volumes = np.sort(np.unique(volume_groups))
    splits = []

    for vol in unique_volumes:
        test_idx = np.where(volume_groups == vol)[0]
        train_idx = np.where(volume_groups != vol)[0]
        splits.append((train_idx, test_idx))

    return splits

def run_lovo_cv(X, y, exp_groups, volume_groups, model_configs, inner_splits=5):
    outer_splitter = make_lovo_splits(volume_groups)

    def fold_label_fn(train_idx, test_idx, fold_idx):
        test_vols = np.unique(volume_groups[test_idx])
        return {"test_volume": test_vols[0]}

    return run_grid_search_cv(
        X=X,
        y=y,
        model_configs=model_configs,
        outer_splitter=outer_splitter,
        inner_groups=exp_groups,   # inner tuning still grouped by experiment
        outer_name="LOVO",
        inner_splits=inner_splits,
        fold_label_fn=fold_label_fn,
    )