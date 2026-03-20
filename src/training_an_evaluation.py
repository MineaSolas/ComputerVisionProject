import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV

def run_nested_cv(X, y, groups, model_configs, outer_splits, inner_splits):
    outer_cv = GroupKFold(n_splits=outer_splits)
    nested_results = {}
    oof_predictions = {}

    for model_name, (pipeline, param_grid) in model_configs.items():
        fold_records = []
        oof_pred = np.full(len(y), np.nan)
        print(f"\n{'='*70}\n{model_name}\n{'='*70}")

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            g_train = groups[train_idx]

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
                "R2": r2_score(y_test, y_pred),
                "best_params": best_params,
            }
            fold_records.append(record)
            print(
                f"Fold {fold_idx}: MAE={record['MAE']:.3f}, RMSE={record['RMSE']:.3f}, "
                f"R2={record['R2']:.3f}, best={best_params}"
            )

        nested_results[model_name] = fold_records
        oof_predictions[model_name] = oof_pred

    return nested_results, oof_predictions


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
