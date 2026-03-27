import torch
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression
from pathlib import Path

def make_pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("model",   estimator),
    ])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 5
BATCH_SIZE = 8
MAX_EPOCHS = 20

CSV_PATH = Path("experiments/experiments_1_36.csv")
TOP_FOLDER = Path("photos/top_view_images")
SIDE_FOLDER = Path("photos/side_view_images")
EMBEDDING_CACHE_DIR = Path("embeddings")
EMBEDDING_CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

BACKBONE_NAMES = [
    # "resnet50",
    # "convnext_tiny",
    # "densenet121",
    "vit_b_16"
]

FUSION_NAMES = [
    "concat",
    # "mean",
    # "max",
    # "concat_abs_diff",
]

REGRESSION_MODEL_CONFIGS = {
    # "dummy_mean": (
    #     DummyRegressor(strategy="mean"),
    #     {},
    # ),

    "linear": (
        make_pipeline(LinearRegression()),
        {},
    ),

    # "ridge": (
    #     make_pipeline(Ridge()),
    #     {
    #         "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    #     },
    # ),
    #
    # "elasticnet": (
    #     make_pipeline(ElasticNet(max_iter=20000)),
    #     {
    #         "model__alpha":    [0.0001, 0.001, 0.01, 0.1, 1.0],
    #         "model__l1_ratio": [0.1, 0.2, 0.5, 0.8, 0.9],
    #     },
    # ),
    #
    # "random_forest": (
    #     Pipeline([
    #         ("imputer", SimpleImputer(strategy="mean")),
    #         ("model",   RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
    #     ]),
    #     {
    #         "model__n_estimators":    [100, 300, 500],
    #         # "model__max_depth":       [None, 5, 10]   # Seemed to always select None anyway
    #         "model__min_samples_leaf": [1, 2, 4],
    #         "model__max_features":    [0.3, 0.5, 1.0]
    #     },
    # ),

    "mlp": (
        make_pipeline(
            MLPRegressor(
                random_state=RANDOM_STATE,
                hidden_layer_sizes=(64,),
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
            )
        ),
        {
            # "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
            "model__alpha": [1e-4, 1e-3, 1e-2],
            "model__learning_rate_init": [1e-3, 3e-4],
        },
    )
}

TRAINING_CONFIGS = {
    "last_stage": [
        {"head_lr": 1e-3, "backbone_lr": 1e-5, "weight_decay": 1e-4},
        # {"head_lr": 3e-4, "backbone_lr": 3e-5, "weight_decay": 1e-4},
    ],
    # "bias": [
    #     {"head_lr": 1e-3, "backbone_lr": 3e-4, "weight_decay": 1e-4},
    #     {"head_lr": 3e-4, "backbone_lr": 1e-4, "weight_decay": 1e-4},
    # ],
    # "adapter": [
    #     {"head_lr": 1e-3, "adapter_lr": 1e-3, "weight_decay": 1e-4, "adapter_dim": 32},
    #     {"head_lr": 3e-4, "adapter_lr": 1e-3, "weight_decay": 1e-4, "adapter_dim": 64},
    # ],
}

HEAD_CONFIGS = {
    # "linear": {},
    "mlp": {"hidden_dim": 64, "dropout": 0.2},
}
