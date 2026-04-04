from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torchvision import models
from tqdm import tqdm

from src.constants import *
from src.data_loading import *
from src.helpers import *

def extract_vit_features(vit_model, x):
    n = x.shape[0]
    x = vit_model._process_input(x)
    cls_token = vit_model.class_token.expand(n, -1, -1)
    x = torch.cat([cls_token, x], dim=1)
    x = vit_model.encoder(x)
    x = x[:, 0]
    return x

def get_finetune_backbone_spec(backbone_name, device):
    if backbone_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT
        backbone = models.densenet121(weights=weights)
        output_dim = 1024
        model_type = "cnn"
    elif backbone_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT
        backbone = models.vit_b_16(weights=weights)
        output_dim = 768
        model_type = "vit"
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    backbone = backbone.to(device)
    preprocess = weights.transforms()

    return {
        "name": backbone_name,
        "weights": weights,
        "preprocess": preprocess,
        "model": backbone,
        "output_dim": output_dim,
        "model_type": model_type,
    }

def unfreeze_last_stage(backbone_name, backbone):
    for p in backbone.parameters():
        p.requires_grad = False

    if backbone_name == "densenet121":
        for p in backbone.features.denseblock4.parameters():
            p.requires_grad = True
        for p in backbone.features.norm5.parameters():
            p.requires_grad = True

    elif backbone_name == "vit_b_16":
        for p in backbone.encoder.layers.encoder_layer_11.parameters():
            p.requires_grad = True
        for p in backbone.encoder.ln.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    return backbone

def backbone_forward(backbone_name, backbone, x):
    if backbone_name == "densenet121":
        x = backbone.features(x)
        x = torch.relu(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

    if backbone_name == "vit_b_16":
        return extract_vit_features(backbone, x)

    raise ValueError(f"Unsupported backbone: {backbone_name}")

def fused_dim_from_parts(part_dim, fusion_name):
    if fusion_name == "concat":
        return part_dim * 2
    if fusion_name == "concat_abs_diff":
        return part_dim * 3
    raise ValueError(f"Unsupported fusion: {fusion_name}")

def fuse_features(top_feat, side_feat, fusion_name):
    if fusion_name == "concat":
        return torch.cat([top_feat, side_feat], dim=1)
    if fusion_name == "concat_abs_diff":
        return torch.cat([top_feat, side_feat, torch.abs(top_feat - side_feat)], dim=1)
    raise ValueError(f"Unsupported fusion: {fusion_name}")

def make_head(head_name, input_dim, config):
    if head_name == "linear":
        return nn.Linear(input_dim, 1)

    if head_name == "mlp":
        hidden_dims = config["hidden_dims"]
        dropout = config.get("dropout", 0.0)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    raise ValueError(f"Unsupported head: {head_name}")

def select_top_mask_path(top_path, top_mask_path1=None, top_mask_path2=None):
    if top_mask_path1 is None or top_mask_path2 is None:
        return None

    top_path = Path(top_path)
    if top_path.name <= "P2050047.JPG":
        return top_mask_path1
    return top_mask_path2

# def load_pair_batch(
#     batch_df,
#     preprocess,
#     device,
#     side_mask_paths=None,
#     top_mask_paths=None,
#     image_cache=None,
#     mask_cache=None,
# ):
#     top_tensors = []
#     side_tensors = []
#     targets = []
#
#     image_cache = image_cache if image_cache is not None else {}
#     mask_cache = mask_cache if mask_cache is not None else {}
#
#     resolved_side_mask_paths = _normalize_mask_paths(side_mask_paths)
#     resolved_top_mask_paths = _normalize_mask_paths(top_mask_paths)
#
#     for _, row in batch_df.iterrows():
#         top_path = Path(row["top_path"])
#         side_path = Path(row["side_path"])
#
#         # top_img = Image.open(top_path).convert("RGB")
#         if top_path in image_cache:
#             top_img = image_cache[top_path].copy()
#         else:
#             top_img = Image.open(top_path).convert("RGB")
#             if len(image_cache) < IMAGE_CACHE_MAX:
#                 image_cache[top_path] = top_img.copy()
#
#         # side_img = Image.open(side_path).convert("RGB")
#         if side_path in image_cache:
#             side_img = image_cache[side_path].copy()
#         else:
#             side_img = Image.open(side_path).convert("RGB")
#             if len(image_cache) < IMAGE_CACHE_MAX:
#                 image_cache[side_path] = side_img.copy()
#
#         top_mask_path = resolve_mask_path(top_path, resolved_top_mask_paths)
#         if top_mask_path:
#             # mask = cv2.imread(str(top_mask_path), cv2.IMREAD_GRAYSCALE)
#             if top_mask_path in mask_cache:
#                 mask = mask_cache[top_mask_path]
#             else:
#                 mask = cv2.imread(str(top_mask_path), cv2.IMREAD_GRAYSCALE)
#                 mask_cache[top_mask_path] = mask
#
#             if mask is not None:
#                 if mask.shape[:2] != (top_img.height, top_img.width):
#                     mask = cv2.resize(mask, (top_img.width, top_img.height), interpolation=cv2.INTER_NEAREST)
#                 top_img_np = np.array(top_img)
#                 top_img_np[mask == 0] = 0
#                 top_img = Image.fromarray(top_img_np)
#
#         side_mask_path_resolved = resolve_mask_path(side_path, resolved_side_mask_paths)
#         if side_mask_path_resolved:
#             # mask = cv2.imread(str(side_mask_path_resolved), cv2.IMREAD_GRAYSCALE)
#             if side_mask_path_resolved in mask_cache:
#                 mask = mask_cache[side_mask_path_resolved]
#             else:
#                 mask = cv2.imread(str(side_mask_path_resolved), cv2.IMREAD_GRAYSCALE)
#                 mask_cache[side_mask_path_resolved] = mask
#
#             if mask is not None:
#                 if mask.shape[:2] != (side_img.height, side_img.width):
#                     mask = cv2.resize(mask, (side_img.width, side_img.height), interpolation=cv2.INTER_NEAREST)
#                 side_img_np = np.array(side_img)
#                 side_img_np[mask == 0] = 0
#                 side_img = Image.fromarray(side_img_np)
#
#         top_tensors.append(preprocess(top_img))
#         side_tensors.append(preprocess(side_img))
#         targets.append(float(row["volume"]))
#
#     top_batch = torch.stack(top_tensors).to(device)
#     side_batch = torch.stack(side_tensors).to(device)
#     y_batch = torch.tensor(targets, dtype=torch.float32, device=device)
#     return top_batch, side_batch, y_batch
#
#
# def iterate_batches(
#     samples_df,
#     indices,
#     preprocess,
#     batch_size,
#     shuffle,
#     seed,
#     device,
#     side_mask_paths=None,
#     top_mask_paths=None,
#     image_cache=None,
#     mask_cache=None
# ):
#     idx = np.array(indices, dtype=int).copy()
#     if shuffle:
#         rng = np.random.default_rng(seed)
#         rng.shuffle(idx)
#
#     for start in range(0, len(idx), batch_size):
#         batch_idx = idx[start:start + batch_size]
#         batch_df = samples_df.iloc[batch_idx]
#         yield load_pair_batch(
#             batch_df,
#             preprocess,
#             device,
#             side_mask_paths=side_mask_paths,
#             top_mask_paths=top_mask_paths,
#             image_cache=image_cache,
#             mask_cache=mask_cache
#         )

def make_optimizer(backbone, head, config):
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = list(head.parameters())

    param_groups = []
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": config["lr_backbone"],
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": config["lr_head"],
        })

    optimizer = torch.optim.Adam(
        param_groups,
        weight_decay=config.get("weight_decay", 0.0),
    )
    return optimizer

def predict_indices(
    samples_df,
    indices,
    preprocess,
    backbone_name,
    backbone,
    head,
    fusion_name,
    batch_size,
    device,
    side_mask_paths=None,
    top_mask_paths=None,
    image_cache=None,
    mask_cache=None
):
    backbone.eval()
    head.eval()

    y_true_all = []
    y_pred_all = []

    loader = make_dataloader(
        samples_df=samples_df,
        indices=indices,
        preprocess=preprocess,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        side_mask_paths=side_mask_paths,
        top_mask_paths=top_mask_paths,
        num_workers=NUM_WORKERS,
    )

    with torch.no_grad():
        for top_batch, side_batch, y_batch in loader:
            top_batch = top_batch.to(device)
            side_batch = side_batch.to(device)
            y_batch = y_batch.to(device)

            top_feat = backbone_forward(backbone_name, backbone, top_batch)
            side_feat = backbone_forward(backbone_name, backbone, side_batch)
            fused = fuse_features(top_feat, side_feat, fusion_name)
            pred = head(fused).squeeze(1)

            y_true_all.extend(y_batch.detach().cpu().numpy().tolist())
            y_pred_all.extend(pred.detach().cpu().numpy().tolist())

    return np.asarray(y_true_all, dtype=float), np.asarray(y_pred_all, dtype=float)

def train_single_fold(
    samples_df,
    train_idx,
    val_idx,
    backbone_name,
    fusion_name,
    head_name,
    config,
    max_epochs,
    batch_size,
    device,
    seed,
    side_mask_paths=None,
    top_mask_paths=None,
    patience=PATIENCE,
    min_delta=0.0,
    image_cache=None,
    mask_cache=None,
    use_early_stopping = True
):
    seed_everything(seed)

    # image_cache = image_cache if image_cache is not None else {}
    mask_cache = mask_cache if mask_cache is not None else {}

    backbone_spec = get_finetune_backbone_spec(backbone_name, device)
    preprocess = backbone_spec["preprocess"]
    backbone = backbone_spec["model"]
    backbone = unfreeze_last_stage(backbone_name, backbone)

    input_dim = fused_dim_from_parts(backbone_spec["output_dim"], fusion_name)
    head = make_head(head_name, input_dim, config).to(device)

    optimizer = make_optimizer(backbone, head, config)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_mae = np.inf
    epochs_without_improvement = 0
    history = []

    epoch_bar = tqdm(
        range(1, max_epochs + 1),
        desc=f"### Epochs",
        leave=False,
    )

    for epoch in epoch_bar:
        backbone.train()
        head.train()
        batch_losses = []

        train_loader = make_dataloader(
            samples_df=samples_df,
            indices=train_idx,
            preprocess=preprocess,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
            side_mask_paths=side_mask_paths,
            top_mask_paths=top_mask_paths,
            num_workers=NUM_WORKERS,
        )

        total_batches = len(train_loader)

        for batch_idx, (top_batch, side_batch, y_batch) in enumerate(train_loader, start=1):
            top_batch = top_batch.to(device)
            side_batch = side_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            top_feat = backbone_forward(backbone_name, backbone, top_batch)
            side_feat = backbone_forward(backbone_name, backbone, side_batch)
            fused = fuse_features(top_feat, side_feat, fusion_name)
            pred = head(fused).squeeze(1)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            batch_losses.append(loss_value)
            # print(f"    epoch={epoch}, batch={batch_idx}/{total_batches}, loss={loss_value:.4f}")

        val_true, val_pred = predict_indices(
            samples_df=samples_df,
            indices=val_idx,
            preprocess=preprocess,
            backbone_name=backbone_name,
            backbone=backbone,
            head=head,
            fusion_name=fusion_name,
            batch_size=batch_size,
            device=device,
            side_mask_paths=side_mask_paths,
            top_mask_paths=top_mask_paths,
            image_cache=image_cache,
            mask_cache=mask_cache
        )
        val_mae = mean_absolute_error(val_true, val_pred)
        train_loss = float(np.mean(batch_losses)) if batch_losses else np.nan

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": float(val_mae),
        })

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_mae=f"{val_mae:.4f}",
            best_val_mae=f"{best_val_mae:.4f}",
        )

        improved = val_mae < (best_val_mae - min_delta)
        if improved:
            best_val_mae = float(val_mae)
            epochs_without_improvement = 0
            best_state = {
                "backbone": {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()},
                "regressor": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }
        else:
            epochs_without_improvement += 1
            if use_early_stopping and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}: train_loss={train_loss:.4f}, val_mae={val_mae:.4f}")
                break

    if use_early_stopping and best_state is not None:
        backbone.load_state_dict(best_state["backbone"])
        head.load_state_dict(best_state["regressor"])
        backbone.to(device)
        head.to(device)

    return backbone_spec, backbone, head, best_val_mae, history

def run_end_to_end_grid_search_cv(
    samples_df,
    backbone_name,
    fusion_name,
    head_configs,
    outer_splitter,
    inner_groups,
    outer_name="CV",
    inner_splits=5,
    batch_size=BATCH_SIZE,
    max_epochs=MAX_EPOCHS,
    device=DEVICE,
    side_mask_paths=None,
    top_mask_paths=None,
    patience=PATIENCE,
    min_delta=0.0,
    image_cache=None,
    mask_cache=None
):
    y = samples_df["volume"].to_numpy(dtype=float)
    results = {}
    oof_predictions = {}
    training_histories = {}

    # image_cache = image_cache if image_cache is not None else {}
    # mask_cache = mask_cache if mask_cache is not None else {}

    for head_name, config_grid in head_configs.items():
        fold_records = []
        oof_pred = np.full(len(samples_df), np.nan, dtype=float)
        training_histories[head_name] = []

        print(f"\n{'='*70}\n{outer_name} | {backbone_name} | {fusion_name} | {head_name}\n{'='*70}")

        fold_bar = tqdm(
            outer_splitter,
            desc=f"Outer Folds",
            total=len(outer_splitter),
            leave=True,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(fold_bar, start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            g_train = inner_groups[train_idx]

            unique_inner_groups = np.unique(g_train)
            n_inner = min(inner_splits, len(unique_inner_groups))
            if n_inner < 2:
                raise ValueError("Need at least 2 groups in the training fold for inner CV.")

            inner_cv = GroupKFold(n_splits=n_inner)
            inner_splits_list = list(inner_cv.split(train_idx, y[train_idx], g_train))

            best_config = None
            best_inner_mae = np.inf
            best_inner_histories = None

            config_bar = tqdm(
                config_grid,
                desc=f"# Configs | Outer Fold {fold_idx}/{len(outer_splitter)}",
                leave=False,
                total=len(config_grid),
            )

            for config_idx, config in enumerate(config_bar, start=1):
                inner_fold_maes = []
                inner_fold_histories = []

                inner_fold_bar = tqdm(
                    inner_splits_list,
                    desc=f"## Inner Folds | Outer Fold {fold_idx}/{len(outer_splitter)} | Config {config_idx}/{len(config_grid)}",
                    leave=False,
                    total=len(inner_splits_list),
                )

                for inner_fold_idx, (inner_tr_rel, inner_val_rel) in enumerate(inner_fold_bar, start=1):
                    inner_tr_idx = train_idx[inner_tr_rel]
                    inner_val_idx = train_idx[inner_val_rel]

                    _, _, _, val_mae, history = train_single_fold(
                        samples_df=samples_df,
                        train_idx=inner_tr_idx,
                        val_idx=inner_val_idx,
                        backbone_name=backbone_name,
                        fusion_name=fusion_name,
                        head_name=head_name,
                        config=config,
                        max_epochs=max_epochs,
                        batch_size=batch_size,
                        device=device,
                        seed=RANDOM_STATE + fold_idx * 100 + inner_fold_idx,
                        side_mask_paths=side_mask_paths,
                        top_mask_paths=top_mask_paths,
                        patience=patience,
                        min_delta=min_delta,
                        image_cache=image_cache,
                        mask_cache=mask_cache
                    )
                    inner_fold_maes.append(val_mae)
                    inner_fold_histories.append(history)

                mean_inner_mae = float(np.mean(inner_fold_maes))
                config_bar.set_postfix(
                    config_idx=f"{config_idx}/{len(config_grid)}",
                    mean_inner_mae=f"{mean_inner_mae:.4f}",
                )
                if mean_inner_mae < best_inner_mae:
                    best_inner_mae = mean_inner_mae
                    best_config = dict(config)
                    best_inner_histories = inner_fold_histories

            backbone_spec, backbone, head, _, final_history = train_single_fold(
                samples_df=samples_df,
                train_idx=train_idx,
                val_idx=test_idx,
                backbone_name=backbone_name,
                fusion_name=fusion_name,
                head_name=head_name,
                config=best_config,
                max_epochs=max_epochs,
                batch_size=batch_size,
                device=device,
                seed=RANDOM_STATE + fold_idx,
                side_mask_paths=side_mask_paths,
                top_mask_paths=top_mask_paths,
                patience=patience,
                min_delta=min_delta,
                image_cache=image_cache,
                mask_cache=mask_cache,
                use_early_stopping=False
            )

            y_test, y_pred = predict_indices(
                samples_df=samples_df,
                indices=test_idx,
                preprocess=backbone_spec["preprocess"],
                backbone_name=backbone_name,
                backbone=backbone,
                head=head,
                fusion_name=fusion_name,
                batch_size=batch_size,
                device=device,
                side_mask_paths=side_mask_paths,
                top_mask_paths=top_mask_paths,
                image_cache=image_cache,
                mask_cache=mask_cache
            )
            oof_pred[test_idx] = y_pred

            record = {
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "inner_MAE": best_inner_mae,
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R2": r2_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else np.nan,
                "best_params": best_config,
            }
            fold_records.append(record)
            training_histories[head_name].append({
                "fold": fold_idx,
                "inner_histories": best_inner_histories,
                "final_history": final_history,
            })

            fold_bar.set_postfix(
                MAE=f"{record['MAE']:.3f}",
                RMSE=f"{record['RMSE']:.3f}",
                best=f"{best_config}",
            )

            print(
                f"Outer Fold {fold_idx}: "
                f"MAE={record['MAE']:.3f}, RMSE={record['RMSE']:.3f}, "
                f"R2={record['R2']}, best={best_config}"
            )

        results[head_name] = fold_records
        oof_predictions[head_name] = oof_pred

    return results, oof_predictions, training_histories
def run_end_to_end_nested_cv(
    samples_df,
    backbone_name,
    fusion_name,
    head_configs,
    outer_splits,
    inner_splits,
    batch_size=BATCH_SIZE,
    max_epochs=MAX_EPOCHS,
    device=DEVICE,
    side_mask_paths=None,
    top_mask_paths=None,
    patience=PATIENCE,
    min_delta=MIN_DELTA,
    image_cache=None,
    mask_cache=None
):
    groups = samples_df["exp_id"].to_numpy()
    y = samples_df["volume"].to_numpy(dtype=float)
    dummy_X = np.zeros((len(samples_df), 1), dtype=float)

    outer_cv = GroupKFold(n_splits=outer_splits)
    outer_splitter = list(outer_cv.split(dummy_X, y, groups))

    # image_cache = image_cache if image_cache is not None else {}
    # mask_cache = mask_cache if mask_cache is not None else {}

    return run_end_to_end_grid_search_cv(
        samples_df=samples_df,
        backbone_name=backbone_name,
        fusion_name=fusion_name,
        head_configs=head_configs,
        outer_splitter=outer_splitter,
        inner_groups=groups,
        outer_name="GroupKFold",
        inner_splits=inner_splits,
        batch_size=batch_size,
        max_epochs=max_epochs,
        device=device,
        side_mask_paths=side_mask_paths,
        top_mask_paths=top_mask_paths,
        patience=patience,
        min_delta=min_delta,
        image_cache=image_cache,
        mask_cache=mask_cache
    )
