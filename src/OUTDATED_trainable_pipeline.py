import copy
import sys

import numpy as np
import torch
import cv2
from PIL import Image
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from torch import nn
from torchvision import models
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from helpers import resolve_mask_path, seed_everything


def _normalize_mask_paths(mask_paths=None):
    if mask_paths is None:
        return None
    if isinstance(mask_paths, (str, Path)):
        return [Path(mask_paths)]
    return [Path(p) for p in mask_paths if p is not None]


def get_backbone_and_preprocess(backbone_name, device):
    if backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)
        out_dim = 2048
    elif backbone_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        backbone = models.convnext_tiny(weights=weights)
        out_dim = 768
    elif backbone_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT
        backbone = models.densenet121(weights=weights)
        out_dim = 1024
    elif backbone_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT
        backbone = models.vit_b_16(weights=weights)
        out_dim = 768
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    preprocess = weights.transforms()
    backbone = backbone.to(device)
    return backbone, preprocess, out_dim

def backbone_forward(backbone_name, backbone, x):
    if backbone_name == "resnet50":
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)
        x = backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    if backbone_name == "convnext_tiny":
        x = backbone.features(x)
        x = backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    if backbone_name == "densenet121":
        x = backbone.features(x)
        x = F.relu(x, inplace=False)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

    if backbone_name == "vit_b_16":
        n = x.shape[0]
        x = backbone._process_input(x)
        cls_token = backbone.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = backbone.encoder(x)
        x = x[:, 0]
        return x

    raise ValueError(f"Unsupported backbone: {backbone_name}")

def get_fused_dim(feature_dim, fusion_name):
    if fusion_name in ["mean", "max"]:
        return feature_dim
    if fusion_name == "concat":
        return 2 * feature_dim
    if fusion_name == "concat_abs_diff":
        return 3 * feature_dim
    raise ValueError(f"Unsupported fusion: {fusion_name}")

def fuse_features(top_feat, side_feat, fusion_name):
    if fusion_name == "concat":
        return torch.cat([top_feat, side_feat], dim=1)
    if fusion_name == "mean":
        return 0.5 * (top_feat + side_feat)
    if fusion_name == "max":
        return torch.maximum(top_feat, side_feat)
    if fusion_name == "concat_abs_diff":
        return torch.cat([top_feat, side_feat, torch.abs(top_feat - side_feat)], dim=1)
    raise ValueError(f"Unsupported fusion: {fusion_name}")

def build_head(fused_dim, head_name, head_cfg, device):
    if head_name == "linear":
        return nn.Linear(fused_dim, 1).to(device)
    if head_name == "mlp":
        hidden_dim = head_cfg.get("hidden_dim", 64)
        dropout = head_cfg.get("dropout", 0.2)
        return nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        ).to(device)
    raise ValueError(f"Unsupported head: {head_name}")

def freeze_all(backbone):
    for p in backbone.parameters():
        p.requires_grad = False

def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True

def enable_bias_only(backbone):
    for name, p in backbone.named_parameters():
        p.requires_grad = name.endswith("bias")

def register_conv_adapter_hooks(modules, out_channels, adapter_dim, adapter_modules, hook_handles, device):
    for module in modules:
        adapter = nn.Sequential(
            nn.Conv2d(out_channels, adapter_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(adapter_dim, out_channels, kernel_size=1),
        ).to(device)
        adapter_modules.append(adapter)

        def hook(_module, _inputs, output, adapter=adapter):
            return output + adapter(output)

        hook_handles.append(module.register_forward_hook(hook))

def register_token_adapter_hooks(modules, token_dim, adapter_dim, adapter_modules, hook_handles, device):
    for module in modules:
        adapter = nn.Sequential(
            nn.Linear(token_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, token_dim),
        ).to(device)
        adapter_modules.append(adapter)

        def hook(_module, _inputs, output, adapter=adapter):
            return output + adapter(output)

        hook_handles.append(module.register_forward_hook(hook))

def configure_backbone_training(backbone_name, backbone, mode_name, cfg, device):
    freeze_all(backbone)
    adapter_modules = []
    hook_handles = []

    if mode_name == "last_stage":
        if backbone_name == "resnet50":
            unfreeze_module(backbone.layer4)
        elif backbone_name == "convnext_tiny":
            unfreeze_module(backbone.features[-1])
        elif backbone_name == "densenet121":
            unfreeze_module(backbone.features.denseblock4)
            unfreeze_module(backbone.features.norm5)
        elif backbone_name == "vit_b_16":
            unfreeze_module(backbone.encoder.layers[-1])
        else:
            raise ValueError(backbone_name)

    elif mode_name == "bias":
        enable_bias_only(backbone)

    elif mode_name == "adapter":
        adapter_dim = cfg.get("adapter_dim", 32)
        if backbone_name == "resnet50":
            last_blocks = list(backbone.layer4)
            register_conv_adapter_hooks(last_blocks, 2048, adapter_dim, adapter_modules, hook_handles, device)
        elif backbone_name == "convnext_tiny":
            last_blocks = list(backbone.features[-1])
            register_conv_adapter_hooks(last_blocks, 768, adapter_dim, adapter_modules, hook_handles, device)
        elif backbone_name == "densenet121":
            register_conv_adapter_hooks([backbone.features.denseblock4], 1024, adapter_dim, adapter_modules, hook_handles, device)
        elif backbone_name == "vit_b_16":
            last_layers = [backbone.encoder.layers[-2], backbone.encoder.layers[-1]]
            register_token_adapter_hooks(last_layers, 768, adapter_dim, adapter_modules, hook_handles, device)
        else:
            raise ValueError(backbone_name)

    else:
        raise ValueError(f"Unsupported mode: {mode_name}")

    return adapter_modules, hook_handles

def cleanup_hooks(hook_handles):
    for handle in hook_handles:
        handle.remove()

def load_pair_batch(
    batch_df,
    preprocess,
    device,
    side_mask_paths=None,
    top_mask_paths=None,
):
    top_tensors = []
    side_tensors = []
    targets = []

    resolved_side_mask_paths = _normalize_mask_paths(side_mask_paths)
    resolved_top_mask_paths = _normalize_mask_paths(top_mask_paths)

    for _, row in batch_df.iterrows():
        top_img = Image.open(row["top_path"]).convert("RGB")
        side_img = Image.open(row["side_path"]).convert("RGB")

        # Apply top mask if provided
        top_path = Path(row["top_path"])
        top_mask_path = resolve_mask_path(top_path, resolved_top_mask_paths)

        if top_mask_path:
            mask = cv2.imread(str(top_mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[:2] != (top_img.height, top_img.width):
                    mask = cv2.resize(mask, (top_img.width, top_img.height), interpolation=cv2.INTER_NEAREST)
                top_img_np = np.array(top_img)
                top_img_np[mask == 0] = 0
                top_img = Image.fromarray(top_img_np)

        # Apply side mask if provided
        side_mask_path_resolved = resolve_mask_path(row["side_path"], resolved_side_mask_paths)
        if side_mask_path_resolved:
            mask = cv2.imread(str(side_mask_path_resolved), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[:2] != (side_img.height, side_img.width):
                    mask = cv2.resize(mask, (side_img.width, side_img.height), interpolation=cv2.INTER_NEAREST)
                side_img_np = np.array(side_img)
                side_img_np[mask == 0] = 0
                side_img = Image.fromarray(side_img_np)

        top_tensors.append(preprocess(top_img))
        side_tensors.append(preprocess(side_img))
        targets.append(float(row["volume"]))

    top_batch = torch.stack(top_tensors).to(device)
    side_batch = torch.stack(side_tensors).to(device)
    y_batch = torch.tensor(targets, dtype=torch.float32, device=device)
    return top_batch, side_batch, y_batch

def iterate_batches(
    samples_df,
    indices,
    preprocess,
    batch_size,
    shuffle,
    seed,
    device,
    side_mask_paths=None,
    top_mask_paths=None,
):
    idx = np.array(indices, dtype=int).copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        batch_df = samples_df.iloc[batch_idx]
        yield load_pair_batch(
            batch_df,
            preprocess,
            device,
            side_mask_paths=side_mask_paths,
            top_mask_paths=top_mask_paths,
        )

def make_group_train_val_split(indices, groups, seed):
    unique_groups = np.unique(groups[indices])
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    n_val_groups = max(1, int(round(0.2 * len(unique_groups))))
    val_groups = set(shuffled[:n_val_groups])

    train_idx = np.array([i for i in indices if groups[i] not in val_groups], dtype=int)
    val_idx = np.array([i for i in indices if groups[i] in val_groups], dtype=int)

    if len(val_idx) == 0 or len(train_idx) == 0:
        splitter = GroupKFold(n_splits=min(2, len(unique_groups)))
        train_idx, val_idx = next(splitter.split(np.zeros((len(indices), 1)), groups[indices], groups[indices]))
        train_idx = indices[train_idx]
        val_idx = indices[val_idx]

    return train_idx, val_idx

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
):
    backbone.eval()
    head.eval()
    preds = []
    ys = []

    with torch.no_grad():
        for top_batch, side_batch, y_batch in iterate_batches(
            samples_df,
            indices,
            preprocess,
            batch_size,
            shuffle=False,
            seed=0,
            device=device,
            side_mask_paths=side_mask_paths,
            top_mask_paths=top_mask_paths,
        ):
            top_feat = backbone_forward(backbone_name, backbone, top_batch)
            side_feat = backbone_forward(backbone_name, backbone, side_batch)
            fused = fuse_features(top_feat, side_feat, fusion_name)
            pred = head(fused).squeeze(1)
            preds.append(pred.detach().cpu().numpy())
            ys.append(y_batch.detach().cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    return y_true, y_pred

def fit_model(
    samples_df,
    train_idx,
    val_idx,
    backbone_name,
    fusion_name,
    head_name,
    mode_name,
    cfg,
    head_cfg,
    batch_size,
    max_epochs,
    seed,
    device,
    side_mask_paths=None,
    top_mask_paths=None,
):
    PATIENCE = 6
    seed_everything(seed)
    backbone, preprocess, feature_dim = get_backbone_and_preprocess(backbone_name, device)
    fused_dim = get_fused_dim(feature_dim, fusion_name)
    head = build_head(fused_dim, head_name, head_cfg, device)
    adapter_modules, hook_handles = configure_backbone_training(backbone_name, backbone, mode_name, cfg, device)

    head_params = [p for p in head.parameters() if p.requires_grad]
    adapter_params = [p for module in adapter_modules for p in module.parameters() if p.requires_grad]
    adapter_ids = {id(p) for p in adapter_params}
    backbone_params = [p for p in backbone.parameters() if p.requires_grad and id(p) not in adapter_ids]

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg["head_lr"]})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.get("backbone_lr", cfg.get("adapter_lr", cfg["head_lr"]))})
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": cfg.get("adapter_lr", cfg["head_lr"])})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.get("weight_decay", 1e-4))
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_mae = float("inf")
    patience_left = PATIENCE

    pbar = tqdm(range(1, max_epochs + 1), desc="Epochs", leave=False)

    for epoch in pbar:
        backbone.train()
        head.train()
        batch_losses = []

        for top_batch, side_batch, y_batch in iterate_batches(
            samples_df,
            train_idx,
            preprocess,
            batch_size,
            shuffle=True,
            seed=seed + epoch,
            device=device,
            side_mask_paths=side_mask_paths,
            top_mask_paths=top_mask_paths,
        ):
            optimizer.zero_grad()
            top_feat = backbone_forward(backbone_name, backbone, top_batch)
            side_feat = backbone_forward(backbone_name, backbone, side_batch)
            fused = fuse_features(top_feat, side_feat, fusion_name)
            pred = head(fused).squeeze(1)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        val_true, val_pred = predict_indices(
            samples_df,
            val_idx,
            preprocess,
            backbone_name,
            backbone,
            head,
            fusion_name,
            batch_size,
            device,
            side_mask_paths=side_mask_paths,
            top_mask_paths=top_mask_paths,
        )
        val_mae = mean_absolute_error(val_true, val_pred)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {
                "backbone": copy.deepcopy(backbone.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
            }
            patience_left = PATIENCE
        else:
            patience_left -= 1

        pbar.set_postfix(
            val=f"{val_mae:.2f}",
            best=f"{best_val_mae:.2f}",
            patience=patience_left
        )

        if patience_left == 0:
            break

    backbone.load_state_dict(best_state["backbone"])
    head.load_state_dict(best_state["head"])
    cleanup_hooks(hook_handles)
    return backbone, head, preprocess, best_val_mae


