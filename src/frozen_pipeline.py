import hashlib
import sys
import numpy as np
import torch
import cv2
from PIL import Image
from torch import nn
from torchvision import models
from tqdm import tqdm
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from helpers import resolve_mask_path


# Create a pretrained torchvision backbone + preprocessing + output dimensions
def get_backbone_spec(backbone_name, device):
    if backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        output_dim = 2048
        model_type = "cnn"

    elif backbone_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        feature_extractor = nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten(1),
        )
        output_dim = 768
        model_type = "cnn"

    elif backbone_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
        feature_extractor = nn.Sequential(
            model.features,
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )
        output_dim = 1024
        model_type = "cnn"

    elif backbone_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
        feature_extractor = model
        output_dim = 768
        model_type = "vit"

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    preprocess = weights.transforms()
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    return {
        "name": backbone_name,
        "weights": weights,
        "preprocess": preprocess,
        "model": feature_extractor,
        "output_dim": output_dim,
        "model_type": model_type,
    }

def extract_vit_features(vit_model, x):
    n = x.shape[0]
    x = vit_model._process_input(x)
    cls_token = vit_model.class_token.expand(n, -1, -1)
    x = torch.cat([cls_token, x], dim=1)
    x = vit_model.encoder(x)
    x = x[:, 0]
    return x

def image_cache_key(image_path, mask_path=None):
    key = str(Path(image_path).resolve())
    if mask_path:
        key += f"_mask_{Path(mask_path).name}"
    return hashlib.md5(key.encode()).hexdigest()

def embedding_cache_path(backbone_name, image_path, cache_dir, mask_path=None):
    return cache_dir / backbone_name / f"{image_cache_key(image_path, mask_path)}.npy"

@torch.no_grad()
def compute_embedding(image_path, backbone_spec, device, mask_path=None):
    image = Image.open(image_path).convert("RGB")

    if mask_path:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Resize mask to image size if necessary
            if mask.shape[:2] != (image.height, image.width):
                mask = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
            
            image_np = np.array(image)
            image_np[mask == 0] = 0
            image = Image.fromarray(image_np)

    x = backbone_spec["preprocess"](image).unsqueeze(0).to(device)

    if backbone_spec["model_type"] == "vit":
        feat = extract_vit_features(backbone_spec["model"], x)
    else:
        feat = backbone_spec["model"](x)

    return feat.squeeze().detach().cpu().numpy().astype(np.float32)

def load_or_compute_embedding(image_path, backbone_name, backbone_spec_cache, cache_dir, device, mask_path=None, embedding_memory_cache=None):
    key = image_cache_key(image_path, mask_path)
    cache_key = f"{backbone_name}_{key}"

    if embedding_memory_cache is not None and cache_key in embedding_memory_cache:
        return embedding_memory_cache[cache_key]

    cache_path = embedding_cache_path(backbone_name, image_path, cache_dir=cache_dir, mask_path=mask_path)

    if cache_path.exists():
        vector = np.load(cache_path)
    else:
        if backbone_name not in backbone_spec_cache:
            backbone_spec_cache[backbone_name] = get_backbone_spec(backbone_name, device)

        vector = compute_embedding(
            image_path,
            backbone_spec_cache[backbone_name],
            device,
            mask_path=mask_path,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, vector)

    if embedding_memory_cache is not None:
        embedding_memory_cache[cache_key] = vector

    return vector

def fuse_pair(top_vec, side_vec, fusion_name):
    if fusion_name == "concat":
        return np.concatenate([top_vec, side_vec], axis=0)
    if fusion_name == "mean":
        return (top_vec + side_vec) / 2.0
    if fusion_name == "max":
        return np.maximum(top_vec, side_vec)
    if fusion_name == "concat_abs_diff":
        return np.concatenate([top_vec, side_vec, np.abs(top_vec - side_vec)], axis=0)
    raise ValueError(f"Unsupported fusion: {fusion_name}")

def _normalize_mask_paths(mask_paths=None):
    if mask_paths is None:
        return None
    if isinstance(mask_paths, (str, Path)):
        return [Path(mask_paths)]
    return [Path(p) for p in mask_paths if p is not None]


def build_feature_matrix(
    samples,
    backbone_name,
    fusion_name,
    cache_dir,
    device,
    side_mask_paths=None,
    top_mask_paths=None,
    embedding_memory_cache=None
):
    backbone_spec_cache = {}
    fused_vectors = []
    resolved_side_mask_paths = _normalize_mask_paths(side_mask_paths)
    resolved_top_mask_paths = _normalize_mask_paths(top_mask_paths)

    if embedding_memory_cache is None:
        embedding_memory_cache = {}

    for _, row in tqdm(samples.iterrows(), total=len(samples), desc=f"{backbone_name} + {fusion_name}"):
        top_mask_path = resolve_mask_path(row["top_path"], resolved_top_mask_paths)
        side_mask_path_resolved = resolve_mask_path(row["side_path"], resolved_side_mask_paths)

        top_vec = load_or_compute_embedding(row["top_path"], backbone_name, backbone_spec_cache, cache_dir, device,
                                            mask_path=top_mask_path, embedding_memory_cache=embedding_memory_cache)
        side_vec = load_or_compute_embedding(row["side_path"], backbone_name, backbone_spec_cache, cache_dir, device,
                                             mask_path=side_mask_path_resolved, embedding_memory_cache=embedding_memory_cache)
        fused_vectors.append(fuse_pair(top_vec, side_vec, fusion_name))

    x = np.vstack(fused_vectors)
    y = samples["volume"].to_numpy(dtype=float)
    groups = samples["exp_id"].to_numpy()
    return x, y, groups

def build_single_view_feature_matrix(
    samples,
    backbone_name,
    image_path_col,
    cache_dir,
    device,
    mask_paths=None,
):

    backbone_spec_cache = {}
    vectors = []
    resolved_mask_paths = _normalize_mask_paths(mask_paths)

    for _, row in tqdm(samples.iterrows(), total=len(samples), desc=f"{backbone_name} | {image_path_col}"):
        current_mask_path = resolve_mask_path(row[image_path_col], resolved_mask_paths)

        vec = load_or_compute_embedding(row[image_path_col], backbone_name, backbone_spec_cache, cache_dir, device, mask_path=current_mask_path)
        vectors.append(vec)

    x = np.vstack(vectors)
    y = samples["volume"].to_numpy(dtype=float)
    groups = samples["exp_id"].to_numpy()
    return x, y, groups