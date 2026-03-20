import hashlib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models
from tqdm import tqdm
from pathlib import Path


# Create a pretrained torchvision backbone + preprocessing + output dimensions
def get_backbone_spec(backbone_name, device):
    if backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        output_dim = 2048
    elif backbone_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        feature_extractor = nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten(1),
        )
        output_dim = 768
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
    }

def image_cache_key(image_path):
    key = str(Path(image_path).resolve())
    return hashlib.md5(key.encode()).hexdigest()

def embedding_cache_path(backbone_name, image_path, cache_dir):
    return cache_dir / backbone_name / f"{image_cache_key(image_path)}.npy"

@torch.no_grad()
def compute_embedding(image_path, backbone_spec, device):
    image = Image.open(image_path).convert("RGB")
    x = backbone_spec["preprocess"](image).unsqueeze(0).to(device)
    feat = backbone_spec["model"](x)
    return feat.squeeze().detach().cpu().numpy().astype(np.float32)

def load_or_compute_embedding(image_path, backbone_name, backbone_spec_cache, cache_dir, device):
    cache_path = embedding_cache_path(backbone_name, image_path, cache_dir=cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return np.load(cache_path)

    if backbone_name not in backbone_spec_cache:
        backbone_spec_cache[backbone_name] = get_backbone_spec(backbone_name, device)

    vector = compute_embedding(image_path, backbone_spec_cache[backbone_name])
    np.save(cache_path, vector)
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

def build_feature_matrix(samples, backbone_name, fusion_name, cache_dir, device):
    backbone_spec_cache = {}
    fused_vectors = []

    for _, row in tqdm(samples.iterrows(), total=len(samples), desc=f"{backbone_name} + {fusion_name}"):
        top_vec = load_or_compute_embedding(row["top_path"], backbone_name, backbone_spec_cache, cache_dir, device)
        side_vec = load_or_compute_embedding(row["side_path"], backbone_name, backbone_spec_cache, cache_dir, device)
        fused_vectors.append(fuse_pair(top_vec, side_vec, fusion_name))

    x = np.vstack(fused_vectors)
    y = samples["volume"].to_numpy(dtype=float)
    groups = samples["exp_id"].to_numpy()
    return x, y, groups