import glob
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

def image_cache_key(image_path, mask_path=None):
    key = str(Path(image_path).resolve())
    if mask_path:
        key += f"_mask_{Path(mask_path).name}"
    return hashlib.md5(key.encode()).hexdigest()

def find_photo(folder, pic_num):
    folder = Path(folder)
    pic_num = int(pic_num)

    matches = []
    for ext in ["jpg", "jpeg", "png"]:
        matches.extend(glob.glob(str(folder / f"*{pic_num:04d}.{ext}")))
        matches.extend(glob.glob(str(folder / f"*{pic_num:04d}.{ext.upper()}")))

    unique_matches = list(dict.fromkeys(matches))

    if not unique_matches:
        raise FileNotFoundError(f"No photo for {pic_num}")

    if len(unique_matches) > 1:
        raise ValueError(f"Multiple matches for {pic_num}: {unique_matches}")

    return unique_matches[0]

def load_image_paths(data_path, top_folder, side_folder):
    data = pd.read_csv(data_path)
    samples = data.copy()

    samples["pic_top"] = pd.to_numeric(samples["pic_top"], errors="coerce")
    samples["pic_side"] = pd.to_numeric(samples["pic_side"], errors="coerce")
    samples = samples[samples["pic_top"].notna() & samples["pic_side"].notna()].copy()

    samples["top_path"] = samples["pic_top"].apply(lambda x: find_photo(top_folder, x))
    samples["side_path"] = samples["pic_side"].apply(lambda x: find_photo(side_folder, x))
    samples = samples[samples["top_path"].notna() & samples["side_path"].notna()].copy()

    image_paths = sorted({Path(p) for p in samples["top_path"].tolist() + samples["side_path"].tolist()})
    return samples.reset_index(drop=True), image_paths

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)