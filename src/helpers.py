import glob
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("model",   estimator),
    ])

def find_photo(folder, pic_num):
    folder = Path(folder)
    pic_num = int(pic_num)

    matches = []
    for ext in ["jpg", "jpeg", "png"]:
        matches.extend(glob.glob(str(folder / f"*{pic_num:04d}.{ext}")))
        matches.extend(glob.glob(str(folder / f"*{pic_num:04d}.{ext.upper()}")))

    if not matches:
        raise FileNotFoundError(f"No photo for {pic_num}")

    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {pic_num}: {matches}")

    return matches[0]

def load_image_paths(data_path, top_folder, side_folder):
    data = pd.read_csv(data_path)
    samples = data.copy()
    samples = samples[samples["volume"] > 0].copy()
    samples = samples[samples["pic_top"].notna() & samples["pic_side"].notna()].copy()

    samples["top_path"] = samples["pic_top"].apply(lambda x: find_photo(top_folder, x))
    samples["side_path"] = samples["pic_side"].apply(lambda x: find_photo(side_folder, x))
    samples = samples[samples["top_path"].notna() & samples["side_path"].notna()].copy()

    image_paths = sorted({Path(p) for p in samples["top_path"].tolist() + samples["side_path"].tolist()})
    return samples.reset_index(drop=True), image_paths