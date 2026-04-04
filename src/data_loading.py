import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from src.helpers import *


def _normalize_mask_paths(mask_paths=None):
    if mask_paths is None:
        return None
    if isinstance(mask_paths, (str, Path)):
        return [Path(mask_paths)]
    return [Path(p) for p in mask_paths if p is not None]

class PairDataset(Dataset):
    def __init__(
        self,
        samples_df,
        indices,
        preprocess,
        side_mask_paths=None,
        top_mask_paths=None,
    ):
        self.samples_df = samples_df
        self.indices = np.asarray(indices, dtype=int)
        self.preprocess = preprocess
        self.resolved_side_mask_paths = _normalize_mask_paths(side_mask_paths)
        self.resolved_top_mask_paths = _normalize_mask_paths(top_mask_paths)
        self.mask_cache = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.samples_df.iloc[self.indices[i]]

        top_path = Path(row["top_path"])
        side_path = Path(row["side_path"])

        top_img = Image.open(top_path).convert("RGB")
        side_img = Image.open(side_path).convert("RGB")

        top_mask_path = resolve_mask_path(top_path, self.resolved_top_mask_paths)
        if top_mask_path:
            if top_mask_path in self.mask_cache:
                mask = self.mask_cache[top_mask_path]
            else:
                mask = cv2.imread(str(top_mask_path), cv2.IMREAD_GRAYSCALE)
                self.mask_cache[top_mask_path] = mask
            if mask is not None:
                if mask.shape[:2] != (top_img.height, top_img.width):
                    mask = cv2.resize(mask, (top_img.width, top_img.height), interpolation=cv2.INTER_NEAREST)
                top_img_np = np.array(top_img)
                top_img_np[mask == 0] = 0
                top_img = Image.fromarray(top_img_np)

        side_mask_path = resolve_mask_path(side_path, self.resolved_side_mask_paths)
        if side_mask_path:
            if side_mask_path in self.mask_cache:
                mask = self.mask_cache[side_mask_path]
            else:
                mask = cv2.imread(str(side_mask_path), cv2.IMREAD_GRAYSCALE)
                self.mask_cache[side_mask_path] = mask
            if mask is not None:
                if mask.shape[:2] != (side_img.height, side_img.width):
                    mask = cv2.resize(mask, (side_img.width, side_img.height), interpolation=cv2.INTER_NEAREST)
                side_img_np = np.array(side_img)
                side_img_np[mask == 0] = 0
                side_img = Image.fromarray(side_img_np)

        top_tensor = self.preprocess(top_img)
        side_tensor = self.preprocess(side_img)
        y = torch.tensor(float(row["volume"]), dtype=torch.float32)

        return top_tensor, side_tensor, y


def make_dataloader(
    samples_df,
    indices,
    preprocess,
    batch_size,
    shuffle,
    seed,
    side_mask_paths=None,
    top_mask_paths=None,
    num_workers=2,
):
    dataset = PairDataset(
        samples_df=samples_df,
        indices=indices,
        preprocess=preprocess,
        side_mask_paths=side_mask_paths,
        top_mask_paths=top_mask_paths,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),      # CPU only
        generator=generator,
    )