import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import (
    DATA_PATH, VIOLENT_LABELS, NUM_JOINTS, IN_CHANNELS,
    MAX_FRAMES, MAX_PERSONS, SPLIT_PROTOCOL, SEED
)


def load_raw_data():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


def normalize_skeleton(kp: np.ndarray) -> np.ndarray:
    hip = (kp[:, :, 11:12, :] + kp[:, :, 12:13, :]) / 2.0
    kp  = kp - hip
    sho = (kp[:, :, 5:6, :] + kp[:, :, 6:7, :]) / 2.0
    scale = np.linalg.norm(sho, axis=-1, keepdims=True).mean() + 1e-6
    return kp / scale


def sample_frames(kp: np.ndarray, n: int) -> np.ndarray:
    T = kp.shape[1]
    if T >= n:
        idx = np.linspace(0, T - 1, n, dtype=int)
        return kp[:, idx]
    pad = np.zeros((kp.shape[0], n - T, NUM_JOINTS, IN_CHANNELS), dtype=kp.dtype)
    return np.concatenate([kp, pad], axis=1)


def pad_persons(kp: np.ndarray, p: int) -> np.ndarray:
    if kp.shape[0] >= p:
        return kp[:p]
    pad = np.zeros((p - kp.shape[0], kp.shape[1], NUM_JOINTS, IN_CHANNELS), dtype=kp.dtype)
    return np.concatenate([kp, pad], axis=0)


def preprocess(ann: dict) -> np.ndarray:
    kp    = ann["keypoint"].astype(np.float32)        # (P, T, J, 2)
    score = ann["keypoint_score"].astype(np.float32)  # (P, T, J)
    kp    = kp * score[:, :, :, np.newaxis]
    kp    = normalize_skeleton(kp)
    kp    = sample_frames(kp, MAX_FRAMES)
    kp    = pad_persons(kp, MAX_PERSONS)
    return kp


class NTU120Dataset(Dataset):
    def __init__(self, annotations: list, augment: bool = False):
        self.data    = annotations
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ann   = self.data[idx]
        kp    = preprocess(ann)
        label = ann["label"]
        is_v  = 1 if label in VIOLENT_LABELS else 0

        x = torch.from_numpy(kp)   # (P, T, J, C)

        if self.augment:
            x = self._augment(x)

        return x, label, is_v

    def _augment(self, x):
        x = x + torch.randn_like(x) * 0.02
        if torch.rand(1).item() > 0.5:
            x = x.clone()
            x[..., 0] = -x[..., 0]
        scale = 0.9 + torch.rand(1).item() * 0.2
        x = x * scale
        return x


def get_dataloaders(batch_size: int = 32):
    print("NTU RGB+D 120")
    raw = load_raw_data()
    ann_by_id = {a["frame_dir"]: a for a in raw["annotations"]}
    train_ids = set(raw["split"][f"{SPLIT_PROTOCOL}_train"])
    val_ids   = set(raw["split"][f"{SPLIT_PROTOCOL}_val"])
    train_ann = [ann_by_id[i] for i in train_ids if i in ann_by_id]
    val_ann   = [ann_by_id[i] for i in val_ids   if i in ann_by_id]
    np.random.seed(SEED)
    val_idx  = np.random.permutation(len(val_ann))
    test_n   = len(val_ann) // 5
    test_ann = [val_ann[i] for i in val_idx[:test_n]]
    val_ann  = [val_ann[i] for i in val_idx[test_n:]]
    print(f"  Train: {len(train_ann):,}  Val: {len(val_ann):,}  Test: {len(test_ann):,}")
    v_train = sum(1 for a in train_ann if a["label"] in VIOLENT_LABELS)
    print(f"  Противоправных в train: {v_train:,} / {len(train_ann):,} "
          f"({100*v_train/len(train_ann):.1f}%)")
    train_loader = DataLoader(
        NTU120Dataset(train_ann, augment=True),
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        NTU120Dataset(val_ann, augment=False),
        batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        NTU120Dataset(test_ann, augment=False),
        batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )
    return train_loader, val_loader, test_loader
