import os
import json
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class WLASLDataset(Dataset):
    """
    WLASL Dataset using OFFICIAL train/val/test splits.
    Different signers in train vs val vs test = realistic evaluation.
    """

    def __init__(self, samples, label_map, max_frames=30, augment=False):
        self.samples     = samples
        self.label_map   = label_map
        self.max_frames  = max_frames
        self.augment     = augment
        self.num_classes = len(label_map)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        keypoint_path, label = self.samples[idx]
        keypoints = np.load(keypoint_path).astype(np.float32)

        if self.augment:
            keypoints = self.augment_sequence(keypoints)

        keypoints = self.pad_or_truncate(keypoints)
        keypoints = self.normalize(keypoints)

        return torch.FloatTensor(keypoints), torch.LongTensor([label])[0]

    def pad_or_truncate(self, seq):
        T = len(seq)
        if T >= self.max_frames:
            start = (T - self.max_frames) // 2
            return seq[start:start + self.max_frames]
        pad = np.zeros((self.max_frames - T, seq.shape[1]), dtype=np.float32)
        return np.concatenate([seq, pad], axis=0)

    def normalize(self, seq):
        mean = seq.mean(axis=0, keepdims=True)
        std  = seq.std(axis=0,  keepdims=True)
        std  = np.where(std < 1e-8, 1.0, std)
        return (seq - mean) / std

    def augment_sequence(self, seq):
        seq = seq.copy()

        # 1. Time scaling
        if np.random.rand() < 0.6:
            scale   = np.random.uniform(0.6, 1.4)
            T       = len(seq)
            new_T   = max(3, int(T * scale))
            indices = np.linspace(0, T - 1, new_T).astype(int)
            seq     = seq[indices]

        # 2. Gaussian noise (simulate sensor noise)
        if np.random.rand() < 0.6:
            seq = seq + np.random.normal(0, 0.02, seq.shape).astype(np.float32)

        # 3. Hand keypoint jitter only
        if np.random.rand() < 0.5:
            seq[:, 132:] += np.random.normal(0, 0.015, seq[:, 132:].shape)

        # 4. Frame dropout
        if np.random.rand() < 0.4 and len(seq) > 3:
            n_drop = np.random.randint(1, max(2, len(seq) // 4))
            drop_i = np.random.choice(len(seq), n_drop, replace=False)
            seq[drop_i] = 0

        # 5. Temporal reverse (sign played backwards = different sign, helps robustness)
        if np.random.rand() < 0.2:
            seq = seq[::-1].copy()

        # 6. Scale keypoints (simulate distance from camera)
        if np.random.rand() < 0.4:
            scale = np.random.uniform(0.85, 1.15)
            seq   = seq * scale

        return seq


def build_dataset_from_csv(processed_dir, csv_path, max_frames=30):
    """
    Build datasets using OFFICIAL WLASL train/val/test splits.
    This is crucial — ensures different signers in each split.
    """
    # Load label map
    label_map_path = os.path.join(processed_dir, 'label_map.json')
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)

    word_to_idx = {v: int(k) for k, v in label_map.items()}

    # Load official splits CSV
    df = pd.read_csv(csv_path)
    print(f"Total entries in CSV : {len(df)}")

    train_s, val_s, test_s = [], [], []
    missing = 0

    for _, row in df.iterrows():
        word     = row['word']
        video_id = row['video_id']
        split    = row['split']       # train / val / test

        if word not in word_to_idx:
            continue

        label    = word_to_idx[word]
        npy_path = os.path.join(processed_dir, word, f"{video_id}.npy")

        if not os.path.exists(npy_path):
            missing += 1
            continue

        sample = (npy_path, label)

        if split == 'train':
            train_s.append(sample)
        elif split == 'val':
            val_s.append(sample)
        elif split == 'test':
            test_s.append(sample)

    print(f"Missing npy files    : {missing}")
    print(f"Train (official)     : {len(train_s)}")
    print(f"Val   (official)     : {len(val_s)}")
    print(f"Test  (official)     : {len(test_s)}")
    print(f"Classes              : {len(label_map)}")

    return (
        WLASLDataset(train_s, label_map, max_frames=max_frames, augment=True),
        WLASLDataset(val_s,   label_map, max_frames=max_frames, augment=False),
        WLASLDataset(test_s,  label_map, max_frames=max_frames, augment=False),
        label_map
    )


def get_dataloaders(processed_dir, csv_path, batch_size=16, max_frames=30, num_workers=0):
    train_ds, val_ds, test_ds, label_map = build_dataset_from_csv(
        processed_dir, csv_path, max_frames
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True, drop_last=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
        label_map
    )
