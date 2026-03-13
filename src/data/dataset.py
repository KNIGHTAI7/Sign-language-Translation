import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class WLASLDataset(Dataset):
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
        keypoints = np.load(keypoint_path)

        # Augment FIRST (may change length)
        if self.augment:
            keypoints = self.augment_sequence(keypoints)

        # THEN fix length
        keypoints = self.pad_or_truncate(keypoints)

        # Per-feature normalization (better than global)
        keypoints = self.normalize(keypoints)

        return torch.FloatTensor(keypoints), torch.LongTensor([label])[0]

    def pad_or_truncate(self, sequence):
        T = len(sequence)
        if T >= self.max_frames:
            start = (T - self.max_frames) // 2
            return sequence[start:start + self.max_frames]
        pad = np.zeros((self.max_frames - T, sequence.shape[1]))
        return np.concatenate([sequence, pad], axis=0)

    def normalize(self, sequence):
        """Per-feature normalization — preserves relative differences between keypoints."""
        mean = sequence.mean(axis=0, keepdims=True)   # (1, 258)
        std  = sequence.std(axis=0,  keepdims=True)   # (1, 258)
        std  = np.where(std < 1e-8, 1.0, std)         # avoid divide by zero
        return (sequence - mean) / std

    def augment_sequence(self, sequence):
        # 1. Mirror flip (simulate left-handed signer)
        if np.random.rand() < 0.5:
            sequence = sequence.copy()
            # Flip x-coordinates for pose (every 4th value starting at 0)
            sequence[:, 0::4] *= -1

        # 2. Time scaling
        if np.random.rand() < 0.4:
            scale   = np.random.uniform(0.7, 1.3)
            T       = len(sequence)
            new_T   = max(2, int(T * scale))
            indices = np.linspace(0, T - 1, new_T).astype(int)
            sequence = sequence[indices]

        # 3. Gaussian noise
        if np.random.rand() < 0.4:
            sequence = sequence + np.random.normal(0, 0.02, sequence.shape)

        # 4. Frame dropout
        if np.random.rand() < 0.3 and len(sequence) > 2:
            drop = np.random.randint(0, len(sequence))
            sequence = sequence.copy()
            sequence[drop] = 0

        return sequence


def build_dataset(processed_dir, max_frames=30):
    label_map_path = os.path.join(processed_dir, 'label_map.json')
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)

    word_to_idx = {v: int(k) for k, v in label_map.items()}
    all_samples = []

    for class_folder in sorted(os.listdir(processed_dir)):
        class_path = os.path.join(processed_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        if class_folder not in word_to_idx:
            continue
        label = word_to_idx[class_folder]
        for npy_file in os.listdir(class_path):
            if npy_file.endswith('.npy'):
                all_samples.append((os.path.join(class_path, npy_file), label))

    print(f"Total samples found  : {len(all_samples)}")
    print(f"Total classes        : {len(set(s[1] for s in all_samples))}")

    train_s, temp_s = train_test_split(
        all_samples, test_size=0.3, random_state=42,
        stratify=[s[1] for s in all_samples]
    )
    val_s, test_s = train_test_split(
        temp_s, test_size=0.5, random_state=42,
        stratify=[s[1] for s in temp_s]
    )

    print(f"Train : {len(train_s)} | Val : {len(val_s)} | Test : {len(test_s)}")

    return (
        WLASLDataset(train_s, label_map, max_frames=max_frames, augment=True),
        WLASLDataset(val_s,   label_map, max_frames=max_frames, augment=False),
        WLASLDataset(test_s,  label_map, max_frames=max_frames, augment=False),
        label_map
    )


def get_dataloaders(processed_dir, batch_size=32, max_frames=30, num_workers=0):
    train_ds, val_ds, test_ds, label_map = build_dataset(processed_dir, max_frames)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        label_map
    )
