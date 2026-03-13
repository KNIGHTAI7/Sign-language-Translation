import json
import os
import csv
from collections import Counter, defaultdict


def build_video_index(
    wlasl_json_path,
    nslt_json_path,
    video_dir,
    output_dir,
    num_classes=20      # ← parameter added here
):
    """
    Build video index using nslt JSON (correct IDs)
    cross-referenced with WLASL_v0.3.json (word names).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load JSONs ─────────────────────────────────────
    with open(wlasl_json_path, 'r') as f:
        wlasl_data = json.load(f)

    with open(nslt_json_path, 'r') as f:
        nslt_data = json.load(f)

    # ── Build class_idx → word map from WLASL ──────────
    idx_to_word = {idx: entry['gloss'] for idx, entry in enumerate(wlasl_data)}

    # ── Find which classes exist in nslt file ──────────
    # Get all unique class indices in nslt
    all_classes_in_nslt = sorted(set(
        info['action'][0] for info in nslt_data.values()
    ))

    # Take only first num_classes
    selected_classes = set(all_classes_in_nslt[:num_classes])

    print(f"Total classes in WLASL    : {len(idx_to_word)}")
    print(f"Total videos in nslt      : {len(nslt_data)}")
    print(f"Selecting first           : {num_classes} classes")
    print(f"Classes selected          : {sorted(selected_classes)[:10]}...")

    # ── Build samples list ─────────────────────────────
    samples = []
    missing = 0
    found   = 0

    for video_id, info in nslt_data.items():
        subset    = info['subset']
        class_idx = info['action'][0]

        # Skip classes not in our selection
        if class_idx not in selected_classes:
            continue

        word = idx_to_word.get(class_idx, f"class_{class_idx}")
        video_path = os.path.join(video_dir, f"{video_id}.mp4")

        if os.path.exists(video_path):
            samples.append({
                'video_path'  : video_path,
                'video_id'    : video_id,
                'label'       : class_idx,
                'word'        : word,
                'split'       : subset,
                'start_frame' : info['action'][1],
                'end_frame'   : info['action'][2]
            })
            found += 1
        else:
            missing += 1

    # ── Remap labels to 0..N-1 ─────────────────────────
    sorted_classes = sorted(selected_classes)
    old_to_new     = {old: new for new, old in enumerate(sorted_classes)}
    label_map      = {new: idx_to_word[old] for old, new in old_to_new.items()}

    for s in samples:
        s['label'] = old_to_new[s['label']]

    print(f"\nVideos found              : {found}")
    print(f"Videos missing            : {missing}")
    print(f"Unique classes            : {len(set(s['label'] for s in samples))}")

    # ── Split distribution ─────────────────────────────
    split_counts = Counter(s['split'] for s in samples)
    print(f"\nSplit distribution:")
    for split, count in sorted(split_counts.items()):
        bar = "█" * (count // 5)
        print(f"  {split:10s}: {count:4d} {bar}")

    # ── Word distribution ──────────────────────────────
    word_counts = defaultdict(int)
    for s in samples:
        word_counts[s['word']] += 1

    print(f"\nWords selected ({len(label_map)}):")
    for idx, word in sorted(label_map.items()):
        count = word_counts[word]
        bar   = "█" * count
        print(f"  {idx:3d}. {word:20s} {bar} ({count})")

    # ── Save label map ─────────────────────────────────
    label_map_path = os.path.join(output_dir, 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)

    # ── Save samples CSV ───────────────────────────────
    csv_path = os.path.join(output_dir, 'all_samples.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'video_path', 'video_id', 'label',
            'word', 'split', 'start_frame', 'end_frame'
        ])
        writer.writeheader()
        writer.writerows(samples)

    print(f"\n✅ Index built successfully!")
    print(f"   Classes    : {len(label_map)}")
    print(f"   Samples    : {len(samples)}")
    print(f"   Label map  → {label_map_path}")
    print(f"   Samples    → {csv_path}")

    return samples, label_map


if __name__ == "__main__":
    samples, label_map = build_video_index(
        wlasl_json_path = "data/raw/wlasl/WLASL_v0.3.json",
        nslt_json_path  = "data/raw/wlasl/nslt_100.json",
        video_dir       = "data/raw/wlasl/videos",
        output_dir      = "data/splits",
        num_classes     = 20   # start small, prove it works, then scale up
    )
