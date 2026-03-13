import json
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 55)
print("       WLASL Dataset Exploration")
print("=" * 55)

# ── Load JSONs ─────────────────────────────────────────────
raw_dir   = "data/raw/wlasl"
json_path = os.path.join(raw_dir, "WLASL_v0.3.json")
n100_path = os.path.join(raw_dir, "nslt_100.json")

with open(json_path, 'r') as f:
    data = json.load(f)

with open(n100_path, 'r') as f:
    nslt_100 = json.load(f)

# ── Basic Stats ────────────────────────────────────────────
print(f"\n📊 Full Dataset (WLASL_v0.3):")
print(f"   Total ASL words     : {len(data)}")

video_counts = [len(e['instances']) for e in data]
print(f"   Total video entries : {sum(video_counts)}")
print(f"   Min per word        : {min(video_counts)}")
print(f"   Max per word        : {max(video_counts)}")
print(f"   Avg per word        : {np.mean(video_counts):.1f}")

print(f"\n📊 NSLT-100 Subset:")
print(f"   Video entries       : {len(nslt_100)}")

# ── First 20 Words ─────────────────────────────────────────
print(f"\n📝 First 20 ASL words in dataset:")
for i, entry in enumerate(data[:20]):
    word  = entry['gloss']
    count = len(entry['instances'])
    bar   = "█" * min(count, 30)
    print(f"   {i:3d}. {word:20s} {bar} ({count})")

# ── Check Actual Videos on Disk ────────────────────────────
print(f"\n📁 Checking actual videos on disk...")
video_dir = os.path.join(raw_dir, "videos")

if os.path.exists(video_dir):
    # Check structure
    contents = os.listdir(video_dir)
    print(f"   Items in videos/    : {len(contents)}")

    # Count mp4 files
    mp4_files = list(
        f for f in os.listdir(video_dir)
        if f.endswith('.mp4')
    )

    # Maybe videos are in subfolders
    mp4_recursive = []
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            if f.endswith('.mp4'):
                mp4_recursive.append(os.path.join(root, f))

    print(f"   MP4 files (flat)    : {len(mp4_files)}")
    print(f"   MP4 files (all)     : {len(mp4_recursive)}")

    if mp4_recursive:
        # Show sample path to understand structure
        print(f"\n   Sample path: {mp4_recursive[0]}")
        print(f"   Sample path: {mp4_recursive[1]}")

        # Analyze a sample video
        sample = mp4_recursive[0]
        cap    = cv2.VideoCapture(sample)
        fps    = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur    = frames / fps if fps > 0 else 0
        cap.release()

        print(f"\n🎥 Sample Video Info:")
        print(f"   File      : {os.path.basename(sample)}")
        print(f"   Resolution: {w} x {h}")
        print(f"   FPS       : {fps:.1f}")
        print(f"   Frames    : {frames}")
        print(f"   Duration  : {dur:.2f} seconds")
else:
    print(f"   videos/ folder not found at {video_dir}")
    print(f"   Contents of raw/: {os.listdir(raw_dir)}")

# ── Plot ───────────────────────────────────────────────────
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(video_counts, bins=40, color='steelblue',
         edgecolor='black', alpha=0.8)
plt.title("Videos per Word — Full WLASL (2000 words)")
plt.xlabel("Number of Videos")
plt.ylabel("Number of Words")
plt.axvline(np.mean(video_counts), color='red',
            linestyle='--', label=f'Mean={np.mean(video_counts):.1f}')
plt.legend()

plt.subplot(1, 2, 2)
top_words  = [data[i]['gloss']            for i in range(15)]
top_counts = [len(data[i]['instances'])   for i in range(15)]
colors     = ['steelblue' if c >= 10 else 'coral' for c in top_counts]
plt.barh(top_words[::-1], top_counts[::-1], color=colors[::-1])
plt.title("First 15 Words — Video Counts\n(blue=10+, red=<10)")
plt.xlabel("Number of Videos")

plt.tight_layout()
plt.savefig("data/wlasl_distribution.png", dpi=150)
print(f"\n✅ Plot saved → data/wlasl_distribution.png")
print("\n🎉 Exploration complete!")