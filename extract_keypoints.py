import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    """
    Extract ONLY pose + hands (no face).
    
    Pose  : 33 × 4 = 132
    LHand : 21 × 3 = 63
    RHand : 21 × 3 = 63
    Total : 258 features (vs 1662 with face)
    """
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                     for lm in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 4)

    left_hand = np.array([[lm.x, lm.y, lm.z]
                           for lm in results.left_hand_landmarks.landmark]).flatten() \
                if results.left_hand_landmarks else np.zeros(21 * 3)

    right_hand = np.array([[lm.x, lm.y, lm.z]
                            for lm in results.right_hand_landmarks.landmark]).flatten() \
                 if results.right_hand_landmarks else np.zeros(21 * 3)

    # NO face landmarks - they are noise for sign language
    return np.concatenate([pose, left_hand, right_hand])  # (258,)


def process_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    keypoints_sequence = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        while len(keypoints_sequence) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)
            keypoints_sequence.append(extract_keypoints(results))

    cap.release()
    return np.array(keypoints_sequence) if keypoints_sequence else None


def process_dataset(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Total videos to process: {len(df)}")
    print(f"Keypoint size per frame : 258 (pose + hands only, no face)\n")

    processed = failed = skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting keypoints"):
        word       = row['word']
        video_id   = row['video_id']
        video_path = row['video_path']

        word_dir    = os.path.join(output_dir, word)
        os.makedirs(word_dir, exist_ok=True)
        output_path = os.path.join(word_dir, f"{video_id}.npy")

        if os.path.exists(output_path):
            skipped += 1
            continue

        keypoints = process_video(video_path)

        if keypoints is not None and len(keypoints) > 0:
            np.save(output_path, keypoints)
            processed += 1
        else:
            failed += 1

    print(f"\n✅ Extraction complete!")
    print(f"   Processed : {processed}")
    print(f"   Skipped   : {skipped} (already done)")
    print(f"   Failed    : {failed}")


if __name__ == "__main__":
    print("Re-extracting keypoints WITHOUT face landmarks...")
    print("This removes noise and reduces input from 1662 → 258\n")

    # Delete old processed folder first
    import shutil
    if os.path.exists("data/processed"):
        shutil.rmtree("data/processed")
        print("Deleted old processed folder\n")

    process_dataset(
        csv_path   = "data/splits/all_samples.csv",
        output_dir = "data/processed"
    )
