import cv2
import mediapipe as mp
import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm


class KeypointExtractor:
    """
    Extracts hand, pose, and face keypoints from videos using MediaPipe.

    Output per frame: 1662 keypoints
    ├── Pose:       33 landmarks × 4 values (x, y, z, visibility) = 132
    ├── Face:       468 landmarks × 3 values (x, y, z)            = 1404
    ├── Left Hand:  21 landmarks × 3 values (x, y, z)             = 63
    └── Right Hand: 21 landmarks × 3 values (x, y, z)             = 63
    """

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing  = mp.solutions.drawing_utils

    def extract_keypoints(self, results):
        """Extract keypoints from MediaPipe results into flat numpy array."""

        # Pose (33 × 4)
        if results.pose_landmarks:
            pose = np.array([
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ]).flatten()
        else:
            pose = np.zeros(33 * 4)

        # Face (468 × 3)
        if results.face_landmarks:
            face = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.face_landmarks.landmark
            ]).flatten()
        else:
            face = np.zeros(468 * 3)

        # Left Hand (21 × 3)
        if results.left_hand_landmarks:
            left_hand = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.left_hand_landmarks.landmark
            ]).flatten()
        else:
            left_hand = np.zeros(21 * 3)

        # Right Hand (21 × 3)
        if results.right_hand_landmarks:
            right_hand = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.right_hand_landmarks.landmark
            ]).flatten()
        else:
            right_hand = np.zeros(21 * 3)

        return np.concatenate([pose, face, left_hand, right_hand])

    def process_video(self, video_path, max_frames=30):
        """
        Extract keypoints from all frames in a video.

        Args:
            video_path : path to .mp4 file
            max_frames : cap frames to this number

        Returns:
            numpy array of shape (num_frames, 1662)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        keypoints_sequence = []

        with self.mp_holistic.Holistic(
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5,
            model_complexity         = 1
        ) as holistic:

            while len(keypoints_sequence) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = holistic.process(frame_rgb)

                keypoints = self.extract_keypoints(results)
                keypoints_sequence.append(keypoints)

        cap.release()

        if len(keypoints_sequence) == 0:
            return None

        return np.array(keypoints_sequence)  # (T, 1662)

    def process_dataset_flat(self, csv_path, output_dir):
        """
        Process all videos listed in CSV file.
        Saves keypoints as .npy files organized by word.

        output_dir/
        └── word/
            └── video_id.npy
        """
        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        print(f"Total videos to process : {len(df)}")

        stats = {"processed": 0, "failed": 0, "skipped": 0}

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting keypoints"):
            word       = row['word']
            video_id   = row['video_id']
            video_path = row['video_path']

            # Save to: processed/word/video_id.npy
            word_dir    = os.path.join(output_dir, word)
            os.makedirs(word_dir, exist_ok=True)
            output_path = os.path.join(word_dir, f"{video_id}.npy")

            # Skip if already done (safe to re-run)
            if os.path.exists(output_path):
                stats["skipped"] += 1
                continue

            keypoints = self.process_video(video_path)

            if keypoints is not None and len(keypoints) > 0:
                np.save(output_path, keypoints)
                stats["processed"] += 1
            else:
                stats["failed"] += 1

        print(f"\n✅ Extraction complete!")
        print(f"   Processed : {stats['processed']}")
        print(f"   Skipped   : {stats['skipped']} (already done)")
        print(f"   Failed    : {stats['failed']}")