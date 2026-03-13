import cv2
import mediapipe as mp
import numpy as np

class KeypointVisualizer:
    """Visualize MediaPipe keypoints on video frames."""

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing  = mp.solutions.drawing_utils
        self.mp_styles   = mp.solutions.drawing_styles

    def draw_landmarks(self, frame, results):
        """Draw all landmarks on a frame."""

        # Draw face
        self.mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            self.mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_styles
                .get_default_face_mesh_contours_style()
        )

        # Draw pose
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            self.mp_styles.get_default_pose_landmarks_style()
        )

        # Draw hands
        self.mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style()
        )
        self.mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style()
        )

        return frame

    def test_on_webcam(self):
        """Test MediaPipe live on your webcam."""
        cap = cv2.VideoCapture(0)
        print("Webcam test running — press Q to quit")

        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process
                frame_rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results            = holistic.process(frame_rgb)
                frame_rgb.flags.writeable = True
                frame              = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Draw
                frame = self.draw_landmarks(frame, results)

                # Show keypoint counts
                pose_detected  = results.pose_landmarks is not None
                lhand_detected = results.left_hand_landmarks is not None
                rhand_detected = results.right_hand_landmarks is not None

                cv2.putText(frame, f"Pose:  {'YES' if pose_detected  else 'NO'}",
                    (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if pose_detected  else (0,0,255), 2)
                cv2.putText(frame, f"LHand: {'YES' if lhand_detected else 'NO'}",
                    (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if lhand_detected else (0,0,255), 2)
                cv2.putText(frame, f"RHand: {'YES' if rhand_detected else 'NO'}",
                    (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if rhand_detected else (0,0,255), 2)

                cv2.imshow('MediaPipe Holistic — Sign Language', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        print("Webcam test done!")

if __name__ == "__main__":
    viz = KeypointVisualizer()
    viz.test_on_webcam()