"""
MediaPipe Pose Extraction for Baseball Pitching Biomechanics
Extracts 33 kinematic features from smartphone video.
"""

import cv2
import numpy as np
from typing import Optional
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass


@dataclass
class PoseLandmarks:
    """Keypose landmark coordinates from MediaPipe."""
    nose: tuple[float, float]
    left_shoulder: tuple[float, float]
    right_shoulder: tuple[float, float]
    left_elbow: tuple[float, float]
    right_elbow: tuple[float, float]
    left_wrist: tuple[float, float]
    right_wrist: tuple[float, float]
    left_hip: tuple[float, float]
    right_hip: tuple[float, float]
    left_knee: tuple[float, float]
    right_knee: tuple[float, float]
    left_ankle: tuple[float, float]
    right_ankle: tuple[float, float]


class PoseExtractor:
    """
    Extract 3D pose landmarks from video frames using MediaPipe.
    Designed for baseball pitching: identifies windup, cocking,
    acceleration, deceleration, and follow-through phases.
    """

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        base_options = python.BaseOptions(model_asset_path="")
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def extract_from_video(self, video_path: str, fps: int = 30) -> list[dict]:
        """
        Process video and return pose landmarks per frame.
        Returns list of dicts with timestamp + 33 landmark coordinates (x, y, z, visibility).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0

        frame_interval = max(1, int(video_fps / fps))
        results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(frame_idx / video_fps * 1000)

                pose_result = self.detector.detect_for_video(mp_image, timestamp_ms)

                if pose_result.pose_landmarks:
                    landmarks = self._extract_landmarks(pose_result.pose_landmarks[0])
                    landmarks["frame_idx"] = frame_idx
                    landmarks["timestamp_s"] = frame_idx / video_fps
                    results.append(landmarks)

            frame_idx += 1

        cap.release()
        return results

    def extract_from_frame(self, frame: np.ndarray, timestamp_ms: int = 0) -> Optional[dict]:
        """Extract pose from a single frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        pose_result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if pose_result.pose_landmarks:
            landmarks = self._extract_landmarks(pose_result.pose_landmarks[0])
            landmarks["timestamp_ms"] = timestamp_ms
            return landmarks
        return None

    @staticmethod
    def _extract_landmarks(pose_landmarks) -> dict:
        """Convert MediaPipe landmark list to dict keyed by body part."""
        lm = pose_landmarks
        return {
            "nose": (lm[0].x, lm[0].y, lm[0].z, lm[0].visibility),
            "left_shoulder": (lm[11].x, lm[11].y, lm[11].z, lm[11].visibility),
            "right_shoulder": (lm[12].x, lm[12].y, lm[12].z, lm[12].visibility),
            "left_elbow": (lm[13].x, lm[13].y, lm[13].z, lm[13].visibility),
            "right_elbow": (lm[14].x, lm[14].y, lm[14].z, lm[14].visibility),
            "left_wrist": (lm[15].x, lm[15].y, lm[15].z, lm[15].visibility),
            "right_wrist": (lm[16].x, lm[16].y, lm[16].z, lm[16].visibility),
            "left_hip": (lm[23].x, lm[23].y, lm[23].z, lm[23].visibility),
            "right_hip": (lm[24].x, lm[24].y, lm[24].z, lm[24].visibility),
            "left_knee": (lm[25].x, lm[25].y, lm[25].z, lm[25].visibility),
            "right_knee": (lm[26].x, lm[26].y, lm[26].z, lm[26].visibility),
            "left_ankle": (lm[27].x, lm[27].y, lm[27].z, lm[27].visibility),
            "right_ankle": (lm[28].x, lm[28].y, lm[28].z, lm[28].visibility),
        }
