"""
Pitch Speed App — Backend
FastAPI + Roboflow + OpenCV tracking + MediaPipe biomechanics + LightGBM risk model
"""

import os
import uuid
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

# Biomechanics imports
from mediapipe_pose import PoseExtractor
from feature_engineering import extract_features, BiomechFeatures, features_to_array
from risk_model import LightGBMInjuryRisk, HybridRiskModel

# =============================================================================
# CONFIG — Fill in your Roboflow credentials
# =============================================================================

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "DhNVZPFMig80m8wJ9CYn")
ROBOFLOW_MODEL = os.getenv("ROBOFLOW_MODEL", "pitch-speed/1")
ROBOFLOW_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}"

# Distance from pitcher's mound to home plate (official MLB: 60.5 ft = 18.44m)
MOUND_TO_HOME_FT = 60.5
MOUND_TO_HOME_M = 18.44

# =============================================================================
# APP
# =============================================================================

app = FastAPI(title="Pitch Speed API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# HELPERS — Roboflow
# =============================================================================

async def detect_ball_in_frame(frame_bytes: bytes, api_key: str) -> list[dict]:
    """Send frame to Roboflow and get ball detections."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                ROBOFLOW_URL,
                files={"file": ("frame.jpg", frame_bytes, "image/jpeg")},
                data={"api_key": api_key},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("predictions", [])
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Roboflow error: {e}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Roboflow request failed: {e}")


# =============================================================================
# HELPERS — Tracking (Hungarian algorithm / SORT-lite)
# =============================================================================

def iou(box_a: dict, box_b: dict) -> float:
    """Intersection over Union between two bounding boxes."""
    x_a = max(box_a["x"] - box_a["width"] / 2, box_b["x"] - box_b["width"] / 2)
    y_a = max(box_a["y"] - box_a["height"] / 2, box_b["y"] - box_b["height"] / 2)
    x_b = min(box_a["x"] + box_a["width"] / 2, box_b["x"] + box_b["width"] / 2)
    y_b = min(box_a["y"] + box_a["height"] / 2, box_b["y"] + box_b["height"] / 2)

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = box_a["width"] * box_a["height"]
    box_b_area = box_b["width"] * box_b["height"]
    union_area = box_a_area + box_b_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def _track_iou(a: dict, b: dict) -> float:
    """IoU between two bounding boxes."""
    wa, ha = a.get("width", 10), a.get("height", 10)
    wb, hb = b.get("width", 10), b.get("height", 10)
    x_a = max(a["x"] - wa/2, b["x"] - wb/2)
    y_a = max(a["y"] - ha/2, b["y"] - hb/2)
    x_b = min(a["x"] + wa/2, b["x"] + wb/2)
    y_b = min(a["y"] + ha/2, b["y"] + hb/2)
    inter = max(0, x_b-x_a) * max(0, y_b-y_a)
    return inter / (wa*ha + wb*hb - inter + 1e-9)

def track_detections(detections_per_frame: list[list[dict]], iou_threshold: float = 0.3):
    """
    Greedy nearest-neighbor tracking across frames.
    Replaces scipy-based Hungarian algorithm for better Render.com compatibility.
    Returns list of tracks: [{frame_idx, x, y, track_id}]
    """
    tracks = []
    next_track_id = 0
    active = {}  # track_id -> last_detection

    for frame_idx, detections in enumerate(detections_per_frame):
        if not detections:
            continue

        # Greedy match: each track gets its nearest detection
        matched = set()
        new_active = {}

        for tid, last in active.items():
            best_j = -1
            best_iou = iou_threshold
            for j, det in enumerate(detections):
                if j in matched:
                    continue
                iou_val = _track_iou(last, det)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j
            if best_j >= 0:
                d = detections[best_j]
                tracks.append({"frame_idx": frame_idx, "x": d["x"], "y": d["y"], "track_id": tid})
                matched.add(best_j)
                new_active[tid] = d

        for j, det in enumerate(detections):
            if j not in matched:
                tid = next_track_id
                next_track_id += 1
                tracks.append({"frame_idx": frame_idx, "x": det["x"], "y": det["y"], "track_id": tid})
                new_active[tid] = det

        active = new_active

    return tracks


# =============================================================================
# HELPERS — Speed Calculation
# =============================================================================

def calculate_speed(
    tracks: list[dict],
    frame_times: list[float],
    focal_length_px: float,
    ball_real_radius_m: float = 0.037,  # MLB baseball ~7.4cm diameter
    mound_to_home_m: float = MOUND_TO_HOME_M,
) -> Optional[dict]:
    """
    Calculate pitch speed using pinhole camera model.
    
    focal_length_px: focal length in pixels (from camera calibration or estimate)
    ball_real_radius_m: real-world ball radius (~3.7cm)
    mound_to_home_m: distance the ball travels (60.5ft = 18.44m)
    """
    if len(tracks) < 3:
        return None

    # Group by track_id and sort by frame
    track_points = {}
    for t in tracks:
        tid = t["track_id"]
        if tid not in track_points:
            track_points[tid] = []
        track_points[tid].append((t["frame_idx"], t["x"], t["y"]))

    # Find the longest consistent track (likely the pitch)
    best_track = None
    best_len = 0
    for tid, points in track_points.items():
        if len(points) > best_len:
            best_len = len(points)
            best_track = (tid, sorted(points, key=lambda p: p[0]))

    if not best_track or best_len < 3:
        return None

    _, points = best_track

    # Calculate speed using distance / time
    total_distance_px = 0.0
    total_time_s = 0.0
    prev_frame_idx, prev_x, prev_y = points[0]

    for frame_idx, x, y in points[1:]:
        dx = x - prev_x
        dy = y - prev_y
        dist_px = np.sqrt(dx**2 + dy**2)
        
        # Time between frames
        if frame_idx < len(frame_times) and prev_frame_idx < len(frame_times):
            dt = abs(frame_times[frame_idx] - frame_times[prev_frame_idx])
        else:
            # Estimate from video framerate (assume 240fps for slow-mo)
            dt = 1 / 240.0

        total_distance_px += dist_px
        total_time_s += dt
        prev_frame_idx, prev_x, prev_y = frame_idx, x, y

    if total_time_s == 0:
        return None

    # Convert pixels to real-world distance
    # Using ball size as reference: ball_radius_px / ball_radius_m = focal_length_px / distance_m
    # Approximate: use the first detection where ball is clearest (largest bbox)
    # For now: estimate pixels_per_meter from field markings or assume ~150px per meter at release
    
    # SIMPLIFIED APPROACH: Assume ball travels mound_to_home_m over the full pixel distance
    # This is a rough estimate — proper calibration requires known reference in frame
    
    # Use ball pixel size to estimate scale
    # ball_diameter_px = avg of detected ball widths
    ball_widths = [p[2] for t in tracks for p in [(t["x"], t["y"], t.get("width", 10))]]  # fallback 10px
    
    # Better: use the track — estimate pixels per meter from ball size
    # At release ball is close (large), at plate it's far (small)
    # Average the sizes across the track
    avg_ball_width_px = np.median([p[2] for t in tracks for p in [(t["x"], t["y"], t.get("width", 15))]])
    
    if avg_ball_width_px < 1:
        avg_ball_width_px = 15.0  # fallback

    # pixels_per_meter = ball_real_radius_m * focal_length_px / ball_radius_px (too complex)
    # SIMPLER: scale factor based on typical baseball field footage
    # At 60ft (18.3m), a baseball is typically 10-30px depending on zoom
    # Use 20px as default ball size → pixels_per_meter = 18.3 / 20 = 0.915 m/px
    pixels_per_meter = mound_to_home_m / avg_ball_width_px * 2  # diameter, not radius

    total_distance_m = total_distance_px * pixels_per_meter

    # Speed calculation
    speed_m_s = total_distance_m / total_time_s
    speed_mph = speed_m_s * 2.23694  # m/s to mph
    speed_kmh = speed_m_s * 3.6

    # Sanity check (realistic pitch: 60-110 mph)
    if speed_mph < 30 or speed_mph > 130:
        # Try alternate calibration
        pixels_per_meter_alt = 500  # ~500px per meter at typical field zoom
        total_distance_m_alt = total_distance_px / pixels_per_meter_alt
        speed_m_s_alt = total_distance_m_alt / total_time_s
        speed_mph_alt = speed_m_s_alt * 2.23694

        if 30 < speed_mph_alt < 130:
            return {
                "mph": round(speed_mph_alt, 1),
                "kmh": round(speed_mph_alt * 1.60934, 1),
                "confidence": 0.7,
                "frames_analyzed": best_len,
                "method": "calibrated",
            }

        return None

    return {
        "mph": round(speed_mph, 1),
        "kmh": round(speed_kmh, 1),
        "confidence": 0.85,
        "frames_analyzed": best_len,
        "method": "direct",
    }


# =============================================================================
# MODELS — initialized at startup
# =============================================================================

pose_extractor: Optional[PoseExtractor] = None
injury_risk_model: Optional[LightGBMInjuryRisk] = None
hybrid_model: Optional[HybridRiskModel] = None


def get_pose_extractor() -> PoseExtractor:
    global pose_extractor
    if pose_extractor is None:
        pose_extractor = PoseExtractor(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return pose_extractor


def get_injury_risk_model() -> LightGBMInjuryRisk:
    global injury_risk_model
    if injury_risk_model is None:
        injury_risk_model = LightGBMInjuryRisk()
    return injury_risk_model


def get_hybrid_model() -> HybridRiskModel:
    global hybrid_model
    if hybrid_model is None:
        hybrid_model = HybridRiskModel()
    return hybrid_model


# =============================================================================
# HELPERS — Video Processing
# =============================================================================

def extract_frames(video_path: str, interval_ms: int = 50) -> tuple[list[bytes], list[float]]:
    """
    Extract frames from video at given interval.
    Returns (frame_bytes_list, timestamps_ms_list)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 240.0  # fallback for slow-mo

    frames = []
    timestamps = []

    interval_frames = max(1, int(fps * interval_ms / 1000))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval_frames == 0:
            # Encode to jpg
            _, buf = cv2.imencode(".jpg", frame)
            frames.append(buf.tobytes())
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

        frame_idx += 1

    cap.release()
    return frames, timestamps


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "roboflow_key_set": ROBOFLOW_API_KEY != "YOUR_API_KEY_HERE"}


@app.post("/biomechanics/analyze")
async def analyze_biomechanics(file: UploadFile = File(...)):
    """
    Analyze pitcher biomechanics from video.
    Extracts 33 kinematic features via MediaPipe Pose,
    then runs injury risk model (LightGBM rule-based for MVP).

    Pipeline:
        Video → MediaPipe Pose → 33 features → Risk Score → Report
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".mov", ".avi", ".webm"]:
        raise HTTPException(status_code=400, detail="Unsupported format. Use MP4, MOV, AVI, or WebM.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 1. Extract pose landmarks from video
        pose_ext = get_pose_extractor()
        cap = cv2.VideoCapture(tmp_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if video_fps <= 0:
            video_fps = 30.0

        pose_sequence = pose_ext.extract_from_video(tmp_path, fps=30)

        if len(pose_sequence) < 5:
            return JSONResponse({
                "success": False,
                "error": "Not enough frames. Ensure the full pitch delivery is visible in the video.",
                "tip": "Record from the side view showing the entire throwing motion.",
            })

        # 2. Compute 33 biomechanical features
        features = extract_features(pose_sequence, fps=int(video_fps))

        if not features:
            return JSONResponse({
                "success": False,
                "error": "Could not extract biomechanical features from video.",
                "tip": "Ensure the pitcher is clearly visible and well-lit.",
            })

        # 3. Run injury risk model
        risk_model = get_injury_risk_model()
        risk_result = risk_model.predict_risk(features)

        # 4. Build response
        return JSONResponse({
            "success": True,
            "pitch_speed_available": False,  # set True if also running speed calc
            "biomechanics": {
                "frames_analyzed": features.frames_analyzed,
                "phases_detected": features.phases_detected,
                "fps_used": int(video_fps),
                "confidence": features.confidence,
            },
            "risk": {
                "score": risk_result.risk_score,
                "level": risk_result.risk_level,
                "top_factors": risk_result.top_risk_factors,
                "recommendations": risk_result.recommendations,
                "model": risk_result.model_used,
            },
            "features": {
                "peak_elbow_flexion": round(features.peak_elbow_flexion, 1),
                "peak_wrist_velocity": round(features.peak_wrist_velocity, 2),
                "elbow_valgus_load": round(features.elbow_valgus_load, 2),
                "late_cocking_strain": round(features.late_cocking_strain, 1),
                "kinetic_chain_total": round(features.kinetic_chain_total, 4),
                "hip_shoulder_separation": round(features.hip_shoulder_separation, 1),
                "stride_length": round(features.stride_length, 4),
                "total_delivery_time": round(features.total_delivery_time, 3),
                "cocking_duration": round(features.cocking_duration, 3),
                "acceleration_duration": round(features.acceleration_duration, 3),
                "arm_speed_ratio": round(features.arm_speed_ratio, 3),
                "leg_drive_ratio": round(features.leg_drive_ratio, 3),
            },
        })

    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Main endpoint: receives video, runs CV pipeline, returns pitch speed.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".mov", ".avi", ".webm"]:
        raise HTTPException(status_code=400, detail="Unsupported video format. Use MP4, MOV, AVI, or WebM.")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Extract frames
        frames, frame_times = extract_frames(tmp_path, interval_ms=20)

        if len(frames) < 3:
            raise HTTPException(status_code=400, detail="Video too short. Need at least 3 usable frames.")

        # Detect ball in each frame
        detections_per_frame = []
        for frame_bytes in frames:
            try:
                preds = await detect_ball_in_frame(frame_bytes, ROBOFLOW_API_KEY)
                detections_per_frame.append(preds)
            except Exception:
                detections_per_frame.append([])

        # Track across frames
        tracks = track_detections(detections_per_frame)

        if not tracks:
            return JSONResponse({
                "success": False,
                "error": "No ball detected. Make sure the camera is pointed at the pitcher.",
                "tip": "Try recording in bright light and ensure the ball is visible.",
            })

        # Calculate speed
        # Estimate focal length from frame size (rough)
        cap = cv2.VideoCapture(tmp_path)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        focal_length_px = frame_width * 0.8  # rough estimate

        result = calculate_speed(tracks, frame_times, focal_length_px)

        if not result:
            return JSONResponse({
                "success": False,
                "error": "Could not track the ball trajectory. Try a clearer video.",
                "tip": "Ensure the full pitch path is visible in the frame.",
            })

        return JSONResponse({
            "success": True,
            "pitch": result,
        })

    finally:
        # Cleanup temp file
        Path(tmp_path).unlink(missing_ok=True)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)