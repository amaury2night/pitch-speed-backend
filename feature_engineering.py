"""
Biomechanical Feature Engineering for Baseball Pitching
Based on: MDPI Applied Sciences 2024 — "Automated Classification of Baseball
Pitching Phases Using Machine Learning and AI-Based Posture Estimation"
https://www.mdpi.com/2076-3417/15/22/12155

Extracts 33 kinematic features per pitch for injury risk modeling.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Key Body Angles
# ─────────────────────────────────────────────────────────────────────────────

def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """
    Calculate angle at vertex B formed by points A-B-C.
    a, b, c are (x, y, z) tuples in normalized [0,1] coordinates.
    Returns angle in degrees.
    """
    a = np.array(a[:3])
    b = np.array(b[:3])
    c = np.array(c[:3])

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def calculate_3d_distance(a: tuple, b: tuple) -> float:
    """Euclidean distance between two 3D points."""
    a = np.array(a[:3])
    b = np.array(b[:3])
    return float(np.linalg.norm(a - b))


# ─────────────────────────────────────────────────────────────────────────────
# Pitch Phase Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_pitch_phase(
    right_shoulder: tuple,
    right_elbow: tuple,
    right_wrist: tuple,
    left_shoulder: tuple,
    right_hip: tuple,
) -> str:
    """
    Classify current frame into one of 5 pitching phases.
    Based on elbow flexion angle and arm position.
    """
    elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    wrist_above_shoulder = right_wrist[1] < right_shoulder[1]  # y is top of frame
    arm_elevation = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Foot-to-ball phase (not applicable in video — skip)
    if elbow_angle > 160 and wrist_above_shoulder:
        return "cocking"
    elif 90 < elbow_angle <= 140 and wrist_above_shoulder:
        return "acceleration"
    elif elbow_angle < 90:
        return "deceleration"
    elif elbow_angle > 140 and not wrist_above_shoulder:
        return "follow_through"
    else:
        return "windup"


# ─────────────────────────────────────────────────────────────────────────────
# 33 Kinematic Features Per Pitch
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BiomechFeatures:
    """All 33 biomechanical features extracted from a single pitch."""
    # ── Peak joint angles ──────────────────────────────────────────────────
    peak_elbow_flexion: float          # 1  max right elbow flexion (degrees)
    peak_shoulder_rotation: float       # 2  max shoulder internal/external rotation
    peak_hip_rotation: float             # 3  max hip internal rotation
    peak_knee_flexion: float             # 4  max knee flexion during stride
    peak_trunk_tilt: float               # 5  max lateral trunk tilt
    peak_trunk_rotation: float           # 6  max trunk rotation velocity
    peak_wrist_velocity: float           # 7  max wrist snap velocity

    # ── Ranges of motion ───────────────────────────────────────────────────
    elbow_extension_range: float         # 8  elbow flexion→extension ROM
    shoulder_arc_range: float           # 9  arm circle arc during cocking
    hip_shoulder_separation: float      # 10 max hip-shoulder angle difference
    knee_extension_range: float          # 11 knee flex/ext ROM
    trunk_rotation_range: float          # 12 trunk rotation ROM

    # ── Temporal features ───────────────────────────────────────────────────
    stride_length: float                # 13 stride length (normalized)
    ground_contact_time: float          # 14 time foot on ground (sec)
    cocking_duration: float             # 15 time in cocking phase (sec)
    acceleration_duration: float        # 16 time in acceleration phase (sec)
    deceleration_duration: float        # 17 time in deceleration phase (sec)
    total_delivery_time: float          # 18 total pitch delivery time (sec)

    # ── Release point features ──────────────────────────────────────────────
    release_height: float               # 19 wrist height at release (normalized)
    release_distance: float             # 20 wrist distance from hip at release
    elbow_linear_velocity: float        # 21 elbow extension linear velocity
    wrist_linear_velocity: float        # 22 wrist snap speed
    shoulder_internal_rotation_vel: float # 23 shoulder IR velocity at release

    # ── Kinetic chain efficiency ────────────────────────────────────────────
    leg_drive_ratio: float             # 24 lower body contribution to velocity
    core_contribution: float            # 25 trunk rotation contribution
    arm_speed_ratio: float             # 26 arm speed vs leg speed ratio
    kinetic_chain_total: float         # 27 sum of segmental velocities

    # ── Risk flags ─────────────────────────────────────────────────────────
    elbow_valgus_load: float            # 28 estimated valgus stress at elbow
    shoulder_abduction_at_release: float # 29 shoulder abduction angle at release
    forward_trunk_tilt_at_release: float # 30 trunk lean at release
    elbow_invaria: float               # 31 elbow varus/valgus instability index
    late_cocking_strain: float         # 32 late cocking medial elbow strain index
    drop_and_drive_index: float        # 33 vertical knee drop at ball release

    # ── Metadata ────────────────────────────────────────────────────────────
    frames_analyzed: int
    phases_detected: list[str]
    confidence: float


def extract_features(pose_sequence: list[dict], fps: int = 30) -> Optional[BiomechFeatures]:
    """
    Compute all 33 biomechanical features from a sequence of pose frames.

    pose_sequence: list of dicts from PoseExtractor.extract_from_video()
    Each dict has keys: nose, left_shoulder, right_shoulder, left_elbow,
    right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee,
    right_knee, left_ankle, right_ankle — each a (x, y, z, visibility) tuple.

    Returns BiomechFeatures or None if not enough frames.
    """
    if len(pose_sequence) < 5:
        return None

    n = len(pose_sequence)
    dt = 1.0 / fps

    def safe_get(frame_idx: int, key: str) -> Optional[tuple]:
        if 0 <= frame_idx < n:
            return pose_sequence[frame_idx].get(key)
        return None

    # ── Helper: compute angle over time series ──────────────────────────────
    def peak_angle(key_a: str, key_b: str, key_c: str) -> float:
        angles = []
        for frame in pose_sequence:
            a, b, c = frame.get(key_a), frame.get(key_b), frame.get(key_c)
            if a and b and c:
                angles.append(calculate_angle(a, b, c))
        return max(angles) if angles else 0.0

    def min_angle(key_a: str, key_b: str, key_c: str) -> float:
        angles = []
        for frame in pose_sequence:
            a, b, c = frame.get(key_a), frame.get(key_b), frame.get(key_c)
            if a and b and c:
                angles.append(calculate_angle(a, b, c))
        return min(angles) if angles else 0.0

    def range_angle(key_a: str, key_b: str, key_c: str) -> float:
        return peak_angle(key_a, key_b, key_c) - min_angle(key_a, key_b, key_c)

    # ── Helper: peak velocity of a landmark ───────────────────────────────
    def peak_velocity(key: str) -> float:
        velocities = []
        for i in range(1, n):
            p_curr = pose_sequence[i].get(key)
            p_prev = pose_sequence[i - 1].get(key)
            if p_curr and p_prev:
                dist = calculate_3d_distance(p_curr, p_prev)
                velocities.append(dist / dt)
        return max(velocities) if velocities else 0.0

    # ── Phase segmentation ──────────────────────────────────────────────────
    phases = []
    for frame in pose_sequence:
        rs = frame.get("right_shoulder")
        re = frame.get("right_elbow")
        rw = frame.get("right_wrist")
        ls = frame.get("left_shoulder")
        rh = frame.get("right_hip")
        if all([rs, re, rw, ls, rh]):
            phases.append(detect_pitch_phase(rs, re, rw, ls, rh))

    phase_counts = {p: phases.count(p) for p in set(phases)}
    cocking_dur = phase_counts.get("cocking", 0) * dt
    accel_dur = phase_counts.get("acceleration", 0) * dt
    decel_dur = phase_counts.get("deceleration", 0) * dt
    total_dur = n * dt

    # ── Stride length (hip displacement during pitch) ──────────────────────
    first_hip = pose_sequence[0].get("right_hip")
    last_hip = pose_sequence[-1].get("right_hip")
    stride = calculate_3d_distance(first_hip, last_hip) if (first_hip and last_hip) else 0.0

    # ── Release point (approximated as frame with max wrist height) ────────
    release_frame_idx = max(
        range(n), key=lambda i: pose_sequence[i].get("right_wrist", (0, 0, 0, 0))[1]
    )
    release_frame = pose_sequence[release_frame_idx]
    rw = release_frame.get("right_wrist")
    rs = release_frame.get("right_shoulder")
    rh = release_frame.get("right_hip")

    release_height = rw[1] if rw else 0.0
    release_distance = calculate_3d_distance(rw, rh) if (rw and rh) else 0.0
    shoulder_abduction = calculate_angle(
        release_frame.get("left_shoulder"),
        release_frame.get("right_shoulder"),
        release_frame.get("right_elbow"),
    ) if all([
        release_frame.get("left_shoulder"),
        release_frame.get("right_shoulder"),
        release_frame.get("right_elbow"),
    ]) else 0.0

    # ── Trunk tilt at release ───────────────────────────────────────────────
    lh = release_frame.get("left_hip")
    rs_r = release_frame.get("right_shoulder")
    if lh and rs_r:
        forward_trunk = calculate_angle(
            (lh[0], lh[1] + 0.1, lh[2]),
            lh,
            (rs_r[0], rs_r[1] - 0.1, rs_r[2]),
        )
    else:
        forward_trunk = 0.0

    # ── Leg drive ratio (hip vs shoulder velocity) ─────────────────────────
    def segment_velocity(key: str) -> float:
        return peak_velocity(key)

    hip_vel = segment_velocity("right_hip")
    shoulder_vel = segment_velocity("right_shoulder")
    leg_drive_ratio = hip_vel / (shoulder_vel + 1e-8)

    # ── Core contribution ───────────────────────────────────────────────────
    elbow_vel = peak_velocity("right_elbow")
    core_contribution = (shoulder_vel - hip_vel) / (elbow_vel + 1e-8)

    # ── Kinetic chain ───────────────────────────────────────────────────────
    wrist_vel = peak_velocity("right_wrist")
    arm_speed_ratio = wrist_vel / (hip_vel + 1e-8)
    kinetic_chain_total = hip_vel + shoulder_vel + elbow_vel + wrist_vel

    # ── Risk indices ────────────────────────────────────────────────────────
    elbow_valgus_load = (180 - peak_angle("right_shoulder", "right_elbow", "right_wrist")) * 0.3
    elbow_invaria = min_angle("right_shoulder", "right_elbow", "right_wrist")
    late_cocking_strain = cocking_dur * peak_angle("right_shoulder", "right_elbow", "right_wrist")

    # Drop-and-drive: knee y drops from set position to ball release
    def get_knee_y(key: str, frame_idx: int) -> float:
        f = pose_sequence[frame_idx] if 0 <= frame_idx < n else None
        return f.get(key, (0, 1.0, 0, 0))[1] if f else 1.0

    stride_start_knee_y = get_knee_y("right_knee", 0)
    release_knee_y = get_knee_y("right_knee", release_frame_idx)
    drop_and_drive_index = abs(stride_start_knee_y - release_knee_y)

    return BiomechFeatures(
        # Peak angles
        peak_elbow_flexion=peak_angle("right_shoulder", "right_elbow", "right_wrist"),
        peak_shoulder_rotation=shoulder_abduction,
        peak_hip_rotation=peak_angle("left_hip", "right_hip", "right_knee"),
        peak_knee_flexion=peak_angle("right_hip", "right_knee", "right_ankle"),
        peak_trunk_tilt=peak_angle("left_hip", "right_hip", "right_shoulder"),
        peak_trunk_rotation=peak_angle("left_shoulder", "right_shoulder", "right_hip"),
        peak_wrist_velocity=wrist_vel * 100,  # scale for readability

        # ROM
        elbow_extension_range=range_angle("right_shoulder", "right_elbow", "right_wrist"),
        shoulder_arc_range=range_angle("left_shoulder", "right_shoulder", "right_elbow"),
        hip_shoulder_separation=peak_angle("right_hip", "right_shoulder", "right_wrist"),
        knee_extension_range=range_angle("right_hip", "right_knee", "right_ankle"),
        trunk_rotation_range=range_angle("nose", "right_hip", "left_hip"),

        # Temporal
        stride_length=stride,
        ground_contact_time=total_dur * 0.6,  # approximate
        cocking_duration=cocking_dur,
        acceleration_duration=accel_dur,
        deceleration_duration=decel_dur,
        total_delivery_time=total_dur,

        # Release
        release_height=release_height,
        release_distance=release_distance,
        elbow_linear_velocity=elbow_vel * 10,
        wrist_linear_velocity=wrist_vel * 10,
        shoulder_internal_rotation_vel=shoulder_vel * 10,

        # Kinetic chain
        leg_drive_ratio=leg_drive_ratio,
        core_contribution=core_contribution,
        arm_speed_ratio=arm_speed_ratio,
        kinetic_chain_total=kinetic_chain_total,

        # Risk
        elbow_valgus_load=elbow_valgus_load,
        shoulder_abduction_at_release=shoulder_abduction,
        forward_trunk_tilt_at_release=forward_trunk,
        elbow_invaria=elbow_invaria,
        late_cocking_strain=late_cocking_strain,
        drop_and_drive_index=drop_and_drive_index,

        # Metadata
        frames_analyzed=n,
        phases_detected=list(set(phases)),
        confidence=min(1.0, n / 60.0),  # more frames = higher confidence
    )


def features_to_dict(f: BiomechFeatures) -> dict:
    """Convert BiomechFeatures to flat dict for CSV / model input."""
    d = {}
    for k, v in f.__dict__.items():
        if isinstance(v, list):
            d[k] = ",".join(v)
        else:
            d[k] = v
    return d


def features_to_array(f: BiomechFeatures) -> np.ndarray:
    """Convert BiomechFeatures to numpy array for model inference."""
    exclude = {"frames_analyzed", "phases_detected", "confidence"}
    vals = [v for k, v in f.__dict__.items() if k not in exclude]
    return np.array(vals, dtype=np.float32)
