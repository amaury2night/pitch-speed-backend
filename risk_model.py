"""
Injury Risk Model — Hybrid LightGBM + TorchKM
Combines video-derived biomechanical features with existing TorchKM model.
Based on: MDPI 2024 LightGBM (99.7% phase accuracy) + XGBoost injury prediction
(AUC 0.81) + DVS Survival analysis.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
import lightgbm as lgb
import xgboost as xgb

from feature_engineering import BiomechFeatures, features_to_array


# ─────────────────────────────────────────────────────────────────────────────
# Risk Level
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskResult:
    risk_score: float          # 0-100 (higher = more risk)
    risk_level: str             # "low" | "moderate" | "high" | "critical"
    top_risk_factors: list[str]
    recommendations: list[str]
    confidence: float
    model_used: str             # "lightgbm" | "xgboost" | "torchkm_hybrid"
    features_used: int


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM Injury Risk Classifier
# ─────────────────────────────────────────────────────────────────────────────

class LightGBMInjuryRisk:
    """
    LightGBM model for pitcher injury risk classification.
    Trained on biomechanical features from video pose estimation.

    Features used: 33 kinematic features from feature_engineering.py
    Target: binary injury risk (high/low) — can be extended to multi-class.
    """

    # Feature names in order (matches feature_engineering.features_to_array order)
    FEATURE_NAMES = [
        "peak_elbow_flexion", "peak_shoulder_rotation", "peak_hip_rotation",
        "peak_knee_flexion", "peak_trunk_tilt", "peak_trunk_rotation",
        "peak_wrist_velocity", "elbow_extension_range", "shoulder_arc_range",
        "hip_shoulder_separation", "knee_extension_range", "trunk_rotation_range",
        "stride_length", "ground_contact_time", "cocking_duration",
        "acceleration_duration", "deceleration_duration", "total_delivery_time",
        "release_height", "release_distance", "elbow_linear_velocity",
        "wrist_linear_velocity", "shoulder_internal_rotation_vel",
        "leg_drive_ratio", "core_contribution", "arm_speed_ratio",
        "kinetic_chain_total", "elbow_valgus_load", "shoulder_abduction_at_release",
        "forward_trunk_tilt_at_release", "elbow_invaria", "late_cocking_strain",
        "drop_and_drive_index",
    ]

    def __init__(self):
        # Default parameters — in production, load from trained model file
        self.params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 5,
            "verbose": -1,
        }
        self.model: Optional[lgb.LGBMClassifier] = None
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train on feature array X (n_samples, 33) and binary labels y."""
        if X.shape[1] != 33:
            raise ValueError(f"Expected 33 features, got {X.shape[1]}")
        if len(X) < 10:
            raise ValueError("Need at least 10 training samples")

        self.model = lgb.LGBMClassifier(**self.params, n_estimators=100)
        self.model.fit(X, y)
        self._trained = True

        train_score = self.model.score(X, y)
        return {"train_accuracy": train_score, "samples": len(X)}

    def predict_risk(self, features: BiomechFeatures) -> RiskResult:
        """Predict injury risk from biomechanical features."""
        if not self._trained:
            # Use rule-based fallback with biomechanical thresholds
            return self._rule_based_risk(features)

        X = features_to_array(features).reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        risk_score = float(proba[1]) * 100  # probability of injury * 100

        return self._build_risk_result(risk_score, features, "lightgbm")

    def _rule_based_risk(self, features: BiomechFeatures) -> RiskResult:
        """Rule-based risk when model isn't trained (MVP phase)."""
        score = 0.0
        factors = []

        # Elbow valgus load (higher = more risk)
        if features.elbow_valgus_load > 30:
            score += 30
            factors.append(f"High elbow valgus load: {features.elbow_valgus_load:.1f}")
        elif features.elbow_valgus_load > 15:
            score += 15

        # Late cocking strain
        if features.late_cocking_strain > 5000:
            score += 25
            factors.append(f"High late cocking strain: {features.late_cocking_strain:.1f}")
        elif features.late_cocking_strain > 3000:
            score += 12

        # Elbow extension range (excessive extension = risk)
        if features.elbow_extension_range > 50:
            score += 15
            factors.append(f"Excessive elbow ROM: {features.elbow_extension_range:.1f}°")

        # Kinetic chain efficiency
        if features.kinetic_chain_total < 0.3:
            score += 20
            factors.append("Poor kinetic chain efficiency")

        # Arm speed ratio (too high = arm lag = risk)
        if features.arm_speed_ratio > 3.0:
            score += 10
            factors.append(f"High arm speed ratio: {features.arm_speed_ratio:.2f}")

        # Cocking duration (too long = red flag)
        if features.cocking_duration > 0.4:
            score += 15
            factors.append(f"Extended cocking phase: {features.cocking_duration:.2f}s")

        # Stride length
        if features.stride_length > 1.2 or features.stride_length < 0.3:
            score += 10

        # Normalize to 0-100
        risk_score = min(100.0, score)

        return self._build_risk_result(risk_score, features, "lightgbm_rule_based")

    def _build_risk_result(self, risk_score: float, features: BiomechFeatures, model: str) -> RiskResult:
        # Determine risk level
        if risk_score >= 75:
            level = "critical"
        elif risk_score >= 50:
            level = "high"
        elif risk_score >= 25:
            level = "moderate"
        else:
            level = "low"

        # Collect top 3 risk factors
        factors = []
        checks = [
            ("Elbow valgus load", features.elbow_valgus_load, 20, 30),
            ("Late cocking strain", features.late_cocking_strain, 3000, 5000),
            ("Elbow ROM", features.elbow_extension_range, 40, 50),
            ("Kinetic chain", features.kinetic_chain_total, 0.3, 0.2),
            ("Arm speed ratio", features.arm_speed_ratio, 2.5, 3.0),
            ("Cocking duration", features.cocking_duration, 0.3, 0.4),
        ]
        for name, val, warn, danger in checks:
            if val >= danger:
                factors.append(f"⚠️ {name}: {val:.1f} (HIGH)")
            elif val >= warn:
                factors.append(f"🔶 {name}: {val:.1f} (ELEVATED)")

        top_factors = factors[:3] if factors else ["No major risk factors detected"]

        # Recommendations
        recs = self._get_recommendations(level, factors, features)

        return RiskResult(
            risk_score=round(risk_score, 1),
            risk_level=level,
            top_risk_factors=top_factors,
            recommendations=recs,
            confidence=features.confidence,
            model_used=model,
            features_used=33,
        )

    @staticmethod
    def _get_recommendations(level: str, factors: list, features: BiomechFeatures) -> list[str]:
        recs = []
        if level in ("critical", "high"):
            recs.append("🔴 Rest and evaluation by sports medicine specialist")
            recs.append("🔴 Consider MRI for elbow (UCL) and shoulder")
            recs.append("🔴 Reduce throwing volume by 50% immediately")
        if any("valgus" in f.lower() for f in factors):
            recs.append("🟠 Work on elbow strengthening and proper follow-through")
        if any("cocking" in f.lower() for f in factors):
            recs.append("🟠 Optimize arm path in late cocking phase")
        if features.kinetic_chain_total < 0.3:
            recs.append("🟡 Improve leg drive and core activation")
        if not recs:
            recs.append("✅ Continue regular monitoring")
        return recs[:5]


# ─────────────────────────────────────────────────────────────────────────────
# TorchKM + LightGBM Hybrid
# ─────────────────────────────────────────────────────────────────────────────

class HybridRiskModel:
    """
    Combines existing TorchKM model with LightGBM biomechanical model.
    TorchKM handles tabular feature risk; LightGBM handles video-derived
    biomechanical risk. Combined via weighted average.
    """

    def __init__(self):
        self.lgbm = LightGBMInjuryRisk()
        # TorchKM placeholder — in production, load the actual TorchKM model
        self.torchkm_weight = 0.4
        self.lgbm_weight = 0.6

    def predict_combined(
        self,
        torchkm_features: Optional[dict] = None,
        biomech_features: Optional[BiomechFeatures] = None,
    ) -> RiskResult:
        """
        Combine TorchKM (tabular) + LightGBM (video biomechanics) risk scores.

        torchkm_features: dict of TorchKM input features (e.g., workload, history)
        biomech_features: BiomechFeatures from video analysis
        """
        scores = []
        weights = []
        models_used = []

        # TorchKM score (if available)
        if torchkm_features:
            torchkm_score = self._torchkm_predict(torchkm_features)
            scores.append(torchkm_score)
            weights.append(self.torchkm_weight)
            models_used.append("torchkm")

        # LightGBM score (if available)
        if biomech_features:
            lgbm_result = self.lgbm.predict_risk(biomech_features)
            scores.append(lgbm_result.risk_score)
            weights.append(self.lgbm_weight)
            models_used.append("lightgbm")
            lgbm_result_msg = lgbm_result
        else:
            lgbm_result_msg = None

        if not scores:
            return RiskResult(
                risk_score=0.0, risk_level="low",
                top_risk_factors=["No input data provided"],
                recommendations=["Provide video or tabular data for risk analysis"],
                confidence=0.0, model_used="none", features_used=0,
            )

        # Weighted average
        total_weight = sum(weights)
        combined_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        # Determine level
        if combined_score >= 75:
            level = "critical"
        elif combined_score >= 50:
            level = "high"
        elif combined_score >= 25:
            level = "moderate"
        else:
            level = "low"

        result = RiskResult(
            risk_score=round(combined_score, 1),
            risk_level=level,
            top_risk_factors=lgbm_result_msg.top_risk_factors if lgbm_result_msg else ["Combined analysis"],
            recommendations=lgbm_result_msg.recommendations if lgbm_result_msg else [],
            confidence=lgbm_result_msg.confidence if lgbm_result_msg else 0.5,
            model_used="+".join(models_used) + "_hybrid",
            features_used=len(torchkm_features) if torchkm_features else 0,
        )
        return result

    @staticmethod
    def _torchkm_predict(features: dict) -> float:
        """TorchKM risk score from tabular features. Placeholder — wire to actual model."""
        # Example: simple weighted sum based on known risk factors
        score = 0.0
        if features.get("velocity_change", 0) > 5:
            score += 20
        if features.get("pitch_mix_variance", 0) > 0.3:
            score += 15
        if features.get("workload_index", 0) > 80:
            score += 25
        if features.get("days_rest", 999) < 3:
            score += 10
        return min(100.0, score)
