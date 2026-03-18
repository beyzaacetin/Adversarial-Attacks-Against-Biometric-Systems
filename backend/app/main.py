"""
FastAPI Backend — Biometric Authentication Demo
=================================================
Endpoints:
  POST /api/enroll          — Enroll a new user (collect baseline)
  POST /api/authenticate    — Authenticate typing/mouse sample
  POST /api/attack/fgsm     — Run FGSM attack on a sample
  POST /api/attack/pgd      — Run PGD attack on a sample
  POST /api/attack/mimicry  — Run statistical mimicry attack
  POST /api/defense/toggle  — Enable/disable defense mechanisms
  GET  /api/stats           — Get model performance stats
  GET  /api/users           — List enrolled users

To run:
  pip install fastapi uvicorn scikit-learn numpy
  cd backend/app
  uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import json
import os
import sys

# Add project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "../models"))
sys.path.insert(0, os.path.join(BASE_DIR, "../attacks"))
sys.path.insert(0, os.path.join(BASE_DIR, "../data"))

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier, MLPRegressor

# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="Biometric Authentication API",
    description="Continuous authentication using behavioral biometrics with adversarial attack simulation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# IN-MEMORY STATE
# ============================================================

class AppState:
    """Holds enrolled users, models, and settings."""
    def __init__(self):
        self.enrolled_users: Dict[str, dict] = {}
        self.models: Dict[str, object] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.defense_enabled: Dict[str, bool] = {
            "input_smoothing": False,
            "feature_squeezing": False,
            "adversarial_training": False,
            "threshold_strictness": False,
        }
        self.auth_history: List[dict] = []
        self.attack_history: List[dict] = []

state = AppState()

# ============================================================
# PYDANTIC MODELS
# ============================================================

class KeystrokeData(BaseModel):
    """Raw keystroke timing data from the frontend."""
    user_id: str
    keystrokes: List[dict]  # [{key, downTime, upTime}, ...]
    text: str

class MouseData(BaseModel):
    """Raw mouse movement data from the frontend."""
    user_id: str
    movements: List[dict]  # [{x, y, timestamp, event_type}, ...]

class EnrollRequest(BaseModel):
    user_id: str
    keystroke_samples: List[List[float]]  # Multiple feature vectors
    mouse_samples: Optional[List[List[float]]] = None

class AuthRequest(BaseModel):
    user_id: str
    keystroke_features: List[float]
    mouse_features: Optional[List[float]] = None

class AttackRequest(BaseModel):
    user_id: str
    attack_type: str  # "fgsm", "pgd", "mimicry"
    epsilon: Optional[float] = 0.2
    n_iterations: Optional[int] = 20
    noise_level: Optional[float] = 0.1

class DefenseToggle(BaseModel):
    defense_name: str
    enabled: bool

# ============================================================
# FEATURE EXTRACTION (from raw browser events)
# ============================================================

def extract_keystroke_features(keystrokes: List[dict]) -> List[float]:
    """
    Extract timing features from raw keystroke events.
    Mirrors the CMU dataset feature format.
    """
    if len(keystrokes) < 2:
        return [0.0] * 20
    
    features = []
    
    # Hold times (key down to key up for each key)
    hold_times = []
    for ks in keystrokes:
        if "downTime" in ks and "upTime" in ks:
            hold = (ks["upTime"] - ks["downTime"]) / 1000.0  # Convert ms to sec
            hold_times.append(max(0.001, hold))
    
    # Down-Down times (time between consecutive key presses)
    dd_times = []
    for i in range(1, len(keystrokes)):
        if "downTime" in keystrokes[i] and "downTime" in keystrokes[i-1]:
            dd = (keystrokes[i]["downTime"] - keystrokes[i-1]["downTime"]) / 1000.0
            dd_times.append(max(0.001, dd))
    
    # Up-Down times (key release to next key press)
    ud_times = []
    for i in range(1, len(keystrokes)):
        if "upTime" in keystrokes[i-1] and "downTime" in keystrokes[i]:
            ud = (keystrokes[i]["downTime"] - keystrokes[i-1]["upTime"]) / 1000.0
            ud_times.append(ud)  # Can be negative (overlapping keys)
    
    # Build feature vector
    # Raw timing features (pad/truncate to fixed size)
    for times in [hold_times, dd_times, ud_times]:
        if len(times) >= 5:
            features.extend(times[:5])
        else:
            features.extend(times + [0.0] * (5 - len(times)))
    
    # Statistical features
    for times in [hold_times, dd_times, ud_times]:
        if times:
            features.append(np.mean(times))
            features.append(np.std(times) if len(times) > 1 else 0.0)
        else:
            features.extend([0.0, 0.0])
    
    # Typing speed (chars per second)
    total_time = sum(dd_times) if dd_times else 1.0
    features.append(len(keystrokes) / total_time)
    
    return features[:20]  # Fixed 20-dim vector

def extract_mouse_features(movements: List[dict]) -> List[float]:
    """Extract behavioral features from mouse movement data."""
    if len(movements) < 3:
        return [0.0] * 15
    
    features = []
    
    xs = [m["x"] for m in movements]
    ys = [m["y"] for m in movements]
    ts = [m["timestamp"] for m in movements]
    
    # Speed features
    speeds = []
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i-1]
        dy = ys[i] - ys[i-1]
        dt = max((ts[i] - ts[i-1]) / 1000.0, 0.001)
        speed = np.sqrt(dx**2 + dy**2) / dt
        speeds.append(min(speed, 5000))
    
    features.append(np.mean(speeds) if speeds else 0)
    features.append(np.std(speeds) if len(speeds) > 1 else 0)
    features.append(np.max(speeds) if speeds else 0)
    
    # Acceleration
    accels = []
    for i in range(1, len(speeds)):
        dt = max((ts[i+1] - ts[i]) / 1000.0, 0.001)
        accels.append((speeds[i] - speeds[i-1]) / dt)
    
    features.append(np.mean(np.abs(accels)) if accels else 0)
    features.append(np.std(accels) if len(accels) > 1 else 0)
    
    # Direction features
    angles = []
    for i in range(1, len(xs)):
        angles.append(np.arctan2(ys[i] - ys[i-1], xs[i] - xs[i-1]))
    
    features.append(np.mean(angles) if angles else 0)
    features.append(np.std(angles) if len(angles) > 1 else 0)
    
    # Path features
    total_dist = sum(np.sqrt((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2) 
                     for i in range(1, len(xs)))
    displacement = np.sqrt((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2)
    features.append(total_dist)
    features.append(displacement)
    features.append(displacement / (total_dist + 1e-6))  # Path efficiency
    
    # Click features (if present)
    clicks = [m for m in movements if m.get("event_type") == "click"]
    features.append(len(clicks))
    
    # Duration
    duration = (ts[-1] - ts[0]) / 1000.0 if len(ts) > 1 else 0
    features.append(duration)
    
    # Curvature
    curvatures = []
    for i in range(1, len(angles)):
        curvatures.append(abs(angles[i] - angles[i-1]))
    features.append(np.mean(curvatures) if curvatures else 0)
    features.append(np.std(curvatures) if len(curvatures) > 1 else 0)
    
    return features[:15]

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {"status": "ok", "message": "Biometric Authentication API", "version": "1.0.0"}

@app.get("/api/users")
def list_users():
    """List all enrolled users."""
    users = []
    for uid, data in state.enrolled_users.items():
        users.append({
            "user_id": uid,
            "n_samples": data.get("n_samples", 0),
            "enrolled_at": data.get("enrolled_at", ""),
            "modalities": data.get("modalities", []),
        })
    return {"users": users}

@app.post("/api/enroll")
def enroll_user(req: EnrollRequest):
    """Enroll a user with baseline behavioral samples."""
    if len(req.keystroke_samples) < 3:
        raise HTTPException(400, "Need at least 3 keystroke samples for enrollment")
    
    X = np.array(req.keystroke_samples)
    
    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train One-Class SVM (anomaly detection — genuine only)
    ocsvm = OneClassSVM(kernel='rbf', nu=0.15, gamma='scale')
    ocsvm.fit(X_scaled)
    
    # Train RF if we have enough samples for binary approach
    rf = None
    if len(req.keystroke_samples) >= 10:
        # Create synthetic impostors by adding noise
        n_imp = len(X)
        X_imp = X + np.random.normal(0, np.std(X, axis=0) * 2, (n_imp, X.shape[1]))
        X_combined = np.vstack([X, X_imp])
        y_combined = np.concatenate([np.ones(len(X)), np.zeros(n_imp)])
        X_combined_scaled = scaler.transform(X_combined)
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        rf.fit(X_combined_scaled, y_combined)
    
    # Store
    state.enrolled_users[req.user_id] = {
        "n_samples": len(req.keystroke_samples),
        "modalities": ["keystroke"],
        "mean_features": X.mean(axis=0).tolist(),
        "std_features": X.std(axis=0).tolist(),
    }
    state.scalers[req.user_id] = scaler
    state.models[f"{req.user_id}_ocsvm"] = ocsvm
    if rf:
        state.models[f"{req.user_id}_rf"] = rf
    
    return {
        "status": "enrolled",
        "user_id": req.user_id,
        "n_samples": len(req.keystroke_samples),
        "models_trained": ["ocsvm"] + (["rf"] if rf else []),
    }

@app.post("/api/authenticate")
def authenticate(req: AuthRequest):
    """Authenticate a user based on behavioral features."""
    if req.user_id not in state.enrolled_users:
        raise HTTPException(404, f"User {req.user_id} not enrolled")
    
    scaler = state.scalers[req.user_id]
    features = np.array(req.keystroke_features).reshape(1, -1)
    
    # Apply defenses if enabled
    if state.defense_enabled.get("input_smoothing"):
        noise = np.random.normal(0, 0.1, features.shape)
        features = (features + features + noise) / 2
    
    if state.defense_enabled.get("feature_squeezing"):
        f_min, f_max = features.min(), features.max()
        f_range = f_max - f_min + 1e-8
        features = np.round((features - f_min) / f_range * 16) / 16 * f_range + f_min
    
    X_scaled = scaler.transform(features)
    
    results = {}
    
    # One-Class SVM
    ocsvm_key = f"{req.user_id}_ocsvm"
    if ocsvm_key in state.models:
        ocsvm = state.models[ocsvm_key]
        pred = ocsvm.predict(X_scaled)[0]
        score = float(ocsvm.decision_function(X_scaled)[0])
        results["ocsvm"] = {
            "prediction": "genuine" if pred == 1 else "impostor",
            "score": score,
            "confidence": min(abs(score) / 2, 1.0),
        }
    
    # Random Forest
    rf_key = f"{req.user_id}_rf"
    if rf_key in state.models:
        rf = state.models[rf_key]
        pred = int(rf.predict(X_scaled)[0])
        proba = float(rf.predict_proba(X_scaled)[0][1])
        results["rf"] = {
            "prediction": "genuine" if pred == 1 else "impostor",
            "score": proba,
            "confidence": abs(proba - 0.5) * 2,
        }
    
    # Ensemble decision
    scores = [r["score"] for r in results.values()]
    avg_score = np.mean(scores) if scores else 0
    
    # Dynamic threshold if defense enabled
    threshold = 0.5
    if state.defense_enabled.get("threshold_strictness"):
        threshold = 0.7  # Stricter
    
    is_genuine = avg_score > threshold
    
    auth_result = {
        "user_id": req.user_id,
        "authenticated": bool(is_genuine),
        "overall_score": float(avg_score),
        "threshold": threshold,
        "model_results": results,
        "defenses_active": [k for k, v in state.defense_enabled.items() if v],
    }
    
    state.auth_history.append(auth_result)
    
    return auth_result

@app.post("/api/attack/simulate")
def simulate_attack(req: AttackRequest):
    """Simulate an adversarial attack against the authentication system."""
    if req.user_id not in state.enrolled_users:
        raise HTTPException(404, f"User {req.user_id} not enrolled")
    
    user_data = state.enrolled_users[req.user_id]
    mean = np.array(user_data["mean_features"])
    std = np.array(user_data["std_features"])
    scaler = state.scalers[req.user_id]
    
    # Generate impostor sample
    impostor = mean + np.random.normal(0, std * 3, mean.shape)
    
    if req.attack_type == "fgsm":
        # FGSM: add perturbation in gradient direction
        epsilon = req.epsilon or 0.2
        # Estimate gradient via finite differences
        ocsvm_key = f"{req.user_id}_ocsvm"
        if ocsvm_key in state.models:
            model = state.models[ocsvm_key]
            X_base = scaler.transform(impostor.reshape(1, -1))
            grad = np.zeros_like(X_base)
            for i in range(X_base.shape[1]):
                X_plus = X_base.copy(); X_plus[0, i] += 1e-4
                X_minus = X_base.copy(); X_minus[0, i] -= 1e-4
                grad[0, i] = (model.decision_function(X_plus) - model.decision_function(X_minus)) / 2e-4
            
            perturbation = epsilon * np.sign(grad)
            X_adv = X_base + perturbation
            
            score_before = float(model.decision_function(X_base)[0])
            score_after = float(model.decision_function(X_adv)[0])
            pred_before = "genuine" if model.predict(X_base)[0] == 1 else "impostor"
            pred_after = "genuine" if model.predict(X_adv)[0] == 1 else "impostor"
        else:
            score_before, score_after = 0, 0
            pred_before, pred_after = "unknown", "unknown"
            perturbation = np.zeros_like(impostor)
        
        result = {
            "attack_type": "FGSM",
            "epsilon": epsilon,
            "score_before": score_before,
            "score_after": score_after,
            "prediction_before": pred_before,
            "prediction_after": pred_after,
            "score_change": score_after - score_before,
            "attack_success": pred_before == "impostor" and pred_after == "genuine",
            "perturbation_magnitude": float(np.linalg.norm(perturbation)),
        }
    
    elif req.attack_type == "pgd":
        epsilon = req.epsilon or 0.2
        n_iter = req.n_iterations or 20
        alpha = epsilon / 4
        
        ocsvm_key = f"{req.user_id}_ocsvm"
        if ocsvm_key in state.models:
            model = state.models[ocsvm_key]
            X_base = scaler.transform(impostor.reshape(1, -1))
            X_adv = X_base.copy()
            
            trajectory = []
            for step in range(n_iter):
                grad = np.zeros_like(X_adv)
                for i in range(X_adv.shape[1]):
                    X_p = X_adv.copy(); X_p[0, i] += 1e-4
                    X_m = X_adv.copy(); X_m[0, i] -= 1e-4
                    grad[0, i] = (model.decision_function(X_p) - model.decision_function(X_m)) / 2e-4
                
                X_adv = X_adv + alpha * np.sign(grad)
                perturbation = np.clip(X_adv - X_base, -epsilon, epsilon)
                X_adv = X_base + perturbation
                
                trajectory.append(float(model.decision_function(X_adv)[0]))
            
            score_before = float(model.decision_function(X_base)[0])
            score_after = float(model.decision_function(X_adv)[0])
            pred_before = "genuine" if model.predict(X_base)[0] == 1 else "impostor"
            pred_after = "genuine" if model.predict(X_adv)[0] == 1 else "impostor"
        else:
            score_before, score_after = 0, 0
            pred_before, pred_after = "unknown", "unknown"
            trajectory = []
        
        result = {
            "attack_type": "PGD",
            "epsilon": epsilon,
            "n_iterations": n_iter,
            "score_before": score_before,
            "score_after": score_after,
            "prediction_before": pred_before,
            "prediction_after": pred_after,
            "attack_success": pred_before == "impostor" and pred_after == "genuine",
            "trajectory": trajectory,
        }
    
    elif req.attack_type == "mimicry":
        noise = req.noise_level or 0.1
        # Generate sample from target's distribution
        mimicry_sample = np.random.normal(mean, std * (1 + noise))
        X_mimicry = scaler.transform(mimicry_sample.reshape(1, -1))
        
        scores = {}
        for model_key_suffix in ["_ocsvm", "_rf"]:
            mk = f"{req.user_id}{model_key_suffix}"
            if mk in state.models:
                m = state.models[mk]
                pred = m.predict(X_mimicry)[0]
                if hasattr(m, 'decision_function'):
                    sc = float(m.decision_function(X_mimicry)[0])
                elif hasattr(m, 'predict_proba'):
                    sc = float(m.predict_proba(X_mimicry)[0][1])
                else:
                    sc = 0
                scores[model_key_suffix.strip("_")] = {
                    "prediction": "genuine" if (pred == 1) else "impostor",
                    "score": sc,
                }
        
        any_fooled = any(s["prediction"] == "genuine" for s in scores.values())
        
        result = {
            "attack_type": "Statistical Mimicry",
            "noise_level": noise,
            "model_results": scores,
            "attack_success": any_fooled,
        }
    
    else:
        raise HTTPException(400, f"Unknown attack type: {req.attack_type}")
    
    state.attack_history.append(result)
    return result

@app.post("/api/defense/toggle")
def toggle_defense(req: DefenseToggle):
    """Enable or disable a defense mechanism."""
    if req.defense_name not in state.defense_enabled:
        raise HTTPException(400, f"Unknown defense: {req.defense_name}")
    
    state.defense_enabled[req.defense_name] = req.enabled
    return {
        "defense": req.defense_name,
        "enabled": req.enabled,
        "all_defenses": state.defense_enabled,
    }

@app.get("/api/stats")
def get_stats():
    """Get authentication and attack statistics."""
    n_auth = len(state.auth_history)
    n_attacks = len(state.attack_history)
    
    genuine_count = sum(1 for a in state.auth_history if a.get("authenticated"))
    attack_success = sum(1 for a in state.attack_history if a.get("attack_success"))
    
    return {
        "total_authentications": n_auth,
        "total_attacks": n_attacks,
        "genuine_rate": genuine_count / max(n_auth, 1),
        "attack_success_rate": attack_success / max(n_attacks, 1),
        "enrolled_users": len(state.enrolled_users),
        "defenses_active": [k for k, v in state.defense_enabled.items() if v],
    }

@app.post("/api/keystroke/extract")
def extract_keystroke(data: KeystrokeData):
    """Extract features from raw keystroke events (browser → features)."""
    features = extract_keystroke_features(data.keystrokes)
    return {
        "user_id": data.user_id,
        "features": features,
        "n_keystrokes": len(data.keystrokes),
        "text_length": len(data.text),
    }

@app.post("/api/mouse/extract")
def extract_mouse(data: MouseData):
    """Extract features from raw mouse movement events."""
    features = extract_mouse_features(data.movements)
    return {
        "user_id": data.user_id,
        "features": features,
        "n_events": len(data.movements),
    }


# ============================================================
# STARTUP: Pre-seed with demo user
# ============================================================

@app.on_event("startup")
def seed_demo_data():
    """Create a demo user with pre-generated enrollment data."""
    np.random.seed(42)
    n_features = 20
    
    # Generate realistic enrollment data for demo_user
    base_profile = np.random.uniform(0.05, 0.3, n_features)
    samples = []
    for _ in range(20):
        sample = base_profile + np.random.normal(0, base_profile * 0.1)
        samples.append(sample.tolist())
    
    # Enroll via the API logic
    req = EnrollRequest(user_id="demo_user", keystroke_samples=samples)
    enroll_user(req)
    print("[OK] Demo user 'demo_user' pre-enrolled with 20 keystroke samples")
