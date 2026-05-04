"""Touchscreen gesture dataset generator and preprocessor."""

import numpy as np
import pandas as pd
import os

np.random.seed(42)


class TouchDataGenerator:
    """Generate synthetic touchscreen gesture data."""
    
    GESTURE_TYPES = ["swipe_up", "swipe_down", "swipe_left", "swipe_right",
                     "tap", "double_tap", "long_press", "pinch_in", "pinch_out"]
    
    def __init__(self, n_users=15, gestures_per_user=200):
        self.n_users = n_users
        self.gestures_per_user = gestures_per_user
    
    def _create_user_profile(self, user_id):
        """Each user has unique touch characteristics."""
        return {
            "swipe_speed": np.random.uniform(400, 1200),      # px/sec
            "swipe_speed_std": np.random.uniform(50, 150),
            "swipe_pressure": np.random.uniform(0.3, 0.9),    # 0-1
            "pressure_std": np.random.uniform(0.02, 0.1),
            "touch_area": np.random.uniform(30, 80),          # pixels
            "area_std": np.random.uniform(5, 15),
            "tap_duration": np.random.uniform(0.05, 0.2),     # seconds
            "long_press_duration": np.random.uniform(0.5, 2.0),
            "double_tap_interval": np.random.uniform(0.1, 0.35),
            "swipe_curvature": np.random.uniform(0.0, 0.3),   # deviation from straight
            "preferred_hand": np.random.choice(["left", "right"], p=[0.1, 0.9]),
            "finger_angle": np.random.uniform(-15, 15),       # degrees tilt
        }
    
    def _generate_swipe(self, profile, direction):
        """Generate a single swipe gesture."""
        speed = max(100, np.random.normal(profile["swipe_speed"], profile["swipe_speed_std"]))
        pressure = np.clip(np.random.normal(profile["swipe_pressure"], profile["pressure_std"]), 0.1, 1.0)
        area = max(10, np.random.normal(profile["touch_area"], profile["area_std"]))
        
        # Start position based on hand preference
        if profile["preferred_hand"] == "right":
            start_x = np.random.uniform(150, 350)
        else:
            start_x = np.random.uniform(30, 200)
        start_y = np.random.uniform(200, 600)
        
        # Direction vectors
        directions = {
            "swipe_up": (0, -1), "swipe_down": (0, 1),
            "swipe_left": (-1, 0), "swipe_right": (1, 0)
        }
        dx, dy = directions[direction]
        
        length = np.random.uniform(150, 400)
        duration = length / speed
        n_points = max(5, int(duration * 60))  # 60Hz
        
        # Generate path with curvature
        t_vals = np.linspace(0, duration, n_points)
        curvature = profile["swipe_curvature"] * np.sin(np.linspace(0, np.pi, n_points))
        
        points = []
        for i, t in enumerate(t_vals):
            progress = t / duration
            x = start_x + dx * length * progress + curvature[i] * dy * 50
            y = start_y + dy * length * progress + curvature[i] * dx * 50
            p = pressure * (1 - 0.2 * abs(progress - 0.5))  # Pressure dips at start/end
            a = area * (1 + 0.1 * np.sin(progress * np.pi))
            points.append([round(t, 4), round(x, 1), round(y, 1), round(p, 3), round(a, 1)])
        
        return pd.DataFrame(points, columns=["time", "x", "y", "pressure", "area"]), duration
    
    def _generate_tap(self, profile, is_double=False):
        """Generate tap or double tap."""
        if profile["preferred_hand"] == "right":
            x = np.random.uniform(150, 350)
        else:
            x = np.random.uniform(30, 200)
        y = np.random.uniform(100, 700)
        
        pressure = np.clip(np.random.normal(profile["swipe_pressure"] * 1.1, profile["pressure_std"]), 0.1, 1.0)
        area = max(10, np.random.normal(profile["touch_area"] * 0.8, profile["area_std"]))
        duration = max(0.02, np.random.normal(profile["tap_duration"], profile["tap_duration"] * 0.2))
        
        points = [
            [0.0, round(x, 1), round(y, 1), round(pressure, 3), round(area, 1)],
            [round(duration, 4), round(x + np.random.normal(0, 2), 1), 
             round(y + np.random.normal(0, 2), 1), round(pressure * 0.5, 3), round(area, 1)]
        ]
        
        if is_double:
            interval = max(0.05, np.random.normal(profile["double_tap_interval"], 0.05))
            t2 = duration + interval
            points.append([round(t2, 4), round(x + np.random.normal(0, 3), 1),
                          round(y + np.random.normal(0, 3), 1), round(pressure, 3), round(area, 1)])
            t3 = t2 + duration
            points.append([round(t3, 4), round(x + np.random.normal(0, 3), 1),
                          round(y + np.random.normal(0, 3), 1), round(pressure * 0.5, 3), round(area, 1)])
            duration = t3
        
        return pd.DataFrame(points, columns=["time", "x", "y", "pressure", "area"]), duration
    
    def _generate_long_press(self, profile):
        """Generate long press gesture."""
        if profile["preferred_hand"] == "right":
            x = np.random.uniform(150, 350)
        else:
            x = np.random.uniform(30, 200)
        y = np.random.uniform(100, 700)
        
        pressure = np.clip(np.random.normal(profile["swipe_pressure"], profile["pressure_std"]), 0.1, 1.0)
        area = max(10, np.random.normal(profile["touch_area"], profile["area_std"]))
        duration = max(0.3, np.random.normal(profile["long_press_duration"], 0.2))
        
        n_points = max(3, int(duration * 10))  # Lower rate for stationary
        points = []
        for i in range(n_points):
            t = duration * i / (n_points - 1)
            # Slight drift
            cx = x + np.random.normal(0, 1.5)
            cy = y + np.random.normal(0, 1.5)
            p = pressure * (0.9 + 0.1 * np.sin(t * 2))
            points.append([round(t, 4), round(cx, 1), round(cy, 1), round(p, 3), round(area, 1)])
        
        return pd.DataFrame(points, columns=["time", "x", "y", "pressure", "area"]), duration
    
    def generate_gesture(self, profile, gesture_type):
        """Generate a single gesture of given type."""
        if gesture_type.startswith("swipe"):
            return self._generate_swipe(profile, gesture_type)
        elif gesture_type == "tap":
            return self._generate_tap(profile, is_double=False)
        elif gesture_type == "double_tap":
            return self._generate_tap(profile, is_double=True)
        elif gesture_type == "long_press":
            return self._generate_long_press(profile)
        elif gesture_type.startswith("pinch"):
            # Simplified: treat as two-finger swipe
            return self._generate_swipe(profile, "swipe_up" if "in" in gesture_type else "swipe_down")
        else:
            return self._generate_tap(profile)
    
    def generate_dataset(self, output_dir="../../datasets/touch"):
        """Generate full touch gesture dataset."""
        os.makedirs(output_dir, exist_ok=True)
        
        all_records = []
        
        for user_id in range(1, self.n_users + 1):
            profile = self._create_user_profile(user_id)
            user_name = f"user{user_id:02d}"
            
            for g_idx in range(self.gestures_per_user):
                gesture_type = np.random.choice(self.GESTURE_TYPES, p=[
                    0.12, 0.12, 0.12, 0.12,  # swipes
                    0.2, 0.1, 0.08,           # taps
                    0.07, 0.07                 # pinch
                ])
                
                gesture_df, duration = self.generate_gesture(profile, gesture_type)
                
                all_records.append({
                    "user": user_name,
                    "gesture_id": g_idx,
                    "gesture_type": gesture_type,
                    "n_points": len(gesture_df),
                    "duration": round(duration, 4),
                    "start_x": gesture_df["x"].iloc[0],
                    "start_y": gesture_df["y"].iloc[0],
                    "end_x": gesture_df["x"].iloc[-1],
                    "end_y": gesture_df["y"].iloc[-1],
                    "mean_pressure": round(gesture_df["pressure"].mean(), 4),
                    "std_pressure": round(gesture_df["pressure"].std(), 4),
                    "max_pressure": round(gesture_df["pressure"].max(), 4),
                    "mean_area": round(gesture_df["area"].mean(), 2),
                    "std_area": round(gesture_df["area"].std(), 2),
                })
        
        df = pd.DataFrame(all_records)
        df.to_csv(os.path.join(output_dir, "touch_gestures.csv"), index=False)
        
        print(f"✓ Touch dataset generated: {output_dir}")
        print(f"  Users: {self.n_users}")
        print(f"  Gestures per user: {self.gestures_per_user}")
        print(f"  Total samples: {len(df)}")
        
        return df


class TouchPreprocessor:
    """Feature extraction and preprocessing for touch gesture data."""
    
    FEATURE_COLS = [
        "duration", "n_points",
        "start_x", "start_y", "end_x", "end_y",
        "mean_pressure", "std_pressure", "max_pressure",
        "mean_area", "std_area",
        # Derived features
        "swipe_distance", "swipe_speed", "swipe_angle",
        "pressure_range", "area_range",
        "gesture_type_encoded",
    ]
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
    
    def load_and_process(self):
        """Load raw data and extract features."""
        self.df = pd.read_csv(self.data_path)
        
        # Derived features
        dx = self.df["end_x"] - self.df["start_x"]
        dy = self.df["end_y"] - self.df["start_y"]
        self.df["swipe_distance"] = np.sqrt(dx**2 + dy**2)
        self.df["swipe_speed"] = self.df["swipe_distance"] / (self.df["duration"] + 1e-6)
        self.df["swipe_angle"] = np.arctan2(dy, dx)
        self.df["pressure_range"] = self.df["max_pressure"] - self.df["mean_pressure"]
        self.df["area_range"] = self.df["std_area"] * 2
        
        # Encode gesture type
        gesture_map = {g: i for i, g in enumerate(TouchDataGenerator.GESTURE_TYPES)}
        self.df["gesture_type_encoded"] = self.df["gesture_type"].map(gesture_map)
        
        print(f"Processed {len(self.df)} touch gestures")
        return self.df
    
    def create_auth_dataset(self, target_user, n_impostors=5):
        """Create authentication dataset for a target user."""
        from sklearn.preprocessing import StandardScaler
        
        genuine = self.df[self.df["user"] == target_user]
        
        # Train/test split (70/30)
        n_train = int(len(genuine) * 0.7)
        genuine_train = genuine.iloc[:n_train]
        genuine_test = genuine.iloc[n_train:]
        
        # Impostors
        other_users = [u for u in self.df["user"].unique() if u != target_user]
        imp_users = np.random.choice(other_users, min(n_impostors, len(other_users)), replace=False)
        impostor = self.df[self.df["user"].isin(imp_users)]
        
        n_imp_train = min(len(genuine_train), len(impostor))
        impostor_train = impostor.sample(n_imp_train, random_state=42)
        impostor_test = impostor[~impostor.index.isin(impostor_train.index)].sample(
            min(len(genuine_test), len(impostor) - n_imp_train), random_state=42
        )
        
        # Combine
        train = pd.concat([genuine_train, impostor_train])
        test = pd.concat([genuine_test, impostor_test])
        
        y_train = (train["user"] == target_user).astype(int).values
        y_test = (test["user"] == target_user).astype(int).values
        
        X_train = train[self.FEATURE_COLS].values
        X_test = test[self.FEATURE_COLS].values
        
        # Clean and normalize
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"Touch auth dataset for {target_user}:")
        print(f"  Train: {X_train.shape} (genuine={sum(y_train)}, impostor={len(y_train)-sum(y_train)})")
        print(f"  Test:  {X_test.shape} (genuine={sum(y_test)}, impostor={len(y_test)-sum(y_test)})")
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TOUCH_DIR = os.path.join(BASE_DIR, "../../datasets/touch")
    
    # Generate
    print("=== Generating Touch Dataset ===")
    gen = TouchDataGenerator(n_users=15, gestures_per_user=200)
    df = gen.generate_dataset(output_dir=TOUCH_DIR)
    
    # Process
    print("\n=== Processing Touch Data ===")
    proc = TouchPreprocessor(os.path.join(TOUCH_DIR, "touch_gestures.csv"))
    proc.load_and_process()
    
    # Auth dataset
    print("\n=== Auth Dataset for user01 ===")
    X_train, X_test, y_train, y_test = proc.create_auth_dataset("user01")
    
    print("\n✓ Touch preprocessing pipeline complete!")
