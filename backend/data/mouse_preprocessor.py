import numpy as np
import pandas as pd
import os
from collections import defaultdict

np.random.seed(42)

class MouseDataGenerator:
    
    def __init__(self, n_users=10, sessions_per_user=12):
        self.n_users = n_users
        self.sessions_per_user = sessions_per_user
        self.user_profiles = {}
    
    def _create_user_profile(self, user_id):
        return {
            "avg_speed": np.random.uniform(200, 800),       # pixels/sec
            "speed_std": np.random.uniform(50, 200),
            "avg_acceleration": np.random.uniform(500, 2000),
            "curvature_tendency": np.random.uniform(0.1, 0.8),  # 0=straight, 1=curvy
            "pause_frequency": np.random.uniform(0.05, 0.3),
            "avg_pause_duration": np.random.uniform(0.3, 2.0),
            "click_speed": np.random.uniform(0.05, 0.3),  # seconds held
            "double_click_interval": np.random.uniform(0.1, 0.4),
            "preferred_direction": np.random.choice(["horizontal", "vertical", "diagonal"]),
            "movement_smoothness": np.random.uniform(0.3, 0.9),
        }
    
    def _generate_mouse_movement(self, profile, duration_sec=60):
        records = []
        t = 0.0
        x, y = 500.0, 400.0  # Start position
        dt = 0.015  # ~67Hz sampling rate (typical)
        
        n_targets = int(duration_sec / 3)  # New target every ~3 seconds
        targets = [(np.random.uniform(50, 1870), np.random.uniform(50, 1030)) 
                   for _ in range(n_targets)]
        
        target_idx = 0
        tx, ty = targets[0]
        state = "Move"
        button = "NoButton"
        
        while t < duration_sec:
            dx = tx - x
            dy = ty - y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < 10:
                records.append([round(t, 4), round(x, 1), round(y, 1), "Pressed", "Left"])
                t += profile["click_speed"] * np.random.uniform(0.8, 1.2)
                records.append([round(t, 4), round(x, 1), round(y, 1), "Released", "Left"])
                t += dt
                
                if np.random.random() < profile["pause_frequency"]:
                    t += profile["avg_pause_duration"] * np.random.uniform(0.5, 1.5)
                
                target_idx = (target_idx + 1) % len(targets)
                tx, ty = targets[target_idx]
                continue
            
            speed = profile["avg_speed"] + np.random.normal(0, profile["speed_std"])
            speed = max(50, speed)
            
            angle = np.arctan2(dy, dx)
            curvature_noise = profile["curvature_tendency"] * np.random.normal(0, 0.3)
            angle += curvature_noise
            
            step = min(speed * dt, dist)
            
            jitter = np.random.normal(0, 1.5 * (1 - profile["movement_smoothness"]))
            
            x += step * np.cos(angle) + jitter
            y += step * np.sin(angle) + jitter
            
            x = np.clip(x, 0, 1920)
            y = np.clip(y, 0, 1080)
            
            records.append([round(t, 4), round(x, 1), round(y, 1), "Move", "NoButton"])
            t += dt
        
        return pd.DataFrame(records, columns=["timestamp", "x", "y", "state", "button"])
    
    def generate_dataset(self, output_dir="../../datasets/mouse"):
        os.makedirs(output_dir, exist_ok=True)
        
        train_dir = os.path.join(output_dir, "training_files")
        test_dir = os.path.join(output_dir, "test_files")
        
        for user_id in range(1, self.n_users + 1):
            user_name = f"user{user_id}"
            profile = self._create_user_profile(user_id)
            self.user_profiles[user_name] = profile
            
            user_train_dir = os.path.join(train_dir, user_name)
            os.makedirs(user_train_dir, exist_ok=True)
            
            n_train = self.sessions_per_user // 2
            for sess_id in range(n_train):
                session_df = self._generate_mouse_movement(profile, duration_sec=60)
                session_df.to_csv(
                    os.path.join(user_train_dir, f"session_{sess_id+1}.csv"),
                    index=False
                )
            
            user_test_dir = os.path.join(test_dir, user_name)
            os.makedirs(user_test_dir, exist_ok=True)
            
            n_test = self.sessions_per_user - n_train
            labels = []
            
            for sess_id in range(n_test):
                if sess_id < n_test // 2:
                    session_df = self._generate_mouse_movement(profile, duration_sec=60)
                    labels.append({"session": f"session_{sess_id+1}.csv", "is_illegal": 0})
                else:
                    imp_id = (user_id % self.n_users) + 1
                    imp_profile = self._create_user_profile(imp_id + 100)  # different seed
                    session_df = self._generate_mouse_movement(imp_profile, duration_sec=60)
                    labels.append({"session": f"session_{sess_id+1}.csv", "is_illegal": 1})
                
                session_df.to_csv(
                    os.path.join(user_test_dir, f"session_{sess_id+1}.csv"),
                    index=False
                )
            
            pd.DataFrame(labels).to_csv(
                os.path.join(user_test_dir, "labels.csv"), index=False
            )
        
        np.save(os.path.join(output_dir, "user_profiles.npy"), self.user_profiles)
        
        print(f"✓ Mouse dataset generated: {output_dir}")
        print(f"  Users: {self.n_users}")
        print(f"  Training sessions per user: {self.sessions_per_user // 2}")
        print(f"  Test sessions per user: {self.sessions_per_user - self.sessions_per_user // 2}")
        
        return output_dir

class MouseFeatureExtractor:
    
    FEATURE_NAMES = [
        "mean_speed", "std_speed", "max_speed", "min_speed",
        "mean_acceleration", "std_acceleration", "max_acceleration",
        "mean_angle", "std_angle", "direction_changes",
        "mean_curvature", "std_curvature", "path_straightness",
        "click_count", "mean_click_duration", "std_click_duration",
        "double_click_count",
        "pause_count", "mean_pause_duration", "total_pause_time",
        "total_distance", "total_displacement", "path_efficiency",
        "mean_jerk", "movement_variability",
        "session_duration", "active_time_ratio",
        "dir_N", "dir_NE", "dir_E", "dir_SE",
        "dir_S", "dir_SW", "dir_W", "dir_NW",
    ]
    
    @staticmethod
    def extract_from_session(session_df):
        df = session_df.copy()
        features = {}
        
        moves = df[df["state"] == "Move"].copy()
        
        if len(moves) < 10:
            return {f: 0.0 for f in MouseFeatureExtractor.FEATURE_NAMES}
        
        moves["dt"] = moves["timestamp"].diff().fillna(0.015)
        moves["dt"] = moves["dt"].clip(lower=0.001)  # Avoid division by zero
        
        moves["dx"] = moves["x"].diff().fillna(0)
        moves["dy"] = moves["y"].diff().fillna(0)
        
        moves["distance"] = np.sqrt(moves["dx"]**2 + moves["dy"]**2)
        moves["speed"] = moves["distance"] / moves["dt"]
        moves["speed"] = moves["speed"].clip(upper=5000)  # Cap outliers
        
        features["mean_speed"] = moves["speed"].mean()
        features["std_speed"] = moves["speed"].std()
        features["max_speed"] = moves["speed"].quantile(0.95)  # Use 95th percentile
        features["min_speed"] = moves["speed"].quantile(0.05)
        
        moves["accel"] = moves["speed"].diff().fillna(0) / moves["dt"]
        moves["accel"] = moves["accel"].clip(lower=-10000, upper=10000)
        
        features["mean_acceleration"] = moves["accel"].abs().mean()
        features["std_acceleration"] = moves["accel"].std()
        features["max_acceleration"] = moves["accel"].abs().quantile(0.95)
        
        moves["angle"] = np.arctan2(moves["dy"], moves["dx"])
        features["mean_angle"] = moves["angle"].mean()
        features["std_angle"] = moves["angle"].std()
        
        angle_diff = moves["angle"].diff().abs()
        features["direction_changes"] = (angle_diff > np.pi/4).sum()
        
        moves["curvature"] = angle_diff.fillna(0) / (moves["distance"] + 1e-6)
        moves["curvature"] = moves["curvature"].clip(upper=10)
        features["mean_curvature"] = moves["curvature"].mean()
        features["std_curvature"] = moves["curvature"].std()
        
        total_dist = moves["distance"].sum()
        displacement = np.sqrt(
            (moves["x"].iloc[-1] - moves["x"].iloc[0])**2 +
            (moves["y"].iloc[-1] - moves["y"].iloc[0])**2
        )
        features["path_straightness"] = displacement / (total_dist + 1e-6)
        
        presses = df[df["state"] == "Pressed"]
        releases = df[df["state"] == "Released"]
        features["click_count"] = len(presses)
        
        if len(presses) > 0 and len(releases) > 0:
            click_durations = []
            for _, press in presses.iterrows():
                matching_release = releases[releases["timestamp"] > press["timestamp"]]
                if len(matching_release) > 0:
                    duration = matching_release.iloc[0]["timestamp"] - press["timestamp"]
                    if duration < 5.0:  # Sanity check
                        click_durations.append(duration)
            
            if click_durations:
                features["mean_click_duration"] = np.mean(click_durations)
                features["std_click_duration"] = np.std(click_durations) if len(click_durations) > 1 else 0
            else:
                features["mean_click_duration"] = 0
                features["std_click_duration"] = 0
            
            if len(presses) > 1:
                press_intervals = presses["timestamp"].diff().dropna()
                features["double_click_count"] = (press_intervals < 0.5).sum()
            else:
                features["double_click_count"] = 0
        else:
            features["mean_click_duration"] = 0
            features["std_click_duration"] = 0
            features["double_click_count"] = 0
        
        large_dt = moves["dt"][moves["dt"] > 0.5]
        features["pause_count"] = len(large_dt)
        features["mean_pause_duration"] = large_dt.mean() if len(large_dt) > 0 else 0
        features["total_pause_time"] = large_dt.sum() if len(large_dt) > 0 else 0
        
        features["total_distance"] = total_dist
        features["total_displacement"] = displacement
        features["path_efficiency"] = features["path_straightness"]
        
        moves["jerk"] = moves["accel"].diff().fillna(0) / moves["dt"]
        features["mean_jerk"] = moves["jerk"].abs().mean()
        features["movement_variability"] = moves["speed"].std() / (moves["speed"].mean() + 1e-6)
        
        features["session_duration"] = df["timestamp"].max() - df["timestamp"].min()
        active_time = moves["dt"][moves["dt"] < 0.5].sum()
        features["active_time_ratio"] = active_time / (features["session_duration"] + 1e-6)
        
        angles = moves["angle"].values
        bins = np.linspace(-np.pi, np.pi, 9)
        hist, _ = np.histogram(angles, bins=bins)
        hist = hist / (hist.sum() + 1e-6)  # Normalize
        
        dir_names = ["dir_E", "dir_NE", "dir_N", "dir_NW", 
                     "dir_W", "dir_SW", "dir_S", "dir_SE"]
        for name, val in zip(dir_names, hist):
            features[name] = val
        
        return features
    
    @staticmethod
    def extract_from_directory(session_dir):
        all_features = []
        
        for fname in sorted(os.listdir(session_dir)):
            if fname.endswith(".csv") and fname.startswith("session"):
                fpath = os.path.join(session_dir, fname)
                session_df = pd.read_csv(fpath)
                features = MouseFeatureExtractor.extract_from_session(session_df)
                features["session_file"] = fname
                all_features.append(features)
        
        return pd.DataFrame(all_features)

class MousePreprocessor:
    
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(dataset_dir, "training_files")
        self.test_dir = os.path.join(dataset_dir, "test_files")
        self.scaler = None
        self.feature_cols = MouseFeatureExtractor.FEATURE_NAMES
    
    def process_all_users(self):
        all_train = []
        all_test = []
        
        users = sorted([d for d in os.listdir(self.train_dir) 
                        if os.path.isdir(os.path.join(self.train_dir, d))])
        
        print(f"Processing {len(users)} users...")
        
        for user in users:
            train_features = MouseFeatureExtractor.extract_from_directory(
                os.path.join(self.train_dir, user)
            )
            train_features["user"] = user
            train_features["label"] = 1  # All training is genuine
            all_train.append(train_features)
            
            test_features = MouseFeatureExtractor.extract_from_directory(
                os.path.join(self.test_dir, user)
            )
            test_features["user"] = user
            
            labels_path = os.path.join(self.test_dir, user, "labels.csv")
            if os.path.exists(labels_path):
                labels = pd.read_csv(labels_path)
                test_features = test_features.merge(
                    labels, left_on="session_file", right_on="session", how="left"
                )
                test_features["label"] = 1 - test_features["is_illegal"].fillna(0)
            else:
                test_features["label"] = -1  # Unknown
            
            all_test.append(test_features)
        
        train_df = pd.concat(all_train, ignore_index=True)
        test_df = pd.concat(all_test, ignore_index=True)
        
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        
        return train_df, test_df
    
    def create_auth_dataset(self, train_df, test_df, target_user):
        from sklearn.preprocessing import StandardScaler
        
        genuine_train = train_df[train_df["user"] == target_user]
        
        impostor_train = train_df[train_df["user"] != target_user]
        n_genuine = len(genuine_train)
        if len(impostor_train) > n_genuine * 2:
            impostor_train = impostor_train.sample(n_genuine * 2, random_state=42)
        
        X_train = pd.concat([genuine_train, impostor_train])
        y_train = X_train["label"].values
        X_train = X_train[self.feature_cols].values
        
        user_test = test_df[test_df["user"] == target_user]
        X_test = user_test[self.feature_cols].values
        y_test = user_test["label"].values
        
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def save_processed(self, train_df, test_df, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "mouse_train_features.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "mouse_test_features.csv"), index=False)
        print(f"✓ Saved to {output_dir}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MOUSE_DIR = os.path.join(BASE_DIR, "../../datasets/mouse")
    PROCESSED_DIR = os.path.join(BASE_DIR, "../../datasets/mouse/processed")
    
    if not os.path.exists(os.path.join(MOUSE_DIR, "training_files")):
        print("=== Generating Synthetic Mouse Data ===")
        gen = MouseDataGenerator(n_users=10, sessions_per_user=12)
        gen.generate_dataset(output_dir=MOUSE_DIR)
    
    print("\n=== Extracting Mouse Features ===")
    preprocessor = MousePreprocessor(MOUSE_DIR)
    train_df, test_df = preprocessor.process_all_users()
    
    print("\n=== Feature Summary ===")
    print(f"Features: {len(MouseFeatureExtractor.FEATURE_NAMES)}")
    for i, name in enumerate(MouseFeatureExtractor.FEATURE_NAMES):
        print(f"  {i+1:2d}. {name}")
    
    print("\n=== Auth Dataset for user1 ===")
    X_train, X_test, y_train, y_test = preprocessor.create_auth_dataset(
        train_df, test_df, "user1"
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    print("\n=== Saving ===")
    preprocessor.save_processed(train_df, test_df, PROCESSED_DIR)
    
    print("\n✓ Mouse preprocessing pipeline complete!")
