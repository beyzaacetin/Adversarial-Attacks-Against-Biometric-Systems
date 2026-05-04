"""CMU-format keystroke dynamics dataset generator."""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Password characters for column naming
KEYS = ["period", "t", "i", "e", "five", "Shift.r", "o", "a", "n", "l", "Return"]

def generate_column_names():
    """Generate the 31 feature column names matching CMU format."""
    cols = []
    for i, key in enumerate(KEYS):
        cols.append(f"H.{key}")  # Hold time
        if i < len(KEYS) - 1:
            next_key = KEYS[i + 1]
            cols.append(f"DD.{key}.{next_key}")  # Down-Down
            cols.append(f"UD.{key}.{next_key}")   # Up-Down
    return cols

def generate_user_profile(user_id):
    """
    Generate a unique typing profile for a user.
    Each user has their own mean and std for each timing feature.
    This simulates real-world individual typing patterns.
    """
    profile = {}
    
    # Base typing speed (some users are faster, some slower)
    speed_factor = np.random.uniform(0.7, 1.5)
    consistency = np.random.uniform(0.05, 0.20)  # How consistent they are
    
    for key in KEYS:
        # Hold times: typically 70-150ms
        profile[f"H.{key}_mean"] = np.random.uniform(0.06, 0.18) * speed_factor
        profile[f"H.{key}_std"] = profile[f"H.{key}_mean"] * consistency
    
    for i in range(len(KEYS) - 1):
        key1, key2 = KEYS[i], KEYS[i + 1]
        
        # DD times: typically 100-300ms
        dd_mean = np.random.uniform(0.08, 0.35) * speed_factor
        profile[f"DD.{key1}.{key2}_mean"] = dd_mean
        profile[f"DD.{key1}.{key2}_std"] = dd_mean * consistency
        
        # UD times: DD - Hold (can be negative for overlapping keystrokes)
        ud_mean = dd_mean - profile[f"H.{key1}_mean"]
        profile[f"UD.{key1}.{key2}_mean"] = ud_mean
        profile[f"UD.{key1}.{key2}_std"] = abs(ud_mean) * consistency * 1.2
    
    return profile

def generate_typing_samples(profile, n_samples=400):
    """Generate n_samples typing samples from a user profile."""
    cols = generate_column_names()
    data = []
    
    for _ in range(n_samples):
        sample = []
        for col in cols:
            mean = profile[f"{col}_mean"]
            std = profile[f"{col}_std"]
            value = np.random.normal(mean, std)
            # Hold and DD times should be positive
            if col.startswith("H.") or col.startswith("DD."):
                value = max(0.01, value)
            sample.append(round(value, 6))
        data.append(sample)
    
    return data

def generate_dataset(n_subjects=51, n_reps=400, output_dir="../../datasets/keystroke"):
    """Generate the full CMU-format keystroke dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    cols = generate_column_names()
    all_rows = []
    user_profiles = {}
    
    print(f"Generating keystroke data for {n_subjects} subjects...")
    
    for subj_idx in range(1, n_subjects + 1):
        subj_id = f"s{subj_idx:03d}"
        profile = generate_user_profile(subj_idx)
        user_profiles[subj_id] = profile
        
        samples = generate_typing_samples(profile, n_reps)
        
        for rep_idx, sample in enumerate(samples):
            session = (rep_idx // 50) + 1
            rep_in_session = (rep_idx % 50) + 1
            row = [subj_id, session, rep_in_session] + sample
            all_rows.append(row)
    
    # Create DataFrame
    meta_cols = ["subject", "sessionIndex", "rep"]
    df = pd.DataFrame(all_rows, columns=meta_cols + cols)
    
    # Save
    csv_path = os.path.join(output_dir, "DSL-StrongPasswordData.csv")
    df.to_csv(csv_path, index=False)
    
    # Save user profiles for later verification
    profiles_path = os.path.join(output_dir, "user_profiles.npy")
    np.save(profiles_path, user_profiles)
    
    print(f"✓ Dataset saved: {csv_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Subjects: {df['subject'].nunique()}")
    print(f"  Features: {len(cols)}")
    print(f"  Samples per subject: {n_reps}")
    print(f"\nFeature columns:")
    for i, col in enumerate(cols):
        print(f"  {i+1:2d}. {col}")
    
    return df

if __name__ == "__main__":
    df = generate_dataset()
    
    # Quick stats
    print("\n--- Sample Statistics (subject s001) ---")
    s001 = df[df["subject"] == "s001"]
    feature_cols = [c for c in df.columns if c not in ["subject", "sessionIndex", "rep"]]
    print(s001[feature_cols].describe().round(4).to_string())
