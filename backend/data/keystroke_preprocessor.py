"""Keystroke dynamics preprocessor for CMU-format data."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import json

class KeystrokePreprocessor:
    """Preprocessor for CMU Keystroke Dynamics data."""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.df = None
        self.user_stats = {}
        
    def load_data(self, path=None):
        """Load keystroke dataset from CSV."""
        path = path or self.data_path
        if path is None:
            raise ValueError("Data path not specified")
        
        self.df = pd.read_csv(path)
        self.feature_cols = [c for c in self.df.columns 
                           if c not in ["subject", "sessionIndex", "rep"]]
        
        print(f"Loaded {len(self.df)} samples, {self.df['subject'].nunique()} subjects")
        print(f"Features: {len(self.feature_cols)}")
        return self.df
    
    def extract_derived_features(self):
        """Add derived features beyond raw timing data."""
        df = self.df.copy()
        
        hold_cols = [c for c in self.feature_cols if c.startswith("H.")]
        df["avg_hold_time"] = df[hold_cols].mean(axis=1)
        df["std_hold_time"] = df[hold_cols].std(axis=1)
        dd_cols = [c for c in self.feature_cols if c.startswith("DD.")]
        df["avg_dd_time"] = df[dd_cols].mean(axis=1)
        df["std_dd_time"] = df[dd_cols].std(axis=1)
        ud_cols = [c for c in self.feature_cols if c.startswith("UD.")]
        df["avg_ud_time"] = df[ud_cols].mean(axis=1)
        df["std_ud_time"] = df[ud_cols].std(axis=1)
        total_time = df[dd_cols].sum(axis=1)
        df["typing_speed"] = 11.0 / total_time
        df["rhythm_cv"] = df["std_dd_time"] / (df["avg_dd_time"] + 1e-8)
        df["overlap_ratio"] = (df[ud_cols] < 0).sum(axis=1) / len(ud_cols)
        
        # Update feature columns
        derived = ["avg_hold_time", "std_hold_time", "avg_dd_time", "std_dd_time",
                   "avg_ud_time", "std_ud_time", "typing_speed", "rhythm_cv", "overlap_ratio"]
        self.feature_cols = self.feature_cols + derived
        self.df = df
        
        print(f"Added {len(derived)} derived features. Total: {len(self.feature_cols)}")
        return df
    
    def compute_user_statistics(self):
        """Compute per-user statistics for anomaly-based authentication."""
        for subject in self.df["subject"].unique():
            user_data = self.df[self.df["subject"] == subject]
            enrollment = user_data[user_data["sessionIndex"] <= 4]
            
            self.user_stats[subject] = {
                "mean": enrollment[self.feature_cols].mean().values,
                "std": enrollment[self.feature_cols].std().values,
                "median": enrollment[self.feature_cols].median().values,
                "n_enrollment": len(enrollment)
            }
        
        print(f"Computed statistics for {len(self.user_stats)} users")
        return self.user_stats
    
    def create_authentication_dataset(self, target_subject, n_impostors=5):
        """Create binary classification dataset for a target user."""
        genuine = self.df[self.df["subject"] == target_subject].copy()
        genuine["label"] = 1
        genuine_train = genuine[genuine["sessionIndex"] <= 4]
        genuine_test = genuine[genuine["sessionIndex"] > 4]
        
        other_subjects = [s for s in self.df["subject"].unique() if s != target_subject]
        impostor_subjects = np.random.choice(other_subjects,
                                             min(n_impostors, len(other_subjects)),
                                             replace=False)
        impostor_data = self.df[self.df["subject"].isin(impostor_subjects)].copy()
        impostor_data["label"] = 0
        impostor_test_samples = []
        for imp_subj in impostor_subjects:
            imp_data = impostor_data[impostor_data["subject"] == imp_subj]
            samples = imp_data.sample(min(5, len(imp_data)), random_state=42)
            impostor_test_samples.append(samples)
        
        impostor_test = pd.concat(impostor_test_samples) if impostor_test_samples else pd.DataFrame()
        impostor_train = impostor_data[~impostor_data.index.isin(impostor_test.index)]
        impostor_train = impostor_train.sample(min(len(genuine_train), len(impostor_train)),
                                               random_state=42)
        train_df = pd.concat([genuine_train, impostor_train])
        test_df = pd.concat([genuine_test, impostor_test])
        
        X_train = train_df[self.feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[self.feature_cols].values
        y_test = test_df["label"].values
        
        # Normalize
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"Authentication dataset for {target_subject}:")
        print(f"  Train: {len(X_train)} ({sum(y_train)} genuine, {len(y_train)-sum(y_train)} impostor)")
        print(f"  Test:  {len(X_test)} ({sum(y_test)} genuine, {len(y_test)-sum(y_test)} impostor)")
        
        return X_train, X_test, y_train, y_test
    
    def create_anomaly_detection_dataset(self, target_subject):
        """Create one-class dataset with genuine-only training samples."""
        genuine = self.df[self.df["subject"] == target_subject]
        train_data = genuine[genuine["sessionIndex"] <= 4]
        X_train = train_data[self.feature_cols].values
        test_genuine = genuine[genuine["sessionIndex"] > 4]
        X_test_genuine = test_genuine[self.feature_cols].values
        other = self.df[self.df["subject"] != target_subject]
        impostor_samples = other.groupby("subject").apply(
            lambda x: x.sample(min(5, len(x)), random_state=42)
        ).reset_index(drop=True)
        X_test_impostor = impostor_samples[self.feature_cols].values
        
        # Normalize
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test_genuine = self.scaler.transform(X_test_genuine)
        X_test_impostor = self.scaler.transform(X_test_impostor)
        
        print(f"Anomaly detection dataset for {target_subject}:")
        print(f"  Train (genuine only): {len(X_train)}")
        print(f"  Test genuine: {len(X_test_genuine)}")
        print(f"  Test impostor: {len(X_test_impostor)}")
        
        return X_train, X_test_genuine, X_test_impostor
    
    def create_sequence_dataset(self, target_subject, seq_length=10):
        """Create sequential dataset for LSTM-based models."""
        X_train, X_test, y_train, y_test = self.create_authentication_dataset(
            target_subject, n_impostors=5
        )
        
        def make_sequences(X, y, seq_len):
            sequences = []
            labels = []
            for i in range(len(X) - seq_len + 1):
                sequences.append(X[i:i+seq_len])
                # Majority vote for sequence label
                labels.append(1 if sum(y[i:i+seq_len]) > seq_len // 2 else 0)
            return np.array(sequences), np.array(labels)
        
        X_train_seq, y_train_seq = make_sequences(X_train, y_train, seq_length)
        X_test_seq, y_test_seq = make_sequences(X_test, y_test, seq_length)
        
        print(f"Sequence dataset (seq_len={seq_length}):")
        print(f"  Train: {X_train_seq.shape}")
        print(f"  Test:  {X_test_seq.shape}")
        
        return X_train_seq, X_test_seq, y_train_seq, y_test_seq
    
    def get_feature_importance_data(self):
        """Get data formatted for feature importance analysis."""
        return {
            "feature_names": self.feature_cols,
            "n_hold_features": len([c for c in self.feature_cols if c.startswith("H.")]),
            "n_dd_features": len([c for c in self.feature_cols if c.startswith("DD.")]),
            "n_ud_features": len([c for c in self.feature_cols if c.startswith("UD.")]),
            "n_derived_features": len(self.feature_cols) - 31  # 31 original CMU features
        }
    
    def save_processed_data(self, output_dir):
        """Save all processed data for later use."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed DataFrame
        self.df.to_csv(os.path.join(output_dir, "keystroke_processed.csv"), index=False)
        
        # Save feature columns list
        with open(os.path.join(output_dir, "feature_cols.json"), "w") as f:
            json.dump(self.feature_cols, f)
        
        # Save scaler parameters
        if hasattr(self.scaler, "mean_"):
            np.save(os.path.join(output_dir, "scaler_mean.npy"), self.scaler.mean_)
            np.save(os.path.join(output_dir, "scaler_scale.npy"), self.scaler.scale_)
        
        print(f"Processed data saved to {output_dir}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "../../datasets/keystroke")
    PROCESSED_DIR = os.path.join(BASE_DIR, "../../datasets/keystroke/processed")
    
    # Step 0: Generate synthetic data if not exists
    csv_path = os.path.join(DATA_DIR, "DSL-StrongPasswordData.csv")
    if not os.path.exists(csv_path):
        print("=== Generating synthetic CMU-format data ===")
        from generate_keystroke_data import generate_dataset
        generate_dataset(output_dir=DATA_DIR)
    
    # Step 1: Load
    print("\n=== Loading Data ===")
    preprocessor = KeystrokePreprocessor(csv_path)
    preprocessor.load_data()
    
    # Step 2: Extract derived features
    print("\n=== Extracting Derived Features ===")
    preprocessor.extract_derived_features()
    
    # Step 3: Compute user statistics
    print("\n=== Computing User Profiles ===")
    preprocessor.compute_user_statistics()
    
    # Step 4: Create datasets for subject s001
    print("\n=== Creating Binary Classification Dataset (s001) ===")
    X_train, X_test, y_train, y_test = preprocessor.create_authentication_dataset("s001")
    
    print("\n=== Creating Anomaly Detection Dataset (s001) ===")
    X_train_ad, X_test_gen, X_test_imp = preprocessor.create_anomaly_detection_dataset("s001")
    
    print("\n=== Creating Sequence Dataset (s001) ===")
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = preprocessor.create_sequence_dataset("s001")
    
    # Step 5: Feature info
    print("\n=== Feature Summary ===")
    feat_info = preprocessor.get_feature_importance_data()
    print(f"  Hold features:    {feat_info['n_hold_features']}")
    print(f"  DD features:      {feat_info['n_dd_features']}")
    print(f"  UD features:      {feat_info['n_ud_features']}")
    print(f"  Derived features: {feat_info['n_derived_features']}")
    print(f"  Total features:   {len(feat_info['feature_names'])}")
    
    # Step 6: Save
    print("\n=== Saving Processed Data ===")
    preprocessor.save_processed_data(PROCESSED_DIR)
    
    print("\n✓ Keystroke preprocessing pipeline complete!")
