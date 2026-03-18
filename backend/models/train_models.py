"""
Model Training Runner
======================
Trains all authentication models across all biometric modalities
and produces comprehensive evaluation results.

Usage:
    python train_models.py
"""

import sys
import os
import json
import time
import numpy as np

# Add paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "../data"))
sys.path.insert(0, BASE_DIR)

from auth_models import (
    OneClassSVMAuth, RandomForestAuth, MLPAuth, AutoencoderAuth,
    MultimodalFusion, ModelTrainer
)
from keystroke_preprocessor import KeystrokePreprocessor
from mouse_preprocessor import MousePreprocessor, MouseDataGenerator
from touch_preprocessor import TouchPreprocessor, TouchDataGenerator


def train_keystroke_models(trainer):
    """Train models on keystroke dynamics data."""
    print("\n" + "=" * 60)
    print("  KEYSTROKE DYNAMICS MODEL TRAINING")
    print("=" * 60)
    
    DATA_DIR = os.path.join(BASE_DIR, "../../datasets/keystroke")
    csv_path = os.path.join(DATA_DIR, "DSL-StrongPasswordData.csv")
    
    if not os.path.exists(csv_path):
        print("Generating keystroke data first...")
        from generate_keystroke_data import generate_dataset
        generate_dataset(output_dir=DATA_DIR)
    
    proc = KeystrokePreprocessor(csv_path)
    proc.load_data()
    proc.extract_derived_features()
    
    # Train for subject s001
    X_train, X_test, y_train, y_test = proc.create_authentication_dataset("s001", n_impostors=10)
    
    models, results = trainer.train_all_models(
        X_train, X_test, y_train, y_test, modality_name="keystroke"
    )
    
    # Feature importance from Random Forest
    rf_model = models["keystroke_rf"]
    feat_importance = rf_model.get_feature_importance(proc.feature_cols)
    print("\n  Top 10 Keystroke Features:")
    for fi in feat_importance[:10]:
        print(f"    {fi['feature']:30s} → {fi['importance']:.4f}")
    
    # Autoencoder error distribution
    ae_model = models["keystroke_ae"]
    X_train_ad, X_test_gen, X_test_imp = proc.create_anomaly_detection_dataset("s001")
    error_dist = ae_model.get_error_distribution(X_test_gen, X_test_imp)
    print(f"\n  Autoencoder Error Distribution:")
    print(f"    Genuine mean error:  {error_dist['genuine_mean']:.6f}")
    print(f"    Impostor mean error: {error_dist['impostor_mean']:.6f}")
    print(f"    Threshold:           {error_dist['threshold']:.6f}")
    print(f"    Separation ratio:    {error_dist['impostor_mean'] / (error_dist['genuine_mean'] + 1e-8):.2f}x")
    
    return models, results, X_test, y_test


def train_mouse_models(trainer):
    """Train models on mouse dynamics data."""
    print("\n" + "=" * 60)
    print("  MOUSE DYNAMICS MODEL TRAINING")
    print("=" * 60)
    
    MOUSE_DIR = os.path.join(BASE_DIR, "../../datasets/mouse")
    
    # Always regenerate with more sessions for better training
    gen = MouseDataGenerator(n_users=10, sessions_per_user=30)
    gen.generate_dataset(output_dir=MOUSE_DIR)
    
    proc = MousePreprocessor(MOUSE_DIR)
    train_df, test_df = proc.process_all_users()
    X_train, X_test, y_train, y_test = proc.create_auth_dataset(train_df, test_df, "user1")
    
    models, results = trainer.train_all_models(
        X_train, X_test, y_train, y_test, modality_name="mouse"
    )
    
    return models, results, X_test, y_test


def train_touch_models(trainer):
    """Train models on touch gesture data."""
    print("\n" + "=" * 60)
    print("  TOUCH GESTURE MODEL TRAINING")
    print("=" * 60)
    
    TOUCH_DIR = os.path.join(BASE_DIR, "../../datasets/touch")
    csv_path = os.path.join(TOUCH_DIR, "touch_gestures.csv")
    
    if not os.path.exists(csv_path):
        gen = TouchDataGenerator(n_users=15, gestures_per_user=200)
        gen.generate_dataset(output_dir=TOUCH_DIR)
    
    proc = TouchPreprocessor(csv_path)
    proc.load_and_process()
    X_train, X_test, y_train, y_test = proc.create_auth_dataset("user01", n_impostors=5)
    
    models, results = trainer.train_all_models(
        X_train, X_test, y_train, y_test, modality_name="touch"
    )
    
    return models, results, X_test, y_test


def test_multimodal_fusion(trainer, all_test_data):
    """Test multimodal fusion across modalities."""
    print("\n" + "=" * 60)
    print("  MULTIMODAL FUSION")
    print("=" * 60)
    
    # Use Random Forest scores from each modality
    # Since test sets have different sizes, we use the smallest common size
    min_size = min(len(d["y_test"]) for d in all_test_data.values())
    
    score_lists = {}
    y_common = None
    
    for modality, data in all_test_data.items():
        rf_key = f"{modality}_rf"
        if rf_key in trainer.models and rf_key in trainer.results and "error" not in trainer.results[rf_key]:
            model = trainer.models[rf_key]
            X_sub = data["X_test"][:min_size]
            y_sub = data["y_test"][:min_size]
            
            try:
                scores = model.predict_score(X_sub)
                score_lists[modality] = scores
                y_common = y_sub
            except Exception as e:
                print(f"  Skipping {modality}: {e}")
    
    if len(score_lists) < 2 or y_common is None:
        print("  Not enough modalities for fusion test")
        return {}
    
    print(f"  Using {len(score_lists)} modalities, {min_size} samples each")
    
    fusion_results = {}
    
    for strategy in ['average', 'weighted_average', 'min', 'max']:
        fusion = MultimodalFusion(strategy=strategy)
        if strategy == 'weighted_average':
            # Weight by individual model AUC
            weights = []
            for mod in score_lists.keys():
                auc = trainer.results.get(f"{mod}_rf", {}).get("auc_roc", 0.5)
                weights.append(auc)
            fusion.set_weights(weights)
        
        fused = fusion.fuse_scores(list(score_lists.values()))
        
        # Evaluate fused scores
        from sklearn.metrics import roc_auc_score, accuracy_score
        threshold = np.median(fused)
        y_pred = (fused >= threshold).astype(int)
        
        try:
            auc = float(roc_auc_score(y_common, fused))
        except ValueError:
            auc = 0.5
        
        acc = float(accuracy_score(y_common, y_pred))
        
        fusion_results[strategy] = {
            "auc_roc": auc,
            "accuracy": acc,
        }
        
        print(f"  {strategy:20s} → AUC: {auc:.4f}, Acc: {acc:.4f}")
    
    return fusion_results


def print_final_comparison(trainer, fusion_results):
    """Print comprehensive comparison table."""
    print("\n" + "=" * 60)
    print("  FINAL MODEL COMPARISON")
    print("=" * 60)
    
    comparison = trainer.compare_models()
    
    print(f"\n  {'Model':<25s} {'Accuracy':>9s} {'AUC-ROC':>9s} {'EER':>8s} {'FAR':>8s} {'FRR':>8s} {'F1':>8s}")
    print("  " + "-" * 77)
    
    for entry in comparison:
        print(f"  {entry['model']:<25s} "
              f"{entry['accuracy']:>8.4f} "
              f"{entry['auc_roc']:>9.4f} "
              f"{entry['eer']:>8.4f} "
              f"{entry['far']:>8.4f} "
              f"{entry['frr']:>8.4f} "
              f"{entry['f1']:>8.4f}")
    
    if fusion_results:
        print(f"\n  {'Fusion Strategy':<25s} {'AUC-ROC':>9s} {'Accuracy':>9s}")
        print("  " + "-" * 45)
        for strategy, metrics in fusion_results.items():
            print(f"  {strategy:<25s} {metrics['auc_roc']:>9.4f} {metrics['accuracy']:>9.4f}")


def main():
    start_time = time.time()
    
    print("=" * 60)
    print("  BIOMETRIC AUTHENTICATION — MODEL TRAINING")
    print("=" * 60)
    
    trainer = ModelTrainer()
    all_test_data = {}
    
    # ---- Keystroke ----
    ks_models, ks_results, ks_X_test, ks_y_test = train_keystroke_models(trainer)
    all_test_data["keystroke"] = {"X_test": ks_X_test, "y_test": ks_y_test}
    
    # ---- Mouse ----
    ms_models, ms_results, ms_X_test, ms_y_test = train_mouse_models(trainer)
    all_test_data["mouse"] = {"X_test": ms_X_test, "y_test": ms_y_test}
    
    # ---- Touch ----
    tc_models, tc_results, tc_X_test, tc_y_test = train_touch_models(trainer)
    all_test_data["touch"] = {"X_test": tc_X_test, "y_test": tc_y_test}
    
    # ---- Multimodal Fusion ----
    fusion_results = test_multimodal_fusion(trainer, all_test_data)
    
    # ---- Final Comparison ----
    print_final_comparison(trainer, fusion_results)
    
    # ---- Save Everything ----
    MODEL_DIR = os.path.join(BASE_DIR, "../../datasets/trained_models")
    trainer.save_all(MODEL_DIR)
    
    # Save fusion results
    with open(os.path.join(MODEL_DIR, "fusion_results.json"), 'w') as f:
        json.dump(fusion_results, f, indent=2)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — {elapsed:.1f} seconds")
    print(f"  Models saved to: {MODEL_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
