"""Runs all adversarial attacks against trained biometric models."""

import sys
import os
import json
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "../data"))
sys.path.insert(0, os.path.join(BASE_DIR, "../models"))
sys.path.insert(0, BASE_DIR)

from adversarial import (
    FGSMAttack, PGDAttack, StatisticalMimicryAttack,
    NoiseInjectionAttack, AdversarialDefense, AttackOrchestrator
)
from auth_models import OneClassSVMAuth, RandomForestAuth, MLPAuth, AutoencoderAuth
from keystroke_preprocessor import KeystrokePreprocessor
from touch_preprocessor import TouchPreprocessor, TouchDataGenerator


def run_keystroke_attacks():
    """Run adversarial attacks on keystroke authentication models."""
    print("\n" + "=" * 60)
    print("  ADVERSARIAL ATTACKS ON KEYSTROKE AUTHENTICATION")
    print("=" * 60)
    
    # Load and prepare data
    DATA_DIR = os.path.join(BASE_DIR, "../../datasets/keystroke")
    csv_path = os.path.join(DATA_DIR, "DSL-StrongPasswordData.csv")
    
    if not os.path.exists(csv_path):
        from generate_keystroke_data import generate_dataset
        generate_dataset(output_dir=DATA_DIR)
    
    proc = KeystrokePreprocessor(csv_path)
    proc.load_data()
    proc.extract_derived_features()
    X_train, X_test, y_train, y_test = proc.create_authentication_dataset("s001", n_impostors=10)
    
    orchestrator = AttackOrchestrator()
    all_results = {}
    
    # Test attacks on each model type
    models_to_test = {
        "Random Forest": RandomForestAuth(n_estimators=100, max_depth=10),
        "MLP Neural Network": MLPAuth(hidden_layers=(128, 64, 32)),
        "Autoencoder": AutoencoderAuth(encoding_dim=8),
        "One-Class SVM": OneClassSVMAuth(kernel='rbf', nu=0.1),
    }
    
    for model_name, model in models_to_test.items():
        model.train(X_train, y_train)
        attack_res, defense_res = orchestrator.run_full_evaluation(
            model, X_train, X_test, y_train, y_test,
            model_name=f"Keystroke — {model_name}"
        )
        all_results[f"keystroke_{model_name}"] = {
            "attacks": attack_res,
            "defenses": defense_res,
        }
    
    return all_results, orchestrator


def run_touch_attacks():
    """Run adversarial attacks on touch gesture authentication."""
    print("\n" + "=" * 60)
    print("  ADVERSARIAL ATTACKS ON TOUCH AUTHENTICATION")
    print("=" * 60)
    
    TOUCH_DIR = os.path.join(BASE_DIR, "../../datasets/touch")
    csv_path = os.path.join(TOUCH_DIR, "touch_gestures.csv")
    
    if not os.path.exists(csv_path):
        gen = TouchDataGenerator(n_users=15, gestures_per_user=200)
        gen.generate_dataset(output_dir=TOUCH_DIR)
    
    proc = TouchPreprocessor(csv_path)
    proc.load_and_process()
    X_train, X_test, y_train, y_test = proc.create_auth_dataset("user01", n_impostors=5)
    
    orchestrator = AttackOrchestrator()
    
    # Test on Random Forest (best performing model)
    model = RandomForestAuth(n_estimators=100, max_depth=10)
    model.train(X_train, y_train)
    
    attack_res, defense_res = orchestrator.run_full_evaluation(
        model, X_train, X_test, y_train, y_test,
        model_name="Touch — Random Forest"
    )
    
    return {"touch_rf": {"attacks": attack_res, "defenses": defense_res}}, orchestrator


def print_summary(all_results):
    """Print a summary table of attack effectiveness."""
    print("\n" + "=" * 60)
    print("  ATTACK EFFECTIVENESS SUMMARY")
    print("=" * 60)
    
    print(f"\n  {'Model':<35s} {'FGSM':>8s} {'PGD':>8s} {'Mimicry':>8s} {'Noise':>8s}")
    print(f"  {'':35s} {'(ε=0.2)':>8s} {'(ε=0.2)':>8s} {'(n=0.1)':>8s} {'(σ=0.5)':>8s}")
    print("  " + "-" * 67)
    
    for model_key, data in all_results.items():
        attacks = data.get("attacks", {})
        
        # Get FGSM ASR at epsilon=0.2
        fgsm_asr = 0
        for r in attacks.get("fgsm", []):
            if abs(r.get("epsilon", 0) - 0.2) < 0.01:
                fgsm_asr = r.get("attack_success_rate", 0)
        
        # Get PGD ASR at epsilon=0.2
        pgd_asr = 0
        for r in attacks.get("pgd", []):
            if abs(r.get("epsilon", 0) - 0.2) < 0.01:
                pgd_asr = r.get("attack_success_rate", 0)
        
        # Get Mimicry ASR at noise=0.1
        mimicry_asr = 0
        for r in attacks.get("mimicry", []):
            if abs(r.get("noise_level", 0) - 0.1) < 0.01:
                mimicry_asr = r.get("attack_success_rate", 0)
        
        # Get Noise ASR at std=0.5
        noise_asr = 0
        for r in attacks.get("noise", []):
            if abs(r.get("noise_std", 0) - 0.5) < 0.01:
                noise_asr = r.get("attack_success_rate", 0)
        
        print(f"  {model_key:<35s} "
              f"{fgsm_asr:>7.3f} "
              f"{pgd_asr:>8.3f} "
              f"{mimicry_asr:>8.3f} "
              f"{noise_asr:>8.3f}")
    
    # Defense summary
    print(f"\n  {'Defense Summary':^60s}")
    print("  " + "-" * 60)
    
    for model_key, data in all_results.items():
        defenses = data.get("defenses", {})
        if defenses:
            print(f"\n  {model_key}:")
            for def_key, def_data in defenses.items():
                rate = def_data.get("defense_rate", 0)
                name = def_data.get("defense", def_key)
                print(f"    {name:30s} → {rate:.3f} blocked")


def main():
    start = time.time()
    
    print("=" * 60)
    print("  ADVERSARIAL ATTACK EVALUATION SUITE")
    print("=" * 60)
    
    all_results = {}
    
    # Keystroke attacks
    ks_results, ks_orch = run_keystroke_attacks()
    all_results.update(ks_results)
    
    # Touch attacks
    tc_results, tc_orch = run_touch_attacks()
    all_results.update(tc_results)
    
    # Summary
    print_summary(all_results)
    
    # Save results
    OUTPUT_DIR = os.path.join(BASE_DIR, "../../datasets/attack_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Clean results for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(os.path.join(OUTPUT_DIR, "attack_results.json"), 'w') as f:
        json.dump(clean_for_json(all_results), f, indent=2)
    
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  ATTACK EVALUATION COMPLETE — {elapsed:.1f} seconds")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
