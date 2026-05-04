"""Master data pipeline runner for all biometric preprocessing steps."""

import os
import sys
import time

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

def run_pipeline():
    start = time.time()
    
    print("=" * 60)
    print("  BIOMETRIC AUTHENTICATION - DATA PIPELINE")
    print("=" * 60)
    
    # ========================================
    # 1. KEYSTROKE DYNAMICS
    # ========================================
    print("\n" + "=" * 60)
    print("  MODULE 1: KEYSTROKE DYNAMICS (CMU Format)")
    print("=" * 60)
    
    from generate_keystroke_data import generate_dataset
    from keystroke_preprocessor import KeystrokePreprocessor
    
    KS_DIR = os.path.join(BASE_DIR, "../../datasets/keystroke")
    KS_PROC = os.path.join(KS_DIR, "processed")
    
    csv_path = os.path.join(KS_DIR, "DSL-StrongPasswordData.csv")
    if not os.path.exists(csv_path):
        generate_dataset(output_dir=KS_DIR)
    
    ks_proc = KeystrokePreprocessor(csv_path)
    ks_proc.load_data()
    ks_proc.extract_derived_features()
    ks_proc.compute_user_statistics()
    
    # Create datasets for first 3 users as examples
    for subj in ["s001", "s002", "s003"]:
        ks_proc.create_authentication_dataset(subj, n_impostors=5)
    
    ks_proc.save_processed_data(KS_PROC)
    
    # ========================================
    # 2. MOUSE DYNAMICS
    # ========================================
    print("\n" + "=" * 60)
    print("  MODULE 2: MOUSE DYNAMICS (Balabit Format)")
    print("=" * 60)
    
    from mouse_preprocessor import MouseDataGenerator, MousePreprocessor
    
    MOUSE_DIR = os.path.join(BASE_DIR, "../../datasets/mouse")
    MOUSE_PROC = os.path.join(MOUSE_DIR, "processed")
    
    if not os.path.exists(os.path.join(MOUSE_DIR, "training_files")):
        gen = MouseDataGenerator(n_users=10, sessions_per_user=12)
        gen.generate_dataset(output_dir=MOUSE_DIR)
    
    mouse_proc = MousePreprocessor(MOUSE_DIR)
    train_df, test_df = mouse_proc.process_all_users()
    mouse_proc.save_processed(train_df, test_df, MOUSE_PROC)
    
    # ========================================
    # 3. TOUCH GESTURES
    # ========================================
    print("\n" + "=" * 60)
    print("  MODULE 3: TOUCHSCREEN GESTURES")
    print("=" * 60)
    
    from touch_preprocessor import TouchDataGenerator, TouchPreprocessor
    
    TOUCH_DIR = os.path.join(BASE_DIR, "../../datasets/touch")
    
    gen = TouchDataGenerator(n_users=15, gestures_per_user=200)
    gen.generate_dataset(output_dir=TOUCH_DIR)
    
    proc = TouchPreprocessor(os.path.join(TOUCH_DIR, "touch_gestures.csv"))
    proc.load_and_process()
    
    # ========================================
    # SUMMARY
    # ========================================
    elapsed = time.time() - start
    
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"""
    Keystroke Dynamics:
      - Dataset: CMU-format, 51 subjects × 400 reps
      - Features: 31 raw + 9 derived = 40 total
      - Split: Sessions 1-4 (train) / 5-8 (test)
      
    Mouse Dynamics:
      - Dataset: Balabit-format, 10 users × 12 sessions
      - Features: 35 behavioral features extracted
      - Split: Training (genuine) / Test (genuine + impostor)
      
    Touch Gestures:
      - Dataset: Synthetic, 15 users × 200 gestures
      - Features: 16 gesture features
      - 9 gesture types simulated
      
    Total time: {elapsed:.1f} seconds
    """)
    
    # Print directory structure
    print("  Dataset directory structure:")
    for root, dirs, files in os.walk(os.path.join(BASE_DIR, "../../datasets")):
        level = root.replace(os.path.join(BASE_DIR, "../../datasets"), "").count(os.sep)
        indent = "    " + "  " * level
        dirname = os.path.basename(root)
        if level < 3:
            n_files = len(files)
            print(f"{indent}{dirname}/ ({n_files} files)")
    
    print("\n✓ All data pipelines complete!")
    return True


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
