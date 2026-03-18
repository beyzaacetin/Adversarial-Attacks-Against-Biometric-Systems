"""
Adversarial Attacks Against Biometric Authentication
=====================================================
Implements attacks that fool trained biometric models:

ATTACKS:
1. FGSM (Fast Gradient Sign Method)
   - Single-step gradient-based perturbation
   - Fast but less effective
   
2. PGD (Projected Gradient Descent)
   - Iterative FGSM with projection back to epsilon-ball
   - Stronger attack, gold standard for robustness evaluation
   
3. Statistical Mimicry Attack
   - Crafts samples matching target user's statistical profile
   - More realistic: simulates an attacker who observed the target

4. Noise Injection Attack
   - Adds calibrated random noise to impostor samples
   - Tests model robustness to input perturbation

DEFENSES:
1. Adversarial Training - Retrain with adversarial examples
2. Input Smoothing - Gaussian smoothing of input features
3. Feature Squeezing - Reduce feature precision
4. Anomaly-Aware Thresholding - Dynamic threshold adjustment
5. Ensemble Defense - Combine multiple model decisions

Security Metrics:
- Attack Success Rate (ASR): % of adversarial samples that fool the model
- Perturbation Budget (ε): Maximum allowed distortion
- Robustness Score: 1 - ASR under strongest attack
"""

import numpy as np
import json
import os
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from copy import deepcopy


# ============================================================
# GRADIENT ESTIMATION FOR SKLEARN MODELS
# ============================================================

class GradientEstimator:
    """
    Estimate gradients for sklearn models that don't expose them.
    
    Since sklearn's RF and SVM don't provide gradients natively,
    we use finite differences (numerical gradient estimation).
    For MLP, we can also use the internal structure.
    
    This is a key concept: adversarial attacks fundamentally need
    gradient information. When the model is a black box, we must
    estimate gradients — this is the "black-box attack" scenario.
    """
    
    @staticmethod
    def numerical_gradient(model_fn, X, epsilon=1e-4):
        """
        Compute numerical gradient using central finite differences.
        
        ∂f/∂x_i ≈ (f(x + εe_i) - f(x - εe_i)) / (2ε)
        
        Args:
            model_fn: Function that returns scores (higher = more genuine)
            X: Input samples (n_samples, n_features)
            epsilon: Step size for finite differences
        Returns:
            gradients: (n_samples, n_features) gradient matrix
        """
        n_samples, n_features = X.shape
        gradients = np.zeros_like(X)
        
        for i in range(n_features):
            X_plus = X.copy()
            X_minus = X.copy()
            X_plus[:, i] += epsilon
            X_minus[:, i] -= epsilon
            
            score_plus = model_fn(X_plus)
            score_minus = model_fn(X_minus)
            
            gradients[:, i] = (score_plus - score_minus) / (2 * epsilon)
        
        return gradients
    
    @staticmethod
    def zoo_gradient(model_fn, X, epsilon=1e-3, n_queries=None):
        """
        Zeroth-Order Optimization gradient estimation.
        Uses random direction sampling — more query-efficient for high-dim.
        
        This simulates a realistic black-box attack where the attacker
        can only query the model API, not access internals.
        """
        n_samples, n_features = X.shape
        if n_queries is None:
            n_queries = min(n_features, 20)  # Budget limit
        
        gradients = np.zeros_like(X)
        
        for _ in range(n_queries):
            # Random direction
            direction = np.random.randn(n_samples, n_features)
            direction = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8)
            
            score_plus = model_fn(X + epsilon * direction)
            score_minus = model_fn(X - epsilon * direction)
            
            # Project gradient onto random direction
            grad_estimate = ((score_plus - score_minus) / (2 * epsilon))[:, np.newaxis]
            gradients += grad_estimate * direction
        
        gradients /= n_queries
        return gradients


# ============================================================
# ATTACK 1: FGSM (Fast Gradient Sign Method)
# ============================================================

class FGSMAttack:
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014)
    
    x_adv = x + ε · sign(∇_x L(x, y))
    
    The attacker adds a small perturbation in the direction that
    maximizes the model's loss. The perturbation is bounded by ε.
    
    In biometric context: An impostor's typing/mouse data is slightly
    modified to look more like the target user.
    
    Threat model: White-box (attacker knows model) or
                  Black-box (using gradient estimation)
    """
    
    def __init__(self, epsilon=0.1, targeted=True):
        """
        Args:
            epsilon: Maximum perturbation magnitude (L∞ norm)
            targeted: If True, push toward genuine class (evasion)
                     If False, push away from genuine class (DoS)
        """
        self.epsilon = epsilon
        self.targeted = targeted
        self.name = "FGSM"
    
    def attack(self, model_score_fn, X_impostor, gradient_method='numerical'):
        """
        Generate adversarial examples from impostor samples.
        
        Args:
            model_score_fn: Function returning genuine probability scores
            X_impostor: Impostor samples to perturb
            gradient_method: 'numerical' or 'zoo'
        
        Returns:
            X_adversarial: Perturbed samples
            perturbation: The added noise
        """
        # Estimate gradient
        if gradient_method == 'zoo':
            gradients = GradientEstimator.zoo_gradient(model_score_fn, X_impostor)
        else:
            gradients = GradientEstimator.numerical_gradient(model_score_fn, X_impostor)
        
        # FGSM: perturb in direction of gradient sign
        if self.targeted:
            # Move TOWARD genuine class (increase score)
            perturbation = self.epsilon * np.sign(gradients)
        else:
            # Move AWAY from genuine class (decrease score)  
            perturbation = -self.epsilon * np.sign(gradients)
        
        X_adversarial = X_impostor + perturbation
        
        return X_adversarial, perturbation
    
    def evaluate_attack(self, model_predict_fn, model_score_fn, X_impostor, X_adversarial):
        """Evaluate attack effectiveness."""
        # Original predictions on impostor data
        orig_pred = model_predict_fn(X_impostor)
        orig_scores = model_score_fn(X_impostor)
        
        # Predictions on adversarial data
        adv_pred = model_predict_fn(X_adversarial)
        adv_scores = model_score_fn(X_adversarial)
        
        # Attack Success Rate: % of impostors now classified as genuine
        n_originally_rejected = np.sum(orig_pred == 0)  # Correctly rejected
        n_now_accepted = np.sum((orig_pred == 0) & (adv_pred == 1))  # Now fooled
        
        asr = n_now_accepted / max(n_originally_rejected, 1)
        
        # Average score increase
        score_increase = np.mean(adv_scores - orig_scores)
        
        return {
            "attack": self.name,
            "epsilon": self.epsilon,
            "attack_success_rate": float(asr),
            "n_samples": len(X_impostor),
            "n_originally_rejected": int(n_originally_rejected),
            "n_now_fooled": int(n_now_accepted),
            "avg_score_before": float(np.mean(orig_scores)),
            "avg_score_after": float(np.mean(adv_scores)),
            "avg_score_increase": float(score_increase),
            "avg_perturbation_l2": float(np.mean(np.linalg.norm(
                X_adversarial - X_impostor, axis=1))),
            "max_perturbation_linf": float(np.max(np.abs(
                X_adversarial - X_impostor))),
        }


# ============================================================
# ATTACK 2: PGD (Projected Gradient Descent)
# ============================================================

class PGDAttack:
    """
    Projected Gradient Descent (Madry et al., 2018)
    
    Iterative version of FGSM:
    x^(t+1) = Π_{x+S}( x^(t) + α · sign(∇_x L(x^(t), y)) )
    
    Where Π projects back onto the ε-ball around original x.
    
    Stronger than FGSM because it takes multiple smaller steps,
    finding better adversarial examples within the budget.
    
    In biometric context: The attacker iteratively refines the
    perturbation, getting closer to fooling the model each step.
    """
    
    def __init__(self, epsilon=0.1, alpha=0.01, n_iterations=40, random_start=True):
        """
        Args:
            epsilon: Maximum perturbation (L∞ ball radius)
            alpha: Step size per iteration
            n_iterations: Number of PGD steps
            random_start: Start from random point in ε-ball
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.random_start = random_start
        self.name = f"PGD-{n_iterations}"
        self.attack_history = []
    
    def attack(self, model_score_fn, X_impostor, gradient_method='numerical'):
        """
        Generate adversarial examples using iterative PGD.
        
        Returns:
            X_adversarial: Best adversarial examples found
            perturbation: Total perturbation applied
        """
        X_orig = X_impostor.copy()
        
        # Random initialization within ε-ball
        if self.random_start:
            X_adv = X_orig + np.random.uniform(
                -self.epsilon, self.epsilon, X_orig.shape
            )
        else:
            X_adv = X_orig.copy()
        
        best_X_adv = X_adv.copy()
        best_scores = model_score_fn(X_adv)
        
        self.attack_history = []
        
        for step in range(self.n_iterations):
            # Estimate gradient
            if gradient_method == 'zoo':
                gradients = GradientEstimator.zoo_gradient(model_score_fn, X_adv)
            else:
                gradients = GradientEstimator.numerical_gradient(model_score_fn, X_adv)
            
            # Gradient ascent step (move toward genuine class)
            X_adv = X_adv + self.alpha * np.sign(gradients)
            
            # Project back onto ε-ball (L∞ projection)
            perturbation = np.clip(X_adv - X_orig, -self.epsilon, self.epsilon)
            X_adv = X_orig + perturbation
            
            # Track best adversarial examples
            current_scores = model_score_fn(X_adv)
            improved = current_scores > best_scores
            best_X_adv[improved] = X_adv[improved]
            best_scores = np.maximum(best_scores, current_scores)
            
            # Log progress
            self.attack_history.append({
                "step": step,
                "avg_score": float(np.mean(current_scores)),
                "max_score": float(np.max(current_scores)),
                "avg_perturbation": float(np.mean(np.abs(perturbation))),
            })
        
        total_perturbation = best_X_adv - X_orig
        return best_X_adv, total_perturbation
    
    def evaluate_attack(self, model_predict_fn, model_score_fn, X_impostor, X_adversarial):
        """Evaluate attack using same interface as FGSM."""
        fgsm_eval = FGSMAttack(self.epsilon)
        result = fgsm_eval.evaluate_attack(
            model_predict_fn, model_score_fn, X_impostor, X_adversarial
        )
        result["attack"] = self.name
        result["n_iterations"] = self.n_iterations
        result["alpha"] = self.alpha
        result["attack_trajectory"] = self.attack_history
        return result


# ============================================================
# ATTACK 3: STATISTICAL MIMICRY
# ============================================================

class StatisticalMimicryAttack:
    """
    Statistical Mimicry Attack
    
    A more realistic attack where the adversary has observed
    the target user's behavioral statistics (e.g., through
    shoulder surfing, data breach, or side-channel).
    
    The attacker generates synthetic samples that match the
    target user's mean and standard deviation for each feature.
    
    Threat model: The attacker knows the target's typing/mouse
    statistics but cannot perfectly replicate them.
    """
    
    def __init__(self, noise_level=0.1):
        """
        Args:
            noise_level: How much random variation to add (0=perfect copy)
        """
        self.noise_level = noise_level
        self.name = f"Mimicry-{noise_level}"
    
    def attack(self, X_genuine_reference, n_samples=50):
        """
        Generate synthetic samples mimicking the target user.
        
        Args:
            X_genuine_reference: Genuine user's enrollment data
            n_samples: Number of fake samples to generate
        
        Returns:
            X_mimicry: Synthetic impostor samples
        """
        mean = np.mean(X_genuine_reference, axis=0)
        std = np.std(X_genuine_reference, axis=0)
        
        # Generate from fitted distribution with noise
        X_mimicry = np.random.normal(
            loc=mean,
            scale=std * (1 + self.noise_level),
            size=(n_samples, X_genuine_reference.shape[1])
        )
        
        return X_mimicry
    
    def evaluate_attack(self, model_predict_fn, model_score_fn, X_mimicry):
        """Evaluate how many mimicry samples are accepted."""
        predictions = model_predict_fn(X_mimicry)
        scores = model_score_fn(X_mimicry)
        
        accepted = np.sum(predictions == 1)
        
        return {
            "attack": self.name,
            "n_samples": len(X_mimicry),
            "n_accepted": int(accepted),
            "attack_success_rate": float(accepted / len(X_mimicry)),
            "avg_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "noise_level": self.noise_level,
        }


# ============================================================
# ATTACK 4: NOISE INJECTION
# ============================================================

class NoiseInjectionAttack:
    """
    Simple noise injection to test model robustness.
    Tests if random perturbation can flip decisions.
    """
    
    def __init__(self, noise_std=0.5):
        self.noise_std = noise_std
        self.name = f"Noise-{noise_std}"
    
    def attack(self, X_impostor):
        noise = np.random.normal(0, self.noise_std, X_impostor.shape)
        return X_impostor + noise, noise
    
    def evaluate_attack(self, model_predict_fn, model_score_fn, X_impostor, X_noisy):
        orig_pred = model_predict_fn(X_impostor)
        noisy_pred = model_predict_fn(X_noisy)
        
        n_flipped = np.sum(orig_pred != noisy_pred)
        n_now_accepted = np.sum((orig_pred == 0) & (noisy_pred == 1))
        
        return {
            "attack": self.name,
            "noise_std": self.noise_std,
            "n_flipped": int(n_flipped),
            "flip_rate": float(n_flipped / len(X_impostor)),
            "n_now_accepted": int(n_now_accepted),
            "attack_success_rate": float(n_now_accepted / max(np.sum(orig_pred == 0), 1)),
        }


# ============================================================
# DEFENSE MECHANISMS
# ============================================================

class AdversarialDefense:
    """Collection of defense mechanisms against adversarial attacks."""
    
    @staticmethod
    def adversarial_training(model_class, model_params, X_train, y_train,
                             attack_fn, n_augment_ratio=0.3):
        """
        Defense 1: Adversarial Training
        
        Augment training data with adversarial examples and retrain.
        The model learns to be robust against known attack patterns.
        
        Args:
            model_class: sklearn model class
            model_params: Model constructor parameters
            X_train, y_train: Original training data
            attack_fn: Function that generates adversarial samples
            n_augment_ratio: Fraction of adversarial samples to add
        
        Returns:
            Retrained model
        """
        # Generate adversarial versions of genuine training samples
        genuine_mask = y_train == 1
        X_genuine = X_train[genuine_mask]
        
        n_adv = int(len(X_genuine) * n_augment_ratio)
        X_adv_subset = X_genuine[:n_adv]
        
        # Apply attack to create adversarial examples
        X_adv, _ = attack_fn(X_adv_subset)
        
        # Augmented training set: original + adversarial (labeled as impostor)
        X_augmented = np.vstack([X_train, X_adv])
        y_augmented = np.concatenate([y_train, np.zeros(len(X_adv))])
        
        # Retrain model
        model = model_class(**model_params)
        model.fit(X_augmented, y_augmented)
        
        return model, {
            "n_original": len(X_train),
            "n_adversarial_added": len(X_adv),
            "total_training_samples": len(X_augmented),
        }
    
    @staticmethod
    def input_smoothing(X, sigma=0.5):
        """
        Defense 2: Input Smoothing
        
        Apply Gaussian smoothing to input features before classification.
        Removes high-frequency adversarial perturbations while preserving
        the genuine behavioral signal.
        
        In biometric context: Smooth out suspicious micro-variations
        in typing timing or mouse trajectory.
        """
        noise = np.random.normal(0, sigma, X.shape)
        # Average with neighbors (simple moving average per feature)
        # For independent features, this acts as a low-pass filter
        X_smoothed = X + noise
        # Average original and noisy version (denoising)
        X_smoothed = (X + X_smoothed) / 2
        return X_smoothed
    
    @staticmethod
    def feature_squeezing(X, n_bits=4):
        """
        Defense 3: Feature Squeezing
        
        Reduce the precision of input features.
        Adversarial perturbations often exploit fine-grained precision.
        
        Quantize features to fewer bits, eliminating small perturbations.
        """
        # Compute range per feature
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_range = X_max - X_min + 1e-8
        
        # Normalize to [0, 1]
        X_norm = (X - X_min) / X_range
        
        # Quantize
        n_levels = 2 ** n_bits
        X_quantized = np.round(X_norm * n_levels) / n_levels
        
        # Restore original scale
        X_squeezed = X_quantized * X_range + X_min
        
        return X_squeezed
    
    @staticmethod
    def anomaly_threshold_adjustment(base_scores, genuine_scores, 
                                      confidence=0.99):
        """
        Defense 4: Anomaly-Aware Dynamic Thresholding
        
        Instead of fixed threshold, use adaptive threshold based on
        the distribution of genuine scores during enrollment.
        
        Tighter threshold = more false rejections but harder to fool.
        """
        mean_score = np.mean(genuine_scores)
        std_score = np.std(genuine_scores)
        
        # Set threshold at confidence interval boundary
        from scipy.stats import norm
        z = norm.ppf(1 - (1 - confidence) / 2)
        strict_threshold = mean_score - z * std_score
        
        # Apply to base scores
        adjusted_predictions = (base_scores >= strict_threshold).astype(int)
        
        return adjusted_predictions, strict_threshold
    
    @staticmethod
    def ensemble_defense(models, X, strategy='unanimous'):
        """
        Defense 5: Ensemble Defense
        
        Require agreement from multiple models/modalities.
        An adversarial example that fools one model may not fool others.
        
        Strategies:
        - 'majority': >50% must agree
        - 'unanimous': ALL must agree (strictest)
        - 'any': At least one accepts (most permissive)
        """
        predictions = []
        for model in models:
            pred = model.predict(X)
            predictions.append(pred)
        
        preds = np.column_stack(predictions)
        
        if strategy == 'unanimous':
            return (preds.min(axis=1) == 1).astype(int)
        elif strategy == 'majority':
            return (preds.mean(axis=1) >= 0.5).astype(int)
        elif strategy == 'any':
            return (preds.max(axis=1) == 1).astype(int)
        else:
            return (preds.mean(axis=1) >= 0.5).astype(int)


# ============================================================
# ATTACK ORCHESTRATOR
# ============================================================

class AttackOrchestrator:
    """
    Orchestrates attacks and defenses, producing comprehensive
    security evaluation results.
    """
    
    def __init__(self):
        self.attack_results = []
        self.defense_results = []
    
    def run_full_evaluation(self, model, X_train, X_test, y_train, y_test, 
                            model_name="model"):
        """
        Run all attacks at various epsilon levels against a model,
        then test all defenses.
        """
        print(f"\n{'='*55}")
        print(f"  ADVERSARIAL EVALUATION: {model_name}")
        print(f"{'='*55}")
        
        # Define model interface functions
        def predict_fn(X):
            return model.predict(X)
        
        def score_fn(X):
            return model.predict_score(X)
        
        # Separate genuine and impostor test data
        genuine_mask = y_test == 1
        impostor_mask = y_test == 0
        X_genuine = X_test[genuine_mask]
        X_impostor = X_test[impostor_mask]
        
        if len(X_impostor) == 0:
            print("  No impostor samples in test set, skipping attack evaluation")
            return {}, {}
        
        # ---- BASELINE ----
        print(f"\n  Baseline (no attack):")
        baseline_pred = predict_fn(X_test)
        baseline_acc = accuracy_score(y_test, baseline_pred)
        print(f"    Accuracy: {baseline_acc:.4f}")
        
        all_attack_results = {"baseline_accuracy": baseline_acc}
        
        # ---- FGSM at various epsilon ----
        print(f"\n  --- FGSM Attack ---")
        epsilons = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        fgsm_results = []
        
        for eps in epsilons:
            attack = FGSMAttack(epsilon=eps)
            X_adv, _ = attack.attack(score_fn, X_impostor)
            result = attack.evaluate_attack(predict_fn, score_fn, X_impostor, X_adv)
            fgsm_results.append(result)
            print(f"    ε={eps:.2f} → ASR={result['attack_success_rate']:.3f}, "
                  f"Score: {result['avg_score_before']:.3f}→{result['avg_score_after']:.3f}")
        
        all_attack_results["fgsm"] = fgsm_results
        
        # ---- PGD Attack ----
        print(f"\n  --- PGD Attack ---")
        pgd_results = []
        
        for eps in [0.05, 0.1, 0.2, 0.5]:
            attack = PGDAttack(epsilon=eps, alpha=eps/4, n_iterations=20)
            X_adv, _ = attack.attack(score_fn, X_impostor)
            result = attack.evaluate_attack(predict_fn, score_fn, X_impostor, X_adv)
            pgd_results.append(result)
            print(f"    ε={eps:.2f}, steps=20 → ASR={result['attack_success_rate']:.3f}")
        
        all_attack_results["pgd"] = pgd_results
        
        # ---- Statistical Mimicry ----
        print(f"\n  --- Statistical Mimicry Attack ---")
        mimicry_results = []
        X_genuine_train = X_train[y_train == 1]
        
        for noise in [0.0, 0.1, 0.3, 0.5, 1.0]:
            attack = StatisticalMimicryAttack(noise_level=noise)
            X_mimicry = attack.attack(X_genuine_train, n_samples=50)
            result = attack.evaluate_attack(predict_fn, score_fn, X_mimicry)
            mimicry_results.append(result)
            print(f"    noise={noise:.1f} → ASR={result['attack_success_rate']:.3f}, "
                  f"Avg score={result['avg_score']:.3f}")
        
        all_attack_results["mimicry"] = mimicry_results
        
        # ---- Noise Injection ----
        print(f"\n  --- Noise Injection ---")
        noise_results = []
        for std in [0.1, 0.5, 1.0, 2.0]:
            attack = NoiseInjectionAttack(noise_std=std)
            X_noisy, _ = attack.attack(X_impostor)
            result = attack.evaluate_attack(predict_fn, score_fn, X_impostor, X_noisy)
            noise_results.append(result)
            print(f"    σ={std:.1f} → Flip rate={result['flip_rate']:.3f}, "
                  f"ASR={result['attack_success_rate']:.3f}")
        
        all_attack_results["noise"] = noise_results
        
        # ==== DEFENSES ====
        print(f"\n  === DEFENSE EVALUATION ===")
        all_defense_results = {}
        
        # Find the strongest FGSM attack for defense testing
        best_fgsm_eps = max(fgsm_results, key=lambda x: x["attack_success_rate"])["epsilon"]
        fgsm_attack = FGSMAttack(epsilon=best_fgsm_eps)
        X_adv_strong, _ = fgsm_attack.attack(score_fn, X_impostor)
        
        # Defense 1: Input Smoothing
        print(f"\n  Defense: Input Smoothing")
        for sigma in [0.1, 0.3, 0.5]:
            X_adv_smoothed = AdversarialDefense.input_smoothing(X_adv_strong, sigma=sigma)
            smoothed_pred = predict_fn(X_adv_smoothed)
            n_blocked = np.sum(smoothed_pred == 0)
            n_total_adv = len(X_adv_strong)
            defense_rate = n_blocked / n_total_adv
            print(f"    σ={sigma:.1f} → Blocked {n_blocked}/{n_total_adv} = {defense_rate:.3f}")
            all_defense_results[f"smoothing_s{sigma}"] = {
                "defense": "input_smoothing",
                "sigma": sigma,
                "n_blocked": int(n_blocked),
                "defense_rate": float(defense_rate),
            }
        
        # Defense 2: Feature Squeezing
        print(f"\n  Defense: Feature Squeezing")
        for n_bits in [2, 4, 6]:
            X_adv_squeezed = AdversarialDefense.feature_squeezing(X_adv_strong, n_bits=n_bits)
            squeezed_pred = predict_fn(X_adv_squeezed)
            n_blocked = np.sum(squeezed_pred == 0)
            defense_rate = n_blocked / len(X_adv_strong)
            print(f"    {n_bits}-bit → Blocked {n_blocked}/{len(X_adv_strong)} = {defense_rate:.3f}")
            all_defense_results[f"squeezing_{n_bits}bit"] = {
                "defense": "feature_squeezing",
                "n_bits": n_bits,
                "defense_rate": float(defense_rate),
            }
        
        # Defense 3: Adversarial Training
        print(f"\n  Defense: Adversarial Training")
        if hasattr(model, 'model') and hasattr(model.model, 'fit'):
            try:
                def attack_fn(X):
                    fgsm = FGSMAttack(epsilon=best_fgsm_eps)
                    return fgsm.attack(score_fn, X)
                
                # Retrain with adversarial examples
                retrained_model = deepcopy(model)
                
                # For binary classifiers
                if model.model_type == 'binary':
                    X_aug = np.vstack([X_train, X_adv_strong])
                    y_aug = np.concatenate([y_train, np.zeros(len(X_adv_strong))])
                    retrained_model.model.fit(X_aug, y_aug)
                    
                    # Test retrained model against same attack
                    X_adv_retest, _ = fgsm_attack.attack(
                        lambda X: retrained_model.predict_score(X), X_impostor
                    )
                    retrained_pred = retrained_model.predict(X_adv_retest)
                    n_blocked = np.sum(retrained_pred == 0)
                    defense_rate = n_blocked / len(X_adv_retest)
                    print(f"    After retraining → Blocked {n_blocked}/{len(X_adv_retest)} = {defense_rate:.3f}")
                    
                    # Also check clean accuracy didn't degrade too much
                    clean_pred = retrained_model.predict(X_test)
                    clean_acc = accuracy_score(y_test, clean_pred)
                    print(f"    Clean accuracy: {baseline_acc:.4f} → {clean_acc:.4f}")
                    
                    all_defense_results["adversarial_training"] = {
                        "defense": "adversarial_training",
                        "defense_rate": float(defense_rate),
                        "clean_accuracy_before": float(baseline_acc),
                        "clean_accuracy_after": float(clean_acc),
                    }
            except Exception as e:
                print(f"    Failed: {e}")
        
        self.attack_results.append(all_attack_results)
        self.defense_results.append(all_defense_results)
        
        return all_attack_results, all_defense_results
    
    def generate_report(self):
        """Generate summary report of all attacks and defenses."""
        report = {
            "attacks": self.attack_results,
            "defenses": self.defense_results,
        }
        return report
