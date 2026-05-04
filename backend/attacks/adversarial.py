"""Adversarial attack and defense implementations for biometric models."""

import numpy as np
import json
import os
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from copy import deepcopy


class GradientEstimator:
    """Estimate gradients for sklearn models using finite differences."""
    
    @staticmethod
    def numerical_gradient(model_fn, X, epsilon=1e-4):
        """Compute numerical gradient using central finite differences."""
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
        """Zeroth-Order Optimization gradient estimation via random direction sampling."""
        n_samples, n_features = X.shape
        if n_queries is None:
            n_queries = min(n_features, 20)
        
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


class FGSMAttack:
    """Fast Gradient Sign Method (FGSM) adversarial attack."""
    
    def __init__(self, epsilon=0.1, targeted=True):
        self.epsilon = epsilon
        self.targeted = targeted
        self.name = "FGSM"
    
    def attack(self, model_score_fn, X_impostor, gradient_method='numerical'):
        """Generate adversarial examples from impostor samples."""
        # Estimate gradient
        if gradient_method == 'zoo':
            gradients = GradientEstimator.zoo_gradient(model_score_fn, X_impostor)
        else:
            gradients = GradientEstimator.numerical_gradient(model_score_fn, X_impostor)
        
        if self.targeted:
            perturbation = self.epsilon * np.sign(gradients)
        else:
            perturbation = -self.epsilon * np.sign(gradients)
        
        X_adversarial = X_impostor + perturbation
        
        return X_adversarial, perturbation
    
    def evaluate_attack(self, model_predict_fn, model_score_fn, X_impostor, X_adversarial):
        """Evaluate attack effectiveness."""
        orig_pred = model_predict_fn(X_impostor)
        orig_scores = model_score_fn(X_impostor)
        adv_pred = model_predict_fn(X_adversarial)
        adv_scores = model_score_fn(X_adversarial)
        
        n_originally_rejected = np.sum(orig_pred == 0)
        n_now_accepted = np.sum((orig_pred == 0) & (adv_pred == 1))
        
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


class PGDAttack:
    """Projected Gradient Descent (PGD) iterative adversarial attack."""
    
    def __init__(self, epsilon=0.1, alpha=0.01, n_iterations=40, random_start=True):
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.random_start = random_start
        self.name = f"PGD-{n_iterations}"
        self.attack_history = []
    
    def attack(self, model_score_fn, X_impostor, gradient_method='numerical'):
        """Generate adversarial examples using iterative PGD."""
        X_orig = X_impostor.copy()
        
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
            
            X_adv = X_adv + self.alpha * np.sign(gradients)
            perturbation = np.clip(X_adv - X_orig, -self.epsilon, self.epsilon)
            X_adv = X_orig + perturbation
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


class StatisticalMimicryAttack:
    """Statistical mimicry attack that generates samples matching the target user's profile."""
    
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level
        self.name = f"Mimicry-{noise_level}"
    
    def attack(self, X_genuine_reference, n_samples=50):
        """Generate synthetic samples mimicking the target user's statistical profile."""
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


class AdversarialDefense:
    """Collection of defense mechanisms against adversarial attacks."""
    
    @staticmethod
    def adversarial_training(model_class, model_params, X_train, y_train,
                             attack_fn, n_augment_ratio=0.3):
        """Augment training data with adversarial examples and retrain the model."""
        genuine_mask = y_train == 1
        X_genuine = X_train[genuine_mask]
        n_adv = int(len(X_genuine) * n_augment_ratio)
        X_adv_subset = X_genuine[:n_adv]
        X_adv, _ = attack_fn(X_adv_subset)
        X_augmented = np.vstack([X_train, X_adv])
        y_augmented = np.concatenate([y_train, np.zeros(len(X_adv))])
        model = model_class(**model_params)
        model.fit(X_augmented, y_augmented)
        
        return model, {
            "n_original": len(X_train),
            "n_adversarial_added": len(X_adv),
            "total_training_samples": len(X_augmented),
        }
    
    @staticmethod
    def input_smoothing(X, sigma=0.5):
        """Apply input smoothing to reduce adversarial perturbations."""
        noise = np.random.normal(0, sigma, X.shape)
        X_smoothed = X + noise
        X_smoothed = (X + X_smoothed) / 2
        return X_smoothed
    
    @staticmethod
    def feature_squeezing(X, n_bits=4):
        """Reduce feature precision to eliminate small adversarial perturbations."""
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_range = X_max - X_min + 1e-8
        X_norm = (X - X_min) / X_range
        n_levels = 2 ** n_bits
        X_quantized = np.round(X_norm * n_levels) / n_levels
        X_squeezed = X_quantized * X_range + X_min
        
        return X_squeezed
    
    @staticmethod
    def anomaly_threshold_adjustment(base_scores, genuine_scores,
                                      confidence=0.99):
        """Compute an adaptive threshold from the genuine score distribution."""
        mean_score = np.mean(genuine_scores)
        std_score = np.std(genuine_scores)
        from scipy.stats import norm
        z = norm.ppf(1 - (1 - confidence) / 2)
        strict_threshold = mean_score - z * std_score
        adjusted_predictions = (base_scores >= strict_threshold).astype(int)
        
        return adjusted_predictions, strict_threshold
    
    @staticmethod
    def ensemble_defense(models, X, strategy='unanimous'):
        """Combine predictions from multiple models using the given strategy."""
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


class AttackOrchestrator:
    """Orchestrates attacks and defenses across models."""
    
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
