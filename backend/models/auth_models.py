"""ML authentication models for behavioral biometric data."""

import numpy as np
import json
import os
import pickle
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler


class BaseAuthModel:
    """Base class for all authentication models."""
    
    def __init__(self, name, model_type):
        self.name = name
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.training_metrics = {}
    
    def train(self, X_train, y_train=None):
        raise NotImplementedError
    
    def predict(self, X_test):
        raise NotImplementedError
    
    def predict_score(self, X_test):
        """Return continuous authentication score (higher = more likely genuine)."""
        raise NotImplementedError
    
    def evaluate(self, X_test, y_test):
        """Compute comprehensive security metrics."""
        y_pred = self.predict(X_test)
        scores = self.predict_score(X_test)
        
        metrics = {
            "model": self.name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        
        # ROC AUC (needs continuous scores)
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_test, scores))
        except ValueError:
            metrics["auc_roc"] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["FAR"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            metrics["FRR"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            metrics["true_positive"] = int(tp)
            metrics["true_negative"] = int(tn)
            metrics["false_positive"] = int(fp)
            metrics["false_negative"] = int(fn)
        
        # EER (Equal Error Rate) computation
        try:
            fpr, tpr, thresholds = roc_curve(y_test, scores)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.abs(fpr - fnr))
            metrics["EER"] = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
            metrics["EER_threshold"] = float(thresholds[eer_idx])
        except (ValueError, IndexError):
            metrics["EER"] = 1.0
            metrics["EER_threshold"] = 0.5
        
        return metrics
    
    def get_roc_data(self, X_test, y_test):
        """Get ROC curve data points for visualization."""
        scores = self.predict_score(X_test)
        try:
            fpr, tpr, thresholds = roc_curve(y_test, scores)
            return {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
                "auc": float(roc_auc_score(y_test, scores))
            }
        except ValueError:
            return {"fpr": [0, 1], "tpr": [0, 1], "thresholds": [1, 0], "auc": 0.5}
    
    def save(self, path):
        """Save trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        """Load trained model."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class OneClassSVMAuth(BaseAuthModel):
    """One-Class SVM for anomaly-based authentication."""
    
    def __init__(self, kernel='rbf', nu=0.1, gamma='scale'):
        super().__init__("One-Class SVM", "anomaly")
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.nu = nu
    
    def train(self, X_train, y_train=None):
        """Train on genuine samples only."""
        if y_train is not None:
            X_genuine = X_train[y_train == 1]
        else:
            X_genuine = X_train
        
        self.model.fit(X_genuine)
        self.is_trained = True
        
        train_pred = self.model.predict(X_genuine)
        self.training_metrics = {
            "n_genuine_train": len(X_genuine),
            "train_inlier_rate": float(np.mean(train_pred == 1)),
        }
        return self.training_metrics
    
    def predict(self, X_test):
        """Returns 1 (genuine) or 0 (impostor)."""
        raw = self.model.predict(X_test)
        return (raw == 1).astype(int)
    
    def predict_score(self, X_test):
        """Decision function score (higher = more likely genuine)."""
        return self.model.decision_function(X_test)


class RandomForestAuth(BaseAuthModel):
    """Random Forest binary classifier for authentication."""
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        super().__init__("Random Forest", "binary")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'
        )
    
    def train(self, X_train, y_train):
        """Train on labeled genuine + impostor data."""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_pred = self.model.predict(X_train)
        self.training_metrics = {
            "n_train": len(X_train),
            "n_genuine": int(sum(y_train)),
            "n_impostor": int(len(y_train) - sum(y_train)),
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "n_estimators": self.model.n_estimators,
        }
        return self.training_metrics
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_score(self, X_test):
        """Probability of being genuine."""
        proba = self.model.predict_proba(X_test)
        if proba.shape[1] == 1:
            # Only one class seen during training
            return proba[:, 0] if self.model.classes_[0] == 1 else 1 - proba[:, 0]
        return proba[:, 1]
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance ranking."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        result = []
        for i in indices[:20]:  # Top 20
            name = feature_names[i] if feature_names else f"feature_{i}"
            result.append({
                "feature": name,
                "importance": float(importances[i])
            })
        return result


class MLPAuth(BaseAuthModel):
    """MLP neural network classifier for authentication."""
    
    def __init__(self, hidden_layers=(128, 64, 32), max_iter=500, random_state=42):
        super().__init__("MLP Neural Network", "binary")
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=random_state,
            verbose=False
        )
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        self.training_metrics = {
            "n_train": len(X_train),
            "n_layers": len(self.model.hidden_layer_sizes),
            "hidden_sizes": list(self.model.hidden_layer_sizes),
            "n_iterations": self.model.n_iter_,
            "final_loss": float(self.model.loss_),
            "train_accuracy": float(self.model.score(X_train, y_train)),
        }
        return self.training_metrics
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_score(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]
    
    def get_loss_curve(self):
        """Get training loss curve for visualization."""
        if hasattr(self.model, 'loss_curve_'):
            return self.model.loss_curve_
        return []


class AutoencoderAuth(BaseAuthModel):
    """Autoencoder for reconstruction-based anomaly detection."""
    
    def __init__(self, encoding_dim=8, threshold_percentile=95, random_state=42):
        super().__init__("Autoencoder", "anomaly")
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.random_state = random_state
        self.model = None
    
    def _build_model(self, input_dim):
        """Build autoencoder architecture."""
        
        hidden_sizes = (64, 32, self.encoding_dim, 32, 64)
        
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_sizes,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=self.random_state,
            verbose=False
        )
    
    def _reconstruction_error(self, X):
        """Compute per-sample reconstruction error (MSE)."""
        X_reconstructed = self.model.predict(X)
        errors = np.mean((X - X_reconstructed) ** 2, axis=1)
        return errors
    
    def train(self, X_train, y_train=None):
        """Train autoencoder on genuine samples."""
        if y_train is not None:
            X_genuine = X_train[y_train == 1]
        else:
            X_genuine = X_train
        
        input_dim = X_genuine.shape[1]
        self._build_model(input_dim)
        
        self.model.fit(X_genuine, X_genuine)
        train_errors = self._reconstruction_error(X_genuine)
        self.threshold = np.percentile(train_errors, self.threshold_percentile)
        
        self.is_trained = True
        self.training_metrics = {
            "n_genuine_train": len(X_genuine),
            "input_dim": input_dim,
            "encoding_dim": self.encoding_dim,
            "threshold": float(self.threshold),
            "mean_train_error": float(np.mean(train_errors)),
            "std_train_error": float(np.std(train_errors)),
            "n_iterations": self.model.n_iter_,
        }
        return self.training_metrics
    
    def predict(self, X_test):
        """Classify as genuine (1) or impostor (0) based on reconstruction error."""
        errors = self._reconstruction_error(X_test)
        return (errors <= self.threshold).astype(int)
    
    def predict_score(self, X_test):
        """Return negative reconstruction error (higher = more genuine)."""
        errors = self._reconstruction_error(X_test)
        return -errors
    
    def get_error_distribution(self, X_genuine, X_impostor):
        """Get error distributions for visualization."""
        genuine_errors = self._reconstruction_error(X_genuine)
        impostor_errors = self._reconstruction_error(X_impostor)
        return {
            "genuine_errors": genuine_errors.tolist(),
            "impostor_errors": impostor_errors.tolist(),
            "threshold": float(self.threshold),
            "genuine_mean": float(np.mean(genuine_errors)),
            "impostor_mean": float(np.mean(impostor_errors)),
        }


class MultimodalFusion:
    """Combines scores from multiple biometric modalities."""
    
    def __init__(self, strategy='weighted_average'):
        self.strategy = strategy
        self.weights = None
        self.meta_model = None
    
    def set_weights(self, weights):
        """Set manual weights for weighted average fusion."""
        self.weights = np.array(weights)
        self.weights = self.weights / self.weights.sum()
    
    def fuse_scores(self, score_list):
        """Fuse scores from multiple models/modalities."""
        scores = np.column_stack(score_list)
        
        if self.strategy == 'average':
            return np.mean(scores, axis=1)
        
        elif self.strategy == 'weighted_average':
            if self.weights is None:
                self.weights = np.ones(len(score_list)) / len(score_list)
            return np.average(scores, weights=self.weights, axis=1)
        
        elif self.strategy == 'max':
            return np.max(scores, axis=1)
        
        elif self.strategy == 'min':

            return np.min(scores, axis=1)
        
        else:
            return np.mean(scores, axis=1)
    
    def fuse_decisions(self, prediction_list):
        """Majority vote fusion of binary predictions."""
        preds = np.column_stack(prediction_list)
        return (np.mean(preds, axis=1) >= 0.5).astype(int)
    
    def train_meta_classifier(self, score_list, y_true):
        """Train a learned fusion model (stacking)."""
        scores = np.column_stack(score_list)
        self.meta_model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.meta_model.fit(scores, y_true)
        self.strategy = 'learned'
    
    def predict_learned(self, score_list):
        """Predict using the learned meta-classifier."""
        if self.meta_model is None:
            raise ValueError("Meta-classifier not trained")
        scores = np.column_stack(score_list)
        return self.meta_model.predict(scores)


class ModelTrainer:
    """Orchestrates training and evaluation of all models across all biometric modalities."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_all_models(self, X_train, X_test, y_train, y_test, modality_name="keystroke"):
        """Train all 4 model types on a single modality."""
        
        print(f"\n{'='*50}")
        print(f"  Training models for: {modality_name}")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"{'='*50}")
        
        models = {
            f"{modality_name}_ocsvm": OneClassSVMAuth(kernel='rbf', nu=0.1),
            f"{modality_name}_rf": RandomForestAuth(n_estimators=100, max_depth=10),
            f"{modality_name}_mlp": MLPAuth(hidden_layers=(128, 64, 32)),
            f"{modality_name}_ae": AutoencoderAuth(encoding_dim=8),
        }
        
        results = {}
        
        for model_key, model in models.items():
            print(f"\n--- {model.name} ---")
            
            try:
                # Train
                train_info = model.train(X_train, y_train)
                print(f"  Training: {train_info}")
                
                # Evaluate
                metrics = model.evaluate(X_test, y_test)
                results[model_key] = metrics
                
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  AUC-ROC:  {metrics.get('auc_roc', 0):.4f}")
                print(f"  EER:      {metrics.get('EER', 1):.4f}")
                print(f"  FAR:      {metrics.get('FAR', 0):.4f}")
                print(f"  FRR:      {metrics.get('FRR', 0):.4f}")
                print(f"  F1:       {metrics['f1']:.4f}")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                results[model_key] = {"error": str(e)}
        
        self.models.update(models)
        self.results.update(results)
        
        return models, results
    
    def compare_models(self):
        """Generate comparison table of all trained models."""
        comparison = []
        for key, metrics in self.results.items():
            if "error" not in metrics:
                comparison.append({
                    "model": key,
                    "accuracy": metrics.get("accuracy", 0),
                    "auc_roc": metrics.get("auc_roc", 0),
                    "eer": metrics.get("EER", 1),
                    "far": metrics.get("FAR", 0),
                    "frr": metrics.get("FRR", 0),
                    "f1": metrics.get("f1", 0),
                })
        
        return sorted(comparison, key=lambda x: x["auc_roc"], reverse=True)
    
    def save_all(self, output_dir):
        """Save all models and results."""
        os.makedirs(output_dir, exist_ok=True)
        
        for key, model in self.models.items():
            model.save(os.path.join(output_dir, f"{key}.pkl"))
        
        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            clean_results = {}
            for k, v in self.results.items():
                clean_results[k] = {
                    mk: float(mv) if isinstance(mv, (np.floating, float)) else mv
                    for mk, mv in v.items()
                }
            json.dump(clean_results, f, indent=2, default=str)
        
        print(f"\n✓ All models and results saved to {output_dir}")
