"""
Benchmark script for PathSnap Member 2: Model Experimentation & Comparison.
Trains and compares 5 distinct algorithm families using Group-aware Cross-Validation.
Evaluates on the student-testing partition and exports comparison tables, plots, and final model.
"""

from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    accuracy_score,
    roc_auc_score
)


def compute_top1_accuracy(df_candidates: pd.DataFrame, probas: np.ndarray) -> float:
    """
    Computes Observation-level Top-1 Accuracy.
    For each (trip_id, obs_idx), the candidate arc with the highest predicted
    score is selected. Returns 1.0 if that candidate has target == 1, else 0.0.
    """
    temp = df_candidates[['trip_id', 'obs_idx', 'target']].copy()
    temp['score'] = probas
    
    # Identify the highest-scoring candidate per observation
    idx_max = temp.groupby(['trip_id', 'obs_idx'])['score'].idxmax()
    top1_hits = temp.loc[idx_max, 'target']
    return float(top1_hits.mean())


def run_benchmark(
    train_path: str = "data/processed/train/train_features.parquet",
    test_path: str = "data/processed/student_test/train_features.parquet",
    features_json: str = "data/processed/train/feature_columns.json",
    artifacts_dir: str = "artifacts",
    reports_dir: str = "reports/member2"
):
    artifacts_path = Path(artifacts_dir)
    reports_path = Path(reports_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    print("[1/5] Loading datasets and feature specifications...")
    with open(features_json, "r") as f:
        feature_cols = json.load(f)
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Filter training-eligible candidates
    train_df = train_df[train_df['training_eligible'] == True].reset_index(drop=True)
    test_df = test_df[test_df['training_eligible'] == True].reset_index(drop=True)
    
    print(f"Train samples: {len(train_df):,} | Test samples: {len(test_df):,}")
    print(f"Target distribution (train): {dict(train_df['target'].value_counts())}")
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    groups_train = train_df['trip_id'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    # 5 Model Families
    models = {
        "Logistic Regression": {
            "family": "Linear & Probabilistic",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, C=1.0, random_state=42))
            ]),
            "config": "StandardScaler + LogisticRegression(C=1.0, solver='lbfgs')"
        },
        "Distance-Weighted k-NN": {
            "family": "Instance-based",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=-1))
            ]),
            "config": "StandardScaler + KNeighborsClassifier(k=9, weights='distance')",
            "subsample": 60000  # k-NN inference scales O(N), subsample for practical speed
        },
        "Random Forest": {
            "family": "Tree Ensemble",
            "model": RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_split=10,
                random_state=42, n_jobs=-1
            ),
            "config": "RandomForest(n_estimators=100, max_depth=12)"
        },
        "HistGradientBoosting": {
            "family": "Gradient Boosting",
            "model": HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.08, max_leaf_nodes=31,
                random_state=42
            ),
            "config": "HistGradientBoosting(max_iter=150, lr=0.08, max_leaf_nodes=31)"
        },
        "Multi-Layer Perceptron (MLP)": {
            "family": "Neural Network",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(64, 32), max_iter=25,
                    early_stopping=True, random_state=42
                ))
            ]),
            "config": "MLPClassifier(hidden_layer_sizes=(64,32), early_stopping=True)"
        }
    }
    
    results = []
    gkf = GroupKFold(n_splits=5)
    
    # Representative fold for fast cross-validation comparison
    train_fold_idx, val_fold_idx = next(gkf.split(X_train, y_train, groups=groups_train))
    
    print("\n[2/5] Benchmarking 5 model families with Group-Aware Validation...")
    fitted_models = {}
    
    for name, spec in models.items():
        print(f"\n---> Training: {name} ({spec['family']})")
        model = spec["model"]
        
        # Prepare CV fold training data
        subsample = spec.get("subsample", None)
        if subsample and len(train_fold_idx) > subsample:
            rng = np.random.RandomState(42)
            fold_sub_idx = rng.choice(train_fold_idx, size=subsample, replace=False)
            X_tr, y_tr = X_train[fold_sub_idx], y_train[fold_sub_idx]
        else:
            X_tr, y_tr = X_train[train_fold_idx], y_train[train_fold_idx]
            
        X_val, y_val = X_train[val_fold_idx], y_train[val_fold_idx]
        val_df_fold = train_df.iloc[val_fold_idx].copy()
        
        t0 = time.time()
        model.fit(X_tr, y_tr)
        fit_time = time.time() - t0
        
        # Predict on validation fold
        val_preds = model.predict(X_val)
        val_probas = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else val_preds
        
        cv_macro_f1 = f1_score(y_val, val_preds, average="macro")
        cv_top1 = compute_top1_accuracy(val_df_fold, val_probas)
        cv_loss = log_loss(y_val, val_probas)
        
        # Now refit on full training set (or subsample for k-NN) for student-test evaluation
        if subsample and len(X_train) > subsample:
            rng = np.random.RandomState(42)
            full_sub_idx = rng.choice(len(X_train), size=subsample, replace=False)
            model.fit(X_train[full_sub_idx], y_train[full_sub_idx])
        else:
            model.fit(X_train, y_train)
            
        fitted_models[name] = model
        
        # Predict on student_test partition
        t_infer_start = time.time()
        test_preds = model.predict(X_test)
        infer_time = (time.time() - t_infer_start) / (len(X_test) / 1000.0) # ms per 1k observations
        
        test_probas = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else test_preds
        test_macro_f1 = f1_score(y_test, test_preds, average="macro")
        test_top1 = compute_top1_accuracy(test_df, test_probas)
        test_prec = precision_score(y_test, test_preds, zero_division=0)
        test_rec = recall_score(y_test, test_preds, zero_division=0)
        test_loss = log_loss(y_test, test_probas)
        
        print(f"      CV Macro F1: {cv_macro_f1:.4f} | CV Top-1 Acc: {cv_top1:.4f}")
        print(f"      Test Macro F1: {test_macro_f1:.4f} | Test Top-1 Acc: {test_top1:.4f}")
        print(f"      Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | Inference: {infer_time:.2f} ms/1k")
        
        results.append({
            "Model": name,
            "Algorithm Family": spec["family"],
            "Key Configuration": spec["config"],
            "CV Macro F1-score": round(cv_macro_f1, 4),
            "Top-1 Accuracy": round(test_top1, 4),
            "Student-Test Macro F1-score": round(test_macro_f1, 4),
            "Precision": round(test_prec, 4),
            "Recall": round(test_rec, 4),
            "Log Loss": round(test_loss, 4),
            "Training Time (s)": round(fit_time, 2),
            "Inference Time (ms/1k)": round(infer_time, 2)
        })
        
    print("\n[3/5] Generating Consolidated Model Comparison Table...")
    df_results = pd.DataFrame(results)
    csv_path = reports_path / "model_comparison.csv"
    md_path = reports_path / "model_comparison.md"
    json_path = reports_path / "model_accuracy_log.json"
    df_results.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "metric_definition": {
            "cv_macro_f1": "Macro F1 on the representative GroupKFold validation fold",
            "student_test_top1_accuracy": "Observation-level candidate Top-1 accuracy",
            "student_test_macro_f1": "Candidate-level Macro F1 on student_test",
            "log_loss": "Binary candidate probability log loss on student_test",
        },
        "models": results,
    }, indent=2) + "\n")
    
    with open(md_path, "w") as f:
        f.write("# Model Experimentation and Benchmark Results\n\n")
        f.write(df_results.to_markdown(index=False))
        f.write("\n")
        
    print(f"Saved comparison to {csv_path}, {md_path}, and {json_path}")
    print("\n" + df_results.to_string(index=False))
    
    print("\n[4/5] Serializing Champion Model...")
    # Select best model based on Student-Test Top-1 Accuracy and Macro F1
    best_row = df_results.sort_values(by=["Top-1 Accuracy", "Student-Test Macro F1-score"], ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = fitted_models[best_model_name]
    print(f"Selected Champion Model: {best_model_name}")
    
    model_save_path = artifacts_path / "model_final.joblib"
    joblib.dump(best_model, model_save_path)
    print(f"Serialized champion model to {model_save_path}")
    
    with open(artifacts_path / "model_metadata.json", "w") as f:
        json.dump({
            "champion_model": best_model_name,
            "family": best_row["Algorithm Family"],
            "features": feature_cols,
            "cv_macro_f1": best_row["CV Macro F1-score"],
            "test_top1_accuracy": best_row["Top-1 Accuracy"],
            "test_macro_f1": best_row["Student-Test Macro F1-score"],
            "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
        
    print("\n[5/5] Generating Comparison Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Macro F1 Comparison
    models_list = df_results["Model"]
    x = np.arange(len(models_list))
    width = 0.35
    
    axes[0].bar(x - width/2, df_results["CV Macro F1-score"], width, label="CV Macro F1", color="#3b82f6")
    axes[0].bar(x + width/2, df_results["Student-Test Macro F1-score"], width, label="Student-Test Macro F1", color="#10b981")
    axes[0].set_ylabel("Macro F1-Score")
    axes[0].set_title("Cross-Validation vs. Student-Test Macro F1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models_list, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    
    # Plot 2: Observation Top-1 Accuracy
    colors = ["#6366f1", "#8b5cf6", "#ec4899", "#f59e0b", "#14b8a6"]
    axes[1].bar(models_list, df_results["Top-1 Accuracy"], color=colors)
    axes[1].set_ylabel("Top-1 Accuracy")
    axes[1].set_title("Observation-level Top-1 Accuracy on Student-Test")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models_list, rotation=25, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_path = reports_path / "model_comparison_plots.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved benchmark plot to {plot_path}")
    print("\n=== Benchmark Completed Successfully! ===")


if __name__ == "__main__":
    run_benchmark()
