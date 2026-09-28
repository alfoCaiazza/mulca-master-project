"""
CONDITIONING INDEX (CI): CI for conversation i is defined as the posterior 
probability of conditioning given its set of extracted Principal Components (PCs):

    CI_i = P(Y_i = conditioning | PC_{i,1}, ..., PC_{i,k})

Raw classifier outputs / logits are mapped to calibrated probabilities 
via Sigmoid Calibration (Platt Scaling):

    P(Y_i = 1 | s_i) = 1 / (1 + exp(A * s_i + B))

where parameters A and B are estimated via maximum likelihood on a hold-out 
calibration set.

Validation Protocol (Out-of-Distribution & Robustness Checks):
The script performs out-of-distribution (OOD) cross-validation across three 
leave-one-group-out evaluation regimes:
1. Unseen Claims: ests generalization across novel topics/arguments not observed during training.
2. Unseen Models: Evaluates cross-architecture transferability across different target LLMs.
3. Unseen Categories: Assesses macro-level robustness to entirely held-out thematic domains.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve 
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

FUNCTIONAL_COLUMNS = [
    "alignment_score",
    "epistemic_uncertainty",
    "hedging_score",
    "entrainment_score",
]

VARIANCE_THRESHOLD = 0.90
CLAIM_CV_SPLITS = 5
CALIBRATION_FOLDS = 3
CALIBRATION_METHOD = "sigmoid"
CLASSIFICATION_THRESHOLD = 0.50
RANDOM_STATE = 42

# BOOLEAN PARSING
def parse_bool_series(series):
    if series.dtype == bool:
        return series.astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    parsed = (series.astype(str).str.strip().str.lower().map(mapping))

    if parsed.isna().any():
        unknown = series[parsed.isna()].unique()

        raise ValueError( f"Unable to parse boolean values: {unknown}")

    return parsed.astype(bool)

# TRAPEZOIDAL QUADRATURE
def trapezoidal_weights(t):
    t = np.asarray(t, dtype=float)

    if len(t) < 2:
        raise ValueError("At least two time points are required.")

    w = np.zeros_like(t)
    w[0] = (t[1] - t[0]) / 2.0
    w[-1] = (t[-1] - t[-2]) / 2.0

    if len(t) > 2:
        w[1:-1] = (t[2:] - t[:-2]) / 2.0

    return w

# GLOBAL UNSUPERVISED FUNCTIONAL SUPPORT
def determine_common_support(functional_df, functional_columns):
    session_ids = (functional_df["session_id"].drop_duplicates().tolist())
    supports = {}

    for column in functional_columns:
        pivot = (functional_df.pivot(
                index="session_id",
                columns="t",
                values=column
            )
            .reindex(index=session_ids)
            .sort_index(axis=1)
        )

        valid = pivot.notna().all(axis=0)
        grid = (pivot.columns[valid].to_numpy(dtype=float))

        if len(grid) < 2:
            raise ValueError(f"Insufficient common support for {column}")

        supports[column] = grid

    return supports

# EXTRACT FUNCTIONAL MATRIX
def extract_component_matrix(functional_df, session_ids, column, grid):
    pivot = (functional_df[functional_df["session_id"].isin(session_ids)]
        .pivot(
            index="session_id",
            columns="t",
            values=column
        )
        .reindex(
            index=session_ids,
            columns=grid
        )
    )

    if pivot.isna().any().any():
        missing = int(pivot.isna().sum().sum())

        raise ValueError(f"{column}: {missing} missing values inside the predefined functional support.")

    return pivot.to_numpy(dtype=float)

# FIT FPCA ON TRAINING SET
def fit_fpca_train(functional_df, train_session_ids, supports):
    component_models = {}
    weighted_blocks = []

    for column in FUNCTIONAL_COLUMNS:
        grid = supports[column]
        X = extract_component_matrix(functional_df, train_session_ids, column, grid)

        # Training mean function
        mean_function = X.mean(axis=0)
        centered = (X - mean_function)

        # Functional L2 scale
        weights = trapezoidal_weights(grid)
        integrated_variance = np.mean(np.sum(centered ** 2 * weights[None, :], axis=1))
        scale = np.sqrt(integrated_variance)

        if scale <= 1e-12:
            raise ValueError(f"{column} has near-zero functional variance.")

        standardized = (centered / scale)
        weighted = (standardized * np.sqrt(weights)[None, :])
        weighted_blocks.append(weighted)

        component_models[column] = {
            "grid": grid,
            "mean_function": mean_function,
            "scale": scale,
            "weights": weights,
        }

    # Multivariate functional matrix
    Z_train = np.concatenate(weighted_blocks, axis=1)
    max_components = min(Z_train.shape[0] - 1, Z_train.shape[1])

    pca = PCA(n_components=max_components, svd_solver="full")
    scores_train_all = (pca.fit_transform(Z_train))
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    n_selected = int(np.searchsorted(cumulative_variance, VARIANCE_THRESHOLD) + 1)

    scores_train = (scores_train_all[:, :n_selected])

    return {
        "components": component_models,
        "pca": pca,
        "n_selected": n_selected,
        "cumulative_variance": cumulative_variance,
        "scores_train": scores_train,
        "n_functional_features": Z_train.shape[1],
    }

# PROJECT TEST DATA INTO TRAINING FPCA SPACE
def transform_fpca(functional_df, session_ids, fpca_model,):
    weighted_blocks = []

    for column in FUNCTIONAL_COLUMNS:
        model = ( fpca_model["components"][column])

        X = extract_component_matrix(functional_df, session_ids, column, model["grid"])
        centered = (X - model["mean_function"])
        standardized = (centered / model["scale"])
        weighted = (standardized * np.sqrt(model["weights"])[None, :])

        weighted_blocks.append(weighted)

    Z = np.concatenate(weighted_blocks, axis=1)
    scores_all = (fpca_model["pca"].transform(Z))

    return scores_all[:,:fpca_model["n_selected"]]

# CLASSIFIER + CI CALIBRATION
def build_logistic_model():
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )

def fit_ci_models(X_train, y_train, fold_seed):
    """
    Returns:

    raw_model:
        logistic regression probability.

    calibrated_model:
        sigmoid-calibrated probability = CI.
    """

    # Raw classifier
    raw_model = build_logistic_model()
    raw_model.fit(X_train, y_train)

    # Calibration
    class_counts = np.bincount(y_train, minlength=2)
    minimum_class_count = int(class_counts.min())
    calibration_folds = min( CALIBRATION_FOLDS, minimum_class_count)

    if calibration_folds < 2:
        return (raw_model, None, "none")

    inner_cv = StratifiedKFold(n_splits=calibration_folds, shuffle=True, random_state=fold_seed,)

    calibrated_model = (
        CalibratedClassifierCV(
            estimator=build_logistic_model(),
            method=CALIBRATION_METHOD,
            cv=inner_cv,
            ensemble=False,
        )
    )

    calibrated_model.fit( X_train, y_train)

    return (raw_model, calibrated_model, CALIBRATION_METHOD)

# METRICS
def expected_calibration_error(y_true, probabilities, n_bins=10):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)

    bin_ids = np.digitize(probabilities, edges[1:-1], right=True)
    ece = 0.0

    for bin_idx in range(n_bins):
        mask = (bin_ids == bin_idx)

        if not mask.any():
            continue

        confidence = (probabilities[mask].mean())
        accuracy = (y_true[mask].mean())
        weight = (mask.sum() / len(y_true))
        
        ece += (weight * abs(accuracy - confidence))

    return float(ece)

def safe_roc_auc(y_true, probability):
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, probability)

def safe_average_precision(y_true, probability):
    if len(np.unique(y_true)) < 2:
        return np.nan

    return average_precision_score(y_true, probability)

def compute_metrics(y_true,  ci_score, raw_probability=None,):
    y_true = np.asarray(y_true, dtype=int)
    ci_score = np.asarray(ci_score, dtype=float)
    prediction = (ci_score >= CLASSIFICATION_THRESHOLD).astype(int)

    metrics = {
        "n_samples": len(y_true),
        "positive_rate": float(y_true.mean()),
        "roc_auc": safe_roc_auc(y_true, ci_score),
        "average_precision": safe_average_precision(y_true, ci_score),
        "brier_score":brier_score_loss(y_true, ci_score),
        "log_loss": log_loss(y_true, ci_score, labels=[0, 1]),
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision":precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "ece": expected_calibration_error(y_true, ci_score),
    }

    if raw_probability is not None:
        metrics["raw_brier_score"] = brier_score_loss(y_true, raw_probability)
        metrics["raw_log_loss"] = log_loss(y_true, raw_probability, labels=[0, 1])

    return metrics

# SESSION-LEVEL TABLE
def build_session_table(functional_df, labels_df):
    metadata = (functional_df.groupby("session_id", as_index=False).first())

    required_metadata = [
        "session_id",
        "model",
        "claim",
        "claim_category",
    ]

    metadata = metadata[required_metadata]
    labels = labels_df[["session_id","conditioning_candidate",]].copy()
    labels["conditioning_candidate"] = parse_bool_series(labels["conditioning_candidate"])

    session_df = metadata.merge(labels, on="session_id", how="inner", validate="1:1")

    if len(session_df) != (functional_df["session_id"].nunique()):
        raise ValueError("Mismatch between functional data and session labels.")

    return session_df

# CROSS-VALIDATION SPLITS
def generate_splits(session_df, scheme):
    y = (session_df["conditioning_candidate"].astype(int).to_numpy())

    if scheme == "claim":
        groups = (session_df["claim"].astype(str).to_numpy())
        n_groups = (session_df["claim"].nunique())
        n_splits = min(CLAIM_CV_SPLITS, n_groups)

        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        yield from cv.split(session_df, y, groups)
    elif scheme == "model":
        groups = (session_df["model"].astype(str).to_numpy())
        cv = LeaveOneGroupOut()

        yield from cv.split(session_df, y, groups)
    elif scheme == "category":
        groups = (session_df["claim_category"].astype(str).to_numpy())

        cv = LeaveOneGroupOut()

        yield from cv.split(session_df, y, groups)
    else:
        raise ValueError(f"Unknown CV scheme: {scheme}")

# RUN ONE CV SCHEME
def run_cv_scheme(functional_df, session_df, supports, scheme,):
    predictions = []
    fold_metrics = []
    y_all = (session_df["conditioning_candidate"].astype(int).to_numpy())

    for fold_id, (train_idx, test_idx) in enumerate(generate_splits(session_df, scheme), start=1):
        train_meta = (session_df.iloc[train_idx].copy())
        test_meta = (session_df.iloc[test_idx].copy())
        train_ids = (train_meta["session_id"].tolist())

        test_ids = (test_meta["session_id"].tolist())
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        if len(np.unique(y_train)) < 2:
            raise ValueError(f"{scheme} fold {fold_id}: training set contains only one class.")

        # 1. TRAIN-ONLY FPCA
        fpca_model = fit_fpca_train(functional_df, train_ids, supports,)

        X_train = (fpca_model["scores_train"])
        X_test = transform_fpca(functional_df, test_ids, fpca_model)

        # 2. TRAIN-ONLY PC STANDARDIZATION
        score_scaler = (StandardScaler())

        X_train_scaled = (score_scaler.fit_transform(X_train))
        X_test_scaled = ( score_scaler.transform(X_test))

        # 3. LOGISTIC MODEL + CALIBRATION
        (raw_model, calibrated_model, calibration_method,) = fit_ci_models(
            X_train_scaled,
            y_train,
            fold_seed=(RANDOM_STATE + fold_id)
        )

        raw_probability = (raw_model.predict_proba(X_test_scaled)[:, 1])

        if calibrated_model is not None:
            ci_score = (calibrated_model.predict_proba(X_test_scaled)[:, 1])
        else:
            ci_score = (raw_probability.copy())

        predicted = (ci_score >= CLASSIFICATION_THRESHOLD).astype(int)

        # 4. SAVE INDIVIDUAL OOF PREDICTIONS
        for local_idx, (session_id,  y_true, raw_prob, ci, pred) in enumerate(zip(test_ids, y_test, raw_probability, ci_score, predicted)):
            row_meta = (test_meta.iloc[local_idx])

            predictions.append({
                "scheme": scheme,
                "fold": fold_id,
                "session_id": session_id,
                "model": row_meta["model"],
                "claim": row_meta["claim"],
                "claim_category": row_meta["claim_category"],
                "true_label": int(y_true),
                "raw_probability": float(raw_prob),
                "ci_score": float(ci),
                "predicted_label": int(pred),
                "n_fpca_components": fpca_model["n_selected"],
            })

        # 5. FOLD METRICS
        metrics = compute_metrics(y_test, ci_score, raw_probability,)
        metrics.update({
            "scheme": scheme,
            "fold": fold_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_positive_rate": float(y_train.mean()),
            "n_fpca_components": fpca_model["n_selected"],
            "fpca_cumulative_variance": float(fpca_model["cumulative_variance"][fpca_model["n_selected"] - 1]),
            "n_functional_features": fpca_model["n_functional_features"],
            "calibration_method": calibration_method,
        })

        fold_metrics.append(metrics)

        print(
            f"[{scheme}] "
            f"Fold {fold_id}: "
            f"train={len(train_idx)}, "
            f"test={len(test_idx)}, "
            f"PCs={fpca_model['n_selected']}, "
            f"AUC={metrics['roc_auc']:.3f}, "
            f"AP={metrics['average_precision']:.3f}"
        )

    return (pd.DataFrame(predictions), pd.DataFrame(fold_metrics),)

# POOLED CV SUMMARY
def build_cv_summary(prediction_df, fold_metrics_df):
    rows = []

    for scheme, group in ( prediction_df.groupby("scheme")):
        y = (group["true_label"].astype(int).to_numpy())
        ci = (group["ci_score"].astype(float).to_numpy())
        raw = (group["raw_probability"].astype(float).to_numpy())

        metrics = compute_metrics(y, ci, raw)

        scheme_folds = (fold_metrics_df[fold_metrics_df["scheme"] == scheme])

        metrics.update({
            "scheme": scheme,
            "n_folds": len(scheme_folds),
            "mean_fpca_components": scheme_folds["n_fpca_components"].mean(),
            "std_fpca_components": scheme_folds["n_fpca_components"].std(),
            "ci_mean_conditioned": group.loc[group["true_label"] == 1, "ci_score"].mean(),
            "ci_mean_non_conditioned": group.loc[ group["true_label"] == 0,"ci_score"].mean(),
        })

        rows.append(metrics)

    return pd.DataFrame(rows)

# CALIBRATION TABLE
def build_calibration_table(prediction_df, n_bins=10):
    rows = []

    for scheme, group in (prediction_df.groupby("scheme")):
        y = (group["true_label"].astype(int).to_numpy())

        probability = (group["ci_score"].astype(float).to_numpy())

        observed, predicted = (calibration_curve(y, probability, n_bins=n_bins, strategy="quantile"))

        for bin_idx, (pred_prob, obs_prob) in enumerate(zip( predicted,observed), start=1):
            rows.append({
                "scheme": scheme,
                "bin": bin_idx,
                "mean_predicted_ci": pred_prob,
                "observed_conditioning_rate": obs_prob,
            })

    return pd.DataFrame(rows)

# CALIBRATION PLOTS
def plot_calibration(calibration_df, figures_dir):
    for scheme, group in (calibration_df.groupby("scheme")):
        fig, ax = plt.subplots(figsize=(7, 7))

        ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        ax.plot(group["mean_predicted_ci"], group["observed_conditioning_rate" ], marker="o", label="Observed")
        ax.set_xlabel( "Mean predicted Conditioning Index")
        ax.set_ylabel( "Observed conditioning frequency")
        ax.set_title(f"CI calibration: {scheme}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"calibration_{scheme}.png"), dpi=300, bbox_inches="tight",)
        plt.close(fig)

# ===================================================================================================================
def main():
    main_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))
    functional_path = os.path.join(main_dir, "FUNCTIONAL_GRID.csv")
    labels_path = os.path.join(main_dir, "SESSION_LABELS.csv")
    cv_dir = os.path.join(main_dir, "ci_cross_validation")
    figures_dir = os.path.join(cv_dir,"figures")

    functional_df = pd.read_csv(functional_path)
    labels_df = pd.read_csv(labels_path)

    # Session-level metadata + labels
    session_df = build_session_table(functional_df, labels_df)

    print("Dataset:")
    print(f"Sessions: {len(session_df)}")
    print("Conditioning prevalence:",session_df["conditioning_candidate"].mean())

    # Fixed common functional support
    supports = determine_common_support(functional_df, FUNCTIONAL_COLUMNS)
    total_features = sum(len(grid)for grid in supports.values())
    print("Functional coordinates:", total_features)

    # Cross-validation schemes
    schemes = ["claim", "model","category",]
    all_predictions = []
    all_fold_metrics = []

    for scheme in schemes:
        print("\n" + "=" * 60)
        print(f"Running {scheme} cross-validation")
        print( "=" * 60)

        predictions, fold_metrics = (run_cv_scheme( functional_df, session_df, supports, scheme))
        all_predictions.append(predictions)
        all_fold_metrics.append(fold_metrics)

    prediction_df = pd.concat(all_predictions, ignore_index=True)
    fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)

    # Pooled OOF summary
    summary_df = build_cv_summary(prediction_df, fold_metrics_df)

    # Calibration analysis
    calibration_df = (build_calibration_table(prediction_df))

    prediction_df.to_csv(os.path.join(cv_dir, "CI_OOF_PREDICTIONS.csv"), index=False)
    fold_metrics_df.to_csv(os.path.join(cv_dir, "CI_FOLD_METRICS.csv"), index=False)
    summary_df.to_csv(os.path.join(cv_dir, "CI_CV_SUMMARY.csv"), index=False)
    calibration_df.to_csv(os.path.join(cv_dir, "CI_CALIBRATION.csv"), index=False)

    # Calibration figures
    plot_calibration(calibration_df, figures_dir)

    # Metadata
    metadata = {
        "functional_columns": FUNCTIONAL_COLUMNS,
        "variance_threshold": VARIANCE_THRESHOLD,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "calibration_method": CALIBRATION_METHOD,
        "claim_cv_splits": CLAIM_CV_SPLITS,
        "random_state": RANDOM_STATE,
        "cv_schemes": schemes,
        "n_sessions": len(session_df),
        "conditioning_prevalence": float(session_df["conditioning_candidate"].mean()),
        "functional_support": {
            column: {
                "n_points": len(grid),
                "t_min":float(grid.min()),
                "t_max": float(grid.max()),
            } for column, grid in supports.items()
        }
    }

    with open(os.path.join(cv_dir, "CI_CV_METADATA.json"), "w", encoding="utf-8",) as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(summary_df[
            [
                "scheme",
                "roc_auc",
                "average_precision",
                "brier_score",
                "balanced_accuracy",
                "f1",
                "ece",
                "mean_fpca_components",
            ]].to_string(index=False)
    )

    print(f"\nResults saved in:\n{cv_dir}")

if __name__ == "__main__":
    main()


