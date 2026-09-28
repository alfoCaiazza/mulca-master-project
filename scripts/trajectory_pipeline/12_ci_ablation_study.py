"""
FEATURE ABLATION SUITE
1. FULL = [A, E, H, M]
   Complete feature space combining Alignment (A) with Epistemic Uncertainty (E),
   Entropy/Hedging (H), and Entrainment (M).

2. ALIGNMENT ONLY = [A]
   Baseline isolating stance displacement and directional alignment dynamics.

3. NO ALIGNMENT = [E, H, M]
   Ablated space evaluating whether non-alignment conversational dynamics 
   (uncertainty, lexical/syntactic entrainment, entropy) suffice for detection.

- Evaluation Metrics:
  * Pooled Metrics: Concatenated out-of-fold predictions evaluated globally.
  * Fold-level Statistics: Macro-averaged metric mean +/- standard deviation (SD) 
    computed across folds.
  * Cross-Configuration Comparisons: Direct paired differences (delta) and fold-wise 
    hypothesis testing against FULL.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

# CONFIGURATION
ABLATION_CONFIGS = {
    "full": [
        "alignment_score",
        "epistemic_uncertainty",
        "hedging_score",
        "entrainment_score",
    ],

    "alignment_only": [
        "alignment_score",
    ],

    "no_alignment": [
        "epistemic_uncertainty",
        "hedging_score",
        "entrainment_score",
    ],
}

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
        unknown = (series[parsed.isna()].unique())

        raise ValueError(f"Unable to parse boolean values: {unknown}")

    return parsed.astype(bool)

# TRAPEZOIDAL WEIGHTS
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

# COMMON SUPPORT
def determine_common_support(functional_df, functional_columns):
    session_ids = (functional_df["session_id"].drop_duplicates().tolist())
    supports = {}

    for column in functional_columns:
        pivot = (functional_df
                .pivot(index="session_id", columns="t", values=column)
                .reindex(index=session_ids).sort_index(axis=1))

        valid = (pivot.notna().all(axis=0))
        grid = (pivot.columns[valid].to_numpy(dtype=float))

        if len(grid) < 2:
            raise ValueError(f"Insufficient common support for {column}")

        supports[column] = grid

    return supports

# EXTRACT FUNCTIONAL MATRIX
def extract_component_matrix(functional_df, session_ids, column, grid,):
    pivot = (functional_df[functional_df["session_id"].isin(session_ids)]
        .pivot(index="session_id", columns="t", values=column)
        .reindex(index=session_ids, columns=grid,))

    if pivot.isna().any().any():
        missing = int(pivot.isna().sum().sum())

        raise ValueError(f"{column}: {missing} missing values inside functional support.")

    return pivot.to_numpy(dtype=float)

# TRAIN-ONLY MULTIVARIATE FPCA
def fit_fpca_train(functional_df, train_session_ids, functional_columns, supports):
    component_models = {}
    weighted_blocks = []

    for column in functional_columns:
        grid = supports[column]

        X = extract_component_matrix(functional_df, train_session_ids, column,grid)

        # Pointwise functional mean
        mean_function = (X.mean(axis=0))
        centered = (X - mean_function)

        # Functional variance scaling
        weights = (trapezoidal_weights(grid))

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

    # Multivariate functional representation
    Z_train = np.concatenate( weighted_blocks, axis=1,)
    max_components = min(Z_train.shape[0] - 1, Z_train.shape[1],)

    pca = PCA(n_components=max_components, svd_solver="full",)

    scores_all = (pca.fit_transform(Z_train))
    cumulative_variance = (np.cumsum(pca.explained_variance_ratio_))

    n_selected = int(np.searchsorted(cumulative_variance, VARIANCE_THRESHOLD) + 1)

    scores_train = (scores_all[:, :n_selected])

    return {
        "components": component_models,
        "pca": pca,
        "n_selected": n_selected,
        "scores_train": scores_train,
        "cumulative_variance": cumulative_variance,
        "n_functional_features": Z_train.shape[1],
    }

# PROJECT TEST SET
def transform_fpca(functional_df, session_ids, functional_columns, fpca_model,):
    weighted_blocks = []
    for column in functional_columns:
        model = (fpca_model["components"][column])

        X = extract_component_matrix(functional_df, session_ids, column, model["grid"],)

        centered = (X - model["mean_function"])
        standardized = (centered / model["scale"])
        weighted = (standardized * np.sqrt(model["weights"])[None, :])

        weighted_blocks.append(weighted)

    Z = np.concatenate(weighted_blocks, axis=1)

    scores = (fpca_model["pca"].transform(Z))

    return scores[:, :fpca_model["n_selected"]]

# CLASSIFIER
def build_logistic_model():
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )

# FIT RAW + CALIBRATED MODEL
def fit_ci_models(X_train, y_train, fold_seed):
    # Raw classifier
    raw_model = build_logistic_model()
    raw_model.fit(X_train, y_train)

    # Calibration
    class_counts = np.bincount(y_train, minlength=2)
    minimum_class_count = int(class_counts.min())
    calibration_folds = min( CALIBRATION_FOLDS, minimum_class_count)

    if calibration_folds < 2:
        return (raw_model, None, "none")

    inner_cv = StratifiedKFold(n_splits=calibration_folds, shuffle=True, random_state=fold_seed)

    calibrated_model = (
        CalibratedClassifierCV(
            estimator=build_logistic_model(),
            method=CALIBRATION_METHOD,
            cv=inner_cv,
            ensemble=False
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

# SESSION TABLE
def build_session_table(functional_df, labels_df,):
    metadata = (functional_df.groupby("session_id", as_index=False).first())

    required_metadata = [
        "session_id",
        "model",
        "claim",
        "claim_category",
    ]

    metadata = metadata[required_metadata]
    labels = (labels_df[["session_id", "conditioning_candidate"]].copy())
    labels["conditioning_candidate"] = parse_bool_series(labels[ "conditioning_candidate"])

    session_df = metadata.merge(labels, on="session_id", how="inner", validate="1:1")

    return session_df

# GENERATE IDENTICAL CV SPLITS
def generate_split_list(session_df, scheme):
    y = ( session_df["conditioning_candidate"].astype(int).to_numpy())

    if scheme == "claim":
        groups = (session_df["claim"].astype(str).to_numpy())
        n_groups = (session_df["claim"].nunique())
        n_splits = min(CLAIM_CV_SPLITS, n_groups,)

        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    elif scheme == "model":
        groups = (session_df["model"].astype(str).to_numpy())
        cv = LeaveOneGroupOut()
    elif scheme == "category":
        groups = (session_df["claim_category"].astype(str).to_numpy())
        cv = LeaveOneGroupOut()
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    return list(cv.split(session_df, y, groups))

# RUN ONE ABLATION CONFIGURATION
def run_ablation_configuration(functional_df, session_df, supports, scheme, split_list, config_name, functional_columns,):
    prediction_rows = []
    fold_rows = []

    y_all = (session_df["conditioning_candidate"].astype(int).to_numpy())

    for fold_id, (train_idx, test_idx,) in enumerate(split_list, start=1):
        train_meta = (session_df.iloc[train_idx].copy())
        test_meta = (session_df.iloc[test_idx].copy())

        train_ids = (train_meta["session_id"].tolist())
        test_ids = (test_meta["session_id"].tolist())

        y_train = (y_all[train_idx])
        y_test = (y_all[test_idx])

        if len(np.unique(y_train)) < 2:
            raise ValueError(f"{scheme} / {config_name} / fold {fold_id}: only one class in training.")

        # FPCA TRAIN ONLY
        fpca_model = fit_fpca_train(functional_df, train_ids, functional_columns, supports,)

        X_train = (fpca_model["scores_train"])
        X_test = transform_fpca(functional_df, test_ids, functional_columns, fpca_model,)

        # PC STANDARDIZATION
        scaler = StandardScaler()
        X_train_scaled = (scaler.fit_transform(X_train))
        X_test_scaled = (scaler.transform(X_test))

        # CLASSIFIER + CALIBRATION
        raw_model, calibrated_model, calibration_method = fit_ci_models(
            X_train_scaled,
            y_train,
            fold_seed=(RANDOM_STATE + fold_id)
        )

        raw_probability = (raw_model.predict_proba(X_test_scaled)[:, 1])

        if calibrated_model is None:
            ci_score = (raw_probability.copy())
        else:
            ci_score = (calibrated_model.predict_proba(X_test_scaled)[:, 1])

        predicted = (ci_score >= CLASSIFICATION_THRESHOLD).astype(int)

        # INDIVIDUAL PREDICTIONS
        for local_idx, session_id in enumerate(test_ids):
            meta = (test_meta.iloc[local_idx])

            prediction_rows.append({
                "scheme": scheme,
                "configuration": config_name,
                "fold": fold_id,
                "session_id":session_id,
                "model": meta["model"],
                "claim": meta["claim"],
                "claim_category": meta["claim_category"],
                "true_label":int(y_test[local_idx]),
                "raw_probability":float(raw_probability[local_idx]),
                "ci_score":float(ci_score[local_idx]),
                "predicted_label":int(predicted[local_idx]),
                "n_fpca_components":int(fpca_model["n_selected"])
            })

        # FOLD METRICS
        metrics = compute_metrics(y_test, ci_score, raw_probability)
        metrics.update({
            "scheme": scheme,
            "configuration": config_name,
            "fold": fold_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_positive_rate": float(y_train.mean()),
            "n_fpca_components": int(fpca_model["n_selected"]),
            "fpca_cumulative_variance": (fpca_model["cumulative_variance"][fpca_model["n_selected"] - 1]),
            "n_functional_features": int(fpca_model["n_functional_features"]),
            "calibration_method": calibration_method,
        })

        fold_rows.append(metrics)

        print(
            f"[{scheme}] "
            f"[{config_name}] "
            f"Fold {fold_id}: "
            f"PCs={fpca_model['n_selected']}, "
            f"AUC={metrics['roc_auc']:.3f}, "
            f"AP={metrics['average_precision']:.3f}, "
            f"Brier={metrics['brier_score']:.3f}"
        )

    return (pd.DataFrame(prediction_rows), pd.DataFrame(fold_rows))

# FOLD-LEVEL SUMMARY
def build_fold_summary(fold_metrics_df,):
    metric_columns = [
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
        "balanced_accuracy",
        "f1",
        "ece",
        "n_fpca_components",
    ]

    rows = []

    for(scheme, configuration), group in fold_metrics_df.groupby(["scheme", "configuration"]):
        row = {
            "scheme": scheme,
            "configuration": configuration,
            "n_folds": len(group),
        }

        for metric in metric_columns:
            values = (group[metric].astype(float))

            row[f"{metric}_mean"] = values.mean()
            row[ f"{metric}_std"] = values.std(ddof=1)

        rows.append(row)

    return pd.DataFrame(rows)

# POOLED OOF SUMMARY
def build_pooled_summary(prediction_df):
    rows = []

    for(scheme, configuration), group in prediction_df.groupby(["scheme", "configuration"]):
        y = (group["true_label"].astype(int).to_numpy())
        ci = ( group["ci_score"].astype(float).to_numpy())

        raw = (group["raw_probability"].astype(float).to_numpy())
        metrics = compute_metrics(y, ci, raw,)

        metrics.update({
            "scheme": scheme,
            "configuration": configuration,
        })

        rows.append( metrics)

    return pd.DataFrame(rows)

# PAIRED DELTAS AGAINST FULL MODEL
def build_ablation_deltas(fold_metrics_df):
    metrics = [
        "roc_auc",
        "average_precision",
        "brier_score",
        "balanced_accuracy",
        "f1",
        "ece",
    ]

    rows = []

    for scheme in (fold_metrics_df["scheme"].unique()):
        scheme_df = (fold_metrics_df[fold_metrics_df["scheme"] == scheme])
        
        full = (scheme_df[scheme_df["configuration"] == "full"].set_index("fold"))

        for configuration in (ABLATION_CONFIGS.keys()):
            if configuration == "full":
                continue

            current = (scheme_df[scheme_df["configuration"] == configuration].set_index("fold"))
            common_folds = (full.index.intersection(current.index))

            for metric in metrics:
                delta = (current.loc[common_folds, metric] - full.loc[common_folds, metric])

                rows.append({
                    "scheme": scheme,
                    "configuration": configuration,
                    "reference":"full",
                    "metric": metric,
                    "mean_delta_vs_full": delta.mean(),
                    "std_delta_vs_full": delta.std(ddof=1),
                    "n_paired_folds": len(common_folds),
                })

    return pd.DataFrame(rows)

# =======================================================================================================================
def main():
    main_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))
    functional_path = os.path.join(main_dir, "FUNCTIONAL_GRID.csv")
    labels_path = os.path.join(main_dir, "SESSION_LABELS.csv")
    output_dir = os.path.join(main_dir, "ablation_study",)

    os.makedirs(output_dir, exist_ok=True)

    functional_df = pd.read_csv(functional_path)
    labels_df = pd.read_csv(labels_path)
    
    # Session-level metadata + labels
    session_df = build_session_table(functional_df, labels_df)

    print("Dataset:")
    print(f"Sessions: {len(session_df)}")
    print("Conditioning prevalence:",session_df["conditioning_candidate"].mean())

    # Support for every possible functional variable
    all_columns = sorted({column for columns in ABLATION_CONFIGS.values() for column in columns})
    supports = (determine_common_support(functional_df, all_columns))
    schemes = [
        "claim",
        "model",
        "category",
    ]

    all_predictions = []
    all_fold_metrics = []

    # Build splits ONCE per scheme. All ablations receive exactly the same folds.
    for scheme in schemes:
        print("\n" + "=" * 70)
        print(f"ABLATION STUDY: {scheme.upper()}")
        print("=" * 70)

        split_list = (generate_split_list(session_df, scheme))

        for (config_name, functional_columns) in ABLATION_CONFIGS.items():
            print(f"\nConfiguration: {config_name}")
            print("Dimensions:", functional_columns)
            predictions, fold_metrics = run_ablation_configuration(
                functional_df,
                session_df,
                supports,
                scheme,
                split_list,
                config_name,
                functional_columns
            )

            all_predictions.append(predictions)
            all_fold_metrics.append(fold_metrics)

    # CONCATENATE RESULTS
    prediction_df = pd.concat(all_predictions, ignore_index=True,)
    fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True,)

    fold_summary_df = (build_fold_summary(fold_metrics_df))
    pooled_summary_df = (build_pooled_summary(prediction_df))
    delta_df = (build_ablation_deltas(fold_metrics_df))

    prediction_df.to_csv(os.path.join(output_dir, "ABLATION_OOF_PREDICTIONS.csv"), index=False)
    fold_metrics_df.to_csv(os.path.join(output_dir, "ABLATION_FOLD_METRICS.csv"), index=False)
    fold_summary_df.to_csv(os.path.join(output_dir, "ABLATION_OOF_PREDICTIONS.csv"), index=False)
    pooled_summary_df.to_csv(os.path.join(output_dir, "ABLATION_POOLED_SUMMARY.csv"), index=False)
    delta_df.to_csv(os.path.join(output_dir, "ABLATION_DELTAS_VS_FULL.csv"), index=False)

    # METADATA
    metadata = {
        "ablation_configs": ABLATION_CONFIGS,
        "variance_threshold": VARIANCE_THRESHOLD,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "calibration_method": CALIBRATION_METHOD,
        "claim_cv_splits": CLAIM_CV_SPLITS,
        "random_state": RANDOM_STATE,
        "n_sessions": len(session_df),
        "conditioning_prevalence": float(session_df["conditioning_candidate"].mean()),
    }

    with open(os.path.join(output_dir, "ABLATION_METADATA.json",), "w", encoding="utf-8",) as f:
        json.dump(metadata, f, indent=4)

    columns_to_print = [
        "scheme",
        "configuration",
        "roc_auc_mean",
        "roc_auc_std",
        "average_precision_mean",
        "average_precision_std",
        "brier_score_mean",
        "brier_score_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "f1_mean",
        "f1_std",
        "n_fpca_components_mean",
    ]

    print("\n" + "=" * 70)
    print("ABLATION STUDY — FOLD-LEVEL SUMMARY")
    print( "=" * 70)
    print(fold_summary_df[columns_to_print].sort_values( ["scheme","configuration"]).to_string( index=False))

    print( "\n" + "=" * 70)
    print("DELTA VS FULL MODEL")
    print( "=" * 70)
    print(delta_df[delta_df["metric"].isin(["roc_auc", "average_precision", "brier_score"])].to_string(index=False))

    print(f"\nResults saved in: {output_dir}")


if __name__ == "__main__":
    main()