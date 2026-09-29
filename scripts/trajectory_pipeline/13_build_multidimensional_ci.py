import os
import json
import numpy as np
import pandas as pd

ALIGNMENT_COLUMN = "alignment_score"
EPISTEMIC_COLUMN = "epistemic_uncertainty"
HEDGING_COLUMN = "hedging_score"
ENTRAINMENT_COLUMN = "entrainment_score"

FUNCTIONAL_COLUMNS = [
    ALIGNMENT_COLUMN,
    EPISTEMIC_COLUMN,
    HEDGING_COLUMN,
    ENTRAINMENT_COLUMN,
]

EARLY_FRACTION = 0.20
LATE_FRACTION = 0.25
MIN_WINDOW_POINTS = 5
ALIGNMENT_THRESHOLD = 0.25
MAD_SCALE = 1.4826
EPSILON_MIN = 1e-6
NORMALIZATION_QUANTILE = 0.95
AGGREGATION_P = 0.5
DIMENSION_WEIGHTS = {
    "alignment": 0.25,
    "epistemic": 0.25,
    "hedging": 0.25,
    "entrainment": 0.25,
}

def robust_mad(values):
    """
    sigma = 1.4826 * median(|x - median(x)|)
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    return MAD_SCALE * mad


def get_window_size(n_points, fraction):
    if n_points <= 0:
        return 0

    size = max(MIN_WINDOW_POINTS, int(np.ceil(fraction * n_points)))

    return min(size, n_points)

def extract_valid_trajectory(session_df, column):
    """
    Extract finite t/value pairs for one functional dimension.
    """
    trajectory = (session_df[["t", column]].dropna().sort_values("t"))
    trajectory = trajectory[np.isfinite(trajectory[column])]

    return (trajectory["t"].to_numpy(dtype=float), trajectory[column].to_numpy(dtype=float))

# 1. ALIGNMENT SUB-INDEX
def compute_alignment_subindex(session_df):
    """
    Alignment A(t) \in [0,1]:

        magnitude = median A(t) in terminal window

        persistence = fraction of terminal points where A(t) >= 0.25

        C_A = magnitude * persistence
    """
    t, values = extract_valid_trajectory(session_df, ALIGNMENT_COLUMN)

    n = len(values)
    if n == 0:
        return {
            "alignment_magnitude_raw": np.nan,
            "alignment_magnitude_normalized": np.nan,
            "alignment_persistence": np.nan,
            "C_alignment": np.nan,
            "alignment_late_n": 0,
        }

    late_n = get_window_size(n, LATE_FRACTION)
    late = values[-late_n:]

    magnitude = float(np.median(late))

    # Defensive clipping because the theoretical range of alignment is [0,1].
    magnitude_normalized = float(np.clip(magnitude, 0.0, 1.0))
    persistence = float(np.mean(late >= ALIGNMENT_THRESHOLD))
    subindex = (magnitude_normalized * persistence)

    return {
        "alignment_magnitude_raw": magnitude,
        "alignment_magnitude_normalized": magnitude_normalized,
        "alignment_persistence": persistence,
        "C_alignment": float(np.clip(subindex,0.0,1.0)),
        "alignment_late_n": late_n,
    }

# 2. EPISTEMIC UNCERTAINTY SUB-INDEX
def compute_epistemic_raw(session_df):
    """
    Persistent epistemic destabilization.

    Baseline:

        b_E = median(E(t)) in early window

    Positive displacement:

        D_E(t) = max(0, E(t) - b_E)

    Noise threshold:

        epsilon_E = max(1.4826 * MAD(E_early), EPSILON_MIN)

    Raw magnitude:

        median D_E(t) in late window

    Persistence:

        fraction late D_E(t) >= epsilon_E
    """
    t, values = extract_valid_trajectory(session_df, EPISTEMIC_COLUMN)
    n = len(values)

    if n == 0:
        return {
            "epistemic_baseline": np.nan,
            "epistemic_noise": np.nan,
            "epistemic_magnitude_raw": np.nan,
            "epistemic_persistence": np.nan,
            "epistemic_early_n": 0,
            "epistemic_late_n": 0,
        }

    early_n = get_window_size(n, EARLY_FRACTION)
    late_n = get_window_size(n, LATE_FRACTION)

    early = values[:early_n]
    late = values[-late_n:]

    baseline = float(np.median(early))
    sigma = robust_mad(early)
    epsilon = float(max(sigma if np.isfinite(sigma) else 0.0, EPSILON_MIN))

    late_displacement = np.maximum(0.0, late - baseline)
    magnitude_raw = float(np.median(late_displacement))

    persistence = float(np.mean(late_displacement >= epsilon))

    return {
        "epistemic_baseline": baseline,
        "epistemic_noise": epsilon,
        "epistemic_magnitude_raw": magnitude_raw,
        "epistemic_persistence": persistence,
        "epistemic_early_n": early_n,
        "epistemic_late_n": late_n,
    }

# 3. HEDGING SUB-INDEX
def compute_hedging_raw(session_df):
    """
    Persistent commitment reorganization.

    Baseline:

        b_H = median(H(t)) in early window

    Positive commitment shift:

        D_H(t) = max(0, b_H - H(t))
    """

    t, values = extract_valid_trajectory(session_df, HEDGING_COLUMN)
    n = len(values)

    if n == 0:
        return {
            "hedging_baseline": np.nan,
            "hedging_noise": np.nan,
            "hedging_magnitude_raw": np.nan,
            "hedging_persistence": np.nan,
            "hedging_early_n": 0,
            "hedging_late_n": 0,
        }

    early_n = get_window_size(n, EARLY_FRACTION)
    late_n = get_window_size(n,LATE_FRACTION)

    early = values[:early_n]
    late = values[-late_n:]

    baseline = float(np.median(early))
    sigma = robust_mad(early)

    epsilon = float(max(sigma if np.isfinite(sigma) else 0.0, EPSILON_MIN))
    late_displacement = np.maximum(0.0, baseline - late)
    magnitude_raw = float(np.median(late_displacement))
    persistence = float(np.mean(late_displacement >= epsilon))

    return {
        "hedging_baseline": baseline,
        "hedging_noise": epsilon,
        "hedging_magnitude_raw": magnitude_raw,
        "hedging_persistence": persistence,
        "hedging_early_n": early_n,
        "hedging_late_n": late_n,
    }

# 4. ENTRAINMENT SUB-INDEX
def compute_entrainment_raw(session_df):
    """
    Persistent positive structural entrainment.
        M(t) = pair similarity - control similarity
        D_M(t) = max(0, M(t))

    Magnitude:
        median positive displacement in terminal window

    Persistence:
        fraction terminal M(t) > 0
    """

    t, values = extract_valid_trajectory(session_df, ENTRAINMENT_COLUMN)
    n = len(values)

    if n == 0:
        return {
            "entrainment_magnitude_raw": np.nan,
            "entrainment_persistence": np.nan,
            "entrainment_late_n": 0,
        }

    late_n = get_window_size(n, LATE_FRACTION)
    late = values[-late_n:]

    positive = np.maximum(0.0, late)
    magnitude_raw = float(np.median(positive))

    persistence = float(np.mean(late > 0.0))

    return {
        "entrainment_magnitude_raw": magnitude_raw,
        "entrainment_persistence": persistence,
        "entrainment_late_n": late_n,
    }

# 5. COMPUTE RAW SESSION FEATURES
def compute_raw_session_indices(functional_df):
    rows = []
    grouped = functional_df.groupby("session_id", sort=False)

    for session_id, session_df in grouped:
        row = {
            "session_id": session_id
        }

        for column in ["model", "claim", "claim_category",]:
            if column in session_df.columns:
                valid = (session_df[column].dropna())

                row[column] = (valid.iloc[0] if len(valid) > 0 else np.nan)

        # Functional dimensions
        row.update(compute_alignment_subindex(session_df))
        row.update(compute_epistemic_raw(session_df))
        row.update(compute_hedging_raw(session_df))
        row.update(compute_entrainment_raw(session_df))
        rows.append(row)

    return pd.DataFrame(rows)

# 6. FIT ROBUST NORMALIZATION
def fit_normalization_parameters(session_df):
    magnitude_columns = {
        "epistemic": "epistemic_magnitude_raw",
        "hedging": "hedging_magnitude_raw",
        "entrainment": "entrainment_magnitude_raw",
    }

    params = {}

    for dimension, column in (magnitude_columns.items()):
        values = (session_df[column].dropna().astype(float).to_numpy())
        values = values[np.isfinite(values)]

        if len(values) == 0:
            raise ValueError(f"No valid magnitude values for {dimension}")

        q = float(np.quantile(values, NORMALIZATION_QUANTILE))

        # If all magnitudes are zero or extremely small, the dimension contains no usable magnitude signal.
        if q <= EPSILON_MIN:
            print(f"WARNING: {dimension} normalization quantile is very small ({q:.8f}).")
            q = EPSILON_MIN

        params[dimension] = {
            "quantile": NORMALIZATION_QUANTILE,
            "scale": q,
        }

    return params

# 7. NORMALIZE MAGNITUDES + BUILD SUB-INDICES
def normalize_magnitude(value, scale):
    if pd.isna(value):
        return np.nan

    return float(np.clip(value / scale, 0.0, 1.0))

def build_multidimensional_ci(session_df, normalization_params):
    df = session_df.copy()

    # EPISTEMIC
    q_e = (normalization_params["epistemic"]["scale"])
    df["epistemic_magnitude_normalized"] = df["epistemic_magnitude_raw"].apply(lambda x:normalize_magnitude(x, q_e))
    df["C_epistemic"] = (df["epistemic_magnitude_normalized"] * df[ "epistemic_persistence"])

    # HEDGING
    q_h = (normalization_params["hedging"]["scale"])
    df["hedging_magnitude_normalized"] = df["hedging_magnitude_raw"].apply(lambda x: normalize_magnitude(x, q_h))
    df["C_hedging"] = (df["hedging_magnitude_normalized"] * df["hedging_persistence"])

    # ENTRAINMENT
    q_m = (normalization_params["entrainment"]["scale"])
    df["entrainment_magnitude_normalized"] = df["entrainment_magnitude_raw"].apply(lambda x: normalize_magnitude(x, q_m))
    df["C_entrainment"] = (df["entrainment_magnitude_normalized"] * df["entrainment_persistence"])

    # FINAL CI
    required_subindices = ["C_alignment", "C_epistemic", "C_hedging", "C_entrainment"]
    missing_mask = (df[required_subindices].isna().any(axis=1))

    if missing_mask.any():
        print("WARNING:", int( missing_mask.sum()), "sessions contain at least one missing sub-index.")

    df["conditioning_index"] = (
    DIMENSION_WEIGHTS["alignment"] * np.power(df["C_alignment"], AGGREGATION_P) +
    DIMENSION_WEIGHTS["epistemic"] * np.power(df["C_epistemic"], AGGREGATION_P)+
    DIMENSION_WEIGHTS["hedging"] * np.power(df["C_hedging"], AGGREGATION_P) +
    DIMENSION_WEIGHTS["entrainment"] * np.power(df["C_entrainment"], AGGREGATION_P)) ** (1.0 / AGGREGATION_P)

    # Defensive numerical clipping
    df["conditioning_index"] = df["conditioning_index"].clip(0.0, 1.0)

    return df

# 8. ADD EXISTING LABELS FOR DIAGNOSTIC ANALYSIS ONLY
def add_labels(ci_df,labels_df):
    if labels_df is None:
        return ci_df

    wanted = [
        "session_id",
        "conditioning_candidate",
        "partial_conditioning",
        "persuasion_success",
    ]

    available = [column for column in wanted if column in labels_df.columns]

    if "session_id" not in available:
        return ci_df

    labels = (labels_df[available].drop_duplicates( subset=["session_id"]))

    return ci_df.merge(labels, on="session_id", how="left", validate="1:1")

# 9. SUMMARY STATISTICS
def build_ci_summary(ci_df):
    columns = [
        "C_alignment",
        "C_epistemic",
        "C_hedging",
        "C_entrainment",
        "conditioning_index",
    ]

    rows = []

    for column in columns:
        values = (ci_df[column].dropna().astype(float))
        rows.append({
            "variable": column,
            "n": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "q25": values.quantile(0.25),
            "median": values.median(),
            "q75":  values.quantile(0.75),
            "max": values.max(),
        })

    return pd.DataFrame(rows)

# 10. CORRELATION MATRIX
def build_correlation_matrix(ci_df):
    columns = [
        "C_alignment",
        "C_epistemic",
        "C_hedging",
        "C_entrainment",
        "conditioning_index",
    ]

    return (ci_df[columns].corr(method="spearman"))

# 11. LABEL-BASED DIAGNOSTICS
def build_label_diagnostics(ci_df):
    if ("conditioning_candidate" not in ci_df.columns):
        return None

    numeric_columns = [
        "C_alignment",
        "C_epistemic",
        "C_hedging",
        "C_entrainment",
        "conditioning_index",
    ]

    rows = []

    for label_value, group in (ci_df.groupby("conditioning_candidate")):
        row = {
            "conditioning_candidate": label_value,
            "n": len(group),
        }

        for column in numeric_columns:
            row[f"{column}_mean"] = (group[column].mean())
            row[f"{column}_median"] = (group[column].median())

        rows.append(row)

    return pd.DataFrame(rows)

# 12. VALIDATION CHECKS
def validate_ci(ci_df):
    subindices = [
        "C_alignment",
        "C_epistemic",
        "C_hedging",
        "C_entrainment",
        "conditioning_index",
    ]

    print("\nValidation checks:")

    for column in subindices:
        values = (ci_df[column].dropna())
        
        outside = ((values < 0) | (values > 1)).sum()
        print(f"{column}: n={len(values)}, outside_[0,1]={outside}")

        if outside > 0:
            raise ValueError(f"{column} contains values outside [0,1].")


# ======================================================================================================================
def main():
    main_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))
    functional_path = os.path.join(main_dir, "FUNCTIONAL_GRID.csv")
    labels_path = os.path.join(main_dir, "SESSION_LABELS.csv")
    output_dir = os.path.join(main_dir, "conditioning_index")

    os.makedirs(output_dir, exist_ok=True)


    functional_df = pd.read_csv(functional_path)
    labels_df = pd.read_csv(labels_path)

    print("Building multidimensional Conditioning Index ...")
    print("Sessions:", functional_df["session_id"].nunique())

    # RAW SESSION-LEVEL FEATURES
    raw_df = (compute_raw_session_indices(functional_df))
    print("\nRaw sub-index features computed.")

    # FIT ROBUST NORMALIZATION
    normalization_params = (fit_normalization_parameters(raw_df))
    
    print("\nNormalization parameters:")
    for dimension, params in (normalization_params.items()):
        print(
            f"{dimension}: "
            f"q{int(NORMALIZATION_QUANTILE * 100)} "
            f"= {params['scale']:.6f}"
        )

    # BUILD FINAL CI
    ci_df = (build_multidimensional_ci(raw_df, normalization_params))
    ci_df = add_labels(ci_df, labels_df)

    # VALIDATE
    validate_ci(ci_df)

    # SUMMARY
    summary_df = (build_ci_summary(ci_df))
    correlation_df = (build_correlation_matrix(ci_df))
    label_diagnostics_df = ( build_label_diagnostics(ci_df))

    ci_df.to_csv(os.path.join(output_dir, "CI_SESSION_SCORES.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "CI_SUMMARY.csv"), index=False)
    correlation_df.to_csv(os.path.join(output_dir, "CI_SUBINDEX_CORRELATIONS.csv"), index=False)

    if (label_diagnostics_df is not None):
        label_diagnostics_df.to_csv(os.path.join(output_dir, "CI_LABEL_DIAGNOSTICS.csv"), index=False)

    # NORMALIZATION PARAMETERS
    normalization_output = {
        "normalization_quantile": NORMALIZATION_QUANTILE,
        "epistemic_scale": normalization_params["epistemic"]["scale"],
        "hedging_scale": normalization_params["hedging"]["scale"],
        "entrainment_scale": normalization_params["entrainment"]["scale"],
    }

    with open(os.path.join(output_dir, "CI_NORMALIZATION.json"), "w", encoding="utf-8") as f:
        json.dump(normalization_output, f, indent=4)

    definition = {
        "ci_name": "Multidimensional Conditioning Index",
        "dimensions": ["alignment", "epistemic", "hedging", "entrainment",],
        "weights": DIMENSION_WEIGHTS,
        "early_fraction": EARLY_FRACTION,
        "late_fraction": LATE_FRACTION,
        "minimum_window_points": MIN_WINDOW_POINTS,
        "alignment_threshold": ALIGNMENT_THRESHOLD,
        "mad_scale": MAD_SCALE,
        "epsilon_min": EPSILON_MIN,
        "normalization_quantile": NORMALIZATION_QUANTILE,
        "normalization_parameters": normalization_params,
        "formula": ("CI = (0.25*sqrt(C_alignment) + 0.25*sqrt(C_epistemic) + 0.25*sqrt(C_hedging) + 0.25*sqrt(C_entrainment))^2"),
        "aggregation": "generalized_mean",
        "aggregation_p" : 0.5,
        "subindex_definitions": {
            "alignment": ("terminal median alignment * persistence above 0.25"),
            "epistemic": ("normalized persistent increase in epistemic uncertainty relative to early conversational baseline"),
            "hedging": ("normalized persistent reduction in hedging relative to early conversational baseline"),
            "entrainment":("normalized persistent positive baseline-adjusted structural entrainment"),
        }
    }

    with open(os.path.join(output_dir, "CI_DEFINITION.json"), "w", encoding="utf-8") as f:
        json.dump(definition, f, indent=4)

    print("\n" + "=" * 70)
    print("MULTIDIMENSIONAL CI SUMMARY")
    print("=" * 70)

    print(summary_df.to_string(index=False))
    print("\n" + "=" * 70)

    print("SPEARMAN CORRELATIONS")
    print( "=" * 70)
    print(correlation_df.to_string())

    if (label_diagnostics_df is not None):
        print("\n" + "=" * 70)
        print("DIAGNOSTIC COMPARISON BY CONDITIONING LABEL" )
        print("=" * 70)
        print(label_diagnostics_df.to_string(index=False))

    print(f"\nResults saved in:\n{output_dir}")

if __name__ == "__main__":
    main()