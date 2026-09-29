import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SUBINDEX_COLUMNS = [
    "C_alignment",
    "C_epistemic",
    "C_hedging",
    "C_entrainment",
]

DIMENSION_NAMES = [
    "alignment",
    "epistemic",
    "hedging",
    "entrainment",
]

N_DIMENSIONS = len(SUBINDEX_COLUMNS)


# ------------------------------------------------------------
# Generalized mean sensitivity
#
# p = 1     -> arithmetic mean
# p -> 0    -> geometric mean
#
# Lower p means lower compensability.
# ------------------------------------------------------------

GENERALIZED_MEAN_P = [
    0.75,
    0.50,
    0.25,
]


# ------------------------------------------------------------
# Thresholds for active-dimension criteria
# ------------------------------------------------------------

ACTIVE_THRESHOLDS = [
    0.10,
    0.20,
    0.25,
    0.30,
]


# ------------------------------------------------------------
# Perturbation robustness
# ------------------------------------------------------------

PERTURBATION_SIGMAS = [
    0.01,
    0.025,
    0.05,
]

N_PERTURBATIONS = 500

RANDOM_STATE = 42

ZERO_TOL = 1e-12


# ============================================================
# INPUT
# ============================================================

def validate_input(df):

    missing = [
        column
        for column in SUBINDEX_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required sub-index columns: {missing}"
        )

    X = (
        df[SUBINDEX_COLUMNS]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    if X.isna().any().any():

        raise ValueError(
            "Sub-index matrix contains missing "
            "or non-numeric values."
        )

    outside = (
        (X < 0)
        | (X > 1)
    )

    if outside.any().any():

        raise ValueError(
            "All sub-indices must lie in [0,1]."
        )

    return X.astype(float)


# ============================================================
# AGGREGATION OPERATORS
# ============================================================

def generalized_mean(X, p):
    """
    Generalized/power mean:

        G_p(x) =
            [mean(x^p)]^(1/p)

    Valid here for p > 0.
    """

    if p <= 0:

        raise ValueError(
            "Use geometric_mean() for p = 0."
        )

    return np.power(
        np.mean(
            np.power(X, p),
            axis=1,
        ),
        1.0 / p,
    )


def geometric_mean(X):
    """
    Geometric mean.

    If any dimension is exactly zero:

        G_0 = 0
    """

    result = np.zeros(
        X.shape[0],
        dtype=float,
    )

    positive_rows = np.all(
        X > 0,
        axis=1,
    )

    if positive_rows.any():

        result[
            positive_rows
        ] = np.exp(
            np.mean(
                np.log(
                    X[
                        positive_rows
                    ]
                ),
                axis=1,
            )
        )

    return result


def compute_operators(X):
    """
    Compute all aggregation strategies.

    No existing conditioning_index column is used.
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    results = {}

    # ========================================================
    # Fully compensatory reference
    # ========================================================

    arithmetic = np.mean(
        X,
        axis=1,
    )

    results[
        "CI_arithmetic"
    ] = arithmetic


    # ========================================================
    # Generalized means
    # ========================================================

    for p in GENERALIZED_MEAN_P:

        suffix = str(p).replace(
            ".",
            ""
        )

        results[
            f"CI_generalized_p{suffix}"
        ] = generalized_mean(
            X,
            p,
        )


    # ========================================================
    # Geometric mean
    # ========================================================

    results[
        "CI_geometric"
    ] = geometric_mean(
        X
    )


    # ========================================================
    # Minimum
    # ========================================================

    results[
        "CI_minimum"
    ] = np.min(
        X,
        axis=1,
    )


    # ========================================================
    # Active-dimension criteria
    # ========================================================

    for tau in ACTIVE_THRESHOLDS:

        tau_name = (
            f"{int(round(tau * 100)):02d}"
        )

        K = np.sum(
            X >= tau,
            axis=1,
        )

        active_fraction = (
            K / N_DIMENSIONS
        )

        # Pure dimensional breadth:
        #
        # 0 dimensions active -> 0
        # 4 dimensions active -> 1

        results[
            f"CI_active_count_t{tau_name}"
        ] = active_fraction


        # Magnitude × dimensional breadth

        results[
            f"CI_active_adjusted_t{tau_name}"
        ] = (
            arithmetic
            * active_fraction
        )

    return results


# ============================================================
# PROFILE MULTIDIMENSIONALITY
# ============================================================

def compute_profile_diagnostics(X):
    """
    Diagnostics describing how distributed the four
    sub-indices are within each conversation.

    profile_entropy:
        1 -> perfectly balanced across dimensions
        0 -> concentrated in one dimension

    max_dimension_share:
        0.25 -> perfectly balanced
        1    -> one-dimensional profile
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    totals = X.sum(
        axis=1
    )

    shares = np.zeros_like(
        X,
        dtype=float,
    )

    valid = totals > 0

    shares[
        valid
    ] = (
        X[valid]
        /
        totals[
            valid,
            None,
        ]
    )

    safe_shares = np.where(
        shares > 0,
        shares,
        1.0,
    )

    entropy = -np.sum(
        np.where(
            shares > 0,
            shares
            * np.log(
                safe_shares
            ),
            0.0,
        ),
        axis=1,
    )

    entropy = (
        entropy
        / np.log(
            N_DIMENSIONS
        )
    )

    entropy[
        ~valid
    ] = np.nan

    max_share = np.max(
        shares,
        axis=1,
    )

    max_share[
        ~valid
    ] = np.nan

    return (
        entropy,
        max_share,
    )


# ============================================================
# DISTRIBUTION DIAGNOSTICS
# ============================================================

def tie_observation_rate(values):
    """
    Fraction of observations belonging to an exact
    numerical tie after rounding to 12 decimals.
    """

    rounded = pd.Series(
        np.round(
            values,
            12,
        )
    )

    counts = rounded.value_counts()

    tied_values = counts[
        counts > 1
    ].index

    if len(tied_values) == 0:
        return 0.0

    return float(
        rounded.isin(
            tied_values
        ).mean()
    )


def distribution_summary(
    operators
):

    rows = []

    for name, values in (
        operators.items()
    ):

        values = np.asarray(
            values,
            dtype=float,
        )

        series = pd.Series(
            values
        )

        rows.append({

            "operator":
                name,

            "n":
                len(values),

            "mean":
                series.mean(),

            "std":
                series.std(
                    ddof=1
                ),

            "min":
                series.min(),

            "q25":
                series.quantile(
                    0.25
                ),

            "median":
                series.median(),

            "q75":
                series.quantile(
                    0.75
                ),

            "max":
                series.max(),

            "iqr":
                (
                    series.quantile(
                        0.75
                    )
                    -
                    series.quantile(
                        0.25
                    )
                ),

            "zero_rate":
                float(
                    np.mean(
                        np.isclose(
                            values,
                            0.0,
                            atol=ZERO_TOL,
                        )
                    )
                ),

            "n_unique":
                int(
                    pd.Series(
                        np.round(
                            values,
                            12,
                        )
                    )
                    .nunique()
                ),

            "unique_fraction":
                float(
                    pd.Series(
                        np.round(
                            values,
                            12,
                        )
                    )
                    .nunique()
                    /
                    len(values)
                ),

            "tie_observation_rate":
                tie_observation_rate(
                    values
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# RANK CORRELATIONS
# ============================================================

def safe_spearman(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    if valid.sum() < 3:

        return np.nan

    if (
        np.std(
            x[valid]
        )
        == 0
        or
        np.std(
            y[valid]
        )
        == 0
    ):

        return np.nan

    return float(
        spearmanr(
            x[valid],
            y[valid],
        ).statistic
    )


def build_rank_correlation_matrix(
    operators
):

    names = list(
        operators.keys()
    )

    matrix = pd.DataFrame(
        index=names,
        columns=names,
        dtype=float,
    )

    for name_a in names:

        for name_b in names:

            matrix.loc[
                name_a,
                name_b
            ] = safe_spearman(
                operators[
                    name_a
                ],
                operators[
                    name_b
                ],
            )

    return matrix


# ============================================================
# COMPENSATION / JOINTNESS DIAGNOSTICS
# ============================================================

def build_compensation_diagnostics(
    operators,
    profile_entropy,
    max_dimension_share,
):
    """
    Compare conjunctive operators with the arithmetic mean.

    ratio =
        CI_operator / CI_arithmetic

    A strongly conjunctive operator should penalize
    unbalanced profiles more strongly.

    Therefore we expect:

        rho(profile_entropy, ratio) > 0

    and:

        rho(max_dimension_share, ratio) < 0
    """

    arithmetic = operators[
        "CI_arithmetic"
    ]

    rows = []

    # Pure active-count scores are excluded from ratio analysis
    # because they measure breadth rather than magnitude.

    names = [
        name
        for name in operators
        if "active_count" not in name
    ]

    for name in names:

        values = operators[
            name
        ]

        valid = (
            arithmetic > ZERO_TOL
        )

        ratio = np.full(
            len(arithmetic),
            np.nan,
            dtype=float,
        )

        ratio[
            valid
        ] = (
            values[
                valid
            ]
            /
            arithmetic[
                valid
            ]
        )

        rows.append({

            "operator":
                name,

            "mean_ratio_vs_arithmetic":
                float(
                    np.nanmean(
                        ratio
                    )
                ),

            "median_ratio_vs_arithmetic":
                float(
                    np.nanmedian(
                        ratio
                    )
                ),

            "mean_penalty_vs_arithmetic":
                float(
                    1.0
                    -
                    np.nanmean(
                        ratio
                    )
                ),

            "rho_entropy_ratio":
                safe_spearman(
                    profile_entropy,
                    ratio,
                ),

            "rho_maxshare_ratio":
                safe_spearman(
                    max_dimension_share,
                    ratio,
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# ACTIVE DIMENSION ANALYSIS
# ============================================================

def build_active_dimension_analysis(
    X,
    arithmetic,
):

    rows = []

    for tau in ACTIVE_THRESHOLDS:

        K = np.sum(
            X >= tau,
            axis=1,
        )

        for k in range(
            N_DIMENSIONS + 1
        ):

            mask = (
                K == k
            )

            n = int(
                mask.sum()
            )

            rows.append({

                "threshold":
                    tau,

                "active_dimensions":
                    k,

                "n":
                    n,

                "proportion":
                    n / len(X),

                "arithmetic_mean":
                    (
                        float(
                            arithmetic[
                                mask
                            ].mean()
                        )
                        if n > 0
                        else np.nan
                    ),

                "arithmetic_median":
                    (
                        float(
                            np.median(
                                arithmetic[
                                    mask
                                ]
                            )
                        )
                        if n > 0
                        else np.nan
                    ),
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# SYNTHETIC COMPENSATION STRESS TEST
# ============================================================

def build_stress_test():
    """
    Theoretical profiles used to illustrate how each
    aggregation operator reacts to concentration vs
    multidimensional balance.
    """

    profiles = {

        "all_zero":
            [0.0, 0.0, 0.0, 0.0],

        "one_dimension":
            [1.0, 0.0, 0.0, 0.0],

        "two_dimensions":
            [1.0, 1.0, 0.0, 0.0],

        "three_dimensions":
            [1.0, 1.0, 1.0, 0.0],

        "all_dimensions":
            [1.0, 1.0, 1.0, 1.0],

        # Same arithmetic mean = 0.5
        "balanced_half":
            [0.5, 0.5, 0.5, 0.5],

        "unbalanced_same_mean":
            [1.0, 1.0, 0.0, 0.0],

        "dominant_profile":
            [1.0, 0.1, 0.1, 0.1],

        "balanced_low":
            [0.2, 0.2, 0.2, 0.2],

        "balanced_mid":
            [0.4, 0.4, 0.4, 0.4],
    }

    names = list(
        profiles.keys()
    )

    X = np.asarray(
        [
            profiles[name]
            for name in names
        ],
        dtype=float,
    )

    operators = compute_operators(
        X
    )

    output = pd.DataFrame({
        "profile":
            names,

        "C_alignment":
            X[:, 0],

        "C_epistemic":
            X[:, 1],

        "C_hedging":
            X[:, 2],

        "C_entrainment":
            X[:, 3],
    })

    for name, values in (
        operators.items()
    ):

        output[
            name
        ] = values

    return output


# ============================================================
# PERTURBATION ROBUSTNESS
# ============================================================

def perturbation_robustness(
    X,
):
    """
    Add small Gaussian perturbations to the four sub-indices:

        C'_d = clip(C_d + epsilon, 0, 1)

    and measure Spearman rank stability of each aggregation
    operator.

    No re-fitting or re-normalization occurs.
    """

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    baseline_operators = (
        compute_operators(
            X
        )
    )

    rows = []

    for sigma in PERTURBATION_SIGMAS:

        correlations = {
            name: []
            for name
            in baseline_operators
        }

        for _ in range(
            N_PERTURBATIONS
        ):

            noise = rng.normal(
                loc=0.0,
                scale=sigma,
                size=X.shape,
            )

            perturbed_X = np.clip(
                X + noise,
                0.0,
                1.0,
            )

            perturbed_operators = (
                compute_operators(
                    perturbed_X
                )
            )

            for name in (
                baseline_operators
            ):

                rho = safe_spearman(
                    baseline_operators[
                        name
                    ],
                    perturbed_operators[
                        name
                    ],
                )

                if np.isfinite(
                    rho
                ):

                    correlations[
                        name
                    ].append(
                        rho
                    )

        for name, values in (
            correlations.items()
        ):

            values = np.asarray(
                values,
                dtype=float,
            )

            rows.append({

                "operator":
                    name,

                "noise_sigma":
                    sigma,

                "n_valid":
                    len(values),

                "spearman_mean":
                    (
                        float(
                            values.mean()
                        )
                        if len(values)
                        else np.nan
                    ),

                "spearman_std":
                    (
                        float(
                            values.std(
                                ddof=1
                            )
                        )
                        if len(values) > 1
                        else np.nan
                    ),

                "spearman_q025":
                    (
                        float(
                            np.quantile(
                                values,
                                0.025
                            )
                        )
                        if len(values)
                        else np.nan
                    ),

                "spearman_median":
                    (
                        float(
                            np.median(
                                values
                            )
                        )
                        if len(values)
                        else np.nan
                    ),

                "spearman_q975":
                    (
                        float(
                            np.quantile(
                                values,
                                0.975
                            )
                        )
                        if len(values)
                        else np.nan
                    ),
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# SUBGROUP ROBUSTNESS
# ============================================================

def build_subgroup_diagnostics(
    source_df,
    operators,
):

    grouping_columns = [
        column
        for column in [
            "model",
            "claim_category",
            "claim",
        ]
        if column in source_df.columns
    ]

    rows = []

    arithmetic = operators[
        "CI_arithmetic"
    ]

    for group_column in grouping_columns:

        for group_value, indices in (
            source_df
            .groupby(
                group_column
            )
            .groups
            .items()
        ):

            indices = np.asarray(
                list(indices),
                dtype=int,
            )

            for operator_name, values in (
                operators.items()
            ):

                subset = values[
                    indices
                ]

                rows.append({

                    "grouping_variable":
                        group_column,

                    "group":
                        group_value,

                    "operator":
                        operator_name,

                    "n":
                        len(indices),

                    "mean":
                        float(
                            np.mean(
                                subset
                            )
                        ),

                    "median":
                        float(
                            np.median(
                                subset
                            )
                        ),

                    "std":
                        (
                            float(
                                np.std(
                                    subset,
                                    ddof=1
                                )
                            )
                            if len(subset) > 1
                            else np.nan
                        ),

                    "zero_rate":
                        float(
                            np.mean(
                                np.isclose(
                                    subset,
                                    0.0,
                                    atol=ZERO_TOL,
                                )
                            )
                        ),

                    "rho_vs_arithmetic":
                        safe_spearman(
                            arithmetic[
                                indices
                            ],
                            subset,
                        ),
                })

    return pd.DataFrame(
        rows
    )


# ============================================================
# SESSION OUTPUT
# ============================================================

def build_session_output(
    source_df,
    X,
    operators,
):

    metadata_columns = [
        column
        for column in [
            "session_id",
            "model",
            "claim",
            "claim_category",
            "attacker_persona",
        ]
        if column in source_df.columns
    ]

    output = (
        source_df[
            metadata_columns
            +
            SUBINDEX_COLUMNS
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    profile_entropy, max_share = (
        compute_profile_diagnostics(
            X
        )
    )

    output[
        "profile_entropy"
    ] = profile_entropy

    output[
        "max_dimension_share"
    ] = max_share

    # --------------------------------------------------------
    # Dimension responsible for maximum sub-index
    # --------------------------------------------------------

    dominant_idx = np.argmax(
        X,
        axis=1,
    )

    total = X.sum(
        axis=1
    )

    output[
        "dominant_dimension"
    ] = [
        (
            DIMENSION_NAMES[idx]
            if total[i] > 0
            else "none"
        )
        for i, idx
        in enumerate(
            dominant_idx
        )
    ]

    # --------------------------------------------------------
    # Number of active dimensions
    # --------------------------------------------------------

    for tau in ACTIVE_THRESHOLDS:

        tau_name = (
            f"{int(round(tau * 100)):02d}"
        )

        output[
            f"K_active_t{tau_name}"
        ] = np.sum(
            X >= tau,
            axis=1,
        )

    # --------------------------------------------------------
    # Aggregation variants
    # --------------------------------------------------------

    for name, values in (
        operators.items()
    ):

        output[
            name
        ] = values

    return output


# ============================================================
# SANITY CHECK AGAINST EXISTING CI
# ============================================================

def compare_existing_ci(
    source_df,
    arithmetic,
):
    """
    Existing conditioning_index is never used as an input.

    If available, we only verify that it corresponds to the
    arithmetic baseline generated by the previous script.
    """

    if (
        "conditioning_index"
        not in source_df.columns
    ):

        return None

    old_ci = pd.to_numeric(
        source_df[
            "conditioning_index"
        ],
        errors="coerce",
    ).to_numpy()

    valid = np.isfinite(
        old_ci
    )

    if not valid.any():
        return None

    diff = np.abs(
        old_ci[
            valid
        ]
        -
        arithmetic[
            valid
        ]
    )

    return {
        "n_compared":
            int(
                valid.sum()
            ),

        "max_absolute_difference":
            float(
                diff.max()
            ),

        "mean_absolute_difference":
            float(
                diff.mean()
            ),
    }


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    script_dir = os.path.dirname(
        os.path.abspath(
            __file__
        )
    )

    default_results_dir = os.path.abspath(
        os.path.join(
            script_dir,
            "..",
            "..",
            "data",
            "generative",
            "results",
        )
    )

    parser = argparse.ArgumentParser(
        description=(
            "Sensitivity analysis for alternative "
            "multidimensional CI aggregation operators."
        )
    )

    parser.add_argument(
        "--input",
        default=os.path.join(
            default_results_dir,
            "conditioning_index",
            "CI_SESSION_SCORES.csv",
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            default_results_dir,
            "conditioning_index",
            "aggregation_sensitivity",
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # ========================================================
    # LOAD
    # ========================================================

    df = pd.read_csv(
        args.input
    )

    subindex_df = validate_input(
        df
    )

    X = subindex_df.to_numpy(
        dtype=float
    )

    print(
        "CI aggregation sensitivity analysis"
    )

    print(
        "Sessions:",
        len(df)
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Existing conditioning_index is NOT "
        "used to compute any alternative CI."
    )

    # ========================================================
    # OPERATORS
    # ========================================================

    operators = compute_operators(
        X
    )

    print(
        "\nOperators evaluated:"
    )

    for name in operators:
        print(
            " ",
            name
        )

    # ========================================================
    # PROFILE DIAGNOSTICS
    # ========================================================

    (
        profile_entropy,
        max_dimension_share,
    ) = compute_profile_diagnostics(
        X
    )

    # ========================================================
    # DISTRIBUTIONS
    # ========================================================

    distributions_df = (
        distribution_summary(
            operators
        )
    )

    # ========================================================
    # RANK CORRELATIONS
    # ========================================================

    rank_corr_df = (
        build_rank_correlation_matrix(
            operators
        )
    )

    # ========================================================
    # COMPENSATION DIAGNOSTICS
    # ========================================================

    compensation_df = (
        build_compensation_diagnostics(
            operators,
            profile_entropy,
            max_dimension_share,
        )
    )

    # ========================================================
    # ACTIVE DIMENSIONS
    # ========================================================

    active_df = (
        build_active_dimension_analysis(
            X,
            operators[
                "CI_arithmetic"
            ],
        )
    )

    # ========================================================
    # THEORETICAL STRESS TEST
    # ========================================================

    stress_df = (
        build_stress_test()
    )

    # ========================================================
    # PERTURBATION ROBUSTNESS
    # ========================================================

    print(
        "\nRunning perturbation robustness..."
    )

    perturbation_df = (
        perturbation_robustness(
            X
        )
    )

    # ========================================================
    # SUBGROUP DIAGNOSTICS
    # ========================================================

    subgroup_df = (
        build_subgroup_diagnostics(
            df.reset_index(
                drop=True
            ),
            operators,
        )
    )

    # ========================================================
    # SESSION-LEVEL OUTPUT
    # ========================================================

    session_output_df = (
        build_session_output(
            df,
            X,
            operators,
        )
    )

    # ========================================================
    # SANITY CHECK WITH OLD ARITHMETIC CI
    # ========================================================

    old_ci_check = compare_existing_ci(
        df,
        operators[
            "CI_arithmetic"
        ],
    )

    # ========================================================
    # SAVE
    # ========================================================

    session_output_df.to_csv(
        os.path.join(
            args.output_dir,
            "CI_AGGREGATION_SESSION_SCORES.csv",
        ),
        index=False,
    )

    distributions_df.to_csv(
        os.path.join(
            args.output_dir,
            "CI_AGGREGATION_DISTRIBUTIONS.csv",
        ),
        index=False,
    )

    rank_corr_df.to_csv(
        os.path.join(
            args.output_dir,
            "CI_AGGREGATION_RANK_CORRELATIONS.csv",
        )
    )

    compensation_df.to_csv(
        os.path.join(
            args.output_dir,
            "CI_COMPENSATION_DIAGNOSTICS.csv",
        ),
        index=False,
    )

    active_df.to_csv(
        os.path.join(
            args.output_dir,
            "CI_ACTIVE_DIMENSION_ANALYSIS.csv",
        ),
        index=False,
    )

    stress_df.to_csv(
        os.path.join(
            args.output_dir,
            "CI_AGGREGATION_STRESS_TEST.csv",
        ),
        index=False,
    )

    perturbation_df.to_csv(
        os.path.join(
            args.output_dir,
            "CI_PERTURBATION_ROBUSTNESS.csv",
        ),
        index=False,
    )

    subgroup_df.to_csv(
        os.path.join(
            args.output_dir,
            "CI_AGGREGATION_SUBGROUPS.csv",
        ),
        index=False,
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "analysis":
            "CI aggregation sensitivity",

        "input":
            os.path.abspath(
                args.input
            ),

        "n_sessions":
            len(df),

        "subindices":
            SUBINDEX_COLUMNS,

        "existing_conditioning_index_used":
            False,

        "generalized_mean_p":
            GENERALIZED_MEAN_P,

        "active_thresholds":
            ACTIVE_THRESHOLDS,

        "perturbation_sigmas":
            PERTURBATION_SIGMAS,

        "n_perturbations":
            N_PERTURBATIONS,

        "random_state":
            RANDOM_STATE,

        "automatic_winner_selection":
            False,

        "existing_ci_sanity_check":
            old_ci_check,

        "operator_interpretation": {

            "CI_arithmetic":
                (
                    "Fully compensatory arithmetic mean."
                ),

            "generalized_means":
                (
                    "Progressively lower compensability "
                    "as p approaches zero."
                ),

            "CI_geometric":
                (
                    "Strongly conjunctive; equals zero "
                    "if any sub-index equals zero."
                ),

            "CI_minimum":
                (
                    "Strict conjunctive lower-bound "
                    "operator."
                ),

            "CI_active_count":
                (
                    "Fraction of dimensions above an "
                    "activation threshold."
                ),

            "CI_active_adjusted":
                (
                    "Arithmetic magnitude multiplied by "
                    "fraction of simultaneously active "
                    "dimensions."
                ),
        },
    }

    with open(
        os.path.join(
            args.output_dir,
            "CI_AGGREGATION_METADATA.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    # ========================================================
    # CONSOLE REPORT
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "AGGREGATION DISTRIBUTIONS"
    )

    print(
        "=" * 90
    )

    print(
        distributions_df[
            [
                "operator",
                "mean",
                "median",
                "std",
                "zero_rate",
                "unique_fraction",
                "tie_observation_rate",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "COMPENSATION / JOINTNESS DIAGNOSTICS"
    )

    print(
        "=" * 90
    )

    print(
        compensation_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "ACTIVE DIMENSION DISTRIBUTION"
    )

    print(
        "=" * 90
    )

    print(
        active_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "THEORETICAL STRESS TEST"
    )

    print(
        "=" * 90
    )

    print(
        stress_df.to_string(
            index=False
        )
    )

    if old_ci_check is not None:

        print(
            "\n"
            + "=" * 90
        )

        print(
            "SANITY CHECK: EXISTING CI VS ARITHMETIC BASELINE"
        )

        print(
            "=" * 90
        )

        print(
            "Max absolute difference:",
            old_ci_check[
                "max_absolute_difference"
            ]
        )

        print(
            "Mean absolute difference:",
            old_ci_check[
                "mean_absolute_difference"
            ]
        )

    print(
        "\nNo aggregation operator has been "
        "automatically selected."
    )

    print(
        "\nResults saved in:"
    )

    print(
        args.output_dir
    )


if __name__ == "__main__":
    main()