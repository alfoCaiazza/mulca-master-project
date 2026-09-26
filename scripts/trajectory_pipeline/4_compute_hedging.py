import re
import pandas as pd
import os

HEDGE_PATTERNS = [
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bpossibly\b",
    r"\bprobably\b",
    r"\bseems?\b",
    r"\bappears?\b",
    r"\bi think\b",
    r"\bi believe\b",
    r"\bi guess\b",
    r"\bi'm not sure\b",
    r"\bnot entirely clear\b",
    r"\bnot necessarily\b",
    r"\bit depends\b",
    r"\bcould\b",
    r"\bmight\b",
    r"\bmay\b",
    r"\barguably\b",
]


def compute_hedging_score(text):
    if pd.isna(text):
        return float("nan")

    text = str(text).lower()
    words = re.findall(r"\b\w+\b", text)

    if not words:
        return 0.0

    hedge_count = sum(
        len(re.findall(pattern, text))
        for pattern in HEDGE_PATTERNS
    )

    # Normalize by response length
    return hedge_count / len(words)


def add_hedging_trajectory(trajectory_df):
    df = trajectory_df.copy()

    df["hedge_count"] = df["text"].apply(
        lambda x: (
            float("nan")
            if pd.isna(x)
            else sum(
                len(re.findall(pattern, str(x).lower()))
                for pattern in HEDGE_PATTERNS
            )
        )
    )

    df["hedging_score"] = df["text"].apply(
        compute_hedging_score
    )

    return df

def main():
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))
    trajectory_dir = os.path.join(main_dir, "TRAJECTORY.csv")
    trajectory_df = pd.read_csv(trajectory_dir)

    print("Computing hedging trajectory ...")
    trajectory_df = add_hedging_trajectory(trajectory_df)
    trajectory_df.to_csv(trajectory_dir, index=False)
    print("Operation completed successfully!")
    

if __name__ == "__main__":
    main()