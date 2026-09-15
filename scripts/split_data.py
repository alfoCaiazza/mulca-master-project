import pandas as pd
import os

def split_dataframe(df: pd.DataFrame, split_attribute: str, output_dest: str):
    os.makedirs(output_dest, exist_ok=True)

    # Split by the specified attribute
    for value in df[split_attribute].unique().dropna():
        subset = df[df[split_attribute] == value]

        filename = f"{value}.csv"
        filepath = os.path.join(output_dest, filename)

        subset.to_csv(filepath, index=False)

    print(f"Saved {df[split_attribute].nunique()} CSV files to {output_dest}")

if __name__ == "__main__":
    output_path = "../data/splits"
    print("Splitting the dataframe ...")

    df = pd.read_csv('../data/processed/ENRICHED_WITH_EMBEDDINGS.csv')
    split_dataframe(df, "category", output_path)
    