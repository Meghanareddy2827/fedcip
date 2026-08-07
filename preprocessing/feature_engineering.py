import pandas as pd
import numpy as np
import os

INPUT_FILE = "datasets/processed/merged_dataset.csv"
OUTPUT_FILE = "datasets/processed/engineered_dataset.csv"


def feature_engineering():

    print("Loading merged dataset...")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Replace infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove rows with missing values
    df.dropna(inplace=True)

    # Keep Label column separately
    label_column = df["Label"]

    # Remove Label temporarily
    features = df.drop(columns=["Label"])

    # Convert object columns to numeric where possible
    for col in features.columns:
        if features[col].dtype == object:
            features[col] = pd.to_numeric(features[col], errors="coerce")

    # Fill any remaining NaN values
    features = features.fillna(0)

    # Remove constant columns
    constant_cols = [
        c for c in features.columns
        if features[c].nunique() <= 1
    ]

    features.drop(columns=constant_cols, inplace=True)

    # Add Label back
    features["Label"] = label_column.values

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature engineering completed.")
    print("Final Shape:", features.shape)
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    feature_engineering()