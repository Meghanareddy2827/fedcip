import os
import glob
import numpy as np
import pandas as pd

RAW_DATA_PATH = "datasets/raw"
OUTPUT_PATH = "datasets/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def clean_dataframe(df):

    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Replace infinity values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove rows containing missing values
    df.dropna(inplace=True)

    # Remove columns with only one unique value
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df.drop(columns=constant_cols, inplace=True)

    return df


def clean_all_datasets():

    csv_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.csv"))

    print(f"\nFound {len(csv_files)} CSV files.\n")

    for file in sorted(csv_files):

        filename = os.path.basename(file)

        print("=" * 70)
        print(f"Cleaning: {filename}")

        df = pd.read_csv(file, low_memory=False)

        print("Original Shape :", df.shape)

        df = clean_dataframe(df)

        print("Cleaned Shape  :", df.shape)

        output_file = os.path.join(
            OUTPUT_PATH,
            filename.replace(".csv", "_clean.csv")
        )

        df.to_csv(output_file, index=False)

        print("Saved ->", output_file)

    print("\nAll datasets cleaned successfully.")


if __name__ == "__main__":
    clean_all_datasets()