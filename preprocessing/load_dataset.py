import os
import glob
import pandas as pd

RAW_DATA_PATH = "datasets/raw"


def load_all_datasets():
    csv_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No CSV files found inside datasets/raw/"
        )

    print(f"\nFound {len(csv_files)} CSV files.\n")

    dataframes = []

    for file in sorted(csv_files):
        print(f"Loading: {os.path.basename(file)}")

        df = pd.read_csv(
            file,
            low_memory=False,
            encoding="utf-8"
        )

        dataframes.append(df)

    return dataframes


if __name__ == "__main__":

    dfs = load_all_datasets()

    merged_preview = pd.concat(
        dfs,
        ignore_index=True
    )

    print("\nDataset Loaded Successfully!")
    print("-" * 50)

    print("Shape :", merged_preview.shape)

    print("\nColumns:")
    print(list(merged_preview.columns))

    print("\nFirst 5 Rows:")
    print(merged_preview.head())

    print("\nMissing Values:")
    print(merged_preview.isnull().sum())