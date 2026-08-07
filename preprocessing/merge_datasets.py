import os
import glob
import pandas as pd

PROCESSED_PATH = "datasets/processed"
OUTPUT_FILE = os.path.join(PROCESSED_PATH, "merged_dataset.csv")


def merge_datasets():

    csv_files = sorted(
        glob.glob(os.path.join(PROCESSED_PATH, "*_clean.csv"))
    )

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No cleaned datasets found in datasets/processed/"
        )

    print(f"\nFound {len(csv_files)} cleaned datasets.\n")

    merged_df = []

    for file in csv_files:

        print(f"Adding: {os.path.basename(file)}")

        df = pd.read_csv(file, low_memory=False)

        merged_df.append(df)

    merged_df = pd.concat(
        merged_df,
        ignore_index=True
    )

    print("\nFinal Shape :", merged_df.shape)

    merged_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nMerged dataset saved at:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    merge_datasets()