import pandas as pd
from pathlib import Path


RAW_DIR = Path("datasets/raw")


def inspect_file(file_path):
    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    df = pd.read_csv(file_path, low_memory=False)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns:")
    for i, column in enumerate(df.columns):
        print(f"{i}: {repr(column)}")

    print("\nLabel distribution:")

    # CICIDS2017 normally uses the "Label" column
    label_column = None

    for column in df.columns:
        if column.strip().lower() == "label":
            label_column = column
            break

    if label_column:
        print(df[label_column].value_counts(dropna=False))
    else:
        print("WARNING: Label column not found!")

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values found.")
    else:
        print(missing)

    print("\nDuplicate rows:")
    print(df.duplicated().sum())


def main():
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print("ERROR: No CSV files found in datasets/raw/")
        return

    print(f"Found {len(csv_files)} CSV files.")

    for file_path in csv_files:
        inspect_file(file_path)


if __name__ == "__main__":
    main()