import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# FEDCIP - DATA PREPARATION AND 3-CLIENT SPLITTING
# ============================================================

RAW_DIR = "datasets/raw"
PROCESSED_DIR = "datasets/processed"
CLIENT_DIR = "datasets/processed/clients"

RANDOM_SEED = 42

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CLIENT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD RAW CICIDS2017 DATA
# ============================================================

def load_raw_data():

    files = sorted(
        glob.glob(
            os.path.join(RAW_DIR, "*.csv")
        )
    )

    if not files:
        raise FileNotFoundError(
            "No CSV files found in datasets/raw/"
        )

    print("=" * 70)
    print("LOADING RAW CICIDS2017 DATA")
    print("=" * 70)

    dataframes = []

    for file in files:

        print("Loading:", os.path.basename(file))

        df = pd.read_csv(
            file,
            low_memory=False,
            encoding="utf-8"
        )

        df.columns = df.columns.str.strip()

        dataframes.append(df)

    combined = pd.concat(
        dataframes,
        ignore_index=True
    )

    print("\nRaw combined shape:", combined.shape)

    return combined


# ============================================================
# 2. CLEAN DATA
# ============================================================

def clean_data(df):

    print("\n" + "=" * 70)
    print("CLEANING DATA")
    print("=" * 70)

    before_duplicates = len(df)

    df = df.drop_duplicates()

    print(
        "Duplicates removed:",
        before_duplicates - len(df)
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    feature_columns = [
        column
        for column in df.columns
        if column != "Label"
    ]

    for column in feature_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    before_missing = len(df)

    df = df.dropna()

    print(
        "Rows removed because of missing/invalid values:",
        before_missing - len(df)
    )

    df = df.reset_index(drop=True)

    print(
        "Cleaned dataset shape:",
        df.shape
    )

    return df


# ============================================================
# 3. NORMALIZE LABEL NAMES
# ============================================================

def normalize_labels(df):

    print("\n" + "=" * 70)
    print("NORMALIZING LABELS")
    print("=" * 70)

    label_mapping = {

        "Web Attack � Brute Force":
            "Web Attack - Brute Force",

        "Web Attack � XSS":
            "Web Attack - XSS",

        "Web Attack � Sql Injection":
            "Web Attack - Sql Injection",

        "Web Attack – Brute Force":
            "Web Attack - Brute Force",

        "Web Attack – XSS":
            "Web Attack - XSS",

        "Web Attack – Sql Injection":
            "Web Attack - Sql Injection"
    }

    df["Label"] = df["Label"].replace(
        label_mapping
    )

    print("\nLabels found:")

    print(
        df["Label"]
        .value_counts()
        .sort_index()
    )

    return df


# ============================================================
# 4. GLOBAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def create_global_split(df):

    print("\n" + "=" * 70)
    print("CREATING GLOBAL TRAIN / VALIDATION / TEST")
    print("=" * 70)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["Label"]
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["Label"]
    )

    print("\nGlobal dataset sizes:")

    print("Train      :", train_df.shape)
    print("Validation :", validation_df.shape)
    print("Test       :", test_df.shape)

    train_df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "global_train.csv"
        ),
        index=False
    )

    validation_df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "global_validation.csv"
        ),
        index=False
    )

    test_df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "global_test.csv"
        ),
        index=False
    )

    return train_df, validation_df, test_df


# ============================================================
# 5. CLIENT DISTRIBUTION DESIGN
# ============================================================

CLIENT_RATIOS = {

    # -------------------------
    # BENIGN
    # -------------------------
    "BENIGN": {
        "client_1": 0.60,
        "client_2": 0.25,
        "client_3": 0.15
    },

    # -------------------------
    # DoS
    # -------------------------
    "DoS Hulk": {
        "client_1": 0.70,
        "client_2": 0.20,
        "client_3": 0.10
    },

    "DoS GoldenEye": {
        "client_1": 0.70,
        "client_2": 0.20,
        "client_3": 0.10
    },

    "DoS slowloris": {
        "client_1": 0.70,
        "client_2": 0.20,
        "client_3": 0.10
    },

    "DoS Slowhttptest": {
        "client_1": 0.70,
        "client_2": 0.20,
        "client_3": 0.10
    },

    # -------------------------
    # DDoS
    # -------------------------
    "DDoS": {
        "client_1": 0.20,
        "client_2": 0.10,
        "client_3": 0.70
    },

    # -------------------------
    # PortScan
    # -------------------------
    "PortScan": {
        "client_1": 0.10,
        "client_2": 0.10,
        "client_3": 0.80
    },

    # -------------------------
    # Web Attacks
    # -------------------------
    "Web Attack - Brute Force": {
        "client_1": 0.10,
        "client_2": 0.80,
        "client_3": 0.10
    },

    "Web Attack - XSS": {
        "client_1": 0.10,
        "client_2": 0.80,
        "client_3": 0.10
    },

    "Web Attack - Sql Injection": {
        "client_1": 0.10,
        "client_2": 0.80,
        "client_3": 0.10
    },

    # -------------------------
    # Infiltration
    # -------------------------
    "Infiltration": {
        "client_1": 0.00,
        "client_2": 1.00,
        "client_3": 0.00
    },

    # -------------------------
    # Bot
    # -------------------------
    "Bot": {
        "client_1": 0.10,
        "client_2": 0.10,
        "client_3": 0.80
    },

    # -------------------------
    # FTP Patator
    # -------------------------
    "FTP-Patator": {
        "client_1": 0.10,
        "client_2": 0.10,
        "client_3": 0.80
    },

    # -------------------------
    # SSH Patator
    # -------------------------
    "SSH-Patator": {
        "client_1": 0.10,
        "client_2": 0.10,
        "client_3": 0.80
    },

    # -------------------------
    # Heartbleed
    # -------------------------
    "Heartbleed": {
        "client_1": 0.00,
        "client_2": 0.00,
        "client_3": 1.00
    }
}


# ============================================================
# 6. SPLIT TRAINING DATA INTO CLIENTS
# ============================================================

def split_clients(train_df):

    print("\n" + "=" * 70)
    print("CREATING 3 FEDERATED CLIENTS")
    print("=" * 70)

    client_data = {
        "client_1": [],
        "client_2": [],
        "client_3": []
    }

    for label, class_df in train_df.groupby(
        "Label",
        sort=False
    ):

        class_df = class_df.sample(
            frac=1,
            random_state=RANDOM_SEED
        ).reset_index(drop=True)

        if label in CLIENT_RATIOS:

            ratios = CLIENT_RATIOS[label]

        else:

            ratios = {
                "client_1": 1 / 3,
                "client_2": 1 / 3,
                "client_3": 1 / 3
            }

        total = len(class_df)

        n1 = int(
            total * ratios["client_1"]
        )

        n2 = int(
            total * ratios["client_2"]
        )

        client_1_part = class_df.iloc[
            :n1
        ]

        client_2_part = class_df.iloc[
            n1:n1 + n2
        ]

        client_3_part = class_df.iloc[
            n1 + n2:
        ]

        if len(client_1_part) > 0:
            client_data["client_1"].append(
                client_1_part
            )

        if len(client_2_part) > 0:
            client_data["client_2"].append(
                client_2_part
            )

        if len(client_3_part) > 0:
            client_data["client_3"].append(
                client_3_part
            )

    clients = {}

    for client_name, parts in client_data.items():

        client_df = pd.concat(
            parts,
            ignore_index=True
        )

        client_df = client_df.sample(
            frac=1,
            random_state=RANDOM_SEED
        ).reset_index(drop=True)

        clients[client_name] = client_df

        output_file = os.path.join(
            CLIENT_DIR,
            f"{client_name}.csv"
        )

        client_df.to_csv(
            output_file,
            index=False
        )

        print(
            f"\n{client_name}:",
            client_df.shape
        )

        print(
            client_df["Label"]
            .value_counts()
            .sort_index()
        )

    return clients


# ============================================================
# 7. VERIFY CLIENT SPLIT
# ============================================================

def verify_clients(train_df, clients):

    print("\n" + "=" * 70)
    print("VERIFYING CLIENT SPLIT")
    print("=" * 70)

    total_client_rows = sum(
        len(client_df)
        for client_df in clients.values()
    )

    print(
        "Original training rows :",
        len(train_df)
    )

    print(
        "Client training rows   :",
        total_client_rows
    )

    if total_client_rows == len(train_df):

        print(
            "PASS: No training rows were lost."
        )

    else:

        print(
            "ERROR: Client row count does not match training data."
        )

    combined_clients = pd.concat(
        clients.values(),
        ignore_index=True
    )

    duplicate_count = (
        combined_clients
        .duplicated()
        .sum()
    )

    print(
        "Duplicate rows across clients:",
        duplicate_count
    )

    if duplicate_count == 0:

        print(
            "PASS: No duplicate training rows across clients."
        )

    else:

        print(
            "WARNING: Duplicate rows detected."
        )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    df = load_raw_data()

    df = clean_data(df)

    df = normalize_labels(df)

    train_df, validation_df, test_df = (
        create_global_split(df)
    )

    clients = split_clients(
        train_df
    )

    verify_clients(
        train_df,
        clients
    )

    print("\n" + "=" * 70)
    print("CLIENT SPLITTING COMPLETED")
    print("=" * 70)

    print("\nFiles created:")

    print(
        "datasets/processed/global_train.csv"
    )

    print(
        "datasets/processed/global_validation.csv"
    )

    print(
        "datasets/processed/global_test.csv"
    )

    print(
        "datasets/processed/clients/client_1.csv"
    )

    print(
        "datasets/processed/clients/client_2.csv"
    )

    print(
        "datasets/processed/clients/client_3.csv"
    )


if __name__ == "__main__":
    main()