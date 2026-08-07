import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

INPUT_FILE = "datasets/processed/engineered_dataset.csv"
OUTPUT_FILE = "datasets/processed/final_dataset.csv"

# Number of top features to keep
TOP_FEATURES = 30


def feature_selection():

    print("Loading engineered dataset...")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    df.columns = df.columns.str.strip()

    X = df.drop(columns=["Label"])
    y = df["Label"]

    # Encode target labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    print("\nTraining CatBoost for feature importance...")

    model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        verbose=False,
        random_seed=42
    )

    model.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.get_feature_importance()
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop Selected Features:\n")
    print(importance.head(TOP_FEATURES))

    selected_features = importance.head(TOP_FEATURES)["Feature"].tolist()

    final_df = df[selected_features].copy()
    final_df["Label"] = df["Label"]

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature Selection Completed")
    print("Selected Features :", len(selected_features))
    print("Final Dataset Shape :", final_df.shape)
    print("Saved to :", OUTPUT_FILE)


if __name__ == "__main__":
    feature_selection()