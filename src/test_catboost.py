import os
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/saved_models/catboost_final_model.cbm"
TEST_PATH = "datasets/processed/test.csv"

# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("CATBOOST FINAL MODEL TEST")
print("=" * 70)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

print("\nModel found:", MODEL_PATH)
print("Test dataset found:", TEST_PATH)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading CatBoost model...")

model = CatBoostClassifier()
model.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Number of model features:", len(model.feature_names_))

# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_PATH)

print("Test dataset shape:", test_df.shape)

# ============================================================
# IDENTIFY TARGET COLUMN
# ============================================================

print("\nTest dataset columns:")
print(test_df.columns.tolist())

# Change this only if your target column has a different name.
TARGET_COLUMN = "Label"

if TARGET_COLUMN not in test_df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found in test dataset."
    )

# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X_test = test_df.drop(columns=[TARGET_COLUMN])
y_test = test_df[TARGET_COLUMN]

print("\nFeatures in test data:", X_test.shape[1])
print("Target column:", TARGET_COLUMN)

# ============================================================
# MATCH MODEL FEATURES
# ============================================================

model_features = model.feature_names_

if not model_features:
    raise ValueError("The saved model does not contain feature names.")

missing_features = [
    feature for feature in model_features
    if feature not in X_test.columns
]

if missing_features:
    print("\nMissing features:")
    for feature in missing_features:
        print(feature)

    raise ValueError(
        "Test dataset does not contain all features required by the model."
    )

# Use exactly the feature order used by the trained model.
X_test = X_test[model_features]

print("Feature order matched successfully.")

# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

# CatBoost may return shape (n, 1)
y_pred = y_pred.ravel()

print("Predictions generated successfully.")

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)

# ============================================================
# SAVE TEST RESULTS
# ============================================================

results_path = "models/saved_models/catboost_test_results.txt"

with open(results_path, "w", encoding="utf-8") as f:

    f.write("CATBOOST FINAL MODEL TEST RESULTS\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Test dataset shape: {test_df.shape}\n")
    f.write(f"Number of features: {X_test.shape[1]}\n\n")

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n\n")

    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 70 + "\n")

    f.write(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    f.write("\n\nCONFUSION MATRIX\n")
    f.write("=" * 70 + "\n")

    f.write(cm_df.to_string())

print("\nTest results saved to:")
print(results_path)

print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)