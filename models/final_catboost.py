# ============================================================
# FEDCIP - FINAL OPTIMIZED CATBOOST MODEL
#
# Dataset:
# final_dataset -> existing train/validation/test CSV files
#
# This is Model 3:
# Optimized version of the successful weighted CatBoost model
# ============================================================

import os
import time
import numpy as np
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
# 1. PATH CONFIGURATION
# ============================================================

# This file is inside:
# fedcip/models/final_catboost.py

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "saved_models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


TRAIN_PATH = os.path.join(
    DATA_DIR,
    "train.csv"
)

VALIDATION_PATH = os.path.join(
    DATA_DIR,
    "validation.csv"
)

TEST_PATH = os.path.join(
    DATA_DIR,
    "test.csv"
)


# ============================================================
# 2. MODEL SETTINGS
# ============================================================

TARGET = "Label"

RANDOM_SEED = 42

# Reduced from Model 2:
ITERATIONS = 150

# Same learning rate used in Model 2
LEARNING_RATE = 0.1

# Slightly smaller tree depth
DEPTH = 7

# Stop if validation loss doesn't improve
EARLY_STOPPING = 20

# Controlled class weighting
ALPHA = 0.25


# ============================================================
# 3. START TIMER
# ============================================================

total_start = time.time()


print("=" * 75)
print("FEDCIP - FINAL OPTIMIZED CATBOOST MODEL")
print("=" * 75)


# ============================================================
# 4. CHECK DATASET
# ============================================================

print("\nChecking dataset files...")

for path in [
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nDataset file not found:\n{path}"
        )

    print("Found:", path)


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\nLoading datasets...")

load_start = time.time()


train_df = pd.read_csv(
    TRAIN_PATH
)

validation_df = pd.read_csv(
    VALIDATION_PATH
)

test_df = pd.read_csv(
    TEST_PATH
)


load_time = (
    time.time()
    - load_start
)


print(
    "\nTrain shape      :",
    train_df.shape
)

print(
    "Validation shape :",
    validation_df.shape
)

print(
    "Test shape       :",
    test_df.shape
)

print(
    f"\nData loading time: "
    f"{load_time / 60:.2f} minutes"
)


# ============================================================
# 6. SEPARATE FEATURES AND LABEL
# ============================================================

if TARGET not in train_df.columns:

    raise ValueError(
        f"Target column '{TARGET}' "
        f"not found in train.csv"
    )


X_train = train_df.drop(
    columns=[TARGET]
)

y_train = train_df[TARGET].astype(int)


X_validation = validation_df.drop(
    columns=[TARGET]
)

y_validation = validation_df[TARGET].astype(int)


X_test = test_df.drop(
    columns=[TARGET]
)

y_test = test_df[TARGET].astype(int)


print(
    "\nNumber of features:",
    X_train.shape[1]
)


# ============================================================
# 7. CHECK FEATURES
# ============================================================

if list(X_train.columns) != list(
    X_validation.columns
):

    raise ValueError(
        "Train and validation "
        "features do not match."
    )


if list(X_train.columns) != list(
    X_test.columns
):

    raise ValueError(
        "Train and test "
        "features do not match."
    )


# ============================================================
# 8. HANDLE INF VALUES
# ============================================================

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_validation = X_validation.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 9. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 75)


class_counts = (
    y_train
    .value_counts()
    .sort_index()
)

classes = sorted(
    y_train.unique()
)

number_of_classes = len(
    classes
)


for class_id in classes:

    print(
        f"Class {class_id:2d}: "
        f"{class_counts[class_id]:,}"
    )


# ============================================================
# 10. CONTROLLED CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 75)
print("CALCULATING CONTROLLED CLASS WEIGHTS")
print("=" * 75)


total_samples = len(
    y_train
)


class_weights = {}


for class_id in classes:

    count = class_counts[
        class_id
    ]

    # Standard balanced weight
    balanced_weight = (
        total_samples /
        (
            number_of_classes *
            count
        )
    )

    # Controlled weighting
    controlled_weight = (
        balanced_weight ** ALPHA
    )

    class_weights[
        class_id
    ] = controlled_weight


# Normalize around 1
mean_weight = np.mean(
    list(
        class_weights.values()
    )
)


for class_id in classes:

    class_weights[
        class_id
    ] /= mean_weight


print("\nControlled class weights:")


for class_id in classes:

    print(
        f"Class {class_id:2d} | "
        f"Samples: "
        f"{class_counts[class_id]:8,d} | "
        f"Weight: "
        f"{class_weights[class_id]:7.3f}"
    )


weights_list = [

    class_weights[class_id]

    for class_id in classes

]


# ============================================================
# 11. CREATE MODEL
# ============================================================

print("\n" + "=" * 75)
print("CREATING OPTIMIZED CATBOOST MODEL")
print("=" * 75)


model = CatBoostClassifier(

    iterations=ITERATIONS,

    learning_rate=LEARNING_RATE,

    depth=DEPTH,

    loss_function="MultiClass",

    # IMPORTANT:
    # Keep training metric fast.
    # Macro F1 is calculated after prediction.
    eval_metric="MultiClass",

    class_weights=weights_list,

    early_stopping_rounds=EARLY_STOPPING,

    random_seed=RANDOM_SEED,

    task_type="CPU",

    verbose=25,

    use_best_model=True

)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\n" + "=" * 75)
print("TRAINING FINAL OPTIMIZED MODEL")
print("=" * 75)


print(
    "\nTarget configuration:"
)

print(
    f"Iterations       : {ITERATIONS}"
)

print(
    f"Depth            : {DEPTH}"
)

print(
    f"Learning rate    : {LEARNING_RATE}"
)

print(
    f"Class weight α   : {ALPHA}"
)

print(
    f"Early stopping   : {EARLY_STOPPING}"
)


train_start = time.time()


model.fit(

    X_train,

    y_train,

    eval_set=(
        X_validation,
        y_validation
    )

)


train_time = (
    time.time()
    - train_start
)


print("\nTraining completed.")

print(
    f"Training time: "
    f"{train_time / 60:.2f} minutes"
)

print(
    "Best iteration:",
    model.get_best_iteration()
)


# ============================================================
# 13. VALIDATION PREDICTION
# ============================================================

print("\n" + "=" * 75)
print("VALIDATION EVALUATION")
print("=" * 75)


validation_start = time.time()


y_validation_pred = model.predict(
    X_validation
)


y_validation_pred = (
    y_validation_pred
    .flatten()
    .astype(int)
)


validation_time = (
    time.time()
    - validation_start
)


# ============================================================
# 14. VALIDATION METRICS
# ============================================================

validation_accuracy = accuracy_score(
    y_validation,
    y_validation_pred
)

validation_precision = precision_score(
    y_validation,
    y_validation_pred,
    average="weighted",
    zero_division=0
)

validation_recall = recall_score(
    y_validation,
    y_validation_pred,
    average="weighted",
    zero_division=0
)

validation_weighted_f1 = f1_score(
    y_validation,
    y_validation_pred,
    average="weighted",
    zero_division=0
)

validation_macro_f1 = f1_score(
    y_validation,
    y_validation_pred,
    average="macro",
    zero_division=0
)


print(
    f"\nValidation Accuracy    : "
    f"{validation_accuracy:.4f}"
)

print(
    f"Validation Precision   : "
    f"{validation_precision:.4f}"
)

print(
    f"Validation Recall      : "
    f"{validation_recall:.4f}"
)

print(
    f"Validation Weighted F1 : "
    f"{validation_weighted_f1:.4f}"
)

print(
    f"Validation Macro F1    : "
    f"{validation_macro_f1:.4f}"
)

print(
    f"\nValidation prediction time: "
    f"{validation_time / 60:.2f} minutes"
)


# ============================================================
# 15. TEST PREDICTION
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)


test_start = time.time()


y_test_pred = model.predict(
    X_test
)


y_test_pred = (
    y_test_pred
    .flatten()
    .astype(int)
)


test_time = (
    time.time()
    - test_start
)


# ============================================================
# 16. TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

test_precision = precision_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)

test_weighted_f1 = f1_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)

test_macro_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# 17. FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL OPTIMIZED MODEL - TEST PERFORMANCE")
print("=" * 75)


print(
    f"\nTest Accuracy       : "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Precision      : "
    f"{test_precision:.4f}"
)

print(
    f"Test Recall         : "
    f"{test_recall:.4f}"
)

print(
    f"Test Weighted F1    : "
    f"{test_weighted_f1:.4f}"
)

print(
    f"Test Macro F1       : "
    f"{test_macro_f1:.4f}"
)

print(
    f"\nTest prediction time: "
    f"{test_time / 60:.2f} minutes"
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)


report = classification_report(

    y_test,

    y_test_pred,

    labels=classes,

    target_names=[
        str(c)
        for c in classes
    ],

    zero_division=0

)


print(report)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)


cm = confusion_matrix(

    y_test,

    y_test_pred,

    labels=classes

)


cm_df = pd.DataFrame(

    cm,

    index=classes,

    columns=classes

)


print(cm_df)


# ============================================================
# 20. SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_PATH = os.path.join(

    MODEL_DIR,

    "catboost_final_model.cbm"

)


model.save_model(
    FINAL_MODEL_PATH
)


print(
    "\nFinal model saved:"
)

print(
    FINAL_MODEL_PATH
)


# ============================================================
# 21. SAVE CLASS WEIGHTS
# ============================================================

WEIGHTS_PATH = os.path.join(

    MODEL_DIR,

    "final_class_weights.csv"

)


weights_df = pd.DataFrame({

    "class": classes,

    "training_samples": [

        class_counts[c]

        for c in classes

    ],

    "weight": [

        class_weights[c]

        for c in classes

    ]

})


weights_df.to_csv(

    WEIGHTS_PATH,

    index=False

)


print(
    "\nClass weights saved:"
)

print(
    WEIGHTS_PATH
)


# ============================================================
# 22. SAVE METRICS
# ============================================================

METRICS_PATH = os.path.join(

    MODEL_DIR,

    "catboost_final_metrics.txt"

)


with open(
    METRICS_PATH,
    "w"
) as f:

    f.write(
        "FEDCIP - FINAL OPTIMIZED CATBOOST MODEL\n"
    )

    f.write(
        "=======================================\n\n"
    )

    f.write(
        "Dataset: final_dataset\n"
    )

    f.write(
        "Train file: train.csv\n"
    )

    f.write(
        "Validation file: validation.csv\n"
    )

    f.write(
        "Test file: test.csv\n\n"
    )

    f.write(
        f"Features: {X_train.shape[1]}\n"
    )

    f.write(
        f"Classes: {number_of_classes}\n\n"
    )

    f.write(
        f"Iterations: {ITERATIONS}\n"
    )

    f.write(
        f"Depth: {DEPTH}\n"
    )

    f.write(
        f"Learning rate: {LEARNING_RATE}\n"
    )

    f.write(
        f"Alpha: {ALPHA}\n\n"
    )

    f.write(
        f"Best iteration: "
        f"{model.get_best_iteration()}\n"
    )

    f.write(
        f"Training time: "
        f"{train_time / 60:.2f} minutes\n\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}\n"
    )

    f.write(
        f"Test Precision: "
        f"{test_precision:.4f}\n"
    )

    f.write(
        f"Test Recall: "
        f"{test_recall:.4f}\n"
    )

    f.write(
        f"Test Weighted F1: "
        f"{test_weighted_f1:.4f}\n"
    )

    f.write(
        f"Test Macro F1: "
        f"{test_macro_f1:.4f}\n"
    )


print(
    "\nMetrics saved:"
)

print(
    METRICS_PATH
)


# ============================================================
# 23. TOTAL TIME
# ============================================================

total_time = (
    time.time()
    - total_start
)


print("\n" + "=" * 75)
print("FINAL OPTIMIZED CATBOOST MODEL COMPLETED")
print("=" * 75)


print(
    f"\nTotal execution time: "
    f"{total_time / 60:.2f} minutes"
)


print("\n" + "=" * 75)
print("MODEL 1 vs MODEL 2 vs FINAL MODEL")
print("=" * 75)

print(
    "\nModel 1:"
)

print(
    "Accuracy = 93.64%"
)

print(
    "Weighted F1 = 96.04%"
)

print(
    "Macro F1 ≈ 54%"
)


print(
    "\nModel 2:"
)

print(
    "Accuracy = 99.69%"
)

print(
    "Weighted F1 = 99.68%"
)

print(
    "Macro F1 = 76.34%"
)

print(
    "Training time ≈ 257 minutes"
)


print(
    "\nNow compare the FINAL MODEL results "
    "with Model 1 and Model 2."
)