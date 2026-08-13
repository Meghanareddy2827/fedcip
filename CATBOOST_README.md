# CatBoost Model - FEDCIP

## What is this?

This folder contains the final CatBoost Machine Learning model
developed for the FEDCIP cybersecurity project.

The model is used to classify network traffic into different
cybersecurity attack categories.

## Final Model

The trained model is:

models/saved_models/catboost_final_model.cbm

This is the main file you need if you want to use the trained
CatBoost model.

You do NOT need to train the model again.

## Model Information

- Model: CatBoost Classifier
- Dataset: CICIDS2017
- Number of features: 23
- Test samples: 385,896

## Final Test Results

The model was tested on the test dataset.

- Accuracy: 99.68%
- Precision: 99.67%
- Recall: 99.68%
- F1 Score: 99.66%

## Important Files

### 1. Trained Model

models/saved_models/catboost_final_model.cbm

This is the actual trained model.

### 2. Training Code

models/final_catboost.py

This contains the code used to train the CatBoost model.

### 3. Test Code

src/test_catboost.py

This contains the code used to test the saved model.

### 4. Test Results

models/saved_models/catboost_test_results.txt

Contains the final test results and confusion matrix.

### 5. Class Weights

models/saved_models/final_class_weights.csv

Contains the class weights used during training to handle
class imbalance.

### 6. Class Distribution

models/class_distribution.csv

Contains information about the distribution of attack classes.

### 7. Model Metrics

models/saved_models/catboost_final_metrics.txt

Contains the model evaluation metrics from the development stage.

## Dataset

The dataset files are NOT stored in GitHub because they are very
large.

The required datasets are available in the team's shared
Google Drive.

The main files required are:

- train.csv
- validation.csv
- test.csv

## How to Load the Model

Install CatBoost if it is not already installed:

pip install catboost

Then:

from catboost import CatBoostClassifier

model = CatBoostClassifier()

model.load_model(
    "models/saved_models/catboost_final_model.cbm"
)

The model expects the same 23 features that were used during
training.

Keep the feature names and feature order the same when using
the model.

## For Team Members

If you are working on the next stage of the FEDCIP project,
you can directly use:

models/saved_models/catboost_final_model.cbm

You do not need to retrain the CatBoost model.

Use this trained model as the standard ML model for the next
stage of the project.

## Important

Do not delete or rename:

catboost_final_model.cbm

This is the final trained CatBoost model.