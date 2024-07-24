import pandas as pd
from sklearn.model_selection import KFold
import numpy as np


def epitope_dataset():
    file_path_train = './datasets/bepi3_epitopes_30.csv'
    file_path_eval = './datasets/bepi3_epitopes_eval.csv'
    df_train = pd.read_csv(file_path_train)
    df_eval = pd.read_csv(file_path_eval)

    return df_train.loc[df_train['PEPTIDE'].str.len() <= 30],  df_eval.loc[df_eval['PEPTIDE'].str.len() <= 30]


def perform_k_fold_cross_validation(allSequences_train, y_train, n_splits, model):
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # List to hold performance metrics
    performance_metrics = []

    # Iterate over folds
    for train_index, val_index in kf.split(allSequences_train):
        # Split datasets
        X_train_fold, X_val_fold = allSequences_train[:
                                                      train_index], allSequences_train[:val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Train your model here (example: Logistic Regression)
        model.fit(X_train_fold, y_train_fold)

        # Predict on validation set
        predictions = model.predict(X_val_fold)

        # Calculate and append performance metric (example: accuracy)
        accuracy = np.mean(predictions == y_val_fold)
        performance_metrics.append(accuracy)

    return np.mean(performance_metrics)
