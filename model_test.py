# Example usage
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from helpers.DatasetReader import GetAllAminoacids
from helpers.epitope_classifier import epitope_dataset, perform_k_fold_cross_validation
from helpers.epitope_encoder import embedding_epitopes
from models.LeoModelsBuilder import build_LSTM_model

aminoacids = GetAllAminoacids()

epitopes_train, epitopes_eval = epitope_dataset()
epitopes_train.head()

labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(epitopes_train['FLAG'])
y_eval = labelencoder.fit_transform(epitopes_eval['FLAG'])

allSequences_train = embedding_epitopes(
    epitopes_train['PEPTIDE'].values, aminoacids, 30)

allSequences_eval = embedding_epitopes(
    epitopes_eval['PEPTIDE'].values, aminoacids, 30)

X_train, X_test, y_train, y_test = train_test_split(
    allSequences_train, y_train, test_size=0.3)


model = build_LSTM_model(len(X_train), 17, 30)

n_splits = 5
performance = perform_k_fold_cross_validation(
    allSequences_train, y_train, n_splits, model)
print(f"Average Performance Across All Folds: {performance}")
