import numpy as np

from classes.ModelWrapper import ModelWrapper
from helpers.epitope_classifier import GetEpitopeDataset
from helpers.epitope_encoder import embedding_epitopes
from helpers.DatasetReader import GetAllAminoacids
from models.LeoModelsBuilder import build_GRU_model, build_LSTM_model, build_RNN_model
# from EmbbederModel import TrainAndSaveEmbedder
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split


# TrainAndSaveEmbedder()

aminoacids = GetAllAminoacids(isHybrid=True)
epitopes_train, _ = GetEpitopeDataset()

labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(epitopes_train['FLAG'])

allSequences_train = embedding_epitopes(
    epitopes_train['PEPTIDE'].values, aminoacids, 30)

X_train, X_test, y_train, y_test = train_test_split(
    allSequences_train, y_train, test_size=0.5, shuffle=True)

models = [
    ModelWrapper(name='RNN', instance=build_RNN_model(
        len(X_train), len(aminoacids[0].embedding), 64), prediction=[], scores=[], accuracy=0),
    ModelWrapper(name='LSTM', instance=build_LSTM_model(
        len(X_train), len(aminoacids[0].embedding), 64), prediction=[], scores=[], accuracy=0),
    ModelWrapper(name='GRU', instance=build_GRU_model(
        len(X_train), len(aminoacids[0].embedding), 64), prediction=[], scores=[], accuracy=0)
]



for model in models:
    model.instance.fit(np.array(X_train), np.array(y_train), verbose=True,
                       epochs=150, batch_size=len(X_train), validation_split=0.5)

    scores, acc = model.instance.evaluate(np.array(X_test), np.array(y_test))
    model.scores.append(scores)
    model.prediction = model.instance.predict(np.array(X_test))
    model.accuracy = acc

for model in models:
    model.ReportROC(expected=y_test)
    model.Histogram(expected=y_test)
    model.ConfusionMatrix(expected=y_test, show=True)
    model.ReportCrossValidation()
    model.ReportAcuracy(expected=y_test)
