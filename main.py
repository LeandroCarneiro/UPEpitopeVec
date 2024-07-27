import numpy as np

from classes.ModelWrapper import ModelWrapper
from helpers.epitope_classifier import GetEpitopeDataset
from helpers.epitope_encoder import embedding_epitopes
from helpers.DatasetReader import GetAllAminoacids
from models.LeoModelsBuilder import build_GRU_model, build_LSTM_model, build_RNN_model

from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split


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
        len(X_train), len(aminoacids[0].embedding), 128), prediction=[], scores=[]),
    ModelWrapper(name='LSTM', instance=build_LSTM_model(
        len(X_train), len(aminoacids[0].embedding), 128), prediction=[], scores=[]),
    ModelWrapper(name='GRU', instance=build_GRU_model(
        len(X_train), len(aminoacids[0].embedding), 128), prediction=[], scores=[])
]

for model in models:
    model.instance.fit(np.array(X_train), np.array(y_train), verbose=False,
                       epochs=1, batch_size=len(X_train), validation_split=0.4)

    scores = model.instance.evaluate(np.array(X_test), np.array(y_test))
    model.scores.append(scores)
    model.prediction = model.instance.predict(np.array(X_test))

for model in models:
    model.ReportROC(expected=y_test)
    model.ConfusionMatrix(expected=y_test, show=True)

# history = model.fit(np.array(X_train), np.array(y_train), verbose=True, epochs=10,
#                     batch_size=len(X_train), validation_split=0.3)

# # # print(history)
#     test_loss, test_acc = model.evaluate(
#         np.array(X_test), np.array(y_test))
# print(f'Test accuracy: {test_acc}')

# prediction = model.predict(np.array(allSequences_eval))
# y_pred = (prediction > 0.5).astype(int)

# epochs = range(len(history.history['accuracy']))
# # Confusion Matrix
# cm = confusion_matrix(y_eval, y_pred)
# # Compute ROC curve and ROC area for each class
# fpr, tpr, thresholds = roc_curve(
#     np.array(y_eval), prediction)
# roc_auc = auc(fpr, tpr)

# kappa = cohen_kappa_score(y_eval, y_pred)
# print(kappa)

# # Create a 2x2 grid of subplots
# fig, axs = plt.subplots(2, 2, figsize=(22, 15))

# # Plot Training and Validation Accuracy
# axs[0, 0].plot(
#     epochs, history.history['accuracy'], label='Training Accuracy')
# axs[0, 0].plot(epochs, history.history['val_accuracy'],
#                label='Validation Accuracy')
# axs[0, 0].set_title('Training and Validation Accuracy')
# axs[0, 0].set_xlabel('Epochs')
# axs[0, 0].set_ylabel('Accuracy')
# axs[0, 0].legend()

# # Plot Training and Validation Loss
# axs[0, 1].plot(epochs, history.history['loss'],
#                label='Training Loss')
# axs[0, 1].plot(epochs, history.history['val_loss'],
#                label='Validation Loss')
# axs[0, 1].set_title('Training and Validation Loss')
# axs[0, 1].set_xlabel('Epochs')
# axs[0, 1].set_ylabel('Loss')
# axs[0, 1].legend()

# # Plot Confusion Matrix Heatmap
# sns.heatmap(cm, annot=True, fmt='d',
#             cmap='Blues', ax=axs[1, 0])
# axs[1, 0].set_xlabel('Predicted')
# axs[1, 0].set_ylabel('Truth')
# axs[1, 0].set_title('Confusion Matrix')

# # Plot ROC Curve
# axs[1, 1].plot(fpr, tpr, color='darkorange', lw=2,
#                label=f'ROC curve (area = {roc_auc:.2f})')
# axs[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# axs[1, 1].set_xlim([0.0, 1.0])
# axs[1, 1].set_ylim([0.0, 1.05])
# axs[1, 1].set_xlabel('False Positive Rate')
# axs[1, 1].set_ylabel('True Positive Rate')
# axs[1, 1].set_title('Receiver Operating Characteristic')
# axs[1, 1].legend(loc="lower right")

# Adjust layout to prevent overlap
# plt.tight_layout()

# plt.show()
