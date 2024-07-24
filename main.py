import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from helpers.epitope_classifier import epitope_dataset
from helpers.epitope_encoder import embedding_epitopes
from helpers.DatasetReader import GetAllAminoacids, GetAllAminoAciddsHibridModel
from models.LeoModelsBuilder import build_GRU_model, build_LSTM_model, build_RNN_model, build_MLP_model

from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, cohen_kappa_score
from itertools import combinations
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

aminoacids = GetAllAminoacids(isHybrid=True)
# for i in range(len(aminoacids)):
#     print(f"{aminoacids[i].letter}: {aminoacids[i].embedding}")

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

model = build_LSTM_model(len(X_train), len(aminoacids[0].embedding), 128)

#     models = [
#   ('LSTM', build_LSTM_model(len(X_train),
#                              len(aminoacids[0].embedding), 128)),
#    ('RNN', build_RNN_model(len(X_train), len(aminoacids[0].embedding), 128)),
#     ('GRU', build_GRU_model(len(X_train), len(aminoacids[0].embedding), 128))
# ]
# model.summary()
# print(np.array(X_train).shape)

history = model.fit(np.array(X_train), np.array(y_train), verbose=True, epochs=10,
                    batch_size=len(X_train), validation_split=0.3)

# print(history)
test_loss, test_acc = model.evaluate(
    np.array(X_test), np.array(y_test))
print(f'Test accuracy: {test_acc}')

prediction = model.predict(np.array(allSequences_eval))
y_pred = (prediction > 0.5).astype(int)

epochs = range(len(history.history['accuracy']))
# Confusion Matrix
cm = confusion_matrix(y_eval, y_pred)
# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(
    np.array(y_eval), prediction)
roc_auc = auc(fpr, tpr)

kappa = cohen_kappa_score(y_eval, y_pred)
print(kappa)

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(22, 15))

# Plot Training and Validation Accuracy
axs[0, 0].plot(
    epochs, history.history['accuracy'], label='Training Accuracy')
axs[0, 0].plot(epochs, history.history['val_accuracy'],
               label='Validation Accuracy')
axs[0, 0].set_title('Training and Validation Accuracy')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].legend()

# Plot Training and Validation Loss
axs[0, 1].plot(epochs, history.history['loss'],
               label='Training Loss')
axs[0, 1].plot(epochs, history.history['val_loss'],
               label='Validation Loss')
axs[0, 1].set_title('Training and Validation Loss')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()

# Plot Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d',
            cmap='Blues', ax=axs[1, 0])
axs[1, 0].set_xlabel('Predicted')
axs[1, 0].set_ylabel('Truth')
axs[1, 0].set_title('Confusion Matrix')

# Plot ROC Curve
axs[1, 1].plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (area = {roc_auc:.2f})')
axs[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[1, 1].set_xlim([0.0, 1.0])
axs[1, 1].set_ylim([0.0, 1.05])
axs[1, 1].set_xlabel('False Positive Rate')
axs[1, 1].set_ylabel('True Positive Rate')
axs[1, 1].set_title('Receiver Operating Characteristic')
axs[1, 1].legend(loc="lower right")

# Adjust layout to prevent overlap
plt.tight_layout()

plt.show()
