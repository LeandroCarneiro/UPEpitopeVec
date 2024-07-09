import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder

from helpers.epitope_classifier import epitope_dataset
from helpers.epitope_encoder import embedding_epitopes
from models.LeoModelsBuilder import build_GRU_model, build_LSTM_model, build_RNN_model
from helpers.DatasetReader import GetAllAminoacids
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

aminoacids = GetAllAminoacids()
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

# model = build_LSTM_model(len(X_train), 17, 30)
model = build_RNN_model(len(X_train), 17, 30)
# model = build_GRU_model(len(X_train), 17, 30)
model.summary()

history = model.fit(np.array(X_train), np.array(y_train), epochs=100,
                    batch_size=len(X_train), validation_split=0.3)

# print(history)
test_loss, test_acc = model.evaluate(np.array(X_test), np.array(y_test))
print(f'Test accuracy: {test_acc}')


prediction = model.predict(np.array(allSequences_eval))
y_pred = (prediction > 0.5).astype(int)

epochs = range(len(history.history['accuracy']))
# Confusion Matrix
cm = confusion_matrix(y_eval, y_pred)
# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(np.array(y_eval), y_pred)
roc_auc = auc(fpr, tpr)


# Plotting accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, history.history['loss'], label='Training Loss')
plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure(figsize=(10, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

plt.show()

# Plotting accuracy
# sns.heatmap(cm, annot=True, fmt="d")
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()
# print("finished")
# model.save('leo_lstm_model.h5')

# # Plotting accuracy
# plt.figure(figsize=(12, 8))
# plt.plot(test_loss, label='Training Accuracy')
# plt.plot(test_acc, label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(loc='upper left')
# plt.show()

# kfold
