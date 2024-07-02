import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder

from helpers.epitope_classifier import epitope_dataset
from helpers.epitope_encoder import embedding_epitopes
from models.LeoLSTM import build_model
from helpers.DatasetReader import GetAllAminoacids
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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

print(len(X_train))
model = build_model(len(X_train), 17, 64)
model.summary()

print(np.array(X_train).shape)
print(len(np.array(X_train)))

history = model.fit(np.array(X_train), np.array(y_train), epochs=100,
                    batch_size=len(X_train), validation_split=0.3)

# print(history)
test_loss, test_acc = model.evaluate(np.array(X_test), np.array(y_test))
print(f'Test accuracy: {test_acc}')


prediction = model.predict(np.array(allSequences_eval))
y_pred = (prediction > 0.5).astype(int)

cm = confusion_matrix(y_eval, y_pred)

# Plotting accuracy
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
print("finished")

# # Plotting accuracy
# plt.figure(figsize=(12, 8))
# plt.plot(train_acc_history, label='Training Accuracy')
# plt.plot(valid_acc_history, label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(loc='upper left')
# plt.show()
