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

epitopes = epitope_dataset()
epitopes.head()

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(epitopes['FLAG'])

allSequences = embedding_epitopes(epitopes['PEPTIDE'].values, aminoacids, 30)

X_train, X_test, y_train, y_test = train_test_split(
    allSequences, y, test_size=0.3)

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

ep_test = embedding_epitopes(['QTGNVYSLEAIEELNLKPGHLKD', 'ETVDELNAAHYSQGR',
                             'AATNAACAWLEAQEEE', 'GYVGAEFPLDITAGTE', 'GAAQEALEAYAAAERS', 'IKHQGLPQGVLNENLLRFFV', 'APFPEVFGKEKVNELSTDIG', 'SESTEDQAMEDIKQMEAE'], aminoacids, 30)
# pos, pos, pos, pos
# pos, neg, neg, neg

print(np.array(ep_test).shape)

prediction = model.predict(np.array(X_test))
y_pred = (prediction > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)

# Plotting accuracy
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# # Plotting accuracy
# plt.figure(figsize=(12, 8))
# plt.plot(train_acc_history, label='Training Accuracy')
# plt.plot(valid_acc_history, label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(loc='upper left')
# plt.show()