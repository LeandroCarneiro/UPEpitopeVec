import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder

from helpers.epitope_classifier import epitope_dataset
from helpers.epitope_encoder import embedding_epitopes
from models.LeoLSTM import build_model
from helpers.DatasetReader import GetAllAminoacids
from sklearn.model_selection import train_test_split

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

model = build_model(30, 17, 64)
model.summary()

print(np.array(X_train).shape)

history = model.fit(np.array(X_train), np.array(y_train), epochs=10,
                    batch_size=32, validation_split=0.2)

print(history)
test_loss, test_acc = model.evaluate(np.array(X_test), np.array(y_test))
print(f'Test accuracy: {test_acc}')
