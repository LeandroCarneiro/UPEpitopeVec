import numpy as np
import matplotlib.pyplot as plt

from helpers.DatasetReader import GetAllAminoacids
from helpers.epitope_classifier import epitope_dataset
from helpers.distance_calculator import calculate_distance, plot_heatmap
from models.LeoModelsBuilder import build_MLP_embedder

epitopes_train, epitopes_eval = epitope_dataset()
aminoacids = GetAllAminoacids(False)
# print(aminoacids)

sequences = epitopes_eval['PEPTIDE'].values

bigrams = []
for words_list in sequences:
    for i in range(len(words_list) - 1):
        for j in range(i+1, len(words_list)):
            bigrams.append([words_list[i], words_list[j]])
            bigrams.append([words_list[j], words_list[i]])


all_words = [aa.letter for aa in aminoacids]

# print(all_words)
# print("Total number of unique words are:", len(all_words))

words_dict = {}

counter = 0
for word in all_words:
    words_dict[word] = counter
    counter += 1

# print(words_dict)
X = []
Y = []

for bi in bigrams:
    X.append(aminoacids[aminoacids.index(bi[0])].onehot_encode)
    Y.append(aminoacids[aminoacids.index(bi[1])].onehot_encode)

X = np.array(X)
Y = np.array(Y)

# print(Y.shape[1])
# print(len(aminoacids))

model = build_MLP_embedder(Y.shape[1], len(aminoacids))
model.summary()
model.fit(X, Y, epochs=1000, batch_size=len(sequences), verbose=True)

# print(X[0], ': ', Y[0])

weights = model.get_weights()[0]

word_embeddings = {}
for word in all_words:
    word_embeddings[word] = weights[words_dict[word]]

# print(word_embeddings)


# plt.figure(figsize = (10, 10))
for word in list(words_dict.keys()):
    coord = word_embeddings.get(word)
    plt.scatter(coord[0], coord[1])
    plt.annotate(word, (coord[0], coord[1]))

calculate_distance(aminoacids)
# plot_heatmap()

plt.show()
plt.savefig('plot_len.10.png')
# model.save('leo_enbedder_model.h5')
model.save('leo_enbedder_model_new.h5')
