# import iedb

# # Previsão de ligação de peptídeo MHC-I
# mhci_res = iedb.query_mhci_binding(method="recommended", sequence="ARFTGIKTA", allele="HLA-A*02:01", length=8)

# # Previsão de ligação de peptídeo MHC-II
# mhcii_res = iedb.query_mhcii_binding(method="nn_align", sequence="SLYNTVATLYCVHQRIDV", allele="HLA-DRB1*01:01", length=None)

# # Previsão de epítopo T-célula
# tcell_res = iedb.query_tcell_epitope(method="smm", sequence="SLYNTVATLYCVHQRIDV", allele="HLA-A*01:01", length=9, proteasome='immuno')

# # Avaliação da probabilidade de um peptídeo ser naturalmente processado pelo MHC
# pep_res = iedb.query_peptide_prediction(method="mhcnp", sequence="SLYNTVATLYCVHQRIDV", allele="HLA-A*02:01", length=9)

# # Previsão de epítopo B-célula
# bcell_res = iedb.query_bcell_epitope(method="Emini", sequence="VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTE", window_size=9)


# print(mhci_res)
# print(mhcii_res)
# print(tcell_res)
# print(pep_res)
# print(bcell_res)


from keras.models import load_model
from matplotlib import pyplot as plt

from helpers.DatasetReader import GetAllAminoacids
from helpers.distance_calculator import calculate_distance, plot_heatmap
from gensim.models import Word2Vec


aminoacids = GetAllAminoacids(False)
# model = Word2Vec.load('leo_embeder_word2vec.h5')
model = load_model('leo_embeder_new.h5')


weights = model.get_weights()[0]
# print(word_embeddings)


# plt.figure(figsize = (10, 10))
for word in aminoacids:
    word.embedding = weights[word.letter]
    coord = word.embedding
    plt.scatter(coord[0], coord[1])
    plt.annotate(word, (coord[0], coord[1]))

manhattan_dist, euclidean_dist, cosine_dist = calculate_distance(aminoacids)
print(f'manhattan_dist: {manhattan_dist}')
print(f'euclidean_dist: {euclidean_dist}')
print(f'cosine_dist: {cosine_dist}')


# for word in aminoacids:
#     word.embedding = model.wv[word.letter]
#     coord = word.embedding
#     plt.scatter(coord[0], coord[1])
#     plt.annotate(word, (coord[0], coord[1]))

# words_list = []
# for aa in aminoacids:
#     words_list.append(aa.letter)

# manhattan_dist, euclidean_dist, cosine_dist = calculate_distance(aminoacids)
# plot_heatmap(words_list, manhattan_dist, 'manhattan_dist')
# plot_heatmap(words_list, euclidean_dist, 'euclidean_dist')
# plot_heatmap(words_list, cosine_dist, 'cosine_dist')
