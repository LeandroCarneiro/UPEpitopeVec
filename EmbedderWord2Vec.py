import matplotlib.pyplot as plt

from helpers.DatasetReader import GetAllAminoacids
from helpers.epitope_classifier import GetEpitopes
from helpers.distance_calculator import calculate_distance, plot_heatmap

from gensim.models import Word2Vec


def TrainAndSaveEmbedder():

    epitopes = GetEpitopes()
    aminoacids = GetAllAminoacids(False)
    sequences = []
    sequences = epitopes['PEPTIDE'].values

    # Skip-Gram (sg=1)
    # size (dimensionality of the word vectors)
    # window (maximum distance between the current and predicted word within a sentence)
    # min_count (ignore all words with total frequency lower than this)
    # workers (number of worker threads to train the model).
    model = Word2Vec(sentences=sequences, sg=1, vector_size=30, batch_words=20,
                     window=1, min_count=1, workers=4, max_vocab_size=21)
    model.train(sequences, total_examples=len(
        sequences), epochs=1000, total_words=21)

    for aa in aminoacids:
        aa.embedding = model.wv[aa.letter]

    plt.savefig('embedder_word2vec.png')
    model.save('embedder_word2vec.h5')


def Word2VecReport():
    aminoacids = GetAllAminoacids(True)

    manhattan_dist, euclidean_dist, cosine_dist = calculate_distance(
        aminoacids)

    words_list = []
    for aa in aminoacids:
        coord = aa.embedding
        plt.scatter(coord[0], coord[1])
        plt.annotate(aa.name, (coord[0], coord[1]))
        words_list.append(aa.letter)

    plot_heatmap(words_list, manhattan_dist, 'manhattan_dist')
    plot_heatmap(words_list, euclidean_dist, 'euclidean_dist')
    plot_heatmap(words_list, cosine_dist, 'cosine_dist')
    plt.show()

Word2VecReport()