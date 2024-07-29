import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

from helpers.DatasetReader import GetAllAminoacids
from helpers.epitope_classifier import GetEpitopes
from helpers.distance_calculator import calculate_distance, plot_heatmap
from models.LeoModelsBuilder import build_MLP_embedder


def TrainAndSaveEmbedder():

    epitopes = GetEpitopes()
    aminoacids = GetAllAminoacids(False)

    sequences = epitopes['PEPTIDE'].values[:2]

    aminoacids_dict = {aa.letter: aa for aa in aminoacids}

    bigrams = []
    for seq in sequences:
        for i in range(len(seq) - 1):
            gramI_id = seq[i]
            gramI = aminoacids_dict.get(gramI_id)
            for j in range(i + 1, len(seq)):
                gramJ_id = seq[j]
                gramJ = aminoacids_dict.get(gramJ_id)
                for k in range(j + 1, len(seq)):
                    gramK_id = seq[k]
                    gramK = aminoacids_dict.get(gramK_id)
                    bigrams.append([gramI, gramJ, gramK])
                    bigrams.append([gramK, gramJ, gramI])

    X = []
    Y = []

    for bi in bigrams:
        X.append([bi[0].onehot_encode, bi[1].onehot_encode, bi[2].onehot_encode])
        Y.append([bi[2].onehot_encode, bi[0].onehot_encode, bi[0].onehot_encode])

    X = np.array(X)
    Y = np.array(Y)

    model = build_MLP_embedder(Y.shape[1], (len(aminoacids)))
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X, Y, epochs=200, batch_size=len(sequences),
              verbose=True, callbacks=[early_stopping])

    weights = model.get_weights()[0]

    for aa in aminoacids:
        aa.embedding = weights[aa.id-1]

    for aa in aminoacids:
        coord = aa.embedding
        plt.scatter(coord[0], coord[1])
        plt.annotate(aa.name, (coord[0], coord[1]))

    model.save('embedder_model_only_positive_2.h5')
    calculate_distance(aminoacids)

    plt.savefig('embedder_model_only_positive_2.png')
    # plt.show()
    # plot_heatmap()

TrainAndSaveEmbedder()