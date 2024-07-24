import csv
from typing import List
import numpy as np
from classes.AminoAcid import AminoAcid

from keras.models import load_model


def get_amino_acids_dataset():
    amino_acids = []

    with open('./datasets/aminoacids.csv', 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            amino_acid = AminoAcid(**row)
            amino_acids.append(amino_acid)

    return amino_acids


def GetAllAminoAciddsHibridModel(embeddings):
    aas = get_amino_acids_dataset()
    for aa in aas:
        aa.embedding = embeddings[aa.letter]

    return aas


def GetAllAminoacids(isHybrid: bool):
    aas = get_amino_acids_dataset()
    return embed_aminoacids(aas, isHybrid)


def embed_aminoacids(allAminoacids: List[AminoAcid], isHybrid: bool):
    if isHybrid:
        model = load_model('leo_enbedder_model_new.h5')
        weights = model.get_weights()[0]

        for aa in allAminoacids:
            aa.embedding = weights[aa.id-1]
    else:
        data = []
        xData = []

        for aa in allAminoacids:
            if (aa != 'X'):
                data.append(aa.get_features())
            else:
                xData = aa.get_features()

        # data = np.array(vectors)
        # Calculate min and max for each feature
        min_values = np.min(data)
        max_values = np.max(data)

        # Normalize the data
        normalized_data = (data - min_values) / (max_values - min_values)

        for i in range(len(allAminoacids)):
            # set one hot encode
            allAminoacids[i].onehot_encode = allAminoacids[i].get_encode()

            if (i == len(allAminoacids)-1):
                allAminoacids[i].embedding = xData
            else:
                allAminoacids[i].embedding = normalized_data[i]

    return allAminoacids
