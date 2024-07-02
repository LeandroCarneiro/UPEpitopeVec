import csv
import numpy as np
from classes.AminoAcid import AminoAcid


def get_amino_acids_dataset():
    amino_acids = []

    with open('./datasets/aminoacids.csv', 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            amino_acid = AminoAcid(**row)
            amino_acids.append(amino_acid)

    return amino_acids


def GetAllAminoacids():
    aas = get_amino_acids_dataset()
    data = []
    xData = []
    for aa in aas:
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

    for i in range(len(aas)):
        if (i == len(aas)-1):
            aas[i].embedding = xData
        else:
            aas[i].embedding = normalized_data[i]

    return aas
