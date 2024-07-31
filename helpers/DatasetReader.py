import csv
from typing import List
import numpy as np
from classes.AminoAcid import AminoAcid

from gensim.models import Word2Vec


def get_amino_acids_dataset():
    amino_acids = []

    with open('./datasets/aminoacids.csv', 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            amino_acid = AminoAcid(**row)
            amino_acids.append(amino_acid)

    return amino_acids


def GetAllAminoacids(withEmbedding):
    aas = get_amino_acids_dataset()
    if withEmbedding:
        return embed_aminoacids(aas)

    return aas


def embed_aminoacids(allAminoacids: List[AminoAcid]):
    model = Word2Vec.load('embedder_word2vec.h5')
    for aa in allAminoacids:
        aa.embedding = model.wv[aa.letter]

    return allAminoacids
