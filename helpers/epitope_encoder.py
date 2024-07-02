import numpy as np

from typing import List
from classes.AminoAcid import AminoAcid


def embedding_epitope(epitope: str, allAminoacids: List[AminoAcid]):
    enconded_features = [
        allAminoacids[allAminoacids.index(aa)].embedding for aa in epitope]

    return enconded_features


def embedding_epitopes(epitopes: [], allAminoacids: List[AminoAcid], limit_length):
    embeddings = []

    for e in epitopes:
        padX = (limit_length-len(e))
        e = f"{e}{'X' * padX}"

        embeddings.append(embedding_epitope(e, allAminoacids))

    return embeddings
