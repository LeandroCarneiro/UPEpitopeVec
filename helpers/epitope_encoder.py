from typing import List
from classes.AminoAcid import AminoAcid
import numpy as np


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


def get_peptides(molecule, min_win, max_win):
    peptides = []
    length = len(molecule)
    for i in range(0, length-min_win, min_win):
        for j in range(0, max_win-min_win, 1):
            peptide = molecule[i:i+j+min_win]
            peptides.append(peptide)

    return peptides
