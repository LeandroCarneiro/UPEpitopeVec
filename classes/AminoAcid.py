import numpy as np


class AminoAcid:
    def __init__(self, name: str, abbr: str, letter: str, molecular_formula: str, residue_formula: str, molecular_weight: float, residue_weight: float,
                 pka1: float, pkb2: float, pkx3: float, pl4: float, h: float, vsc: float, p1: float, p2: float, sasa: float, ncisc: float, carbon: float, hydrogen: float, nitrogen: float, oxygen: float, sulfur: float):
        self.name = name
        self.abbr = abbr
        self.letter = letter
        self.molecular_formula = molecular_formula
        self.residue_formula = residue_formula
        self.molecular_weight = molecular_weight
        self.residue_weight = residue_weight
        self.pka1 = pka1
        self.pkb2 = pkb2
        self.pkx3 = pkx3
        self.pl4 = pl4
        self.h = h  # Assuming 'h' is a typo and meant to be 'vsc'
        self.vsc = vsc
        self.p1 = p1
        self.p2 = p2
        self.sasa = sasa
        self.ncisc = ncisc
        self.carbon = carbon
        self.hydrogen = hydrogen
        self.nitrogen = nitrogen
        self.oxygen = oxygen
        self.sulfur = sulfur
        self.embedding = []

    def return_amino_acid(self):
        return self

    def __hash__(self):
        return hash((self.letter))

    def __eq__(self, other):
        return (self.letter) == (other)

    def get_features(self):
        data = np.array([self.molecular_weight,
                         self.residue_weight,
                         self.pka1,
                         self.pkb2,
                         self.pkx3,
                         self.pl4,
                         self.h,
                         self.vsc,
                         self.p1,
                         self.p2,
                         self.sasa,
                         self.ncisc,
                         self.carbon,
                         self.hydrogen,
                         self.nitrogen,
                         self.oxygen,
                         self.sulfur])

        data = data.astype(float)

        return data
