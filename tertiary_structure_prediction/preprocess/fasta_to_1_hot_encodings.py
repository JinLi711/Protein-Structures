"""
Convert each amino acid sequence into its one hot encoded 
representations and save it as a file.

NOTE: need to edit this so this file can generalize to 
new fasta sequences (not in the train set).
"""

from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
import sys


def convert_SeqRecord_to_1_hot(seqrecord):
    """
    Given an amino acid sequence in SeqRecord object,
    convert the amino acid into one hot encoding.

    Some sequences may include a character that is not an
    typical amino acid.

    For example:
        B denotes Aspartate (D) or Asparagine (N)
        J denotes Leucine (L) or Isoleucine (I)
        O denotes Pyrrolysine, encoded by a stop codon. 
            Most similar to Lysine (K)
        X denotes "any"
        U denotes Selenocysteine, most similar to Cysteine (C)
        Z denotes Glutamate (E) or Glutamine (Q)
        * denotes translation stop
        - denotes indeterminate length
    
    These characters are replaced accordingly.

    :param seqrecord: a sequence from the fasta file
    :type  seqrecord: Bio.SeqRecord.SeqRecord
    :returns: 1 hot encoded array
    :rtype:   numpy.ndarray
    """

    seq = seqrecord.seq

    new_seq = []
    for aa in seq:
        if aa == "B":
            new_seq.append(
                random.choice(['D', 'N'])
            )
        elif aa == 'O':
            new_seq.append(
                'K'
            )
        elif aa == "J":
            new_seq.append(
                random.choice(['L', 'I'])
            )
        elif aa == "X":
            # replace by most frequently occuring amino acids
            # leucine, serine, lysine, and glutamic acid
            new_seq.append(
                random.choice(['L', 'S', 'K', 'E'])
            )
        elif aa == 'U':
            new_seq.append(
                'C'
            )
        elif aa == "Z":
            new_seq.append(
                random.choice(['E', 'Q'])
            )
        else:
            new_seq.append(aa)

    seq = np.array(new_seq).reshape(-1, 1)
    seq_1_hot = aa_enc.transform(seq)

    return seq_1_hot.toarray()


def convert_fasta_to_1_hot(path):
    """
    Given a fasta file, create a dictionary
    mapping PDB ID to its sequence in 1 hot encoded form.

    :param path: path to wanted amino acid sequences
    :type  path: str
    :returns: dictionary mapping PDB ID to its amino acid 
              sequence in 1 hot encoded form.
    :rtype:   dict
    """

    fasta_seq = SeqIO.to_dict(
        SeqIO.parse(path + "wanted_aa.fasta", "fasta")
    )

    one_hot_encodes = {}
    for pdb_id, seqrecord in fasta_seq.items():
        one_hot_enc = convert_SeqRecord_to_1_hot(seqrecord)
        one_hot_encodes[pdb_id] = one_hot_enc
    return one_hot_encodes


if __name__ == "__main__":
    # all_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    all_amino_acids = open("../preprocess/amino_acid_letters.txt", 'r')
    all_amino_acids = all_amino_acids.read()
    amino_acid_array = np.array(list(all_amino_acids)).reshape(-1, 1)

    aa_enc = OneHotEncoder()
    aa_enc.fit(amino_acid_array)
    # to get the categories corresponding to each slot
    # aa_enc.categories_

    path = "../data/cull%i/" % int (sys.argv[1])
    one_hot_encodings = convert_fasta_to_1_hot(path)
    np.save(path + 'amino_acids_1_hot.npy', one_hot_encodings)