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
import helper_func as hf


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


def sequences_to_dict(seq_gen):
    """
    Create a dictionary mapping the PDB ID to
    the sequence.
    This is an alternative to SeqIO.to_dict, which
    can not handle duplicate keys.
    This will only get the first key if there
    are duplicate keys

    :param: a generator for the fasta sequences
    :type:  generator
    :returns: dictionary mapping key to sequence
    :rtype:   dict
    """

    dict1 = {}

    for seq in seq_gen:
        try:
            dict1[seq.name] = seq
        except ValueError:
            pass

    return dict1


@hf.timing_val
def convert_fasta_to_1_hot(path, fasta_file, train=True):
    """
    Given a fasta file, create a dictionary
    mapping PDB ID to its sequence in 1 hot encoded form.

    :param path: path to wanted amino acid sequences
    :type  path: str
    :param fasta_file: fasta file
    :type  fasta_file: str
    :returns: dictionary mapping PDB ID to its amino acid 
              sequence in 1 hot encoded form.
    :rtype:   dict
    """

    if train:
        fasta_seq = SeqIO.to_dict(
            SeqIO.parse(path + fasta_file, "fasta")
        )
    else:
        seq_gen = SeqIO.parse(path, "fasta")
        fasta_seq = sequences_to_dict(seq_gen)

    one_hot_encodes = {}
    for pdb_id, seqrecord in fasta_seq.items():
        if not train:
            # deal with annoying commas attached to the ID
            pdb_id = pdb_id[:5]

        one_hot_enc = convert_SeqRecord_to_1_hot(seqrecord)
        one_hot_encodes[pdb_id] = one_hot_enc
    return one_hot_encodes

def find_avg_aa_length(aa_dict):
    """
    Find the average amino acid length from a dictionary.
    """
    
    sizes = [array.shape[0] for array in aa_dict.values()]
    return sum(sizes) / len(aa_dict)


all_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
amino_acid_array = np.array(list(all_amino_acids)).reshape(-1, 1)

aa_enc = OneHotEncoder()
aa_enc.fit(amino_acid_array)
# to get the categories corresponding to each slot
# aa_enc.categories_


if __name__ == "__main__":

    import logging
    import argparse

    parser = argparse.ArgumentParser(
        description='Get only the fasta sequences that we want.')

    parser.add_argument(
        'cull_path', 
        type=str,
        help='The path to the cull directory')

    parser.add_argument(
        'fasta', 
        type=str,
        help='The fasta file.')

    parser.add_argument(
        'one_hot_file', 
        type=str,
        help='The file to save the numpy matrix of the 1 hot amino acids.')

    parser.add_argument(
        '--log', 
        type=str,
        help='Log file')

    args = parser.parse_args()


    path = args.cull_path
    one_hot_encodings, time_str = convert_fasta_to_1_hot(path, args.fasta)
    np.save(path + args.one_hot_file, one_hot_encodings)

    avg_aa_length = find_avg_aa_length(one_hot_encodings)

    logging.info(
        time_str
    )

    logging.info(
        "Average amino acid length: {}".format(avg_aa_length)
    )