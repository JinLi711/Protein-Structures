"""
Write out a fasta file that contains only the
amino acid sequences that we want.

Output file: 
    wanted_aa.fasta
"""


import sys
from os import listdir
from os.path import isfile, join
from Bio import SeqIO
import helper_func as hf


@hf.timing_val
def extract_wanted_fasta(path, fasta_file):
    """
    Create a fasta file containing 
    only the sequences that we want 
    (i.e. the ones left in the PDB folder)

    :param path: path to the data
    :type  path: str
    :returns: list of wanted amino acid sequences
    :rtype:   list
    """
    
    wanted_seq = []
    for seq_record in SeqIO.parse(path + fasta_file, "fasta"):
        # remove the last letter, which describes the chain
        pdb_id = seq_record.id[:-1].lower()
        if pdb_id in pdb_ids:
            seq_record.id = pdb_id
            seq_record.name = pdb_id
            wanted_seq.append(seq_record)
    return wanted_seq


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
        'fasta_file', 
        type=str,
        help='Amino acid fasta file.')

    parser.add_argument(
        'out_fasta_file', 
        type=str,
        help='Amino acid fasta file that contains the filtered amino acids')

    parser.add_argument(
        'pdb_dir', 
        type=str,
        help='Directory that contains the PDB files.')

    parser.add_argument(
        '--log', 
        type=str,
        help='Log file')

    args = parser.parse_args()

    path = args.cull_path
    mypath = path + args.pdb_dir
    pdb_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    pdb_ids = set([file.split('.')[0] for file in pdb_files])


    wanted_seq, time_str = extract_wanted_fasta(path, args.fasta_file)
    SeqIO.write(wanted_seq, path + args.out_fasta_file, "fasta")

    logging.info(
        time_str
    )