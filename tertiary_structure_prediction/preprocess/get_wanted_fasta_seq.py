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


path = "../data/cull%i/" % int (sys.argv[1])
mypath = path + 'pdb_files/'
pdb_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
pdb_ids = set([file.split('.')[0] for file in pdb_files])


def extract_wanted_fasta(path):
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
    for seq_record in SeqIO.parse(path + "amino_acids.fasta", "fasta"):
        # remove the last letter, which describes the chain
        pdb_id = seq_record.id[:-1].lower()
        if pdb_id in pdb_ids:
            seq_record.id = pdb_id
            seq_record.name = pdb_id
            wanted_seq.append(seq_record)
    return wanted_seq


if __name__ == "__main__":
    wanted_seq = extract_wanted_fasta(path)
    SeqIO.write(wanted_seq, path + "wanted_aa.fasta", "fasta")