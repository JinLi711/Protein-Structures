"""
Find the matrix of the contact maps
"""


import numpy as np


def calc_residue_dist(residue_one, residue_two):
    """
    Returns the C-alpha distance between two residues.
    Return N distance if C-alpha does not exist.

    :param residue_one: single amino acid
    :type  residue_one: Bio.PDB.Residue.Residue
    :param residue_two: single amino acid
    :type  residue_two: Bio.PDB.Residue.Residue
    """

    try:
        diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
    except KeyError:
        """
        In some PDB files, it may only contain a few atoms
        in a residue. This mostly occurs near the beginning or
        the end of the PDB file because the ends oscillate 
        a lot, making it difficult for X-ray Crystollagraphy
        to capture the coordinates.
        As a result, the there are no CA atoms.
        Therefore, we will get the first possible atom: N.

        I decided to not ignore this (meaning don't
        calculate the distance at all) because even though
        not all positions of the atoms are recorded,
        it still shows up in the FASTA file of amino acid
        sequence.
        """
        diff_vector = residue_one["N"].coord - residue_two["N"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))


def calc_dist_matrix(chain_one, chain_two):
    """
    Find matrix of C-alpha distances between two chains

    :param chain_one: chain of amino acids
    :type  chain_one: Bio.PDB.Chain.Chain 
    :param chain_two: chain of amino acids
    :type  chain_two: Bio.PDB.Chain.Chain 
    """

    answer = np.zeros((len(chain_one), len(chain_two)), np.float)
    for row, residue_one in enumerate(chain_one):
        for col, residue_two in enumerate(chain_two):
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer


def get_only_residues(chain):
    """
    Given a chain, get only the amino acids.
    Removes all water molecules, ligands, etc.

    :param chain: amino acid chain
    :type  chain: Bio.PDB.Chain.Chain  
    """

    """
    HUGE PROBLEM:
        need to reinsert missing residues, elsewise
        the contact map is entirely misaligned.
        Right now, this only calculates contact 
        for existing residues.
    """
    new_chain = []
    for residue in chain:
        # http://biopython.org/DIST/docs/tutorial/Tutorial.html#htoc174
        if residue.id[0] == ' ':
            new_chain.append(residue)
    return new_chain