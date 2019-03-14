"""
Find the matrix of the contact maps
"""


import numpy as np


def calc_residue_dist(residue_one, residue_two):
    """
    Returns the C-alpha distance between two residues

    :param residue_one: single amino acid
    :type  residue_one: Bio.PDB.Residue.Residue
    :param residue_two: single amino acid
    :type  residue_two: Bio.PDB.Residue.Residue
    """

    diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
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

    new_chain = []
    for residue in chain:
        # http://biopython.org/DIST/docs/tutorial/Tutorial.html#htoc174
        if residue.id[0] == ' ':
            new_chain.append(residue)
    return new_chain