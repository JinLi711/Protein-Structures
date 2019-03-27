"""
Given a culled list of PDB files 
from Dunbrack's server, 
get only the PDB IDs, ignoring the 
length, Exptl., resolution, R-factor, FreeRvalue,
and chain.

Output a pdb_ids.txt
"""


import sys
from collections import defaultdict


def find_ids_with_single_chain(pdb_ids):
    """
    Given a list of pdb_ids with chains attached to them,
    find the ids that have multiple chains and the ids
    that have a single chain.

    Note, just because a PDB ID has a single chain here,
    it does not mean that the PDB file has only one chain.
    This is because when culling, there's a threshold of protein
    chain lengths, and the culling ignores chains that are 
    not within the threshold.

    This is mainly used to reduce the amount of downloadings
    required, since we are not going to use proteins with
    multiple chains.

    :param pdb_ids: list of pdb ids with its chain
    :type  pdb_ids: list
    :returns: list of ids with only one chain
    :rtype:   list
    """

    pdb_ids_and_chains = defaultdict(lambda: [])
    for pdb_id in pdb_ids:
        id_only = pdb_id[:4]
        current_chains = pdb_ids_and_chains[id_only]
        pdb_ids_and_chains[id_only] = current_chains + [pdb_id[4:]]

    one_chain_ids = []
    multiple_chains_ids = []

    for pdb_id, chains in pdb_ids_and_chains.items():
        if len(chains) == 1:
            one_chain_ids.append(pdb_id)
        else:
            multiple_chains_ids.append(pdb_id)
    return one_chain_ids


if __name__ == "__main__":
    path = "../data/cull%i/" % int (sys.argv[1])

    with open(path + "pdb_ids.dat") as f:
        pdb_ids_with_chain = [line.split()[0] for line in f][1:] 

    pdb_ids = find_ids_with_single_chain(pdb_ids_with_chain) 
    pdb_ids = ','.join(pdb_ids)

    pdb_ids_with_chain = ','.join(pdb_ids_with_chain)

    out_file = open(path + "pdb_ids.txt", 'w') 
    out_file.write(pdb_ids)
    out_file.close()