"""
This script moves unwanted PDB files.

We do not want PDB files with:
    multiple chains since it:
        introduces extra complications
        creates discontinuities that are hard to deal with


"""


import sys
from Bio.PDB.PDBParser import PDBParser


parser = PDBParser(PERMISSIVE=1)
path = "../data/cull%i/" % int (sys.argv[1])


def move_pdb_with_multiple_chains(path):
    """
    Move pdb files that have more than one chain 
    into folder called: removed_pdb_files
    
    :param path: path to the pdb files
    :type  path: str
    """

    from os import listdir
    from shutil import move
    from os.path import isfile, join

    mypath = path + 'pdb_files/'
    pdb_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for pdb_file in pdb_files:
        structure_id = pdb_file.split('.')[0]
        filename = path + "pdb_files/" + pdb_file
        structure = parser.get_structure(structure_id, filename)
        model = structure[0]

        if len(list(model)) != 1:
            move(mypath + pdb_file, path + "removed_pdb_files/" + pdb_file)
