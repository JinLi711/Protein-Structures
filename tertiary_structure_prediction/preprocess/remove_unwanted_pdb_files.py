"""
This script moves unwanted PDB files.

We do not want PDB files with:
    multiple chains since it:
        introduces extra complications
        creates discontinuities that are hard to deal with
    does not end in .pdb


"""


import sys
from os import listdir
from shutil import move
from os.path import isfile, join
from Bio.PDB.PDBParser import PDBParser
import warnings 


# it's just discontinues chains warnings
warnings.filterwarnings("ignore")


def move_pdb_file(path):
    """
    Move pdb file if:
        it has more than one chain 
        does not end with .pdb

    The file is moved to folder called: 
        removed_pdb_files
    
    :param path: path to the pdb files
    :type  path: str
    """

    mypath = path + 'pdb_files/'
    pdb_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for pdb_file in pdb_files:
        splited_file = pdb_file.split('.')
        structure_id = splited_file[0]
        extension = splited_file[1]

        if (extension != 'pdb'):
            move(mypath + pdb_file, path + "removed_pdb_files/" + pdb_file)

        filename = path + "pdb_files/" + pdb_file
        structure = parser.get_structure(structure_id, filename)
        model = structure[0]

        if (len(list(model)) != 1):
            move(mypath + pdb_file, path + "removed_pdb_files/" + pdb_file)


if __name__ == "__main__":
    parser = PDBParser(PERMISSIVE=1)
    path = "../data/cull%i/" % int (sys.argv[1])
    move_pdb_file(path)
