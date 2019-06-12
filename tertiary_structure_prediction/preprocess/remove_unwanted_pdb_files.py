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
import helper_func as hf


# it's just discontinues chains warnings
warnings.filterwarnings("ignore")

@hf.timing_val
def move_pdb_file(path, pdb_parser, rm_file_dir, pdb_dir):
    """
    Move pdb file if:
        it has more than one chain 
        does not end with .pdb

    The file is moved to folder called: 
        removed_pdb_files
    
    :param path: path to the pdb files
    :type  path: str
    :param pdb_parser: a parser for pdb files
    :type  pdb_parser:
    :param rm_file_dir: directory to send unwanted files
    :type  rm_file_dir: str
    :param pdb_dir: directory that contains the pdb files
    :type  pdb_dir: str
    """

    mypath = path + pdb_dir
    pdb_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    non_pdb_files = []
    unwanted_pdb_files = []

    for pdb_file in pdb_files:
        splited_file = pdb_file.split('.')
        structure_id = splited_file[0]
        extension = splited_file[1]

        if (extension != 'pdb'):
            move(mypath + pdb_file, path + rm_file_dir + pdb_file)
            non_pdb_files.append(pdb_file)
            continue

        filename = path + pdb_dir + pdb_file
        structure = pdb_parser.get_structure(structure_id, filename)
        model = structure[0]

        if (len(list(model)) != 1):
            move(mypath + pdb_file, path + rm_file_dir + pdb_file)
            unwanted_pdb_files.append(structure_id)

    return non_pdb_files, unwanted_pdb_files

if __name__ == "__main__":

    import logging
    import argparse
    from os.path import basename

    parser = argparse.ArgumentParser(
        description='Remove unwanted pdb files')

    parser.add_argument(
        'cull_path', 
        type=str,
        help='The path to the cull directory')

    parser.add_argument(
        'rm_dir', 
        type=str,
        help='Location of removed PDB files')

    parser.add_argument(
        'pdb_dir', 
        type=str,
        help='Location of PDB files')

    parser.add_argument(
        '--log', 
        type=str,
        help='Log file')

    args = parser.parse_args()
    
    pdb_parser = PDBParser(PERMISSIVE=1)
    path = args.cull_path
    (non_pdb_files, unwanted_pdb_files), time_str = move_pdb_file(path, pdb_parser, args.rm_dir, args.pdb_dir)

    logging.info(
        "Non pdb files: {}\n\t Unwanted PDB files: {}".format(
            non_pdb_files, 
            unwanted_pdb_files)
    )

    logging.info(
        time_str
    )
