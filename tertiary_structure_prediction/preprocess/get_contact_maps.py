"""
Find the contact map matrixes.
Need to insert filler residues to
make sure the chain aligns correctly
with the FASTA sequence.
"""


import sys
import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBParser import PDBParser
import warnings
warnings.filterwarnings("ignore")


parser = PDBParser(PERMISSIVE=1)
path = "../data/cull%i/" % int (sys.argv[1])

def calc_residue_dist(residue_one, residue_two):
    """
    Returns the C-alpha distance between two residues.
    Return N distance if C-alpha does not exist.

    :param residue_one: single amino acid
    :type  residue_one: Bio.PDB.Residue.Residue
    :param residue_two: single amino acid
    :type  residue_two: Bio.PDB.Residue.Residue
    :returns: the distance
    :rtype:   float
    """

    try:
        diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
        return np.sqrt(np.sum(diff_vector * diff_vector))
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

        try:
            diff_vector = residue_one["N"].coord - residue_two["N"].coord
            return np.sqrt(np.sum(diff_vector * diff_vector))
        except KeyError:
            """
            If neither exists, set distance to infinity.
            Then contact will always be false.
            May need to change this later,
            though not sure about the alternatives.
            """
            return np.inf


def calc_dist_matrix(chain_one, chain_two):
    """
    Find matrix of C-alpha distances between two chains

    :param chain_one: chain of amino acids
    :type  chain_one: Bio.PDB.Chain.Chain
    :param chain_two: chain of amino acids
    :type  chain_two: Bio.PDB.Chain.Chain
    :returns: matrix
    :rtype:   numpy array
    """

    answer = np.zeros((len(chain_one), len(chain_two)), np.float)
    for row, residue_one in enumerate(chain_one):
        for col, residue_two in enumerate(chain_two):
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer


def get_only_residues(chain, allowed_residues):
    """
    Given a chain, get only the amino acids.
    Removes all water molecules, ligands, etc.

    :param chain: amino acid chain
    :type  chain: Bio.PDB.Chain.Chain
    :param allowed_residues: list of allowed residues
    :type  allowed_residues: list
    :returns: list of residues
    :rtype:   list
    """

    allowed_ids = ["H_" + residue for residue in allowed_residues]
    allowed_ids.append(' ')
    allowed_ids = set(allowed_ids)
    new_chain = []
    for residue in chain:
        # http://biopython.org/DIST/docs/tutorial/Tutorial.html#htoc174
        residue_id = residue.id[0]
        if residue_id in allowed_ids:
            new_chain.append(residue)
    return new_chain


def find_missing_resid(line):
    """
    Given a line from PDB file, check if it is 
    the line that indicates a missing residue.
    
    example of pdb lines:
    
    REMARK 465   M RES C SSSEQI                                                     
    REMARK 465     GLN A   333                                                      
    REMARK 465     SER A   446                                                      
    REMARK 465     GLN A   447
    
    :param line: line from PDB file
    :type  line: str
    :returns: position of the missing residue
    :rtype:   int
    """

    if line[0:10] == 'REMARK 465':
        split = [line[:10], line[15:18], line[21:26]]
        # RES is one of the header label
        if split[1] != "RES":
            try:
                missing_resid = int(split[2])
                return missing_resid
            except ValueError:
                return None
    else:
        return None


def find_mutated_resid(line):
    """
    Given a line from PDB file, check if it is 
    the line that indicates a mutated residue.

    example of PDB line:

    MODRES 1B0B SAC A    1  SER  N-ACETYL-SERINE

    :param line: line from PDB file
    :type  line: str
    :returns: 3 letter code for the mutated residue
    :rtype:   string
    """

    if line[:6] == 'MODRES':
        mutated_resid = line.split()[2]
    else:
        mutated_resid = None
    return mutated_resid


def parse_pdb_file(path, file_name):
    """
    Parse a PDB file again for information that is needed.

    Will extract:
        missing residues
        mutated residues  

    :param path: path to data
    :type  path: str
    :param file_name: name of PDB file
    :type  file_name: str
    :returns: dictionary mapping extracted info to its info
    :rtype:   dict
    """

    missing_resids = []
    mutated_resids = []
    file = open(path + file_name, 'r').readlines()
    for line in file:
        missing_resid = find_missing_resid(line)
        missing_resids.append(missing_resid)

        mutated_resid = find_mutated_resid(line)
        mutated_resids.append(mutated_resid)

    def remove_none(list1):
        rm_none = set(list1).difference({None})
        return list(rm_none)

    missing_resids = remove_none(missing_resids)
    mutated_resids = remove_none(mutated_resids)

    parsed_info = {}
    parsed_info["Missing Residues"] = missing_resids
    parsed_info["Mutated Residues"] = mutated_resids

    return parsed_info


def insert_missing_residues(residues, parse_info):
    """
    Create a fake residue to fill in
    missing residues. The xyz coordinates will
    be infinity (so it will always output no contact).

    :param residues: list of only residues
    :type  residues: list
    :param parse_info: dictionary mapping extracted info to its info
    :type  parse_info: dict
    :returns: list of residues with inserts
    :rtype:   list
    """

    new_residues = residues
    missing_resid_position = parse_info["Missing Residues"]

    for position in missing_resid_position:
        new_residue = Residue((' ', position, ' '), "GLY", '    ')
        new_atom = Atom(
            name="CA",
            coord=np.array([np.inf, np.inf, np.inf]),
            bfactor=0,
            occupancy=0,
            altloc=' ',
            fullname=" CA ",
            serial_number=0
        )
        new_residue.add(new_atom)
        new_residues.append(new_residue)

    new_residues.sort()
    return new_residues


def get_contact_map(model, path, pdb_file, cutoff, contact=True):
    """
    Parse a PDB file and
    get the contact map.
    If contact is set to false,
    return only the distance matrix.

    :param model: Model from Biopython PDB
    :type  model: Bio.PDB.Model.Model
    :param path: path to data
    :type  path: str
    :param pdb_file: name of pdb file
    :type  pdb_file: str
    :param cutoff: cutoff for contact distance
    :type  cutoff: float
    :param contact: output contact map or distance map
    :type  contact: bool
    :returns: either a distance matrix or contact matrix
    :rtype:   numpy array
    """

    # Note that we are only getting the first chain
    # We will only get training data for proteins with
    # a single chain.
    chain = list(model)[0]

    parse_info = parse_pdb_file(path, pdb_file)

    residues = get_only_residues(chain, parse_info["Mutated Residues"])

    # add in missing residues
    residues = insert_missing_residues(residues, parse_info)
    dist_matrix = calc_dist_matrix(
        residues,
        residues
    )

    if contact:
        contact_map = dist_matrix < cutoff
    return contact_map


def get_contact_maps(path, cutoff=12.0):
    """
    Create a dictionary mapping the PDB ID
    to its contact map.

    :param path: path to data
    :type  path: str
    :param cutoff: cutoff for contact distance
    :type  cutoff: float
    :returns: dict mapping PDB file to contact matrix
    :rtype:   dict
    """

    from os import listdir
    from os.path import isfile, join

    mypath = path + 'pdb_files/'
    pdb_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    contact_maps = {}

    # contains pdb files with more than one chain
    # will not calculate contact map if it has more than one chain
    other_pdb_files = []

    for pdb_file in pdb_files:
        pdb_id = pdb_file.split('.')[0]
        print("PDB File: ", pdb_file)

        structure_id = pdb_file.split('.')[0]
        filename = path + "pdb_files/" + pdb_file
        structure = parser.get_structure(structure_id, filename)
        model = structure[0]

        if len(list(model)) == 1:
            c_map = get_contact_map(
                model,
                mypath,
                pdb_file,
                cutoff=cutoff
            )
            contact_maps[pdb_id] = c_map
        else:
            print("\tThis protein has more than 1 chain.")
            other_pdb_files.append(pdb_file)

    return contact_maps


c_maps = get_contact_maps(path)
np.save(path + 'contact_map_matrices.npy', c_maps)