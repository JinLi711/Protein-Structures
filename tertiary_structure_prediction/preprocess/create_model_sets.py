"""
Create the train, validation, and development test sets.

Note that the output is a dictionary mapping
the PDB ID to the amino acid sequence and contact
maps. 
"""


from random import shuffle
import os
import numpy as np
import sys


def get_items_from_dict(dict1, list1):
    """
    Create a new dictionary given a list 
    of keys.
    
    :returns: dictionary
    :rtype:   dict
    """

    dict2 = {}
    for item in list1:
        #         print(item)
        dict2[item] = dict1[item]
    return dict2


def create_train_valid_devtest_sets(
        aa,
        c_maps,
        path,
        train_size=0.7,
        valid_size=0.2):
    """
    Create the train, validation, and development test sets.
    Save the sets to the folder model_data.

    :param aa: dictionary mapping PDB ID to amino acid
               sequence in 1 hot encoded form
    :type  aa: dict
    :param c_maps: dictionary mapping PDB ID to
                   contact maps
    :type  c_maps: dict
    :param path: path to data
    :type  path: str
    :param train_size: Proportion of train set
    :type  train_size: float
    :param valid_size: Proportion of validation set
    :type  valid_size: float
    :returns: None
    """

    # note that the ids are gathered from the
    # fasta file and not the downloaded
    # pdb files
    # Will need to check whether size of aa
    # matches that of c_maps.

    if (train_size + valid_size >= 1):
        print(
            """error[create_train_valid_devtest_sets]
            (Train and validate proportion must be less than 1)
            """)
        return None

    pdb_ids = list(aa.keys())
    shuffle(pdb_ids)

    id_len = len(pdb_ids)
    train_size = round(id_len * train_size)
    valid_size = round(id_len * valid_size)

    train_list = pdb_ids[:train_size]
    valid_list = pdb_ids[train_size:train_size + valid_size]
    devtest_list = pdb_ids[train_size + valid_size:]

    train_cmap_dict = get_items_from_dict(c_maps, train_list)
    train_aa_dict = get_items_from_dict(aa, train_list)

    valid_cmap_dict = get_items_from_dict(c_maps, valid_list)
    valid_aa_dict = get_items_from_dict(aa, valid_list)

    devtest_cmap_dict = get_items_from_dict(c_maps, devtest_list)
    devtest_aa_dict = get_items_from_dict(aa, devtest_list)

    directory = path + "model_data/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save(directory + 'train_cmap_dict.npy', train_cmap_dict)
    np.save(directory + 'train_aa_dict.npy', train_aa_dict)

    np.save(directory + 'valid_cmap_dict.npy', valid_cmap_dict)
    np.save(directory + 'valid_aa_dict.npy', valid_aa_dict)

    np.save(directory + 'devtest_cmap_dict.npy', devtest_cmap_dict)
    np.save(directory + 'devtest_aa_dict.npy', devtest_aa_dict)


path = "../data/cull%i/" % int (sys.argv[1])
c_maps = np.load(path + 'contact_map_matrices.npy')[()]
aa_1_hot = np.load(path + 'amino_acids_1_hot.npy')[()]

create_train_valid_devtest_sets(
    aa_1_hot,
    c_maps,
    path
)