"""
Create the train, validation, and development test sets.
Do not include any mismatched input and output sizes.
Record which PDB ID had mismatches.

Note that the output is a dictionary mapping
the PDB ID to the amino acid sequence and contact
maps. 
"""


from random import shuffle
import os
import numpy as np
import sys
import helper_func as hf

def remove_mismatches(aa, c_map):
    """
    If input and output dimensions do not match, 
    remove the PDB ID, its 1 hot encoding, and
    its contact map.

    :param aa: dictionary mapping PDB ID to amino acid
               sequence in 1 hot encoded form
    :type  aa: dict
    :param c_maps: dictionary mapping PDB ID to
                   contact maps
    :returns: (input, output) with removed mismatches, 
              list of pdb ids with mismatches
    :rtype:   (dict, dict, list)
    """

    mismatched_dim_ids = []
    inputs = aa.copy()
    outputs = c_map.copy()

    for pdb_id, one_hot in aa.items():
        aa_dim = one_hot.shape[0]
        cmap_dim = outputs[pdb_id].shape

        if not ((aa_dim == cmap_dim[0]) and aa_dim == cmap_dim[1]):
            inputs.pop(pdb_id)
            outputs.pop(pdb_id)
            mismatched_dim_ids.append(pdb_id)

    return inputs, outputs, mismatched_dim_ids


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
        aa_file,
        c_map_file,
        npy_path,
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
    :param aa_file: file name for aa 1 hot encodings
    :type  aa_file: str
    :param c_map_file: file name for cmaps
    :type  c_map_file: str
    :param npy_path: path to store the splited data
    :type  npy_path: str
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
        raise ValueError(
            "Train and validate proportion must be less than 1"
        )

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

    directory = path + npy_path
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save(directory + 'train_' + c_map_file, train_cmap_dict)
    np.save(directory + 'train_' + aa_file, train_aa_dict)

    np.save(directory + 'valid_' + c_map_file, valid_cmap_dict)
    np.save(directory + 'valid_' + aa_file, valid_aa_dict)

    np.save(directory + 'devtest_' + c_map_file, devtest_cmap_dict)
    np.save(directory + 'devtest_' + aa_file, devtest_aa_dict)


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
        'c_map', 
        type=str,
        help='Numpy file for contact map matrices')

    parser.add_argument(
        'aa', 
        type=str,
        help='Numpy file for amino acid 1 hot encodings.')

    parser.add_argument(
        'data_dir', 
        type=str,
        help='Directory contained the split train, validate, and test set.')

    parser.add_argument(
        '--log', 
        type=str,
        help='Log file')

    args = parser.parse_args()

    path = args.cull_path
    c_maps = np.load(path + args.c_map)[()]
    aa_1_hot = np.load(path + args.aa)[()]

    aa_1_hot, c_maps, mismatched_dim_ids = remove_mismatches(aa_1_hot, c_maps)

    create_train_valid_devtest_sets(
        aa_1_hot,
        c_maps,
        path,
        args.aa,
        args.c_map,
        args.data_dir
    )

    logging.info(
        "PDB IDS with mismatched aa/PDB dimensions: {}".format(
            mismatched_dim_ids
        )
    )