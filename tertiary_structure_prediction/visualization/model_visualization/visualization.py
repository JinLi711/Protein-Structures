"""
Visualize the results of applying the model to 
the test CASP data.
Also, extract RR format from contact maps.

So far, we will visualize:
    contact maps (predicted, and actual)
"""


import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib
import pandas as pd


def make_prediction(model, fasta_seqs):
    """
    Given fasta sequences, make the contact map 
    predictions
    """
    
    cmap_predictions = dict()
    
    for pdb_id, aa_seq in fasta_seqs.items():
        one_hot = aa_seq.reshape((1,) + aa_seq.shape)
        c_map_pred = model.predict(one_hot)
        length = one_hot.shape[1]
        c_map_pred = c_map_pred.reshape((length, length))
        cmap_predictions[pdb_id] = c_map_pred
        
    return cmap_predictions
    
def plot_contact_maps(c_maps_preds, c_maps, save_dir=None):
    """
    Plot the actual contact maps
    and the predicted contact maps.

    :param c_maps_preds: dictionary mapping PDB ID to predicted c_maps
    :type  c_maps_preds: dict
    :param c_maps: dictionary mapping PDB ID to c_map
    :type  c_maps: dict
    :param save_dir: directory to save plots
    :type  save_dir: str
    """

    for pdb_id, cmap in c_maps.items():
        c_map_pred = c_maps_preds[pdb_id]
        
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle("PDB ID: " + pdb_id)

        ax1 = plt.subplot(221)
        ax1.set_title("Predicted: ")
        plt.imshow(c_map_pred > 0.5)

        ax2 = plt.subplot(222)
        ax2.set_title("Actual: ")
        plt.imshow(cmap)

        if save_dir != None:
            import os
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_dir + pdb_id + "_cmap.png")


def contacts_in_RR_format(contact_map, threshold=8):
    """
    Convert the contact map matrix into the RR format.
    (Note that this is currently unordered)

    Format is:

    i  j  d1  d2  p

    i, j are the indices of the residues in contact.
    i < j (since the matrix is symmetrical)

    d1 and d2 indicates the threshold for contact.
    d1 = 0, d2 = 8 Angstrom is the norm.

    p indicates the probability of the two residues 
    in contact. (0.0-1.0)
    Contacts should be listed in decreasing order

    Any pair not listed is considered to not be in contact

    See here for more information:
    http://predictioncenter.org/casp13/index.cgi?page=format

    CONFOLD server requires:
        E-mail Address
        Job Id
        Sequence
        Secondary Structure
        Contacts
        
    :param contact_map: contact matrix
    :type  contact_map: numpy array
    :param threshold: threshold of contact
    :param threshold: int
    :returns: a string in the correct format
    :rtype:   str
    """

    df = pd.DataFrame(contact_map)
    columns = df.columns

    contacts = {}
    for index, row in df.iterrows():
        for col_num, col in enumerate(columns):
            prob = row[col]
            if prob > 0.5:
                min1 = min(index, col_num)
                max1 = max(index, col_num)
                contacts[str(min1) + ' ' + str(max1)] = prob
    contact_str = ""
    for resids, prob in contacts.items():
        contact_str += resids + " 0 " + str(threshold) + " " + str(prob) + '\n'
    return contact_str


def write_out_all_predictions(
    cmaps, path='coordinate_prediction/', maxlen=500):
    """
    Write out the information for all the inputs
    required for PDB file reconstruction.

    Since Confold server only takes in proteins of maximum length 500,
    we will impose a cap.

    :param cmaps: dictionary mapping PDB ID to cmap
    :type  cmaps: dict
    :param path: path to write out
    :type  path: str
    :param 
    """

    import os

    for pdb_id, cmap in cmaps.items():
        length = int((cmap.shape[0]))

        if length < maxlen:
            if not os.path.exists(path):
                os.makedirs(path)

            out = contacts_in_RR_format(cmap)
            out_file = open(path + pdb_id + '.txt', "w+")
            out_file.write(out)
            out_file.close()


if __name__ == "__main__":
    import logging
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize the results of the trained model.')

    parser.add_argument(
        'model_path', 
        type=str,
        help='Path to the trained model weights.')

    parser.add_argument(
        'aa_path', 
        type=str,
        help='Path to the amino acid inputs. Can be preprocessed or unprocessed.')

    parser.add_argument(
        'cmap_path', 
        type=str,
        help='Path to the output. Can be preprocessed or unprocessed.')

    parser.add_argument(
        '--num_layers', 
        type=int,
        nargs='?',
        default=14,
        help='Number of layers of the second residue net of the architecture.')

    parser.add_argument(
        '--window_size', 
        type=int,
        nargs='?',
        default=3,
        help='Window size of the second residue net of the architecture.')

    parser.add_argument(
        '--preprocess', 
        help='Activate this tag to preprocess the inputs. The amino acids should be in a fasta format, and the PDB files should all be in the same directory. Otherwise, the inputs and outputs should be in a dictionary stored in a numpy format.',
        action='store_true')

    parser.add_argument(
        '--log', 
        type=str,
        help='Log file')

    args = parser.parse_args()

    logging.basicConfig(
    filename=args.log,  
    level=logging.INFO,
    filemode='w',
    format='%(levelname)s - %(filename)s \n\t %(message)s')


    path = "../../"
    sys.path.insert(0, path + "models/model_functions")
    sys.path.insert(0, path + "preprocess")

    import primary_model as pm
    import fasta_to_1_hot_encodings as fthe
    import get_contact_maps as gcm
    import visualization as tv
    import eval_metrics as em

    model = pm.create_architecture(
        args.window_size,
        args.num_layers)
    model.load_weights(args.model_path)

    if args.preprocess:
        aa_dict = fthe.convert_fasta_to_1_hot(
            args.aa_path, train=False)

        c_maps, time_str = gcm.get_contact_maps(
            args.cmap_path, train=False)
        c_maps = c_maps[0]
    else:
        aa_dict = np.load(args.aa_path)[()]
        c_maps = np.load(args.cmap_path)[()]
        # aa_dict, c_maps = em.sample_dict(aa_dict, c_maps, 10)

    c_map_preds = make_prediction(model, aa_dict)
    plot_contact_maps(c_map_preds, c_maps, save_dir="plots/")
    rmse = em.calc_rmse(c_map_preds, c_maps)

    logging.info(
        "Root mean squared error: {}".format(rmse)
    )

    write_out_all_predictions(c_map_preds)