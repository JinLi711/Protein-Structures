"""
Visualize the results of applying the model to 
the test CASP data.

So far, we will visualize:
    contact maps (predicted, and actual)
"""


import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib


def plot_contact_maps(model, fasta_seqs, c_maps, save_dir=None):
    """
    Plot the actual contact maps
    and the predicted contact maps.

    :param model: trained keras model
    :type  model:
    :param fasta_seq: dictionary mapping PDB ID to 1 hot 
    :type  fasta_seq: dict
    :param c_maps: dictionary mapping PDB ID to c_map
    :type  c_maps: dict
    :param save_dir: directory to save plots
    :type  save_dir: str
    """

    for pdb_id, cmap in c_maps.items():
        one_hot = fasta_seqs[pdb_id]
        one_hot = one_hot.reshape((1,) + one_hot.shape)
        c_map_pred = model.predict(one_hot)

        length = one_hot.shape[1]

        c_map_pred = c_map_pred.reshape((length, length))
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
    # model_path = path + "models/"
    # test_path = path + "data/test/"
    # model_path_cull = model_path + "cull%i/" % args.cull_num
    # devtest_path = path + "data/cull5/model_data/"
    # fasta_seq_path = test_path + "casp11.fasta"
    # pdb_path = test_path + "casp11.targets_refine/"

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
        aa_dict, c_maps = em.sample_dict(aa_dict, c_maps, 10)

    plot_contact_maps(model, aa_dict, c_maps, save_dir="plots/")
    rmse = em.calc_rmse(model, aa_dict, c_maps)

    logging.info(
        "Root mean squared error: {}".format(rmse)
    )