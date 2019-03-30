"""
Visualize the results of applying the model to 
the test CASP data.

So far, we will visualize:
    contact maps (predicted, and actual)
"""


import tensorflow as tf
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import importlib


def plot_contact_maps(model, fasta_seqs, c_maps, save_dir="plots/"):
    """
    If possible, plot the actual contact maps
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

        plt.savefig(save_dir + pdb_id + "cmap.png")


if __name__ == "__main__":
    import sys 

    path = "../../"
    model_path = path + "models/"
    test_path = path + "data/test/"
    model_path_cull = model_path + "cull%i/" % int (sys.argv[1])
    sys.path.insert(0, model_path + "model_functions")
    sys.path.insert(0, path + "preprocess")

    import primary_model as pm
    import fasta_to_1_hot_encodings as fthe
    import get_contact_maps as gcm

    model = tf.keras.models.load_model(
        model_path_cull + 'my_model.h5',
        custom_objects={"OuterProduct": pm.OuterProduct()}
    )

    fasta_seq_path = test_path + "casp11.fasta"
    pdb_path = test_path + "casp11.targets_refine/"

    fasta_seqs = fthe.convert_fasta_to_1_hot(
        fasta_seq_path, 
        train=False
    )

    c_maps = gcm.get_contact_maps(
        pdb_path, 
        train=False
    )


    plot_contact_maps(model, fasta_seqs, c_maps)