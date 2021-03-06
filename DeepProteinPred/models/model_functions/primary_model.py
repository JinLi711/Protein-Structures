"""
This module contains the network used in:

*Wang S, Sun S, Li Z, Zhang R, Xu J (2017) 
Accurate De Novo Prediction of Protein Contact Map by 
Ultra-Deep Learning Model. PLoS Comput Biol 13(1): e1005324. 
https://doi.org/10.1371/journal.pcbi.1005324*

Here are some attributes they used in the paper:

Inputs:
    Along with 1 hot encoding of amino acids to a 20 dimension
    1 hot encoding, they also included another 6 dimensions,
    3-state secondary structure and 3-state solvent accessibility.
    This was predicted using another neural network.
    pairwise features: 
        mutual information, 
        the EC information calculated by CCMpred, 
        and pair- wise contact potential
    were concatenated after the outer product layer

Activation layer:
    ReLU after every layer.
    Batch normalization before activation layer.
    (though did not say whether this was after 
    or before the convolution layer)

Residual network:
    the number of features of the next layer is greater or
    equal to the one below it, so they had to pad
    the previous layer with zeros to allow the skip adding.
    For 1D residual network:
        window size: 17 (fixed)
        number of layers: 6 (fixed)
    For 2D residual network:
        window size: (3,3) or (5,5)
        number of layers: ~60
        number of hidden neurons per layer: ~60

Loss function:
    negative log-likelihood averaged over all 
    the residue pairs of the training proteins.
    Since outputs were unbalanced, they assign a larger 
    weight to the residue pairs forming a contact.
    The weight is assigned such that the total weight 
    assigned to contacts is approximately 1/8 of the number 
    of non-contacts in the training set.

Mini-batches:
    can have mini-batches, but they sorted the training set 
    and then grouped batches by related size. 
    Then they did some extra padding to make sure all proteins
    in the batch had the same size.

Others:
    L2 normalization
    stochastic gradient descent
    drop out was never mentioned
    20-30 epochs


Things that I did not implement, even though I wish I did:
    * the extra six dimensions added to input
    * pairwise features
    * different layer sizes in residual network
    used binary crossentropy instead of log
    * my weighing was simply 8 times higher for contact than
    noncontact
    * did not have 60 layers for second residual network. 
    (Too many parameters, memory exploded, not sure 
    how to deal with this yet).
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import tensorflow as tf


#----------------------------------------------------------------------
# Special Layers
#----------------------------------------------------------------------


class OuterProduct(tf.keras.layers.Layer):
    """
    Given a layer of size (B, L, N), create 
    a layer of size (B, L, L, 3N).
    If we have {v1, ..., vm},
    for the i, j entry, we have (vi, v((i+j)/2), vj).
    """

    def __init__(self, **kwargs):
        super(OuterProduct, self).__init__()

    def call(self, incoming):
        """
        Create the layer.

        :param incoming: tensor of size (B, L, N)
        :type  incoming: tensorflow.python.framework.ops.Tensor
        :returns: tensor of size (B, L, L, 3N)
        :rtype:   tensorflow.python.framework.ops.Tensor
        """

        L = tf.shape(incoming)[1]
        # save the indexes of each position
        v = tf.range(0, L, 1)

        i, j = tf.meshgrid(v, v)

        m = tf.cast((i+j)/2, tf.int32)

        # switch batch dim with L dim to put L at first
        incoming2 = tf.transpose(incoming, perm=[1, 0, 2])

        # full matrix i with element in incomming2 indexed i[i][j]
        out1 = tf.nn.embedding_lookup(incoming2, i)
        out2 = tf.nn.embedding_lookup(incoming2, j)
        out3 = tf.nn.embedding_lookup(incoming2, m)

        # concatanate final feature dim together
        out = tf.concat([out1, out2, out3], axis=3)
        # return to original dims
        output = tf.transpose(
            out,
            perm=[2, 0, 1, 3],
            name="outer_product"
        )
        return output


class OuterProduct2(tf.keras.layers.Layer):
    """
    Given a layer of size (B, L, N), create 
    a layer of size (B, L, L, N).

    This is done by:
        switching B and L : (L, B, N)
        extending the dimension: (1, L, B, N)
        getting some random index: (L, 1)
        computing tensorproduct: (L, 1) x (1, L, B, N)
    """

    def __init__(self, **kwargs):
        super(OuterProduct2, self).__init__()

    def call(self, incoming):
        """
        Create the layer.

        :param incoming: tensor of size (B, L, N)
        :type  incoming: tensorflow.python.framework.ops.Tensor
        :returns: tensor of size (B, L, L, N)
        :rtype:   tensorflow.python.framework.ops.Tensor
        """

        # switch batch dim with L dim to put L at first
        incoming2 = tf.transpose(incoming, perm=[1, 0, 2])

        # get a random index value at tensor index 2 and 3
        L2 = tf.shape(incoming2)[1]
        L3 = tf.shape(incoming2)[2]
        index2 = tf.random.uniform((1,), dtype="int32", maxval=(L2))[0]
        index3 = tf.random.uniform((1,), dtype="int32", maxval=(L3))[0]

        # compute the tensor product
        inputa = tf.expand_dims(incoming2[:, index2][:, index3], 1)
        incoming3 = tf.expand_dims(incoming2, 0)
        tensorproduct = tf.tensordot(inputa, incoming3, axes=1)
        tensorproduct = tf.transpose(
            tensorproduct,
            perm=[2, 0, 1, 3],
            name="outer_product2"
        )

        return tensorproduct


def residual_conv_block(
        x,
        convnet,
        stride,
        num_layers,
        regularizer=None,
        activation="relu",
        padding="same"):
    """
    Create a residual convolution block, either 
    in 1 or 2 dimensions.

    :param x: 
    :type x:  
    :param convnet: indicates which type of layer to use
    :type  convnet: string
    :param stride: stride
    :type  stride: int
    :param num_layers: number of layers for the entire residual network
    :type  num_layers: int
    :param regularizer: 
    :type  regularizer: 
    :param activation: 
    :type  activation: str
    :param padding:
    :type  padding: str
    :returns: result of the residual network
    :rtype:   tensorflow.python.framework.ops.Tensor
    """

    size = int(x.shape[-1])
    y = x

    if num_layers % 2 != 0:
        raise ValueError("The number of layers must be even")

    def one_dim_block(x, i):
        """
        Create the duo layer for conv1d.

        :param x: input
        :type  x: tensorflow.python.framework.ops.Tensor
        :param i: position of that duo layer
        :type  i: int
        :returns: 
        :rtype:   tensorflow.python.framework.ops.Tensor
        """

        i += 1
        z = tf.keras.layers.Conv1D(
            size,
            stride,
            activation=activation,
            padding=padding,
            kernel_regularizer=regularizer,
            name=convnet + "_layer{}a".format(i),
        )(x)
        z = tf.keras.layers.BatchNormalization(
            name=convnet + "_batch_norm{}a".format(i),
        )(z)

        z = tf.keras.layers.Conv1D(
            size,
            stride,
            activation=activation,
            padding=padding,
            kernel_regularizer=regularizer,
            name=convnet + "_layer{}b".format(i),
        )(z)
        z = tf.keras.layers.BatchNormalization(
            name=convnet + "_batch_norm{}b".format(i),
        )(z)

        z = tf.keras.layers.add(
            [z, x],
            name=convnet + "_residual_block{}".format(i)
        )

        return z

    def two_dim_block(x, i):
        """
        Create the duo layer for conv2d.

        :param x: input
        :type  x: tensorflow.python.framework.ops.Tensor
        :param i: position of that duo layer
        :type  i: int
        :returns: 
        :rtype:   tensorflow.python.framework.ops.Tensor
        """

        i += 1
        z = tf.keras.layers.Conv2D(
            size,
            stride,
            activation=activation,
            padding=padding,
            kernel_regularizer=regularizer,
            name=convnet + "_layer{}a".format(i),
        )(x)
        z = tf.keras.layers.BatchNormalization(
            name=convnet + "_batch_norm{}a".format(i),
        )(z)

        z = tf.keras.layers.Conv2D(
            size,
            stride,
            activation=activation,
            padding=padding,
            kernel_regularizer=regularizer,
            name=convnet + "_layer{}b".format(i),
        )(z)
        z = tf.keras.layers.BatchNormalization(
            name=convnet + "_batch_norm{}b".format(i),
        )(z)

        z = tf.keras.layers.add(
            [z, x],
            name=convnet + "_residual_block{}".format(i)
        )

        return z

    if convnet == "1d_convnet":
        for i in range(int(num_layers / 2)):
            y = one_dim_block(y, i)

    elif convnet == "2d_convnet":
        for i in range(int(num_layers / 2)):
            y = two_dim_block(y, i)

    else:
        raise ValueError("Not an available convnet dimension")

    return y


def inception_module (x):
    """
    Create the inception V3 module
    
    :param x: Tensor input continuing from the chain
    :type  x: tensorflow.python.framework.ops.Tensor
    
    :return: The concatenated output of the four branches
    :rtype:  tensorflow.python.framework.ops.Tensor
    """

    # from tensorflow.keras import layers

    branch_a = tf.keras.layers.Conv2D(
        128,
        1,
        activation='relu',
        strides=2,
        padding="same"
    )(x)

    branch_b = tf.keras.layers.Conv2D(
        128,
        1,
        activation='relu'
    )(x)
    branch_b = tf.keras.layers.Conv2D(
        128,
        3,
        activation='relu',
        strides=2,
        padding="same"
    )(branch_b)

    branch_c = tf.keras.layers.AveragePooling2D(
        3,
        strides=2,
        padding="same"
    )(x)
    branch_c = tf.keras.layers.Conv2D(
        128,
        3,
        activation='relu',
        padding="same"
    )(branch_c)

    branch_d = tf.keras.layers.Conv2D(
        128,
        1,
        activation='relu'
    )(x)
    branch_d = tf.keras.layers.Conv2D(
        128,
        3,
        activation='relu',
        padding="same"
    )(branch_d)
    branch_d = tf.keras.layers.Conv2D(
        128,
        3,
        activation='relu',
        strides=2,
        padding="same"
    )(branch_d)

    output = tf.keras.layers.concatenate(
        [branch_a, branch_b, branch_c, branch_d],
        name="Inception_V3"
    )
    return output


#----------------------------------------------------------------------
# Functions Wrapped in Lambda Layers
#----------------------------------------------------------------------


def drop_last_dim(x):
    """
    Reshape a tensor of size (None, None, None, 1)
    to (None, None, None)

    :param x: input
    :type  x: tensorflow tensor
    :returns: reshaped input
    :rtype:   tensorflow tensor
    """

    x_shape = tf.keras.backend.shape(x)
    x_shape = x_shape[:-1]
    return tf.keras.backend.reshape(x, x_shape)


#----------------------------------------------------------------------
# Generators
#----------------------------------------------------------------------


def aa_generator_batch(x, y, batch_size=1):
    """
    A generator for batches of sorted size.

    :param x: dictionary mapping keys to aa 1 hot
    :type  x: dict
    :param y: dictionary mapping keys to cmaps
    :type  y: dict
    :param batch_size: size of batch
    :type  batch_size: int
    :returns: aa batch, cmap batch
    :rtype:   (np.array, np.array)
    """

    aa = [(pdb_id, array) for (pdb_id, array) in x.items()]
    aa.sort(
        key=lambda x: x[1].shape[0],
        reverse=True
    )
    cmaps = [(pdb_id, array) for (pdb_id, array) in y.items()]
    cmaps.sort(
        key=lambda x: x[1].shape[0],
        reverse=True
    )

    def check_if_aligned(x, y):
        """
        Check if the sorting is aligned.

        :param x: list of tuples (a, c)
        :type  x: list
        :param y: list of tuples (a, b)
        :type  y: list
        """

        if len(x) != len(y):
            raise ValueError("Lengths do not match.")

        for i in range(len(x)):
            if (x[i][0] != y[i][0]):
                raise ValueError("Not sorted correctly")

    check_if_aligned(aa, cmaps)

    def create_batches(aa, cmaps, batch_size):
        """
        Create the batches.

        :param aa: list of tuple (pdb_id, aa 1 hot)
        :type  aa: list
        :param cmaps: list of tuple (pdb_id, cmap)
        :type  cmaps: list
        :param batch_size: size of batch
        :type  batch_size: int
        :returns: dictionary mapping a key to a tuple
                  (aa batch, cmap batch)
        :rtype:   dict
        """

        all_batches = {}
        for i in range(0, len(aa), batch_size):
            aa_batch = aa[i:i+batch_size]
            cmap_batch = cmaps[i:i+batch_size]
            max_length = aa[i][1].shape[0]

            aa_batch = [
                np.pad(
                    array,
                    ((0, max_length - array.shape[0]),
                     (0, 0)),
                    'constant')
                for (pdb_id, array) in aa_batch
            ]

            cmap_batch = [
                np.pad(
                    array,
                    ((0, max_length - array.shape[0]),
                     (0, max_length - array.shape[1])),
                    'constant')
                for (pdb_id, array) in cmap_batch
            ]

            # example:
            # shape of (4,5,6) becomes (4,30)
            # equivalent to tf.flatten
            cmap_batch = [
                batch.flatten()
                for batch in cmap_batch
            ]  

            # shape of (4,30) becomes (4,30,1)
            cmap_batch = [
                np.expand_dims(batch, axis=2)
                for batch in cmap_batch
            ]
            
                
            stacked_aa_batch = np.stack(aa_batch, axis=0)
            stacked_cmap_batch = np.stack(cmap_batch, axis=0)

            all_batches["batch" +
                        str(i)] = (stacked_aa_batch, stacked_cmap_batch)

        return all_batches

    all_batches = create_batches(aa, cmaps, batch_size)
    keys = set(all_batches.keys())
    

    while True:
        try:
            key = random.sample(keys, 1)[0]
            keys.remove(key)
            aa_batch, cmap_batch = all_batches[key]


            class_weight = np.zeros(
                cmap_batch.shape
            )

            class_weight += cmap_batch * 8
            class_weight[class_weight == 0] = 1
            class_weight = class_weight.reshape(
                (class_weight.shape[:2])
            )

            yield aa_batch, cmap_batch, class_weight

        except ValueError:
            # if out of keys, reinsert back the keys
            keys = set(all_batches.keys())


#----------------------------------------------------------------------
# Create the model
#----------------------------------------------------------------------


def create_architecture(
    resid_layer2_window_size, 
    resid_layer2_num_layers):
    """
    Create the basic architecture. 
    1d residual network followed by 2d residual network.

    :param resid_layer2_window_size: window size
    :type  resid_layer2_window_size: int
    :param resid_layer2_num_layers: number of layers
    :type  resid_layer2_num_layers: int
    :returns: training model
    :rtype:   tensorflow.python.keras.engine.training.Model
    """

    input_tensor = tf.keras.Input(
        shape=(None, 20),
        name="input_layer"
    )

    x = residual_conv_block(
        input_tensor,
        "1d_convnet",
        17,
        num_layers=6,
        regularizer=tf.keras.regularizers.l2(0.001)
    )

    x = OuterProduct(
    )(x)

    # x = tf.keras.layers.Conv2D(
    #     60,
    #     1,
    #     activation='relu',
    #     padding='same',
    #     kernel_regularizer=tf.keras.regularizers.l2(0.001)
    # )(x)

    x = residual_conv_block(
        x,
        "2d_convnet",
        resid_layer2_window_size,
        num_layers=resid_layer2_num_layers,
        regularizer=tf.keras.regularizers.l2(0.001)
    )

    x = tf.keras.layers.Conv2D(
        1,
        1,
        activation='relu',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)

    x = tf.keras.layers.BatchNormalization(
    )(x)

    x = tf.keras.layers.Flatten(
    )(x)

    # x = tf.expand_dims(
    #     x,
    #     axis=2
    # )

    x = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(x, axis=2)
    )(x)

    x = tf.keras.layers.Activation(
        activation="sigmoid"
    )(x)

    # x = tf.keras.layers.Lambda(
    #     lambda x: drop_last_dim(x)
    # )(x)

    # x = tf.keras.layers.Dropout(
    #     0.5,
    #     name="Drop-Out"
    # )(x)

    model = tf.keras.models.Model(
        input_tensor,
        x
    )

    return model


#----------------------------------------------------------------------
# Callbacks
#----------------------------------------------------------------------


weight_path = "best_weight/{}_weights.best.hdf5".format(
    'Double_Resid_Network'
)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    weight_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=True
)

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='Logs',
)

reduceLROnPlat = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    verbose=1,
    mode='min',
    min_delta=0.0001,
    cooldown=2,
    min_lr=1e-7
)


def step_decay(epoch):
    """
    Reduce learning rate after epochs.
    """

    import math 

    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(
        drop, math.floor((1+epoch)/epochs_drop)
    )
    return lrate

lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)

early = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    # monitor='acc'
    mode="min",
    verbose=2,
    # training is interrupted when the monitor argument 
    # stops improving after n steps
    patience=3
)

callbacks_list = [
    checkpoint, 
    early, 
    reduceLROnPlat,
    tensorboard
]



if __name__ == "__main__":

    import sys
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description='Train the model')

    parser.add_argument(
        'cull_num', 
        type=str,
        help='Cull number')

    parser.add_argument(
        'c_map', 
        type=str,
        help='Numpy file for contact map matrices')

    parser.add_argument(
        'aa', 
        type=str,
        help='Numpy file for amino acid 1 hot encodings.')

    parser.add_argument(
        'save_model', 
        type=str,
        help='File to save trained model')

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
        '--batch_size', 
        type=int,
        nargs='?',
        default=1,
        help='Batch size for training.')

    parser.add_argument(
        '--epochs', 
        type=int,
        nargs='?',
        default=20,
        help='Number of epochs to train for.')

    parser.add_argument(
        '--log', 
        type=str,
        help='Log file')

    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log,  
        level=logging.INFO,
        format='%(levelname)s - %(filename)s \n\t %(message)s')


    data_path = "../../data/cull{}/model_data/".format(args.cull_num)

    cmap_matrices = args.c_map
    aa_1_hot_matrix = args.aa
    
    # devtest_aa_dict = np.load(data_path + 'devtest_' + aa_1_hot_matrix)[()]
    # devtest_cmap_dict = np.load(data_path + 'devtest_' + cmap_matrices)[()]
    train_aa_dict = np.load(data_path + 'train_' + aa_1_hot_matrix)[()]
    train_cmap_dict = np.load(data_path + 'train_' + cmap_matrices)[()]
    valid_aa_dict = np.load(data_path + 'valid_' + aa_1_hot_matrix)[()]
    valid_cmap_dict = np.load(data_path + 'valid_' + cmap_matrices)[()]


    model = create_architecture(
        int(args.window_size), 
        int(args.num_layers)
    )

    logging.info(
        model.summary()
    )

    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    train_steps = round(len(train_aa_dict) / batch_size)
    valid_steps = round(len(valid_aa_dict) / batch_size)

    logging.info(
        "Batch size: {}".format(batch_size)
    )
    logging.info(
        "Epochs: {}".format(epochs)
    )
    logging.info(
        "Train steps: {}".format(train_steps)
    )
    logging.info(
        "Validation steps: {}".format(valid_steps)
    )

    # from sys import exit
    # exit()

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        sample_weight_mode="temporal",
        metrics=['accuracy']
    )

    history = model.fit_generator(
        aa_generator_batch(
            train_aa_dict, 
            train_cmap_dict, 
            batch_size),
        validation_data=aa_generator_batch(
            valid_aa_dict, 
            valid_cmap_dict, 
            batch_size),
        steps_per_epoch=train_steps,
        # steps_per_epoch=6,
        epochs=epochs,
        validation_steps=valid_steps,
        # validation_steps=1,
        callbacks=callbacks_list,
    )

    model.save(args.save_model) 

    # to load back the model:
    # model = tf.keras.models.load_model('my_model.h5')

    # note: make sure the versions are correct
    # if the model is saved from tensorflow 1.12.0,
    # it should not be loaded into a python program
    # using tensorflow 1.13.0

    # need to also call the custom layers when loading