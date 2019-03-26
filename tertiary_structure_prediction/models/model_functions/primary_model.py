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
    the extra six dimensions added to input
    pairwise features
    different layer sizes in residual network
    used sparse categorical crossentropy instead of log
    no weighing of outputs
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

    def __init__(self):
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
# Generator
#----------------------------------------------------------------------


def aa_generator(x, y):
    """
    Generator for feeding a single instance of an 
    input and an output.
    The generator is reset when all elements are used.

    :param: input
    :type:  dict
    :param: label
    :type:  dict
    :returns: a single instance of input and label
    :rtype:   (numpy array, numpy array)
    """

    keys = set(x.keys())

    while True:
        try:
            key = random.sample(keys, 1)[0]
            keys.remove(key)

            one_hot_aa = x[key]
            one_hot_aa = np.reshape(
                one_hot_aa, (1,) + one_hot_aa.shape
            )
            cmap = y[key]
            cmap = np.reshape(cmap, (1,) + cmap.shape + (1,))
            yield one_hot_aa, cmap

        except ValueError:
            # if out of keys, reinsert back the keys
            keys = set(x.keys())


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

    x = residual_conv_block(
        x,
        "2d_convnet",
        resid_layer2_window_size,
        num_layers=resid_layer2_num_layers,
        regularizer=tf.keras.regularizers.l2(0.001)
    )

    x = tf.keras.layers.Conv2D(
        2,
        1,
        activation='relu',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)

#     x = tf.keras.layers.Dropout(
#         0.5,
#         name="Drop-Out"
#     )(x)

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

# tensorboard = tf.keras.callback.TensorBoard(
#     log_dir='Logs',
# )

reduceLROnPlat = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=1,
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
    patience=5
)

callbacks_list = [checkpoint, early, reduceLROnPlat]

if __name__ == "__main__":

    import sys

    data_path = "../../data/cull%i/model_data/" % int (sys.argv[1])

    train_aa_dict = np.load(data_path + 'train_aa_dict.npy')[()]
    train_cmap_dict = np.load(data_path + 'train_cmap_dict.npy')[()]
    valid_aa_dict = np.load(data_path + 'valid_aa_dict.npy')[()]
    valid_cmap_dict = np.load(data_path + 'valid_cmap_dict.npy')[()]
    devtest_aa_dict = np.load(data_path + 'devtest_aa_dict.npy')[()]
    devtest_cmap_dict = np.load(data_path + 'devtest_cmap_dict.npy')[()]


    model = create_architecture(3, 60)
    print (model.summary())

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        sample_weight_mode="temporal",
        metrics=['accuracy']
    )

    history = model.fit_generator(
        aa_generator(train_aa_dict, train_cmap_dict),
        validation_data=aa_generator(valid_aa_dict, valid_cmap_dict),
        steps_per_epoch=len(train_aa_dict), 
        epochs=20,
        validation_steps=10,
        callbacks=callbacks_list
    )

    model.save('my_model.h5') 

    # to load back the model:
    # model = tf.keras.models.load_model('my_model.h5')