Here are the notes for research papers. I only took notes on what I thought was important for this project.


# Protein secondary structure prediction with a neural network

[LINK](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC286422/)


*Holley LH, Karplus M. Protein secondary structure prediction with a neural network. Proc Natl Acad Sci U S A. 1989;86(1):152-6.*

## Abstract

* The training was intended to teach networks to find relationships between secondary structures and amino acid sequences.
* sample set was 48 proteins
* test set: 14 proteins. 63% accuracy.
* tried to predict 3 structures: helix, sheet, coil.

## Introduction

* Used feedfoward neural networks organized into layers.
* Weights and biases are updated to minimize a loss function

## Methods

* 3 classes: helices (H), sheets (E), and coils
* one input layer, one hidden layer, one output layer
* input layer encodes a moving window of size 17, and is used to predict the central residue
* uses one hot encoding to size of 21 (20 different amino acids plus null input for when the moving window overlaps the terminal end of protein)
* input is essentially a 2-tensor
* output layer: (1,0): helix, (0, 1): sheet, (0,0): coil
* run the numbers through the network, outputs some decimal, then round based on some threshold.
* propogation and gradient descent is used

## Results

* Includes percentage correct under different parameters (like number of hidden layers, window sizes), correlation coefficients, comparison with other algorithms
* proposed physicohemical encoding: encode amino acid sequences based on their physicohemical properties of side chains