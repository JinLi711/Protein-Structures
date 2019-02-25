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







# Improved prediction of protein secondary structure by use of sequence profiles and neural networks

[LINK](https://www.pnas.org/content/90/16/7558.short)

*Improved prediction of protein secondary structure by use of sequence profiles and neural networks. B Rost, C Sander. Proceedings of the National Academy of Sciences Aug 1993, 90 (16) 7558-7562; DOI: 10.1073/pnas.90.16.7558*

## Abstract 

* combines multiple sequence alignments, balanced training, and structure context training
* accuracy of 70%

## Point of Reference

* baseline of 62% accuracy

## Use of Multiple Sequence Alignments

* the idea is to leverage the fact that proteins with similar sequences also have similar 3 dimensional folds
* multiple sequence alignments rather than a single sequence are the inputs

## Balanced Training

* loops are predicted well, helices are predicted rather well, and strands are predicted poorly
* this is partly due to the inbalance of the training set, which can be solved if we have the training set have the same proportions (1/3 each)

## Training on Structure Context

* though a prediction may have high accuracy, it can be bad at predicting the lengths of sequences 
* can address this problem by feeding the three state prediction output of the first into a second network, which is trained to recognize the structural context of single residue states, without reference to sequence information

## Jury of Networks

* hard voting of 12 different networks






# PHD: Predicting one-dimensional protein structure by profile-based neural networks

[LINK](https://www.sciencedirect.com/science/article/pii/S0076687996660339)

Read only abstract


## Abstract

* generate multiple sequence alignment
* feed alignment into neural network system

