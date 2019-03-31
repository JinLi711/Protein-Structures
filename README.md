# Protein Structure Prediction

The basic objective of this repository is to predict the contact map given a sequence of amino acids.

ALMOST THERE!



## Motivation

Proteins are one of the most important fundamental building blocks of life, making it very important to study their functions. However, the current methods of figuring out a protein's structure, like X-ray crystallography, are incredibly difficult and time consuming. 

The Nobel Prize winning experiment, the Anfinsen experiment, showed that the shape of the protein determines its function, and that the protein's primary structure (its amino acid sequence) determines its shape under correct conditions. However, predicting the protein's shape just from its amino acid sequence is not a simple task: Levinthal's paradox suggests that trying out all possible configurations of a short 100 residue sequence requires more time than the age of the universe!






## Method

Although there are many methods for determining the shape of a protein just from its amino acid, this repository will mainly focus on predicting the contact map from the starting amino acid sequence. A contact map for an amino acid sequence of length L is simply a matrix of size L x L, where each position i,j is either 1 if residue i, j are close together (they are in contact), or 0 if they are not. The contact map, along with the predicted secondary structure, is enough to predict the coordinate positions of each residue.

This repository contains a deep neural network composed of two residual networks. The amino acid sequence is feed into the first residual network, which is a series of residual 1 dimensional convolution layers. This network is then connected to a residual 2 dimensional network, and outputs the predicted contact map.





### Preprocess

1. Download the PDB IDs (IDs that represent a protein) that meet certain conditions. Our parameters were:
    * Sequence percentage identity: <= 25
    * Resolution                  : 0.0 ~ 2.5
    * R-factor                    : 0.3
    * Sequence length             : 40 ~ 700

2. Download the fasta file of the PDB IDs (a file containing the amino acid sequence).

3. Use the PDB IDs to download the PDB files (files containing the target coordinates of each atom of the protein). For simplicity and a more accurate model, we only consider the proteins with 1 chain.

4. Convert the amino acid sequence into one hot encoding (this is the input).

5. Compute the target contact map from the PDB files (this is the output).

6. Split the data into a train, validation, and developement test set. The final test set is on the CASP11 dataset.






### Neural Network Set Up

This is the deep neural network that I tried to build:

![2017 Deep Residual Network](https://github.com/JinLi711/Protein-Structures/blob/master/research/images/journal.pcbi.1005324.g001.PNG)

Note that `L` is the sequence length, which can be arbitrary, and `n` is the number of features learned from the convnet.


[Here](https://github.com/JinLi711/Protein-Structures/blob/master/tertiary_structure_prediction/visualization/test_visualization/chosen_plots/graph_run%3D.png) is a tensorboard visualization of the created model. Note that this has a few differences from the above model (see [Acknowledgements](https://github.com/JinLi711/Protein-Structures#acknowledgements)).






## Results

NOTE THAT THIS IS THE PRELIMARY MODEL RESULTS (not trained on all the data, and for a small number of steps)

![Predicted vs Actual Contact Map](tertiary_structure_prediction/visualization/test_visualization/plots/TR217cmap.png)

INSERT HERE RESULTS FROM CNS SOLVE


<img src="https://github.com/JinLi711/Protein-Structures/blob/master/tertiary_structure_prediction/visualization/test_visualization/chosen_plots/TR823cmap.png" alt="Alignment" width="300" height="300">

![Contact Map Prediction](https://github.com/JinLi711/Protein-Structures/blob/master/tertiary_structure_prediction/visualization/test_visualization/chosen_plots/TR823cmap.png)


<p class="aligncenter">

<img src="https://github.com/JinLi711/Protein-Structures/blob/master/tertiary_structure_prediction/visualization/test_visualization/chosen_plots/align.png" alt="Alignment" width="400" height="400">

</p>


## Future Steps

* Learn about the existing evaluation metrics for predicting protein structures and compare my model with existing models.

* Predict a distance matrix rather than a contact map. Then compare this model with my current model.

* Implement another deep neural network to predict secondary structure and solvent accessibility.

* Predict torsion angles using the amino acid sequence.

* Train multiple models with different parameters and compare.




## Extra Notes 

* All the python notebooks are scrap work for testing purposes. The cleaned code is in the scripts.






## Acknowledgements

The model is mostly based off this paper:

*Wang S, Sun S, Li Z, Zhang R, Xu J (2017) Accurate De Novo Prediction of Protein Contact Map by Ultra-Deep Learning Model. PLoS Comput Biol 13(1): e1005324. https://doi.org/10.1371/journal.pcbi.1005324*

However, they did not provide any code, so this repository tries to replicate the described model as closely as possible.

Some major differences between this model and the model from the paper:
* I did not include pairwise features, predicted secondary structure, predicted solvent accessibility as inputs (all these features individual require an entirely different project in itself)
* I used binary crossentropy instead of average log.
* The second residual network is only 14 layers rather than 60 because I came across out of memory issues on Google Colab.