Here are the notes for research papers. I only took notes on what I thought was important for this project.






# Protein secondary structure prediction with a neural network

[LINK](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC286422/)

1989

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







# Neural network analysis of protein tertiary structure

[LINK](https://www.sciencedirect.com/science/article/pii/089855299090052A)

1990

*Neural network analysis of protein tertiary structure. George L. Wilcox. Marius Poliac. Tetrahedron Computer Methodology Volume 3, Issues 3–4, 1990, Pages 191-204, IN4, 205-211*

## Abstract 

* describes large scale back-propagation neural network for secondary and tertiary structure prediction
* uses 15 proteins as training

## Introduction

* the predicted output will be a distance matrix

## Methods, Data, and Analysis

* tried several networks, including no layers, 1 layer, 2 layers, layers with direct connections between input and output
* converted the amino acids into "alphabets" corresponding to their hydrophobicity, as it is an important physico-chemical interaction that drives protein folding.
    * ex. tyrosine was assigned -3.4, lycine was assigned 3.3, etc.
    * these numbers were then normalized from -1 to 1
    * the problem is that many of the residues have similar hydrophobicity
* the input was 1* 140 (each protein was less than 140 residues long)
* target was a distance matrix of 140 * 140.
    * calculated from alpha-carbon coordinates of PDB file
    * distances were normalized to 1 by dividing by the maximum distance
    * note that the matrix is symmetrical since d(i,j) = d(j,i). This does create some bias.

## Results

* weights of the neural network was initialized randomly
* some networks did not converge
* though RMS errors were low after training, generalization to new protein structures were very poor
* decreasing the learning rate seemed to have the greatest effect








# Improved prediction of protein secondary structure by use of sequence profiles and neural networks

[LINK](https://www.pnas.org/content/90/16/7558.short)

1993

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







# Combining evolutionary information and neural networks to predict protein secondary structure

[LINK](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.340190108)

1994

*Rost, Burkhard, and Chris Sander. “Combining Evolutionary Information and Neural Networks to Predict Protein Secondary Structure.” Proteins: Structure, Function, and Genetics, vol. 19, no. 1, 1994, pp. 55–72., doi:10.1002/prot.340190108.*

Read only abstract

## Abstract

* uses position-specific conservation weight as part of the input
* include number of insertions and deletions
* include global amino acid content
* 71.6% accuracy







# PHD: Predicting one-dimensional protein structure by profile-based neural networks

[LINK](https://www.sciencedirect.com/science/article/pii/S0076687996660339)

1996

*Rost, Burkhard. “[31] PHD: Predicting One-Dimensional Protein Structure by Profile-Based Neural Networks.” Methods in Enzymology Computer Methods for Macromolecular Sequence Analysis, 1996, pp. 525–539., doi:10.1016/s0076-6879(96)66033-9.*

Read only abstract

## Abstract

* generate multiple sequence alignment
* feed alignment into neural network system






# Recovery of protein structure from contact maps

[LINK](https://www.sciencedirect.com/science/article/pii/S1359027897000412)

1997







# Mining Protein Contact Maps

[LINK](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.19.488&rep=rep1&type=pdf)

2002







# Machine learning methods for protein structure prediction

[LINK](https://www.ncbi.nlm.nih.gov/pubmed/22274898)

2008

*Cheng J, Tegge AN, Baldi P. Machine Learning Methods for Protein Structure Prediction. IEEE Reviews in Biomedical Engineering. 2008;1:41–49. pmid:22274898*

## Abstract

* reviews hidden Markov models, neural networks, support vector machines, Bayesian methods, and clustering methods in 1-D, 2-D, 3-D, and 4-D protein structure predictions

## Introduction

* tertiary structure is described by x,y,z coordinates of atoms
* protein function is determined by structure
* 40,000 out of 2.5 million known sequences available have solved structures (determined experimentally)
* 2d prediction focuses on predicting spatial relationships between residues (like distance, contact map predictions, disulfide bond predictions)
* 2d predictions are independent of rotations and translations of protein
* to predict 3d structure, we can use information from 1d or 2d structures.

* Contact Map:
<img src="https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/4664312/4689462/4664428/4664428-fig-3-source-large.gif" height="500" width="500">

## Machine Learning Methods for 1-D Structure Prediction

* input: protein primary sequence
* ideas include probabilistic models, ensembles of neural networks, SVMs
* prediction limit of 88%

## Machine Learning Methods for 2-D Structure Prediction

* predict contact maps, which is just a matrix. Each element in this matrix M[i,j] is either 1 or 0, which will depend on whether the Euclidean distance between 2 amino acids at position i,j is above a specified distance threshold. We can measure those distances using the backbone.
* coarser contact map: use only the secondary structure elements
* finer contact map: use every atom
* we use contact maps because they are invariant to translations and rotations
* we can also use contact maps to infer protein folding rates
* some machine learning methods include neural networks, self-organizing maps, SVMs
* use two windows to target 2 amino acids, determine if they are in contact or not (makes this a binary classification). Each position in the window is a vector of 20 numbers (corresponding to 20 profile probabilities)
* can include other 1D info, including predicted secondary structure 
* 2D recursive neural network: to address the problem that amino acids outside the window are not being considered.
* can also try to predict disulfide bonds, very important for structure
* can also predict beta strand pairing
* can use Monte Carlo methods to reconstruct 3D structures from contact maps. Though this is usually unreliable.

## Machine Learning Methods for 3-D Structure Prediction

* WILL COME BACK TO THIS

## Machine Learning Methods for 4-D Structure Prediction

* WILL COME BACK TO THIS






# Protein Secondary Structure Prediction Using Cascaded Convolutional and Recurrent Neural Networks

[LINK](https://arxiv.org/pdf/1604.07176.pdf#page=8&zoom=100,0,445)

2016


# Protein contact maps: A binary depiction of protein 3D structures

[LINK](https://www.sciencedirect.com/science/article/pii/S0378437116305507)

2017




# Accurate De Novo Prediction of Protein Contact Map by Ultra-Deep Learning Model

[LINK](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005324)

2017

*Wang S, Sun S, Li Z, Zhang R, Xu J (2017) Accurate De Novo Prediction of Protein Contact Map by Ultra-Deep Learning Model. PLoS Comput Biol 13(1): e1005324. https://doi.org/10.1371/journal.pcbi.1005324*

## Abstract

* uses evolutionary coupling(ec) and sequence conservation information with deep residual nn

## Introduction

* de novo protein structure prediction
* direct evolutionary coupling analysis (DCA)
* evolutionary coupling analysis (ECA): predicts contacts by identifying co-evolved residues in a protein
    * but needs a lot of matches to be effective
* predicting contact map: sort of like pixel level labeling 
    * though some problems include:
        * not a ton of research in ML community on pixel level labeling 
        * contact maps cannot be resized like in actual images
        * number of positive and negative labels are unbalanced
* model was trained on solved protein structures, and tested on CASP and CAMEO

* ![Deep Learning Contact Model](/images/journal.pcbi.1005324.g001.PNG)




# Prediction of 8-state protein secondary structures by a novel deep learning architecture

2018

[LINK](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2280-5)