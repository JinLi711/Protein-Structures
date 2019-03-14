Notes on the articles that I've read.


# Guide to Understanding PDB Data

[LINK](https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/introduction)

## Introduction to PDB Data

* typical PDB file will contain header info summarizing the protein.
* some structures may contain information only on the functional part
* portions of the molecule may be missing
* most crystallographic structure do not have information on hydrogen atoms

## Dealing with Coordinates

* for each atom, there is the name, number in the file, the name and number of the residue it belongs to, one letter to specify the chain (in oligomeric proteins), its x, y, and z coordinates, and an occupancy and temperature factor

## R-value and R-free

* measure of the quality of atomic model from crystallography
* typical values are 0.20
* random values are 0.63, and perfect is 0

## Small Molecule Ligands

* need to remove ligands






# THE BASICS OF PROTEIN STRUCTURE AND FUNCTION

[LINK](http://www.interactive-biology.com/6711/the-basics-of-protein-structure-and-function/)

* This site contains a brief description of the very basics of proteins. It talks about protein functions, structures, and stability.






# AlphaFold @ CASP13: “What just happened?”

[LINK](https://moalquraishi.wordpress.com/2018/12/09/alphafold-casp13-what-just-happened/)

2018

* NEED TO GET BACK TO THIS SINCE

* AlphaFold: co-evolutionary based method: extract evolutionary couplings from protein MSA. Predict whether two amino acids are in contact or not using MSAs. Feed it through an algorithm to predict 3D structure.
* AlphaFold used a softmax over discretized spatial ranges as outputs, predicting probability distribution over distances
* Insights from AlphaFold: don't just predict contact maps but also distances





# Biopython Tutorial

* can be used to parse through different file formats

## 11.2 Structure representation

* layout of Structure object is: SMCRA (Structure/Model/Chain/Residue/Atom)
    * A structure consists of models
    * A model consists of chains
    * A chain consists of residues
    * A residue consists of atoms
