#!/bin/sh


# the integer corresponds to the cull folder
cullnum=1


# convert a culled pdb_ids.dat into just a text file 
# containing just the PDB IDs
# example output: pdb_ids.txt  
# Inside that text file is:
# PDBID1,PDBID2,PDBID3
# We need this text file for downloading PDB files
# from the Protein Data Bank Website
python ../preprocess/get_pdb_ids.py $cullnum


# Move unwanted pdb files to another folder
python ../preprocess/remove_unwanted_pdb_files.py $cullnum


# Write out a fasta file that contains only the
# amino acid sequences that we want.
python ../preprocess/get_wanted_fasta_seq.py $cullnum


# write out contact maps
# python ../preprocess/get_contact_maps.py $cullnum


# using the wanted fasta file, write out the amino acid
# sequence in 1 hot encoded form.
# python ../preprocess/fasta_to_1_hot_encodings.py $cullnum


# Create train, validate, and development test set.
python ../preprocess/create_model_sets.py $cullnum


# remove intermediary files
# I can actually delete all files in this folder
# except the final result folder.

# rm ../data/cull$cullnum/amino_acids_1_hot.npy
# rm ../data/cull$cullnum/contact_map_matrices.npy
# rm ../data/cull$cullnum/amino_acids.fasta
# rm -r ../data/cull$cullnum/removed_pdb_files