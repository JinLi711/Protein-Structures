#!/bin/sh


# convert a culled pdb_ids.dat into just a text file 
# containing just the PDB IDs
# example output: pdb_ids.txt  
# Inside that text file is:
# PDBID1,PDBID2,PDBID3
# We need this text file for downloading PDB files
# from the Protein Data Bank Website

# takes an integer as an argument
# the integer corresponds to the cull folder
cullnum=1
python ../preprocess/get_pdb_ids.py $cullnum

# Move unwanted pdb files to another folder
python ../preprocess/remove_unwanted_pdb_files.py $cullnum

# Write out a fasta file that contains only the
# amino acid sequences that we want.
python ../preprocess/get_wanted_fasta_seq.py $cullnum

python ../preprocess/get_contact_maps.py $cullnum