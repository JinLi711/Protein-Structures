#!/bin/sh

# Notes to self:
# Ctrl + C to stop the program from running


# the integer corresponds to the cull folder
cullnum=2
# whether to run preprocess or not
runpreprocess=true

#######################################################################
# STEP 1: Create folder
#######################################################################

# create folder called cull$cullnum
cd ../data

if [ ! -d "cull$cullnum" ]; then
    mkdir "cull$cullnum"
fi

cd "cull$cullnum"

if [ ! -f "pdb_ids.dat" ]; then
    touch "pdb_ids.dat"
fi

if [ ! -f "amino_acids.fasta" ]; then
    touch "amino_acids.fasta"
fi

cd ../../scripts

# In this directory, you only need to start with 2 files: 
#   amino_acids.fasta
#   pdb_ids.dat
# These files are downloaded from the Dunbrack server.
# The reason why there are multiple cull folders
# is that the culled files may have different parameters,
# affecting the type of data used.
# The parameters are described in:
#   cull_parameters.md


#######################################################################
# STEP 2: Get PDB IDs
#######################################################################

# convert the culled pdb_ids.dat into just a text file 
# containing just the PDB IDs
# example output: pdb_ids.txt  
# Inside that text file is:
# 7ODC,3SWH,2AXP,4EIU
# We need this text file for downloading PDB files
# from the Protein Data Bank Website
python ../preprocess/get_pdb_ids.py $cullnum


#######################################################################
# STEP 3: Get PDB files
#######################################################################

# Now that we have the PDB IDs, we have to download the PDB IDs from 
# RCSB server. It's as simple as uploading the created text file from
# step 2.

cd ../data/cull$cullnum

if [ ! -d "pdb_files" ]; then
    mkdir "pdb_files"
fi

cd ../../scripts


# All steps below requires you to have downloaded the pdb files

if [ "$runpreprocess" = true ]; then


#######################################################################
# STEP 4: Preprocess
#######################################################################


# Move unwanted pdb files to another folder
# Conditions so far that warrant removal:
#   multiple chains
#   does not end in .pdb

cd ../data/cull$cullnum

if [ ! -d "removed_pdb_files" ]; then
    mkdir "removed_pdb_files"
fi

cd ../../scripts

# python ../preprocess/remove_unwanted_pdb_files.py $cullnum


# Write out a fasta file that contains only the
# amino acid sequences that we want.
# python ../preprocess/get_wanted_fasta_seq.py $cullnum


# write out contact maps
# python ../preprocess/get_contact_maps.py $cullnum


# using the wanted fasta file, write out the amino acid
# sequence in 1 hot encoded form.
# python ../preprocess/fasta_to_1_hot_encodings.py $cullnum


# Create train, validate, and development test set.
# python ../preprocess/create_model_sets.py $cullnum


# remove intermediary files
# I can actually delete all files in this folder
# except the final result folder.

# rm ../data/cull$cullnum/amino_acids_1_hot.npy
# rm ../data/cull$cullnum/contact_map_matrices.npy

# rm ../data/cull$cullnum/amino_acids.fasta
# rm -r ../data/cull$cullnum/removed_pdb_files

# create the model

fi


#######################################################################
# STEP 5: Create Model
#######################################################################