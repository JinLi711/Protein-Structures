#!/bin/sh

#TODO: time functions in log
#TODO: have a reset parameter
#TODO: calculate total time of preprocess.sh execution

# This file is for preprocessing the protein sequence data for training.

# The first things to do is gather data from the Dunbrack server.
# LINK: http://dunbrack.fccc.edu/Guoli/PISCES.php
# The server sends the data to your email.


# Notes:
# Ctrl + C to stop the program from running


# the integer corresponds to the cull folder
# we want multiple cull folders for different parameters
cullnum=4
# whether to run preprocess (after all the directories are set up) or not
runpreprocess=true

log_file="preprocess.log"
data_dir="../data/cull$cullnum/"

#######################################################################
# STEP 1: Set up the cull directory.
#######################################################################

# create the cull folder
cd ../data

if [ ! -d "cull$cullnum" ]; then
    mkdir "cull$cullnum"
fi

cd "cull$cullnum"

# create the files to hold the Dunbrack cull
pdb_id_file="pdb_ids.dat"

if [ ! -f $pdb_id_file ]; then
    touch $pdb_id_file
fi

aa_file="amino_acids.fasta"
if [ ! -f $aa_file ]; then
    touch $aa_file
fi

param_file="cull_parameters.md"
if [ ! -f $param_file ]; then
    touch $param_file
fi

if [ ! -s $pdb_id_file ] && [ ! -s $aa_file ] && [ ! -s $param_file ]; then
    echo "Download the cull metadata from Dunbrack server and place it in the files first. \nhttp://dunbrack.fccc.edu/Guoli/PISCES.php"
    exit 1
fi


cd ../../tools

rm preprocess.log
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
only_pdb_ids_file="pdb_ids.txt"
python3 ../preprocess/get_pdb_ids.py $data_dir $pdb_id_file $only_pdb_ids_file --log $log_file



#######################################################################
# STEP 3: Get PDB files
#######################################################################

# Now that we have the PDB IDs, we have to download the PDB IDs from 
# RCSB server. It's as simple as uploading the created text file from
# step 2.
# Check only the "PDB" box for "Coordinates"
# "Experimental Data" boxes should not be checked.
# The compression type should be "uncompressed"
# Click "Launch Download", and the server will download a file called "download_rcsb.jnlp"
# Use that to download the PDB files into pdb_file directory.

cd $data_dir

pdb_file_dir="pdb_files/"
if [ ! -d $pdb_file_dir ]; then
    mkdir $pdb_file_dir
fi

if [ -z "$(ls -A $pdb_file_dir)" ]; then
    echo "You need to download PDB file data from RCSB server. LINK: \nhttps://www.rcsb.org/#Subcategory-download_structures"
    exit 1
fi

cd ../../tools



# All steps below requires you to have downloaded the pdb files

if [ "$runpreprocess" = true ]; then


#######################################################################
# STEP 4: Preprocess
#######################################################################


# Move unwanted pdb files to another folder
# Conditions so far that warrant removal:
#   multiple chains
#   does not end in .pdb

cd $data_dir

rm_dir="removed_pdb_files/"

if [ ! -d $rm_dir ]; then
    mkdir $rm_dir
fi

cd ../../tools

python3 ../preprocess/remove_unwanted_pdb_files.py $data_dir $rm_dir $pdb_file_dir --log $log_file


# Write out a fasta file that contains only the
# amino acid sequences that we want.
wanted_aa="wanted_aa.fasta"
python3 ../preprocess/get_wanted_fasta_seq.py $data_dir $aa_file $wanted_aa $pdb_file_dir --log $log_file

cmap_matrices="contact_map_matrices.npy"
# write out contact maps
python3 ../preprocess/get_contact_maps.py $data_dir $pdb_file_dir $cmap_matrices --log $log_file


# using the wanted fasta file, write out the amino acid
# sequence in 1 hot encoded form.
aa_1_hot_matrix="amino_acids_1_hot.npy"
python3 ../preprocess/fasta_to_1_hot_encodings.py $data_dir $wanted_aa $aa_1_hot_matrix --log $log_file


# Create train, validate, and development test set.
# python3 ../preprocess/create_model_sets.py $cullnum


# remove intermediary files
# I can actually delete all files in this folder
# except the final result folder.

# rm "$data_diramino_acids_1_hot.npy"
# rm "$data_dircontact_map_matrices.npy"

# rm "$data_diramino_acids.fasta"
# rm "$data_dirremoved_pdb_files"

fi


#######################################################################
# STEP 5: Create Model
#######################################################################