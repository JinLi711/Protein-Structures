#!/bin/sh

# This file is for preprocessing the protein sequence data for training.

# The first things to do is gather data from the Dunbrack server.
# LINK: http://dunbrack.fccc.edu/Guoli/PISCES.php
# The server sends the data to your email.


# Notes:
# To stop the program from running:
#   `Ctrl + C` to 
# To time the script:
#   `time ./yourscript.sh` 


# The cull folder number (for different parameters)
cullnum=5
# whether to process the PDB files (after all the directories are set up) 
runpreprocess=true
# whether to reset the files created
reset=true

# log is stored in scripts folder
log_file="preprocess.log"

# These are stored in the cull folder
pdb_id_file="pdb_ids.dat"
aa_file="amino_acids.fasta"
param_file="cull_parameters.md"
only_pdb_ids_file="pdb_ids.txt"
# folder that contains the PDB files downloaded from RCSB
pdb_file_dir="pdb_files/"
# folder that contains the removed PDB files
rm_dir="removed_pdb_files/"
# Fasta file of the remaining PDB IDs
wanted_aa="wanted_aa.fasta"
cmap_matrices="cmap.npy"
aa_1_hot_matrix="aa.npy"
split_data_dir="model_data/"

data_dir="../data/cull$cullnum/"

if [ "$reset" = true ]; then
    rm -f $log_file

    cd $data_dir
    rm -f $only_pdb_ids_file
    rm -f $wanted_aa
    mv "$rm_dir"* "$pdb_file_dir"
    rm -r $split_data_dir
    rm -r $rm_dir
    cd ../../tools
fi

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
if [ ! -f $pdb_id_file ]; then
    touch $pdb_id_file
fi
if [ ! -f $aa_file ]; then
    touch $aa_file
fi
if [ ! -f $param_file ]; then
    touch $param_file
fi
if [ ! -s $pdb_id_file ] && [ ! -s $aa_file ] && [ ! -s $param_file ]; then
    echo "Download the cull metadata from Dunbrack server and place it in the files first." 
    echo "http://dunbrack.fccc.edu/Guoli/PISCES.php"
    exit 1
fi

cd ../../tools


#######################################################################
# STEP 2: Get PDB IDs
#######################################################################

# Extract only the PDB IDs
# example output:
#   7ODC,3SWH,2AXP,4EIU
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
if [ ! -d $pdb_file_dir ]; then
    mkdir $pdb_file_dir
fi
if [ -z "$(ls -A $pdb_file_dir)" ]; then
    echo "You need to download PDB file data from RCSB server. LINK: \nhttps://www.rcsb.org/#Subcategory-download_structures"
    exit 1
fi
cd ../../tools


# Steps below requires you to have downloaded the pdb files
if [ "$runpreprocess" = true ]; then

#######################################################################
# STEP 4: Preprocess
#######################################################################

# Move unwanted pdb files to another folder
# Conditions so far that warrant removal:
#   multiple chains
#   does not end in .pdb

cd $data_dir
if [ ! -d $rm_dir ]; then
    mkdir $rm_dir
fi
cd ../../tools

python3 ../preprocess/remove_unwanted_pdb_files.py $data_dir $rm_dir $pdb_file_dir --log $log_file

# Get a fasta file that contains only the wanted amino acid sequences.
python3 ../preprocess/get_wanted_fasta_seq.py $data_dir $aa_file $wanted_aa $pdb_file_dir --log $log_file

# Get the contact maps.
python3 ../preprocess/get_contact_maps.py $data_dir $pdb_file_dir $cmap_matrices --log $log_file

# Using the wanted fasta file, write out the amino acid sequence in 1 hot encoded form.
python3 ../preprocess/fasta_to_1_hot_encodings.py $data_dir $wanted_aa $aa_1_hot_matrix --log $log_file

# Create train, validate, and development test set.
python3 ../preprocess/create_model_sets.py $data_dir $cmap_matrices $aa_1_hot_matrix $split_data_dir --log $log_file

# remove intermediary files
rm "$data_dir""$aa_1_hot_matrix"
rm "$data_dir""$cmap_matrices"

fi