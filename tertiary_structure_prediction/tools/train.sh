#!/bin/sh

# This file is for training the model.

cullnum=5
log_file="train.log"
cmap_matrices="cmap.npy"
aa_1_hot_matrix="aa.npy"
split_data_dir="model_data/"
saved_model="my_model.h5"

cd ../models/cull$cullnum/

if [ ! -d "best_weight/" ]; then
    mkdir "best_weight/"
fi

python3 ../model_functions/primary_model.py $cullnum $cmap_matrices $aa_1_hot_matrix $saved_model --num_layers 10 --epochs 2 --log $log_file