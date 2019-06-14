#!/bin/sh

# Visualize the results of the model.

cullnum=5
log_file="visualize.log"
viz_dir="visualization/model_visualization/"

model_weights="../../models/cull$cullnum/best_weight/Double_Resid_Network_weights.best.hdf5"
# aa_path="../../data/test/casp11.fasta"
# cmap_path="../../data/test/casp11.targets_refine/"
aa_path="../../data/cull$cullnum/model_data/devtest_aa.npy"
cmap_path="../../data/cull$cullnum/model_data/devtest_cmap.npy"

cd ../$viz_dir

python3 visualization.py $model_weights $aa_path $cmap_path --num_layers 14 --log $log_file