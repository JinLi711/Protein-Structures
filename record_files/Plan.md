* C - Complete
* D - Doing
* NW - Not working on yet
* ND - Decided not to do this


# Step 1: Set Up

| Progress | Date Finished | Task                  
|----------|---------------|-----
|C         | 2/23/2019     | Create all the starting files and directories.
|C         | 3/17/2019     | Find programs to help with protein modeling.
|C         | 3/13/2019     | Gather PDB ids files for preliminary testing.
|C         | 3/27/2019     | Add license.
|C         | 3/27/2019     | Find final test set (from CASP or something)




# Step 1: Research 

| Progress | Date Finished | Task                  
|----------|---------------|-----
|C         | 3/12/2019     | Research on how to read PDB file and things I need to worry about for PDB files
|C         | 3/17/2019     | Read through early literature of secondary structure predictions
|C         | 3/17/2019     | Read literature on contact map predictions.

# Step 2: Visualization

| Progress | Date Finished | Task                  
|----------|---------------|-----
| NW | NAN| Find aggregate information on the PDB files / fasta sequences.
| C        | 3/28/2019     | Visualize contact maps.
| NW | NAN| Tensorboard visualization.


# Step 3: Preprocess

| Progress | Date Finished | Task                  
|----------|---------------|-----
|C         | 3/17/2019     | Parse through the culled files from Dunbrack's server to get only the PDB IDs seperated by commas.
|C         | 3/26/2019     | Remove all PDB IDs that are known to have more than one chain.
|C         | 3/26/2019     | Download PDB files from RSCB server.
|C         | 3/22/2019     | Move PDB files that we do not want to another folder.
|C         | 3/22/2019     | Extract only the FASTA sequence that I need.
|C         | 3/22/2019     | Align the PDB file sequence with FASTA sequence.
|C         | 3/22/2019     | Calculate contact map from PDB file and store it as a numpy array. (Need to deal with missing residues)
|C         | 3/23/2019     | Convert the amino acid sequence into one hot encodings.
|C         | 3/23/2019     | Compress data to reduce memory
|C         | 3/23/2019     | Create two files, one containing one hot encodings and another containing the contact maps. The two files have to align. Note that amino acids have varying lengths
|C         | 3/23/2019     | Split data into train, valid, and developement test set.
|C         | 3/23/2019     | Delete intermediary files that I created to reduce space use.
|C         | 3/25/2019     |  Check that the shapes of inputs and outputs match.
|C         | 3/25/2019     | Remove all inputs/outputs from dataset where the shape doesn't match. Note that this should be the very last step; I want to remove as few as possible from this step. Count how many were removed (and which were removed).
|C         | 3/26/2019     |  Edit bash script with more documentation and make sure everything flows nicely.
|C         | 3/26/2019     |  Make sure everything runs for a PDB file size of 200.
| NW | NAN| Create a preprocess output file that describes all relevant information.




# Step 4: Model Building

| Progress | Date Finished | Task                  
|----------|---------------|-----
| C        | 3/23/2019     | Research how to create a model that takes in a variable length input.
| C        | 3/24/2019     | Research how to have variable size inputs and outputs for the neural net.
| C        | 3/24/2019     | Research how to deal with 2 dimensional outputs.
| C        | 3/24/2019     | Create my own outer concatenation layer using Keras backend.
| C        | 3/24/2019     | Create 1D and 2D residual network block.
| C        | 3/24/2019     | Create an iterator to feed the neural net.
|ND        | NaN           | Sort train set by amino acid sequence length.
| C        | 3/25/2019     | Add in callbacks to the model.
| C        | 3/25/2019     | Create plotting functions to plot the change in loss function.
| C        | 3/26/2019     | Build the model as best as possible.
| C        | 3/27/2019     |  Set up the training on Google Colab
| C        | 3/27/2019     |  Create another iterator that produces batches instead of a single file.
| D         | NAN     |  Make sure the model makes sense for a PDB file size of 2000.



# Step 5: Model Evaluation

| Progress | Date Finished | Task                  
|----------|---------------|-----
| C        | 3/27/2019     | Make sure fasta_to_1_hot_encoding.py is able to take in a never before seem sequence.
| C        | 3/27/2019     | Set up a short pipeline so I can predict a new fasta sequence.
| NW | NAN | Learn how to use CNS Solve.
| NW | NAN | Since we do not have a model to predict secondary structures, we have to run our test amino acid sequences through another server to get more required information to create the tertiary structure in CNS Solve.
| NW | NAN | Use the contact map to create PDB files of predictions.
| NW | NAN | Use VMD to superimpose the actual protein structure and the predicted protein structure.



# Step 6: Final Adjustments 

| Progress | Date Finished | Task  
|----------|---------------|-----
| C        | 3/27/2019     | Improve code documentations on preprocess files.
| NW | NAN| Update the README.md file.
| NW | NAN| Move all scrape work files to a scrape work folder.
| NW | NAN| build documentation with Sphinx
| NW | NAN | Create requirements.txt
| NW | NAN | Create directory tree.
| NW | NAN | Remove warnings.