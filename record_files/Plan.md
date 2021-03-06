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
|C         | 6/12/2019     | Find final test set (from CASP or something)




# Step 1: Research 

| Progress | Date Finished | Task                  
|----------|---------------|-----
|C         | 3/12/2019     | Research on how to read PDB file and things I need to worry about for PDB files
|C         | 3/17/2019     | Read through early literature of secondary structure predictions
|C         | 3/17/2019     | Read literature on contact map predictions.

# Step 2: Visualization

| Progress | Date Finished | Task                  
|----------|---------------|-----
| C        | 3/28/2019     | Visualize contact maps.
| C        | 3/30/2019     | Tensorboard visualization. Just create an image of the entire model.
| ND | NAN| Find aggregate information on the PDB files / fasta sequences.


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
|C         | 3/25/2019     | Check that the shapes of inputs and outputs match.
|C         | 3/25/2019     | Remove all inputs/outputs from dataset where the shape doesn't match. Note that this should be the very last step; I want to remove as few as possible from this step. Count how many were removed (and which were removed).
|C         | 6/12/2019     | Create a preprocess output file that describes all relevant information.
|C         | 6/12/2019     | Add argument parsing.
|C         | 6/12/2019     | Add logging
| ND | NAN | Enable multiprocessing for get_contact_maps.py.
|C         | 6/12/2019     |  Edit bash script with more documentation and make sure everything flows nicely.
|C         | 6/12/2019     |  Make sure everything runs for a PDB file size of 200.




# Step 4: Model Building

| Progress | Date Finished | Task                  
|----------|---------------|-----
| C        | 3/23/2019     | Research how to create a model that takes in a variable length input.
| C        | 3/24/2019     | Research how to have variable size inputs and outputs for the neural net.
| C        | 3/24/2019     | Research how to deal with 2 dimensional outputs.
| C        | 3/24/2019     | Create my own outer concatenation layer using Keras backend.
| C        | 3/24/2019     | Create 1D and 2D residual network block.
| C        | 3/24/2019     | Create an iterator to feed the neural net.
| C        | 3/28/2019     | Sort train set by amino acid sequence length.
| C        | 3/25/2019     | Add in callbacks to the model.
| C        | 3/25/2019     | Create plotting functions to plot the change in loss function.
| C        | 3/26/2019     | Build the model as best as possible.
| C        | 3/27/2019     |  Set up the training on Google Colab
| C        | 3/27/2019     |  Create another iterator that produces batches instead of a single file.
| C        | 3/28/2019     | Create another Outer Product layer.
| C        | 3/28/2019     | Change the sparse cross entropy to binary cross entropy and change the generator accordingly.
| C        | 3/29/2019     | Implement weighing for a 3 tensor output (implement in generator).
| C        | 3/30/2019     |  Make sure the model makes sense for a PDB file size of 2000.
| C        | 6/12/2019     | Create a train script.
| C        | 6/12/2019     | Add arg parsing to model train script.
| C        | 6/12/2019     | Add logging.



# Step 5: Model Evaluation

| Progress | Date Finished | Task                  
|----------|---------------|-----
| C        | 3/27/2019     | Make sure fasta_to_1_hot_encoding.py is able to take in a never before seem sequence.
| C        | 3/27/2019     | Set up a short pipeline so I can predict a new fasta sequence.
| ND | NAN | Learn how to use CNS Solve.
| C        | 6/13/2019     | Since we do not have a model to predict secondary structures, we have to run our test amino acid sequences through another server to get more required information to create the tertiary structure in CNS Solve.
| C        | 3/29/2019     | Create a file that contains the contact map in PFRMAT RR format, the predicted secondary structure, and the amino acid sequence.
| C        | 3/30/2019     | Use the contact map (along with secondary structure and amino acid sequence) to create PDB files of predictions using CONFOLD.
| C        | 3/30/2019     | Use RSCB alignment tool to superimpose the actual protein structure and the predicted protein structure.
| C        | 6/13/2019     | Calculate RMSD of prediction and actual for test set.
| C        | 6/13/2019     | Write an visualization evaluation script.


# Step 6: Final Adjustments 

| Progress | Date Finished | Task  
|----------|---------------|-----
| C        | 6/14/2019     | Improve code documentations.
| C        | 6/13/2019     | Go through all my scrape work to see what I forgot to describe.
| ND | NAN| Move all scrape work files to a scrape work folder.
| ND | NAN| build documentation with Sphinx
| ND | NAN | Create directory tree.
| ND | NAN | Remove warnings.
| ND | NAN | Check pep8 standards.
| ND | NAN | Remove unneccesary commits.
| C        | 6/14/2019     | Make everything more presentable to people without the background.
| NW | NAN | Make a quick use example.
| NW | NAN | Create a Pypi package.
| NW | NAN | Create requirements.txt
| C        | 6/14/2019     | Update the README.md file.