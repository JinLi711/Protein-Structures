C - Complete
D - Doing
NW - Not working on yet
ND - Decided not to do this


# Step 1: Set Up

| Progress | Date Finished | Task                  
|----------|---------------|-----
|C         | 2/23/2019     | Create all the starting files and directories.
|C         | 3/17/2019     | Find programs to help with protein modeling.
|C         | 3/13/2019     | Gather PDB ids files.
|D         | NaN    | Add license.
|D         | NaN    | Find final test set (from CASP or something)




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


# Step 3: Preprocess

| Progress | Date Finished | Task                  
|----------|---------------|-----
|C         | 3/17/2019     | Parse through the culled files from Dunbrack's server to get only the PDB IDs seperated by commas.
|C         | 3/22/2019     | Move PDB files that we do not want to another folder.
|C         | 3/22/2019     | Extract only the FASTA sequence that I need.
|C         | 3/22/2019     | Align the PDB file sequence with FASTA sequence.
|C         | 3/22/2019     | Calculate contact map from PDB file and store it as a numpy array. (Need to deal with missing residues)
|C         | 3/23/2019     | Convert the amino acid sequence into one hot encodings.
|C         | 3/23/2019     | Compress data to reduce memory
|C         | 3/23/2019     | Create two files, one containing one hot encodings and another containing the contact maps. The two files have to align. Note that amino acids have varying lengths
|C         | 3/23/2019    | Split data into train, valid, and developement test set.
|C         | 3/23/2019     | Delete intermediary files that I created to reduce space use.



# Step 4: Model Building

| Progress | Date Finished | Task                  
|----------|---------------|-----
| C        | 3/23/2019     | Research how to create a model that takes in a variable length input.
| NW | NAN| Research how to deal with 2 dimensional outputs.
| NW | NAN| Create my own outer concatenation layer using Keras backend.
|NW         | NaN    | Sort train set by amino acid sequence length.
| NW | NAN| Build a baseline model



# Step 5: Model Evaluation

| Progress | Date Finished | Task                  
|----------|---------------|-----



# Step 6: Final Adjustments 

| Progress | Date Finished | Task  
|----------|---------------|-----
| NW | NAN | Improve code documentations.
| NW | NAN| build documentation with Sphinx
| NW | NAN | Create requirements.txt
| NW | NAN | Create directory tree.
| NW | NAN | Remove warnings.