Here is a log of my thoughts, questions, what I could have done better, and whatever else is on my mind.

# February 23, 2019

There seems to be many different output labels of protein structure. Some papers have done DSSP classification, others have predicted the angles and positions in space.

# February 24, 2019

Starting from the earliest literature and reading from ground up. Going to mainly try to predict tertiary structures (positions in space), but this will still require me understanding how to predict the primary and secondary structures.

Will eventually need to learn how one converts contact maps into tertiary structures. There are many algorithms, not sure which one would be best.

# March 9, 2019

Just some small notes: when downloading PDB file, we may only want a small chain from it. Is there a way for me to download only that small chain and not the entire molecule since the PDB file for that molecule may be huge?

Using Dunbrack to find PDB ideas, need to parse that file and remove all chain indicators. For example, as one of the results from the search, we got: 7ODCA. We want to only search for 7ODC, but keep in mind that we only want the A chain after the molecule has been downloaded.

# March 12, 2019

Use pylint or other something else to check pep8 standards. Need to write good software.

Performed cull1 from Dunbrack.

# March 13, 2019

Spent 2 hours trying to get DSSP to work. What a waste of time. Gave up, going to use RaptorX instead to predict secondary structure.

# March 17, 2019

Tried to look for scripts that fills in missing residues / atoms into PDB file. Couldn't find any that were well integrated into Python.

# March 22, 2019

Couldn't get RaptorX secondary structure prediction to work. It doesn't work and doesn't have any error messages. Great.

Decided that I'm not going to use any existing feature extraction methods (like finding secondary structure, solvent accessibility, etc). I need a proof of concept for now, and I actually plan on creating my own feature extractions (probably in the summer). I also can't find a model that works, has good documentation, and is backed by actualy publications.

# March 23, 2019

It may be interesting to embedd amino acids into vectors instead of one hot encoding it. But all the research papers I've read only used one hot encoding.

Need to download the full cull.

# March 24, 2019

Took forever to create the inner product layer. Note to self: don't mix the Keras inside Tensorflow with Keras outside.

# March 25, 2019

There was a mismatch in size between the input amino acid 1 hot encoding, and the output contact map for some of the proteins.

Some causes that I was able to figure out:
* when extracting the residues from the PDB IDs, BioPython does not count some codes as residues, even though those residues appear in the fasta sequence. For example, MSE is a modified residue, which is not counted as a residue. There may be others, and I will just need to find the most common ones. Then I will just remove the other proteins that have uncommon residues in the chain. 
Or I can parse MODRES in the PDB file.

There may be others that I have not found out yet, but sure.

# March 26, 2019

Problem:

```
/Users/jinli/anaconda3/lib/python3.6/site-packages/tensorflow/python/ops/gradients_impl.py:112: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
```

Most likely cause:

```
tf.nn.embedding_lookup # which is like tf.gather
```

See this [StackOverFlow](https://stackoverflow.com/questions/35892412/tensorflow-dense-gradient-explanation#) post.

This is just a warning, but when trying to run the model with 2 million parameters, the amount of memory used exploded. 

I think I need to replace tf.nn.embedding_lookup with tf.dynamic_partition, but not so sure how I'm going to do that. I think it would be find just using a smaller model for now.

Some possible suggestions to improve:
* predict distance rather contact. Then when generating the 3D protein, convert the distance to a contact matrix.


# March 27, 2019

Training on Google Colab has been super slow (like 4 hours for 1 epoch). Some possible fixes:
* increase the batch size. The current batch size is 1, because I didn't think it mattered. Turns out it matters ALOT.
* Fix the sparse IndexedSlices.

Things I need to check / change before running the final model:
* check the validation steps for fit_generator
* check that the data path is correct (i.e the cull number)

# March 28, 2019

Couldn't fix the exploding memory problem, so I created another OuterProduct layer. Not sure if this is going to be better.

Ok, and another part causing out of memory is just the sheer size of the neural net. For example, if we feed in a batch of amino acids of size 400, batch size of 15, we have a size of 15 * 400 * 400 * 60 = 144000000, which is huge. And that's just for one layer (there are 60 of these layers in just the second residual network). I have no clue how the paper was able to pull off such a huge model training. But for my case, I will have to tone down the size of the model.

This seemed to do the trick.

```
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
```


# March 29, 2019

Another problem encountered: training seems to result in the optimal prediction of no contact for every residue, which is bad. Possible causes:
* unbalanced outputs (alot more no contacts than contacts)
  * will solve this by increasing class weight for contact
* noisy data

Possible solutions:
* increase dataset size
* decrease fully connected layers
* increase number of epochs
* implement more regularizers
* make sure there's correct normalizations going on


Job for secondary structure prediction [here](http://raptorx.uchicago.edu/StructurePropertyPred/status/89121447/)

THINGS FINALLY WORKED!!! :smile: :smile: :smile: 

Bit annoying: Confold Server only takes in proteins of max length 500.

Just remember to upload the data for final training.