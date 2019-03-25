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