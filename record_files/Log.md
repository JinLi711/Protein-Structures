Here is a log of my thoughts, questions, what I could have done better, and whatever else is on my mind. This is mainly for me to jot down ideas, but it can also be used to get insight into my thinking.

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