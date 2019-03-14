"""
Given a culled list of PDB files 
from Dunbrack's server, 
get only the PDB IDs, ignoring the 
length, Exptl., resolution, R-factor, and FreeRvalue.
"""

import sys

with open("../data/cull%i/pdb_ids.dat" % int (sys.argv[1])) as f:
    pdb_ids = [line.split()[0] for line in f]  
pdb_ids = ','.join(pdb_ids[1:])

out_file = open("../data/cull%i/pdb_ids.txt" % int (sys.argv[1]), 'w') 
out_file.write(pdb_ids)