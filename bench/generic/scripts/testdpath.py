import numpy as np
import inpout.seppy as seppy

sep = seppy.sep()

sep.write_file("a.H",np.ones(10),dpath="/data/sep/joseph29/scratch/resmigflts/")
