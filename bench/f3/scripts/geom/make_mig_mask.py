"""
Makes a mask from the migration cube

@author: Joseph Jennings
@version: 2020.09.28
"""
import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in the migration cube
maxes,mig = sep.read_file("./mig/mig.H")
mig = mig.reshape(maxes.n,order='F')
[ot,ox,oy] = maxes.o; [dt,dx,dy] = maxes.d

migabs = np.abs(mig)
migene = np.sum(migabs,axis=0)

msk = migene != 0

msk = msk.astype('float32')

sep.write_file("./vels/migmsk.H",msk,os=[ox,oy],ds=[dx,dy],dpath='/net/brick5/data3/northsea_dutch_f3/vels/')

