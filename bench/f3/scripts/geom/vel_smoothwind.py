import inpout.seppy as seppy
from scaas.trismooth import smooth
import numpy as np

sep = seppy.sep()
vaxes,vel = sep.read_file("./vels/migvelslint.H")
vels = vel.reshape(vaxes.n,order='F').T.astype('float32')

nans = np.isnan(vels)
vels[nans] = 0.0

# Window the velocities
velsw = vels[5:505,200:1200,:]

velssm = smooth(velsw,rect1=60,rect2=60,rect3=30)

sep.write_file("./vels/migvelssm.H",velssm.T,os=vaxes.o,ds=vaxes.d)

