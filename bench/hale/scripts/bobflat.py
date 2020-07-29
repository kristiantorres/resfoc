import inpout.seppy as seppy
import numpy as np
from oway.mute import mute
import matplotlib.pyplot as plt

sep = seppy.sep()

saxes,stack = sep.read_file("shots.H")
[nt,nrx,nsx] = saxes.n; [dt,drx,dsx] = saxes.d; [ot,orx,osx] = saxes.o
stack = np.ascontiguousarray(stack.reshape(saxes.n,order='F').T)

# Compute the source coordinates
srcx = np.linspace(osx,osx+(nsx-1)*dsx,nsx)
idx = 25

# Windowed source axis
nsxw = nsx - idx; osxw = srcx[idx]
srcxw = np.linspace(osxw,osxw+(nsxw-1)*dsx,nsxw)

# Windowed data
stackw = stack[idx:,:,:]
ntrw = nsxw*nrx

# Apply a mute
stackwmut = mute(stackw,dt=dt,dx=dsx,v0=1.4,t0=0.2,half=False)

# Get the number of receivers per shot
nrec = np.zeros(nsxw,dtype='float32') + 48

# Compute the receiver coordinates
offs = np.linspace(orx,orx+(nrx-1)*drx,nrx)
recx = np.zeros([nsxw,nrx],dtype='float32')
for isx in range(nsxw):
  recx[isx,:] = srcxw[isx] + offs[:]

# Flatten the receiver coords and the data
recx = recx.reshape([ntrw])
stackwmut = stackwmut.reshape([ntrw,nt])

# Write out
sep.write_file("hale_shotflatbob2.H",stackwmut.T,ds=[dt,1.0])
sep.write_file("hale_srcxflatbob2.H",srcxw)
sep.write_file("hale_recxflatbob2.H",recx)
sep.write_file("hale_nrecbob2.H",nrec)

