import inpout.seppy as seppy
import numpy as np
from scaas.trismooth import smooth
from oway.mute import mute
from resfoc.semb import pick
import matplotlib.pyplot as plt
from genutils.movie import viewimgframeskey

sep = seppy.sep()

# Read in both semblance scans
aaxes,ascn = sep.read_file("ascn.H")
ascn = np.ascontiguousarray(ascn.reshape(aaxes.n,order='F').T).astype('float32')
[nt,nv,nma] = aaxes.n; [ot,ov,oma] = aaxes.o; [dt,dv,dma] = aaxes.d

baxes,bscn = sep.read_file("bscn.H")
bscn = np.ascontiguousarray(bscn.reshape(baxes.n,order='F').T).astype('float32')
[nt,nv,nmb] = baxes.n; [ot,ov,omb] = baxes.o; [dt,dv,dmb] = baxes.d

# Mute the semblance panels
ascnmut = mute(ascn,dt,dv,ot=ot,ox=ov,v0=0.67,x0=1.5,half=False)
bscnmut = mute(bscn,dt,dv,ot=ot,ox=ov,v0=0.67,x0=1.5,half=False)

#viewimgframeskey(bscnmut,cmap='jet',interp='bilinear')

tscn = np.zeros([nma+nmb,nv,nt],dtype='float32')

# Combine the semblance panels (even and odds)
tscn[0::2,:,:] = ascnmut[:]
tscn[1::2,:,:] = bscnmut[:]

#viewimgframeskey(tscn,cmap='jet',interp='bilinear')

velrms = pick(tscn,ov,dv,rectz=40,rectx=15,verb=False)

omo = 7.035; dmo = dma/2

sep.write_file("velrmscomb.H",velrms.T,os=[ot,omo],ds=[dt,dmo])


