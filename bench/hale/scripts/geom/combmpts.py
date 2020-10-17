import inpout.seppy as seppy
import numpy as np

sep = seppy.sep()

# Read in both semblance scans
aaxes,amid = sep.read_file("aden.H")
amid = np.ascontiguousarray(amid.reshape(aaxes.n,order='F').T).astype('float32')
[nt,nh,nma] = aaxes.n; [ot,oh,oma] = aaxes.o; [dt,dh,dma] = aaxes.d

baxes,bmid = sep.read_file("bden.H")
bmid = np.ascontiguousarray(bmid.reshape(baxes.n,order='F').T).astype('float32')
[nt,nh,nmb] = baxes.n; [ot,oh,omb] = baxes.o; [dt,dh,dmb] = baxes.d

tmid = np.zeros([nma+nmb,nh,nt],dtype='float32')

# Combine the semblance panels (even and odds)
tmid[0::2,:,:] = amid[:]
tmid[1::2,:,:] = bmid[:]

oho = 0.264; dho = 0.134
omo = 7.035; dmo = dma/2

sep.write_file("allmidptsden.H",tmid.T,os=[ot,oho,omo],ds=[dt,dho,dmo])


