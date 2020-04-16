import inpout.seppy as seppy
import h5py
import numpy as np
import deeplearn.utils as dlut
import matplotlib.pyplot as plt

sep = seppy.sep()

faxes,foc = sep.read_file("../focdat/fltimg-00760.H")
foc = foc.reshape(faxes.n,order='F')
zfoc = foc[:,:,16]

raxes,res = sep.read_file("../focdat/resfltimg-00760.H")
[nz,nx,nh,nro] = raxes.n
res = res.reshape(raxes.n,order='F')
zres = res[:,:,16]

paxes,ptb = sep.read_file("../focdat/resfltptb-00760.H")
ptb = ptb.reshape(paxes.n,order='F')

caxes,cnv = sep.read_file("velfltimg0760j.H")
cnv = cnv.reshape(caxes.n,order='F').T
cnvi = (dlut.resample(cnv,[nx,nz],kind='linear'))

laxes,lbl = sep.read_file("velfltlbl0760j.H")
lbl = lbl.reshape(laxes.n,order='F').T
flti = dlut.thresh((dlut.resample(lbl,[nx,nz],kind='linear')),0)

# Plot label on image
dlut.plotseglabel(cnvi.T,flti.T,color='red',fname='./fig/lbl760',xlabel='X (km)',
    ylabel='Z (km)',xmax=(nx)*0.01,zmax=(nz)*0.01,show=False,vmin=-3,vmax=3)

# Open H5 files
hfr = h5py.File('res760.h5','w')
hfi = h5py.File('foc760.h5','w')
hfp = h5py.File('ptb760.h5','w')
hfl = h5py.File('lbl760.h5','w')
hfc = h5py.File('cnv760.h5','w')

hfr.create_dataset("res0", (nro,nx,nz), data=zres.T, dtype=np.float32)
hfi.create_dataset("img0", (nx,nz), data=zfoc.T, dtype=np.float32)
hfp.create_dataset("ptb0", (nx,nz), data=ptb.T, dtype=np.float32)
hfl.create_dataset("flt0", (nx,nz), data=flti, dtype=np.float32)
hfc.create_dataset("cnv0", (nx,nz), data=cnvi, dtype=np.float32)

hfr.close(); hfi.close(); hfp.close(); hfl.close(); hfc.close()

