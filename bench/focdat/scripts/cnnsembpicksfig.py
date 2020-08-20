import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt

# Windowing parameters
fxl =  49; nxl = 400
fzl = 120; nzl = 300

# Read residually migrated angle gathers
#laxes,lng = sep.read_file("./dat/refocus/mltest/mltestdogang.H")
#lng = lng.reshape(laxes.n,order='F').T
#lngw = lng[:,fxl:fxl+nxl,:,fzl:fzl+nzl]

laxes,lng= sep.read_file("./dat/refocus/mltest/mltestdogang2mask.H")
lng = lng.reshape(laxes.n,order='F').T
lngw = lng[:,fxl:fxl+nxl,:,fzl:fzl+nzl]

# Read in all semblance
#saxes,smb = sep.read_file("../focdat/dat/refocus/mltest/mltestdogsmb.H")
#smb = smb.reshape(saxes.n,order='F')
#smb = np.ascontiguousarray(smb.T).astype('float32')
#smbw = smb[fxl:fxl+nxl,:,fzl:fzl+nzl]

saxes,smb = sep.read_file("../focdat/dat/refocus/mltest/mltestdogsmbmask2.H")
smb = smb.reshape(saxes.n,order='F')
smb = np.ascontiguousarray(smb.T).astype('float32')
smbw = smb[fxl:fxl+nxl,:,fzl:fzl+nzl]

# Read in all picks
#paxes,pck = sep.read_file('../focdat/dat/refocus/mltest/mltestdogrho.H')
#pck = pck.reshape(paxes.n,order='F')
#pck = np.ascontiguousarray(pck.T).astype('float32')
#pckw = pck[fxl:fxl+nxl,fzl:fzl+nzl]

paxes,pck = sep.read_file('../focdat/dat/refocus/mltest/mltestdogrhomask2.H')
pck = pck.reshape(paxes.n,order='F')
pck = np.ascontiguousarray(pck.T).astype('float32')
pckw = pck[fxl:fxl+nxl,fzl:fzl+nzl]

