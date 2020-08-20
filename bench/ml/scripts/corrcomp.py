import inpout.seppy as seppy
import numpy as np
from deeplearn.focuslabels import corrsim
from resfoc.gain import agc
import matplotlib.pyplot as plt

sep = seppy.sep()

faxes,foc = sep.read_file("zofogresmigro1.H")
foc = agc(foc.reshape(faxes.n,order='F'),transp=True).T

saxes,smb = sep.read_file("rfismbcomp.H")
smb = agc(smb.reshape(saxes.n,order='F'),transp=True).T

caxes,cnn = sep.read_file("rficnnfoc.H")
cnn = agc(cnn.reshape(caxes.n,order='F'),transp=True).T

#TODO: make a patch wise metric. This would give an actual map
smbsim,smbimg = corrsim(smb,foc,corrimg=True)
cnnsim,cnnimg = corrsim(cnn,foc,corrimg=True)

plt.figure()
plt.imshow(smbimg,cmap='jet')
plt.colorbar()
plt.figure()
plt.imshow(cnnimg,cmap='jet')
plt.colorbar()
plt.show()

print("SMB=%f CNN=%f"%(smbsim,cnnsim))

fsize = 16
fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(foc,cmap='gray',interpolation='sinc')
ax.set_title('Well-focused',fontsize=fsize)

fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(smb,cmap='gray',interpolation='sinc')
ax.set_title('Semblance',fontsize=fsize)

fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(cnn,cmap='gray',interpolation='sinc')
ax.set_title('CNN',fontsize=fsize)

plt.show()
