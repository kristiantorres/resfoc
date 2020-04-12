import inpout.seppy as seppy
import numpy as np
import deeplearn.utils as util
import matplotlib.pyplot as plt

sep = seppy.sep([])
iaxes,img = sep.read_file(None,ifname='img0019.H')
img = img.reshape(iaxes.n,order='F')

imgt = img.T

oimg = imgt[0,9,8,:,:]
fimg = imgt[0,:,:,:]

# Resamp to power of 2
loimg = util.resizepow2(oimg)
flimg = util.resizepow2(fimg,kind='cubic')
print(flimg.shape)

# Go back
smimg = util.resample(flimg,[400,256],kind='cubic')

plt.figure()
plt.imshow(oimg.T,cmap='gray')

plt.figure()
plt.imshow(loimg.T,cmap='gray')

plt.figure()
plt.imshow(flimg[9,8,:,:].T,cmap='gray')

plt.figure()
plt.imshow(fimg[9,8,:,:].T,cmap='gray')

plt.figure()
plt.imshow(smimg[9,8,:,:].T - fimg[9,8,:,:].T,cmap='gray',vmin=np.min(oimg),vmax=np.max(oimg))

plt.show()

