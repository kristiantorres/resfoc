import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()

iaxes,img = sep.read_file("sigoverw5msk.H")
img = img.reshape(iaxes.n,order='F').T

iaxes,img2 = sep.read_file("sigoverwmsk.H")
img2 = img2.reshape(iaxes.n,order='F').T

taxes,tru = sep.read_file("sigextmasked.H")
tru = tru.reshape(taxes.n,order='F').T

sc = 0.2
vmin = sc*np.min(img); vmax = sc*np.max(img);

print(img.shape)
plt.figure()
plt.imshow(img[20,0,100:250,500:1000].T,cmap='gray',interpolation='bilinear',aspect='auto',vmin=vmin,vmax=vmax)
plt.figure()
plt.imshow(img2[20,0,100:250,500:1000].T,cmap='gray',interpolation='bilinear',aspect='auto',vmin=vmin,vmax=vmax)
plt.figure()
plt.imshow(tru[20,0,100:250,500:1000].T,cmap='gray',interpolation='bilinear',aspect='auto',vmin=vmin,vmax=vmax)
plt.show()

