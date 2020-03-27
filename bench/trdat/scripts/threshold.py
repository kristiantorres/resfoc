import inpout.seppy as seppy
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

sep = seppy.sep([])

laxes,lbl = sep.read_file(None,ifname='lbl.H')
lbl = lbl.reshape(laxes.n,order='F')

vaxes,vel = sep.read_file(None,ifname='me.H')
vel = vel.reshape(vaxes.n,order='F')

#idx = np.abs(lbl) > 50
#lbl[ idx] = 1
#lbl[~idx] = 0

plt.figure()
plt.imshow(lbl[:,:,300])
plt.show()

mask = np.ma.masked_where(lbl[:,:,300] == 0, lbl[:,:,300])
cmap = colors.ListedColormap(['red','white'])

plt.figure()
plt.imshow(vel[:,:,300],cmap='jet')
plt.imshow(mask,cmap=cmap)
plt.show()
