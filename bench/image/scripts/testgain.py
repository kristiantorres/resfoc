import numpy as np
import inpout.seppy as seppy
from resfoc.gain import agc
import matplotlib.pyplot as plt

sep = seppy.sep()

iaxes,img = sep.read_file("fltimgzoff.H")

img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)

plt.figure(1)
plt.imshow(img.T,cmap='gray')
plt.figure(2)
plt.imshow(agc(img).T,cmap='gray')
plt.show()

