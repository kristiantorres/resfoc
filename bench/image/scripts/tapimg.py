import numpy as np
import inpout.seppy as seppy
from scaas.gradtaper import build_taper
import matplotlib.pyplot as plt

sep = seppy.sep()

iaxes,img = sep.read_file("./extimg.H")
img = img.reshape(iaxes.n,order='F')

nz = img.shape[0]; nx = img.shape[1]; nh = img.shape[2]
tap1d,tap = build_taper(nx,nz,50,200)

timg = np.zeros(img.shape)

for iimg in range(nh):
  timg[:,:,iimg] = img[:,:,iimg]*tap


plt.imshow(tap*img[:,:,16],cmap='gray')
plt.show()

sep.write_file("extimgtap.H",timg,ofaxes=iaxes)
