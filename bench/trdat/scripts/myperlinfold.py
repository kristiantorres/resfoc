import numpy as np
from scaas.velocity import noise_generator
import scipy.ndimage as sciim
import matplotlib.pyplot as plt

nz = 256; nx = 256
img = np.zeros([nx,nz])

idx = np.random.randint(0,256,30)

img[:,idx] = 1

plt.figure()
plt.imshow(img.T,cmap='gray',vmin=-1,vmax=1)

# Create a perlin function
amp = 8
shifts = noise_generator.perlin(x=np.linspace(0,2,256), octaves=3, period=80, Ngrad=80, persist=0.6, ncpu=1)
shifts -= np.mean(shifts); shifts *= 10*amp
plt.figure()
plt.plot(shifts)

oimg = np.array([sciim.interpolation.shift(img[ix],shifts[ix],order=1) for ix in range(nx)])

## Loop over each x point and apply the perlin shift
#oimg = np.zeros(img.shape)
#for ix in range(nx):
#  oimg[ix] = np.roll(img[ix],int(shifts[ix]))

  
plt.figure()
plt.imshow(oimg.T,cmap='gray',vmin=-1,vmax=1)
plt.show()
