import numpy as np
from resfoc.tpow import tpow
import matplotlib.pyplot as plt

nz = 256; oz = 0.0; dz = 20
nx = 400; nh = 21; nro = 11
onear = np.ones([nz,nx,nh,nro],dtype='float32')

tpowed = tpow(onear,nz,oz,dz,nx,1,nh,nro)

ih = np.random.randint(0,21)
ir = np.random.randint(0,11)
print(ih,ir)
plt.imshow(tpowed[:,:,ih,ir],cmap='jet')
plt.show()

onear2 = np.ones([nz,nx,nro],dtype='float32')

tpowed2 = tpow(onear2,nz,oz,dz,nx,1,nh=None,nro=nro)
plt.imshow(tpowed2[:,:,ir],cmap='jet')
plt.show()

