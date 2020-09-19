import inpout.seppy as seppy
import numpy as np
from oway.utils import interp_vel
import matplotlib.pyplot as plt

sep = seppy.sep()
vaxes,vels = sep.read_file("sigsbee_trvels.H")
vels = np.ascontiguousarray(vels.reshape(vaxes.n,order='F'))
[nz,nxv,nm] = vaxes.n; [oz,oxv,om] = vaxes.o; [dz,dxv,dm] = vaxes.d
nyv = 1; dyv = 1.0; oyv = 0.0

# Imaging grid
nxi = 600; oxi = 1.043945; dxi = 0.0457199

velin = np.zeros([nz,nyv,nxv],dtype='float32')
velin[:,0,:] = vels[:,:,0]

vel = interp_vel(nz,
                 nyv,oyv,dyv,
                 nxi,oxi,dxi,
                 velin,dxv,dyv,oxv,oyv)

print(vel.shape)

plt.imshow(vel[:,0,:],cmap='jet',interpolation='bilinear')
plt.show()


