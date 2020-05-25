import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection

#hf = h5py.File('/net/fantastic/scr2/joseph29/angfaultfocptch.h5')
hf = h5py.File('/net/fantastic/scr2/joseph29/angfaultdefptch.h5')

keys = list(hf.keys())

iptch = np.random.randint(105)
iex = np.random.randint(1000)
print(iex,iptch)
onex = hf[keys[iex]][iptch,:,:,:,0]

hf.close()

# onex [ang,z,x]

stk = np.sum(onex,axis=0)
slc = onex[:,32,:]
ang = onex[:,:,32]

fig = plt.figure()
ax = fig.gca(projection='3d')

nx = 64; dx = 0.01; ox = 0.0
na = 64; da = 2.22; oa = -70.0

xbeg = ox; xend = ox+(nx-1)*dx
abeg = oa; aend = oa+(na-1)*da

#X = np.linspace(0, 63, 64)
#Y = np.linspace(0, 63, 64)
X = np.linspace(xbeg, xend, nx)
#Y = np.linspace(xbeg, xend, nx)
Y = np.linspace(abeg, aend, na)
print(X,Y)
xg, yg = np.meshgrid(X, Y)
xg, zg = np.meshgrid(X, X)

vmin1 = np.min(stk); vmax1 = np.max(stk)
levels1 = np.linspace(vmin1,vmax1,200)

#vmin2 = np.min(ang); vmax2= np.max(ang)
vmin2 = np.min(onex); vmax2= np.max(onex)
levels2 = np.linspace(vmin2,vmax2,200)

cset = [[],[],[]]

cset[0] = ax.contourf(xg, yg, slc, zdir='z', offset=xbeg,
                      levels=levels2,cmap='gray')

# now, for the x-constant face, assign the contour to the x-plot-variable:
cset[1] = ax.contourf(ang, yg, np.flip(xg), zdir='x', offset=xend,
                      levels=levels2,cmap='gray')

# likewise, for the y-constant face, assign the contour to the y-plot-variable:
cset[2] = ax.contourf(xg, stk, np.flip(zg), zdir='y', offset=abeg,
                      levels=levels1,cmap='gray')

ax.set(xlim=[xbeg,xend],ylim=[abeg,aend],zlim=[xend,xbeg])

ax.set_xlabel('\nX (km)',fontsize=15)
ax.set_ylabel('\nAngle'r'$\degree$',fontsize=15)
ax.set_zlabel('\nZ (km)',fontsize=15)
ax.tick_params(labelsize=15)

#zi = [0,0.64]
#xi = [0.32,0.32]
#
#pts = [[(0.32,0.0),(0.32,0.64)]]
#pts2 = [[(0.0,0.32),(0.64,0.32)]]
#
#lines = LineCollection(pts,zorder=1000,color='k',lw=2)
#lines2 = LineCollection(pts2,zorder=1000,color='k',lw=2)
#ax.add_collection3d(lines,zdir='y',zs=abeg)
#ax.add_collection3d(lines2,zdir='y',zs=abeg)
#
#
#from mpl_toolkits.mplot3d.art3d import Line3DCollection
#
#class FixZorderCollection(Line3DCollection):
#    _zorder = 1000
#
#    @property
#    def zorder(self):
#        return self._zorder
#
#    @zorder.setter
#    def zorder(self, value):
#        pass
#
#ax.collections[-1].__class__ = FixZorderCollection

plt.show()
