import inpout.seppy as seppy
from inpout.seppy import bytes2float
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()
# Read in the source coordinates
#xaxes,xcrd = sep.read_file("allsrcxsep.H")
#yaxes,ycrd = sep.read_file("allsrcysep.H")
xaxes,xcrd = sep.read_file("sepsxwind.H")
yaxes,ycrd = sep.read_file("sepsywind.H")
npt = len(xcrd)
coords = np.zeros([npt,2])
coords[:,0] = ycrd
coords[:,1] = xcrd

# 6072299, new y origin (gives positive for receivers and sources)
ucoords = np.unique(coords,axis=0)

srcy = ucoords[:,0]
srcx = ucoords[:,1]

# Read in migration cube
maxes,mig = sep.read_file("./mig/mig.T")
nz,nx,ny = maxes.n; oz,ox,oy = maxes.o; dz,dx,dy = maxes.d
mig = mig.reshape(maxes.n,order='F')

print(np.min(srcy),np.min(srcx),np.max(srcy),np.max(srcx))

fig = plt.figure(); ax = fig.gca()
ax.imshow(np.flipud(bytes2float(mig[400].T)),cmap='gray',extent=[0,nx*dx,0,ny*dy])
ax.scatter(srcx,srcy)

grid = True
if(grid):
  bg1 = np.min(srcx); eg1 = np.max(srcx); dg1 = 0.025
  xticks = np.arange(bg1,eg1,dg1)
  print(len(xticks),bg1,dg1)
  bg2 = np.min(srcy); eg2 = np.max(srcy); dg2 = 0.05
  yticks = np.arange(bg2,eg2,dg2)
  print(len(yticks),bg2,dg2)
  ax.set_xticks(xticks)
  ax.set_yticks(yticks)
  plt.rc('grid',linestyle="-",color='black')
  plt.grid()

plt.show()
