import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt

# Grid origin and sampling
ox,oy = 469800,6072350
dz,dx,dy = 10,25,25

sep = seppy.sep()
maxes,mig = sep.read_file("/data3/northsea_dutch_f3/mig/mig.T")
mig = mig.reshape(maxes.n,order='F')
migw  = mig[:,200:1200,5:505]
migslc = migw[400]

# Windowed grid
oxw = ox + 200*dx; oyw = oy + 5*dy
nzw1,nxw1,nyw1 = migw.shape

# Plotting axes
oxp,oyp = oxw*0.001,oyw*0.001
dzp,dxp,dyp = dz*0.001,dx*0.001,dy*0.001

sxaxes,srcx = sep.read_file("/data3/northsea_dutch_f3/f3_srcx2.H")
syaxes,srcy = sep.read_file("/data3/northsea_dutch_f3/f3_srcy2.H")

rxaxes,recx = sep.read_file("/data3/northsea_dutch_f3/f3_recx2.H")
ryaxes,recy = sep.read_file("/data3/northsea_dutch_f3/f3_recy2.H")

naxes,nrec= sep.read_file("/data3/northsea_dutch_f3/f3_nrec2.H")
nrec = nrec.astype('int32')

#fig = plt.figure(figsize=(10,5)); ax = fig.gca()
#im = ax.imshow(np.flipud(migslc.T),extent=[oxp,oxp+nxw1*dxp,oyp,oyp+nyw1*dyp],
#               interpolation='bilinear',cmap='gray')
#ax.scatter(srcx/1000.0,srcy/1000.0,marker='*',color='tab:red')
#plt.show()

idx = 0
for isht in range(len(nrec)):
  fig = plt.figure(figsize=(10,5)); ax = fig.gca()
  im = ax.imshow(np.flipud(migslc.T),extent=[oxp,oxp+nxw1*dxp,oyp,oyp+nyw1*dyp],
                 interpolation='bilinear',cmap='gray')
  ax.scatter(srcx[isht]/1000.0,srcy[isht]/1000.0,marker='*',color='tab:red')
  ax.scatter(recx[idx:idx+nrec[isht]]/1000.0,recy[idx:idx+nrec[isht]]/1000.0,marker='v',color='tab:green')
  idx += nrec[isht]
  plt.show()

