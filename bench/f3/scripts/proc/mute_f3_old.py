import inpout.seppy as seppy
import numpy as np
from oway.mute import mute
from genutils.plot import plot_dat2d, plot_img2d
from genutils.ptyprint import progressbar
import matplotlib.pyplot as plt

# Data
sep = seppy.sep()
#saxes,sht = sep.read_wind("/data3/northsea_dutch_f3/f3_shots2.H",fw=0,nw=700000)
saxes,sht = sep.read_file("/data3/northsea_dutch_f3/f3_shots2.H")
sht = np.ascontiguousarray(sht.reshape(saxes.n,order='F').T).astype('float32')
smute = np.zeros(sht.shape,dtype='float32')
ntr,nt = sht.shape
dt = 0.002

# Geometry
sxaxes,srcx = sep.read_file("/data3/northsea_dutch_f3/f3_srcx2.H")
syaxes,srcy = sep.read_file("/data3/northsea_dutch_f3/f3_srcy2.H")
rxaxes,recx = sep.read_file("/data3/northsea_dutch_f3/f3_recx2.H")
ryaxes,recy = sep.read_file("/data3/northsea_dutch_f3/f3_recy2.H")

naxes,nrec = sep.read_file("/data3/northsea_dutch_f3/f3_nrec2.H")
nrec = nrec.astype('int32')
nsht = len(nrec)

maxes,mig = sep.read_file("/data3/northsea_dutch_f3/mig/mig.T")
mig = mig.reshape(maxes.n,order='F')
migw  = mig[:,200:1200,5:505]
migslc = migw[400]

dx=25
dy=25
ox=469800
oy=6072350

oxw = ox + 200*dx; oyw = oy + 5*dy
nxw1,nyw1 = migslc.shape

# Plotting axes
oxp,oyp = oxw*0.001,oyw*0.001
dxp,dyp = dx*0.001,dy*0.001

beg,end = 0,0
ctr = 0
i,k,totntr = 0,0,nrec[0]

qc = False
recxs,recys = [],[]
while(ctr < ntr):
  # Get the source and receiver coordinates
  isrcx,isrcy = srcx[k],srcy[k]
  irecx,irecy = recx[ctr],recy[ctr]
  dist = np.sqrt((isrcx-irecx)**2 + (isrcy-irecy)**2)
  if(qc):
    recxs.append(irecx/1000.0)
    recys.append(irecy/1000.0)
    fig = plt.figure(); ax = fig.gca()
    ax.imshow(np.flipud(migslc.T),extent=[oxp,oxp+nxw1*dxp,oyp,oyp+nyw1*dyp],
              interpolation='none',cmap='gray')
    ax.scatter(isrcx/1000.0,isrcy/1000.0,marker='*',color='tab:red')
    ax.scatter(recxs,recys,marker='v',color='tab:green')
    # Compute the distance to determine the type of mute
    plt.show()
  print(ctr,dist,flush=True)
  smute[ctr:ctr+120] = np.squeeze(mute(sht[ctr:ctr+120],dt=dt,dx=0.025,v0=1.5,t0=0.2,half=False))
  #plot_dat2d(sht[ctr:ctr+120,:1500],pclip=0.01,dt=dt,show=False,aspect=50)
  #plot_dat2d(smute[ctr:ctr+120,:1500],pclip=0.03,dt=dt,aspect=50)
  # Update counters
  ctr += 120
  if(ctr >= totntr):
    k += 1
    if(k < nsht):
      totntr += nrec[k]
    recxs,recys = [],[]
    #if(qc): print()
    print(ctr,totntr)
    print()

print(ctr,totntr)
#dmin,dmax = np.min(smute),np.max(smute)
#plot_dat2d(sht[:,:1500],dmin=dmin,dmax=dmax,pclip=0.05,dt=dt,aspect='auto',show=False)
#plot_dat2d(smute[:,:1500],dmin=dmin,dmax=dmax,pclip=0.05,dt=dt,aspect='auto')
