import inpout.seppy as seppy
import numpy as np
import scipy.ndimage as sciim
from pef.stat.pef1d import pef1d, pef1dmask
from pef.stat.conv1dm import conv1dm
from opt.linopt.essops.identity import identity
from opt.linopt.combops import chainop
from opt.linopt.cd import cd
from genutils.plot import plot_dat2d, plot_wavelet
import matplotlib.pyplot as plt

def hyperb(v,t0,sign,nx,ox,dx,nt,ot,dt,kind='linear') -> np.ndarray:
  """
  Models a hyperbola given the data, a velocity and a t0.

  Parameters:
    v    - constant velocity
    t0   - origin time of hyperbola
    nx   - number of x samples
    ox   - x-axis origin
    dx   - x-axis sampling
    nt   - number of time samples
    ot   - t-axis origin
    dt   - time sampling
    kind - type of interpolation to apply

  Returns a modeled hyperbola [nt,nx]
  """
  T = dt*(nt-1)
  samp = int(t0/dt)
  if(samp >= nt):
    raise Exception("t0 should not be greater than nt")
  spk = np.zeros(nt)
  spk[samp] = 1
  rpt = np.tile(spk,(nx,1)).T
  x   = np.linspace(ox,dx*(nx-1),nx)
  tsq = sign * x*x/(v*v)
  shift  = tsq/dt
  ishift = shift.astype(int)

  if(kind == 'linear'):
    nmo = np.array([sciim.interpolation.shift(rpt[:,ix],shift[ix],order=1) for ix in range(nx)]).T
  else:
    nmo = np.array([np.roll(rpt[:,ix],ishift[ix]) for ix in range(nx)]).T

  return nmo.T


# Build a fake gather
nt,ot,dt = 3073,0.0,0.002
nx,ox,dx = 120,0.0,0.025
hyp  = hyperb(2.5,1.0,1.0,nx,ox,dx,nt,ot,dt)
#hyp += hyperb(3,2.0,1.0,nx,ox,dx,nt,ot,dt)

# Convolve with the bubble
sep = seppy.sep()
baxes,bub = sep.read_file("nucleus.H")
ns = 0
#plot_wavelet(bub,dt)

bubc = np.array([np.convolve(hyp[ix,:],bub) for ix in range(nx)])[:,ns:nt+ns].astype('float32')
dbb  = np.zeros(bubc.shape,dtype='float32')

lags = np.arange(69,300,1).astype('int32')
lags[0] = 0

nlag= len(lags)
pef = pef1d(nt,nlag,lags,aux=bubc[0])
idat = pef.create_data()

flt = np.zeros(nlag,dtype='float32')
flt[0] = 1.0

mask = pef1dmask()
idop = identity()
zro = np.zeros(nlag,dtype='float32')
dkop = chainop([mask,pef],pef.get_dims())

pefres = []
invflt = cd(dkop,idat,flt,niter=300,rdat=zro,eps=0.01,ress=pefres)

cop = conv1dm(nt,nlag,lags,flt=invflt)

for itr in range(nx):
  cop.forward(False,bubc[itr],dbb[itr])
  #plt.figure(); plt.plot(bubc[itr])
  #plt.figure(); plt.plot(err); plt.show()
  #dbb[itr,:] = err[:]
#
plot_dat2d(bubc,dt=dt,dx=dx,aspect='auto',dmin=-2.5,dmax=2.5,show=False)
plot_dat2d(dbb,dt=dt,dx=dx,aspect='auto',dmin=-2.5,dmax=2.5)
#
##plot_dat2d(hyp,dt=dt,dx=dx,aspect='auto',dmin=-0.1,dmax=0.1,show=False)
##plot_dat2d(bubc,dt=dt,dx=dx,aspect='auto',dmin=-2.5,dmax=2.5)


