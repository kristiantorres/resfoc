"""
Creates simple layered with three faults training data

Outputs the following:
  (a) A layered v(z) velocity model with three faults
  (b) The associated reflectivity model
  (c) The associated fault labels
  (d) A velocity anomaly
  (e) Poorly focused migrated image (no anomaly)
  (f) Prestack Residual migration (subsurface offsets)
  (g) Rho field picked from semblance
  (h) Refocused image

@author: Joseph Jennings
@version: 2020.05.01
"""
import numpy as np
import inpout.seppy as seppy
from velocity.stdmodels import layeredfaults2d
from scaas.velocity import create_randomptbs_loc
from scaas.trismooth import smooth
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
from resfoc.gain import agc
from utils.plot import plot_imgvelptb, plot_wavelet
import matplotlib.pyplot as plt

# Create layered model
vel,ref,cnv,lbl = layeredfaults2d(nx=1300,ofx=0.4,dfx=0.08)
#vel,ref,cnv,lbl = layeredfaults2d(nx=1000,ofx=0.55)
[nz,nx] = vel.shape
dx = 10; dz = 10

# Create migration velocity
velsm = smooth(vel,rect1=30,rect2=30)

# Create a random perturbation
ano = create_randomptbs_loc(nz,nx,nptbs=3,romin=0.95,romax=1.00,
                            minnaz=100,maxnaz=150,minnax=100,maxnax=400,mincz=100,maxcz=150,mincx=250,maxcx=700,
                            mindist=100,nptsz=2,nptsx=2,octaves=2,period=80,persist=0.2,ncpu=1,sigma=20)

# Create velocity with anomaly
velwr = velsm*ano
velptb = velwr - velsm
plot_imgvelptb(ref,velptb,dz,dx,velmin=-100,velmax=100,thresh=5,agc=False,show=True)

# Acquisition geometry
dsx = 20; bx = 50; bz = 50
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=66,dsx=dsx,bx=bx,bz=bz)
#prp = geom.defaultgeom(nx,dx,nz,dz,nsx=51,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(velsm,cmap='jet',show=False)
prp.plot_acq(ref,cmap='gray',show=False)

# Create data axes
ntu = 6500; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)
plot_wavelet(wav,dtu)

# Model linearized data
dtd = 0.004
allshot = prp.model_lindata(velsm,ref,wav,dtd,verb=True,nthrds=24)

# Taper for migration
prp.build_taper(70,150)
prp.plot_taper(ref,cmap='gray')

# Wave equation depth migration
img = prp.wem(velsm,allshot,wav,dtd,nh=16,lap=True,verb=True,nthrds=20)
nh,oh,dh = prp.get_off_axis()
imgt = np.transpose(img,(1,2,0))

#TODO: Residual migration, conversion to time and angle, semblance, picking

# Write outputs to file
sep = seppy.sep()
sep.write_file("veltest.H",vel,ds=[dz,dx])
sep.write_file("lbltest.H",lbl,ds=[dz,dx])
sep.write_file("anotest.H",ano,ds=[dz,dx])
sep.write_file("imgtest.H",imgt,ds=[dz,dx,dh],os=[0.0,0.0,oh])

