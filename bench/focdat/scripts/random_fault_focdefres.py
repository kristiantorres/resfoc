"""
Creates four different types of images for the same
reflectivity model:

  (a) Convolution in depth with a zero-phase wavelet
  (b) Migration with the correct migration velocity (offset and angle)
  (c) Migration with an incorrect migration velocity (offset and angle)
  (d) Residual migration with a relatively large rho value (offset and angle)

@author: Joseph Jennings
@version: 2020.06.06
"""
import sys, os, argparse, configparser
import numpy as np
import inpout.seppy as seppy
from velocity.stdmodels import velfaultsrandom
from scaas.velocity import create_randomptbs_loc
from scaas.trismooth import smooth
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
from scaas.gradtaper import build_taper
from resfoc.resmig import preresmig,convert2time,get_rho_axis
from scaas.off2ang import off2ang
import matplotlib.pyplot as plt
from genutils.plot import plot_imgvelptb

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "y",
    "na": 64,
    "nro": 21,
    "oro": 1.0,
    "dro": 0.00125,
    "offset": 5
    }
if args.conf_file:
  config = configparser.ConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

# Set defaults
parser.set_defaults(**defaults)

# Input files
ioArgs = parser.add_argument_group('Inputs and outputs')
ioArgs.add_argument("-vel",help="Output velocity model",type=str,required=True)
ioArgs.add_argument("-ref",help="Output reflectivity",type=str,required=True)
ioArgs.add_argument("-lbl",help="Output label",type=str,required=True)
ioArgs.add_argument("-cnv",help="Focused convolution image",type=str,required=True)
ioArgs.add_argument("-ptb",help="Velocity anomaly",type=str,required=True)
ioArgs.add_argument("-fimgo",help="Focused migrated image",type=str,required=True)
ioArgs.add_argument("-fimga",help="Focused migrated image",type=str,required=True)
ioArgs.add_argument("-dimgo",help="Defocused migrated image",type=str,required=True)
ioArgs.add_argument("-dimga",help="Defocused migrated image",type=str,required=True)
ioArgs.add_argument("-rimgo",help="Residually defocused image",type=str,required=True)
ioArgs.add_argument("-rimga",help="Residually defocused image",type=str,required=True)
ioArgs.add_argument("-dpath",help="Output binary datapath",type=str,required=True)
# Other arguments
parser.add_argument("-na",help="Number of angles to compute [64]",type=int)
parser.add_argument("-nro",help="Number of rhos [21]",type=int)
parser.add_argument("-oro",help="Origin of residual migration axis",type=float)
parser.add_argument("-dro",help="Sampling of residual migration axis",type=float)
parser.add_argument("-offset",help="Offset from rho=1",type=int)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Create IO for writing at the end
sep = seppy.sep()

# Create layered model
nx = 1024; nz = 512
vel,ref,cnv,lbl = velfaultsrandom(nz=nz,nx=nx)
dx = 10; dz = 10

plt.figure(1)
plt.imshow(vel,cmap='jet',extent=[0,nx*0.01,nz*0.01,0.0])
plt.figure(2)
plt.imshow(lbl,cmap='jet',extent=[0,nx*0.01,nz*0.01,0.0])
plt.figure(3)
plt.imshow(ref,cmap='gray',extent=[0,nx*0.01,nz*0.01,0.0])
plt.figure(4)
plt.imshow(cnv,cmap='gray',extent=[0,nx*0.01,nz*0.01,0.0])
plt.show()

# Create migration velocity
velmig = smooth(vel,rect1=30,rect2=30)

# Create a random perturbation
ano = create_randomptbs_loc(nz,nx,nptbs=3,romin=0.95,romax=1.05,
                            minnaz=100,maxnaz=150,minnax=100,maxnax=400,mincz=100,maxcz=150,mincx=200,maxcx=800,
                            mindist=100,nptsz=2,nptsx=2,octaves=2,period=80,persist=0.2,ncpu=1,sigma=20)

# Create velocity with anomaly
veltru = velmig*ano
velptb = veltru - velmig
plot_imgvelptb(ref,-velptb,dz,dx,velmin=-100,velmax=100,thresh=5,agc=False,show=True)

# Acquisition geometry
dsx = 10; bx = 50; bz = 50
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=103,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(veltru,cmap='jet',show=True)

# Create data axes
ntu = 6500; dtu = 0.001;
freq = 20; amp = 100.0; dly = 0.2;
wav = ricker(ntu,dtu,freq,amp,dly)

# Model true linearized data
dtd = 0.008
allshot = prp.model_lindata(veltru,ref,wav,dtd,verb=True,nthrds=24)

# Taper for migration
prp.build_taper(70,150)

# Wave equation depth migration
imgr = prp.wem(veltru,allshot,wav,dtd,nh=16,lap=True,verb=True,nthrds=24)
#imgw = prp.wem(velmig,allshot,wav,dtd,nh=16,lap=True,verb=True,nthrds=24)
nh,oh,dh = prp.get_off_axis()

# Transpose
imgrt = np.ascontiguousarray(np.transpose(imgr,(0,2,1))) # [nh,nz,nx] -> [nh,nx,nz]
#imgwt = np.ascontiguousarray(np.transpose(imgw,(0,2,1))) # [nh,nz,nx] -> [nh,nx,nz]

# Residual migration for a random rho
nro = args.nro; oro = args.oro; dro = args.dro
foro = oro - (nro-1)*dro; fnro = 2*nro-1
rhos = np.linspace(foro,foro + (fnro-1)*dro,2*nro-1)
# Choose a rho for defocusing
if(np.random.choice([0,1])):
  rho = np.random.randint(0,nro-args.offset)*dro + foro
else:
  rho = np.random.randint(nro+args.offset+1,fnro)*dro + foro

# Apply a secondary taper to the image before resmig
z1t = 70; z2t = 150
restap1d,restap = build_taper(nx,nz,z1t,z2t)
imgrttap = np.asarray([restap.T*imgrt[ih] for ih in range(nh)])

# Depth Residual migration for one rho
rmig  = preresmig(imgrttap,[dh,dx,dz],nps=[2049,nx+1,nz+1],nro=1,oro=rho,dro=dro,time=False,nthreads=1,verb=True)
onro,ooro,odro = get_rho_axis(nro=nro,dro=dro)

# Conversion to time
rmigt = convert2time(rmig,dz,dt=dtd,oro=rho,dro=odro,verb=True)

# Transpose
rmigtt = np.ascontiguousarray(np.transpose(rmigt,(0,1,3,2))).astype('float32') # [nro,nh,nx,nz] -> [nro,nh,nz,nx]

# Convert to angle gathers
imgrang  = prp.to_angle(imgr,na=args.na,verb=True,nthrds=24)
#imgwang  = prp.to_angle(imgw,na=args.na,verb=True,nthrds=24)
stormang = off2ang(rmigtt,oh,dh,dz,na=args.na,oro=ooro,dro=odro,verb=True,nthrds=24)
na,oa,da = prp.get_ang_axis()

# Output inputs
sep.write_file(args.vel,vel,ds=[dz,dx],dpath=args.dpath+'/')
sep.write_file(args.ref,ref,ds=[dz,dx],dpath=args.dpath+'/')
sep.write_file(args.lbl,lbl,ds=[dz,dx],dpath=args.dpath+'/')
sep.write_file(args.cnv,cnv,ds=[dz,dx],dpath=args.dpath+'/')
sep.write_file(args.ptb,-velptb,ds=[dz,dx],dpath=args.dpath+'/')
# Output images
sep.write_file(args.fimgo,imgrt.T,   ds=[dz,dx,dh],os=[0,0,oh],dpath=args.dpath+'/')
sep.write_file(args.fimga,imgrang.T, ds=[dz,da,dx],os=[0,oa,0],dpath=args.dpath+'/')
#sep.write_file(args.dimgo,imgwt.T,   ds=[dz,dx,dh],os=[0,0,oh],dpath=args.dpath+'/')
#sep.write_file(args.dimga,imgwang.T, ds=[dz,da,dx],os=[0,oa,0],dpath=args.dpath+'/')
sep.write_file(args.rimgo,rmig.T,    ds=[dz,dx,dh],os=[0,0,oh],dpath=args.dpath+'/')
sep.write_file(args.rimga,stormang.T,ds=[dz,da,dx],os=[0,oa,0],dpath=args.dpath+'/')

# Write the residual migration parameter to the files
with open(args.rimgo,'a') as f:
  f.write('rho=%f\n'%(rho))

with open(args.rimga,'a') as f:
  f.write('rho=%f\n'%(rho))

# Flag for cluster manager
print("Success!")

