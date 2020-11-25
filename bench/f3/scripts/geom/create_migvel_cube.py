"""
Creates a cube from the migration velocity file

@author: Joseph Jennings
@version: 2020.09.27
"""
import inpout.seppy as seppy
from inpout.seppy import bytes2float
import numpy as np
from scipy.interpolate import griddata
from genutils.ptyprint import progressbar
import matplotlib.pyplot as plt

# Initialize IO
sep = seppy.sep()

# Output time axis
nt = 1500; dt = 4
times = np.linspace(0,(nt-1)*dt,nt)

# Parse the migration velocity .vel file
vline = []; lnums = []
vssp  = []; vnums = []; vlnums = []
veff  = []; teff  = []
with open('./vels/C00154001.vel','r') as f:
#with open('./vels/C00152346.vel','r') as f:
  for iline in f.readlines():
    lsp = iline.split()
    if(lsp[0] == 'INFO'): continue
    elif(lsp[0] == 'LINE'):
      if(len(vssp) != 0):
        lnums.append(lprev)
        vlnums.append(vnums)
        vline.append(vssp)
      lprev = lsp[1]
      vssp = []; vnums = []
      veff = []; teff = []
    elif(lsp[0] == 'VELSSP'):
      if(len(veff) != 0):
        vnums.append(vprev)
        # Interpolate the times and velocities
        veff = np.asarray(veff)
        teff = np.asarray(teff)
        velint = np.interp(times,teff,veff)
        vssp.append(list(velint))
      vprev = lsp[2]
      veff = []; teff = []
    elif(lsp[0] == 'VEFF'):
      # Append until new VELSSP
      veff += list(map(float,lsp[1::2]))
      teff += list(map(float,lsp[2::2]))

# Get coordinates of velocity picks
xc = []
yc = []
lnums = np.asarray(lnums,dtype='float').astype('int')
for iline in range(len(lnums)):
  cdps = np.asarray(vlnums[iline],dtype='float').astype('int')
  for issp in range(len(cdps)):
    yc.append(lnums[iline])
    xc.append(vlnums[iline][issp])

# Get corresponding velocity values
k = 0
vc = np.zeros([nt,len(xc)],dtype='float32')
for iline in range(len(lnums)):
  cdps = np.asarray(vlnums[iline],dtype='float').astype('int')
  for issp in range(len(cdps)):
    for it in range(nt):
      vc[it,k] = vline[iline][issp][it]
    k += 1

#TODO: need to change the ox and the oy
# Get spatial axes from migration cube
maxes = sep.read_header("./mig/mig.H")
[nt,nx,ny] = maxes.n; [ot,ox,oy] = maxes.o; [dt,dx,dy] = maxes.d

# Read in mask
kaxes,msk = sep.read_file("./vels/migmsk.H")
msk = msk.reshape(kaxes.n,order='F').T

yc = (np.asarray(yc,dtype='float').astype('int') - 99)*dy + oy
xc = (np.asarray(xc,dtype='float').astype('int'))*dx + ox

xi = np.linspace(ox,ox+(nx-1)*dx,nx)
yi = np.linspace(oy,oy+(ny-1)*dy,ny)
xi,yi = np.meshgrid(xi,yi)

# Output migration velocity
vels = np.zeros([nt,ny,nx],dtype='float32')

# Now grid the data for each time slice
for it in progressbar(range(nt),"nt:"):
  vels[it] = griddata((xc,yc),vc[it],(xi,yi),method='linear')
  #vels[it] = griddata((xc,yc),vc[it],(xi,yi),method='cubic')
  vels[it] *= msk

vels *= 0.001
velst = np.transpose(vels,axes=(0,2,1))
sep.write_file("./vels/migvelslint.H",velst,ds=maxes.d,os=maxes.o,dpath="/net/brick5/data3/northsea_dutch_f3/vels/")

