import numpy as np
from scaas.noise_generator import perlin
import matplotlib.pyplot as plt

nx = 250; ny = 200; 
nptsx = 2; nptsy = 3

# Default arguments
octave = 4; period = 10.0; amp=1.0; persist=0.5; Ngrad = 80; ncpu = 1

## 2D plots
# Plotting octaves
nocts = 8
octs1 = np.zeros([nocts,ny,nx])
fig,ax = plt.subplots(1,7,figsize=(14,7))
for ioct in range(1,nocts):
  octs1[ioct-1] = perlin(x=np.linspace(0,nptsx,nx),y=np.linspace(0,nptsy,ny),octaves=ioct,amp=amp,persist=persist,Ngrad=Ngrad,period=period,ncpu=ncpu)
  im = ax[ioct-1].imshow(octs1[ioct-1],cmap='gray')
  ax[ioct-1].tick_params(labelsize=10)
  if(ioct != 1):
    ax[ioct-1].set_yticks([])
  ax[ioct-1].set_title('%d Octaves'%ioct,fontsize=14)

# Plotting persist
npsts = 5; dpst = 0.2; opst = 0.1
psts1 = np.zeros([npsts,ny,nx])
fig,ax = plt.subplots(1,npsts,figsize=(14,7))
for ipst in range(npsts):
  pst = opst + dpst*ipst
  psts1[ipst] = perlin(x=np.linspace(0,nptsx,nx),y=np.linspace(0,nptsy,ny),octaves=octave,amp=amp,persist=pst,Ngrad=Ngrad,period=period,ncpu=ncpu)
  im = ax[ipst].imshow(psts1[ipst],cmap='gray')
  ax[ipst].tick_params(labelsize=10)
  if(ipst != 0):
    ax[ipst].set_yticks([])
  ax[ipst].set_title('%.2f persist'%pst,fontsize=14)

# Plotting nptsx
nnpts = 10
pts1 = np.zeros([nnpts,ny,nx])
fig,ax = plt.subplots(1,5,figsize=(14,7))
k = 0
for ipts in range(1,nnpts,2):
  pts1[k] = perlin(x=np.linspace(0,ipts,nx),y=np.linspace(0,nptsy,ny),octaves=2,amp=amp,persist=persist,Ngrad=Ngrad,period=period,ncpu=1)
  im = ax[k].imshow(pts1[k],cmap='gray')
  ax[k].tick_params(labelsize=10)
  if(k != 0):
    ax[k].set_yticks([])
  ax[k].set_title("%d points"%ipts,fontsize=14)
  k += 1

# Plotting nptsy
nnpts = 10
pts1 = np.zeros([nnpts,ny,nx])
fig,ax = plt.subplots(1,5,figsize=(14,7))
k = 0
for ipts in range(1,nnpts,2):
  pts1[k] = perlin(x=np.linspace(0,nptsx,nx),y=np.linspace(0,ipts,ny),octaves=2,amp=amp,persist=persist,Ngrad=Ngrad,period=period,ncpu=1)
  im = ax[k].imshow(pts1[k],cmap='gray')
  ax[k].tick_params(labelsize=10)
  if(k != 0):
    ax[k].set_yticks([])
  ax[k].set_title("%d points"%ipts,fontsize=14)
  k += 1

# Plotting nptsx and nptsy
nnpts = 10
pts1 = np.zeros([nnpts,ny,nx])
fig,ax = plt.subplots(1,5,figsize=(14,7))
k = 0
for ipts in range(1,nnpts,2):
  pts1[k] = perlin(x=np.linspace(0,ipts,nx),y=np.linspace(0,ipts,ny),octaves=2,amp=amp,persist=persist,Ngrad=Ngrad,period=period,ncpu=1)
  im = ax[k].imshow(pts1[k],cmap='gray')
  ax[k].tick_params(labelsize=10)
  if(k != 0):
    ax[k].set_yticks([])
  ax[k].set_title("%d points"%ipts,fontsize=14)
  k += 1

# Plotting periods
prd = 1.0
fig,ax = plt.subplots(1,3,figsize=(10,7))
per1 = perlin(x=np.linspace(0,nptsx,nx),y=np.linspace(0,nptsy,ny),octaves=4,amp=amp,persist=persist,Ngrad=Ngrad,period=prd,ncpu=1)
im = ax[0].imshow(per1,cmap='gray')
ax[0].set_title('Period=%.2f Nptsx=%d Nptsy=%d'%(prd,nptsx,nptsy))
# Keep same period but increase ptsx
per2 = perlin(x=np.linspace(0,nptsx*2,nx),y=np.linspace(0,nptsy,ny),octaves=4,amp=amp,persist=persist,Ngrad=Ngrad,period=prd,ncpu=1)
im = ax[1].imshow(per2,cmap='gray')
ax[1].set_yticks([])
ax[1].set_title('Period=%.2f Nptsx=%d Nptsy=%d'%(prd,2*nptsx,nptsy))
# Keep same period but increase ptsx and ptsy
per3 = perlin(x=np.linspace(0,nptsx*2,nx),y=np.linspace(0,nptsy*2,ny),octaves=4,amp=amp,persist=persist,Ngrad=Ngrad,period=prd,ncpu=1)
im = ax[2].imshow(per3,cmap='gray')
ax[2].set_yticks([])
ax[2].set_title('Period=%.2f Nptsx=%d Nptsy=%d'%(prd,2*nptsx,2*nptsy))

plt.show()

