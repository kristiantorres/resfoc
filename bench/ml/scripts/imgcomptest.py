import inpout.seppy as seppy
import numpy as np
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from deeplearn.focuslabels import mse, focdefocflt_labels, extract_defocpatches, corrsim
from deeplearn.utils import normalize
import resfoc.depth2time as d2t
from resfoc.gain import agc
from resfoc.ssim import ssim,cwssim
from skimage.measure import compare_ssim
from utils.ptyprint import create_inttag
import matplotlib.pyplot as plt
from utils.movie import viewimgframeskey
from scipy.signal.signaltools import correlate2d

def pmetpptch(img,tgt,show=False):
  print("MSE=%f"%(mse(img,tgt)))
  print("SSIM=%f"%(ssim(img,tgt)))
  plt.figure()
  plt.imshow(tgt,interpolation='sinc',vmin=-2.5,vmax=2.5,cmap='gray')
  plt.figure()
  plt.imshow(img,interpolation='sinc',vmin=-2.5,vmax=2.5,cmap='gray')
  if(show):
    plt.show()

sep = seppy.sep()

tag = create_inttag(np.random.randint(0,1000),9999)
#tag = '0007'
#tag = '0228'
#tag = '0376'
#tag = '0935'

print(tag)

# Read in the images
foaxes,fog = sep.read_file("./dat/focdefoc/fog-%s.H"%(tag))
fog = fog.reshape(foaxes.n,order='F')
fog = np.ascontiguousarray(fog.T).astype('float32')
fogt = np.zeros(fog.shape,dtype='float32')
zofog = fog[16]

# Get axes
[nz,nx,nh] = foaxes.n; [oz,ox,oh] = foaxes.o; [dz,dx,dh] = foaxes.d

faaxes,fag = sep.read_file("./dat/focdefoc/fag-%s.H"%(tag))
fag = fag.reshape(faaxes.n,order='F')
fstk = np.sum(fag,axis=1)

doaxes,dog = sep.read_file("./dat/focdefoc/dog-%s.H"%(tag))
dog = dog.reshape(doaxes.n,order='F')
dog = np.ascontiguousarray(dog.T).astype('float32')
dogt = np.zeros(dog.shape,dtype='float32')
zodog = dog[16]

daaxes,dag = sep.read_file("./dat/focdefoc/dag-%s.H"%(tag))
dag = dag.reshape(daaxes.n,order='F')
dstk = np.sum(dag,axis=1)

faxes,flt = sep.read_file("./dat/focdefoc/lbl-%s.H"%(tag))
flt = flt.reshape(faxes.n,order='F')

vaxes,vel = sep.read_file("./dat/focdefoc/vel-%s.H"%(tag))
vel = vel.reshape(vaxes.n,order='F')
vel = np.ascontiguousarray(vel.T).astype('float32')
velc = np.zeros(fog.shape,dtype='float32')
for ih in range(nh):
  velc[ih,:,:] = vel[120:120+1024,:]

# Convert to time
#nt = nz; ot = 0.0; dt = 0.0112
#d2t.convert2time(nh,nx,nz,oz,dz,nt,ot,dt,velc,fog,fogt)
#d2t.convert2time(nh,nx,nz,oz,dz,nt,ot,dt,velc,dog,dogt)

#gzofogt = agc(fogt[16].astype('float32')).T
#gzodogt = agc(dogt[16].astype('float32')).T
gzofog  = agc(zofog.astype('float32')).T
gzodog  = agc(zodog.astype('float32')).T

#plt.figure(1)
#plt.imshow(gzofogt,cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5)
plt.figure(2)
plt.imshow(gzofog,cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5)
#plt.figure(3)
#plt.imshow(gzodogt,cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5)
plt.figure(4)
plt.imshow(gzodog,cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5)
plt.show()

#msed,msef,msel,mlblimg,mseimg = focdefocflt_labels(gzodog,gzofog,flt,metric='mse',focthresh=100,imgs=True)
ssid,ssif,ssil,slblimg,ssmimg = focdefocflt_labels(gzodog,gzofog,flt,strdz=32,strdx=32,metric='ssim',focthresh=0.2,imgs=True)
defocs = extract_defocpatches(gzodog,gzofog,flt,pixthresh=30,focthresh=0.65)

#plt.figure(1)
#plt.imshow(gzodog,cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5)
#plt.imshow(ssmimg,cmap='jet',interpolation='bilinear',vmin=0.0,alpha=0.1)

#plt.figure(2)
#plt.imshow(gzodog,cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5)
#plt.imshow(slblimg,cmap='jet',interpolation='bilinear',alpha=0.1)

#plt.figure(3)
#plt.imshow(gzofog,cmap='gray',interpolation='sinc',vmin=-2.5,vmax=2.5)

#viewimgframeskey(defocs,transp=False,interp='sinc',vmin=-2.5,vmax=2.5,show=True)

nzp   = 64; nxp   = 64
strdz = 64; strdx = 64

## Patch the images
pe = PatchExtractor((nzp,nxp),stride=(strdz,strdx))

zofptch = pe.extract(gzofog)
zodptch = pe.extract(gzodog)
#
stkfptch = pe.extract(fstk)
stkdptch = pe.extract(dstk)

numpz = zofptch.shape[0]; numpx = zofptch.shape[1]

zofptchf = zofptch.reshape([numpz*numpx,nzp,nxp])
zodptchf = zodptch.reshape([numpz*numpx,nzp,nxp])

for ip in range(len(ssid)):
  #print("ip=%d MSE=%f"%(ip,mse(ssid[ip],ssif[ip])))
  #print("ip=%d SSIM=%f"%(ip,ssim(ssid[ip],ssif[ip])))
  normd = normalize(ssid[ip]); normf = normalize(ssif[ip])
  corr = np.max(correlate2d(normd,normf,mode='same'))/np.sqrt((np.max(correlate2d(normd,normd,mode='same'))*np.max(correlate2d(normf,normf,mode='same'))))
  print("ip=%d CORR=%f"%(ip,corr))
  plt.figure(1)
  plt.imshow(ssif[ip],interpolation='sinc',vmin=-2.5,vmax=2.5,cmap='gray')
  plt.figure(2)
  plt.imshow(ssid[ip],interpolation='sinc',vmin=-2.5,vmax=2.5,cmap='gray')
  plt.show()

# Try correlation
#testfptch = normalize(ssif[0])
#testdptch = normalize(ssid[0])
#corr1 = correlate2d(testfptch,testdptch, mode='same')
#
#testfptch = normalize(ssif[16])
#testdptch = normalize(ssid[16])
#corr2 = correlate2d(testfptch,testdptch, mode='same')

#print(np.max(corr1),np.max(corr2))
#plt.figure()
#plt.imshow(corr1,cmap='jet',interpolation='bilinear')
#plt.figure()
#plt.imshow(corr2,cmap='jet',interpolation='bilinear')
#plt.show()

# 0007
#print("Patch 75:")
#pmetpptch(zodptchf[75],zofptchf[75])
#
#print("Patch 85:")
#pmetpptch(zodptchf[85],zofptchf[85],show=True)

# 0228
#print("Patch 52:")
#pmetpptch(zodptchf[52],zofptchf[52])

#print("Patch 56:")
#pmetpptch(zodptchf[56],zofptchf[56],show=True)

#viewimgframeskey(zofptchf,transp=False,show=False,vmin=-2.5,vmax=2.5,interp='sinc')
#viewimgframeskey(zodptchf,transp=False,vmin=-2.5,vmax=2.5,interp='sinc')

