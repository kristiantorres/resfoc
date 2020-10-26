import inpout.seppy as seppy
import numpy as np
import time
from oway.hessnchunkr import hessnchunkr
from server.distribute import dstr_sum
from server.utils import startserver, stopserver
from scaas.wavelet import ricker
from oway.costaper import costaper
from scaas.trismooth import smooth
from deeplearn.utils import next_power_of_2
from resfoc.resmig import rand_preresmig, convert2time, get_rho_axis
from scaas.off2ang import off2angkzx, get_angkzx_axis
from client.sshworkers import launch_sshworkers, kill_sshworkers
from genutils.ptyprint import progressbar

# IO
sep = seppy.sep()

# Start workers
hosts = ['fantastic','storm','jarvis','thing','torch']
cfile = "/homes/sep/joseph29/projects/scaas/oway/hessnworker.py"
launch_sshworkers(cfile,hosts=hosts,sleep=1,verb=1,clean=True)

# Read in velocity
vaxes,vels = sep.read_file("hale_trvels.H")
vels = np.ascontiguousarray(vels.reshape(vaxes.n,order='F')).astype('float32')
[nz,nvx,nm] = vaxes.n; [dz,dvx,dm] = vaxes.d; [oz,ovx,om] = vaxes.o
# Read in reflectivity
raxes,refs = sep.read_file("hale_trrefs.H")
[nz,nrx,nm] = raxes.n; [dz,drx,dm] = raxes.d; [oz,orx,om] = raxes.o
refs = np.ascontiguousarray(refs.reshape(raxes.n,order='F')).astype('float32')
ny = 1; dy = 1

# Residual migration axis
nro = 41; dro = 0.001250

# Read in anomalies
aaxes,anos = sep.read_file("hale_tranos.H")
anos = np.ascontiguousarray(anos.reshape(aaxes.n,order='F')).astype('float32')

# Read in the acquisition geometry
saxes,srcx = sep.read_file("hale_srcxflatbob.H")
raxes,recx = sep.read_file("hale_recxflatbob.H")
_,nrec= sep.read_file("hale_nrecbob.H")
nrec = nrec.astype('int')

# Convert velocity to slowness
slo = np.zeros([nz,ny,nvx],dtype='float32')
ano = np.zeros([nz,ny,nvx],dtype='float32')
ref = np.zeros([nz,ny,nrx],dtype='float32')

# Create ricker wavelet
n1   = 1500; d1 = 0.004;
freq = 20; amp = 0.5; t0 = 0.2;
wav  = ricker(n1,d1,freq,amp,t0)

# Start server
context,socket = startserver()

print("Image grid: nxi=%d oxi=%f dxi=%f"%(nrx,orx,drx))

# Loop over all models
beg,end = 2,50
for im in progressbar(range(beg,end),"nmod:"):
  # Get the current example
  velin = vels[:,:,im]
  refin = refs[:,:,im]
  anoin = anos[:,:,im]

  # Smooth in slowness
  slo[:,0,:] = smooth(1/velin,rect1=40,rect2=30)
  vel = 1/slo

  # Build the reflectivity
  reftap = costaper(refin,nw1=16)
  ref[:,0,:] = reftap[:]

  # Prepare the anomaly
  ano[:,0,:] = anoin

  # One example without anomaly, one with
  for k in range(2):
    # Introduce anomaly
    if(k == 0):
      velmig = vel
    if(k == 1):
      velmig = vel*ano

    nchnk = len(hosts)
    hcnkr = hessnchunkr(nchnk,
                        drx,dy,dz,
                        ref,vel,velmig,wav,d1,t0,minf=1.0,maxf=51.0,
                        nrec=nrec,srcx=srcx,recx=recx,
                        ovx=ovx,dvx=dvx,ox=orx,verb=False)

    hcnkr.set_hessn_pars(ntx=16,nhx=20,nthrds=40,nrmax=20,mpx=100,sverb=False)
    gen = iter(hcnkr)

    # Distribute work to workers and sum over results
    img = dstr_sum('cid','result',nchnk,gen,socket,hcnkr.get_img_shape())

    # Get offset axis
    nhx,ohx,dhx = hcnkr.get_offx_axis()

    # Transpose the image
    imgt = np.ascontiguousarray(np.transpose(img,(0,1,3,4,2))) # [nhy,nhx,nz,ny,nx] -> [nhy,nhx,ny,nx,nz]

    # Create a residually-defocused image
    if(k == 0):
      nsin  = [nhx,nrx,nz]
      nps = [next_power_of_2(nin)+1 for nin in nsin]
      rmig,rho  = rand_preresmig(imgt[0,:,0,:,:],[dhx,drx,dz],nps=nps,nro=nro,dro=dro,offset=14,verb=False)
      rmigt = convert2time(rmig,dz,dt=d1,oro=rho,dro=dro,verb=False)[0]
      rmige = rmigt[np.newaxis,:,np.newaxis,:,:]

    # Convert to angle
    na = 64
    if(k == 0):
      iang = off2angkzx(imgt ,ohx,dhx,dz,na=na,nthrds=10,transp=True,cverb=False)
      rang = off2angkzx(rmige,ohx,dhx,dz,na=na,nthrds=10,transp=True,cverb=False)
    elif(k == 1):
      iang = off2angkzx(imgt ,ohx,dhx,dz,na=na,nthrds=10,transp=True,cverb=False)
    na,oa,da = get_angkzx_axis(na,amax=60)
    iangc = iang[0]
    rangc = rang[0]

    # Write the example to file
    if(im == 0):
      if(k == 0):
        sep.write_file("hale_foctrimgs.H",iangc.T,os=[oz,oa,0.0,orx],ds=[dz,da,1.0,drx])
        sep.write_file("hale_restrimgs.H",rangc.T,os=[oz,oa,0.0,orx],ds=[dz,da,1.0,drx])
        sep.write_file("hale_randrhos.H",np.asarray([rho]))
      elif(k == 1):
        sep.write_file("hale_deftrimgs.H",iangc.T,os=[oz,oa,0.0,orx],ds=[dz,da,1.0,drx])
    elif(im == 1):
      if(k == 0):
        sep.append_file("hale_foctrimgs.H",iangc.T,newaxis=True)
        sep.append_file("hale_restrimgs.H",rangc.T,newaxis=True)
        sep.append_file("hale_randrhos.H",np.asarray([rho]))
      elif(k == 1):
        sep.append_file("hale_deftrimgs.H",iangc.T,newaxis=True)
    else:
      if(k == 0):
        sep.append_file("hale_foctrimgs.H",iangc.T)
        sep.append_file("hale_restrimgs.H",rangc.T)
        sep.append_file("hale_randrhos.H",np.asarray([rho]))
      elif(k == 1):
        sep.append_file("hale_deftrimgs.H",iangc.T)

# Clean up
kill_sshworkers(cfile,hosts,verb=False)

