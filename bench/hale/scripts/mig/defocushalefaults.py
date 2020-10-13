import inpout.seppy as seppy
import numpy as np
from scaas.velocity import create_randomptbs_loc
from oway.imagechunkr import imagechunkr
from server.distribute import dstr_collect, dstr_sum
from server.utils import startserver
from client.sshworkers import launch_sshworkers, kill_sshworkers
from genutils.plot import plot_vel2d
from genutils.ptyprint import progressbar

# IO
sep = seppy.sep()

# Start workers
hosts = ['fantastic','storm', 'thing', 'torch', 'jarvis','jarvis']
cfile = "/homes/sep/joseph29/projects/scaas/oway/imageworker.py"
launch_sshworkers(cfile,hosts=hosts,sleep=1,verb=1,clean=True)

# Read in data
daxes,dat = sep.read_file("hale_shotflatbob.H")
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
[nt,ntr] = daxes.n; [ot,_] = daxes.o; [dt,_] = daxes.d

# Read in the velocity model
vaxes,vel = sep.read_file("vintzcomb.H")
vel = vel.reshape(vaxes.n,order='F')
[nz,nvx] = vaxes.n; [dz,dvx] = vaxes.d; [oz,ovx] = vaxes.o
ny = 1; dy = 1.0
velin = np.zeros([nz,ny,nvx],dtype='float32')
velin[:,0,:] = vel
ano = np.zeros(velin.shape,dtype='float32')

# Read in coordinates
saxes,srcx = sep.read_file("hale_srcxflatbob.H")
raxes,recx = sep.read_file("hale_recxflatbob.H")
_,nrec= sep.read_file("hale_nrecbob.H")
nrec = nrec.astype('int')

nxi = int(2*nvx); dxi = dvx/2; oxi = ovx

# Bind to socket
ctx,socket = startserver()

nvel = 12
for ivel in progressbar(range(2,nvel),"nvels:",verb=True):
  print("Vel %d/%d"%(ivel,nvel))
  # Create velocity anomaly
  anoin = create_randomptbs_loc(nz,nvx,nptbs=3,romin=0.93,romax=1.00,
                                minnaz=100,maxnaz=150,minnax=50,maxnax=150,
                                mincz=int(0.13*nz),maxcz=int(0.22*nz),
                                mincx=int(0.25*nvx),maxcx=int(0.75*nvx),
                                mindist=50,nptsz=2,nptsx=2,octaves=2,period=80,persist=0.2,sigma=20)
  ano[:,0,:] = anoin

  # Apply to migration velocity
  velmig = velin*ano
  #plot_vel2d(velin[:,0,:] ,dx=dvx,ox=ovx,dz=dz,title='Vel',show=False)
  #plot_vel2d(ano[:,0,:],dx=dvx,ox=ovx,dz=dz,cmap='seismic',title='Ano',show=False)
  #plot_vel2d(velmig[:,0,:],dx=dvx,ox=ovx,dz=dz,title='Vel+Ano',show=True)

  nchnk = len(hosts)
  icnkr = imagechunkr(nchnk,
                      nxi,dxi,ny,dy,nz,dz,velmig,
                      dat,dt,minf=1.0,maxf=51.0,
                      nrec=nrec,srcx=srcx,recx=recx,
                      ovx=ovx,dvx=dvx,ox=oxi,verb=False)

  icnkr.set_image_pars(ntx=16,nhx=20,nthrds=40,nrmax=10,sverb=False)
  gen = iter(icnkr)

  # Distribute work to workers and sum over results
  img = dstr_sum('cid','result',nchnk,gen,socket,icnkr.get_img_shape())

  # Transpose the image
  imgt = np.transpose(img,(2,4,3,1,0)) # [nhy,nhx,nz,ny,nx] -> [nz,nx,ny,nhx,nhy]
  imgts = imgt[0]
  # Get offset axis
  nhx,ohx,dhx = icnkr.get_offx_axis()

  # Write the result
  if(ivel == 0):
    sep.write_file("faultfocus.H",imgt,os=[oz,oxi,0.0,ohx],ds=[dz,dxi,dy,dhx])
    sep.write_file("velfltfocus.H",velmig,os=[oz,0.0,ovx],ds=[dz,dy,dvx])
  # Append
  elif(ivel == 1):
    sep.append_file("faultfocus.H",imgt,newaxis=True)
    sep.append_file("velfltfocus.H",velmig,newaxis=True)
  else:
    sep.append_file("faultfocus.H",imgt)
    sep.append_file("velfltfocus.H",velmig)

kill_sshworkers(cfile,hosts,verb=False)

