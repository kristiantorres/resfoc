import inpout.seppy as seppy
import numpy as np
from velocity.custommodels import random_hale_vel, fake_fault_img
from genutils.plot import plot_img2d, plot_vel2d
from deeplearn.utils import plot_seglabel

sep = seppy.sep()

sep = seppy.sep()
iaxes,img = sep.read_file("spimgextbobdistr.H")
[nz,nx,ny,nhx] = iaxes.n; [oz,ox,oy,ohx] = iaxes.o; [dz,dx,dy,dhx] = iaxes.d
stk = img.reshape(iaxes.n,order='F').T
stk = stk[20,0]

vaxes,hvel = sep.read_file("vintzcomb.H")
nvz,nvx = vaxes.n; ovz,ovx = vaxes.o; dvz,dvx = vaxes.d
hvel = hvel.reshape(vaxes.n,order='F').T

vzin = hvel[150,45:]
vel1,ref1,lbl1 = random_hale_vel(vzin)
vel2,ref2,lbl2 = fake_fault_img(hvel,stk)
#plot_img2d(ref2,pclip=0.4)

#plot_img2d(ref[:,120:660],show=False)
#plot_img2d(stkp[120:660].T,show=True)
#plot_vel2d(vel,show=False)
#plot_img2d(ref,show=False)
#plot_seglabel(ref,lbl,show=True)
#plot_img2d(stkp.T,show=True)

ovels = np.zeros([2,vel1.shape[1],vel1.shape[0]],dtype='float32')
orefs = np.zeros([2,ref1.shape[1],ref1.shape[0]],dtype='float32')
olbls = np.zeros([2,lbl1.shape[1],lbl1.shape[0]],dtype='float32')

ovels[0] = vel1.T; ovels[1] = vel2.T
orefs[0] = ref1.T; orefs[1] = ref2.T
olbls[0] = lbl1.T; olbls[1] = lbl2.T

sep.write_file("nvels.H",ovels.T,ds=[dz,dx,1.0],os=[0.0,5.36,0.0])
sep.write_file("nlbls.H",olbls.T,ds=[dz,dx,1.0],os=[0.0,5.36,0.0])
sep.write_file("nrefs.H",orefs.T,ds=[dz,dx,1.0],os=[0.0,5.36,0.0])

