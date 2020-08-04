import inpout.seppy as seppy
import numpy as np
from resfoc.resmig import preresmig,convert2time,get_rho_axis
from scaas.off2ang import off2ang, get_ang_axis

sep = seppy.sep()

# Read in the defocused image
iaxes,img = sep.read_file("spimgbobfullext.H")
img  = img.reshape(iaxes.n,order='F')
imgt = np.ascontiguousarray(img.T).astype('float32')
imgtw = imgt[:,0,:,:]

# Get axes
[nz,nx,ny,nhx] = iaxes.n; [oz,ox,oy,ohx] = iaxes.o; [dz,dx,dy,dhx] = iaxes.d

# Depth Residual migration
inro = 21; idro = 0.00125
rmig = preresmig(imgtw,[dhx,dx,dz],nps=[65,1025,1025],nro=inro,dro=idro,time=False,nthreads=18,verb=True)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

# Convert to time
dtd = 0.004
rmigt = convert2time(rmig,dz,dt=dtd,dro=odro,verb=True)

# Convert to angle
#rmigtt = np.ascontiguousarray(np.transpose(rmigt,(0,1,3,2))).astype('float32') # [nro,nh,nx,nz] -> [nro,nh,nz,nx]
#stormang = off2ang(rmigtt,ohx,dhx,dz,ota=-0.02,oro=ooro,dro=odro,verb=True,nthrds=24,amax=60)
#na,oa,da = get_ang_axis()

# Write to file
sep.write_file("haleres.H",rmig.T,ds=[dz,dx,dhx,odro],os=[0,0,ohx,ooro])
sep.write_file("halerest.H",rmigt.T,ds=[dz,dx,dhx,odro],os=[0,0,ohx,ooro])
#sep.write_file("haleresang.H",stormang.T,ds=[dz,da,dx,odro],os=[0,oa,0,ooro])

