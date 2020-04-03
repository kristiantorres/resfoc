import inpout.seppy as seppy
import numpy as np
from scaas.off2ang import off2ang, get_ang_axis
from resfoc.resmig import preresmig, get_rho_axis
from resfoc.gain import tpow
from utils.movie import viewimgframeskey
from utils.plot import plot_imgpang, plot_allanggats, plot_anggatrhos

sep = seppy.sep()

iaxes,img = sep.read_file("rugimgsmalltaper.H")
img = img.reshape(iaxes.n,order='F')

dz = iaxes.d[0]; dx = iaxes.d[1]; dh = iaxes.d[2]

imgt = np.ascontiguousarray(np.transpose(img,(2,0,1))).astype('float32')
#viewimgframeskey(imgt,transp=False)

# First do residual migration
inro = 3; idro = 0.005
storm = preresmig(imgt,[dh,dz,dx],nro=inro,dro=idro,time=False,transp=True,verb=True,nthreads=1)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

nh = storm.shape[1]
# View the zero offset residual migration
#viewimgframeskey(tpow(storm[:,int(nh/2),:,:],dz,tpow=2),transp=False,ttlstring='rho=%.3f',ottl=ooro,dttl=odro)

# Convert to angle
oh = iaxes.o[2]
stormang = off2ang(storm,oh,dh,dz,oro=ooro,dro=odro,transp=True,verb=True,nthrds=4)
na,oa,da = get_ang_axis()

plot_imgpang(stormang[0],dx/1000.0,dz/1000.0,128,oa,da,aaspect=300,show=False)

plot_allanggats(stormang[0],dx/1000.0,dz/1000.0,transp=True,jx=4,pclip=0.7,show=False)

plot_anggatrhos(stormang,128,dz/1000.0,dx/1000.0,ooro,odro,transp=True)

## Write to file
stormangt = np.transpose(stormang,(2,1,3,0))
sep.write_file("rugimgsmallresmigang.H",stormangt,ors=[0,oa,0,ooro],ds=[dz,da,dx,odro])

