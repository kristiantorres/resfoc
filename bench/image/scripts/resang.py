import inpout.seppy as seppy
import numpy as np
from scaas.off2ang import off2ang,get_ang_axis
from resfoc.resmig import preresmig,get_rho_axis
from resfoc.gain import tpow
from genutils.movie import viewimgframeskey
from genutils.plot import plot_imgpang, plot_allanggats, plot_anggatrhos

sep = seppy.sep()

iaxes,img = sep.read_file("fltimgextprcnew2.H")
img = img.reshape(iaxes.n,order='F')
[dz,dx,dh] = iaxes.d

imgt = np.ascontiguousarray(np.transpose(img,(2,0,1))).astype('float32')

# First do residual migration
inro = 10; idro = 0.0025
storm = preresmig(imgt,[dh,dz,dx],nps=[2049,513,1025],nro=inro,dro=idro,time=True,transp=True,verb=True,nthreads=24)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

# Convert to angle
oh = iaxes.o[2]
stormang = off2ang(storm,oh,dh,dz,oro=ooro,dro=odro,transp=True,verb=True,nthrds=24)
na,oa,da = get_ang_axis()

# Look at the data
plot_imgpang(stormang[9],dx/1000.0,dz/1000.0,512,oa,da,aaspect=300,show=False)
plot_allanggats(stormang[9],dx/1000.0,dz/1000.0,transp=True,jx=10,pclip=0.7,show=False)
plot_anggatrhos(stormang,512,dz/1000.0,dx/1000.0,ooro,odro,transp=True)

# Write to file
stormangt = np.transpose(stormang,(2,1,3,0))
sep.write_file("bigresmigtest2time.H",stormangt,os=[0,oa,0,ooro],ds=[dz,da,dx,odro])

