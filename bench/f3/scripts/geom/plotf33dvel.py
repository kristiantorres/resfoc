import inpout.seppy as seppy
import numpy as np
from scaas.trismooth import smooth
from genutils.movie import viewcube3d
from genutils.plot import plot3d

sep = seppy.sep([])

#Inline 112
#daxes,dat = sep.read_file('./vels/migvelslint.H')
#daxes,dat = sep.read_file('./vels/migvelscub.H')
daxes,dat = sep.read_file('./vels/miglintz.H')
#daxes,dat = sep.read_file('./vels/migvelint.H')

dat = dat.reshape(daxes.n,order='F').astype('float32')

ds = [0.01,0.025,0.025]
os = [0.0,0.0,0.0]
#ds = [0.004,0.025,0.025]
#os = [300*ds[0],0.0,0.0]

#nans = np.isnan(dat)
#dat[nans] = 0

#datsm = smooth(dat,rect1=100,rect2=100,rect3=100)

#viewcube3d(dat,ds=ds,os=os,vmin=1.5,vmax=5.0,
#           interp='none',width1=8,label1='X (Km)',label2='Y (km)',label3='Z (km)',
#           transp=False,cbar=True,cmap='jet',barlabel='Velocity (km/s)')

#viewcube3d(dat[300:700,:,:],ds=ds,os=os,vmin=1.5,vmax=5.0,loc1=6.5,#loc2=7.2,loc3=2.0,
#           interp='none',width1=8,label1='X (Km)',label2='Y (km)',label3='Time (s)',
#           transp=False,cbar=True,cmap='jet',barlabel='Velocity (km/s)',labelsize=15,ticksize=15)


#plot3d(dat[300:700,:,:],ds=ds,os=os,vmin=1.5,vmax=5.0,loc1=6.5,loc2=7.2,loc3=2.0,
#       interp='none',width1=8,label1='X (km)',label2='Y (km)',label3='Time (s)',
#       transp=False,cbar=True,cmap='jet',barlabel='Velocity (km/s)',labelsize=15,ticksize=15,
#       figname='./fig/intvelt.png',show=False)

plot3d(dat[:500,:,:],ds=ds,os=os,vmin=1.5,vmax=5.0,loc1=6.5,loc2=7.2,loc3=2.0,
       interp='none',width1=8,label1='X (km)',label2='Y (km)',label3='Z (km)',
       transp=False,cbar=True,cmap='jet',barlabel='Velocity (km/s)',labelsize=15,ticksize=15,
       figname='./fig/intvelz.png',show=False)

