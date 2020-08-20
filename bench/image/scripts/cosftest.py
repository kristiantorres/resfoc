import numpy as np
import inpout.seppy as seppy
from resfoc.resmig import preresmig,get_rho_axis
from resfoc.cosft import cosft,icosft
from genutils.movie import viewcube3d

sep = seppy.sep()

iaxes,img = sep.read_file("rugimgsmall.H")

img = img.reshape(iaxes.n,order='F')
dz = iaxes.d[0]; dx = iaxes.d[1]; dh = iaxes.d[2]

imgt = np.ascontiguousarray(np.transpose(img,(2,0,1))).astype('float32')

viewcube3d(img,show=False)

imgft = cosft(imgt,axis1=1,axis2=1,axis3=1).astype('float32')

#imgftt = np.transpose(imgft,(1,2,0))
#viewcube3d(imgftt,pclip=0.1)

imgift = icosft(imgft,axis1=1,axis2=1,axis3=1)
imgiftt = np.transpose(imgift,(1,2,0))
viewcube3d(imgiftt,show=False)

inro = 1; idro = 0.005
storm = preresmig(imgt,[dh,dz,dx],nro=inro,dro=idro,time=False,transp=True,verb=True,nthreads=2)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

stormt = np.transpose(storm,(0,2,3,1))
viewcube3d(stormt[0]-img,cbar=True)

