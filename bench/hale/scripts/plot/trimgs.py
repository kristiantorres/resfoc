import inpout.seppy as seppy
from genutils.plot import plot_img2d

sep = seppy.sep()
iaxes,img = sep.read_file("hale_foctrstks.H")
img = img.reshape(iaxes.n,order='F').T

iaxes = sep.read_header("spimgbobang.H")
dz,da,dy,dx = iaxes.d; oz,oa,oy,ox = iaxes.o

begx,endx = 140,652
begz,endz = 100,356

imgw = img[1,begx:endx,0,begz:endz]

#plot_img2d(imgw.T,pclip=0.5,oz=100*dz,dz=dz,dx=dx,ox=ox+30*dx,aspect=3.0,figname='./fig/fakefault/psf.png')

saxes,stk = sep.read_file("nrefs.H")
stk = stk.reshape(saxes.n,order='F').T

stkw = stk[1,begx:endx,begz:endz]

plot_img2d(stkw.T,pclip=0.5,oz=100*dz,dz=dz,dx=dx,ox=ox+30*dx,aspect=3.0,figname='./fig/fakefault/ipsf.png')

#iaxes,img = sep.read_file("./dat/split/img-00272.H")
#img = img.reshape(iaxes.n,order='F')
#
#saxes,smt = sep.read_file("./dat/split/img-00272-sos.H")
#smt = smt.reshape(saxes.n,order='F')
#
#plot_img2d(img,pclip=0.5,oz=100*dz,dz=dz,dx=dx,ox=ox+30*dx,aspect=3.0,figname='./fig/trimgs/trex.png')
#plot_img2d(smt,pclip=0.5,oz=100*dz,dz=dz,dx=dx,ox=ox+30*dx,aspect=3.0,figname='./fig/trimgs/trexsm.png')
