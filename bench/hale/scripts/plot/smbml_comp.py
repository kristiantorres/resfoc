import inpout.seppy as seppy
from genutils.plot import plot_rhoimg2d, plot_img2d
from genutils.movie import viewimgframeskey

sep = seppy.sep()

# Read in poorly focused stack
saxes,stk = sep.read_file("faultfocusstkwind.H")
dz,dx = saxes.d; oz,ox = saxes.o
stk = stk.reshape(saxes.n,order='F')

# Read in estimate rhos
raxes,srho = sep.read_file("faultfocusrhowind.H")
srho = srho.reshape(raxes.n,order='F')

raxes,drho = sep.read_file("realtorch_rho.H")
drho = drho.reshape(raxes.n,order='F')

# Read in refocused images
iaxes,simg = sep.read_file("faultfocusrfiwind.H")
simg = simg.reshape(iaxes.n,order='F')

iaxes,dimg = sep.read_file("realtorch_rfi.H")
dimg = dimg.reshape(iaxes.n,order='F')

# Plot the rhos on the stack
plot_rhoimg2d(stk,srho,dz=dz,dx=dx,oz=oz,ox=ox,title='Semblance',show=False,aspect=2.0)

plot_rhoimg2d(stk,drho,dz=dz,dx=dx,oz=oz,ox=ox,title='DL',show=False,aspect=2.0)

viewimgframeskey([dimg,simg,stk],dx=dx,dz=dz,ox=ox,oz=oz,transp=False,pclip=0.6)

