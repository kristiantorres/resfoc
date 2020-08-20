import inpout.seppy as seppy
import numpy as np
import subprocess
from resfoc.estro import refocusimg

sep = seppy.sep()
#TODO: create a loop over all files

# Semblance file to be processed
fin = "smbthing2.H"
base = fin.split(".H")[0]

# Read in the angle stack
saxes,stk = sep.read_file("stkthing2.H")
stk = stk.reshape(saxes.n,order='F')
stk = np.ascontiguousarray(stk.T).astype('float32')

[nz,nx,nro] = saxes.n; [dz,dx,dro] = saxes.d

# Normalize and pick semblance
sp = subprocess.check_call("Scale scale_to=1 < %s > %s-norm.H"%(fin,base),shell=True)
sp = subprocess.check_call("sfpick vel0=1.0 rect1=40 rect2=20 < %s-norm.H > rhopcksthing2.H"%(base),shell=True)
sp = subprocess.check_call("Rm %s-norm.H"%(base),shell=True)

# Read in picks and perform refocusing
raxes,rho = sep.read_file("rhopcksthing2.H")
rho = rho.reshape(raxes.n,order='F')
rho = np.ascontiguousarray(rho.T).astype('float32')

# Refocus the image with the picks
rfis = refocusimg(stk,rho,dro)
sep.write_file("rfimgthing2.H",rfis.T,ds=[dz,dx])

