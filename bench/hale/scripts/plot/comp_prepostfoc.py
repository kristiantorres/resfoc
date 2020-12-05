import inpout.seppy as seppy
import numpy as np
from genutils.movie import viewimgframeskey

sep = seppy.sep()
taxes,post = sep.read_file("faultfocusrfiwindpost.H")
post = post.reshape(taxes.n,order='F').T

paxes,pre = sep.read_file("faultfocusrfiwind.H")
pre = pre.reshape(paxes.n,order='F').T

saxes,stk = sep.read_file("faultfocusstkwind.H")
stk = stk.reshape(saxes.n,order='F').T

dz,dx = saxes.d; oz,ox = saxes.o

viewimgframeskey([stk,post,pre],ox=ox,dx=dx,dz=dz,oz=oz,pclip=0.5)

