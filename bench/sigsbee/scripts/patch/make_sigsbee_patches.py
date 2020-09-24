import inpout.seppy as seppy
from resfoc.gain import agc
from deeplearn.utils import normextract
from deeplearn.dataloader import WriteToH5
from deeplearn.focuslabels import label_defocused_patches
from genutils.plot import plot_cubeiso
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in focused residual migration
raxes,res = sep.read_file("sigsbeeresmsk.H")
[nz,na,nx,nro] = raxes.n
res  = res.reshape(raxes.n,order='F').T
rest = np.ascontiguousarray(np.transpose(res,(0,2,3,1)))

# Take three slices of the residual migration image
rol = rest[12]
ro1 = rest[20]
rog = rest[25]

# Read in unfocused image
daxes,dfc = sep.read_file("resmskoverwro1.H")
dfc  = dfc.reshape(daxes.n,order='F').T
dfct = np.ascontiguousarray(np.transpose(dfc,(1,2,0)))

# Make a list of the images
imgs = [rol,ro1,rog,dfct]

# Define window
bxw = 20;  exw = 200
bzw = 400; ezw = 900

# Patches
nzp = 64; nxp = 64
strdz = int(nzp/2 + 0.5)
strdx = int(nxp/2 + 0.5)

# Window each of the images
imgsw = [ img[:,bzw:ezw,bxw:exw] for img in imgs ]
aagcs = [ agc(img) for img in imgsw ]
iagcs = [ agc(np.sum(img,axis=0)) for img in imgsw ]

aptchs = [ normextract(img,nzp=nzp,nxp=nxp,strdz=strdz,strdx=strdx,norm=True) for img in aagcs ]
iptchs = [ normextract(img,nzp=nzp,nxp=nxp,strdz=strdz,strdx=strdx,norm=True) for img in iagcs ]

nex = aptchs[0].shape[0]


# Label the defocused patches
lmet,rollbls = label_defocused_patches(aptchs[0],aptchs[1],qc=True)
gmet,roglbls = label_defocused_patches(aptchs[2],aptchs[1],qc=True)
dmet,deflbls = label_defocused_patches(aptchs[3],aptchs[1],qc=True)

titles = ['RL-Defocused','Focused','RU-Defocused','Defocused']
mets = [lmet,None,gmet,dmet]
lbls = [rollbls,None,roglbls,deflbls]
for iex in range(nex):
  for i in range(4):
    if(i == 0 or i == 2 or i == 3):
      print("Fsemb=%f Dsemb=%f Rat=%f"%(mets[i]['fsemb'],mets[i]['dsemb'],mets[i]['sembrat']))
      if(lbls[i][iex] == 0): print("Defocused")
      plot_cubeiso(aptchs[i][iex],stack=True,elev=15,show=False,verb=False,title=titles[i])
      plot_cubeiso(aptchs[1][iex],stack=True,elev=15,show=True,verb=False,title=titles[1])

