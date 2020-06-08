import numpy as np
import h5py
from utils.plot import plot_cubeiso
import matplotlib.pyplot as plt

#hf = h5py.File('/net/fantastic/scr2/joseph29/angfaultfocptch.h5')
hf = h5py.File('/net/fantastic/scr2/joseph29/angfaultdefptch.h5')

keys = list(hf.keys())

iptch = np.random.randint(105)
iex = np.random.randint(1000)
#iptch = 1; iex = 266
print(iex,iptch)
onex = hf[keys[iex]][iptch,:,:,:,0]

hf.close()

print(onex.shape)

plot_cubeiso(onex,os=[-70.0,0.0,0.0],ds=[2.22,0.01,0.01],stack=True,show=False,hbox=8,wbox=8,elev=15,
             x1label='\nX (km)',x2label='\nAngle'+r'($\degree$)',x3label='\nDepth (km)',verb=False)


mystk = np.sum(onex,axis=0)
plt.figure()
plt.imshow(mystk,cmap='gray',interpolation='sinc')
plt.figure()
plt.imshow(onex[:,:,32].T,cmap='gray',interpolation='sinc')
plt.show()

