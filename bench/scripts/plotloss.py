import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep([])

laxes,loss = sep.read_file(None,ifname='./dat/nnhistory/fltclasswgt5-gpu10ep_loss.H')
vaxes,vlss = sep.read_file(None,ifname='./dat/nnhistory/fltclasswgt5-gpu10ep_vlss.H')
aaxes,accu = sep.read_file(None,ifname='./dat/nnhistory/fltclasswgt5-gpu10ep_accu.H')
caxes,vacu = sep.read_file(None,ifname='./dat/nnhistory/fltclasswgt5-gpu10ep_vacu.H')

fig = plt.figure(1,figsize=(7,7)); ax = fig.gca()
plt.plot(loss/np.max(loss))
plt.plot(vlss/np.max(vlss))
ax.set_xlabel('Epochs',fontsize=18)
ax.set_ylabel('Loss',fontsize=18)
ax.tick_params(labelsize=18)
plt.savefig('./fig/ep10loss.png',bbox_inches='tight',dpi=150,transparent=True)

fig = plt.figure(2,figsize=(7,7)); ax = fig.gca()
plt.plot(accu)
plt.plot(vacu)
ax.set_xlabel('Epochs',fontsize=18)
ax.set_ylabel('Accuracy (%)',fontsize=18)
ax.tick_params(labelsize=18)
plt.savefig('./fig/ep10accu.png',bbox_inches='tight',dpi=150,transparent=True)

