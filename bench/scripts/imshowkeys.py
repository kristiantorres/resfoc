import numpy as np
import matplotlib.pyplot as plt 

def plotframes(frames):
  curr_pos = 0 

  def key_event(e):
    nonlocal curr_pos,l

    if e.key == "right":
        curr_pos = curr_pos + 1 
    elif e.key == "left":
        curr_pos = curr_pos - 1 
    else:
        return
    curr_pos = curr_pos % frames.shape[0]

    ax.cla()
    ax.imshow(frames[curr_pos,:,:],cmap='gray')
    ax.set_title('%d'%(curr_pos))
    #l.set_data(frames[curr_pos,:,:])
    fig.canvas.draw()

  fig = plt.figure()
  fig.canvas.mpl_connect('key_press_event', key_event)
  ax = fig.add_subplot(111)
  # Show the first frame
  l = ax.imshow(frames[0,:,:],cmap='gray')
  ax.set_title('%d'%(curr_pos))
  plt.show()

#data = np.random.rand(100,1024,512)
#plotframes(data)

from utils.movie import viewframeskey
import numpy as np
import inpout.seppy as seppy


sep = seppy.sep([])

daxes,dat = sep.read_file(None,ifname='f3cube.H')
dat = dat.reshape(daxes.n,order='F').T

viewframeskey(dat,fast=False,interp='none',pclip=0.3,wbox=8,hbox=4)
#plotframes(dat)

