"""
Utility functions for manipulating/processing the F3
dataset

@author: Joseph Jennings
@version: 2020.12.15
"""
import numpy as np
from oway.mute import mute
import subprocess
import matplotlib.pyplot as plt

#TODO: add a parameter for streamer size. A list with the size of each streamer (near 120)
#TODO: find the streamer header and use that to determine if you are on a new streamer line
#      seems that the header ShotPoint is promising for this
def mute_f3shot(dat,isrcx,isrcy,inrec,recx,recy,tp=0.5,vel=1450.0,dymin=15,dt=0.002,dx=0.025,
                close=False) -> np.ndarray:
  """
  Mutes a shot from the F3 dataset

  Parameters:
    dat   - an input shot gather from the F3 dataset [ntr,nt]
    isrcx - x source coordinate of the shot [float]
    isrcy - y source coordinate of the shot [float]
    inrec - number of receivers for this shot [int]
    recx  - x receiver coordinates for this shot [ntr]
    recy  - y receiver coordinates for this shot [ntr]
    vel   - water velocity [1450.0]
    tp    - length of taper [0.5s]
    dy    - minimum distance between streamers [20 m]
    dt    - temporal sampling interval [0.002]
    dx    - spacing between receivers [25 m]
    close - A last resort method if streamer searching will not work [False]

  Returns a muted shot gather
  """
  mut = np.zeros(dat.shape,dtype='float32')
  v0 = vel*0.001
  if(inrec%120 == 0 or close):
    nstream = inrec//120
    if(close):
      diff = 480 - inrec
      nstream += diff
    k = 0
    for istr in range(nstream):
      irecx,irecy = recx[k],recy[k]
      dist = np.sqrt((isrcx-irecx)**2 + (isrcy-irecy)**2)
      t0 = dist/vel
      if(t0 > 0.15):
        t0 = dist/(1500.0)
        v0 = 1.5
      else:
        v0 = vel*0.001
      mut[k:k+120] = np.squeeze(mute(dat[k:k+120],dt=dt,dx=dx,v0=v0,t0=t0,tp=tp,half=False,hyper=True))
      k += 120
  else:
    print("Warning: irregular number of receivers. Searching for different streamer lines")
    t0s = []
    dist = np.sqrt((isrcx-recx[0])**2 + (isrcy-recy[0])**2)
    t0 = dist/vel
    if(t0 > 0.15):
      t0 = dist/(1500.0)
      v0 = 1.5
    else:
      v0 = vel*0.001
    t0s.append(t0)
    beg,k,nstrm = 0,0,1
    # First find the near offset receivers
    nrecxs,nrecys = [],[]
    for itr in range(1,inrec):
      dy = np.abs(recy[itr] - recy[itr-1])
      k += 1
      # Check if you moved to another streamer
      if(dy >= dymin):
        # Compute the distance
        dist = np.sqrt((isrcx-recx[itr])**2 + (isrcy-recy[itr])**2)
        t0 = dist/vel
        if(t0 > 0.15):
          t0 = dist/1500.0
          v0 = 1.5
        else:
          v0 = vel*0.001
        t0s.append(t0)
        mut[beg:beg+k] = np.squeeze(mute(dat[beg:beg+k],dt=dt,dx=dx,v0=v0,t0=t0s[nstrm-1],tp=tp,
                                    half=False,hyper=True))
        beg += k
        k = 0
        nstrm += 1
    # Mute the last streamer
    mut[beg:] = np.squeeze(mute(dat[beg:],dt=dt,dx=dx,v0=v0,t0=t0s[nstrm-1],tp=tp,half=False,hyper=True))

  return mut

def compute_batches(batchin,totnsht):
  """
  Computes the starting and stoping points for reading in
  batches from the F3 data file.

  Parameters:
    batchin - target batch size
    totnsht - total number of shots to read in

  Returns the batch size and the start and end of
  each batch
  """
  divs = np.asarray([i for i in range(1,totnsht) if(totnsht%i == 0)])
  bsize = divs[np.argmin(np.abs(divs - batchin))]
  nb = totnsht//bsize

  return bsize,nb

def plot_acq(srcx,srcy,recx,recy,slc,ox,oy,
             dx=0.025,dy=0.025,srcs=True,recs=False,figname=None,**kwargs):
  """
  Plots the acqusition geometry on a depth/time slice

  Parameters:
    srcx    - source x coordinates
    srcy    - source y coordinates
    recx    - receiver x coordinatesq
    recy    - receiver y coordinates
    slc     - time or depth slice [ny,nx]
    ox      - slice x origin
    oy      - slice y origin
    dx      - slice x sampling [0.025]
    dy      - slice y sampling [0.025]
    recs    - plot only the receivers (toggles on/off the receivers)
    cmap    - 'grey' (colormap grey for image, jet for velocity)
    figname - output name for figure [None]
  """
  ny,nx = slc.shape
  cmap = kwargs.get('cmap','gray')
  fig = plt.figure(figsize=(14,7)); ax = fig.gca()
  ax.imshow(np.flipud(slc),cmap=cmap,extent=[ox,ox+nx*dx,oy,oy+ny*dy])
  if(srcs):
    ax.scatter(srcx,srcy,marker='*',color='tab:red')
  if(recs):
    ax.scatter(recx,recy,marker='v',color='tab:green')
  ax.set_xlabel('X (km)',fontsize=kwargs.get('fsize',15))
  ax.set_ylabel('Y (km)',fontsize=kwargs.get('fsize',15))
  ax.tick_params(labelsize=kwargs.get('fsize',15))
  if(figname is not None):
    plt.savefig(figname,dpi=150,transparent=True,bbox_inches='tight')
  if(kwargs.get('show',True)):
    plt.show()

def sum_extimgs(migdir,fout):
  """
  Sums partial extended images to form the full F3 image

  Parameters:
    migdir - directory containing migration images (string)
    fout   - output file that will contain the output image (string)

  Returns nothing
  """
  pyexec = "/sep/joseph29/anaconda3/envs/py37/bin/python"
  summer = "/homes/sep/joseph29/projects/resfoc/bench/f3/scripts/mig/MigSum.py"
  subprocess.Popen([pyexec,summer,"-migdir",migdir,"-fout",fout])
