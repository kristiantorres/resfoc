"""
Keras callbacks for QCing training

@author: Joseph Jennings
@version: 2020.02.21
"""
import inpout.seppy as seppy
import h5py
from tensorflow.keras import callbacks
from deeplearn.dataloader import load_allflddata
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from deeplearn.utils import thresh, plotseglabel
from utils.ptyprint import create_inttag
import numpy as np
import matplotlib.pyplot as plt

class F3Pred(callbacks.Callback):

  def __init__(self,f3path,dsize,psize,ssize):
    # Load in the f3 data
    self.f3dat = load_allflddata(f3path,dsize)
    self.dsize = dsize
    # Create Patch extractor
    self.pe = PatchExtractor(psize,stride=ssize)
    # Dummy array to set dimensions
    self.dummy = np.zeros([512,1024]); dptch = self.pe.extract(self.dummy)
    # Plot one xline
    self.xlidx = 1; self.fs = 200; self.thresh = 0.5
    # Do it only every few epochs
    self.skip = 2

  def on_epoch_end(self,epoch,logs={}):
    if(epoch%self.skip == 0):
      # Make a prediction on the f3 data and save it
      print("Predicting F3 dataset...")
      pred = self.model.predict(self.f3dat,verbose=1)
      # Save predictions to file
      with h5py.File("/net/fantastic/scr2/joseph29/f3preds/ep%s.h5"%(create_inttag(epoch,100)),"w") as hf:
        hf.create_dataset("pred", self.f3dat.shape, data=pred, dtype=np.float32)
      # Reconstruct a single inline
      iimg = self.f3dat[self.xlidx*self.dsize:(self.xlidx+1)*self.dsize,:,:]
      iimg = iimg.reshape([7,15,128,128])
      rimg = self.pe.reconstruct(iimg)
      # Reconstruct the predictions
      ipred = pred[self.xlidx*self.dsize:(self.xlidx+1)*self.dsize,:,:]
      ipred = ipred.reshape([7,15,128,128])
      rpred = self.pe.reconstruct(ipred)
      # Apply threshold and plot
      tpred = thresh(rpred,self.thresh)
      nt = rimg.shape[0]; nx = rimg.shape[1]
      plotseglabel(rimg[self.fs:,:],tpred[self.fs:,:],color='blue',
          xlabel='Inline',ylabel='Time (s)',xmin=0.0,xmax=(nx-1)*25/1000.0,
          zmin=(self.fs-1)*0.004,zmax=(nt-1)*0.004,vmin=-2.5,vmax=2.5,interp='sinc',aratio=6.5)
      plt.savefig('./fig/f3preds/ep%s.png'%(create_inttag(epoch,100)),bbox_inches='tight',dpi=150)


