"""
Useful functions for plotting. No interactive plots.
See utils.movie for interactive plotting
@author: Joseph Jennings
@version: 2020.03.24
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_wavelet(wav,dt,show=True,**kwargs):
  """
  Makes a plot for a wavelet

  Parameters
    wav  - input wavelet
    dt   - sampling rate of wavelet
    show - flag whether to display the plot or not
  """
  fig = plt.figure(figsize=(kwargs.get('wbox',15),kwargs.get('hbox',3)))
  ax = fig.gca()
  t = np.linspace(0.0,(wav.shape[0]-1)*dt,wav.shape[0])
  ax.plot(t,wav)
  ax.set_xlabel('Time (s)',fontsize=kwargs.get('labelsize',14))
  ax.tick_params(labelsize=kwargs.get('labelsize',14))
  maxval = np.max(wav)*1.5
  plt.ylim([-maxval,maxval])
  if(show):
    plt.show()

