from utils.signal import butter_bandpass_filter, ampspec1d
import numpy as np
import matplotlib.pyplot as plt

dt = 20.0
sig = np.zeros(256)

sig[128] = 1.0

sigflt = butter_bandpass_filter(sig,0.002,0.015,1/dt)
spec,fr = ampspec1d(sigflt,dt)

plt.plot(sigflt)
plt.show()

plt.plot(fr,spec); plt.show()

