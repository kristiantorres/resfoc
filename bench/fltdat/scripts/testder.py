import numpy as np
import matplotlib.pyplot as plt

stp = np.zeros(100)
der = np.zeros(100)
stp[49:] = 1.0

for i in range(0,100):
  if(i == 99):
    der[i] = stp[i] - stp[i-1]
  else:
    der[i] = stp[i+1] - stp[i]


plt.figure()
plt.plot(stp)
plt.figure()
plt.plot(der)
plt.show()
