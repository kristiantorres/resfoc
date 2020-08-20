import numpy as np
from multiprocessing import Pool,current_process

n = 300

# Create a 4D array
data = np.ones([24,n,n,n])

def mysum(i):
  print(current_process())
  chunk = data[i]
  return np.sum(chunk)

def chunkit(n):
  for i in range(n):
    yield i

with Pool() as pool:
  resit = pool.imap_unordered(mysum, chunkit(24))


