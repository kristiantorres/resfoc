from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from dask.distributed import progress
import numpy as np
import time

def slow_increment(x,y,z,n=3):
  time.sleep(1)
  a = np.zeros(n)
  a[0] = x+1; a[1] = y+2; a[2] = z+3
  return a

cluster = SLURMCluster(queue='twohour',cores=24,memory="30GB")
client = Client(cluster)

cluster.scale(jobs=10)

#futures = client.map(slow_increment,range(5000))
futures = client.map(slow_increment,range(5000),range(5000),range(5000))
progress(futures)

res = client.gather(futures)

print(type(res))
