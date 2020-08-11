import numpy as np
from oway.ompwrapper import ompwrap
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from dask.distributed import progress
import time

a = np.random.rand(100).astype('float32')
scale = 10

#ret = ompwrap(1.0,np.random.rand(100))

cluster = SLURMCluster(queue='twohour',cores=24,memory="30GB")
client = Client(cluster)

cluster.scale(jobs=10)

futures = client.map(ompwrap,np.arange(100))
#x = []
beg = time.time()
#for i in range(100):
#  x.append(client.submit(ompwrap,float(i)).result())

progress(futures)

res = client.gather(futures)
print(res[20])

#print(x[20])

print("%f seconds"%(time.time() - beg))

