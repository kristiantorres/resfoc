import inpout.seppy as seppy
import numpy as np
import segyio

sep = seppy.sep()

maxes,dat = sep.read_file("./dat/mymidpts.H")
[nt,nh,nm] = maxes.n; [ot,oh,om] = maxes.o; [dt,dh,dm] = maxes.d

dat = np.ascontiguousarray(dat.reshape(maxes.n,order='F').T)
datr = dat.reshape([nm*nh,nt])

mids = np.linspace(om,om+(nm-1)*dm,nm)
offs = np.linspace(oh,oh+(nh-1)*dh,nh)

midsall = np.zeros([nm,nh],dtype='float32')
offsall = np.zeros([nm,nh],dtype='float32')

for im in range(nm):
  midsall[im,:] = mids[im]
  offsall[im,:] = offs[:]

midsflt = midsall.flatten()
offsflt = offsall.flatten()

shtsall = 10000*(midsflt - offsflt)
recsall = 10000*(midsflt + offsflt)

print(np.min(shtsall),np.max(shtsall))
print(np.min(recsall),np.max(recsall))

spec = segyio.spec()
spec.samples = list(range(nt))
spec.ilines = list(range(nm*nh))
spec.xlines = [1]
spec.format = 1

print(shtsall[0],recsall[0])
with segyio.create("test.segy", spec) as dst:
  dst.trace = datr
  dst.header = { segyio.tracefield.TraceField.TRACE_SAMPLE_INTERVAL: 4000 }
  for itr in range(nh*nm):
    #print(shtsall[itr],recsall[itr])
    dst.header[itr] = { segyio.tracefield.TraceField.SourceX: shtsall[itr],
                        segyio.tracefield.TraceField.GroupX: recsall[itr]}
                        
