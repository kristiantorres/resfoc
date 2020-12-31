import numpy as np
from seis.f3utils import select_f3shot
from genutils.plot import plot_dat2d

sx1,sy1 = 478203,6076288
sx2,sy2 = 478002,6076298
hdr1,dat1 = select_f3shot(sx1,sy1,allkeys=True)
hdr2,dat2 = select_f3shot(sx2,sy2,allkeys=True)

f1 = open('weird.txt','w')
for key in hdr1:
  f1.write('%s\n'%(key))
  f1.write(str(hdr1[key])+'\n')
f1.close()

f2 = open('normal.txt','w')
for key in hdr2:
  f2.write('%s\n'%(key))
  f2.write(str(hdr2[key])+'\n')
f2.close()

print(dat1.shape,dat2.shape)

plot_dat2d(dat1,aspect='auto',dt=0.002,pclip=0.02,title='Weird shot',show=False)
plot_dat2d(dat2,aspect='auto',dt=0.002,pclip=0.02,title='Normal shot',show=True)
