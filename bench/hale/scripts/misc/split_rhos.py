import numpy as np
from server.utils import splitnum

def rhos(nro,oro,dro):
  fnro = 2*nro - 1; foro = oro - (nro-1)*dro
  #print(fnro,foro)
  return np.linspace(foro,foro+dro*(fnro-1),fnro)

def chunks(l,sizes):
  i = beg = end = 0
  while i < len(sizes):
    end += sizes[i]
    yield l[beg:end]
    beg = end; i += 1

def force_odd(nums):
  onums = np.copy(nums)
  for i in range(len(onums)):
    if(onums[i]%2 == 0):
      onums[i] += 1
      if(i < len(onums)-1): onums[i+1] -= 1

  return onums

nro = 81; dro = 0.001250; oro = 1.0

# Split 81 into the chunks (40,41)
rhotot = rhos(nro,oro,dro)

sizes = force_odd(splitnum(len(rhotot),3))
chks = list(chunks(rhotot,sizes))

#TODO: force chunks to be odd
myros = []
beg = end = 0
for ichk in chks:
  foro = ichk[0]
  fnro = len(ichk)
  nro = (fnro + 1)//2
  oro = foro + (nro-1)*dro
  #print(nro,oro,dro)
  mycmp = rhos(nro,oro,dro)
  myros += list(mycmp)
  end += fnro
  print(beg,end)
  print(rhotot[beg:end])
  beg = end

#print(np.sum(np.asarray(myros)-rhotot))

