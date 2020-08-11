import glob

ofiles = sorted(glob.glob("./dat/resdefoc/reso*.H"))
afiles = sorted(glob.glob("./dat/resdefoc/resa*.H"))

odict = {}
for iafile in afiles:
  with open(iafile,'r') as f:
    fnum = iafile.split('./dat/resdefoc/resa-')[1].split('.H')[0]
    for line in f.readlines():
      lsplit = line.split()
      if(len(lsplit) > 0):
        entry = lsplit[0]
        if(entry.split('=')[0] == 'rho'):
          odict[fnum] = float(entry.split('=')[1])

print(odict)
keys = list(odict.keys()); k = 0
with open('./dat/resdefoc/rhos1.txt','w') as f:
  for key in keys:
    f.write('./dat/resdefoc/resa-%s.H %f\n'%(key,odict[key]))

