import inpout.seppy as seppy
import subprocess

sep = seppy.sep()

faxes = sep.read_header("sigsbee_foctrimgs.H")
raxes = sep.read_header("sigsbee_restrimgs.H")
haxes = sep.read_header("sigsbee_randrhos.H")
daxes = sep.read_header("sigsbee_deftrimgs.H")
nf = faxes.n[-1]; nr = raxes.n[-1]; nd = daxes.n[-1]; nh = haxes.n[-1]
if(nr != nf and nr != nh):
  raise Exception("Something is odd. nr should = nf...")

fdict = sep.read_header_dict("sigsbee_foctrimgs.H")
rdict = sep.read_header_dict("sigsbee_restrimgs.H")
hdict = sep.read_header_dict("sigsbee_randrhos.H")

fbin = fdict['in'][1:-1]
rbin = rdict['in'][1:-1]
hbin = hdict['in'][1:-1]

windfoc = "Window3d n5=%d < sigsbee_foctrimgs.H > sigsbee_foctrimgswind.H"%(nd)
windres = "Window3d n5=%d < sigsbee_restrimgs.H > sigsbee_restrimgswind.H"%(nd)
windrho = "Window n1=%d < sigsbee_randrhos.H > sigsbee_randrhoswind.H"%(nd)

print(windfoc)
#sp = subprocess.check_call(windfoc,shell=True)
print(windres)
#sp = subprocess.check_call(windres,shell=True)
print(windrho)
#sp = subprocess.check_call(windrho,shell=True)

fwdict = sep.read_header_dict("sigsbee_foctrimgswind.H")
rwdict = sep.read_header_dict("sigsbee_restrimgswind.H")
hwdict = sep.read_header_dict("sigsbee_randrhoswind.H")

fwbin = fwdict['in'][1:-1]
rwbin = rwdict['in'][1:-1]
hwbin = hwdict['in'][1:-1]

mvfoc = "mv %s %s"%(fwbin,fbin)
mvres = "mv %s %s"%(rwbin,rbin)
mvrho = "mv %s %s"%(hwbin,hbin)

print(mvfoc)
#sp = subprocess.check_call(mvfoc,shell=True)
print(mvres)
#sp = subprocess.check_call(mvres,shell=True)
print(mvrho)
#sp = subprocess.check_call(mvrho,shell=True)

#names = ['sigsbee_foctrimgs.H','sigsbee_restrimgs.H', 'sigsbee_randrhos.H']
#for name in names:
#  f = open(name,'r')
#  flines = f.readlines()
#  f.close()
#
#  f = open(name,'w')
#  for line in flines[:-1]:
#    f.write(line)
#  f.close()

clean = 'rm sigsbee_foctrimgswind.H sigsbee_restrimgswind.H sigsbee_randrhoswind.H'
print(clean)
#sp = subprocess.check_call(clean,shell=True)

