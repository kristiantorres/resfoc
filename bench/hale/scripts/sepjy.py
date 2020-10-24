import os,sys,socket,datetime
from edu.mines.jtk.dsp import *
from edu.mines.jtk.io import *
from edu.mines.jtk.util.ArrayMath import *

def readFile(fileIn):
  """ Reads in a SEPlib file """
  # Get the axis information
  hdict,smps = readHeader(fileIn)
  # Get the input stream
  ais = ArrayInputStream(hdict['in'])
  # Allocate the array
  if(len(smps) == 2):
    data = zerofloat(smps[0].getCount(),smps[1].getCount())
  elif(len(smps) == 3):
    data = zerofloat(smps[0].getCount(),smps[1].getCount(),smps[2].getCount())
  else:
    raise Exception("Can only handle up to 3D inputs")
  ais.readFloats(data)
  ais.close()

  return smps,data

def writeFile(fileOut,data,smps=None):
  """ Writes a SEPlib file """
  opath = writeHeader(fileOut,smps)
  aos = ArrayOutputStream(opath)
  aos.writeFloats(data)
  aos.close()

def readHeader(fileIn,hdict=True):
  """ Reads in SEPlib header file """
  hdict = {}
  k = 0
  for line in open(fileIn).readlines():
    splitspace = line.split(' ')
    for item in splitspace:
      spliteq = item.split('=')
      if(len(spliteq) == 1): continue
      spliteq[0] = spliteq[0].replace('\n','')
      spliteq[0] = spliteq[0].replace('\t','')
      spliteq[1] = spliteq[1].replace('\n','')
      spliteq[1] = spliteq[1].replace('\t','')
      hdict[spliteq[0]] = spliteq[1]
  # Check if it found a binary
  assert("in" in hdict), "Error: header in file %s does not have an associated binary."%(fileIn)
  hdict["in"] = hdict["in"].replace('"','')
  # Read the header info into a list of axes
  ns = []; os = []; ds = []
  for n in range(1,7):
    nkey   = "n" + str(n)
    okey   = "o" + str(n)
    dkey   = "d" + str(n)
    lblkey = "label" + str(n)
    if n == 1:
      assert (nkey in hdict), "Error: header in file %s has no n1."%(hin)
    if nkey in hdict and okey in hdict and dkey in hdict:
      ns.append(int(hdict[nkey]))
      os.append(float(hdict[okey]))
      ds.append(float(hdict[dkey]))

  # Remove ones at the end
  for n in ns:
    if(ns[-1] == 1.0 and len(ns) != 1):
      del ns[-1]; del os[-1]; del ds[-1]

  # Take care of the remaining
  if(ns[-1] == 1.0 and len(ns) != 1):
    del ns[-1]; del os[-1]; del ds[-1]

  # Make a list of Sampling objects
  smps = []
  for iax in range(len(ns)):
    smps.append(Sampling(ns[iax],ds[iax],os[iax]))

  if(hdict):
    return hdict,smps
  else:
    return smps

def writeHeader(fileOut,smps=None):
  """ Writes out a SEPlib file """
  fout = open(fileOut,"w+")
  # Write the first line
  fout.write('\n' + get_fline()+'\n')
  # Get the datapath
  if(len(fileOut.split('/')) > 1):
    fileOut = fileOut.split('/')[-1]
  opath = get_datapath() + fileOut + "@"
  fout.write('\t\tsets next: in="%s"\n'%(opath))
  if(smps is not None):
    # Print axes
    ndims = len(smps)
    for k in range(ndims):
      fout.write("\t\tn%d=%d o%d=%f d%d=%.12f\n"%
          (k+1,smps[k].getCount(),k+1,smps[k].getFirst(),k+1,smps[k].getDelta()))
  else:
    ndim = len(smps)
    os = np.zeros(ndim)
    ds = np.ones(ndim)
    for k in range(ndim):
      fout.write("\t\tn%d=%d o%d=%f d%d=%.12f\n"%
          (k+1,ns[k],k+1,os[k],k+1,ds[k]))

  esize = 4
  fout.write('\t\tdata_format="xdr_float" esize=%d\n'%(esize))
  fout.close()

  return opath

def get_fline():
  """ Returns the first line of the program header """
  fline = sys.argv[0]
  # Get user and hostname
  username = 'joseph29'
  fline += ":\t" + username + "@" + gethostname() + "\t\t"
  # Get time and date
  time = datetime.datetime.today()
  fline += time.strftime("%a %b %d %H:%M:%S %Y")

  return fline

def get_datapath():
  """ Gets the set datpath for writing SEP binaries """
  dpath = None
  # Look in home directory
  datstring = os.environ['HOME'] + "/.datapath"
  if(os.path.exists(datstring) == True):
    nohost = ''
    # Assumes structure as host datapath=path
    for line in open(datstring).readlines():
      hostpath = line.split()
      if(len(hostpath) == 1):
        nohost = hostpath
      elif(len(hostpath) == 2):
        # Check if hostname matches
        if(gethostname() == hostpath[0]):
          dpath = hostpath[1].split('=')[1]
          break
    if(dpath == None and nohost != None):
      dpath = nohost[0].split('=')[1]
  # Lastly, look at environment variable
  elif(dpath == None and "DATAPATH" in os.environ):
    dpath = os.environ['DATAPATH']
  #else:
  #  dpath = '/tmp/'

  return dpath

def gethostname(alias=True):
  """
  An extension of socket.gethostname() where the hostname alias
  is returned if requested. This makes this code more
  compatible with the IO in SEPlib
  """
  hname = socket.gethostname()
  if(alias):
    f = open('/etc/hosts','r')
    for line in f.readlines():
      cols = line.lower().split()
      if(len(cols) > 1):
        if(hname == cols[1] and len(cols) > 2):
          hname = cols[2]
    f.close()

  return hname

