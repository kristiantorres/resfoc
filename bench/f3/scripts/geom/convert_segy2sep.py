"""
Converts all of the F3 SEGY files to SEP3D files

@author: Joseph Jennings
@version: 2020.09.26
"""
import os
import glob
import subprocess
from genutils.ptyprint import progressbar

# Get all of the SEGY files
segys = glob.glob("./segy/*.segy")

# Data path for writing SEP3D files
rpath = "/net/brick5/data3/northsea_dutch_f3/"
dpath = rpath + "sep/"

# Path for SEP files
if(not os.path.exists("./su")):
  os.mkdir ("./su")

keepsu = True; verb = False
for isegy in progressbar(range(len(segys)),"nfiles:"):
  # Get the base name
  bname = os.path.basename(segys[isegy])
  fname = os.path.splitext(bname)[0]
  # Output file names
  sufile     = rpath + "su/" + fname + '.su'
  sephfile   = fname + '.H'
  sephhfile  = dpath + fname + '.H@@'
  # Convert to SU
  if(verb): print(suconv)
  suconv = "segyread tape=%s 2>/dev/null | segyclean > %s "%(segys[isegy],sufile)
  sp = subprocess.check_call(suconv,shell=True)
  # Convert to SEP
  sepconv = "Su2sep < %s 2>/dev/null > %s hff=%s datapath=%s"%(sufile,sephfile,sephhfile,dpath)
  sepmov  = "mv %s %s"%(sephfile,dpath)
  if(verb): print(sepconv)
  sp = subprocess.check_call(sepconv,shell=True)
  if(verb): print(sepmov)
  sp = subprocess.check_call(sepmov,shell=True)
  if(not keepsu):
    sp = subprocesss.check_call("rm %s"%(sufile))

sp = subprocess.check_call("rm binary header",shell=True)
if(not keepsu):
  sp = subprocess.check_call("rm -rf ./su/")
