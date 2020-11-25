"""
Su2sep does not work properly and for some reason
for some files, does not convert all of the headers.
This happens to be only the headers related to time.
This script changes those headers in the .su files
(makes a dummy file) and then converts the dummy file
to the .H file.

Fortunately, it seems to only affect the time-related (hour,minute,second)
headers to this should not affect the imaging of the data.
Although, all time-related information for those files
is lost

@author: Joseph Jennings
@version: 2020.09.30
"""

import os,glob
import inpout.seppy as seppy
import subprocess
from genutils.ptyprint import progressbar

sep = seppy.sep()
seps = glob.glob("./sep/*.H")

# Data path for writing SEP3D files
rpath = "/net/brick5/data3/northsea_dutch_f3/"
dpath = rpath + "sep/"

nfiles = len(seps)

verb = True
for isep in progressbar(range(nfiles),"nfiles",verb=(not verb)):
  # Read in the .H@@ file
  hdict = sep.read_header_dict(seps[isep]+'@@')
  if(int(hdict['n1']) < 27):
    # Remove this file
    rm1 = 'Rm %s'%(seps[isep])
    if(verb): print(rm1)
    else: sp = subprocess.check_call(rm1,shell=True)
    # Get the corresponding su file
    bname = os.path.basename(seps[isep])
    name = os.path.splitext(bname)[0]
    sufile = './su/' + name + '.su'
    sushw = 'sushw key=hour,minute,sec a=20,20,20 < %s > dummy.su'%(sufile)
    if(verb): print(sushw)
    else: sp = subprocess.check_call(sushw,shell=True)
    # Convert to SEP again
    sephfile = name + '.H'
    sephhfile = dpath + name + '.H@@'
    su2sep = 'Su2sep < dummy.su 2>/dev/null > %s hff=%s datapath=%s'%(sephfile,sephhfile,dpath)
    if(verb): print(su2sep)
    else: sp = subprocess.check_call(su2sep,shell=True)
    sepmov = 'mv %s %s'%(sephfile,dpath)
    if(verb): print(sepmov)
    else: sp = subprocess.check_call(sepmov,shell=True)
    rm2 = 'rm dummy.su'
    if(verb): print(rm2)
    else: sp = subprocess.check_call(rm2,shell=True)
    if(verb): print()

