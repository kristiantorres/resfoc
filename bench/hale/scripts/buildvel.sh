#! /bin/bash

set -x

# Preprocess
Window < ./dat/midpts.H > ./dat/mymidpts.H
echo "d2=0.067 o2=.132" >> ./dat/mymidpts.H
sfmutter v0=1.4 < ./dat/mymidpts.H > midmute.H

# Velocity analysis
sfvscan < midmute.H semblance=y v0=1.4 nv=51 dv=0.025 > myscn.H

sfmutter x0=1.5 v0=0.67 half=n < myscn.H > scnmute.H
sfpick rect1=40 rect2=15 < scnmute.H > myvelrms.H

# NMO and stack
sfnmo velocity=myvelrms.H < midmute.H > mynmo.H
