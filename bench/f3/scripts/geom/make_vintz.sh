#! /bin/bash

sfdix rect1=10 rect2=1 rect3=1 < vels/migvelssm.H > vels/migvelint.H

sftime2depth velocity=migvelint.H intime=y nz=1000 z0=0 dz=0.01 < migvelint.H > miglintz.H
