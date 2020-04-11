#! /bin/bash

#PBS -N trdatex
#PBS -l nodes=1:ppn=8
#PBS -q sep
#PBS -o /data/sep/joseph29/projects/resfoc/testout
#PBS -e /data/sep/joseph29/projects/resfoc/testerr
cd $PBS_O_WORKDIR
#
/data/sep/joseph29/opt/anaconda3/envs/py35/bin/python ./scripts/trdata_ptscat.py -c ./par/ptscatdata0.par
#
# End of script
