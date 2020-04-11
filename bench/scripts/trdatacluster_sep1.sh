#! /bin/bash

#PBS -N ptscatdata1
#PBS -l nodes=1:ppn=8
#PBS -q sep
#PBS -o /data/sep/joseph29/projects/resfoc/log/ptscatdatao1
#PBS -e /data/sep/joseph29/projects/resfoc/log/ptscatdatae1
cd $PBS_O_WORKDIR
#
/data/sep/joseph29/opt/anaconda3/envs/py35/bin/python ./scripts/trdata_ptscat_allsep.py -c ./par/ptscatdatasep1.par
#
# End of script
