#! /bin/tcsh

#PBS -N test
#PBS -l mem=60gb,nodes=1:ppn=6,host=rcf130
#PBS -q default 
#PBS -o blockjob
#PBS -e blockjob 
cd $PBS_O_WORKDIR
#
sleep  7200
#
# End of script
