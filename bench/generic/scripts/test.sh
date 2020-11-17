#PBS -N velflt
#PBS -l nodes=1:ppn=16
#PBS -q sep
#PBS -o ./log/velflts/test.out
#PBS -e ./log/velflts/test.err
cd $PBS_O_WORKDIR
#
/data/sep/joseph29/opt/anaconda3/envs/py35/bin/python ./scripts/create_random_faultvel.py -c ./par/faultvel.par
#
# End of script
