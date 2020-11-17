#PBS -N velflt
#PBS -l nodes=1:ppn=16
#PBS -q sep
#PBS -o ./log/velflts/test.out
#PBS -e ./log/velflts/test.err
cd $PBS_O_WORKDIR
#
sleep 30
echo "Success!"
#
# End of script
