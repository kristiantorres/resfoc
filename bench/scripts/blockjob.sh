#! /bin/tcsh
#SBATCH --job-name block5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=twohour
#SBATCH --time=02:00:00
#SBATCH --output=./log/vels/block_out.log
#SBATCH --error=./log/vels/block_err.log
#SBATCH --nodelist=maz052
cd $SLURM_SUBMIT_DIR
#
echo $SLURMD_NODENAME > block-node.txt
sleep 7100
#sleep 30
#
# End of script
