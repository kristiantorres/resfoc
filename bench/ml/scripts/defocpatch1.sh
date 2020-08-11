#! /bin/bash
#SBATCH --job-name defocpatch1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=sep
#SBATCH --time=02:00:00
#SBATCH --output=./log/defocptch1_out.log
#SBATCH --error./=log/defocptch1_err.log
cd $SLURM_SUBMIT_DIR
#
echo $SLURMD_NODENAME > defocpatch1-node.txt
/data/biondo/joseph29/opt/anaconda3/envs/py35/bin/python scripts/defocpatch_data.py -c ./par/defocpatch1.par
#
# End of script
