#!/bin/bash
#SBATCH --mem=200g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --output="logs/embedding.out"
#SBATCH --error="logs/embedding.err"
#SBATCH --cpus-per-task=12
#SBATCH --partition=
#SBATCH --account=
#SBATCH --job-name=Test
#SBATCH --time=48:00:00 # hh:mm:ss for the job
#SBATCH --gpus-per-node=4
#SBATCH --mail-type=end
#SBATCH --mail-type=fail


module purge # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
module load gcc
# module load anaconda3  # ... or any appropriate modules
module load openmpi

# source activate torch
ldd /usr/bin/ssh

conda activate ""
cd ""
ifconfig

NODE_LIST=$( scontrol show hostname "$SLURM_JOB_NODELIST" | sed -z 's/\n/\:4,/g' )
NODE_LIST=${NODE_LIST%?}
echo "Job is starting on $(hostname) with $SLURM_NTASKS tasks on $NODE_LIST"
mpiexec -np "$SLURM_NTASKS" python alpaca/context_embedding.py
