#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50g
#SBATCH --output="logs/benign-zeroshot.out"
#SBATCH --error="logs/benign-zeroshot.err"
#SBATCH --cpus-per-task=12
#SBATCH --partition=compute
#SBATCH --job-name=benign-zeroshot
#SBATCH --time=12:00:00 # hh:mm:ss for the job
#SBATCH --gpus-per-node=1

module purge
source /opt/rh/devtoolset-10/enable
conda init bash
conda activate alpaca
module load mpi/openmpi/4.1.4-gcc

echo "$LD_LIBRARY_PATH"
which mpiexec
which python

cd /data/chenhui_zhang/SemAttack || exit
ifconfig

NODE_LIST=$( scontrol show hostname "$SLURM_JOB_NODELIST" | sed -z 's/\n/\:4,/g' )
NODE_LIST=${NODE_LIST%?}

echo "Job is starting on $(hostname) with $SLURM_NTASKS tasks on $NODE_LIST"
for model in "chavinlo/alpaca-native" "eachadea/legacy-vicuna-13b" "TheBloke/stable-vicuna-13B-HF"
do
    python attacks/model.py --model "${model}"
done