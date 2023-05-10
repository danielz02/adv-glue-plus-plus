#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=17
#SBATCH --mem=100g
#SBATCH --output="logs/embedding.out"
#SBATCH --error="logs/embedding.err"
#SBATCH --cpus-per-task=12
#SBATCH --partition=compute
#SBATCH --job-name=Test
#SBATCH --time=24:00:00 # hh:mm:ss for the job
#SBATCH --gpus-per-node=4
#SBATCH --mail-type=end
#SBATCH --mail-type=fail

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

RANKFILE="./.cache/rankfile.txt"
> $RANKFILE

RANK=0
NODE_INDEX=0
for NODE in $(scontrol show hostname "$SLURM_JOB_NODELIST"); do
    NUM_RANKS_PER_NODE=4

    if [ $NODE_INDEX -eq 0 ]; then
        NUM_RANKS_PER_NODE=5
    fi

    SLOT=0
    for SLOT in $(seq 0 $((NUM_RANKS_PER_NODE-1))); do
        echo "rank $RANK=$NODE slot=$SLOT" >> $RANKFILE
        RANK=$((RANK+1))
    done

    NODE_INDEX=$((NODE_INDEX+1))
done

echo "Job is starting on $(hostname) with $SLURM_NTASKS tasks on $NODE_LIST"
mpiexec --mca btl vader,self,tcp -np "$SLURM_NTASKS" --rankfile $RANKFILE --oversubscribe python alpaca/context_embedding.py

