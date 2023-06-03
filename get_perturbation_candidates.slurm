#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100g
#SBATCH --output="logs/perturb_candidates.out"
#SBATCH --error="logs/perturb_candidates.err"
#SBATCH --cpus-per-task=64
#SBATCH --partition=compute
#SBATCH --job-name=GetFC
#SBATCH --time=48:00:00 # hh:mm:ss for the job
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

#pids=""
COUNTER=0
for task in "rte" # "sst2" "mnli" "mnli-mm" "qnli" "qqp" "rte"
do
    echo $COUNTER

    python alpaca/get_FC.py --task "${task}" --model TheBloke/stable-vicuna-13B-HF --word-list ./static/TheBloke/stable-vicuna-13B-HF/word_list.npy --embedding-space ./static/TheBloke/stable-vicuna-13B-HF/s.npy
    python alpaca/get_FT.py --task "${task}" --model TheBloke/stable-vicuna-13B-HF --word-list ./static/TheBloke/stable-vicuna-13B-HF/word_list.npy --embedding-space ./static/TheBloke/stable-vicuna-13B-HF/s.npy
    python alpaca/get_FK.py --task "${task}" --model TheBloke/stable-vicuna-13B-HF --word-list ./static/TheBloke/stable-vicuna-13B-HF/word_list.npy --embedding-space ./static/TheBloke/stable-vicuna-13B-HF/s.npy
#    pids="$pids $!"
    COUNTER=$((COUNTER+1))
done

#for pid in $pids; do
#    wait "$pid" || ((RESULT=1))
#done
#
#if [ "$RESULT" == "1" ];
#    then
#       exit 1
#fi
