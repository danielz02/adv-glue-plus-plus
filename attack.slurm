for task in "mnli" "mnli-mm" "qnli" "qqp" "rte"
do
    for fix in 0
    do

    cat << EOF | sbatch
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50g
#SBATCH --output="logs/attack_${task}-${fix}.out"
#SBATCH --error="logs/attack_${task}-${fix}.err"
#SBATCH --cpus-per-task=12
#SBATCH --partition=compute
#SBATCH --job-name=${task}-${fix}
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
python alpaca/attack.py --bf16 --decode-adv --l1 --task ${task} --word-list ./static/TheBloke/stable-vicuna-13B-HF/word_list.npy --embedding-space ./static/TheBloke/stable-vicuna-13B-HF/s.npy --model TheBloke/stable-vicuna-13B-HF --fix-sentence ${fix}
#python alpaca/attack.py --bf16 --decode-adv --task ${task} --word-list ./static/TheBloke/stable-vicuna-13B-HF/word_list.npy --embedding-space ./static/TheBloke/stable-vicuna-13B-HF/s.npy --model TheBloke/stable-vicuna-13B-HF --fix-sentence ${fix}
EOF

    done

done