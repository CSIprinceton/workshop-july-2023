#!/bin/bash
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1         
#SBATCH --nodes=1         
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --job-name="Train2" 
#SBATCH --gres=gpu:1             # number of gpus per node

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

pwd; hostname; date

conda activate deepmd-2.1.3

dp compress -t input-compress.json -i frozen_model_2.pb -o frozen_model_2_compressed.pb

date
