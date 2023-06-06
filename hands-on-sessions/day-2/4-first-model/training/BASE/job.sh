#!/bin/bash
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1         
#SBATCH --nodes=1         
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --job-name="TrainREPLACE" 
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --exclude=della-i14g[1-9],della-i14g1[1-9],della-i14g20

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

pwd; hostname; date

conda activate deepmd-2.1.3

dp train input.json

date
