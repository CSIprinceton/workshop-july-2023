#!/bin/bash
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G       # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --job-name="si-1b-1700K" 
#SBATCH --gres=gpu:1             # number of gpus per node

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PLUMED_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH/usr/local/cudnn/cuda-11.3/8.2.0/lib64:"

pwd; hostname; date

module purge
module load anaconda3/2021.5
conda activate deepmd-2.1.3

############################################################################
# Variables definition
############################################################################
LAMMPS_EXE=lmp
############################################################################

############################################################################
# Run
############################################################################
srun $LAMMPS_EXE -in start.lmp

date
