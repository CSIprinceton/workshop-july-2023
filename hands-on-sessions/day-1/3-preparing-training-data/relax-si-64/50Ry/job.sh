#!/bin/bash
#SBATCH --job-name=si-50         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4      # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                # total memory per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gpu-mps                # enable cuda multi-process service

module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --mpi=pmi2 \
singularity run --nv \
     /scratch/gpfs/ppiaggi/Simulations/QuantumEspressoGPU/quantum_espresso_qe-7.1.sif \
     pw.x -input pw-si.in -npool 2
