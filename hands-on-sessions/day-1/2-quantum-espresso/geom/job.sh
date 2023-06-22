#!/bin/bash
#SBATCH --job-name=si            # Create a short name for your job
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=4      # Number of tasks per node
#SBATCH --cpus-per-task=1        # Number of CPU cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # Total memory per node
#SBATCH --gres=gpu:1             # Number of GPUs per node
#SBATCH --time=00:15:00          # Total run time limit (HH:MM:SS)
##SBATCH --gpu-mps                # Enable CUDA multi-process service

module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --mpi=pmi2 \
singularity run --nv \
     /scratch/gpfs/taehunl/program_della/qe_gpu/quantum_espresso_qe-7.0.sif \
     pw.x -input pw-si-vc_relax.in > pw-si-vc_relax.out

