#!/bin/bash -l

#######################################
# example for a OpenMP job #
#######################################

#SBATCH --job-name=stat_cloud_detect

# we ask for OPEN MPI tasks with 20 cores each
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=20
#SBATCH --time=08:00:00
#SBATCH --partition=nmipt

export FTDIR=/scratch/mtool/selvarajd/cache

# you may not place bash commands before the last SBATCH directive

# define and create a unique scratch directory
RUN_DIR=/home/gmap/mrmn/selvarajd/SAVE/python/SWD

SCRATCH_DIRECTORY=/scratch/work/selvarajd/python/SWD
cd ${SCRATCH_DIRECTORY}

# we copy everything we need to the scratch directory
# ${SLURM_SUBMIT_DIR} points to the path where this script was submitted from

CURRENT_DIR=scores_dble
cd ${CURRENT_DIR}
ls
# we set OMP_NUM_THREADS to the number of available cores
export OMP_NUM_THREADS=128

module load openmpi
module use ~mary/public/modulefiles
module load epygram
module load python/3.7.6

export NUMEXPR_MAX_THREADS=128
echo $NUMEXPR_MAX_THREADS

# we execute the job and time it
python statistiques_cloud_occurrence_parallel_eps_dble.py > statistiques_cloud_occurrence_parallel_eps_dble.txt

cd ${RUN_DIR}

