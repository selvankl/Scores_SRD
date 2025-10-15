#!/bin/bash -l
# -*- coding: utf-8 -*-

#SBATCH --job-name=transfer_pearome_grib_hendrix_to_cache
#SBATCH --time=06:00:00
#SBATCH --mem=3072
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=transfert

module use ~mary/public/modulefiles

module load epygram
module load python/3.7.6

SCRATCH_DIRECTORY=/scratch/work/selvarajd/python/coding
cd ${SCRATCH_DIRECTORY}

export MTOOL_STEP_CACHE=/scratch/mtool/selvarajd/cache
python transfer_pearome_750m_grib_hendrix_to_cache.py > transfer_pearome_750m_grib_hendrix_to_cache.txt 
