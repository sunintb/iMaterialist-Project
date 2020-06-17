#!/bin/bash -l
#PBS -N GpuTest
#PBS -q gpuq
#PBS -l nodes=4:ppn=8
#PBS -l gpus=1
#PBS -l feature=gpu
#PBS -A NCCC
#PBS -l walltime=02:00:00
#PBS -M david.chen.gr@dartmouth.edu
#PBS -m ea
#PBS -j oe

# Change to the directory that the job was submitted from
cd $PBS_O_WORKDIR

# Begin virtual environment created
source activate kaggle

# Parse the PBS_GPUFILE to determine which GPU is assigned & unset CUDA_VISIBLE_DEVICES
gpuNum=`cat $PBS_GPUFILE | sed -e 's/.*-gpu//g'`
unset CUDA_VISIBLE_DEVICES

CUDA_DEVICE=$gpuNum

# Pass the GPU number as an argument to your program
python3 main.py $gpuNum

# Upon completion:
source deactivate kaggle
exit 0
