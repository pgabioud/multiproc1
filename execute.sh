#!/bin/bash
#SBATCH --workdir /home/your_user_name
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account cs307
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

length=50
iterations=1

module load gcc cuda

echo STARTING AT `date`
make all

./assignment4 $length $iterations
echo FINISHED at `date`
