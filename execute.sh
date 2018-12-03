#!/bin/bash
#SBATCH --workdir /home/your_user_name
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1

length=50
iterations=1

module load gcc cuda

echo STARTING AT `date`
make all

./assignment4 $length $iterations
echo FINISHED at `date`
