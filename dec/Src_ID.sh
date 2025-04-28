#!/bin/bash

#SBATCH --job-name=Src_Id            # Job name
#SBATCH --output=Src_Id_o%j.txt # Standard output log, %j will be replaced with job ID
#SBATCH --error=Src_Id_e%j.txt   # Standard error log, %j will be replaced with job ID
#SBATCH --ntasks=1                 # Run on a single CPU
#SBATCH --mem=10G                        # Memory limit
#SBATCH --time=10:00:00                 # Time limit hrs:min:sec
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1

nvidia-smi

python /scratch/hajorlou/D-VAE/dag_conv_nn/src_ID.py
