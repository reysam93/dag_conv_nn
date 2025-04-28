#!/bin/bash

#SBATCH --job-name=Thames            # Job name
#SBATCH --output=Thames_o%j.txt # Standard output log, %j will be replaced with job ID
#SBATCH --error=Thames_e%j.txt   # Standard error log, %j will be replaced with job ID
#SBATCH --ntasks=1                 # Run on a single CPU
#SBATCH --mem=40G                        # Memory limit
#SBATCH --time=10:00:00                 # Time limit hrs:min:sec
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1

nvidia-smi


python /scratch/hajorlou/D-VAE/dag_conv_nn/Thames.py



