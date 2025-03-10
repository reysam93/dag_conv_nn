#!/bin/bash

#SBATCH --job-name=Diffusion            # Job name
#SBATCH --output=Diffusion_o%j.txt # Standard output log, %j will be replaced with job ID
#SBATCH --error=Diffusion_e%j.txt   # Standard error log, %j will be replaced with job ID
#SBATCH --ntasks=1                 # Run on a single CPU
#SBATCH --mem=40G                        # Memory limit
#SBATCH --time=10:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodelist=bhg[0041,0042,0043]

nvidia-smi 

python /scratch/hajorlou/D-VAE/dag_conv_nn/Diffusion.py



