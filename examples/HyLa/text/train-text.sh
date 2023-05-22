#!/bin/bash
#SBATCH -J hyla_text
#SBATCH -o hyla_text.o%j
#SBATCH -e hyla_text.o%j
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=5000
#SBATCH -t 12:00:00
#SBATCH --partition=gpu  --gres=gpu:1
       
python3 TextHyLa.py \
       -manifold PoincareBall \
       -model hyla \
       -dataset ohsumed \


       # -inductive