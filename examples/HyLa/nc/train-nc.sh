#!/bin/bash
#SBATCH -J hyla_nc
#SBATCH -o hyla_nc.o%j
#SBATCH -e hyla_nc.o%j
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=5000
#SBATCH -t 12:00:00
#SBATCH --partition=gpu  --gres=gpu:1


       # "he_dim": 16,
       # "hyla_dim": 250,
       # "order": 2,
       # "lambda_scale": 0.1,
       # "lr_e": 0.1, 
       # "lr_c": 0.01,
       # "epochs": 100,
       # "quiet": false
       
# python3 HyLa.py \
#        -manifold PoincareBall \
#        -model hyla \
#        -he_dim 16 \
#        -hyla_dim 250 \
#        -lr_e 0.1 \
#        -lr_c 0.01 \
#        -lambda_scale 0.1 \
#        -dataset reddit \
#        -use_feats -tuned -inductive


# python3 HyLa.py \
#        -manifold PoincareBall \
#        -model hyla \
#        -he_dim 16 \
#        -hyla_dim 1000 \
#        -order 2 \
#        -lr_e 0.5 \
#        -lr_c 0.1 \
#        -lambda_scale 0.01 \
#        -dataset airport \
#        -tuned 

#  "he_dim": 16,
#      "hyla_dim": 1000,
#      "order": 2,
#      "lambda_scale": 0.01,
#      "lr_e": 0.5, 
#      "lr_c": 0.1,
# 

python3 HyLa.py \
       -manifold PoincareBall \
       -model hyla \
       -he_dim 16 \
       -hyla_dim 100 \
       -order 2 \
       -lr_e 0.05 \
       -lr_c 0.01 \
       -lambda_scale 0.01 \
       -dataset cora \
       -use_feats -tuned 



# remove -use_feats option for airport, add -inductive option for inductive training on reddit