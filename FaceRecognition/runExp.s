#!/bin/bash

#SBATCH --job-name=FaceRecog
#SBATCH -t72:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu
#SBATCH --output=out.faceRecog.%j

module load pytorch/python2.7/0.3.0_4
module load pytorch/python3.6/0.3.0_4
python ./faceRecog.py