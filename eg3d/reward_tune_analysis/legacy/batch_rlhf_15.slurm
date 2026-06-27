#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Thu May 06 2021 19:54:03 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=gpu-a100

# Multithreaded (SMP) job: must run on one node
#SBATCH --nodes=1

# The project ID which this job should run under:
#SBATCH --account="punim1573"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


# Number of GPUs requested per node:
#SBATCH --gres=gpu:2

# The amount of memory in megabytes per process in the job:
#SBATCH --mem=60000

# Setting output files:
##### SBATCH -o /data/projects/punim1573/amoore/eg3d/eg3d/ffhq_slurm_test_04_07_2023.out

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0:30:00

#SBATCH --job-name eg3d_ffhq

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

cd "/data/projects/punim1573/amoore/eg3d_rlhf/";

# The job command(s):
source /usr/local/module/spartan_new.sh;
module load anaconda3/2021.11;
eval "$(conda shell.bash hook)";
conda activate eg3d_spartan;
module load fosscuda/2020b;

#srun python train.py --outdir=/data/projects/punim1573/amoore/eg3d/eg3d/training-runs --cfg=ffhq --data=/data/projects/punim1573/amoore/eg3d/eg3d_rebal_02_07_2023_10k_uniform_yaws.zip --gpus=2 --batch=32 --gamma=1 --gen_pose_cond=True;


srun python eg3d/train_rlhf.py --data=/data/projects/punim1573/amoore/eg3d/eg3d_rebal_02_07_2023_10k_uniform_yaws.zip --gpus=2 --batch=32 --gamma=1 --rlhf_config_fn=./eg3d_rlhf_hpms_dm3.yaml --gen_pose_cond=True  --neural_rendering_resolution_final=128 --outdir=/data/projects/punim1573/amoore/eg3d_rlhf/eg3d/training-runs --cfg=ffhq
