#!/bin/bash

#SBATCH --time=00:10:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH -C 'pascal' # features syntax (use quotes): -C 'a&b&c&d'
#SBATCH --mem-per-cpu=10240M   # memory per CPU core
#SBATCH -J "egb-start-follow-read"   # job name
#SBATCH --mail-user=burdett1@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=test
#SBATCH --gid=

# Compatibility variables for PBS. Delete if not needed.
export PBS_NODEFILE=`/fslapps/fslutils/generate_pbs_nodefile`
export PBS_JOBID=$SLURM_JOB_ID
export PBS_O_WORKDIR="$SLURM_SUBMIT_DIR"
export PBS_QUEUE=batch

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# Load the Conda Environment
eval "$(/fslhome/burdett1/anaconda3/bin/conda shell.bash hook)"

conda activate sfr_env

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
# module load cuda/8.0
# module load cudnn/6.0_cuda-8.0
# module load python
# module load python-pytorch
# module load opencv/3/0

python run_hwr.py ../french_imgs/ sample_config_60.yaml ../egb-out-gpu/ 2>&1 &

python run_decode.py sample_config_60.yaml ../egb-out-gpu/ 2>&1 &

wait

