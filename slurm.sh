#!/bin/bash
# Author(s): James Owers (james.f.owers@gmail.com)
#
# example usage:
# ```
# EXPT_FILE=experiments.txt  # <- this has a command to run on each line
# NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
# MAX_PARALLEL_JOBS=3 
# sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} slurm_arrayjob.sh $EXPT_FILE
# ```
#
# or, equivalently and as intended, with provided `run_experiement`:
# ```
# run_experiment -b slurm_arrayjob.sh -e experiments.txt -m 12
# ```

# ====================
# Options for sbatch
# ====================

#SBATCH -J my_project

# Maximum number of nodes to use for the job
#SBATCH --nodes=1

# Organisation account
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=16000

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:1

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=16

# Maximum time for the job to run, format: days-hours:minutes:seconds
#SBATCH --time=01:00:00

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output="/home/%u/slurm_logs/%A_%a.out"
# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error="/home/%u/slurm_logs/%A_%a.err"

# 90 seconds before training ends send a SIGUSR1 signal to the process to requeue
# SBATCH --signal=SIGUSR1@90

# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

dest_path=/path/to/my/data
output_path=/path/to/my/output

# Activate your conda environment
CONDA_ENV_NAME=my_project
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

# ==============================
# Finally, run the experiment!
# ==============================
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

experiment_text_file=$1
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND} ++num_workers=${SLURM_CPUS_PER_TASK} ++paths.input_dir=${dest_path} ++paths.output_dir=${output_path}"
echo "Command ran successfully!"

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"