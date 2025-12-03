#!/bin/bash
#SBATCH --account=def-rmcintos
#SBATCH --mail-user=masoumeh_golmohamadian@sfu.ca
#SBATCH --mail-type=FAIL
#SBATCH --mem=8000MB
#SBATCH --time=0-1:00

# This script submits one or more simulations in a single job for PSE

### START MODIFYING HERE
### ====================================

# Location of your submission script directory
SUBMISSION_SCRIPT_DIR='/scratch/mgolmoha/TVB_Brown_Bag/tvb_study_group/Session_7-SBI/SBI-TVB'

# Load in env
module load scipy-stack
. ~/TVB/env/bin/activate

### STOP MODIFYING HERE
### ====================================


# Parameters from arguments
param_file=$1
subject=$2
num_sims_per_job=$3
time_per_sim=$4
weights_file_pattern=$5
results_file_pattern=$6
FCD_file_pattern=$7

# Calculate the starting and ending line based on the SLURM_ARRAY_TASK_ID
start_line=$(((SLURM_ARRAY_TASK_ID - 1) * num_sims_per_job + 1))
end_line=$((start_line + num_sims_per_job - 1))




# Read the parameters for the current job from start_line to end_line
sed -n "${start_line},${end_line}p" $param_file | while IFS=' ' read -r noise G dt sim_len; do
  python ${SUBMISSION_SCRIPT_DIR}/single_sim_handler.py $subject $noise $G $dt $sim_len $time_per_sim $SLURM_JOB_ID $SLURM_ARRAY_TASK_ID $weights_file_pattern $results_file_pattern $FCD_file_pattern
done
