#!/bin/bash
#SBATCH --job-name=cell_triang
#SBATCH -N 1 #Ensure all cores on one machine
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00 # HH:MM:SS
#SBATCH --mem=32g

#delaunay_job_v0.job <--Current file name
ver=0
cd $SLURM_SCRATCH
jobid=$SLURM_ARRAY_TASK_ID

echo "SLURM IN JOB:" $jobid 
echo $SLURM_SCRATCH

data_src_path=$1
data_dest_path=$2
script_path=$3
log_path_base=$4
prj=$5
proc_ver=0
date_str=$(date "+%Y-%m-%d_%H-%M-%S")
log_path=$log_path_base"job_"$SLURM_ARRAY_JOB_ID"_"$jobid".txt"
scratch_path=$SLURM_SCRATCH/
input_file_type=".txt"

#Move file selected by jobid from data source path to scratch path
python -u $script_path/slurm_job_move_file.py \
    $data_src_path $scratch_path \
    $jobid $input_file_type

#Run analysis on file in scratch and put outputs in scratch/proc:
echo "Beginning python preprocessing script:"
python -u $script_path"vectra_delaunay_triang_v"$proc_ver".py" \
    $scratch_path \
    $data_dest_path \
    $input_file_type \
    $jobid

#python -u $script_path"test.py" \
#    $scratch_path $data_dest_path $input_file_type

