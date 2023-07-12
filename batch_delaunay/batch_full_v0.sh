#!/bin/bash
echo "IN BATCH SCRIPT CALLING SBATCH-->"
#batch_full_v0.sh <--Current file name
ver=0
module purge
module load pathml/1.0.3 

data_src_path=$1
data_dest_path=$2"/v"$ver
fstart=$3
fstop=$4
script_path=$5
log_path_base=$6
prj=$7

echo "Beginning batch process... on jobs" $fstart "to" $fstop

sbatch --array=$fstart-$fstop \
    $script_path"delaunay_job_v"$ver".sh" \
    $data_src_path \
    $data_dest_path \
    $script_path \
    $log_path_base \
    $prj
echo "Batch sent!"
