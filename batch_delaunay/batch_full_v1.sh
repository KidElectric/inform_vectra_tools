#!/bin/bash
echo "IN BATCH SCRIPT CALLING SBATCH-->"
#batch_full_v1.sh <--Current file name
# Version 1: run 3 times per core: TLS, TLS Neighborhood, Non-TLS area of core.

job_ver=1 #delaunay_job_v(N).sh version to use and therefore results output location
run_num=2
module purge
module load pathml/1.0.3 

data_src_path=$1
echo "Data src: " $data_src_path
data_dest_path=$2"/v"$run_num
echo "Data destination: " $data_dest_path
fstart=$3
fstop=$4
script_path=$5
prj=$6

echo "Beginning batch process... on jobs" $fstart "to" $fstop

# sbatch --array=$fstart-$fstop \
#     $script_path"/delaunay_job_v"$job_ver".sh" \
#     $data_src_path \
#     $data_dest_path \
#     $script_path \
#     $prj

sbatch --array=29,238 \
    $script_path"/delaunay_job_v"$job_ver".sh" \
    $data_src_path \
    $data_dest_path \
    $script_path \
    $prj
    
echo "Batch sent!"
