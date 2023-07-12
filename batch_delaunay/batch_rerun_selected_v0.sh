#!/bin/bash
module purge
module load pathml/1.0.3 

echo "IN BATCH SCRIPT CALLING SBATCH-->"
#batch_full_v0.sh <--Current file name
ver=0
# Version 0: generate lists of connected cells via Delaunay triangulation 

prj="HCC-CBS-028-Pfizer-TBruno"
base_path="/ix/rbao/Projects/"$prj"/data/TBruno_TMA_Phenotyping/Cell_segmentation"

#top level processed data folder:
data_dest="/ix/rbao/Projects/"$prj"/results/neighborhoods"

#Local Log destination folder:
log_dest=$data_dest"/logs/"
script_path="/ix/rbao/Projects/"$prj"/scripts/python/tbruno_028/batch_delaunay/"

data_dest_path=$data_dest"/v"$ver

echo "Beginning batch process... on select jobs"
#Raw data folder 1:
data_src1=$base_path"/TMA01/" #inForm cell segmentation .txt files
sbatch --array=133 \
    $script_path"delaunay_job_v"$ver".sh" \
    $data_src1 \
    $data_dest_path \
    $script_path \
    $log_dest \
    $prj
    
echo "Batch 1 sent!"

#Raw data folder 2:
data_src2=$base_path"/TMA02/" #inForm cell segmentation .txt files
sbatch --array=28,31,33 \
    $script_path"delaunay_job_v"$ver".sh" \
    $data_src2 \
    $data_dest_path \
    $script_path \
    $log_dest \
    $prj
    
echo "Batch 2 sent!"
