#!/bin/bash
# Version 0: generate lists of connected cells via Delaunay triangulation 
ver=0
prj="HCC-CBS-028-Pfizer-TBruno"
base_path="/ix/rbao/Projects/"$prj"/data/TBruno_TMA_Phenotyping/Cell_segmentation"
#Raw data folder 1:
data_src1=$base_path"/TMA01/" #inForm cell segmentation .txt files

#Raw data folder 2:
data_src2=$base_path"/TMA02/" #inForm cell segmentation .txt files

#top level processed data folder:
data_dest="/ix/rbao/Projects/"$prj"/results/neighborhoods"

#Local Log destination folder:
log_dest=$data_dest"/logs/"
script_path="/ix/rbao/Projects/"$prj"/scripts/python/tbruno_028/batch_delaunay/"

#Batch out on each TMA:
n1=$(expr $(ls $data_src1 | wc -l) - 1)
echo $n1"+1 files in " $data_src1
bash $script_path"/batch_full_v"$ver".sh" $data_src1 $data_dest \
    0 $n1 $script_path $log_dest $prj

n2=$(expr $(ls $data_src2 | wc -l) - 1)
echo $n2"+1 files in " $data_src2
bash $script_path"/batch_full_v"$ver".sh" $data_src2 $data_dest \
    0 $n2 $script_path $log_dest $prj 
