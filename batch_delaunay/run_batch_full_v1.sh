#!/bin/bash
# Version 1: generate lists of connected cells via Delaunay triangulation for TLS separate from Neighborhood  Etc

batch_ver=1 #Version of batch_full_v(N).sh code
echo "Version"$batch_ver
prj="HCC-CBS-028-Pfizer-TBruno"
seg_type="v7.qc0.nr200um.cd20-1000.mincd20-15.ct-14"

#Raw data folder 1:
data_src1="/ix/rbao/Projects/"$prj"/results/proc_tma_tls/cell_seg/"$seg_type #Output of 0_.ipynb "Save new copy of segmentation sheets with paired down columns and TLS "

#top level processed data folder (note: batch_full creates run version subfolders):
data_dest="/ix/rbao/Projects/"$prj"/results/neighborhoods/"$seg_type

#Script location path:
script_path="/ix/rbao/Projects/"$prj"/scripts/python/tbruno_028/batch_delaunay"

#Batch out on each TMA:
n1=$(expr $(ls $data_src1 | wc -l) - 1)
echo $n1"+1 files in " $data_src1
bash $script_path"/batch_full_v"$batch_ver".sh" $data_src1 $data_dest \
    0 $n1 $script_path $prj
