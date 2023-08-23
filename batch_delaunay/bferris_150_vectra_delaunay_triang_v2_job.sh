#!/bin/bash
#SBATCH --job-name=cell_triang
#SBATCH -N 1 #Ensure all cores on one machine
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00 # HH:MM:SS
#SBATCH --mem=32g

module load pathml/1.0.3 #has cv2
VER=2 #Round version
JOB_ID=$SLURM_ARRAY_TASK_ID
echo "SLURM IN JOB:" $JOB_ID
echo $SLURM_SCRATCH
PRJ='HCC-CBS-150-Hillman-BFerris-15132-HN-Vectra'
BASE='/ix/rbao/Projects/'$PRJ
DATA_SRC=$BASE'/bi_results/qupath/sica_gabe_annotations_bi_round_4/cell_exports'
DEST=$BASE'/bi_results/spatial/round_'$VER
SCRIPT_PATH=$BASE'/scripts'
FILE_TYPE=".txt"
JOB_SCRIPT=$SCRIPT_PATH"/inform_vectra_tools/batch_delaunay/bferris_150_vectra_delaunay_triang_v2.py"


#Run analysis on file in scratch and put outputs in scratch/proc:
echo "Beginning python preprocessing script:"
echo $JOB_SCRIPT

#If paths are all made relative to /mnt then using singularity image might be more useful:

# singularity exec -B /ix/rbao/Projects/$PRJ:/mnt \
#  /ix/rbao/images/pathml_jupyter.sif \
#  /opt/conda/envs/py38/bin/

python -u $JOB_SCRIPT \
    $DATA_SRC \
    $DEST \
    $FILE_TYPE \
    $JOB_ID \
    $SCRIPT_PATH
