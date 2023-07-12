import os
import glob
import sys
from pathlib import Path
from shutil import copyfile
# slurm_job_move_file.py <--Current file name

src = Path(sys.argv[1]) #For example /ix/etc/etc/ or /scratch/slurm-
dest = Path(sys.argv[2]) #I.e. /scratch/slurm- or /ix/etc/etc
jobid = int(sys.argv[3])
file_type= sys.argv[4] 

all_files=glob.glob(str(src.joinpath("*" + file_type)))
all_files.sort()
if "scratch" in str(src): #will assume there is one file to move from scratch of this job
    fn = Path(all_files[0])
    move_direction = "scratch_to_ix"
else: #Will assume there are many src files that must be chosen from using jobid
    fn = Path(all_files[jobid])
    move_direction = "ix_to_scratch"
print("Moving %s" % move_direction)

if dest.exists() == False:
    print("Make folder %s" % dest)
    os.makedirs(dest)

job_str = "_job_%d%s" % (jobid,file_type)
if move_direction == "ix_to_scratch":
    #Add jobid to filename:
    new_fn=fn.parts[-1].split('.')[0] + job_str
else: #Assume this is a transfer from /scratch to /ix
    new_fn=fn.parts[-1] #Keep the same

new_dest=dest.joinpath(new_fn)
print("Copy %s to %s, for job %d" % (fn,new_dest,jobid))
copyfile(fn,new_dest)
print("Move complete.")
