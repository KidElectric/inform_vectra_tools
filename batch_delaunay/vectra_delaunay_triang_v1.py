import time
import os
import sys
import pdb
import math
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Custom helper functions:
sys.path.append('/ihome/rbao/bri8/scripts/')
from helpers import tls
from helpers import delaunay as delHelpers
from helpers import delaunayPlots as delPlots
start = time.time()

#Begin the job
base = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
file_type = sys.argv[3]
jobid = sys.argv[4]
time_start_str = time.strftime('%Y-%m-%d %H:%M:%S%p', time.localtime(start))

print('Job log beginning at %s' % (time_start_str))
print("Base path: %s\n" % str(base))

files_avail=glob.glob(str(base.joinpath('*' + file_type)))
files_avail.sort()
if len(files_avail)>0:
    #For now, assume there is only one file moved for this job instance:
    fn=Path(files_avail[0])
else:
    print('No Files found.')
    
df = pd.read_csv(fn, sep='\t')

qct = 0
scale = 1
max_dist = 25
cell_names = ['CD20+',
              'CD4+',
              'CD8+',
              'CD68+',
              'FoxP3+',
              'PanCK+',
              'Tumor',
              'Immune Cell Cluster',
              'AF/Macrophage Cluster']
tissue_types = ['Tumor',
              'Immune Cell Cluster',
              'AF/Macrophage Cluster']

print('QC threshold: %d, scale_factor: %d, max neighbor dist: %d' 
        % (qct,scale,max_dist))

qc = tls.get_qc(df,qc_thresh=qct)


tls_idx = ~df['tls_id'].isna()
neighbor_idx = df.loc[:,'is_neighbor'].values
idxs = [tls_idx & qc,
        neighbor_idx & qc,
       (~tls_idx & ~neighbor_idx) & qc]

labs = ['tls', 'tlsneighbors', 'remaining_core']

keep_cx = []
fig = plt.figure(figsize=(18,11))
i = 1
start = time.time()
out_stack = []

for idx,lab in zip(idxs,labs):
    subset=df.loc[idx,:]

    print('Beginning %s cell connection detection...' % lab)
    connections = delHelpers.df_to_connections_output(subset,
                                                      scale = scale,
                                                      max_dist = max_dist,
                                                      )
    
    delcx = delHelpers.generate_log_odds_matrix(connections,
                                                 cell_names,
                                                 tissue_types,
                                                 version = 2)
       
    out_stack.append(delcx)
    ax = fig.add_subplot(2,len(labs),i,aspect='equal')
    ax = delPlots.connection_heatmap(delcx, 
                                     cell_names,
                                     ax = ax
                                    )
    ax.set_title(lab)
    
    if i > 1:
        ylabel=''
    else:
        ylabel = 'Spoke Cell'
    ax.set_ylabel(ylabel)
    new_fn =jobid + '_' \
            +  fn.parts[-1].split(']_')[0] \
            + ']_delaunay_cx_%s' % lab

    print('Saving %s' % new_fn )
    
    #Save full connectivity:
    out_fn = output.joinpath(new_fn + '.csv')
    connections.to_csv(out_fn)
    
    #Save stack out log-odds connectivity matrices
    # out_fn = output.joinpath(new_fn + '.npy')
    # np.save(out_fn, arr=delcx, allow_pickle=False)
    
    i = i + 1

# Plot difference of tls and neighborhood from rest of core:            
non_tls = out_stack[2]
for lab,delcx in zip(labs[0:2], out_stack[0:2]):
    ax = fig.add_subplot(2,len(labs),i,aspect='equal')         
    ax = delPlots.connection_heatmap(delcx-non_tls, 
                                     cell_names,
                                     ax = ax,
                                     vmin = -4,
                                     vmax = 4,
                                     label = 'log odds'
                                    )
    ax.set_title(lab + ' - remaining core')
    if i > 1:
        ylabel=''
    else:
        ylabel = 'Spoke Cell'
    ax.set_ylabel(ylabel)
    i = i + 1

# Lastly add TLS - Neighborhood:
ax = fig.add_subplot(2,len(labs),i,aspect='equal')
ax = delPlots.connection_heatmap(out_stack[0]-out_stack[1], 
                                 cell_names,
                                 vmin = -4,
                                 vmax = 4,
                                 ax = ax,
                                 label = 'log odds'
                                )
ax.set_title('TLS - Neighborhood')   
ax.set_ylabel('')
            
img_fn = jobid + '_' \
         +  fn.parts[-1].split(']_')[0] \
         +  ']_delaunay_cx_tls-neigh-remain.png'
print('Saving %s' % img_fn)
plt.savefig(output.joinpath(img_fn))
            
stop = time.time()
print('Processing time: %2.2f minutes.' % ((stop-start)/60))
print('Finished!')
