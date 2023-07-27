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


scale = 1
max_dist = 35 #microns
tissue_crit = {}

col_dict = {'tissue_col': 'Parent',
            'cell_col': 'Class',
            'cell_x_pos' : 'Centroid X µm',
            'cell_y_pos' : 'Centroid Y µm'}

multi_label_types = {
     #PDL1-
     'PDL1-_tiv_inner': {'PD-L1': False,
                        'tissue':'Inner margin'}, 
     'PDL1-_tiv_outer': {'PD-L1': False,
                        'tissue':'Outer margin'},   
     'PDL1-_central_tumor': {'PD-L1': False,
                           'tissue': 'Center'}, 
     #PDL1+
     'PDL1+_tiv_inner': {'PD-L1': True,
                        'tissue':'Inner margin'}, 
     'PDL1+_tiv_outer': {'PD-L1': True,
                        'tissue':'Outer margin'},   
     'PDL1+_central_tumor': {'PD-L1': True,
                           'tissue': 'Center'},   
     # T-Cell CD8 CD3
     'CD8_CD3_tiv_inner': {'CD3':True, 'CD8': True,
                           'tissue':'Inner margin'},      
     'CD8_CD3_tiv_outer': {'CD3':True, 'CD8': True,
                           'tissue':'Outer margin'},   
     'CD8_CD3_central_tumor': {'CD3':True, 'CD8': True, 
                           'tissue': 'Center'},  
     #PD1+ T-cell CD8 CD3
     'PD1_CD8_CD3_tiv_inner': {'CD3':True, 'CD8': True, 'PD-1': True,
                               'tissue':'Inner margin'}, 
     'PD1_CD8_CD3_tiv_outer': {'CD3':True, 'CD8': True, 'PD-1': True,
                               'tissue':'Outer margin'},   
     'PD1_CD8_CD3_central_tumor': {'CD3':True, 'CD8': True, 'PD-1': True,
                               'tissue': 'Center'},  
     #Tregs:
     'Treg_tiv_inner': {'CD3':True, 'FOXP3':True,
                        'tissue':'Inner margin'}, 
     'Treg_tiv_outer': {'CD3':True, 'FOXP3':True,
                        'tissue':'Outer margin'},   
     'Treg_central_tumor': {'CD3':True, 'FOXP3':True,
                        'tissue': 'Center'},  
     #PD1+ Tregs:
     'PD1_Treg_tiv_inner': {'CD3':True, 'FOXP3':True, 'PD-1': True,
                            'tissue':'Inner margin'}, 
     'PD1_Treg_tiv_outer': {'CD3':True, 'FOXP3':True, 'PD-1': True,
                            'tissue':'Outer margin'},   
     'PD1_Treg_central_tumor': {'CD3':True, 'FOXP3':True, 'PD-1': True,
                            'tissue': 'Center'},
    }
df = vecUtils.vectra_if_types_to_cell_types(df.copy(),
                                 multi_label_dict= multi_label_types,
                                 col_dict= col_dict,
                                 verbose = True,
                                 type_adds_tissue = True,
                                 exclusive = False,
                                 output_cell_type_col = 'cell_type',
                                )
hub_cells = ['PDL1+_tiv_inner',
             'CD8_CD3_tiv_outer',
             'PD1_CD8_CD3_tiv_outer']

# Check any hub cells in df

tissue_types = ['Inner margin',
              'Outer margin',
               'Center',
               'Stroma']

print('scale_factor: %d, max neighbor dist: %d' 
        % (scale,max_dist))

inner_idx = df.loc[:,col_dict['tissue_col'].values == 'Inner margin'
outer_idx = df.loc[:,col_dict['tissue_col'].values == 'Outer margin'   
center_idx = df.loc[:,col_dict['tissue_col'].values == 'Center'
stroma_idx = df.loc[:,col_dict['tissue_col'].values == 'Stroma'
idxs = [center_idx,
        inner_idx & outer_idx,
        stroma_idx
       ]

labs = ['tumor_center','full_margin', 'stroma']

keep_cx = []
fig = plt.figure(figsize=(18,11))
i = 1
start = time.time()
out_stack = []
idx = df.loc[:,col_dict['cell_col']].isin(cell_names)
                   
for idx,lab in zip(idxs,labs):
    subset=df.loc[idx,:]
    print('Beginning %s cell connection detection...' % lab)
    
    # if evaluating tumor margin, require minimum number of stromal cells of any kind
    # be present--> delHelpers.cell_dist_criterion()
    connections = delHelpers.df_to_connections_output(subset,
                                                      scale = scale,
                                                      max_dist = max_dist,
                                                      col_dict = col_dict
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
    out_fn = output.joinpath(new_fn + '.npy')
    np.save(out_fn, arr=delcx, allow_pickle=False)
    
    i = i + 1

# # Plot difference of tls and neighborhood from rest of core:            
# non_tls = out_stack[2]
# for lab,delcx in zip(labs[0:2], out_stack[0:2]):
#     ax = fig.add_subplot(2,len(labs),i,aspect='equal')         
#     ax = delPlots.connection_heatmap(delcx-non_tls, 
#                                      cell_names,
#                                      ax = ax,
#                                      vmin = -4,
#                                      vmax = 4,
#                                      label = 'log odds'
#                                     )
#     ax.set_title(lab + ' - remaining core')
#     if i > 1:
#         ylabel=''
#     else:
#         ylabel = 'Spoke Cell'
#     ax.set_ylabel(ylabel)
#     i = i + 1

# # Lastly add TLS - Neighborhood:
# ax = fig.add_subplot(2,len(labs),i,aspect='equal')
# ax = delPlots.connection_heatmap(out_stack[0]-out_stack[1], 
#                                  cell_names,
#                                  vmin = -4,
#                                  vmax = 4,
#                                  ax = ax,
#                                  label = 'log odds'
#                                 )
# ax.set_title('TLS - Neighborhood')   
# ax.set_ylabel('')
            
img_fn = jobid + '_' \
         +  fn.parts[-1].split(']_')[0] \
         +  ']_delaunay_cx_tls-neigh-remain.png'
print('Saving %s' % img_fn)
plt.savefig(output.joinpath(img_fn))
            
stop = time.time()
print('Processing time: %2.2f minutes.' % ((stop-start)/60))
print('Finished!')
