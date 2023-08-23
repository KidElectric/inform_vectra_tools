import time
import os
import sys
import math
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

start = time.time()
debug = False #Set false for full run
#Begin the job
base = Path(sys.argv[1])
print('Data source: %s\n' % str(base))
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
print('Data dest: %s\n' % str(output))
file_type = sys.argv[3]
jobid = int(sys.argv[4])
scripts = Path(sys.argv[5])
print('Scripts src: %s \n' % scripts)
#Custom helper functions:
sys.path.append(str(scripts.joinpath('inform_vectra_tools')))
from inform_vectra_tools import vecutils as vecutils
from inform_vectra_tools import vectra as vectra
from delaunay_tools import delaunay as delHelpers
from delaunay_tools import delaunayPlots as delPlots

time_start_str = time.strftime('%Y-%m-%d %H:%M:%S%p', time.localtime(start))
print('Job log beginning at %s' % (time_start_str))
print("Base path: %s\n" % str(base))

files_avail=glob.glob(str(base.joinpath('*' + file_type)))
files_avail.sort()
if len(files_avail)>0:
    #For now, assume there is only one file moved for this job instance:
    fn=Path(files_avail[jobid])
    tissue_id = fn.parts[-1].split('_')[0]
else:
    print('No Files found.')
print('Loading %s' % fn.parts[-1])    
df = pd.read_csv(fn, sep='\t',index_col=1)
scale = 1
max_dist = 35 #microns

col_dict = {'tissue_col': 'Parent',
            'cell_col': 'Class',
            'cell_x_pos' : 'Centroid X µm',
            'cell_y_pos' : 'Centroid Y µm',
            'output_cell_col': 'cell_type'}

# Cell type definitions file:
# Ideally make into .json or yaml ?
# multi_label_types = {
#      #PDL1-
#      'PDL1neg_tiv_inner': {'PD-L1': False,
#                         'tissue':'Inner margin'}, 
#      'PDL1neg_tiv_outer': {'PD-L1': False,
#                         'tissue':'Outer margin'},   
#      'PDL1neg_central_tumor': {'PD-L1': False,
#                            'tissue': 'Center tumor'}, 
#      #PDL1+
#      'PDL1_tiv_inner': {'PD-L1': True,
#                         'tissue':'Inner margin'}, 
#      'PDL1_tiv_outer': {'PD-L1': True,
#                         'tissue':'Outer margin'},   
#      'PDL1_central_tumor': {'PD-L1': True,
#                            'tissue': 'Center tumor'},   
#      # T-Cell CD8 CD3
#      'CD8_CD3_tiv_inner': {'CD3':True, 'CD8': True,
#                            'tissue':'Inner margin'},      
#      'CD8_CD3_tiv_outer': {'CD3':True, 'CD8': True,
#                            'tissue':'Outer margin'},   
#      'CD8_CD3_central_tumor': {'CD3':True, 'CD8': True, 
#                            'tissue': 'Center tumor'},  
#     # "CD4" cell
#      'CD4_tiv_inner' : {'CD3':True, 'CD8': False, 'FOXP3':False,
#                                   'tissue':'Inner margin'},  
#      'CD4_tiv_outer' : {'CD3':True, 'CD8': False, 'FOXP3':False,
#                                   'tissue':'Outer margin'},  
#      'CD4_central_tumor' : {'CD3':True, 'CD8': False, 'FOXP3':False,
#                                   'tissue':'Center tumor'},  
#      #PD1+ T-cell CD8 CD3
#      'PD1_CD8_CD3_tiv_inner': {'CD3':True, 'CD8': True, 'PD-1': True,
#                                'tissue':'Inner margin'}, 
#      'PD1_CD8_CD3_tiv_outer': {'CD3':True, 'CD8': True, 'PD-1': True,
#                                'tissue':'Outer margin'},   
#      'PD1_CD8_CD3_central_tumor': {'CD3':True, 'CD8': True, 'PD-1': True,
#                                'tissue': 'Center tumor'},  
#      #Tregs:
#      'Treg_tiv_inner': {'CD3':True, 'FOXP3':True,
#                         'tissue':'Inner margin'}, 
#      'Treg_tiv_outer': {'CD3':True, 'FOXP3':True,
#                         'tissue':'Outer margin'},   
#      'Treg_central_tumor': {'CD3':True, 'FOXP3':True,
#                         'tissue': 'Center tumor'},  
#      #PD1+ Tregs:
#      'PD1_Treg_tiv_inner': {'CD3':True, 'FOXP3':True, 'PD-1': True,
#                             'tissue':'Inner margin'}, 
#      'PD1_Treg_tiv_outer': {'CD3':True, 'FOXP3':True, 'PD-1': True,
#                             'tissue':'Outer margin'},   
#      'PD1_Treg_central_tumor': {'CD3':True, 'FOXP3':True, 'PD-1': True,
#                             'tissue': 'Center tumor'},
    
#     #PDL1- PanCK- PDL1- as estimate of macrophages:
#     'PDL1neg_macro_tiv_inner': {'CD3':False, 'PanCK': False, 'PD-L1': False,
#                         'tissue':'Inner margin'}, 
#     'PDL1neg_macro_tiv_outer': {'CD3':False, 'PanCK': False, 'PD-L1': False,
#                         'tissue':'Outer margin'},   
#     'PDL1neg_macro_central_tumor': {'CD3':False, 'PanCK': False, 'PD-L1': False,
#                         'tissue': 'Center tumor'},
    
#      #PDL1+ CD3- PanCK- as estimate of macrophages:
#     'PDL1_macro_tiv_inner': {'CD3':False, 'PanCK': False, 'PD-L1': True,
#                         'tissue':'Inner margin'}, 
#     'PDL1_macro_tiv_outer': {'CD3':False, 'PanCK': False, 'PD-L1': True,
#                         'tissue':'Outer margin'},   
#     'PDL1_macro_central_tumor': {'CD3':False, 'PanCK': False, 'PD-L1': True,
#                         'tissue': 'Center tumor'},
#     }
#The order matters! -> the more inclusive types must be defined before the more specific types 
# i.e. more specific cells will be removed from broader (single marker) categories

multi_label_types = {
     #PDL1-
     'PDL1neg_tiv_inner': {'PD-L1': False,
                        'tissue':'Inner margin'}, 

     #PDL1+
     'PDL1_tiv_inner': {'PD-L1': True,
                        'tissue':'Inner margin'}, 
   
     # T-Cell CD8 CD3
     'CD8_CD3_tiv_inner': {'CD3':True, 'CD8': True,#'PD-L1':False,
                           'tissue':'Inner margin'},      
     'CD8_CD3_tiv_outer': {'CD3':True, 'CD8': True,#'PD-L1':False,
                           'tissue':'Outer margin'},   

    # "CD4" cell
     'CD4_tiv_inner' : {'CD3':True, 'CD8': False, 'FOXP3':False,#'PD-L1':False,
                                  'tissue':'Inner margin'},  
     'CD4_tiv_outer' : {'CD3':True, 'CD8': False, 'FOXP3':False,#'PD-L1':False,
                                  'tissue':'Outer margin'},  
 
     #PD1+ T-cell CD8 CD3
     'PD1_CD8_CD3_tiv_inner': {'CD3':True, 'CD8': True, 'PD-1': True,#'PD-L1':False,
                               'tissue':'Inner margin'}, 
     'PD1_CD8_CD3_tiv_outer': {'CD3':True, 'CD8': True, 'PD-1': True,#'PD-L1':False,
                               'tissue':'Outer margin'},   

     #Tregs:
     'Treg_tiv_inner': {'CD3':True, 'CD8':False,'FOXP3':True,#'PD-L1':False,
                        'tissue':'Inner margin'}, 
     'Treg_tiv_outer': {'CD3':True, 'CD8':False,'FOXP3':True,#'PD-L1':False,
                        'tissue':'Outer margin'},   

     #PD1+ Tregs:
     'PD1_Treg_tiv_inner': {'CD3':True,'CD8':False, 'FOXP3':True, 'PD-1': True,#'PD-L1':False,
                            'tissue':'Inner margin'}, 
     'PD1_Treg_tiv_outer': {'CD3':True, 'CD8':False,'FOXP3':True, 'PD-1': True,#'PD-L1':False,
                            'tissue':'Outer margin'},   
    
    #PDL1- PanCK- PDL1- as estimate of macrophages:
    'PDL1neg_macro_tiv_outer': {'CD3':False, 'PanCK': False, 'PD-L1': False,
                        'tissue':'Outer margin'},   

    'PDL1_macro_tiv_outer': {'CD3':False, 'PanCK': False, 'PD-L1': True,
                        'tissue':'Outer margin'},   
    }

x = np.array([x for x in multi_label_types.keys()])
cell_names = x
tiv_only = x[pd.Series(x).str.contains('tiv')] #All tumor invasive margin hub types
inner_tiv = x[pd.Series(x).str.contains('inner')] #All inner vasive margin hub types
df = vecutils.vectra_if_types_to_cell_types(
                                 df.copy(),
                                 multi_label_dict = multi_label_types,
                                 col_dict = col_dict,
                                 verbose = True,
                                 type_adds_tissue = True,
                                 exclusive = False,
                                 add_cell_types_as_columns = False,
                                 output_cell_type_col = 'cell_type',
                                )

print(df.groupby('cell_type')['Class'].count())

# if debug:
# hub_cells = ['PDL1_tiv_inner', 'PDL1neg_tiv_outer']
# else:
hub_cells = tiv_only
tissue_crit = {'Outer margin': 0} #Tissue type & min number of cells > 0

print('scale_factor: %d, max neighbor dist: %d' 
        % (scale,max_dist))

keep_cx = []
fig = plt.figure(figsize=(7,7))
i = 1
start = time.time()
out_stack = []
                  
print('Beginning cell connection detection...')
col_dict = {'tissue_col': 'Parent',
            'cell_col': 'cell_type', #Changed from "Class" above
            'cell_x_pos' : 'Centroid X µm',
            'cell_y_pos' : 'Centroid Y µm',
            }
idx = ~df.loc[:,col_dict['cell_col']].isna()
subset = df.loc[idx,:].reset_index()

connections = delHelpers.df_to_connections_output(subset,
                                                  hub_cells = hub_cells,
                                                  scale = scale,
                                                  max_dist = max_dist,
                                                  col_dict = col_dict,
                                                  tissue_crit = tissue_crit,   
                                                  celltypes_as_cols = False,
                                                  )

delcx = delHelpers.generate_log_odds_matrix(connections,
                                             cell_names, 
                                             version = 1)

out_stack.append(delcx)
ax = fig.add_subplot(1,1,i)
col_idx = pd.Series(cell_names).isin(hub_cells)
ax = delPlots.connection_heatmap(delcx,
                                 cell_names,
                                 hub_cells = hub_cells,
                                 ax = ax
                                )
if i > 1:
    ylabel=''
else:
    ylabel = 'Spoke Cell'
new_fn ='%d_%s_delaunay_cx_%d_hubs' % (jobid,tissue_id, len(hub_cells))

print('Saving %s' % new_fn )

#Save full connectivity:
out_fn = output.joinpath(new_fn + '.csv')
connections.to_csv(out_fn)

#Save stack out log-odds connectivity matrices
out_fn = output.joinpath(new_fn + '.npy')
np.save(out_fn, arr=delcx, allow_pickle=False)

#Save heatmap            
img_fn = new_fn + '.png'
print('Saving %s' % img_fn)
plt.savefig(output.joinpath(img_fn))
            
stop = time.time()
print('Processing time: %2.2f minutes.' % ((stop-start)/60))
print('Finished!')
