import time
import os
import sys
import pdb
import math
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay

#Custom helper functions:
sys.path.append('/ihome/rbao/bri8/scripts/')
from helpers import tls
from helpers import delaunay as delHelpers

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

qct = 50
scale = 1
max_dist = 500
print('QC threshold: %d, scale_factor: %d, max neighbor dist: %d' 
        % (qct,scale,max_dist))
qc = tls.get_qc(df,qc_thresh=qct)

# Would be great to put all this in a .yaml file:
# (as well as path variables etc.)

subset=df.loc[qc,('Tissue Category',
                 'Phenotype',
                 'Cell X Position',
                 'Cell Y Position')]

point_lookup = delHelpers.df_to_cell_type_dict(subset)
points = subset.loc[:,('Cell X Position',
                       'Cell Y Position')
                       ].values   
tri = Delaunay(points)

use_cells = [('Phenotype', 'CD20+'),
             ('Phenotype', 'CD4+'),
             ('Phenotype', 'CD8+'),
             ('Phenotype', 'CD68+'),
             ('Phenotype', 'FoxP3+'),
             ('Phenotype', 'PanCK+'),
             ('Tissue Category', 'Tumor'),
             ('Tissue Category', 'Immune Cell Cluster'),
             ('Tissue Category', 'AF/Macrophage Cluster'),
            ]

cell_names = ['CD20+',
              'CD4+',
              'CD8+',
              'CD68+',
              'FoxP3+',
              'PanCK+',
              'Tumor',
              'Immune Cell Cluster',
              'AF/Macrophage Cluster']

i = 0
connections = pd.DataFrame()
print('Beginning cell connection detection...')

for col,cell in use_cells:
    idx = (subset.loc[:,col].values == cell)
    hub_cell_points = subset.loc[idx.T,
                        ('Cell X Position',
                         'Cell Y Position')
                       ].values    
    hub_ids = subset.index[idx.T] #Will allow easier mapping back to x,y coordinates later if desired
    
    #For each hub cell type, count types of connections    
    for i, hub in zip(hub_ids, hub_cell_points):
        vert = delHelpers.point_to_vert(tri.points, hub)
        if vert > 0:
            connected_verts = delHelpers.vert_to_connected_verts(vert, tri)
            connected_points = tri.points[connected_verts]

            #Filter out cells > 100um away:
            is_neighbor, dists = delHelpers.cell_dist_criterion(tri.points[vert],
                                      connected_points,
                                      radius = max_dist,
                                      scale = scale)
            cx_ids = connected_verts[is_neighbor]
            connected_points = connected_points[is_neighbor]
            connected_dists = dists[is_neighbor]
            spoke_cell_types = delHelpers.point_list_to_celltype(connected_points, point_lookup)
            temp = pd.DataFrame(spoke_cell_types, columns = ['cx_cell','cx_tissue'])
            temp['dist_um'] = connected_dists
            temp['hub_id'] = i
            temp['cx_id'] = cx_ids
            temp['hub_cell'] = cell       
            connections = pd.concat((connections,temp),axis=0)
        else:
            print(i,hub,'Not found')
        
stop = time.time()
print('Processing time: %2.2f minutes.' % ((stop-start)/60))
print(connections.head())
new_fn =jobid + '_' \
        +  fn.parts[-1].split(']_')[0] \
        + ']_delaunay_cx_whole_core.csv'

print('Saving %s' % new_fn)
out_fn = output.joinpath(new_fn)
connections.to_csv(out_fn)
print('Finished!')
