import os
import sys
import pdb
import math
import random 
import cv2
from pathlib import Path
import pandas as pd
import numpy as np

def seg_fn_to_unique_tma_code(fns,
                              method=1):
    unique_codes=[]
    for fn in fns:        
        if 'TMA' in fn:
            core = 'Core[' + fn.split('Core[')[1].split(']_')[0] + ']'
            if method==1:
                tma = 'TMA%s_' % fn.split(' #')[1][0:2]
            elif method>=2:
                tma = 'TMA%02.0f_' % int(fn.split('TMA_')[1][0])
            unique_codes.append(tma + core)
        else:
            unique_codes.append(fn.split('_cell_seg_data.txt')[0])    
    return unique_codes

def tissue_id_to_ihc_fn(tissue_id,ihc_slide,method=1):
    if method == 1:
        fn = '%s_%s-%s.tiff' %(ihc_slide, tissue_id[-2],tissue_id[-4])
    return fn

def get_min_columns(df):
    cols = []
    for col in df.columns:
        if ('Nucleus' not in col) \
            and ('Cytoplasm' not in col) \
            and ('Entire Cell' not in col):
            #Keep
            cols.append(col)
    return cols

def get_celltypes(df,
                  method='vectra',
                  ):
    '''
        vectra->Deal with non-string entries in 'Phenotype' column.
        ihc ->
    '''
    new = []
    if method == 'ihc-init':
        ihc_types = df.IHC.unique()
        classes = df.Class.unique()
        for ihc in ihc_types:
            for cl in classes:
                new.append('%s-%s' % (ihc, cl))
    elif method == 'coex_cell':
        new = list(df.coex_cell.unique())
    else:
        xx=df.loc[:,'Phenotype']
        for x in xx:
            if isinstance(x,str):
                new.append(x)
            else:
                new.append(str(x))
    cell_types=np.array(new)
    return cell_types

def multi_to_index(df, 
                   multi_dict, 
                   use,
                   col_types=[],
                   exclusive=False, #I.e. assume all phenotypes not indicated must be negative
                   qc_thresh= 50,
                   verbose = False):
    '''
        Example multi-dict:
        multi_dict=     {'CD3+_CD8+' : {'CD3+':True, 'CD8+':True},
                         'CD3+_CD8+_TBet+': {'CD3+':True, 'CD8+':True, 'TBet+':True},
                         'CD3+_CD4+': {'CD3+':True,'CD4+':True},
                         'CD3+_CD4+_FOXP3+': {'CD3+':True, 'CD4+':True,'FOXP3+':True},
                         'CD3-_CD16+': {'CD3+': False, 'CD16+': True},
                         'CD3-_CD16+_TBet+': {'CD3+': False, 'CD16+': True, 'TBet+': True}}    
    '''
    use_multi_marker_cell = multi_dict[use]
   
    idx = (np.zeros((df.shape[0],)) + 1).astype(bool)     # Start all True:
    if verbose:
        print('Marker combination %s' % use)
    for marker in use_multi_marker_cell.keys():
        marker_col = 'Phenotype-%s' % marker[:-1]
        conf_col = 'Confidence-%s' % marker[:-1]               
        qc = get_qc(df, conf_col, qc_thresh)
        if use_multi_marker_cell[marker]: #If this gene should be +            
            marker_idx = df.loc[:,marker_col].values == marker
            if verbose:
                print('\t%s+, n = %d' % (marker[:-1],np.sum(marker_idx)))
        else: #This gene should be negative ('Other')            
            marker_idx = df.loc[:,marker_col].values != '%s+' %(marker)
            if verbose:
                print('\t%s-, n = %d' % (marker[:-1],np.sum(marker_idx)))
        idx = idx & qc & marker_idx
    if exclusive: #Also require that markers not defined are in fact negative (non overlapping)
        use = [x[:-1] for x in use_multi_marker_cell.keys()]
        if verbose:
            print(use)
        for marker_col in col_types:
            marker = marker_col.split('-')[1]
            if marker not in use:
                neg_idx =  df.loc[:,marker_col].values != '%s+' %(marker)
                if verbose:
                    print('\tExcluding %s' % marker)
                    print('\tn=%d' % np.sum(idx & neg_idx))
                idx = idx & neg_idx
    return idx
 
def get_qc(df, col='Confidence', qc_thresh=0):
    qc=[]
    for v in df[col]:
        if isinstance(v, str):
            qc.append(float(v.split('%')[0]))
        else:
            qc.append(np.nan)
    return np.array(qc) > qc_thresh

def cell_colormap(cell):
    color_dict = {'Others':tuple([0.8]*3),
                  'PanCK+':'black',
                  'CD68+': 'orange',
                  'CD4+':'g',
                  'CD8+':'b',
                  'CD20+':'m',
                  'FoxP3+':'r'}
    return color_dict[cell]

def vectra_um_to_pixel(dat,coord_cols,img_coord,px_width,img_size):
    x = dat.loc[:,coord_cols[0]].values
    x = (x - img_coord[0])/px_width + (img_size[0]/2)  
    y = dat.loc[:,coord_cols[1]].values
    y =  (y - img_coord[1])/px_width + (img_size[1]/2)
    return x,y