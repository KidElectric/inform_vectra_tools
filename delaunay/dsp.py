import sys
import numpy as np
import pandas as pd
from pathlib import Path

def roi_table_to_unique_tma_code(roi):
    unique_codes=[]
    n = roi.shape[0]
    for i in range(0,n,1):
        
        core = 'Core' + roi.loc[i,'tma_core_id']
        tma = 'TMA0%s_' % roi.loc[i,'Scan name'].split(' ')[2]
        unique_codes.append(tma + core)
    return unique_codes

def all_targets_Q3_norm(counts):
    ''' Implement GeoMx-style Q3 all-target normalization'''
    q3 = np.quantile(counts.values,q = (3/4),axis=0)
    mq3 = np.mean(q3)
    norm_counts = pd.DataFrame(counts)
    norm_counts = (norm_counts / q3) * mq3 # Heart of the normalization
    return norm_counts

def load_geomx_xlsx_clean(fn,
                          sheet_name = 'TargetCountMatrix'):
    df = pd.read_excel(fn,
                           sheet_name = sheet_name) 
    df.columns = df.columns.str.replace('|','',regex=False)
    df.columns = df.columns.str.replace('\s+','.',regex=True)
    if 'SegmentProperties' in sheet_name:
        for col in df.columns:
            # if 'SegmentDisplayName' in df.columns:
            # print('Segment display name clean.')
            if isinstance(df.loc[0,col],str):
                df.loc[:,col] = df.loc[:,col].str.replace('|','',regex=False)
                df.loc[:,col] = df.loc[:,col].str.replace('\s+','.',regex=True)
            
    # temp = counts.iloc[0,1:].T.to_frame().reset_index()
    # temp.columns=['joint_seg_label', 'example']
    # t1 =pd.merge(left = temp, right = roi, on='joint_seg_label',how='left')
    # t1.shape
    # t1 = t1.sort_values(by='joint_seg_label').reset_index(drop=True)
    # counts = counts[ ['TargetName'] + list(t1.joint_seg_label.values)] #Match order of ROI table and counts table
    
    old_fn = fn.parts[-1].split('.')[0]
    new_fn = old_fn + '_' + sheet_name + '.tsv'
    df.to_csv(fn.parent.joinpath(new_fn),
                  sep='\t')
    return df,fn.parent.joinpath(new_fn)

def roi_to_counts_col(roi, 
                     method = 'auto'):
    counts_col=[]
    n = roi.shape[0]
    r_scan_names = roi.loc[:,'Scan name'].str.replace('\s+','.',regex=True)
    if 'Segment (Name/ Label)' in roi.columns:
         r_segment_names = roi.loc[:,'Segment (Name/ Label)'].str.replace('\s+','.',regex=True)
    for i in range(0,n,1):
        if (method == 'auto') and ('Segment (Name/ Label)' in roi.columns):
            val = '%s | %03.0f | %s' % (roi.loc[i,'Scan name'],
                                        int(roi.loc[i,'ROI_ID']),
                                        roi.loc[i,'Segment (Name/ Label)'])
        elif method == 'ROI':
            val = '%s | %03.0f' % (roi.loc[i,'Scan name'],
                                   int(roi.loc[i,'ROI_ID']))
        elif method == 'R_ROI':
            val = '%s.%03.0f' % (r_scan_names[i],
                       int(roi.loc[i,'ROI_ID']))
        elif method == 'R_auto':
            val = '%s.%03.0f.%s' % (r_scan_names[i],
                            int(roi.loc[i,'ROI_ID']),
                            r_segment_names[i])
        counts_col.append(val)
    return counts_col

def combine_counts_by_roi(counts,roi):
    # Transpose to merge with ROI annotations:
    c = counts.T.reset_index()
    cols = c.iloc[0,:].values
    c = c.iloc[1:,:]
    cols[0] = 'joint_seg_label'
    markers = cols[1:]
    c.columns=cols
    
    #Merge additional ROI info:
    c = pd.merge(left=c, right=roi, on='joint_seg_label')
    c = c.groupby(by=['Scan name','ROI_ID'])[markers].sum().reset_index()
    joint_col = roi_to_counts_col(c, method='R_ROI')
    c.drop(columns=['Scan name','ROI_ID'],inplace=True)
    c.index = joint_col
    
    #Transpose back to be like original counts matrix:
    c=c.T.reset_index()
    cols = c.columns.values
    cols[0]='TargetName'
    c.columns=cols

    return c
    
def two_gene_scatter(gene_x,
                     gene_y,
                     counts,
                     conds,
                     ax,
                     cdict,
                     bkg_sub = False):
       
        yy = counts.iloc[counts['TargetName'].isin([gene_y]).values,1:].values
        yy= np.reshape(yy,(-1,))
        xx = counts.iloc[counts['TargetName'].isin([gene_x]).values,1:].values
        xx = np.reshape(xx,(-1,))
        if bkg_sub == True:
            bkg = counts.iloc[counts['TargetName'].isin(['Negative Probe']).values,1:].values
            # bkg = np.mean(counts.iloc[:,1:],axis=0)
            bkg = np.reshape(bkg,(-1,))
            axis_label = '%s (bkg subtracted, norm counts)' 
        else:
            bkg = np.zeros(xx.shape)
            axis_label = '%s (norm. counts)' 

        for cond in cdict.keys():
            idx = conds == cond
            ax.scatter(xx[idx]-bkg[idx],
                       yy[idx]-bkg[idx],
                       s = 100,
                       alpha = alpha,
                       c = cdict[cond], label=cond)
        ax.set_xlabel(axis_label % gene_x)
        ax.set_ylabel(axis_label % gene_y)
        ax.legend()
