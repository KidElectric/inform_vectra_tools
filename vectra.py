import os
import sys
import pdb
import math
import random 
import cv2
from pathlib import Path
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.path as mpltPath
from matplotlib.patches import Polygon
from matplotlib.colors import LogNorm

from skimage import measure

from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn import mixture #previously: from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity #As alternative to GMM (implemented but not used)
from sklearn.utils import resample
from sklearn.preprocessing import normalize

import scipy as sp
# import scipy.interpolate
from scipy.optimize import curve_fit

# master_scale = 0.4972 # pixel / um shown in .tif files
master_scale = 1 #I think vectra outputs might already be in microns? 

def seg_fn_to_unique_tma_code(fns,
                              method=1):
    unique_codes=[]
    for fn in fns:
        core = 'Core[' + fn.split('Core[')[1].split(']_')[0] + ']'
        if method==1:
            tma = 'TMA%s_' % fn.split(' #')[1][0:2]
        elif method==2:
            tma = 'TMA%02.0f_' % int(fn.split('TMA_')[1][0])
        unique_codes.append(tma + core)
    return unique_codes

def get_min_columns(df):
    cols = []
    for col in df.columns:
        if ('Nucleus' not in col) \
            and ('Cytoplasm' not in col) \
            and ('Entire Cell' not in col):
            #Keep
            cols.append(col)
    return cols

def get_celltypes(df):
    '''
        Deal with non-string entries in 'Phenotype' column.
    '''
    xx=df.loc[:,'Phenotype']
    new = []
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
                   exclusive=True, #I.e. assume all phenotypes not indicated must be negative
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
    use_dict = multi_dict[use]
   
    idx = (np.zeros((df.iloc[:,0].shape[0],)) + 1).astype(bool)     # Start all True:
    if verbose:
        print('Cell combination %s' % use)
    for cell in use_dict.keys():
        cell_col = cell
        conf_col = 'Conf-%s' % cell               
        qc = get_qc(df, conf_col, qc_thresh)
        if use_dict[cell]: #If this gene should be +            
            cell_idx = df.loc[:,cell].values == cell
            if verbose:
                print('\t%s+, n = %d' % (cell[0:-1],np.sum(cell_idx)))
        else: #This gene should be negative ('Other')            
            cell_idx = df.loc[:,cell].values == 'Other'
            if verbose:
                print('\t%s-, n = %d' % (cell[0:-1],np.sum(cell_idx)))
        idx = idx & qc & cell_idx
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

def tma_scatter(df,
                ax,
                qc_thresh=0,
                size = 6,
                alpha = 0.3,
                cells =  ['Others', 'PanCK+', 'CD68+', 'CD8+','CD4+', 'FoxP3+' ,'CD20+'],
                ):
    qc = get_qc(df, qc_thresh=qc_thresh)
    cell_types = get_celltypes(df)
    for i,cell in enumerate(cells):
        use = qc &   (cell_types == cell)
        x = df.loc[use,'Cell X Position']
        y = df.loc[use,'Cell Y Position']
        ax.scatter(x,y, c=cell_colormap(cell), s = size, alpha=alpha)
    ax.set_xlabel('Original X pixel')
    ax.set_ylabel('Original Y pixel')
    ax.invert_yaxis() 
    return ax

def get_kde_gauss(df, 
                  density_cell='CD20+',
                  n_comp=10,
                  qc_thresh=0,
                  method = 'GMM'):
    qc = get_qc(df,qc_thresh=qc_thresh)
    cell_types=get_celltypes(df)
    cells = np.unique(cell_types)
    p=[]
    for i,cell in enumerate(cells):
        use = qc &  (cell_types == cell)
        x = df.loc[use,'Cell X Position']
        y = df.loc[use,'Cell Y Position']
        p.append((x.values,y.values))
    p = np.concatenate(p,axis=1).T
    cell = density_cell
    use = (cell_types == cell) # & qc
    x = df.loc[use,'Cell X Position']
    y = df.loc[use,'Cell Y Position']
    
    xx = np.linspace(np.min(p[:,0]), np.max(p[:,0]))
    yy = np.linspace(np.min(p[:,1]),np.max(p[:,1]))
    X, Y = np.meshgrid(xx, yy)
    XX = np.array([X.ravel(), Y.ravel()]).T
    
    if method == 'GMM':
        clf = mixture.GaussianMixture(n_components=n_comp,
                                      covariance_type="full",
                                     random_state = 42)
        clf.fit(np.stack((x,y)).T)
        Z = -clf.score_samples(XX)
        
    elif method == 'KDE':
        kde = KernelDensity(kernel='gaussian', 
                            bandwidth=0.5).fit(np.stack((x,y),axis=1))
        Z = -kde.score_samples(XX)
    Z = Z.reshape(X.shape)
    return X,Y,Z

def plot_kde_contour(X,Y,Z,ax,
                     alpha=0.8,
                     linewidths=3,
                     method = 'GMM'):
       
    if method == 'GMM':
        CS = ax.contour(X, Y, Z,
                     norm=LogNorm(vmin=Z.min(), vmax=Z.max()),
                     levels=np.logspace(0, 2.5, 25),
                     alpha=alpha,
                     linewidths=linewidths,
                    )
        
    elif method =='KDE':
        CS = ax.contour(X, Y, Z,
             norm=LogNorm(vmin=Z.min(), vmax=Z.max()),
             levels=np.logspace(0, 7, 10),
             alpha=alpha,
             linewidths=linewidths,
            )        
        
    return CS
    
def df_to_labeled_kde(df, 
                      contour_thresh,
                      density_cell='CD20+',
                      n_comp=10,
                      qc_thresh=0):
    X,Y,Z = get_tls_gauss(df, 
                          density_cell=density_cell,
                          n_comp=n_comp, 
                          qc_thresh=qc_thresh)
    tZ = Z < contour_thresh
    labeled_tls = measure.label(tZ, background=0)
    return labeled_tls

def bin_im_to_contour_xy(df,
                         labeled_tls,
                         tls_num,
                         n_comp=10,
                         qc_thresh = 0,
                         use_blur = False,
                        ):
    bin_im = (labeled_tls==tls_num).astype(np.uint8)
    if use_blur:
        ksize = (5,5)
        bin_im = cv2.blur(bin_im.copy(),ksize)
    X,Y,Z = get_tls_gauss(df, 
                          density_cell='CD20+', 
                          n_comp=n_comp, 
                          qc_thresh=qc_thresh)
    cont,_ = cv2.findContours(bin_im,
                          cv2.RETR_EXTERNAL,
                          cv2.CHAIN_APPROX_SIMPLE)
    tls_xy= np.squeeze(cont)
    if (tls_xy.shape[0] > 0) and (len(tls_xy.shape) ==2) and (tls_xy.shape[1] > 0) :
        tls_xy= np.concatenate((tls_xy,tls_xy[0:1,:]),axis=0) #Close contour
        tls_xy_orig = np.stack((np.round(X[0,tls_xy[:,0]]).astype(np.int32),
                    np.round(Y[tls_xy[:,1],0]).astype(np.int32)),
                    axis=1)
    else:
        tls_xy_orig = tls_xy = np.array([[],[]])
    return tls_xy_orig #, tls_xy

def df_to_contour_xy(df,
                     method='GMM',
                     density_cell='CD20+', 
                     n_comp=10, 
                     qc_thresh=0,
                     use_contour= None):

    X,Y,Z = get_tls_gauss(df, 
                          density_cell=density_cell, 
                          n_comp=n_comp, 
                          qc_thresh=qc_thresh)

    fig =plt.figure(figsize=(12,10))
    ax = fig.add_subplot(1,1,1)
    ax = tma_scatter(df, ax)
    CS = plot_tls_contour(X,Y,Z,ax,
                         method = method)

    
    top = np.argwhere(np.array([len(x) for x in CS.allsegs]) > 0).flatten()
    print(top)
    tls_xy_orig = []
    if use_contour == None:
        if method == 'KDE':
            use_contour = 2
        elif method == 'GMM':
            use_contour =1
        
    tls_xy_orig = CS.allsegs[top[use_contour]]
    plt.close(fig)
    return tls_xy_orig

def cells_in_kde(tls_xy,cell_xy):
    path = mpltPath.Path(tls_xy)

    #Total cells in this TLS
    return path.contains_points(cell_xy)

def cells_in_kde_neighborhood(tls_center_xy,
                              cell_xy,
                              radius = 200, #um
                              scale = master_scale):
    
    dist = pairwise_distances(cell_xy,
                              tls_center_xy.reshape(1, -1),
                              metric='euclidean').flatten()
    is_neighbor = (dist * scale) < radius      
    return is_neighbor

def plot_circle(xy_center,
                radius, #In um
                ax,
                color='b',
                scale = 1): # to scale from pixel to um or vice versa
    circle1 = plt.Circle(xy_center, radius*scale, color=color, fill=False)
    ax.add_patch(circle1)
    return ax

def kde_cell_neighborhood(df,
                tls_df,
                radius=400, #um         
                qc_thresh=0,
                scale = master_scale, #pixel to um
                cells =  ['Others', 'PanCK+', 'CD68+', 'CD8+','CD4+', 'FoxP3+' ,'CD20+'],
                ):
    qc = get_qc(df, qc_thresh=qc_thresh)
    cell_types = get_celltypes(df)
    all_cell_xy =  df.loc[qc,['Cell X Position','Cell Y Position']].values
    total_col = 'tls_neighbor_total_n'
    for i,cell in enumerate(cells):
        use = qc &   (cell_types == cell)
        cell_xy = df.loc[use,['Cell X Position','Cell Y Position']].values
        all_tls = []        
        percents_tls = []
        neighbor_tot_tls = []
        output_col = 'tls_neighbor_xy_'+ cell
        percents_col = 'tls_neighbor_per_'+ cell
        total_col = 'tls_neighbor_total_n'
        #For each region with sufficient CD20 density:
        for tls_num in range(0,tls_df.shape[0]):
            tls_xy = np.array(tls_df.loc[tls_num,'tls_coords'])
            use_cells = ~cells_in_tls(tls_xy,cell_xy)   #Cell of interest not in current TLS
            other_cells = ~cells_in_tls(tls_xy,all_cell_xy)
            neighbor_tot_tls.append(np.sum(other_cells))
            center_coord = tls_df.loc[tls_num,'xy_center']
            
            if any(other_cells):
                all_neighbors_xy = all_cell_xy[other_cells,:]
                dist = pairwise_distances(all_neighbors_xy,
                                              center_coord.reshape(1, -1),
                                              metric='euclidean').flatten()
                n_neighbors = np.sum((dist * scale) < radius)              
                if any(use_cells):
                    neighbor_cell_xy = cell_xy[use_cells,:]

                    dist = pairwise_distances(neighbor_cell_xy,
                                                      center_coord.reshape(1, -1),
                                                      metric='euclidean').flatten()
                    is_neighbor = (dist * scale) < radius
                    all_tls.append(neighbor_cell_xy[is_neighbor,:])
                    percents_tls.append(np.sum(is_neighbor) / n_neighbors * 100)
                else:
                    all_tls.append(np.array([]))
                    percents_tls.append(0)
            else:
                all_tls.append(np.array([]))
                percents_tls.append(0)
        tls_df[percents_col] = percents_tls        
        tls_df[output_col] = all_tls
        tls_df[total_col] = neighbor_tot_tls
    return tls_df

def check_kde_from_labeled_kde(df, 
                               labeled_tls,
                               tls_num,
                               cd20_min = 10,
                               n_comp = 10,
                               min_cd20_density = 0,
                               cd4_min_percent = 0,
                               contour_thresh = 14,
                               qc_thresh = 0,                               
                               scale = master_scale / 1e6,
                               return_df = False,
                               use_cell_types = ['CD20+','CD4+',
                                                'CD8+','CD68+',
                                                'FoxP3+','PanCK+',
                                                'Others'] ):
    
    qc = get_qc(df,
                qc_thresh=qc_thresh)
    cell_xy = df.loc[:,['Cell X Position','Cell Y Position']].values
    cell_types = get_celltypes(df)
    tls_xy_orig = bin_im_to_contour_xy(df,
                                       labeled_tls,
                                       tls_num,
                                       n_comp=n_comp,
                                       qc_thresh = qc_thresh)
    tls_is_good = False
    
    ncd20 = ncd4 = ncd8 = ncd68 = nfoxp3 = npanck = n_total = area = 0
    cell_counts=[]
    
    if (tls_xy_orig.shape[0] > 0) and (len(tls_xy_orig.shape) ==2) and (tls_xy_orig.shape[1] > 0)  :    
        #Total cells in this TLS
        idx = qc
        n_total = np.sum(cells_in_tls(tls_xy_orig,cell_xy[idx,:]))
        
        for cell_type in use_cell_types:
            idx = qc & (cell_types == cell_type)
            cell_counts.append(np.sum(cells_in_tls(tls_xy_orig,cell_xy[idx,:]))) 

        ncd20,ncd4 = cell_counts[0:2]
        area = PolyArea(tls_xy_orig) * scale # min_cd20_density units should match scale
        if (ncd20 > cd20_min) \
            and ((ncd4 / n_total * 100) > cd4_min_percent) \
            and (ncd20 / area) > min_cd20_density:
            tls_is_good = True
        
    if return_df:
        if any(cell_counts):
            df = pd.DataFrame([cell_counts], columns=use_cell_types)
            df['tls_is_good']=tls_is_good
            df['n_total']= n_total
            df['area'] = area
            df['tls_coords'] = [tls_xy_orig]
            df['xy_center']= [np.mean(tls_xy_orig, axis=0)]
        else:
            df = pd.DataFrame(columns=use_cell_types + \
                              ['tls_is_good','n_total',
                               'area','tls_coords','xy_center'])
        return df    
    
    else:
        return tls_is_good, n_total, ncd20, ncd4, area

#Perimeter?
#Cell heterogeneity / randomness of cd20/cd4 distribution

def PolyArea(xy): #Shoe-lace formula
    x=xy[:,0]
    y=xy[:,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def df_to_kde_info(df,
                cd20_min= 10,
                n_comp = 10, #Number of gaussians to use in mixture model
                cd4_min_percent = 0,
                contour_thresh = 14,
                min_cd20_density = 0, #cells / mm^2 ( >5000 looks like a true cluster)
                qc_thresh = 0,
                scale = master_scale / 1e6): #~0.5um / px to square mm
    
    labeled_tls = df_to_labeled_tls(df,
                                    contour_thresh,
                                    qc_thresh = qc_thresh,
                                    n_comp=n_comp
                                   )
    cell_xy = df.loc[:,['Cell X Position','Cell Y Position']].values
    cell_types = get_celltypes(df)
    count=[]
    cd20=[]
    cd4=[]
    good_tls = []
    tls_area = []
    
    #For each TLS region identified (background is 0--skip):
    for i in range(1,(np.max(labeled_tls)+1)):
        tls_is_good, cc, ncd20, ncd4, area = check_tls_from_labeled_tls(df, 
                                                               labeled_tls,
                                                               i,
                                                               cd20_min= cd20_min,
                                                               min_cd20_density =  min_cd20_density ,
                                                               cd4_min_percent = cd4_min_percent,
                                                               contour_thresh = contour_thresh,
                                                               scale = scale,
                                                               qc_thresh=qc_thresh)        
        if tls_is_good:            
            #Total cells in this TLS
            count.append(cc)
            
            #Count celltypes in region
            cd20.append(ncd20)
            cd4.append(ncd4)       
            
            #Append area
            tls_area.append(area)
            
            print('TLS %d/%d, n CD20+: %d, CD4+: %2.1f%%' \
                  % (i,np.max(labeled_tls),ncd20,ncd4/cc*100))
            
            good_tls.append(i)
                
    return good_tls, cd20, cd4, count, tls_area, labeled_tls

def plot_kde_df_coord(tls_df,tls_num,ax):
    xy = tls_df.loc[tls_num,'coords']
    ax.plot(xy[:,0],xy[:,1])
    
def celldf_to_kde_df(df,
                cd20_min= 10,
                n_comp = 10, #Number of gaussians to use in mixture model
                cd4_min_percent = 0,
                contour_thresh = 14,
                min_cd20_density = 0, #cells / mm^2 ( >5000 looks like a true cluster)
                qc_thresh = 0,
                scale = master_scale / 1e6,
                only_keep_good = False,
                ):
    
    labeled_tls = df_to_labeled_tls(df,contour_thresh,
                                    qc_thresh = qc_thresh,
                                    n_comp=n_comp, 
                                    )
    cell_xy = df.loc[:,['Cell X Position','Cell Y Position']].values
    cell_types = get_celltypes(df)
    tls_df = pd.DataFrame()
    
    #For each TLS region identified (background is 0--skip):
    for i in range(1,(np.max(labeled_tls)+1)):
        temp = check_tls_from_labeled_tls(df, 
                                       labeled_tls,
                                       i,
                                       cd20_min= cd20_min,
                                       min_cd20_density =  min_cd20_density ,
                                       cd4_min_percent = cd4_min_percent,
                                       contour_thresh = contour_thresh,
                                       scale = scale,
                                       qc_thresh=qc_thresh,
                                       return_df = True,
                                       )
        if temp.shape[0]>0:
            tls_df = pd.concat((tls_df,temp),axis=0)
    
    #Add number of good TLS as a feature?
    if tls_df.shape[0] > 0:
        is_good = tls_df.loc[:,'tls_is_good'].values
        n_good = np.sum(is_good)
        tls_df['n_good_tls']= [n_good] * tls_df.shape[0]
        if only_keep_good:
            tls_df = tls_df.loc[is_good,:]
    return tls_df.reset_index(drop=True)

def kde_df_to_long_cell_xy(tls_df,
                           use_cells):
    '''
        Take tls_df and make into long-format dataframe with each row as a cell.
    '''
    new_df = pd.DataFrame([])
    tls_ids = np.unique(tls_df.loc[:,'tls_id'].values)
    for tls_id in tls_ids:
        idx = tls_df.loc[:,'tls_id'].values == tls_id
        for cell in use_cells:
            in_out = ['cell_coord_%s' % cell,
                      'tls_neighbor_xy_%s' % cell]
            for loc in in_out:
                temp = pd.DataFrame([])

                # Collect xy coords of cells inside TLS boundary
                xy_coords = tls_df.loc[idx,loc].squeeze()
                if xy_coords.any():
                    temp['Cell X Position'] = xy_coords[:,0]
                    temp['Cell Y Position'] = xy_coords[:,1]
                    temp['Phenotype'] = cell
                    temp['is_neighbor'] = 'neighbor' in loc
                    temp['tls_id'] = tls_id
                    new_df = pd.concat((new_df,temp),axis=0)
    return new_df

def add_percent(tls_df,use_cells):
    for cell in use_cells:
        per = tls_df.loc[:,cell] / tls_df.loc[:,'n_total'] * 100
        col_name = 'per_'+ cell
        tls_df[col_name]=per
    return tls_df

def add_density(tls_df,use_cells):
    for cell in use_cells:
        dense = tls_df.loc[:,cell] / tls_df.loc[:,'area']
        col_name = 'dense_'+ cell
        tls_df[col_name]=dense
    return tls_df

def add_cell_coords(tls_df,
                    df,
                    use_cells,
                    down_sample=None,
                    center = False,
                    qc_thresh=0,
                    sort_by_dist=True):
    ''' 
    Collect cell positions relative to TLS center and measure pairwise distances if requested.
    
    add_cell_coords(tls_df,
                    df,
                    use_cells,
                    down_sample=None,
                    center = False,
                    qc_thresh=0,
                    sort_by_dist=True) #Dist is only used to sort, so set false to prevent dist calc.
    For each TLS region, zero cell coords to TLS center (if desired),
    then calculate pair-wise cell-cell distances, on a down-sampled (constant)
    number of neighboring cells, then sort cells by distance from TLS center (if desired).
    
    '''
    qc = get_qc(df,qc_thresh=qc_thresh)
    cell_types = get_celltypes(df)
    for cell in use_cells:
        col_name = 'cell_coord_'+ cell
        
        if down_sample is not None:
            col_name = col_name +'_ds-%d' % down_sample 
        
        if center == True:
            col_name = col_name + '_centered' 
            
        idx = qc & (cell_types == cell) 
        x = df.loc[idx,'Cell X Position']
        y = df.loc[idx,'Cell Y Position']
        cell_xy = np.stack([x,y],axis=1) 
        all_tls = []
        
        #For each region with sufficient CD20 density:
        for tls_num in range(0,tls_df.shape[0]):
            tls_xy = np.array(tls_df.loc[tls_num,'tls_coords'])
            use_cells = cells_in_tls(tls_xy,cell_xy)   
            center_coord = tls_df.loc[tls_num,'xy_center']
            
            #If any cells found within TLS region:
            if any(use_cells):
                tls_cell_xy = cell_xy[use_cells,:]
                
                #Resample cells to use:
                if down_sample is not None:
                    tls_cell_xy= resample(tls_cell_xy, 
                                  replace = True,
                                  n_samples = down_sample,
                                  random_state = 42)
                    
                #Sort tls_coords by distance to tls center:
                if sort_by_dist == True:                    
                    dist = pairwise_distances(tls_cell_xy,
                                              center_coord.reshape(1, -1),
                                              metric='euclidean')
                    sort_idx = np.argsort(dist, axis = 0).flatten()
                    tls_cell_xy = tls_cell_xy[sort_idx,:]
                
                #Center to mean coordinate of TLS   
                if center == True:                                    
                    tls_cell_xy = tls_cell_xy - center_coord
                          
                all_tls.append(tls_cell_xy)
            else:
                all_tls.append(np.array([]))
                
        tls_df[col_name] = all_tls
    return tls_df

def add_withincell_pairwise_dist(tls_df,
                          use_cells,
                          down_sample = None,
                          upper_triangle = False,
                          ):
    
    #Require that add_cell_coords has already been run
    #use_cells = ['CD20+','CD4+'] etc
    for cell in use_cells:        
        use_col_name = 'cell_coord_'+ cell
        dist_col_name= 'pdist_' + cell
        all_tls = []
        if down_sample is not None:
            use_col_name = use_col_name + '_ds-%d' % down_sample
            dist_col_name = dist_col_name + '_ds-%d' % down_sample
            
        for tls_num in range(0,tls_df.shape[0]): 
            if use_col_name in tls_df.columns:   
                cell_xy = tls_df.loc[tls_num, use_col_name] 
                if cell_xy.size >0:
                    tls_xy = tls_df.loc[tls_num,'tls_coords']
                    use_cells = cells_in_tls(tls_xy,cell_xy)             
                    tls_cell_xy = cell_xy[use_cells,:]
                    dist = pairwise_distances(tls_cell_xy,
                                          metric='euclidean') #'mahalanobis' ?
                    if upper_triangle:
                        mask = ~np.tri(*dist.shape[-2:], k=0, dtype=bool)
                        dist = dist[mask]
                    all_tls.append(dist)
                else:
                    all_tls.append(np.array([]))
            else:
                all_tls.append(np.array([]))
            
        tls_df[dist_col_name] = all_tls
    return tls_df


def add_acrosscell_pairwise_dist(tls_df,
                          cell_1,
                          cell_2,
                          down_sample,
                          upper_triangle = False,
                          ):
    
    #Require that add_cell_coords has already been run and resampled
             
    use_col_name1 = 'cell_coord_'+ cell_1 + '_ds-%d' % down_sample
    use_col_name2 = 'cell_coord_'+ cell_2 + '_ds-%d' % down_sample
    col_names = [use_col_name1, use_col_name2]
    dist_col_name = 'pdist_' + cell_1 + '_' + cell_2 + '_ds-%d' % down_sample
    all_tls = []
    
    for tls_num in range(0,tls_df.shape[0]): 
        tls_cell_xy=[]
        for use_col_name in col_names:
            if use_col_name in tls_df.columns:   
                cell_xy = tls_df.loc[tls_num, use_col_name]                 
                if (cell_xy.size >0):
                    tls_xy = tls_df.loc[tls_num,'tls_coords']
                    use_cells = cells_in_tls(tls_xy,cell_xy)   
                    tls_cell_xy.append(cell_xy[use_cells,:])
        if len(tls_cell_xy) == 2:            
            dist = pairwise_distances(tls_cell_xy[0],tls_cell_xy[1],
                                  metric='euclidean')         
            if upper_triangle:
                mask = ~np.tri(*dist.shape[-2:], k=0, dtype=bool)
                dist = dist[mask]
            all_tls.append(dist)
        else:
            all_tls.append(np.array([]))
    tls_df[dist_col_name] = all_tls
    return tls_df

def add_withincell_center_dist(tls_df,
                          use_cells,
                          down_sample = None,
                          ):
    
    #Require that add_cell_coords has already been run
    #use_cells = ['CD20+','CD4+'] etc
    for cell in use_cells:        
        use_col_name = 'cell_coord_'+ cell
        dist_col_name= 'center_dist_' + cell
        all_tls = []
        if down_sample is not None:
            use_col_name = use_col_name + '_ds-%d' % down_sample
            dist_col_name = dist_col_name + '_ds-%d' % down_sample
            
        for tls_num in range(0,tls_df.shape[0]): 
            if use_col_name in tls_df.columns:   
                cell_xy = tls_df.loc[tls_num, use_col_name] 
                if cell_xy.size >0:
                    tls_xy = tls_df.loc[tls_num,'tls_coords']
                    use_cells = cells_in_tls(tls_xy,cell_xy)             
                    tls_cell_xy = cell_xy[use_cells,:]
                    center_coord = tls_df.loc[tls_num,'xy_center']
                    dist = pairwise_distances(tls_cell_xy,
                                              center_coord.reshape(1, -1),
                                              metric='euclidean').flatten()
                    dist = np.sort(dist).flatten()
                    all_tls.append(dist)
                else:
                    all_tls.append(np.array([]))
            else:
                all_tls.append(np.array([]))
        tls_df[dist_col_name] = all_tls
    return tls_df


def plot_kde_types(output,
                   tls_idx,
                   tls_id_order,
                   r=4,
                   c=6):
    
    
    for clust in tls_id_order:               
        fig, axes = plt.subplots(nrows=r, ncols=c,
                               sharex=True, sharey=True,
                               figsize=(12,8))
        
        exam = np.argwhere(tls_idx==clust).flatten()
        orig_n = len(exam)
        if len(exam) > (r*c):
            exam = exam[0:(r*c)]
        i = 0
        for ex in exam:
            ax = axes.flatten()[i]
            for cell in plot_cells:
                #Plot
                cell_xy = output.loc[ex,'cell_coord_%s' % cell].values[0] 
                if cell_xy.size >0:
                    cell_xy = (cell_xy - output.loc[ex,'xy_center'].values[0]) * scale
                    ax.scatter(cell_xy[:,0],cell_xy[:,1],
                               marker = 'o', c = tls.cell_colormap(cell), s = marker_size,
                               alpha = 0.5)
            ax.set_title('%s]' % (output.loc[ex,'tls_id'].values[0].split(']')[0]), fontsize=10)
            ax.set_xlim([-plot_pixels,plot_pixels])
            ax.set_ylim([-plot_pixels,plot_pixels])
            i = i + 1
        fig.suptitle('Cluster %d examples (n=%d total)' % (clust,orig_n), fontsize=16)
        fig.text(0.5, 0.07, 'X Distance from TLS center (um)', ha='center')
        fig.text(0.07, 0.5, 'Y Distance from TLS center (um)', va='center', rotation='vertical')