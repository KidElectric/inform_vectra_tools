import math
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import mixture #previously: from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity #As alternative to GMM (implemented but not used)
from matplotlib import pyplot as plt
import matplotlib.path as mpltPath
from matplotlib.patches import Polygon
from matplotlib.colors import LogNorm



def cell_colormap(cell):
    color_dict = {'Others':tuple([0.8]*3),
                  'PanCK+':'black',
                  'CD68+': 'orange',
                  'CD4+':'g',
                  'CD8+':'b',
                  'CD20+':'m',
                  'FoxP3+':'r',
                  'PDL1+':'orange'}
    return color_dict[cell]

def get_qc(df, 
           col='Confidence',
           method = 'vectra',
           qc_thresh=0):
    qc=[]
    if method == 'vectra':
        for v in df[col]:
            if isinstance(v, str):
                qc.append(float(v.split('%')[0]))
            else:
                qc.append(np.nan)
    else:
        qc = np.zeros((df.shape[0],))+1
    return np.array(qc) > qc_thresh

def tma_scatter(df,
                ax,
                size = 6,
                alpha = 0.3,
                qc_thresh =0,
                cells =  ['Others', 'PanCK+', 'CD68+', 'CD8+','CD4+', 'FoxP3+' ,'CD20+'],
                method = 'vectra'
                ):
    qc = get_qc(df, qc_thresh=qc_thresh, method = method)
    cell_types = get_celltypes(df, method=method)
    if method == 'vectra':
        cell_x = 'Cell X Position'
        cell_y = 'Cell Y Position'
    elif method == 'qupath':
        cell_x = 'Centroid X µm'
        cell_y = 'Centroid Y µm'
    for i,cell in enumerate(cells):
        use = qc & (cell_types == cell)
        x = df.loc[use,cell_x]
        y = df.loc[use,cell_y]
        ax.scatter(x,y, c=cell_colormap(cell), s = size, alpha=alpha)
    ax.set_xlabel('Original X pixel')
    ax.set_ylabel('Original Y pixel')
    ax.invert_yaxis() 
    return ax

def fit_gauss(df, 
                  density_cell='CD20+',
                  n_comp=10,
                  qc_thresh=0,
                  cell_detect_method = 'vectra',
                  method = 'GMM'):
    qc = get_qc(df, qc_thresh=qc_thresh, method=cell_detect_method)
    cell_types=get_celltypes(df,method=cell_detect_method)
    cells = np.unique(cell_types)
    p=[]
    if cell_detect_method == 'vectra':
        cell_x = 'Cell X Position'
        cell_y = 'Cell Y Position'
    elif cell_detect_method == 'qupath':
        cell_x = 'Centroid X µm'
        cell_y = 'Centroid Y µm'
    for i,cell in enumerate(cells):
        use = qc &  (cell_types == cell)
        x = df.loc[use,cell_x]
        y = df.loc[use,cell_y]
        p.append((x.values,y.values))
    p = np.concatenate(p,axis=1).T
    cell = density_cell
    use = (cell_types == cell) # & qc
    x = df.loc[use,cell_x]
    y = df.loc[use,cell_y]
    
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

def plot_contour(X,Y,Z,ax,
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
        print(LogNorm(vmin=Z.min(), vmax=Z.max()))
        CS = ax.contour(X, Y, Z,
             norm=LogNorm(vmin=Z.min(), vmax=Z.max()),
             levels=np.logspace(0, 7, 10),
             alpha=alpha,
             linewidths=linewidths,
            )        
        
    return CS

def fit_gauss_new(df, 
                  density_cell='CD20+',
                  n_comp=10,
                  qc_thresh=0,
                  cell_detect_method = 'vectra',
                  method = 'GMM'):
    qc = get_qc(df, qc_thresh=qc_thresh, method=cell_detect_method)
    cell_types=get_celltypes(df,method=cell_detect_method)
    cells = np.unique(cell_types)
    p=[]
    if cell_detect_method == 'vectra':
        cell_x = 'Cell X Position'
        cell_y = 'Cell Y Position'
    elif cell_detect_method == 'qupath':
        cell_x = 'Centroid X µm'
        cell_y = 'Centroid Y µm'
    for i,cell in enumerate(cells):
        use = qc &  (cell_types == cell)
        x = df.loc[use,cell_x]
        y = df.loc[use,cell_y]
        p.append((x.values,y.values))
    p = np.concatenate(p,axis=1).T
    cell = density_cell
    use = (cell_types == cell) # & qc
    xc = df.loc[use,cell_x]
    yc = df.loc[use,cell_y]
    
    xx = np.linspace(np.min(p[:,0]), np.max(p[:,0]))
    yy = np.linspace(np.min(p[:,1]),np.max(p[:,1]))
    X, Y = np.meshgrid(xx, yy)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    xy = xy[use]
    
    if method == 'GMM':
        clf = mixture.GaussianMixture(n_components=n_comp,
                                      covariance_type="full",
                                     random_state = 42)
        clf.fit(np.stack((xc,yc)).T)
        Z = -clf.score_samples(xy)
        
    elif method == 'KDE':
        kde = KernelDensity(bandwidth=0.4, #bandwidth=0.5
                            metric="haversine", 
                            kernel="gaussian", 
                            algorithm="ball_tree"
                            ).fit(np.stack((xc,yc),axis=1))
        Z = np.exp(kde.score_samples(xy)) #-kde.score_samples(XX)
    Z = Z.reshape(X.shape)
    return X,Y,Z