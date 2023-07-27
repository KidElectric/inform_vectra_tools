import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def connection_heatmap(output,
                       cell_names,
                       hub_cells = [],
                       label = 'log odds (spoke:NotSpoke)',
                       ylabel='Spoke Cell',
                       vmin = -6,
                       vmax = 4,
                       ax = None):    
    if ax==None:
        fig = plt.figure(figsize=[8,8],)
        ax = fig.add_subplot(1,1,1,aspect='equal')
    if not any(hub_cells):
        hub_cells = cell_names
    col_idx = pd.Series(cell_names).isin(hub_cells)
    
    g = sns.heatmap(output[:,col_idx],
                square=True,
                cmap = 'coolwarm',
                xticklabels=hub_cells,
                yticklabels=cell_names,
                ax=ax,
                vmin = vmin,
                vmax = vmax,
                cbar_kws={"shrink": 0.5,
                         'label': label})

    g.set_facecolor('xkcd:gray')
    ax.set_xlabel('Hub cell')
    ax.set_ylabel(ylabel)
    return ax