from matplotlib import pyplot as plt
import seaborn as sns

def connection_heatmap(output,
                       cell_names,
                       label = 'log odds (spoke:NotSpoke)',
                       ylabel='Spoke Cell',
                       vmin = -6,
                       vmax = 4,
                       ax = None):    
    if ax==None:
        fig = plt.figure(figsize=[8,8],)
        ax = fig.add_subplot(1,1,1,aspect='equal')
    g = sns.heatmap(output,
                square=True,
                cmap = 'coolwarm',
                xticklabels=cell_names,
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