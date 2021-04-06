'''
Utility file for computing and visualizing syllable label scalar and stat embeddings.
'''

import numpy as np
import seaborn as sns
from itertools import product
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def get_Xy_values(syll_means, unique_groups, stat='usage'):
    '''
    Computes the syllable mean statistics for each session, stored in X. Computes and corresponding
     mapped group name value for each of the sessions to be tracked when plotting the values in the embedding steps.

    Parameters
    ----------
    syll_means (pd DataFrame): Dataframe of syllable mean statistics
    unique_groups (1D list): list of unique groups in the syll_means dataframe.
    stat (str): statistic column to read from the syll_means df.

    Returns
    -------
    X (2D np.array): mean syllable statistics for each session. (nsessions x nsyllables)
    y (1D list): list of group names corresponding to each row in X.
    mapping (dict): dictionary conataining mappings from group string to integer for later embedding.
    rev_mapping (dict): inverse mapping dict to retrieve the group names given their mapped integer value.
    '''

    X, y = [], []

    for u in syll_means.uuid.unique():
        uuid_idx = syll_means['uuid'] == u

        X.append(syll_means[uuid_idx][stat])
        y.append(syll_means[uuid_idx]['group'].unique()[0])

    mapping = {g: i for i, g in enumerate(unique_groups)}
    rev_mapping = {v: k for k, v in mapping.items()}

    y = np.array([mapping[l] for l in y])

    return np.array(X), y, mapping, rev_mapping

def run_2d_embedding(mean_df, stat='usage', output_file='2d_embedding.pdf', embedding='LDA', n_components=2):
    '''
    Computes a 2D embedding of the mean syllable statistic of choice. User selects an embedding type, a stat
     to compute the embedding on, and provides a dataframe with the mean syllable information.
     The function will output a figure of the 2D representation of the embedding.

    Parameters
    ----------
    mean_df (pd DataFrame): Dataframe of the mean syllable statistics for all sessions
    output_file (str): path to saved outputted figure
    embedding (str): type of embedding to run. Either ['lda', 'pca']
    n_components (int): Number of components to compute.

    Returns
    -------
    fig (matplotlib figure): figure containing plotted 2d embedding.
    ax (matplonlib axes): axes instance for plotted figure.
    '''

    if embedding.lower() == 'lda':
        embedder = LDA(solver='eigen', shrinkage='auto', n_components=n_components, store_covariance=True)
    elif embedding.lower() == 'pca':
        embedder = PCA(n_components=n_components)
    else:
        print('Unsupported input. Only input embedding="lda" or "pca".')
        return

    syllable_df = mean_df.groupby(['syllable', 'uuid', 'group'], as_index=False).mean()

    unique_groups = syllable_df.group.unique()

    X, y, mapping, rev_mapping = get_Xy_values(syllable_df, unique_groups, stat=stat)

    L = embedder.fit_transform(X, y)

    fig, ax = plot_embedding(L, y, mapping, rev_mapping, output_file=output_file, embedding=embedding)

    return fig, ax

def plot_embedding(L,
                   y,
                   mapping,
                   rev_mapping,
                   output_file='embedding.pdf',
                   embedding='LDA',
                   x_dim=0,
                   y_dim=1,
                   symbols="o*v^s"):
    '''

    Parameters
    ----------
    L (2D np.array): the embedding representations of the mean syllable statistic to plot.
    y (1D list): list of group names corresponding to each row in L.
    mapping (dict): dictionary conataining mappings from group string to integer for later embedding.
    rev_mapping (dict): inverse mapping dict to retrieve the group names given their mapped integer value.
    output_file (str): path to saved outputted figure
    embedding (str): type of embedding to run. Either ['lda', 'pca'].
    x_dim (int): component number to graph on x-axis
    y_dim (int): component number to graph on y-axis
    symbols (str): symbols to use to draw different groups.

    Returns
    -------
    fig (matplotlib figure): figure containing plotted 2d embedding.
    ax (matplonlib axes): axes instance for plotted figure.
    '''

    fig, ax = plt.subplots(1, 1, figsize=(16, 16), facecolor='w')

    # Create color and symbol combination
    colors = sns.color_palette(n_colors=int(((len(y) + 1) / (len(symbols) - 1))))
    symbols, colors = zip(*list(product(symbols, colors)))

    # Set figure axes
    ax.set_xlabel(f'{embedding} 1')
    ax.set_ylabel(f'{embedding} 2')
    ax.set_xticks([])
    ax.set_yticks([])

    # plot each group's embedding
    for i in range(len(mapping)):
        # get embedding indices
        idx = [y == i]

        # plot mean embedding with corresponding symbol and color
        mu = np.nanmean(L[idx], axis=0)
        plt.plot(mu[x_dim], mu[y_dim], symbols[i], color=colors[i], markersize=10)

        # plot text group name indicator at computed mean
        plt.text(mu[x_dim], mu[y_dim], rev_mapping[i] + " (%s)" % symbols[i],
                 fontsize=18,
                 color=colors[i],
                 horizontalalignment='center',
                 verticalalignment='center')

    sns.despine()
    fig.savefig(output_file, bbox_inches='tight', format='pdf')

    return fig, ax