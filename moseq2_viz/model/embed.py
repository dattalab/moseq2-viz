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

    Parameters
    ----------
    syll_means
    unique_groups

    Returns
    -------

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

    Parameters
    ----------
    mean_df
    output_file
    embedding
    n_components

    Returns
    -------

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
                   symbols = "o*v^s"):
    '''

    Parameters
    ----------
    L
    y
    mapping
    rev_mapping
    output_file
    embedding
    x_dim
    y_dim
    symbols

    Returns
    -------

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