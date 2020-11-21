'''

Syllable transition graph creation and utility functions.

'''

import math
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from cytoolz import sliding_window, complement

def get_trans_graph_groups(model_fit):
    '''
    Wrapper helper function to get the groups and their respective session uuids
    to use in transition graph generation.

    Parameters
    ----------
    model_fit (dict): trained model ARHMM containing training data UUIDs.
    index (dict): index file dict containing corresponding UUIDs.
    sorted_index (dict): sorted version of the index dict.

    Returns
    -------
    group (list): list of unique groups included
    label_group (list): list of groups for each included session
    model_uuids (list): list of corresponding UUIDs for each included session in the model
    '''

    model_uuids = model_fit.get('train_list', model_fit['keys'])
    label_group = [model_fit['metadata']['groups'][k] for k in model_uuids]

    return label_group, model_uuids

def get_group_trans_mats(labels, label_group, group, max_sylls, normalize='bigram'):
    '''
    Computes individual transition matrices for each given group.

    Parameters
    ----------
    labels (np.ndarray): list of frame labels for each included session
    label_group (list): list of groups for each included session
    group (list): list of unique groups included
    max_sylls (int): maximum number of syllables to include in transition matrix.

    Returns
    -------
    trans_mats (list of 2D np.ndarrays): list of transition matrices for each given group.
    usages (list of lists): list of corresponding usage statistics for each group.
    '''
    # Importing within function to avoid import loops
    from moseq2_viz.model.util import get_syllable_statistics

    trans_mats = []
    usages = []

    # Computing transition matrices for each given group
    for plt_group in group:
        # Get sessions to include in trans_mat
        use_labels = [lbl for lbl, grp in zip(labels, label_group) if grp == plt_group]
        trans_mats.append(get_transition_matrix(use_labels,
                                                normalize=normalize,
                                                combine=True,
                                                max_syllable=max_sylls))

        # Getting usage information for node scaling
        usages.append(get_syllable_statistics(use_labels, max_syllable=max_sylls)[0])
    return trans_mats, usages

def compute_and_graph_grouped_TMs(config_data, labels, label_group, group):
    '''
    Convenience function to compute a transition matrix for each given group.
    Function will also graph the computed transition matrices, then return the open figure object to be saved.

    Parameters
    ----------
    config_data (dict): configuration dictionary containing graphing parameters
    labels (list): list of 1D numpy arrays containing syllable labels per frame for every included session
    label_group (list): list of corresponding group names to plot transition aggregated transition plots
    group (list): unique list of groups to plot

    Returns
    -------
    plt (pyplot.Figure): open transition graph figure to save
    '''

    trans_mats, usages = get_group_trans_mats(labels, label_group, group, config_data['max_syllable'], config_data['normalize'])

    # Option to not scale node sizes proportional to the syllable usage.
    if not config_data['scale_node_by_usage']:
        usages = None

    print('Creating plot...')
    plt, _, _ = graph_transition_matrix(trans_mats,
                                        **config_data,
                                        usages=usages,
                                        groups=group,
                                        headless=True)

    return plt

def get_transitions(label_sequence):
    '''
    Computes labels switch to another label. Throws out the first state (usually
    labeled as -5).

    Parameters
    ----------
    label_sequence (tuple): a tuple of syllable transitions and their indices

    Returns
    -------
    transitions (np.array): filtered label sequence containing only the syllable changes
    locs (np.array): list of all the indices where the syllable label changes
    '''

    arr = deepcopy(label_sequence)

    # get syllable transition locations
    locs = np.where(arr[1:] != arr[:-1])[0] + 1
    transitions = arr[locs]

    return transitions, locs


def normalize_transition_matrix(init_matrix, normalize):
    '''
    Normalizes a transition matrix by given criteria.

    Parameters
    ----------
    init_matrix (2D np.array): transition matrix to normalize.
    normalize (str): normalization criteria; ['bigram', 'rows', 'columns', or None]

    Returns
    -------
    init_matrix (2D np.array): normalized transition matrix
    '''
    if normalize is None or normalize not in ('bigram', 'rows', 'columns'):
        return init_matrix

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        if normalize == 'bigram':
            init_matrix /= init_matrix.sum()
        elif normalize == 'rows':
            init_matrix /= init_matrix.sum(axis=1, keepdims=True)
        elif normalize == 'columns':
            init_matrix /= init_matrix.sum(axis=0, keepdims=True)

    return init_matrix


def n_gram_transition_matrix(labels, n=2, max_label=99):
    trans_mat = np.zeros((max_label + 1, ) * n, dtype='float')
    for loc in sliding_window(n, labels):
        if any(l >= max_label for l in loc):
            continue
        trans_mat[loc] += 1
    return trans_mat


# per https://gist.github.com/tg12/d7efa579ceee4afbeaec97eb442a6b72
def get_transition_matrix(labels, max_syllable=99, normalize='bigram',
                          smoothing=0.0, combine=False, disable_output=False) -> list:
    '''
    Compute the transition matrix from a set of model labels.

    Parameters
    ----------
    labels (list of np.array of ints): labels loaded from a model fit
    max_syllable (int): maximum syllable number to consider
    normalize (str): how to normalize transition matrix, 'bigram' or 'rows' or 'columns'
    smoothing (float): constant to add to transition_matrix pre-normalization to smooth counts
    combine (bool): compute a separate transition matrix for each element (False)
    or combine across all arrays in the list (True)
    disable_output (bool): verbosity

    Returns
    -------
    transition_matrix (list or np.ndarray): list of 2d np.arrays that represent the transitions
            from syllable i (row) to syllable j (column) or a single transition matrix combined
            from all sessions in `labels`
    '''
    if not isinstance(labels[0], (list, np.ndarray, pd.Series)):
        labels = [labels]

    # Compute a singular transition matrix
    if combine:
        init_matrix = []

        for v in tqdm(labels, disable=disable_output):
            # Get syllable transitions
            transitions = get_transitions(v)[0]

            trans_mat = n_gram_transition_matrix(transitions, n=2, max_label=max_syllable)
            init_matrix.append(trans_mat)

        init_matrix = np.sum(init_matrix, axis=0) + smoothing
        all_mats = normalize_transition_matrix(init_matrix, normalize)
    else:
        # Compute a transition matrix for each session label list
        all_mats = []
        for v in tqdm(labels, disable=disable_output):
            # Get syllable transitions
            transitions = get_transitions(v)[0]

            trans_mat = n_gram_transition_matrix(transitions, n=2, max_label=max_syllable) + smoothing

            # Normalize matrix
            init_matrix = normalize_transition_matrix(trans_mat, normalize)
            all_mats.append(init_matrix)

    return all_mats


def convert_ebunch_to_graph(ebunch):
    '''
    Convert transition matrices to transition DAGs.

    Parameters
    ----------
    ebunch (list of tuples): syllable transition data

    Returns
    -------
    g (networkx.DiGraph): DAG object to graph
    '''

    g = nx.DiGraph()
    g.add_weighted_edges_from(ebunch)

    return g


def convert_transition_matrix_to_ebunch(weights, transition_matrix,
                                        usages=None, usage_threshold=-.1,
                                        speeds=None, speed_threshold=0,
                                        edge_threshold=-.1, indices=None,
                                        keep_orphans=False, max_syllable=None):
    '''
    Computes thresholded syllable transition data. Function thresholds computed transition
    matrix's usages (included nodes) and edges (syllable transitions).

    Parameters
    ----------
    weights (np.ndarray): syllable transition edge weights
    transition_matrix (np.ndarray): syllable transition matrix
    usages (list): list of syllable usages
    usage_threshold (float): threshold syllable usage to include a syllable in list of orphans
    speeds (1D np.array): list of syllable speeds
    speed_threshold (int): threshold value for syllable speeds to include
    usage_threshold (float or tuple): threshold syllable usage to include a syllable in list of orphans
    edge_threshold (float): threshold transition probability to consider an edge part of the graph.
    indices (list): indices of syllable bigrams to plot
    keep_orphans (bool): indicate whether to graph orphan syllables
    max_syllable (bool): maximum numebr of syllables to include in graph

    Returns
    -------
    ebunch (list): syllable transition data.
    orphans (list): syllables with no edges.
    '''
    # TODO: figure out if I ever need the transition_matrix variable
    # Cap the number of included syllable states
    if max_syllable is not None:
        weights = weights[:max_syllable, :max_syllable]
        transition_matrix = transition_matrix[:max_syllable, :max_syllable]

    def _filter_ebunch(arg):
        _, _, w = arg
        w = abs(w)
        if isinstance(edge_threshold, (list, tuple)):
            return (w > edge_threshold[0]) and (w < edge_threshold[1])
        return w > edge_threshold

    orphans = []

    # u, v, w where w is the weights
    ebunch = list(filter(_filter_ebunch, ((*inds, w) for inds, w in np.ndenumerate(weights))))
    if keep_orphans:
        orphans = list(filter(complement(_filter_ebunch), ((*inds, w) for inds, w in np.ndenumerate(weights))))

    if indices is not None:
        if keep_orphans:
            orphans += list(filter(lambda e: e[:-1] not in indices, ebunch))
        ebunch = list(filter(lambda e: e[:-1] in indices, ebunch))

    def _filter_by_stat(arg, stat, stat_threshold):
        _in, _out, _ = arg
        _in = stat[_in]
        _out = stat[_out]
        if isinstance(stat_threshold, (list, tuple)):
            return ((_in > stat_threshold[0] and _in < stat_threshold[1])
                    and (_out > stat_threshold[0] and _out < stat_threshold[1]))
        return _in > stat_threshold and _out > stat_threshold

    if usages is not None:
        ebunch = list(filter(lambda e: _filter_by_stat(e, usages, usage_threshold), ebunch))
    if speeds is not None:
        ebunch = list(filter(lambda e: _filter_by_stat(e, speeds, speed_threshold), ebunch))

    return ebunch, [o[:-1] for o in orphans]


def get_usage_dict(usages):
    '''
    Convert usages numpy array to an OrderedDict

    Parameters
    ----------
    usages (np.ndarray): list of syllable usages for each group

    Returns
    -------
    usages (OrderedDict): OrderedDict corresponding to each syllable and its usage frame-count
    '''

    if usages is not None and isinstance(usages[0], (list, np.ndarray)):
        usage_dicts = []
        for u in usages:
            d = OrderedDict(enumerate(u))
            usage_dicts.append(d)
        usages = usage_dicts

    return usages


def make_graph(tm, ebunch_anchor, edge_threshold, usages_anchor, speeds=None, speed_threshold=-50):
    '''
    Creates networkx graph DiGraph object given a transition matrix and
    shared graphing metadata.

    Parameters
    ----------
    tm (np.ndarray): syllable transition matrix
    ebunch_anchor (list): list of edge transitions from the first transition matrix
    edge_threshold (float): value to threshold transition edges
    usages_anchor (int): node placement reference value

    Returns
    -------
    graph (nx.DiGraph): generated networkx directed Graph with edge weights == syllable transition probabilities
    '''

    ebunch, _ = convert_transition_matrix_to_ebunch(
        tm, tm, edge_threshold=edge_threshold, usages=usages_anchor, speeds=speeds,
        speed_threshold=speed_threshold, indices=ebunch_anchor, keep_orphans=True, max_syllable=tm.shape[0])

    # get graph from ebunch
    graph = convert_ebunch_to_graph(ebunch)

    return graph


def make_difference_graphs(trans_mats, usages, group, group_names, usages_anchor,
                           widths, pos, ebunch_anchor, node_edge_colors, ax=None, node_sizes=[],
                           difference_threshold=0.0005, difference_edge_width_scale=500, font_size=12,
                           usage_scale=1e5, difference_graphs=[], scalars=None, arrows=False):
    '''
    Helper function that computes transition graph differences.

    Parameters
    ----------
    trans_mats (np.ndarray): syllable transition matrix.
    usages (list): list of syllable usage probabilities.
    group (list): list groups to graph transition graphs for.
    group_names (list): list groups names to display with transition graphs.
    usages_anchor (int): node placement reference value.
    widths (list): list of edge widths for each created single-group graph.
    pos (nx.Layout): nx.Layout type object holding position coordinates for the nodes.
    ebunch_anchor (list): list of transition graph metadata for each node and connected edges
    node_edge_colors (list): node edge colors (of type str).
    node_sizes (list): node size scaling factor (of type int)
    difference_threshold (float): threshold to consider 2 graph elements different.
    difference_edge_width_scale (int): scaling factor for edge widths in difference transition graphs.
    usage_scale (float): syllable usage scaling factor.
    difference_graphs (list): list of created difference transition graphs.

    Returns
    -------
    usages (list): list of syllable usage probabilities including usages differences across groups
    group_names (list): list groups names to display with transition graphs including difference graphs.
    difference_graphs (list): list of computed difference graphs
    widths (list): list of edge widths for each created graph appended with difference weights
    node_sizes (2D list): lists of node sizes corresponding to each graph including difference graph node sizes
    node_edge_colors (2D list): lists of node colors corresponding to each graph including difference graph node sizes
    '''

    for i, tm in enumerate(trans_mats):
        for j, tm2 in enumerate(trans_mats[i + 1:]):
            if len(tm2) == 0:
                continue
            # get graph difference
            df = tm2 - tm

            if isinstance(scalars, dict):
                for key in scalars.keys():
                    if len(scalars[key]) > 0:
                        scalar_df = [j2 - i1 for (j2, i1) in zip(scalars[key][i + 1], scalars[key][i])]
                        scalars[key].append(get_usage_dict([scalar_df])[0])

            # make difference graph
            graph = make_graph(df, ebunch_anchor, difference_threshold, usages_anchor)
            difference_graphs.append(graph)

            # get edge widths
            weight = [np.abs(graph[u][v]['weight']) * difference_edge_width_scale
                      for u, v in graph.edges()]

            edge_colors = ['r' if (graph[u][v]['weight'] * difference_edge_width_scale > 0) else 'b'
                           for u, v in graph.edges()]
            widths.append(weight)

            # Handle node size and coloring
            if usages is not None:
                # get usage difference
                df_usage = [usages[j + i + 1][k] - usages[i][k] for k in range(df.shape[0])]

                usages.append(get_usage_dict([df_usage])[0])

                # get node sizes and colors based on usage differences
                node_size = list(np.abs(df_usage) * usage_scale)[:len(graph.nodes)]
                node_edge_color = ['r' if x > 0 else 'b' for x in df_usage][:len(graph.nodes)]

                node_sizes.append(node_size)
                node_edge_colors.append(node_edge_color)
            else:
                # Set default node display values
                node_size = 400
                node_edge_color = 'r'
                node_sizes.append(node_size)
                node_edge_colors.append(node_edge_color)

            # get difference graph name
            curr_name = f'{group[i + j + 1]} - {group[i]}'
            group_names.append(curr_name)

            if np.array(ax).all() != None:
                draw_graph(graph, curr_name, weight, pos, node_color='w', node_size=node_size,
                           node_edge_colors=node_edge_color, arrows=arrows, edge_colors=edge_colors,
                           font_size=font_size, ax=ax, i=i, j=i+j+1)

    return usages, group_names, difference_graphs, widths, node_sizes, node_edge_colors, scalars


def make_transition_graphs(trans_mats, usages, group, group_names, usages_anchor,
                           pos, ebunch_anchor, orphans, edge_threshold=.0025,
                           difference_threshold=.0005, orphan_weight=0,
                           ax=None, edge_width_scale=100, usage_scale=1e5,
                           difference_edge_width_scale=500,
                           font_size=12, scalars=None, arrows=False):
    '''

    Helper function to create transition matrices for all included groups, as well as their
    difference graphs.

    Parameters
    ----------
    trans_mats (np.ndarray): syllable transition matrix.
    usages (list): list of syllable usage probabilities.
    group (list): list groups to graph transition graphs for.
    group_names (list): list groups names to display with transition graphs.
    usages_anchor (int): node placement reference value.
    pos (nx.Layout): nx.Layout type object holding position coordinates for the nodes.
    ebunch_anchor (list): list of transition graph metadata for each node and connected edges.
    edge_threshold (float): threshold to include edge in graph.
    difference_threshold (float): threshold to consider 2 graph elements different.
    orphans (list): list of nodes with no edges.
    orphan_weight (int): scaling factor to plot orphan node sizes.
    ax (np.ndarray matplotlib.pyplot Axis): Optional axes to plot graphs in
    edge_width_scale (int): edge line width scaling factor.
    usage_scale (float): syllable usage scaling factor.
    scalars (dict): dict of syllable scalar data per transition graph

    Returns
    -------
    usages (list): list of syllable usage probabilities including possible appended difference usages.
    group_names (list): list groups names to display with transition graphs including difference graphs.
    widths (list): list of edge widths for each created graph appended with difference weights.
    node_sizes (2D list): lists of node sizes corresponding to each graph.
    node_edge_colors (2D list): lists of node colors corresponding to each graph including difference graph node sizes.
    graphs (list of nx.DiGraph): list of all group and difference transition graphs.
    '''

    graphs = []
    widths = []
    node_sizes = []
    node_edge_colors = []

    for i, tm in enumerate(trans_mats):
        if isinstance(scalars, dict):
            speeds = scalars['speeds_2d'][i]
        else:
            speeds = None

        # make graph from transition matrix
        graph = make_graph(tm, ebunch_anchor, edge_threshold, usages_anchor, speeds)

        # get edge widths
        assert all(edge not in orphans for edge in graph.edges()), 'graph contains orphans'
        width = [tm[u][v] * edge_width_scale if (u, v) not in orphans else orphan_weight
                 for u, v in graph.edges()]
        widths.append(width)

        # set node sizes according to usages
        if usages is not None:
            node_size = [usages[i][k] * usage_scale for k in pos.keys()][:len(graph.nodes)]
            node_sizes.append(node_size)
        else:
            node_size = 400
            node_sizes.append(node_size)

        # Draw network to matplotlib figure
        if ax is not None:
            draw_graph(graph, group_names, width, pos, node_color='w',
                       node_size=node_size, node_edge_colors='r', arrows=arrows,
                       font_size=font_size, ax=ax, i=i, j=i)

        node_edge_colors.append('r')
        graphs.append(graph)

    # get group difference graphs
    if len(group) > 1:
        usages, group_names, graphs, widths, node_sizes, node_edge_colors, scalars = \
            make_difference_graphs(trans_mats, usages, group, group_names,
            usages_anchor, widths, pos, ebunch_anchor,
            ax=ax,
            node_edge_colors=node_edge_colors,
            node_sizes=node_sizes,
            difference_threshold=difference_threshold,
            difference_edge_width_scale=difference_edge_width_scale,
            difference_graphs=graphs,
            scalars=scalars,
            arrows=arrows,
            font_size=font_size)

    return usages, group_names, widths, node_sizes, node_edge_colors, graphs, scalars


def get_pos(graph_anchor, layout, nnodes):
    '''
    Get node positions in the graph based on the graph anchor
    and a user selected layout.

    Parameters
    ----------
    graph_anchor (nx.Digraph): graph to get node layout for
    layout (str): layout type; ['spring', 'circular', 'spectral', 'graphviz']
    nnodes (int): number of nodes in the graph

    Returns
    -------
    pos (nx layout): computed node position layout
    '''

    if isinstance(layout, str) and layout.lower() == 'spring':
        k = 1.5 / np.sqrt(nnodes)
        pos = nx.spring_layout(graph_anchor, k=k, seed=0)
    elif isinstance(layout, str) and layout.lower() == 'circular':
        pos = nx.circular_layout(graph_anchor)
    elif isinstance(layout, str) and layout.lower() == 'spectral':
        pos = nx.spectral_layout(graph_anchor)
    elif isinstance(layout, (dict, OrderedDict)):
        # user passed pos directly
        pos = layout
    else:
        raise RuntimeError('Did not understand layout type')

    return pos


def draw_graph(graph, groups, width, pos, node_color,
               node_size, node_edge_colors, ax, arrows=False,
               font_size=12, edge_colors='k', i=0, j=0):
    '''
    Draws transition graph to existing matplotlib axes.

    Parameters
    ----------
    graph (nx.DiGraph): list of created nx.DiGraphs converted from transition matrices
    groups (list): list of unique groups included.
    width (2D list): list of edge widths corresponding to each graph's edges.
    pos (nx.Layout): nx.Layout type object holding position coordinates for the nodes.
    node_color (2D list): list of node colors for each graph.
        List item can also be a list corresponding to a color for each node.
    node_sizes (int or 1D list): list of node sizes for each graph.
        List item can also be a list corresponding to a color for each node.
    node_edge_colors (list): list of node edge colors for each graph.
        List item can also be a list corresponding to a color for each node.
    arrows (bool): whether to draw arrow edges
    font_size (int): Node label font size
    ax (mpl.pyplot axis object): axis to draw graphs on.

    Returns
    -------
    '''

    # Draw nodes and edges on matplotlib figure
    nx.draw(graph, pos=pos, with_labels=True, font_size=font_size, alpha=1,
            width=width, edgecolors=node_edge_colors, ax=ax[i][j], cmap='turbo',
            node_size=node_size, node_color=node_color, arrows=arrows,
            edge_color=edge_colors, linewidths=1.5)

    # Set titles
    if groups is not None:
        if isinstance(groups, str):
            ax[i][j].set_title(f'{groups}')
        elif len(groups) > 1:
            ax[i][j].set_title(f'{groups[i]}')


def graph_transition_matrix(trans_mats, usages=None, groups=None,
                            edge_threshold=.0025, anchor=0, usage_threshold=0,
                            layout='spring', edge_width_scale=100, fig=None, ax=None,
                            width_per_group=8, headless=False, difference_threshold=.0005,
                            weights=None, usage_scale=1e4, keep_orphans=False,
                            max_syllable=None, orphan_weight=0, arrows=False, font_size=12,
                            difference_edge_width_scale=500, **kwargs):
    '''
    Creates transition graph plot given a transition matrix and some metadata.

    Parameters
    ----------
    trans_mats (np.ndarray): syllable transition matrix
    usages (list): list of syllable usage probabilities
    groups (list): list groups to graph transition graphs for.
    edge_threshold (float): threshold to include edge in graph
    anchor (int): syllable index as the base syllable
    usage_threshold (int): threshold to include syllable usages
    node_color (str): node colors
    node_edge_color (str): node edge color.
    layout (str): layout format
    edge_width_scale (int): edge line width scaling factor
    node_size (int): node size scaling factor
    fig (pyplot figure): figure to plot to
    ax (pyplot Axes): axes object
    width_per_group (int): graph width scaling factor per group
    headless (bool): exclude first node.
    font_size (int): size of node label text.
    plot_differences (bool): plot difference between group transition matrices
    difference_threshold (float): threshold to consider 2 graph elements different
    difference_edge_width_scale (float): difference graph edge line width scaling factor
    weights (list): list of edge weights
    usage_scale (float): syllable usage scaling factor
    arrows (bool): indicate whether to plot arrows as transitions.
    keep_orphans (bool): plot orphans.
    max_syllable (int): number of syllables (nodes) to plot
    orphan_weight (int): scaling factor to plot orphan node sizes
    edge_color (str): edge color
    kwargs (dict): extra keyword arguments

    Returns
    -------
    fig (pyplot figure): figure containing transition graphs.
    ax (pyplot axis): figure axis object.
    pos (dict): dict figure information.
    '''
    from moseq2_viz.model.util import normalize_usages

    if headless:
        plt.switch_backend('agg')

    # Set weights to transition probabilities
    if weights is None:
        weights = trans_mats

    # Ensure transition matrices are N-dim numpy arrays
    assert isinstance(trans_mats, (np.ndarray, list)), "Transition matrix must be a numpy array or list of arrays"

    # Ensure transition matrices are 2D
    if isinstance(trans_mats, np.ndarray):
        assert trans_mats.ndim == 2, 'Transition matrix needs to be 2 dimensional'
        trans_mats = [trans_mats]

    # Get shared node anchors based on usages
    ngraphs = len(trans_mats)
    usages = [normalize_usages(u) for u in usages]
    anchor = anchor if anchor < len(trans_mats) else 0
    usages_anchor = usages[anchor]

    # Create transition graph metadata from transition matrix
    ebunch_anchor, orphans = convert_transition_matrix_to_ebunch(
        weights[anchor], trans_mats[anchor], edge_threshold=edge_threshold,
        keep_orphans=keep_orphans, usages=usages_anchor,
        usage_threshold=usage_threshold, max_syllable=max_syllable)

    # Create transition graph object
    graph_anchor = convert_ebunch_to_graph(ebunch_anchor)
    nnodes = len(graph_anchor.nodes())

    # Get node position layout
    pos = get_pos(graph_anchor, layout, nnodes)

    # Create figure to plot
    if fig is None or ax is None:
        fig, ax = plt.subplots(ngraphs, ngraphs,
                               figsize=(ngraphs*width_per_group,
                                        ngraphs*width_per_group))

    # Format axis
    if ngraphs == 1:
        ax = [[ax]]

    if isinstance(groups, str):
        groups = [groups]

    # Get group name list to append difference graph names
    group_names = deepcopy(groups)

    # Make graphs and difference graphs
    _ = make_transition_graphs(trans_mats, usages, groups, group_names,
        usages_anchor, pos, ebunch_anchor, orphans,
        difference_edge_width_scale=difference_edge_width_scale, edge_threshold=edge_threshold,
        difference_threshold=difference_threshold, orphan_weight=orphan_weight,
        ax=ax, edge_width_scale=edge_width_scale, usage_scale=usage_scale,
        arrows=arrows, font_size=font_size)

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].axis('off')

    return fig, ax, pos