'''

Syllable transition graph creation and utility functions.

'''

import re
import math
import tqdm
import numpy as np
import networkx as nx
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import OrderedDict
from networkx.drawing.nx_agraph import graphviz_layout

def get_trans_graph_groups(model_fit, index, sorted_index):
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
    label_uuids (list): list of corresponding UUIDs for each included session
    '''

    if 'train_list' in model_fit.keys():
        label_uuids = model_fit['train_list']
    else:
        label_uuids = model_fit['keys']

    # Loading modeled groups from index file by looking up their session's corresponding uuid
    if 'group' in index['files'][0].keys():
        label_group = [sorted_index['files'][uuid]['group'] \
                           if uuid in sorted_index['files'].keys() else '' for uuid in label_uuids]
        group = list(set(label_group))
    else:
        # If no index file is found, set session grouping as nameless default to plot a single transition graph
        label_group = [''] * len(model_fit['labels'])
        group = list(set(label_group))

    return group, label_group, label_uuids

def get_group_trans_mats(labels, label_group, group, max_sylls):
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

    # Importing within function to avoid
    from moseq2_viz.model.util import get_syllable_statistics

    trans_mats = []
    usages = []

    # Computing transition matrices for each given group
    for plt_group in group:
        # Get sessions to include in trans_mat
        use_labels = [lbl for lbl, grp in zip(labels, label_group) if grp == plt_group]
        trans_mats.append(get_transition_matrix(use_labels,
                                                normalize='bigram',
                                                combine=True,
                                                max_syllable=max_sylls - 1))

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

    trans_mats, usages = get_group_trans_mats(labels, label_group, group, config_data['max_syllable'])

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

def _get_transitions(label_sequence):
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

def normalize_matrix(init_matrix, normalize):
    '''
    Normalizes a transition matrix by given criteria.

    Parameters
    ----------
    init_matrix (2D np.array): transition matrix to normalize.
    normalize (str): normalization criteria; ['bigram', 'rows', 'columns']

    Returns
    -------
    init_matrix (2D np.array): normalized transition matrix
    '''

    if normalize == 'bigram':
        init_matrix /= init_matrix.sum()
    elif normalize == 'rows':
        init_matrix /= init_matrix.sum(axis=1, keepdims=True)
    elif normalize == 'columns':
        init_matrix /= init_matrix.sum(axis=0, keepdims=True)
    else:
        pass

    return init_matrix

# per https://gist.github.com/tg12/d7efa579ceee4afbeaec97eb442a6b72
def get_transition_matrix(labels, max_syllable=100, normalize='bigram',
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
    transition_matrix (list): list of 2d np.arrays that represent the transitions
            from syllable i (row) to syllable j (column)
    '''

    # Compute a singular transition matrix
    if combine:
        init_matrix = np.zeros((max_syllable + 1, max_syllable + 1), dtype='float32') + smoothing

        for v in labels:
            # Get syllable transitions
            transitions = _get_transitions(v)[0]

            # Populate matrix array with transition data
            for (i, j) in zip(transitions, transitions[1:]):
                if i <= max_syllable and j <= max_syllable:
                    init_matrix[i, j] += 1

            init_matrix = normalize_matrix(init_matrix, normalize)

        all_mats = init_matrix
    else:
        # Compute a transition matrix for each session label list
        all_mats = []
        for v in tqdm.tqdm(labels, disable=disable_output):
            init_matrix = np.zeros((max_syllable + 1, max_syllable + 1), dtype='float32') + smoothing

            # Get syllable transitions
            transitions = _get_transitions(v)[0]

            # Populate matrix array with transition data
            for (i, j) in zip(transitions, transitions[1:]):
                if i <= max_syllable and j <= max_syllable:
                    init_matrix[i, j] += 1

            # Normalize matrix
            init_matrix = normalize_matrix(init_matrix, normalize)
            all_mats.append(init_matrix)

    return all_mats

def convert_ebunch_to_graph(ebunch):
    '''
    Convert transition matrices to tranistion DAGs.

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

def floatRgb(mag, cmin, cmax):
    '''
    Return a tuple of floats between 0 and 1 for R, G, and B.

    Parameters
    ----------
    mag (float): color intensity.
    cmin (float): minimum color value
    cmax (float): maximum color value

    Returns
    -------
    red (float): red value
    green (float): green value
    blue (float): blue value
    '''

    # Normalize to 0-1
    try: x = float(mag-cmin)/(cmax-cmin)
    except ZeroDivisionError: x = 0.5 # cmax == cmin

    blue = min((max((4*(0.75-x), 0.)), 1.))
    red = min((max((4*(x-0.25), 0.)), 1.))
    green = min((max((4*math.fabs(x-0.5)-1., 0.)), 1.))

    return red, green, blue


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
    indices (list): indices of syllables to list as orphans
    keep_orphans (bool): indicate whether to graph orphan syllables
    max_syllable (bool): maximum numebr of syllables to include in graph

    Returns
    -------
    ebunch (list): syllable transition data.
    orphans (list): syllables with no edges.
    '''

    ebunch = []
    orphans = []

    if indices is None and not keep_orphans:
        # Do not keep orphaned nodes to display
        for i, v in np.ndenumerate(transition_matrix):
            if isinstance(edge_threshold, tuple):
                # Add node pair if transition probability is within threshold range
                if np.abs(v) >= edge_threshold[0] and np.abs(v) <= edge_threshold[1]:
                    ebunch.append((i[0], i[1], weights[i[0], i[1]]))
            else:
                # Add node pair if transition probability is larger than edge threshold value
                if np.abs(v) > edge_threshold:
                    ebunch.append((i[0], i[1], weights[i[0], i[1]]))
    elif indices is None and keep_orphans:
        # Keep orphaned nodes to display
        for i, v in np.ndenumerate(transition_matrix):
            if isinstance(edge_threshold, tuple):
                # Add node pair + weight if transition prob. is within edge threshold range
                if np.abs(v) >= edge_threshold[0] and np.abs(v) <= edge_threshold[1]:
                    ebunch.append((i[0], i[1], weights[i[0], i[1]]))
                    # Add out-of-range/orphaned nodes
                    if np.abs(v) < edge_threshold[0] and np.abs(v) > edge_threshold[1]:
                        orphans.append((i[0], i[1]))
            else:
                # Add node pair if transition probability is larger than edge threshold value
                if np.abs(v) > edge_threshold:
                    ebunch.append((i[0], i[1], weights[i[0], i[1]]))
                # Add out-of-range/orphaned nodes
                if np.abs(v) <= edge_threshold:
                    orphans.append((i[0], i[1]))
    elif indices is not None and keep_orphans:
        # Keep orphaned nodes to display
        for i in indices:
            if isinstance(edge_threshold, tuple):
                # Add node pair + weight if weight is within edge threshold range
                if np.abs(weights[i[0], i[1]]) >= edge_threshold[0] and np.abs(weights[i[0], i[1]]) <= edge_threshold[1]:
                    ebunch.append((i[0], i[1], weights[i[0], i[1]]))
                    # Add out-of-range/orphaned nodes
                    if np.abs(weights[i[0], i[1]]) < edge_threshold[0] and np.abs(weights[i[0], i[1]]) > edge_threshold[1]:
                        orphans.append((i[0], i[1]))
            else:
                # Add node pair if transition probability is larger than edge threshold value
                if np.abs(weights[i[0], i[1]]) > edge_threshold:
                    ebunch.append((i[0], i[1], weights[i[0], i[1]]))
                # Add out-of-range/orphaned nodes
                if np.abs(weights[i[0], i[1]]) <= edge_threshold:
                    orphans.append((i[0], i[1]))
    else:
        # Adding all node pairs in included indices
        ebunch = [(i[0], i[1], weights[i[0], i[1]]) for i in indices]

    if usages is not None:
        if isinstance(usage_threshold, tuple):
            # Add nodes if their usage value is within the usage threshold range.
            ebunch = [e for e in ebunch if ((usages[e[0]] >= usage_threshold[0]) and (usages[e[0]] <= usage_threshold[1])) and
                      ((usages[e[1]] >= usage_threshold[0]) and (usages[e[1]] <= usage_threshold[1]))]
        else:
            # Add nodes if their usage value is larger than the usage threshold value.
            ebunch = [e for e in ebunch if usages[e[0]] > usage_threshold and usages[e[1]] > usage_threshold]

    if speeds is not None:
        if isinstance(speed_threshold, tuple):
            ebunch = [e for e in ebunch if
                      ((speeds[e[0]] >= speed_threshold[0]) and (speeds[e[0]] <= speed_threshold[1])) and
                      ((speeds[e[1]] >= speed_threshold[0]) and (speeds[e[1]] <= speed_threshold[1]))]
        else:
            ebunch = [e for e in ebunch if speeds[e[0]] > speed_threshold and speeds[e[1]] > speed_threshold]

    # Cap the number of included syllable states
    if max_syllable is not None:
        ebunch = [e for e in ebunch if e[0] <= max_syllable and e[1] <= max_syllable]

    return ebunch, orphans


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
        for i, u in enumerate(usages):
            d = OrderedDict()
            for j, v in enumerate(u):
                d[j] = v
            usages[i] = d

    return usages


def handle_graph_layout(trans_mats, usages, anchor):
    '''
    Computes node usage "anchors"/positions.

    Parameters
    ----------
    trans_mats (np.ndarray): syllable transition matrix
    usages (OrderedDict): OrderedDict of syllable usage counts
    anchor (int): Center of the transition graph

    Returns
    -------
    usages (OrderedDict): OrderedDict of syllable usage ratios
    anchor (int): Center of the transition graph
    usages_anchor (int): node placement reference value
    ngraphs (int): number of total transition matrices
    '''

    ngraphs = len(trans_mats)

    if anchor > ngraphs:
        print('Setting anchor to 0')
        anchor = 0

    if usages is not None:
        for i in range(len(usages)):
            usage_total = sum(usages[i].values())
            for k, v in usages[i].items():
                usages[i][k] = v / usage_total
        usages_anchor = usages[anchor]
    else:
        usages_anchor = None

    return usages, anchor, usages_anchor, ngraphs

def make_graph(tm, ebunch_anchor, edge_threshold, usages_anchor, speeds=None, speed_threshold=-50):
    '''
    Creates networkx graph DiGraph object given a transition matrix and
    shared graphing metadata.

    Parameters
    ----------
    tm (np.ndarray): syllable transition matrix
    ebunch_anchor (list):
    edge_threshold (float): value to threshold transition edges
    usages_anchor (int): node placement reference value

    Returns
    -------
    graph (nx.DiGraph):
    '''

    ebunch, orphans = convert_transition_matrix_to_ebunch(
        tm, tm, edge_threshold=edge_threshold, usages=usages_anchor, speeds=speeds,
        speed_threshold=speed_threshold, indices=ebunch_anchor, keep_orphans=True, max_syllable=tm.shape[0])

    # get graph from ebunch
    graph = convert_ebunch_to_graph(ebunch)

    return graph

def make_difference_graphs(trans_mats, usages, group, group_names, usages_anchor,
                           widths, pos, ebunch_anchor, node_edge_colors, ax=None, node_sizes=[],
                           difference_threshold=0.0005, difference_edge_width_scale=500,
                           usage_scale=1e5, difference_graphs=[], scalars=None, speed_threshold=-50):
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
            # get graph difference
            df = tm2 - tm

            if isinstance(scalars, dict):
                if len(scalars['speeds']) > 0:
                    speed_df = [j2 - i1 for (j2, i1) in zip(scalars['speeds'][i + 1], scalars['speeds'][i])]
                    scalars['speeds'].append(get_usage_dict([speed_df])[0])
                else:
                    speed_df = None

                if len(scalars['dists']) > 0:
                    dist_df = [j2 - i1 for (j2, i1) in zip(scalars['dists'][i + 1], scalars['dists'][i])]
                    scalars['dists'].append(get_usage_dict([dist_df])[0])
                else:
                    dist_df = None

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
                node_size = list(np.abs(df_usage) * usage_scale)
                node_edge_color = ['r' if x > 0 else 'b' for x in df_usage]

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
                draw_graphs(graph, curr_name, weight, pos, node_color='w', node_size=node_size,
                node_edge_colors=node_edge_color, arrows=False, edge_colors=edge_colors,
                font_size=12, ax=ax, i=i, j=i+1)

    return usages, group_names, difference_graphs, widths, node_sizes, node_edge_colors, scalars

def make_transition_graphs(trans_mats, usages, group, group_names, usages_anchor,
                           pos, ebunch_anchor, edge_threshold,
                           difference_threshold, orphans, orphan_weight,
                           ax=None, edge_width_scale=100, usage_scale=1e5,
                           scalars=None, speed_threshold=-15):
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
    speed_threshold (int): value to threshold syllable states at.

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
            speeds = scalars['speeds'][i]
        else:
            speeds = None

        # make graph from transition matrix
        graph = make_graph(tm, ebunch_anchor, edge_threshold, usages_anchor, speeds)

        # get edge widths
        width = [tm[u][v] * edge_width_scale if (u, v) not in orphans else orphan_weight
                 for u, v in graph.edges()]
        widths.append(width)

        # set node sizes according to usages
        if usages is not None:
            node_size = [usages[i][k] * usage_scale for k in pos.keys()]
            node_sizes.append(node_size)
        else:
            node_size = 400
            node_sizes.append(node_size)

        # Draw network to matplotlib figure
        if np.array(ax).all() != None:
            draw_graphs(graph, group_names, width, pos, node_color='w',
                        node_size=node_size, node_edge_colors='r', arrows=False,
                        font_size=12, ax=ax, i=i, j=i)

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
            difference_graphs=graphs,
            scalars=scalars,
            speed_threshold=speed_threshold)

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

    if type(layout) is str and layout.lower() == 'spring':
        k = 1.5 / np.sqrt(nnodes)
        pos = nx.spring_layout(graph_anchor, k=k)
    elif type(layout) is str and layout.lower() == 'circular':
        pos = nx.circular_layout(graph_anchor)
    elif type(layout) is str and layout.lower() == 'spectral':
        pos = nx.spectral_layout(graph_anchor)
    elif type(layout) is str and layout.lower()[:8] == 'graphviz':
        prog = re.split(r'\:', layout.lower())[1]
        pos = graphviz_layout(graph_anchor, prog=prog)
    elif type(layout) is dict:
        # user passed pos directly
        pos = layout
    else:
        raise RuntimeError('Did not understand layout type')

    return pos

def draw_graphs(graph, groups, width, pos, node_color,
                node_size, node_edge_colors, arrows, font_size,
                ax, edge_colors='k', i=0, j=0):
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

    print(pos)

    # Draw nodes and edges on matplotlib figure
    nx.draw(graph, pos=pos, with_labels=True,
            edgecolors=node_edge_colors, ax=ax[i][j], cmap='jet',
            node_size=node_size, node_color=node_color, arrows=arrows,
            edge_color=edge_colors, linewidths=1.5)

    # Set titles
    if groups is not None:
        if isinstance(groups, str):
            ax[i][j].set_title( '{}'.format(groups))
        elif len(groups) > 1:
            ax[i][j].set_title( '{}'.format(groups[i]))


def graph_transition_matrix(trans_mats, usages=None, groups=None,
                            edge_threshold=.0025, anchor=0, usage_threshold=0,
                            node_color='w', node_edge_color='r', layout='spring',
                            edge_width_scale=100, node_size=400, fig=None, ax=None,
                            width_per_group=8, headless=False, font_size=12,
                            plot_differences=True, difference_threshold=.0005,
                            difference_edge_width_scale=500, weights=None,
                            usage_scale=1e4, arrows=False, keep_orphans=False,
                            max_syllable=None, orphan_weight=0, edge_color='k', **kwargs):
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

    if headless:
        plt.switch_backend('agg')

    # Set weights to transition probabilities
    if weights is None:
        weights = trans_mats

    # Ensure transition matrices are N-dim numpy arrays
    assert isinstance(trans_mats, (np.ndarray, list)), "Transition matrix must be a numpy array or list of arrays"

    # Ensure transition matrices are 2D
    if isinstance(trans_mats, np.ndarray) and trans_mats.ndim == 2:
        trans_mats = [trans_mats]

    # Convert usages np.ndarray to OrderedDict
    usages = get_usage_dict(usages)

    # Get shared node anchors based on usages
    usages, anchor, usages_anchor, ngraphs = handle_graph_layout(trans_mats, usages, anchor)

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
        groups = list(groups)

    # Get group name list to append difference graph names
    group_names = groups.copy()

    # Make graphs and difference graphs
    usages, group_names, widths, node_sizes, node_edge_colors, graphs, _ = \
        make_transition_graphs(trans_mats, usages, groups, group_names,
        usages_anchor, pos, ebunch_anchor, edge_threshold,
        difference_threshold, orphans, orphan_weight,
        ax, edge_width_scale, usage_scale)

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].axis('off')

    return fig, ax, pos