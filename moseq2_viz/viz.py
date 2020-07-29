'''

Visualization model containing all plotting functions and some dependent data pre-processing helper functions.

'''

import re
import math
import cv2
import h5py
import random
import numpy as np
import seaborn as sns
import networkx as nx
from functools import wraps
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Tuple, Iterable
from cytoolz import pluck
from moseq2_viz.util import star
from matplotlib import lines, gridspec
from networkx.drawing.nx_agraph import graphviz_layout
from moseq2_viz.model.label_util import get_sorted_syllable_stat_ordering, get_syllable_muteness_ordering

def clean_frames(frames, medfilter_space=None, gaussfilter_space=None,
                 tail_filter=None, tail_threshold=5):
    '''
    Filters frames using spatial filters such as Median or Gaussian filters.

    Parameters
    ----------
    frames (3D numpy array): frames to filter.
    medfilter_space (list): list of len()==1, must be odd. Median space filter kernel size.
    gaussfilter_space (list): list of len()==2. Gaussian space filter kernel size.
    tail_filter (int): number of iterations to filter over tail.
    tail_threshold (int): filtering threshold value

    Returns
    -------
    out (3D numpy array): filtered numpy array.
    '''

    out = np.copy(frames)


    if tail_filter is not None:
        for i in range(frames.shape[0]):
            mask = cv2.morphologyEx(out[i], cv2.MORPH_OPEN, tail_filter) > tail_threshold
            out[i] = out[i] * mask.astype(frames.dtype)

    if medfilter_space is not None and np.all(np.array(medfilter_space) > 0):
        for i in range(frames.shape[0]):
            for medfilt in medfilter_space:
                if medfilt % 2 == 0:
                    print('Inputted medfilter must be odd. Subtracting input by 1.')
                    medfilt -= 1
                out[i] = cv2.medianBlur(out[i], medfilt)

    if gaussfilter_space is not None and np.all(np.array(gaussfilter_space) > 0):
        for i in range(frames.shape[0]):
            out[i] = cv2.GaussianBlur(out[i], (21, 21),
                                      gaussfilter_space[0], gaussfilter_space[1])

    return out

def save_fig(fig, output_file, name='{}', **kwargs):
    '''

    Convenience function for saving created/open matplotlib figures to PNG and PDF formats.

    Parameters
    ----------
    fig (pyplot.Figure): open figure to save
    output_file (str): path to save figure to
    name (str): dynamic figure name; allows for overriding name with specific value/prefix
    kwargs (dict): dictionary containing additional figure saving parameters. (check plot-stats in wrappers.py)

    Returns
    -------
    None
    '''

    fig.savefig(f'{name}.png'.format(output_file), **kwargs)
    fig.savefig(f'{name}.pdf'.format(output_file), **kwargs)

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
    blue  = min((max((4*(0.75-x), 0.)), 1.))
    red   = min((max((4*(x-0.25), 0.)), 1.))
    green = min((max((4*math.fabs(x-0.5)-1., 0.)), 1.))
    return red, green, blue


def convert_transition_matrix_to_ebunch(weights, transition_matrix,
                                        usages=None, usage_threshold=-.1,
                                        edge_threshold=-.1, indices=None,
                                        keep_orphans=False, max_syllable=None):
    '''

    Parameters
    ----------
    weights (np.ndarray): syllable transition edge weights
    transition_matrix (np.ndarray): syllable transition matrix
    usages (list): list of syllable usages
    usage_threshold (float): threshold syllable usage to include a syllable in list of orphans
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
        for i, v in np.ndenumerate(transition_matrix):
            if np.abs(v) > edge_threshold:
                ebunch.append((i[0], i[1], weights[i[0], i[1]]))
    elif indices is None and keep_orphans:
        for i, v in np.ndenumerate(transition_matrix):
            ebunch.append((i[0], i[1], weights[i[0], i[1]]))
            if np.abs(v) <= edge_threshold:
                orphans.append((i[0], i[1]))
    elif indices is not None and keep_orphans:
        for i in indices:
            ebunch.append((i[0], i[1], weights[i[0], i[1]]))
            if np.abs(weights[i[0], i[1]]) <= edge_threshold:
                orphans.append((i[0], i[1]))
    else:
        ebunch = [(i[0], i[1], weights[i[0], i[1]]) for i in indices]

    if usages is not None:
        ebunch = [e for e in ebunch if usages[e[0]] > usage_threshold and usages[e[1]] > usage_threshold]

    if max_syllable is not None:
        ebunch = [e for e in ebunch if e[0] <= max_syllable and e[1] <= max_syllable]

    return ebunch, orphans


def graph_transition_matrix(trans_mats, usages=None, groups=None,
                            edge_threshold=.0025, anchor=0, usage_threshold=0,
                            node_color='w', node_edge_color='r', layout='spring',
                            edge_width_scale=100, node_size=400, fig=None, ax=None,
                            width_per_group=8, headless=False, font_size=12,
                            plot_differences=True, difference_threshold=.0005,
                            difference_edge_width_scale=500, weights=None,
                            usage_scale=1e5, arrows=False, keep_orphans=False,
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
    height (int): UNUSED.
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

    if weights is None:
        weights = trans_mats

    assert isinstance(trans_mats, (np.ndarray, list)), "Transition matrix must be a numpy array or list of arrays"

    if isinstance(trans_mats, np.ndarray) and trans_mats.ndim == 2:
        trans_mats = [trans_mats]

    if usages is not None and isinstance(usages[0], (list, np.ndarray)):
        from collections import defaultdict
        for i, u in enumerate(usages):
            d = defaultdict(int)
            for j, v in enumerate(u):
                d[j] = v
            usages[i] = d

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

    ebunch_anchor, orphans = convert_transition_matrix_to_ebunch(
        weights[anchor], trans_mats[anchor], edge_threshold=edge_threshold,
        keep_orphans=keep_orphans, usages=usages_anchor,
        usage_threshold=usage_threshold, max_syllable=max_syllable)

    graph_anchor = convert_ebunch_to_graph(ebunch_anchor)
    nnodes = len(graph_anchor.nodes())

    if type(layout) is str and layout.lower() == 'spring':
        if 'k' not in kwargs.keys():
            kwargs['k'] = 1.5 / np.sqrt(nnodes)
        pos = nx.spring_layout(graph_anchor, **kwargs)
    elif type(layout) is str and layout.lower() == 'circular':
        pos = nx.circular_layout(graph_anchor, **kwargs)
    elif type(layout) is str and layout.lower() == 'spectral':
        pos = nx.spectral_layout(graph_anchor, **kwargs)
    elif type(layout) is str and layout.lower()[:8] == 'graphviz':
        prog = re.split(r'\:', layout.lower())[1]
        pos = graphviz_layout(graph_anchor, prog=prog, **kwargs)
    elif type(layout) is dict:
        # user passed pos directly
        pos = layout
    else:
        raise RuntimeError('Did not understand layout type')

    if fig is None or ax is None:
        fig, ax = plt.subplots(ngraphs, ngraphs,
                               figsize=(ngraphs*width_per_group,
                                        ngraphs*width_per_group))

    if ngraphs == 1:
        ax = [[ax]]

    for i, tm in enumerate(trans_mats):

        ebunch, orphans = convert_transition_matrix_to_ebunch(
            tm, tm, edge_threshold=edge_threshold, indices=ebunch_anchor,
            keep_orphans=keep_orphans)
        graph = convert_ebunch_to_graph(ebunch)
        width = [tm[u][v] * edge_width_scale if (u, v) not in orphans else orphan_weight
                 for u, v in graph.edges()]

        if usages is not None:
            node_size = [usages[i][k] * usage_scale for k in pos.keys()]

        nx.draw_networkx_nodes(graph, pos,
                               edgecolors=node_edge_color, node_color=node_color,
                               node_size=node_size, ax=ax[i][i], cmap='jet')
        nx.draw_networkx_edges(graph, pos, graph.edges(), width=width, ax=ax[i][i],
                               arrows=arrows, edge_color=edge_color)
        if font_size > 0:
            nx.draw_networkx_labels(graph, pos,
                                    {k: k for k in pos.keys()},
                                    font_size=font_size,
                                    ax=ax[i][i], font_color='k')

        if groups is not None:
            ax[i][i].set_title('{}'.format(groups[i]))

    if plot_differences and groups is not None and ngraphs > 1:
        for i, tm in enumerate(trans_mats):
            for j, tm2 in enumerate(trans_mats[i+1:]):
                df = tm2 - tm

                ebunch, _ = convert_transition_matrix_to_ebunch(
                    df, df, edge_threshold=difference_threshold, indices=ebunch_anchor)
                graph = convert_ebunch_to_graph(ebunch)

                weight = [np.abs(graph[u][v]['weight'])*difference_edge_width_scale
                          for u, v in graph.edges()]

                if usages is not None:
                    df_usage = [usages[j + i + 1][k] - usages[i][k] for k in pos.keys()]
                    node_size = list(np.abs(df_usage) * usage_scale)
                    node_edge_color = ['r' if x > 0 else 'b' for x in df_usage]

                nx.draw_networkx_nodes(graph, pos, edgecolors=node_edge_color, node_color=node_color,
                                       node_size=node_size, ax=ax[i][j + i + 1], linewidths=1.5)
                colors = []

                for u, v in graph.edges():
                    if graph[u][v]['weight'] > 0:
                        colors.append('r')
                    else:
                        colors.append('b')

                nx.draw_networkx_edges(graph, pos, graph.edges(),
                                       width=weight, edge_color=colors,
                                       ax=ax[i][j + i + 1], arrows=arrows)

                if font_size > 0:
                    nx.draw_networkx_labels(graph, pos,
                                            {k: k for k in pos.keys()},
                                            font_size=font_size,
                                            ax=ax[i][j + i + 1], font_color='k')

                ax[i][j + i + 1].set_title('{} - {}'.format(groups[j + i + 1], groups[i]))

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].axis('off')

    return fig, ax, pos


def crowd_matrix_from_loaded_data(slices: Iterable[Tuple[int, int]], frames, scalars, nexamples=50,
                                  pad=30, dur_clip=1000, raw_size=(512, 424), crop_size=(80, 80)):
    '''
    This function assumes angles have already been treated for flips, if necessary.
    UNUSED

    Parameters
    ----------
    slices
    frames
    scalars
    nexamples
    pad
    dur_clip
    raw_size
    crop_size

    Returns
    -------
    None
    '''

    def dur_filter(slice_):
        return (slice_[1] - slice_[0]) < dur_clip

    slices = filter(dur_filter, slices)
    slices = random.choices(slices, k=nexamples)
    dur = list(s[1] - s[0] for s in slices)
    max_dur = max(dur)
    starts = map(lambda x: x - pad, pluck(0, slices))

    def pad_idx(idx, dur):
        return idx + pad + (max_dur - dur)

    ends = map(star(pad_idx), zip(pluck(1, slices), dur))
    # turn each tuple of indices into a slice object
    slices = map(star(slice), zip(starts, ends))

    crowd_mtx = np.zeros((max_dur + 2 * pad, *reversed(raw_size)), dtype='uint8')

    yc0, xc0 = [x // 2 for x in crop_size]
    # TODO: finish - add the below stuff


# TODO: add option to render w/ text using opencv (easy, this way we can annotate w/ nu, etc.)
def make_crowd_matrix(slices, nexamples=50, pad=30, raw_size=(512, 424), frame_path='frames',
                      crop_size=(80, 80), dur_clip=1000, offset=(50, 50), scale=1,
                      center=False, rotate=False, min_height=10, legacy_jitter_fix=False,
                      **kwargs):
    '''
    Creates crowd movie video numpy array.

    Parameters
    ----------
    slices (numpy array): video slices of specific syllable label
    nexamples (int): maximum number of mice to include in crowd_matrix video
    pad (int): number of frame padding in video
    raw_size (tuple): video dimensions.
    frame_path (str): path to in-h5 frames variable
    crop_size (tuple): mouse crop size
    dur_clip (int): maximum clip duration.
    offset (tuple): centroid offsets from cropped videos
    scale (int): mouse size scaling factor.
    center (bool): indicate whether mice are centered.
    rotate (bool): rotate mice to orient them.
    min_height (int): minimum max height from floor to use.
    legacy_jitter_fix (bool): whether to apply jitter fix for K1 camera.
    kwargs (dict): extra keyword arguments

    Returns
    -------
    crowd_matrix (3D numpy array): crowd movie for a specific syllable.
    '''

    if rotate and not center:
        raise NotImplementedError('Rotating without centering not supported')

    durs = np.array([i[1]-i[0] for i, j, k in slices])

    if dur_clip is not None:
        idx = np.where(np.logical_and(durs < dur_clip, durs > 0))[0]
        use_slices = [_ for i, _ in enumerate(slices) if i in idx]
    else:
        idx = np.where(durs > 0)[0]
        use_slices = [_ for i, _ in enumerate(slices) if i in idx]

    if len(use_slices) > nexamples:
        use_slices = [use_slices[i] for i in np.random.permutation(np.arange(len(use_slices)))[:nexamples]]

    durs = np.array([i[1]-i[0] for i, j, k in use_slices])

    if len(durs) < 1:
        return None

    max_dur = durs.max()

    if max_dur < 0:
        return None

    crowd_matrix = np.zeros((max_dur + pad * 2, raw_size[1], raw_size[0]), dtype='uint8')

    count = 0

    xc0 = crop_size[1] // 2
    yc0 = crop_size[0] // 2

    xc = np.array(list(range(-xc0, +xc0 + 1)), dtype='int16')
    yc = np.array(list(range(-yc0, +yc0 + 1)), dtype='int16')

    for idx, uuid, fname in use_slices:

        # get the frames, combine in a way that's alpha-aware

        h5 = h5py.File(fname, 'r')
        nframes = h5[frame_path].shape[0]
        cur_len = idx[1] - idx[0]
        use_idx = (idx[0] - pad, idx[1] + pad + (max_dur - cur_len))

        if use_idx[0] < 0 or use_idx[1] >= nframes - 1:
            continue

        if 'centroid_x' in h5['scalars'].keys():
            use_names = ('scalars/centroid_x', 'scalars/centroid_y')
        elif 'centroid_x_px' in h5['scalars'].keys():
            use_names = ('scalars/centroid_x_px', 'scalars/centroid_y_px')

        centroid_x = h5[use_names[0]][use_idx[0]:use_idx[1]] + offset[0]
        centroid_y = h5[use_names[1]][use_idx[0]:use_idx[1]] + offset[1]

        if center:
            centroid_x -= centroid_x[pad]
            centroid_x += raw_size[0] // 2
            centroid_y -= centroid_y[pad]
            centroid_y += raw_size[1] // 2

        angles = h5['scalars/angle'][use_idx[0]:use_idx[1]]
        frames = clean_frames((h5[frame_path][use_idx[0]:use_idx[1]] / scale).astype('uint8'), **kwargs)

        if 'flips' in h5['metadata/extraction'].keys():
            # h5 format as of v0.1.3
            flips = h5['metadata/extraction/flips'][use_idx[0]:use_idx[1]]
            angles[np.where(flips == True)] -= np.pi
        elif 'flips' in h5['metadata'].keys():
            # h5 format prior to v0.1.3
            flips = h5['metadata/flips'][use_idx[0]:use_idx[1]]
            angles[np.where(flips == True)] -= np.pi
        else:
            flips = np.zeros(angles.shape, dtype='bool')

        angles = np.rad2deg(angles)

        for i in range(len(centroid_x)):

            if np.isnan(centroid_x[i]) or np.isnan(centroid_y[i]):
                continue

            rr = (yc + centroid_y[i]).astype('int16')
            cc = (xc + centroid_x[i]).astype('int16')

            if (np.any(rr < 1)
                or np.any(cc < 1)
                or np.any(rr >= raw_size[1])
                or np.any(cc >= raw_size[0])
                or (rr[-1] - rr[0] != crop_size[0])
                or (cc[-1] - cc[0] != crop_size[1])):
                continue

            rot_mat = cv2.getRotationMatrix2D((xc0, yc0), angles[i], 1)
            # old_frame = crowd_matrix[i][rr[0]:rr[-1],
            #                             cc[0]:cc[-1]]
            old_frame = crowd_matrix[i]
            new_frame = np.zeros_like(old_frame)
            new_frame_clip = frames[i]

            # change from fliplr, removes jitter since we now use rot90 in moseq2-extract
            if flips[i] and legacy_jitter_fix:
                new_frame_clip = np.fliplr(new_frame_clip)
            elif flips[i]:
                new_frame_clip = np.rot90(new_frame_clip, k=-2)

            new_frame_clip = cv2.warpAffine(new_frame_clip.astype('float32'),
                                            rot_mat, crop_size).astype(frames.dtype)

            if i >= pad and i <= pad + cur_len:
                cv2.circle(new_frame_clip, (xc0, yc0), 3, (255, 255, 255), -1)
            try:
                new_frame[rr[0]:rr[-1], cc[0]:cc[-1]] = new_frame_clip
            except Exception:
                raise Exception

            if rotate:
                rot_mat = cv2.getRotationMatrix2D((raw_size[0] // 2, raw_size[1] // 2),
                                                  -angles[pad] + flips[pad] * 180,
                                                  1)
                new_frame = cv2.warpAffine(new_frame, rot_mat, raw_size).astype(new_frame.dtype)

            # zero out based on min_height before taking the non-zeros
            new_frame[new_frame < min_height] = 0
            old_frame[old_frame < min_height] = 0

            new_frame_nz = new_frame > 0
            old_frame_nz = old_frame > 0

            blend_coords = np.logical_and(new_frame_nz, old_frame_nz)
            overwrite_coords = np.logical_and(new_frame_nz, ~old_frame_nz)

            old_frame[blend_coords] = .5 * old_frame[blend_coords] + .5 * new_frame[blend_coords]
            old_frame[overwrite_coords] = new_frame[overwrite_coords]

            # crowd_matrix[i][rr[0]:rr[-1], cc[0]:cc[-1]] = old_frame
            crowd_matrix[i] = old_frame

        count += 1

        if count >= nexamples:
            break

    return crowd_matrix


def position_plot(scalar_df, centroid_vars=['centroid_x_mm', 'centroid_y_mm'],
                  sort_vars=['SubjectName', 'uuid'], group_var='group', sz=50, **kwargs):
    '''
    Creates a position summary graph that shows all the
    mice's centroid path throughout the respective sessions.

    Parameters
    ----------
    scalar_df (pandas DataFrame): dataframe containing all scalar data
    centroid_vars (list): list of scalar variables to track mouse position
    sort_vars (list): list of variables to sort the dataframe by.
    group_var (str): groups df column to graph position plots for.
    sz (int): plot size.
    kwargs (dict): extra keyword arguments

    Returns
    -------
    fig (pyplot figure): pyplot figure object
    ax (pyplot axis): pyplot axis object
    '''

    grouped = scalar_df.groupby([group_var] + sort_vars)

    groups = [grp[0] for grp in grouped.groups]
    uniq_groups = list(set(groups))
    count = [len([grp1 for grp1 in groups if grp1 == grp]) for grp in uniq_groups]

    grouped = scalar_df.groupby(group_var)

    figsize = (np.round(2.5 * len(uniq_groups)), np.round(2.6 * np.max(count)))
    lims = (np.min(scalar_df[centroid_vars].min()), np.max(scalar_df[centroid_vars].max()))

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=np.max(count),
                           ncols=len(uniq_groups),
                           width_ratios=[1] * len(uniq_groups),
                           wspace=0.0,
                           hspace=0.0)

    for i, (name, group) in enumerate(grouped):

        group_session = group.groupby(sort_vars)

        for j, (name2, group2) in enumerate(group_session):

            ax = plt.subplot(gs[j, i])
            ax.plot(group2[centroid_vars[0]],
                    group2[centroid_vars[1]],
                    linewidth=.5,
                    **kwargs)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.axis('off')

            if group_var == 'group':
                if j == 0:
                    ax.set_title(name)
            else:
                ax.set_title(name)


            if i == 0 and j == len(group_session) - 1:
                y_line = lines.Line2D([lims[0], lims[0]],
                                      [lims[0], lims[0] + sz],
                                      color='b',
                                      alpha=1)
                x_line = lines.Line2D([lims[0], lims[0] + sz],
                                      [lims[0], lims[0]],
                                      color='b',
                                      alpha=1)
                y_line.set_clip_on(False)
                x_line.set_clip_on(False)
                ax.add_line(y_line)
                ax.add_line(x_line)
                ax.text(lims[0] - 10, lims[0] - 60, '{} CM'.format(np.round(sz / 10).astype('int')))

            ax.set_aspect('auto')

    return fig, ax


def scalar_plot(scalar_df, sort_vars=['group', 'uuid'], group_var='group',
                show_scalars=['velocity_2d_mm', 'velocity_3d_mm',
                              'height_ave_mm', 'width_mm', 'length_mm'],
                headless=False, colors=None, **kwargs):
    '''
    Creates scatter plot of given scalar variables representing extraction results.

    Parameters
    ----------
    scalar_df (pandas DataFrame):
    sort_vars (list): list of variables to sort the dataframe by.
    group_var (str): groups df column to graph position plots for.
    show_scalars (list): list of scalar variables to plot.
    headless (bool): exclude head of dataframe from plot.
    colors (list): list of color strings to indicate groups
    kwargs (dict): extra keyword variables

    Returns
    -------
    fig (pyplot figure): plotted scalar scatter plot
    ax (pyplot axis): plotted scalar axis
    '''
    if headless:
        plt.switch_backend('agg')

    if colors == None:
        colors = sns.color_palette()
    elif len(colors) == 0:
        colors = sns.color_palette()


    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    # sort scalars into a neat summary using group_vars
    grp = scalar_df.groupby(sort_vars)[show_scalars]

    summary = {
        'Mean': grp.mean(),
        'STD': grp.std()
    }

    for i, (k, v) in tqdm(enumerate(summary.items())):
        summary[k].reset_index(level=summary[k].index.names, inplace=True)
        summary[k] = summary[k].melt(id_vars=group_var, value_vars=show_scalars)
        sns.swarmplot(data=summary[k], x='variable', y='value', hue=group_var, ax=ax[i], palette=colors, **kwargs)
        ax[i].set_ylabel(k)
        ax[i].set_xlabel('')

    fig.tight_layout()

    return fig, ax

def check_types(function):
    '''
    Decorator function to validate user input parameters for plotting syllable statistics, facilitated using
    functools wraps

    Parameters
    ----------
    function: plot_syll_stats_with_sem - the function to check parameters from.

    Returns
    -------
    wrapped (function) returns the function to run
    '''

    @wraps(function)
    def wrapped(complete_df, stat='usage', ordering=None, max_sylls=None, groups=None, ctrl_group=None, exp_group=None,
                colors=None, figsize=(10, 5), *args, **kwargs):
        '''
        Wrapper function to validate input parameters and adjust parameters according to any user errors to run the
        plotting function with some respective defaulting parameters.

        Parameters
        ----------
        complete_df (pd.DataFrame): dataframe containing the statistical information about syllable data [usages, durs, etc.]
        stat (str): choice of statistic to plot: either usage, duration, or speed
        ordering (str, list, None): "m" for mutated, f"{stat}" for descending ordering with respect to original usage ordering.
        max_sylls (int): maximum number of syllable to include in plot
        groups (list): list of groups to include in plot. If groups=None, all groups will be plotted.
        ctrl_group (str): name of control group to base mutation sorting on.
        exp_group (str): name of experimental group to base mutation sorting on.
        colors (list): list of user-selected colors to represent the data
        figsize (tuple): tuple value of length = 2, representing (columns x rows) of the plotted figure dimensions
        args
        kwargs

        Returns
        -------
        function: executes function with validated input parameters
        '''

        if not isinstance(figsize, tuple) or isinstance(figsize, list):
            print('Invalid figsize. Input a integer-tuple or list of len(figsize) = 2')
            figsize = (10, 5)

        if groups == None or len(groups) == 0:
            groups = list(set(complete_df.group))
        elif isinstance(groups, str):
            groups = [groups]

        if isinstance(groups, list) or isinstance(groups, tuple):
            uniq_groups = set(complete_df.group)
            if not set(groups).issubset(uniq_groups):
                print('Invalid group entered. Displaying all groups.')
                groups = uniq_groups

        if max_sylls == None:
            max_sylls = 41

        if set(stat).issubset(set('usage')):
            stat = 'usage'
            try:
                if (isinstance(ordering, str) or ordering.any() == None) and ordering != 'm':
                    ordering = range(max_sylls)
            except AttributeError:
                ordering = range(max_sylls)
        else:
            if set(stat).issubset(set('duration')):
                stat = 'duration'
            elif set(stat).issubset(set('speed')):
                stat = 'speed'
            if isinstance(ordering, str) and ordering != 'm':
                if not set(ordering).issubset(set('default')):
                    print(f'Reordering syllables with respect to selected statistic: {stat}')
                    ordering, _ = get_sorted_syllable_stat_ordering(complete_df, stat=stat)
                else:
                    ordering = range(max_sylls)

        if isinstance(ordering, str):
            if ordering == 'm':
                if (ctrl_group != None and exp_group != None) and (ctrl_group in groups and exp_group in groups):
                    max_sylls += 1
                    ordering = get_syllable_muteness_ordering(complete_df, ctrl_group=ctrl_group,
                                                                           exp_group=exp_group, max_sylls=max_sylls,
                                                                           stat=stat)
                else:
                    print('You must enter valid control and experimental group names found in your trained model and index file.\nPlotting descending order.')
                    ordering, _ = get_sorted_syllable_stat_ordering(complete_df, stat=stat)

        if colors == None or len(colors) == 0:
            colors = [None] * len(groups)
        else:
            if len(colors) < len(groups):
                print(f'Number of inputted colors {len(colors)} does not match number of groups {len(groups)}. Using default.')
                colors = [None] * len(groups)

        return function(complete_df, stat=stat, ordering=ordering, max_sylls=max_sylls, groups=groups, colors=colors, figsize=figsize,
                        *args, **kwargs)

    return wrapped

@check_types
def plot_syll_stats_with_sem(complete_df, stat='usage', ordering=None, max_sylls=None, groups=None, ctrl_group=None,
                             exp_group=None, colors=None, fmt='o-', figsize=(10, 5)):
    '''
    Plots a line and/or point-plot of a given pre-computed syllable statistic (usage, duration, or speed),
    with a SEM error bar with respect to the group.
    This function is decorated with the check types function that will ensure that the inputted data configurations
    are safe to plot in matplotlib.

    Parameters
    ----------
    complete_df (pd.DataFrame): dataframe containing the statistical information about syllable data [usages, durs, etc.]
    stat (str): choice of statistic to plot: either usage, duration, or speed
    ordering (str, list, None): "m" for mutated, f"{stat}" for descending ordering with respect to original usage ordering.
    max_sylls (int): maximum number of syllable to include in plot
    groups (list): list of groups to include in plot. If groups=None, all groups will be plotted.
    ctrl_group (str): name of control group to base mutation sorting on.
    exp_group (str): name of experimental group to base mutation sorting on.
    colors (list): list of user-selected colors to represent the data
    fmt (str): str to indicate the kind of plot to make. "o-", "o", "--', etc.
    figsize (tuple): tuple value of length = 2, representing (columns x rows) of the plotted figure dimensions

    Returns
    -------
    fig (pyplot figure): plotted scalar scatter plot
    ax (pyplot axis): plotted scalar axis
    '''

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # separates each group's usage data into a separate array element, and computes their respective group-marginalized SEM
    # also reorders data if using mutant ordering
    shift = -(len(groups) - 1) / 10
    for i, group in tqdm(enumerate(groups), total=len(groups)):
        data_df = complete_df[complete_df['group'] == group][['syllable', stat]].groupby('syllable',
                                                                             as_index=False).mean().reindex(ordering)
        sem = complete_df.groupby('syllable')[[stat]].sem()[:max_sylls].reindex(ordering)
        # plot each group with their corresponding SEM error bars
        plt.errorbar(np.asarray(range(max_sylls)) + shift, data_df[stat].to_numpy()[:max_sylls],
                     yerr=sem[stat][:max_sylls], label=group, fmt=fmt, color=colors[i])
        shift += 0.1

    if stat == 'usage':
        ylabel = 'P(syllable)'
        xlabel = 'usage'
    elif stat == 'duration':
        ylabel = 'Mean Syllable Sequence Frame Duration'
        xlabel = 'duration'
    elif stat == 'speed':
        ylabel = 'Mean Syllable Speed (mm/s)'
        xlabel = 'speed'

    lgd = plt.legend(bbox_to_anchor=(1.1, 1.05),
               ncol=1, fancybox=True, shadow=True, fontsize=16)
    plt.xticks(range(max_sylls), ordering)
    plt.ylabel(ylabel, fontsize=12)
    plt.ylim()
    plt.xlabel(f'Syllable Label (indexed by {xlabel})', fontsize=12)

    sns.despine()

    return fig, lgd

def plot_mean_group_heatmap(pdfs, groups):
    '''
    Computes the overall group mean of the computed PDFs and plots them.

    Parameters
    ----------
    pdfs (list): list of 2d probability density functions (heatmaps) describing mouse position.
    groups (list): list of groups to compute means and plot

    Returns
    -------
    fig (pyplot figure): plotted scalar scatter plot
    '''

    uniq_groups = np.unique(groups)

    fig = plt.figure(figsize=((20, 5)))
    gs = plt.GridSpec(1, len(uniq_groups))

    for i, group in tqdm(enumerate(uniq_groups), total=len(uniq_groups)):
        subplot = fig.add_subplot(gs[i])
        idx = np.array(groups) == group

        im = plt.imshow(pdfs[idx].mean(0) / pdfs[idx].mean(0).max())
        plt.colorbar(im, fraction=0.046, pad=0.04)

        plt.xticks([])
        plt.yticks([])

        subplot.set_title(group, fontsize=14)

    return fig

def plot_verbose_heatmap(pdfs, sessions, groups, subjectNames):
    '''
    Plots the PDF position heatmap for each session, titled with the group and subjectName.

    Parameters
    ----------
    pdfs (list): list of 2d probability density functions (heatmaps) describing mouse position.
    groups (list): list of sessions corresponding to the pdfs indices
    groups (list): list of groups corresponding to the pdfs indices
    subjectNames (list): list of subjectNames corresponding to the pdfs indices

    Returns
    -------
    fig (pyplot figure): plotted scalar scatter plot
    '''

    uniq_groups = np.unique(groups)
    count = [len([grp1 for grp1 in groups if grp1 == grp]) for grp in uniq_groups]
    figsize = (np.round(2.5 * len(uniq_groups)), np.round(2.6 * np.max(count)))

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=np.max(count),
                           ncols=len(uniq_groups),
                           width_ratios=[1] * len(uniq_groups),
                           wspace=0.5,
                           hspace=0.5)

    for i, group in tqdm(enumerate(uniq_groups), total=len(uniq_groups)):
        idx = np.array(groups) == group
        tmp_sessions = np.asarray(sessions)[idx]
        names = np.asarray(subjectNames)[idx]
        for j, sess in enumerate(tmp_sessions):
            idx = np.array(sessions) == sess
            plt.subplot(gs[j, i])

            im = plt.imshow(pdfs[idx].mean(0) / pdfs[idx].mean(0).max())
            plt.colorbar(im, fraction=0.046, pad=0.04)

            plt.xticks([])
            plt.yticks([])

            plt.title(f'{group}: {names[j]}', fontsize=10)

    return fig