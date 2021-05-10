'''

Visualization model containing all plotting functions and some dependent data pre-processing helper functions.

'''

import os
import cv2
import h5py
import warnings
import numpy as np
import seaborn as sns
import matplotlib as mpl
from tqdm.auto import tqdm
from os.path import dirname
from scipy.stats import mode
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from moseq2_viz.model.util import sort_syllables_by_stat, sort_syllables_by_stat_difference


def _validate_and_order_syll_stats_params(complete_df, stat='usage', ordering='stat', max_sylls=40, groups=None, ctrl_group=None, exp_group=None,
            colors=None, figsize=(10, 5)):
    '''
    Validates input parameters and adjust parameters according to any user errors to run the
    plotting function with some respective defaulting parameters. Also orders syllable labels
    based on the average `stat` values per syllable.

    Parameters
    ----------
    complete_df (pd.DataFrame): dataframe containing the statistical information about syllable data [usages, durs, etc.]
    stat (str): choice of statistic to plot: either usage, duration, or speed
    ordering (str, list, None): "stat" for sorting syllables by their average `stat`. "diff" for sorting syllables by
        the difference in `stat` between `exp_group` and `ctrl_group`. If a list, the user should supply
        the order of syllable labels to plot. If None, the original syllable IDs are used.
    max_sylls (int): maximum number of syllable to include in plot
    groups (list): list of groups to include in plot. If groups=None, all groups will be plotted.
    ctrl_group (str): name of control group to base mutation sorting on.
    exp_group (str): name of experimental group to base mutation sorting on.
    colors (list): list of user-selected colors to represent the data
    figsize (tuple): tuple value of length = 2, representing (height x width) of the plotted figure dimensions

    Returns
    -------
    ordering (1D list): list of syllable indices to display on x-axis
    groups (1D list): list of unique groups to plot
    colors (1D list): list of unique colors for each plotted group
    figsize (tuple): plotted figure size (height, width)
    '''

    if not isinstance(figsize, (tuple, list)):
        print('Invalid figsize. Input a integer-tuple or list of len(figsize) = 2. Setting figsize to (10, 5)')
        figsize = (10, 5)

    unique_groups = complete_df['group'].unique()

    if groups is None or len(groups) == 0:
        groups = unique_groups
    elif isinstance(groups, str):
        groups = [groups]

    if isinstance(groups, (list, tuple, np.ndarray)):
        diff = set(groups) - set(unique_groups)
        if len(diff) > 0:
            warnings.warn(f'Invalid group(s) entered: {", ".join(diff)}. Using all groups: {", ".join(unique_groups)}.')
            groups = unique_groups

    if stat.lower() not in complete_df.columns:
        raise ValueError(f'Invalid stat entered: {stat}. Must be a column in the supplied dataframe.')

    if ordering is None:
        ordering = np.arange(max_sylls)
    elif ordering == "stat":
        ordering, _ = sort_syllables_by_stat(complete_df, stat=stat, max_sylls=max_sylls)
    elif ordering == "diff":
        if ctrl_group is None or exp_group is None or not np.all(np.isin([ctrl_group, exp_group], groups)):
            raise ValueError(f'Attempting to sort by {stat} differences, but {ctrl_group} or {exp_group} not in {groups}.')
        ordering = sort_syllables_by_stat_difference(complete_df, ctrl_group, exp_group,
                                                     max_sylls=max_sylls, stat=stat)
    if colors is None:
        colors = []
    if len(colors) == 0 or len(colors) != len(groups):
        if len(colors) != len(groups):
            warnings.warn(f'Number of inputted colors {len(colors)} does not match number of groups {len(groups)}. Using default.')
        colors = sns.color_palette(n_colors=len(groups))

    return ordering, groups, colors, figsize


def clean_frames(frames, medfilter_space=None, gaussfilter_space=None,
                 tail_filter=None, tail_threshold=5):
    '''
    Filters frames using spatial filters such as Median or Gaussian filters.

    Parameters
    ----------
    frames (3D numpy array): frames to filter.
    medfilter_space (list): list of len()==1, must be odd. Median space filter kernel size.
    gaussfilter_space (list): list of len()==2. Gaussian space filter kernel size.
    tail_filter (cv2.getStructuringElement): structuringElement to filter out mouse tails.
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
                    warnings.warn(f'medfilter_space kernel must be odd. Reducing {medfilt} to {medfilt - 1}')
                    medfilt -= 1
                out[i] = cv2.medianBlur(out[i], medfilt)

    if gaussfilter_space is not None and np.all(np.array(gaussfilter_space) > 0):
        for i in range(frames.shape[0]):
            out[i] = cv2.GaussianBlur(out[i], (21, 21),
                                      gaussfilter_space[0], gaussfilter_space[1])

    return out


def save_fig(fig, output_file, suffix=None, **kwargs):
    '''
    Convenience function for saving created/open matplotlib figures to PNG and PDF formats.

    Parameters
    ----------
    fig (pyplot.Figure): open figure to save
    output_file (str): path to save figure to (without extension)
    suffix (str): string to append to the end of output_file
    kwargs (dict): dictionary containing additional figure saving parameters. (check plot-stats in wrappers.py)

    Returns
    -------
    None
    '''

    os.makedirs(dirname(output_file), exist_ok=True)

    if suffix is not None:
        output_file = output_file + suffix

    fig.savefig(f'{output_file}.png', **kwargs)
    fig.savefig(f'{output_file}.pdf', **kwargs)


def make_crowd_matrix(slices, nexamples=50, pad=30, raw_size=(512, 424), frame_path='frames',
                      crop_size=(80, 80), max_dur=60, min_dur=0, offset=(50, 50), scale=1,
                      center=False, rotate=False, min_height=10, legacy_jitter_fix=False,
                      **kwargs):
    '''
    Creates crowd movie video numpy array.

    Parameters
    ----------
    slices (np.ndarray): video slices of specific syllable label
    nexamples (int): maximum number of mice to include in crowd_matrix video
    pad (int): number of frame padding in video
    raw_size (tuple): video dimensions.
    frame_path (str): variable to access frames in h5 file
    crop_size (tuple): mouse crop size
    max_dur (int or None): maximum syllable duration.
    min_dur (int): minimum syllable duration.
    offset (tuple): centroid offsets from cropped videos
    scale (int): mouse size scaling factor.
    center (bool): indicate whether mice are centered.
    rotate (bool): rotate mice to orient them.
    min_height (int): minimum max height from floor to use.
    legacy_jitter_fix (bool): whether to apply jitter fix for K1 camera.
    kwargs (dict): extra keyword arguments

    Returns
    -------
    crowd_matrix (np.ndarray): crowd movie for a specific syllable.
    '''

    if rotate and not center:
        raise NotImplementedError('Rotating without centering not supported')

    xc0, yc0 = crop_size[1] // 2, crop_size[0] // 2
    xc = np.arange(-xc0, xc0 + 1, dtype='int16')
    yc = np.arange(-yc0, yc0 + 1, dtype='int16')

    durs = np.array([i[1]-i[0] for i, _, _ in slices])

    if max_dur is not None:
        idx = np.where(np.logical_and(durs < max_dur, durs > min_dur))[0]
        use_slices = [_slice for i, _slice in enumerate(slices) if i in idx]
    else:
        max_dur = durs.max()
        idx = np.where(durs > min_dur)[0]
        use_slices = [_slice for i, _slice in enumerate(slices) if i in idx]

    if len(use_slices) > nexamples:
        use_slices = np.random.permutation(use_slices)[:nexamples]

    if len(use_slices) == 0 or max_dur < 0:
        return None

    crowd_matrix = np.zeros((max_dur + pad * 2, raw_size[1], raw_size[0]), dtype='uint8')

    for idx, _, fname in use_slices:
        use_idx = (idx[0] - pad, idx[0] + max_dur + pad)
        idx_slice = slice(*use_idx)

        # get the frames, combine in a way that's alpha-aware
        with h5py.File(fname, 'r') as h5:
            nframes = len(h5[frame_path])

            if 'centroid_x' in h5['scalars']:
                use_names = ('scalars/centroid_x', 'scalars/centroid_y')
            elif 'centroid_x_px' in h5['scalars']:
                use_names = ('scalars/centroid_x_px', 'scalars/centroid_y_px')

            if use_idx[0] < 0 or use_idx[1] >= nframes - 1:
                continue

            centroid_x = h5[use_names[0]][idx_slice] + offset[0]
            centroid_y = h5[use_names[1]][idx_slice] + offset[1]

            if center:
                centroid_x -= centroid_x[pad]
                centroid_x += raw_size[0] // 2
                centroid_y -= centroid_y[pad]
                centroid_y += raw_size[1] // 2

            angles = h5['scalars/angle'][idx_slice]
            frames = clean_frames((h5[frame_path][idx_slice] / scale).astype('uint8'), **kwargs)

            if 'flips' in h5['metadata/extraction']:
                # h5 format as of v0.1.3
                flips = h5['metadata/extraction/flips'][idx_slice]
                angles[np.where(flips == True)] -= np.pi
            elif 'flips' in h5['metadata']:
                # h5 format prior to v0.1.3
                flips = h5['metadata/flips'][idx_slice]
                angles[np.where(flips == True)] -= np.pi
            else:
                flips = np.zeros(angles.shape, dtype='bool')

        angles = np.rad2deg(angles)

        for i in range(len(centroid_x)):

            if np.any(np.isnan([centroid_x[i], centroid_y[i]])):
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

            old_frame = crowd_matrix[i]
            new_frame = np.zeros_like(old_frame)
            new_frame_clip = frames[i].copy()

            # change from fliplr, removes jitter since we now use rot90 in moseq2-extract
            if flips[i] and legacy_jitter_fix:
                new_frame_clip = np.fliplr(new_frame_clip)
            elif flips[i]:
                new_frame_clip = np.rot90(new_frame_clip, k=-2)

            new_frame_clip = cv2.warpAffine(new_frame_clip.astype('float32'),
                                            rot_mat, crop_size).astype(frames.dtype)

            if i >= pad and i <= pad + (idx[1] - idx[0]):
                cv2.circle(new_frame_clip, (xc0, yc0), 3, (255, 255, 255), -1)
            
            new_frame[rr[0]:rr[-1], cc[0]:cc[-1]] = new_frame_clip

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

            crowd_matrix[i] = old_frame

    return crowd_matrix


def position_plot(scalar_df, centroid_vars=['centroid_x_mm', 'centroid_y_mm'],
                  sort_vars=['SubjectName', 'uuid'], group_var='group', plt_kwargs=dict(linewidth=1)):
    '''
    Creates a position summary graph that shows all the
    mice's centroid path throughout the respective sessions.

    Parameters
    ----------
    scalar_df (pandas DataFrame): dataframe containing all scalar data
    centroid_vars (list): list of scalar variables to track mouse position
    sort_vars (list): list of variables to sort the dataframe by.
    group_var (str): groups df column to graph position plots for.
    plt_kwargs (dict): extra keyword arguments for plt.plot

    Returns
    -------
    fig (pyplot figure): matplotlib figure object
    ax (pyplot axis): matplotlib axis object
    g (sns.FacetGrid): FacetGrid object the data was plotted with
    '''

    assert len(centroid_vars) == 2, 'must supply 2 centroid vars (x, y) to plot position'

    if isinstance(sort_vars, str) :
        sort_vars = [sort_vars]
    else:
        sort_vars = list(sort_vars)

    scalar_df = scalar_df.sort_values(by=[group_var] + sort_vars)

    if 'uuid' in sort_vars:
        uuid_map = scalar_df.groupby('uuid').first()
    
    g = sns.FacetGrid(data=scalar_df, col='uuid', col_wrap=5, height=2.5, hue=group_var)
    g.map(plt.plot, centroid_vars[0], centroid_vars[1], **plt_kwargs)
    g.set_titles(template='{col_name}')
    for i, a in enumerate(g.axes.flat):
        a.set_title(f"{uuid_map.iloc[i]['SubjectName']}\n{uuid_map.iloc[i]['SessionName']}", fontsize=8)
        a.set_aspect('equal')
    g.fig.tight_layout()
    g.add_legend()

    return g.fig, g.axes, g


def scalar_plot(scalar_df, sort_vars=['group', 'uuid'], group_var='group',
                show_scalars=['velocity_2d_mm', 'velocity_3d_mm',
                              'height_ave_mm', 'width_mm', 'length_mm'],
                headless=False, colors=None, plt_kwargs=dict(height=2, aspect=0.8)):
    '''
    Creates scatter plot of given scalar variables representing extraction results.

    Parameters
    ----------
    scalar_df (pandas DataFrame): dataframe containing scalar data.
    sort_vars (list): list of variables to sort the dataframe by.
    group_var (str): groups scalar plots into separate distributions.
    show_scalars (list): list of scalar variables to plot.
    headless (bool): exclude head of dataframe from plot.
    colors (list): list of color strings to indicate groups
    plt_kwargs (dict): extra arguments for the swarmplot

    Returns
    -------
    fig (pyplot figure): plotted scalar scatter plot
    ax (pyplot axis): plotted scalar axis
    '''
    if headless:
        plt.switch_backend('agg')

    if colors is None or len(colors) == 0:
        colors = sns.color_palette()

    plt_kwargs['aspect'] = 0.6 * len(scalar_df[group_var].unique())

    # sort scalars into a neat summary using group_vars
    summary = scalar_df.groupby(sort_vars)[show_scalars].aggregate(['mean', 'std']).reset_index()
    summary = summary.melt(id_vars=group_var, value_vars=show_scalars)
    groups = summary[group_var].unique()
    
    g = sns.FacetGrid(data=summary, row='variable_0', col='variable_1', sharey=False,
                      hue=group_var, hue_order=groups, palette=colors, **plt_kwargs)
    g.map(sns.swarmplot, group_var, 'value', order=groups)
    g.set_titles(template='{col_name}')
    for row_name, a in zip(show_scalars, g.axes[:, 0]):
        a.set_ylabel(row_name)

    # rotate x-axis labels
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    g.fig.tight_layout()
    plt.show()

    return g.fig, g.axes


def plot_syll_stats_with_sem(scalar_df, syll_info=None, sig_sylls=None, stat='usage', ordering='stat', max_sylls=40,
                             groups=None, ctrl_group=None, exp_group=None, colors=None, join=False, figsize=(10, 5)):
    '''
    Plots a line and/or point-plot of a given pre-computed syllable statistic (usage, duration, or speed),
    with a SEM error bar with respect to the group.
    This function is decorated with the check types function that will ensure that the inputted data configurations
    are safe to plot in matplotlib.

    Parameters
    ----------
    scalar_df (pd.DataFrame): dataframe containing the statistical information about syllable data [usages, durs, etc.]
    syll_info (dict): dictionary of syllable numbers mapped to dict containing the label, description and crowd movie path.
     If provided, will add x-tick markers with the labels for each syllable.
    sig_sylls (1d list): List of syllable numbers that are statistically significant to optionally mark in the graph.
    stat (str): choice of statistic to plot: either usage, duration, or speed
    ordering (str, list, None): "stat" for sorting syllables by their average `stat`. "diff" for sorting syllables by
        the difference in `stat` between `exp_group` and `ctrl_group`. If a list, the user should supply
        the order of syllable labels to plot. If None, the original syllable IDs are used.
    max_sylls (int): maximum number of syllable to include in plot. default: 40
    groups (list): list of groups to include in plot. If groups=None, all groups will be plotted.
    ctrl_group (str): name of control group to base mutation sorting on.
    exp_group (str): name of experimental group to base mutation sorting on.
    colors (list): list of user-selected colors to represent the data
    join (bool): flag to connect points of pointplot
    figsize (tuple): tuple value of length = 2, representing (columns x rows) of the plotted figure dimensions

    Returns
    -------
    fig (pyplot figure): plotted scalar scatter plot
    legend (pyplot legend): figure legend
    '''

    xlabel = f'Syllables sorted by {stat}'
    if ordering == 'diff':
        xlabel += ' difference'
    ordering, groups, colors, figsize = _validate_and_order_syll_stats_params(scalar_df,
                                                                              stat=stat,
                                                                              ordering=ordering,
                                                                              max_sylls=max_sylls,
                                                                              groups=groups,
                                                                              ctrl_group=ctrl_group,
                                                                              exp_group=exp_group,
                                                                              colors=colors,
                                                                              figsize=figsize)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # plot each group's stat data separately, computes groupwise SEM, and orders data based on the stat/ordering parameters
    hue = 'group' if groups is not None else None
    ax = sns.pointplot(data=scalar_df, x='syllable', y=stat, hue=hue, order=ordering,
                       join=join, dodge=True, ci=68, ax=ax, hue_order=groups,
                       palette=colors)

    # where some data has already been plotted to ax
    handles, labels = ax.get_legend_handles_labels()

    # add syllable labels if they exist
    if syll_info is not None:
        mean_xlabels = []
        for o in (ordering):
            mean_xlabels.append(f'{syll_info[o]["label"]} - {o}')

        plt.xticks(range(max_sylls), mean_xlabels, rotation=90)

    # if a list of significant syllables is given, mark the syllables above the x-axis
    if sig_sylls is not None:
        markings = []
        for s in sig_sylls:
            markings.append(ordering.index(s))
        plt.scatter(markings, [-.005] * len(markings), color='r', marker='*')

        # manually define a new patch
        patch = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                              markersize=9, label='Significant Syllable')
        handles.append(patch)

    # add legend and axis labels
    legend = ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1, 1))
    plt.xlabel(xlabel, fontsize=12)
    sns.despine()

    return fig, legend


def plot_mean_group_heatmap(pdfs, groups, normalize=True, norm_color=mpl.colors.LogNorm()):
    '''
    Computes the overall group mean of the computed PDFs and plots them.

    Parameters
    ----------
    pdfs (list): list of 2d probability density functions (heatmaps) describing mouse position.
    groups (list): list of groups to compute means and plot
    normalize (bool): flag to normalize the pdfs between 0-1

    Returns
    -------
    fig (pyplot figure): plotted scalar scatter plot
    '''

    uniq_groups = np.unique(groups)
    groups = np.array(groups)
    pdfs = np.array(pdfs)

    fig, ax = plt.subplots(nrows=len(uniq_groups), ncols=1, sharex=True, sharey=True,
                           figsize=(5, 6 * len(uniq_groups)))
    if not isinstance(ax, np.ndarray):
        ax = np.array(ax)
    for a, group in zip(ax.flat, uniq_groups):
        idx = groups == group
        avg_hist = pdfs[idx].mean(axis=0)
        if normalize:
            _min_val = (avg_hist[avg_hist > 0]).min()
            avg_hist = (avg_hist + _min_val) / (avg_hist.max() + _min_val)

        im = a.imshow(avg_hist, norm=norm_color)
        # fraction to make the colorbar match image height
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, format='%.0e')

        a.set_xticks([])
        a.set_yticks([])
        a.set_title(group, fontsize=14)

    return fig


def plot_verbose_heatmap(pdfs, sessions, groups, subjectNames, normalize=False, norm_color=mpl.colors.LogNorm()):
    '''
    Plots the PDF position heatmap for each session, titled with the group and subjectName.

    Parameters
    ----------
    pdfs (list): list of 2d probability density functions (heatmaps) describing mouse position.
    sessions (list): list of sessions corresponding to the pdfs indices
    groups (list): list of groups corresponding to the pdfs indices
    subjectNames (list): list of subjectNames corresponding to the pdfs indices
    normalize (bool): flag to normalize the pdfs between 0-1

    Returns
    -------
    fig (pyplot figure): plotted scalar scatter plot
    '''

    uniq_groups = np.unique(groups)
    count = [len([grp1 for grp1 in groups if grp1 == grp]) for grp in uniq_groups]
    figsize = (np.round(2.8 * len(uniq_groups)), np.round(2.9 * np.max(count)))

    fig, ax = plt.subplots(nrows=np.max(count), ncols=len(uniq_groups), sharex=True,
                           sharey=True, figsize=figsize)

    if not isinstance(ax, np.ndarray):
        ax = np.array([[ax]])
    if ax.ndim < 2:
        ax = ax[:, None]
    for i, group in enumerate(tqdm(uniq_groups)):
        idx = np.array(groups) == group
        tmp_sessions = np.asarray(sessions)[idx]
        names = np.asarray(subjectNames)[idx]
        for sess, name, a in zip(tmp_sessions, names, ax[:, i]):
            idx = np.array(sessions) == sess

            avg_hist = pdfs[idx].mean(axis=0)

            if normalize:
                _min_val = (avg_hist[avg_hist > 0]).min()
                avg_hist = (avg_hist + _min_val) / (avg_hist.max() + _min_val)

            im = a.imshow(avg_hist, norm=norm_color)
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, format='%.0e')

            a.set_xticks([])
            a.set_yticks([])

            a.set_title(f'{group}: {name}', fontsize=10)

    return fig


def plot_cp_comparison(model_results, pc_cps, plot_all=False, best_model=None, bw_adjust=0.4):
    '''
    Plot the duration distributions for model labels and
    principal component changepoints.

    Parameters
    ----------
    model_cps (dict): Multiple parsed model results aggregated into a single dict.
    pc_cps (1D np.array): Computed PC changepoints
    plot_all (bool): Plot all model changepoints for all keys included in model_cps dict.
    best_model (str): key name to the model with the closest median syllable duration
    bw_adjust (float): fraction to modify bandwith of kernel density estimate. (lower = higher definition)

    Returns
    -------
    fig (pyplot figure): syllable usage ordered by frequency, 90% usage marked
    ax (pyplot axis): plotted scalar axis
    '''

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # Plot KDEs
    ax = sns.kdeplot(pc_cps, color='orange', label='PCA Changepoints', ax=ax, bw_adjust=bw_adjust)

    _mdl = model_results[best_model]
    model_cps = _mdl['changepoints']
    kappa = _mdl['model_parameters']['kappa']

    if not plot_all and best_model is not None:
        ax = sns.kdeplot(model_cps, ax=ax, color='blue', label=f'Model Changepoints Kappa={kappa:1.02E}',
                         bw_adjust=bw_adjust)
    else:
        palette = sns.color_palette('dark', n_colors=len(model_results))
        for i, (k, v) in enumerate(model_results.items()):
            # Set default curve formatting
            ls, alpha = '--', 0.5
            if k == best_model:
                ls, alpha = '-', 1 # Solid line for best fit

            kappa = v['model_parameters']['kappa']
            ax = sns.kdeplot(v['changepoints'], ax=ax, linestyle=ls, alpha=alpha, bw_adjust=bw_adjust,
                        color=palette[i], label=f'Model Changepoints Kappa={kappa:1.02E}')
    # Format plot
    ax.set_xlim(0, 2)

    # Plot best model description
    s = f'Best Model CP Stats: Mean, median, mode (s) = {np.nanmean(model_cps):.4f},' \
        f' {np.nanmedian(model_cps):.4f}, {mode(model_cps)[0][0]:.4f}'
    # Plot PC CP description
    t = f'PC CP Stats: Mean, median, mode (s) = {np.nanmean(pc_cps):.4f}, ' \
        f'{np.nanmedian(pc_cps):.4f}, {mode(pc_cps)[0][0]:.4f}'

    ax.text(0.5, 1.8, s, fontsize=12)
    ax.text(0.5, 1.6, t, fontsize=12)
    ax.set_xlabel('Block duration (s)')
    ax.set_ylabel('Probability density')
    ax.legend(frameon=False, bbox_to_anchor=(1, 0), loc='lower left')
    sns.despine()

    return fig, ax

def plot_group_violin_plots(data_df, stat='velocity_2d_mm', order=None, figsize=(10, 7)):
    '''
    Plots violin plots for each group/column provided in the data_df.

    Parameters
    ----------
    data_df (pd.DataFrame): DataFrame containing N rows x M columns, where N represents the number of examples
     and each column M is a unique group.
    stat (str): Name of the statistic being plotted.
    order (1D list or None): list to specify the order of the violin plots.
    figsize (2-tuple): tuple to indicate the outputted figure size.

    Returns
    -------
    fig (pyplot figure): returned violin plot figure.
    ax (pyplot axis): plotted figure axis object.
    '''

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # draw violin plots
    ax = sns.violinplot(data=data_df, order=order)

    # format axes
    plt.xticks(rotation=45)
    plt.ylabel(stat)

    # format plot
    plt.tight_layout()
    sns.despine()

    return fig, ax