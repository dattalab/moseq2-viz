'''
Wrapper functions for all functionality included in MoSeq2-Viz that is accessible via CLI or GUI.
Each wrapper function executes the functionality from end-to-end given it's dependency parameters are inputted.
(See CLI Click parameters)
'''

import os
import re
import h5py
import shutil
import psutil
import joblib
import numpy as np
from sys import platform
import ruamel.yaml as yaml
from tqdm.auto import tqdm
from moseq2_viz.util import parse_index
from moseq2_viz.io.video import write_crowd_movies
from moseq2_viz.scalars.util import scalars_to_dataframe, compute_mean_syll_speed, compute_all_pdf_data, \
                            compute_session_centroid_speeds
from moseq2_viz.viz import (plot_syll_stats_with_sem, scalar_plot, position_plot, graph_transition_matrix, \
                            plot_mean_group_heatmap, plot_verbose_heatmap)
from moseq2_viz.util import (recursive_find_h5s, check_video_parameters, h5_to_dict, clean_dict)
from moseq2_viz.model.util import (relabel_by_usage, parse_model_results, get_syllable_statistics,
                                   merge_models, get_transition_matrix, results_to_dataframe)

def add_group_wrapper(index_file, config_data):
    '''
    Given a pre-specified key and value, the index file will be updated
    with the respective found keys and values.

    Parameters
    ----------
    index_file (str): path to index file
    config_data (dict): dictionary containing the user specified keys and values

    Returns
    -------
    None
    '''

    index = parse_index(index_file)[0]
    h5_uuids = [f['uuid'] for f in index['files']]
    metadata = [f['metadata'] for f in index['files']]

    value = config_data['value']
    key = config_data['key']

    if type(value) is str:
        value = [value]

    for v in value:
        if config_data['exact']:
            v = r'\b{}\b'.format(v)
        if config_data['lowercase'] and config_data['negative']:
            hits = [re.search(v, meta[key].lower()) is None for meta in metadata]
        elif config_data['lowercase']:
            hits = [re.search(v, meta[key].lower()) is not None for meta in metadata]
        elif config_data['negative']:
            hits = [re.search(v, meta[key]) is None for meta in metadata]
        else:
            hits = [re.search(v, meta[key]) is not None for meta in metadata]

        for uuid, hit in zip(h5_uuids, hits):
            position = h5_uuids.index(uuid)
            if hit:
                index['files'][position]['group'] = config_data['group']

    new_index = '{}_update.yaml'.format(index_file.replace('.yaml', ''))

    try:
        with open(new_index, 'w+') as f:
            yaml.safe_dump(index, f)
        shutil.move(new_index, index_file)
    except Exception:
        raise Exception

    print('Group(s) added successfully.')


def plot_scalar_summary_wrapper(index_file, output_file, groupby='group', colors=None):
    '''
    Wrapper function that plots scalar summary graphs.

    Parameters
    ----------
    index_file (str): path to index file.
    output_file (str): path to save graphs.
    groupby (str): scalar_df column to group sessions by when graphing scalar and position summaries
    colors (list): list of colors to serve as the sns palette in the scalar summary

    Returns
    -------
    scalar_df (pandas DataFrame): df containing scalar data per session uuid.
    (Only accessible through GUI API)
    '''

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    index, sorted_index = parse_index(index_file)
    scalar_df = scalars_to_dataframe(sorted_index)

    plt_scalars, _ = scalar_plot(scalar_df, group_var=groupby, colors=colors, headless=True)
    plt_position, _ = position_plot(scalar_df, group_var=groupby)

    plt_scalars.savefig('{}_summary.png'.format(output_file))
    plt_scalars.savefig('{}_summary.pdf'.format(output_file))

    plt_position.savefig('{}_position.png'.format(output_file))
    plt_position.savefig('{}_position.pdf'.format(output_file))

    return scalar_df

def plot_syllable_usages_wrapper(model_fit, index_file, output_file, sort=True, count='usage', group=None, max_syllable=40,
                                 fmt='o-', ordering=None, ctrl_group=None, exp_group=None, colors=None, figsize=(10, 5)):
    '''
    Wrapper function to plot syllable usages.

    Parameters
    ----------
    model_fit (str): path to trained model file.
    index_file (str): path to index file.
    output_file (str): filename for syllable usage graph.
    sort (bool): sort syllables by usage.
    count (str): method to compute usages 'usage' or 'frames'.
    group (tuple, list, None): tuple or list of groups to separately model usages. (None to graph all groups)
    max_syllable (int): maximum number of syllables to plot.
    fmt (str): scatter plot format. "o-" for line plot with vertices at corresponding usages. "o" for just points.
    ordering (list, range, str, None): order to list syllables. Default is None to graph syllables [0-max_syllable).
     Setting ordering to "m" will graph mutated syllable usage difference between ctrl_group and exp_group.
     None to graph default [0,max_syllable] in order. "usage" to plot descending order of usage values.
    ctrl_group (str): Control group to graph when plotting mutation differences via setting ordering to 'm'.
    exp_group (str): Experimental group to directly compare with control group.
    colors (list): list of colors to serve as the sns palette in the scalar summary. If None, default colors are used.
    figsize (tuple): tuple value of length = 2, representing (columns x rows) of the plotted figure dimensions

    Returns
    -------
    plt (pyplot figure): graph to show in Jupyter Notebook.
    '''

    max_syllable += 1  # accounting for last syllable in list

    # if the user passes model directory, merge model states by
    # minimum distance between them relative to first model in list
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    if os.path.isdir(model_fit):
        model_data = merge_models(model_fit, 'p')
    else:
        model_data = parse_model_results(joblib.load(model_fit))

    # if the user passes multiple groups, sort and plot against each other
    # relabel by usage across the whole dataset, gather usages per session per group
    index, sorted_index = parse_index(index_file)
    df, _ = results_to_dataframe(model_data, sorted_index, max_syllable=max_syllable, sort=sort, count=count)

    plt, lgd = plot_syll_stats_with_sem(df, ctrl_group=ctrl_group, exp_group=exp_group, colors=colors, groups=group,
                                      fmt=fmt, ordering=ordering, stat='usage', max_sylls=max_syllable, figsize=figsize)
    plt.savefig('{}.png'.format(output_file), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig('{}.pdf'.format(output_file), bbox_extra_artists=(lgd,), bbox_inches='tight')

    return plt

def plot_syllable_durations_wrapper(model_fit, index_file, output_file, count='usage', max_syllable=40, sort=True, group=None,
                                    ordering=None, ctrl_group=None, exp_group=None, colors=None, fmt='o-', figsize=(10, 5)):
    '''
    Wrapper function that plots syllable durations.

    Parameters
    ----------
    model_fit (str): path to trained model file.
    index_file (str): path to index file.
    output_file (str): filename for syllable duration graph.
    count (str): method to compute usages 'usage' or 'frames'.
    max_syllable (int): maximum number of syllables to plot.
    sort (bool): sort syllables by usage.
    group (tuple, list, None): tuple or list of groups to separately model usages. (None to graph all groups)
    ordering (list, range, str, None): order to list syllables. Default is None to graph syllables [0-max_syllable).
     Setting ordering to "m" will graph mutated syllable usage difference between ctrl_group and exp_group.
     None to graph default [0,max_syllable] in order. "durations" to plot descending order of duration values.
    ctrl_group (str): Control group to graph when plotting mutation differences via setting ordering to 'm'.
    exp_group (str): Experimental group to directly compare with control group.
    colors (list): list of colors to serve as the sns palette in the scalar summary. If None, default colors are used.
    fmt (str): scatter plot format. "o-" for line plot with vertices at corresponding usages. "o" for just points.
    figsize (tuple): tuple value of length = 2, representing (columns x rows) of the plotted figure dimensions

    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''

    # if the user passes model directory, merge model states by
    # minimum distance between them relative to first model in list
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    if os.path.isdir(model_fit):
        model_data = merge_models(model_fit, 'p')
    else:
        model_data = parse_model_results(joblib.load(model_fit))

    max_syllable += 1  # accounting for last syllable in list

    # if the user passes multiple groups, sort and plot against each other
    # relabel by usage across the whole dataset, gather usages per session per group
    index, sorted_index = parse_index(index_file)
    df, _ = results_to_dataframe(model_data, sorted_index, max_syllable=max_syllable, sort=sort, count=count)
    plt, lgd = plot_syll_stats_with_sem(df, ctrl_group=ctrl_group, exp_group=exp_group, colors=colors, groups=group,
                                      ordering=ordering, fmt=fmt, stat='dur', max_sylls=max_syllable, figsize=figsize)

    plt.savefig('{}.png'.format(output_file), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig('{}.pdf'.format(output_file), bbox_extra_artists=(lgd,), bbox_inches='tight')

    return plt

def plot_syllable_speeds_wrapper(model_fit, index_file, output_file, group=None, ordering=None, colors=None,
                                 ctrl_group=None, exp_group=None, max_syllable=40, fmt='o-', figsize=(10, 5)):
    '''
    Wrapper function that computes the average syllable speed by averaging the speed at all occurrences
    of each syllable in [0, max_syllable) in each session. Then plots the results in the desired ordering.

    Parameters
    ----------
    model_fit (str): path to trained model file.
    index_file (str): path to index file.
    output_file (str): filename for syllable speed graph.
    group (tuple, list, None): tuple or list of groups to separately model usages. (None to graph all groups)
    ordering (list, range, str, None): order to list syllables. Default is None to graph syllables [0-max_syllable).
     Setting ordering to "m" will graph mutated syllable usage difference between ctrl_group and exp_group.
     None to graph default [0,max_syllable] in order. "speeds" to plot descending order of speed values.
    colors (list): list of colors to serve as the sns palette in the scalar summary. If None, default colors are used.
    ctrl_group (str): Control group to graph when plotting mutation differences via setting ordering to 'm'.
    exp_group (str): Experimental group to directly compare with control group.
    max_syllable (int): maximum number of syllables to plot.
    fmt (str): scatter plot format. "o-" for line plot with vertices at corresponding usages. "o" for just points.
    figsize (tuple): tuple value of length = 2, representing (columns x rows) of the plotted figure dimensions

    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    index, sorted_index = parse_index(index_file)
    scalar_df = scalars_to_dataframe(sorted_index)

    if os.path.isdir(model_fit):
        model_data = merge_models(model_fit, 'p')
    else:
        model_data = parse_model_results(joblib.load(model_fit))

    max_syllable += 1  # accounting for last syllable in list

    df, label_df = results_to_dataframe(model_data, sorted_index, max_syllable=max_syllable, sort=True, compute_labels=True)

    scalar_df['centroid_speed_mm'] = compute_session_centroid_speeds(scalar_df)

    df = compute_mean_syll_speed(df, scalar_df, label_df, groups=group, max_sylls=max_syllable)

    fig, lgd = plot_syll_stats_with_sem(df, ordering=ordering, fmt=fmt, stat='speed', max_sylls=max_syllable,
                               groups=group, ctrl_group=ctrl_group, exp_group=exp_group, colors=colors, figsize=figsize)

    fig.savefig('{}.png'.format(output_file), bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig('{}.pdf'.format(output_file), bbox_extra_artists=(lgd,), bbox_inches='tight')

    return fig

def plot_mean_group_position_pdf_wrapper(index_file, output_file):
    '''
    Wrapper function that computes the PDF of the rodent's position throughout the respective sessions,
    and averages these values with respect to their groups to graph a mean position heatmap for each group.

    Parameters
    ----------
    index_file (str): path to index file.
    output_file (str): filename for the group heatmap graph.

    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    index, sorted_index = parse_index(index_file)
    scalar_df = scalars_to_dataframe(sorted_index)

    pdfs, groups, sessions, subjectNames = compute_all_pdf_data(scalar_df)

    fig = plot_mean_group_heatmap(pdfs, groups)

    fig.savefig('{}.png'.format(output_file))
    fig.savefig('{}.pdf'.format(output_file))

    return fig

def plot_verbose_pdfs_wrapper(index_file, output_file):
    '''
    Wrapper function that computes the PDF for the mouse position for each session in the index file.
    Will plot each session's heatmap with a "SessionName: Group"-like title.

    Parameters
    ----------
    index_file (str): path to index file.
    output_file (str): filename for the verbose heatmap graph.

    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    index, sorted_index = parse_index(index_file)
    scalar_df = scalars_to_dataframe(sorted_index)

    pdfs, groups, sessions, subjectNames = compute_all_pdf_data(scalar_df)

    fig = plot_verbose_heatmap(pdfs, sessions, groups, subjectNames)

    fig.savefig('{}.png'.format(output_file))
    fig.savefig('{}.pdf'.format(output_file))

    return fig

def plot_transition_graph_wrapper(index_file, model_fit, config_data, output_file):
    '''
    Wrapper function to plot transition graphs.

    Parameters
    ----------
    index_file (str): path to index file
    model_fit (str): path to trained model.
    config_data (dict): dictionary containing the user specified keys and values
    output_file (str): filename for syllable usage graph.

    Returns
    -------
    plt (pyplot figure): graph to show in Jupyter Notebook.
    '''

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    max_syllable = config_data['max_syllable']

    if config_data.get('layout').lower()[:8] == 'graphviz':
        try:
            import pygraphviz
        except ImportError:
            raise ImportError('pygraphviz must be installed to use graphviz layout engines')

    if config_data.get('group') != None:
        group = config_data['group']

    if os.path.isdir(model_fit):
        model_data = merge_models(model_fit, 'p')
    else:
        model_data = parse_model_results(joblib.load(model_fit))

    index, sorted_index = parse_index(index_file)
    labels = model_data['labels']

    if config_data['sort']:
        labels = relabel_by_usage(labels, count=config_data['count'])[0]

    if 'train_list' in model_data.keys():
        label_uuids = model_data['train_list']
    else:
        label_uuids = model_data['keys']

    label_group = []

    print('Sorting labels...')

    if 'group' in index['files'][0].keys() and len(group) > 0:
        for uuid in label_uuids:
            if uuid in sorted_index['files'].keys():
                label_group.append(sorted_index['files'][uuid]['group'])
            else:
                print('WARNING: UUIDs in model results do not match the index file!')
                label_group.append('default')
    else:
        label_group = [''] * len(model_data['labels'])
        group = list(set(label_group))

    print('Computing transition matrices...')
    try:
        trans_mats = []
        usages = []
        for plt_group in group:
            use_labels = [lbl for lbl, grp in zip(labels, label_group) if grp == plt_group]
            trans_mats.append(get_transition_matrix(use_labels, normalize=config_data['normalize'], combine=True,
                                                    max_syllable=max_syllable))
            usages.append(get_syllable_statistics(use_labels)[0])
        if not config_data['scale_node_by_usage']:
            usages = None

        print('Creating plot...')

        plt, _, _ = graph_transition_matrix(trans_mats, usages=usages, width_per_group=config_data['width_per_group'],
                                            edge_threshold=config_data['edge_threshold'],
                                            edge_width_scale=config_data['edge_scaling'],
                                            difference_edge_width_scale=config_data['edge_scaling'],
                                            keep_orphans=config_data['keep_orphans'],
                                            orphan_weight=config_data['orphan_weight'], arrows=config_data['arrows'],
                                            usage_threshold=config_data['usage_threshold'],
                                            layout=config_data['layout'], groups=group,
                                            usage_scale=config_data['node_scaling'], headless=True)
        plt.savefig('{}.png'.format(output_file))
        plt.savefig('{}.pdf'.format(output_file))
    except:
        print('Incorrectly inputted group, plotting all groups.')

        label_group = [f['group'] for f in sorted_index['files'].values()]
        group = list(set(label_group))

        print('Recomputing transition matrices...')

        trans_mats = []
        usages = []
        for plt_group in group:
            use_labels = [lbl for lbl, grp in zip(labels, label_group) if grp == plt_group]
            trans_mats.append(get_transition_matrix(use_labels, normalize=config_data['normalize'], combine=True,
                                                    max_syllable=max_syllable))
            usages.append(get_syllable_statistics(use_labels)[0])

        plt, _, _ = graph_transition_matrix(trans_mats, usages=usages, width_per_group=config_data['width_per_group'],
                                            edge_threshold=config_data['edge_threshold'],
                                            edge_width_scale=config_data['edge_scaling'],
                                            difference_edge_width_scale=config_data['edge_scaling'],
                                            keep_orphans=config_data['keep_orphans'],
                                            orphan_weight=config_data['orphan_weight'], arrows=config_data['arrows'],
                                            usage_threshold=config_data['usage_threshold'],
                                            layout=config_data['layout'], groups=group,
                                            usage_scale=config_data['node_scaling'], headless=True)
        plt.savefig('{}.png'.format(output_file))
        plt.savefig('{}.pdf'.format(output_file))

    return plt

def make_crowd_movies_wrapper(index_file, model_path, config_data, output_dir):
    '''
    Wrapper function to create crowd movie videos and write them to individual
    files depicting respective syllable labels.

    Parameters
    ----------
    index_file (str): path to index file
    model_path (str): path to trained model.
    config_data (dict): dictionary containing the user specified keys and values
    output_dir (str): directory to store crowd movies in.

    Returns
    -------
    None
    '''

    max_syllable = config_data['max_syllable']
    max_examples = config_data['max_examples']

    if platform in ['linux', 'linux2']:
        print('Setting CPU affinity to use all CPUs...')
        cpu_count = psutil.cpu_count()
        proc = psutil.Process()
        proc.cpu_affinity(list(range(cpu_count)))

    clean_params = {
        'gaussfilter_space': config_data['gaussfilter_space'],
        'medfilter_space': config_data['medfilter_space']
    }

    # need to handle h5 intelligently here...

    if model_path.endswith('.p') or model_path.endswith('.pz'):
        model_fit = parse_model_results(joblib.load(model_path))
        labels = model_fit['labels']

        if 'train_list' in model_fit:
            label_uuids = model_fit['train_list']
        else:
            label_uuids = model_fit['keys']
    elif model_path.endswith('.h5'):
        # load in h5, use index found using another function
        pass
    elif os.path.isdir(model_path):
        model_fit = merge_models(model_path, 'p')
        labels = model_fit['labels']
        if 'train_list' in model_fit:
            label_uuids = model_fit['train_list']
        else:
            label_uuids = model_fit['keys']


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    info_parameters = ['model_class', 'kappa', 'gamma', 'alpha']
    info_dict = {k: model_fit['model_parameters'][k] for k in info_parameters}

    # convert numpy dtypes to their corresponding primitives
    for k, v in info_dict.items():
        if isinstance(v, (np.ndarray, np.generic)):
            info_dict[k] = info_dict[k].item()

    info_dict['model_path'] = model_path
    info_dict['index_path'] = index_file
    info_file = os.path.join(output_dir, 'info.yaml')

    with open(info_file, 'w+') as f:
        yaml.safe_dump(info_dict, f)

    if config_data['sort']:
        labels, ordering = relabel_by_usage(labels, count=config_data['count'])
    else:
        ordering = list(range(max_syllable))

    index, sorted_index = parse_index(index_file)
    vid_parameters = check_video_parameters(sorted_index)

    # uuid in both the labels and the index
    uuid_set = set(label_uuids) & set(sorted_index['files'].keys())

    # make sure the files exist
    uuid_set = [uuid for uuid in uuid_set if os.path.exists(sorted_index['files'][uuid]['path'][0])]

    # harmonize everything...
    labels = [label_arr for label_arr, uuid in zip(labels, label_uuids) if uuid in uuid_set]
    label_uuids = [uuid for uuid in label_uuids if uuid in uuid_set]
    sorted_index['files'] = {k: v for k, v in sorted_index['files'].items() if k in uuid_set}

    if vid_parameters['resolution'] is not None:
        config_data['raw_size'] = vid_parameters['resolution']

    if config_data['sort']:
        filename_format = 'syllable_sorted-id-{:d} ({})_original-id-{:d}.mp4'
    else:
        filename_format = 'syllable_{:d}.mp4'

    write_crowd_movies(sorted_index, config_data, filename_format, vid_parameters, clean_params, ordering,
                       labels, label_uuids, max_syllable, max_examples, output_dir)

def copy_h5_metadata_to_yaml_wrapper(input_dir, h5_metadata_path):
    '''
    Copy h5 metadata dictionary contents into the respective file's yaml file.

    Parameters
    ----------
    input_dir (str): path to directory that contains h5 files.
    h5_metadata_path (str): path to data within h5 file to update yaml with.

    Returns
    -------
    None
    '''

    h5s, dicts, yamls = recursive_find_h5s(input_dir)
    to_load = [(tmp, yml, file) for tmp, yml, file in zip(
        dicts, yamls, h5s) if tmp['complete'] and not tmp['skip']]

    # load in all of the h5 files, grab the extraction metadata, reformat to make nice 'n pretty
    # then stage the copy

    for i, tup in tqdm(enumerate(to_load), total=len(to_load), desc='Copying data to yamls'):
        with h5py.File(tup[2], 'r') as f:
            tmp = clean_dict(h5_to_dict(f, h5_metadata_path))
            tup[0]['metadata'] = dict(tmp)

        try:
            new_file = '{}_update.yaml'.format(os.path.basename(tup[1]))
            with open(new_file, 'w+') as f:
                yaml.safe_dump(tup[0], f)
            shutil.move(new_file, tup[1])
        except Exception:
            raise Exception