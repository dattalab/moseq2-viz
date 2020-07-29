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
from sys import platform
import ruamel.yaml as yaml
from tqdm.auto import tqdm
from functools import wraps
from moseq2_viz.io.video import write_crowd_movies, write_crowd_movie_info_file
from moseq2_viz.util import parse_index, recursive_find_h5s, h5_to_dict, clean_dict
from moseq2_viz.scalars.util import scalars_to_dataframe, compute_mean_syll_speed, \
                                    compute_all_pdf_data, compute_session_centroid_speeds
from moseq2_viz.viz import plot_syll_stats_with_sem, scalar_plot, position_plot, plot_mean_group_heatmap, \
                           plot_verbose_heatmap, save_fig
from moseq2_viz.model.util import relabel_by_usage, parse_model_results,  merge_models, results_to_dataframe,\
                                  compute_and_graph_grouped_TMs

def wrapper_function_setup(function):
    '''

    Decorator for all (but add_group) wrapper functions.
    Function will read function name, and read model, index file and output file parameters
    according to their caller's arg order.
    Additionally, it will create any missing subdirectories for plotting functions.

    Parameters
    ----------
    function (function): function to perform preprocessing for prior to its execution

    Returns
    -------
    function (function): same function with an updated kwargs dictionary containing all required loaded data
    '''

    @wraps(function)
    def wrapped(*args, **kwargs):

        caller = function.__name__
        model_fit, output_file = None, None

        # Handle all scalar related functions
        if any(x in caller for x in ['scalar', 'pdf']):
            index_file, output_file = args[0], args[1]

        # Handle plot syllable summary function
        elif 'stat' in caller:
            model_fit, index_file, output_file = args[0], args[1], args[2]

        # Handle transition_graph and crowd_movie functions
        elif any(x in caller for x in ['crowd', 'transition']):
            index_file, model_fit = args[0], args[1]
            if 'transition' in caller:
                output_file = args[3]
            elif 'crowd' in caller:
                output_dir = args[3]
                # Set up output directory to save crowd movies in
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

        # Set up output directory to save plots in
        if output_file != None:
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))

        # Get sorted index dict
        if index_file != None:
            index, sorted_index = parse_index(index_file)
            kwargs['index'] = index
            kwargs['sorted_index'] = sorted_index

        # Load trained model data
        if model_fit != None:
            # If the user passes model directory, merge model states by
            # minimum distance between them relative to first model in list
            if os.path.isdir(model_fit):
                model_data = merge_models(model_fit, 'p')
            elif model_fit.endswith('.p') or model_fit.endswith('.pz'):
                model_data = parse_model_results(joblib.load(model_fit))
            elif model_fit.endswith('.h5'):
                # TODO: add h5 file model parsing capability
                pass

            kwargs['model_data'] = model_data

        return function(*args, **kwargs)
    return wrapped

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

    # Search for inputted key-value pair and relabel all found instances in index
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

        # Updating index dict groups for each found session
        for uuid, hit in zip(h5_uuids, hits):
            position = h5_uuids.index(uuid)
            if hit:
                index['files'][position]['group'] = config_data['group']

    # Atomically write updated index file
    new_index = '{}_update.yaml'.format(index_file.replace('.yaml', ''))
    try:
        with open(new_index, 'w+') as f:
            yaml.safe_dump(index, f)
        shutil.move(new_index, index_file)
    except Exception:
        raise Exception

    print('Group(s) added successfully.')

@wrapper_function_setup
def plot_scalar_summary_wrapper(index_file, output_file, groupby='group', colors=None, **kwargs):
    '''
    Wrapper function that plots scalar summary graphs.

    Note: function is decorated with function performing initialization operations and saving
    the results in the kwargs variable.

    Decorator will retrieve the sorted_index dict.

    Parameters
    ----------
    index_file (str): path to index file.
    output_file (str): path to save graphs.
    groupby (str): scalar_df column to group sessions by when graphing scalar and position summaries
    colors (list): list of colors to serve as the sns palette in the scalar summary
    kwargs (dict): dict containing index dicts from given index file path.

    Returns
    -------
    scalar_df (pandas DataFrame): df containing scalar data per session uuid.
    (Only accessible through GUI API)
    '''

    # Get loaded index dict via decorator
    sorted_index = kwargs['sorted_index']

    # Parse index dict files to return pandas DataFrame of all computed scalars from extraction step
    scalar_df = scalars_to_dataframe(sorted_index)

    # Plot Scalar Summary with specified groupings and colors
    plt_scalars, _ = scalar_plot(scalar_df, group_var=groupby, colors=colors, headless=True)

    # Plot Position Summary of all mice in columns organized by groups
    plt_position, _ = position_plot(scalar_df, group_var=groupby)

    # Write images to files
    save_fig(plt_scalars, output_file, name='{}_summary')
    save_fig(plt_position, output_file, name='{}_position')

    return scalar_df

@wrapper_function_setup
def plot_syllable_stat_wrapper(model_fit, index_file, output_file, stat='usage', sort=True, count='usage', group=None, max_syllable=40,
                                 fmt='o-', ordering=None, ctrl_group=None, exp_group=None, colors=None, figsize=(10, 5), **kwargs):
    '''
    Wrapper function to plot specified syllable statistic.

    Note: function is decorated with function performing initialization operations and saving
    the results in the kwargs variable.

    Decorator will retrieve the sorted_index dict and parse the model results into a single dict.

    Parameters
    ----------
    model_fit (str): path to trained model file.
    index_file (str): path to index file.
    output_file (str): filename for syllable usage graph.
    stat (str): syllable statistic to plot: ['usage', 'speed', 'duration']
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
    kwargs (dict): dict containing loaded model data and index dicts

    Returns
    -------
    plt (pyplot figure): graph to show in Jupyter Notebook.
    '''

    # Edge case: Accounting for mutation sorting
    max_syllable += 1

    # Get loaded model via decorator
    model_data = kwargs['model_data']

    # Get loaded index dict via decorator
    sorted_index = kwargs['sorted_index']

    compute_labels = False
    if stat == 'speed':
        # Load scalar Dataframe to compute syllable speeds
        scalar_df = scalars_to_dataframe(sorted_index)
        compute_labels = True

    # Compute a syllable summary Dataframe containing usage-based
    # sorted/relabeled syllable usage and duration information from [0, max_syllable] inclusive
    df, label_df = results_to_dataframe(model_data, sorted_index, count=count,
                                        max_syllable=max_syllable, sort=sort, compute_labels=compute_labels)
    if stat == 'speed':
        # Compute each rodent's centroid speed in mm/s
        scalar_df['centroid_speed_mm'] = compute_session_centroid_speeds(scalar_df)

        # Compute the average rodent syllable velocity based on the corresponding centroid speed at each labeled frame
        df = compute_mean_syll_speed(df, scalar_df, label_df, groups=group, max_sylls=max_syllable)

    # Plot and save syllable stat plot
    plt, lgd = plot_syll_stats_with_sem(df, ctrl_group=ctrl_group, exp_group=exp_group, colors=colors, groups=group,
                                      fmt=fmt, ordering=ordering, stat=stat, max_sylls=max_syllable, figsize=figsize)

    # Save
    save_fig(plt, output_file, bbox_extra_artists=(lgd,), bbox_inches='tight')

    return plt

@wrapper_function_setup
def plot_mean_group_position_pdf_wrapper(index_file, output_file, **kwargs):
    '''
    Wrapper function that computes the PDF of the rodent's position throughout the respective sessions,
    and averages these values with respect to their groups to graph a mean position heatmap for each group.

    Note: function is decorated with function performing initialization operations and saving
    the results in the kwargs variable.

    Decorator will retrieve the sorted_index dict.

    Parameters
    ----------
    index_file (str): path to index file.
    output_file (str): filename for the group heatmap graph.
    gui (bool): indicate whether GUI is plotting the graphs.
    kwargs (dict): dict containing loaded index dict from given index file path

    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''

    # Get loaded index dicts via decorator
    sorted_index = kwargs['sorted_index']

    # Load scalar dataframe to compute position PDF heatmap
    scalar_df = scalars_to_dataframe(sorted_index)

    # Compute Position PDF Heatmaps for all sessions
    pdfs, groups, sessions, subjectNames = compute_all_pdf_data(scalar_df)

    # Plot the average Position PDF Heatmap for each group
    fig = plot_mean_group_heatmap(pdfs, groups)

    # Save
    save_fig(fig, output_file)

    return fig

@wrapper_function_setup
def plot_verbose_pdfs_wrapper(index_file, output_file, **kwargs):
    '''
    Wrapper function that computes the PDF for the mouse position for each session in the index file.
    Will plot each session's heatmap with a "SessionName: Group"-like title.

    Note: function is decorated with function performing initialization operations and saving
    the results in the kwargs variable.

    Decorator will retrieve the sorted_index dict.

    Parameters
    ----------
    index_file (str): path to index file.
    output_file (str): filename for the verbose heatmap graph.
    kwargs (dict): dict containing loaded index dict from given index file path

    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''

    # Get loaded index dicts via decorator
    sorted_index = kwargs['sorted_index']

    # Load scalar dataframe to compute position PDF heatmap
    scalar_df = scalars_to_dataframe(sorted_index)

    # Compute PDF Heatmaps for all sessions
    pdfs, groups, sessions, subjectNames = compute_all_pdf_data(scalar_df)

    # Plot all session heatmaps in columns organized by groups
    fig = plot_verbose_heatmap(pdfs, sessions, groups, subjectNames)

    # Save
    save_fig(fig, output_file)

    return fig

@wrapper_function_setup
def plot_transition_graph_wrapper(index_file, model_fit, config_data, output_file, **kwargs):
    '''
    Wrapper function to plot transition graphs.

    Note: function is decorated with function performing initialization operations and saving
    the results in the kwargs variable.

    Decorator will retrieve the sorted_index dict and parse the model results into a single dict.

    Parameters
    ----------
    index_file (str): path to index file
    model_fit (str): path to trained model.
    config_data (dict): dictionary containing the user specified keys and values
    output_file (str): filename for syllable usage graph.
    kwargs (dict): dict containing loaded model data and index dicts

    Returns
    -------
    plt (pyplot figure): graph to show in Jupyter Notebook.
    '''

    # Get loaded model via decorator
    model_data = kwargs['model_data']

    # Get loaded index dicts via decorator
    index, sorted_index = kwargs['index'], kwargs['sorted_index']

    # Optionally load pygraphviz for transition graph layout configuration
    if config_data.get('layout').lower()[:8] == 'graphviz':
        try:
            import pygraphviz
        except ImportError:
            raise ImportError('pygraphviz must be installed to use graphviz layout engines')

    # Set groups to plot
    if config_data.get('group') != None:
        group = config_data['group']

    # Get labels and optionally relabel them by usage sorting
    labels = model_data['labels']
    if config_data['sort']:
        labels = relabel_by_usage(labels, count=config_data['count'])[0]

    # Get modeled session uuids to compute group-mean transition graph for
    if 'train_list' in model_data.keys():
        label_uuids = model_data['train_list']
    else:
        label_uuids = model_data['keys']

    # Loading modeled groups from index file by looking up their session's corresponding uuid
    if 'group' in index['files'][0].keys() and len(group) > 0:
        label_group = [sorted_index['files'][uuid]['group'] \
                           if uuid in sorted_index['files'].keys() else 'default' for uuid in label_uuids]
    else:
        # If no index file is found, set session grouping as nameless default to plot a single transition graph
        label_group = [''] * len(model_data['labels'])
        group = list(set(label_group))

    try:
        # Compute and plot Transition Matrices
        plt = compute_and_graph_grouped_TMs(config_data, labels, label_group, group)
    except:
        print('Incorrectly inputted group, plotting all groups.')

        label_group = [f['group'] for f in sorted_index['files'].values()]
        group = list(set(label_group))

        print('Recomputing transition matrices...')
        plt = compute_and_graph_grouped_TMs(config_data, labels, label_group, group)

    # Save
    save_fig(plt, output_file)

    return plt

@wrapper_function_setup
def make_crowd_movies_wrapper(index_file, model_path, config_data, output_dir, **kwargs):
    '''
    Wrapper function to create crowd movie videos and write them to individual
    files depicting respective syllable labels.

    Note: function is decorated with function performing initialization operations and saving
    the results in the kwargs variable.

    Decorator will retrieve the sorted_index dict and parse the model results into a single dict.

    Parameters
    ----------
    index_file (str): path to index file
    model_path (str): path to trained model.
    config_data (dict): dictionary containing the user specified keys and values
    output_dir (str): directory to store crowd movies in.
    kwargs (dict): dict containing loaded model data and index dicts

    Returns
    -------
    None
    '''

    # Get loaded model via decorator
    model_fit = kwargs['model_data']

    # Get loaded index dicts via decorator
    index, sorted_index = kwargs['index'], kwargs['sorted_index']

    # Get number of CPUs to optimize crowd movie creation and writing speed
    if platform in ['linux', 'linux2']:
        print('Setting CPU affinity to use all CPUs...')
        cpu_count = psutil.cpu_count()
        proc = psutil.Process()
        proc.cpu_affinity(list(range(cpu_count)))

    # Get list of syllable labels for all sessions
    labels = model_fit['labels']

    # Get modeled session uuids
    if 'train_list' in model_fit:
        label_uuids = model_fit['train_list']
    else:
        label_uuids = model_fit['keys']

    # Relabel syllable labels by usage sorting and save the ordering for crowd-movie file naming
    if config_data['sort']:
        labels, ordering = relabel_by_usage(labels, count=config_data['count'])
    else:
        ordering = list(range(config_data['max_syllable']))

    # Uuid in both the labels and the index
    uuid_set = set(label_uuids) & set(sorted_index['files'].keys())

    # Make sure the files exist
    uuid_set = [uuid for uuid in uuid_set if os.path.exists(sorted_index['files'][uuid]['path'][0])]

    # Synchronize arrays such that each label array index corresponds to the correct uuid index
    labels = [label_arr for label_arr, uuid in zip(labels, label_uuids) if uuid in uuid_set]
    label_uuids = [uuid for uuid in label_uuids if uuid in uuid_set]
    sorted_index['files'] = {k: v for k, v in sorted_index['files'].items() if k in uuid_set}

    write_crowd_movie_info_file(model_path=model_path, model_fit=model_fit, index_file=index_file, output_dir=output_dir)

    # Write movies
    write_crowd_movies(sorted_index, config_data, ordering, labels, label_uuids, output_dir)

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