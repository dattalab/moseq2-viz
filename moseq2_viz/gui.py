'''

GUI front-end operations. This module contains all the functionality and configurable parameters
users can alter to most accurately process their data.

Note: These functions perform jupyter notebook specific pre-processing, loads in corresponding parameters from the
CLI functions, then call the corresponding wrapper function with the given input parameters.

'''

import os
import ruamel.yaml as yaml
from bokeh.io import  output_notebook
from .cli import plot_transition_graph
from moseq2_viz.helpers.wrappers import add_group_wrapper, plot_syllable_stat_wrapper, \
    plot_scalar_summary_wrapper, plot_transition_graph_wrapper, copy_h5_metadata_to_yaml_wrapper, \
    plot_verbose_pdfs_wrapper, plot_mean_group_position_pdf_wrapper
from moseq2_viz.helpers.i_wrappers import (interactive_group_setting_wrapper, interactive_syllable_labeler_wrapper,
                                           interactive_syllable_stat_wrapper)


def get_groups_command(index_file):
    '''
    Jupyter Notebook to print index file current metadata groupings.

    Parameters
    ----------
    index_file (str): path to index file

    Returns
    -------
    (int): number of unique groups
    '''


    with open(index_file, 'r') as f:
        index_data = yaml.safe_load(f)
    f.close()

    groups, uuids = [], []
    subjectNames, sessionNames = [], []
    for f in index_data['files']:
        if f['uuid'] not in uuids:
            uuids.append(f['uuid'])
            groups.append(f['group'])
            subjectNames.append(f['metadata']['SubjectName'])
            sessionNames.append(f['metadata']['SessionName'])

    print('Total number of unique subject names:', len(set(subjectNames)))
    print('Total number of unique session names:', len(set(sessionNames)))
    print('Total number of unique groups:', len(set(groups)))

    for i in range(len(subjectNames)):
        print('Session Name:', sessionNames[i], '; Subject Name:', subjectNames[i], '; group:', groups[i])

    return len(set(groups))

def add_group(index_file, by='SessionName', value='default', group='default', exact=False, lowercase=False, negative=False):
    '''
    Updates index file SubjectName group names with user defined group names.

    Parameters
    ----------
    index_file (str): path to index file
    value (str): SessionName value to search for
    group (str): group name to allocate.
    exact (bool): indicate whether to search for exact match.
    lowercase (bool): indicate whether to convert all searched for names to lowercase.
    negative (bool): whether to update the inverse of the found selection.

    Returns
    -------
    None
    '''

    if isinstance(value, str):
        gui_data = {
            'key': by,
            'value': value,
            'group': group,
            'exact': exact,
            'lowercase': lowercase,
            'negative': negative
        }
        add_group_wrapper(index_file, gui_data)

    elif isinstance(value, list) and isinstance(group, list):
        if len(value) == len(group):
            for v, g in zip(value, group):
                gui_data = {
                    'key': by,
                    'value': v,
                    'group': g,
                    'exact': exact,
                    'lowercase': lowercase,
                    'negative': negative
                }
                add_group_wrapper(index_file, gui_data)
        else:
            print('ERROR, did not enter equal number of substring values -> groups.')
    get_groups_command(index_file)

def interactive_group_setting(index_file):
    '''

    Notebook function that calls the wrapper.

    Parameters
    ----------
    index_file (str): Path to index file.

    Returns
    -------
    '''

    interactive_group_setting_wrapper(index_file)

def label_syllables(progress_paths, max_syllables=None, n_explained=90):
    '''
    Parses notebook progress dictionary to load required input parameters and
    launch the interactive GUI.

    Parameters
    ----------
    progress_paths (dict): Notebook progress dict containing paths to required function inputs.
    max_syllables (int or None): Manually inputted number of syllables to label. (default = None)
    n_explained (int): Explained variance percentage of dataset to use to select maximum number of syllables

    Returns
    -------
    '''

    # Get required input paths
    model_path = progress_paths['model_path']
    config_file = progress_paths['config_file']
    index_file = progress_paths['index_file']
    crowd_dir = progress_paths['crowd_dir']
    syll_info_path = progress_paths['syll_info']

    # Launch the GUI
    output_notebook()
    interactive_syllable_labeler_wrapper(model_path, config_file,
                                         index_file, crowd_dir, syll_info_path,
                                         max_syllables=max_syllables, n_explained=n_explained)

def interactive_syllable_stats(progress_paths, max_syllable=None):
    '''
    Parses notebook progress paths to load input required input parameters to
     launch the interactive syllable statistics explorer GUI.

    Parameters
    ----------
    progress_paths (dict): Notebook progress dict containing paths to required function inputs.
    max_syllables (int or None): Manually inputted number of syllables to label. (default = None)

    Returns
    -------
    '''

    # Get proper input parameters
    index_file = progress_paths['index_file']
    model_path = progress_paths['model_path']
    syll_info_path = progress_paths['syll_info']
    syll_info_df_path = progress_paths['df_info_path']

    output_notebook()
    interactive_syllable_stat_wrapper(index_file, model_path, syll_info_path,
                                      syll_info_df_path, max_syllables=max_syllable)

def copy_h5_metadata_to_yaml_command(input_dir, h5_metadata_path):
    '''
    Reads h5 metadata from a specified metadata h5 path.

    Parameters
    ----------
    input_dir (str): path to directory containing h5 file
    h5_metadata_path (str): path to metadata within h5 file

    Returns
    -------
    None
    '''

    copy_h5_metadata_to_yaml_wrapper(input_dir, h5_metadata_path)


def make_crowd_movies_command(index_file, model_path, output_dir, max_syllable, max_examples):
    '''
    Runs CLI function to write crowd movies, due to multiprocessing
    compatibilty issues with Jupyter notebook's scheduler.

    Parameters
    ----------
    index_file (str): path to index file.
    model_path (str): path to fit model.
    output_dir (str): path to directory to save crowd movies in.
    max_syllable (int): number of syllables to make crowd movies for.
    max_examples (int): max number of mice to include in a crowd movie.

    Returns
    -------
    (str): Success string.
    '''


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.system(f'moseq2-viz make-crowd-movies --max-syllable {max_syllable} -m {max_examples} -o {output_dir} {index_file} {model_path}')

    if len(os.listdir(output_dir)) >= max_syllable:
        return 'Successfully generated '+str(max_syllable) + ' crowd videos.'

def plot_stats_command(model_fit, index_file, output_file, stat='usage', max_syllable=40, count='usage', group=None, sort=True,
                        ordering=None, ctrl_group=None, exp_group=None, colors=None, fmt='o-', figsize=(10, 5)):
    '''
    Graph given syllable statistic from fit model data.
    Parameters
    ----------
    model_fit (str): path to fit model.
    index_file (str): path to index file
    output_file (str): name of saved usages graph.
    stat (str): syllable statistic to plot: ['usage', 'speed', 'duration']
    max_syllable (int): max number of syllables to plot.
    count (str): method to calculate syllable usages, either by 'frames' or 'usage'
    group (tuple): groups to include in usage plot. If empty, plots default average of all groups.
    sort (bool): sort by usages.
    ordering (list, range, str, None): order to list syllables. Default is None to graph syllables [0-max_syllable).
     Setting ordering to "m" will graph mutated syllable usage difference between ctrl_group and exp_group.
     None to graph default [0,max_syllable] in order. "usage" to plot descending order of usage values.
    ctrl_group (str): Control group to graph when plotting mutation differences via setting ordering to 'm'.
    exp_group (str): Experimental group to directly compare with control group.
    colors (list): list of colors to serve as the sns palette in the scalar summary. If None, default colors are used.
    fmt (str): scatter plot format. "o-" for line plot with vertices at corresponding usages. "o" for just points.
    figsize (tuple): tuple value of length = 2, representing (columns x rows) of the plotted figure dimensions
    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''


    fig = plot_syllable_stat_wrapper(model_fit, index_file, output_file, stat=stat, max_syllable=max_syllable,
                                     sort=sort, count=count, group=group, fmt=fmt, ordering=ordering,
                                     ctrl_group=ctrl_group, exp_group=exp_group, colors=colors, figsize=figsize)

    print(f'{stat} plot successfully generated')
    return fig

def plot_scalar_summary_command(index_file, output_file, colors=None, groupby='group'):
    '''
    Creates a scalar summary graph and a position summary graph.

    Parameters
    ----------
    index_file (str): path to index file
    output_file (str): prefix name of scalar summary images
    colors (list): list of colors to serve as the sns palette in the scalar summary
    groupby (str): scalar_df column to group sessions by when graphing scalar and position summaries

    Returns
    -------
    scalar_df (pandas DataFrame): DataFrame containing all of scalar values for debugging.
    '''

    scalar_df = plot_scalar_summary_wrapper(index_file, output_file, groupby=groupby, colors=colors)
    return scalar_df

def plot_transition_graph_command(index_file, model_fit, config_file, max_syllable, group, output_file):
    '''
    Creates transition graphs given groups to process.

    Parameters
    ----------
    index_file (str): path to index file
    model_fit (str): path to fit model
    config_file (str): path to config file
    max_syllable (int): maximum number of syllables to include in graph
    group (tuple): tuple of names of groups to graph transition graphs for
    output_file (str): name of the transition graph saved image

    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    f.close()

    # Get default CLI params
    objs = plot_transition_graph.params

    params = {tmp.name: tmp.default for tmp in objs if not tmp.required}
    for k, v in params.items():
        if k not in config_data.keys():
            config_data[k] = v

    config_data['max_syllable'] = max_syllable
    config_data['group'] = group

    fig = plot_transition_graph_wrapper(index_file, model_fit, config_data, output_file)

    print('Transition graph(s) successfully generated')
    return fig

def plot_mean_group_position_heatmaps_command(index_file, output_file):
    '''
    Plots the average mouse position in a PDF-derived heatmap for each group found in the inputted index file.
    Parameters
    ----------
    index_file (str): path to index file.
    output_file (str): filename for syllable duration graph.

    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''

    fig = plot_mean_group_position_pdf_wrapper(index_file, output_file)

    return fig

def plot_verbose_position_heatmaps(index_file, output_file):
    '''
    Plots a PDF-derived heatmap of each session found in the index file titled with the session name and group.

    Parameters
    ----------
    index_file (str): path to index file.
    output_file (str): filename for syllable duration graph.

    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''

    fig = plot_verbose_pdfs_wrapper(index_file, output_file)

    return fig
