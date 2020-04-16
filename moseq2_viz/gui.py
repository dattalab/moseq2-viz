import os
import ruamel.yaml as yaml
from .cli import plot_usages, plot_scalar_summary, plot_transition_graph, plot_syllable_durations
from moseq2_viz.helpers.wrappers import add_group_wrapper, plot_syllable_usages_wrapper, plot_scalar_summary_wrapper, \
        plot_syllable_durations_wrapper, plot_transition_graph_wrapper, copy_h5_metadata_to_yaml_wrapper

def get_groups_command(index_file, output_directory=None):
    '''
    Jupyter Notebook to print index file current metadata groupings.

    Parameters
    ----------
    index_file (str): path to index file
    output_directory (str): path to alternative index file path

    Returns
    -------
    (int): number of unique groups
    '''

    if output_directory is not None:
        index_file = os.path.join(output_directory, index_file.split('/')[-1])

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

def add_group_by_session(index_file, value, group, exact, lowercase, negative, output_directory=None):
    '''
    Updates index file SessionName group names with user defined group names.

    Parameters
    ----------
    index_file (str): path to index file
    value (str): SessionName value to search for
    group (str): group name to allocate.
    exact (bool): indicate whether to search for exact match.
    lowercase (bool): indicate whether to convert all searched for names to lowercase.
    negative (bool): whether to update the inverse of the found selection.
    output_directory (str): path to alternative index file path

    Returns
    -------
    None
    '''

    if output_directory is not None:
        index_file = os.path.join(output_directory, index_file.split('/')[-1])

    key = 'SessionName'

    gui_data = {
        'key': key,
        'value': value,
        'group': group,
        'exact': exact,
        'lowercase': lowercase,
        'negative': negative
    }

    add_group_wrapper(index_file, gui_data)
    get_groups_command(index_file)

def add_group_by_subject(index_file, value, group, exact, lowercase, negative, output_directory=None):
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
    output_directory (str): path to alternative index file path

    Returns
    -------
    None
    '''

    if output_directory is not None:
        index_file = os.path.join(output_directory, index_file.split('/')[-1])

    key = 'SubjectName'

    gui_data = {
        'key': key,
        'value': value,
        'group': group,
        'exact': exact,
        'lowercase': lowercase,
        'negative': negative
    }

    add_group_wrapper(index_file, gui_data)
    get_groups_command(index_file)

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


def make_crowd_movies_command(index_file, model_path, output_dir, max_syllable, max_examples, output_directory=None):
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
    output_directory (str): alternative directory prefix to save crowd movies in.

    Returns
    -------
    (str): Success string.
    '''

    if output_directory != None:
        output_dir = os.path.join(output_directory, output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    os.system(f'moseq2-viz make-crowd-movies --max-syllable {max_syllable} -m {max_examples} -o {output_dir} {index_file} {model_path}')

    return 'Successfully generated '+str(max_examples) + ' crowd videos.'

def plot_usages_command(index_file, model_fit, sort, count, max_syllable, group, output_file):
    '''
    Graph syllable usages from fit model data.

    Parameters
    ----------
    index_file (str): path to index file
    model_fit (str): path to fit model.
    sort (bool): sort by usages.
    count (str): method to calculate syllable usages, either by 'frames' or 'usage'
    max_syllable (int): max number of syllables to plot.
    group (tuple): groups to include in usage plot. If empty, plots default average of all groups.
    output_file (str): name of saved usages graph.

    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''


    fig = plot_syllable_usages_wrapper(index_file, model_fit, max_syllable, sort, count, group, output_file, gui=True)
    print('Usage plot successfully generated')
    return fig

def plot_scalar_summary_command(index_file, output_file):
    '''
    Creates a scalar summary graph and a position summary graph

    Parameters
    ----------
    index_file (str): path to index file
    output_file (str): prefix name of scalar summary images

    Returns
    -------
    scalar_df (pandas DataFrame): DataFrame containing all of scalar values for debugging.
    '''

    scalar_df = plot_scalar_summary_wrapper(index_file, output_file, gui=True)
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

    fig = plot_transition_graph_wrapper(index_file, model_fit, config_data, output_file, gui=True)

    print('Transition graph(s) successfully generated')
    return fig

def plot_syllable_durations_command(model_fit, index_file, groups, count, max_syllable, output_file, ylim=None):
    '''
    Plot average syllable durations.

    Parameters
    ----------
    model_fit (str): path to fit model.
    index_file (str): path to index file.
    groups (tuple): tuple groups to separately plot.
    count (str): method to calculate syllable usages, either by 'frames' or 'usage'.
    max_syllable (int): number of syllables to plot durations for.
    output_file (str): name of saved image of durations plot.
    ylim (int): y-axis limit of graph.

    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''

    fig = plot_syllable_durations_wrapper(index_file, model_fit, groups, count, max_syllable, output_file, ylim=ylim, gui=True)

    return fig