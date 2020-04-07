import os
import ruamel.yaml as yaml
from moseq2_viz.helpers.wrappers import add_group_wrapper, plot_syllable_usages_wrapper, plot_scalar_summary_wrapper, \
        plot_syllable_durations_wrapper, plot_transition_graph_wrapper, copy_h5_metadata_to_yaml_wrapper, \
        make_crowd_movies_wrapper

def get_groups_command(index_file, output_directory=None):

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

def add_group_by_session(index_file, value, group, exact, lowercase, negative, output_directory=None):

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

    copy_h5_metadata_to_yaml_wrapper(input_dir, h5_metadata_path)


def make_crowd_movies_command(index_file, model_path, config_file, output_dir, max_syllable, max_examples, output_directory=None):

    if output_directory != None:
        output_dir = os.path.join(output_directory, output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    os.system(f'moseq2-viz make-crowd-movies --max-syllable {max_syllable} -m {max_examples} -o {output_dir} {index_file} {model_path}')

    return 'Successfully generated '+str(max_examples) + ' crowd videos.'

def plot_usages_command(index_file, model_fit, sort, count, max_syllable, group, output_file):

    fig = plot_syllable_usages_wrapper(index_file, model_fit, max_syllable, sort, count, group, output_file, gui=True)
    print('Usage plot successfully generated')
    return fig

def plot_scalar_summary_command(index_file, output_file):

    scalar_df = plot_scalar_summary_wrapper(index_file, output_file, gui=True)
    return scalar_df

def plot_transition_graph_command(index_file, model_fit, config_file, max_syllable, group, output_file):

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    f.close()

    config_data['max_syllable'] = max_syllable
    config_data['group'] = group

    fig = plot_transition_graph_wrapper(index_file, model_fit, config_data, output_file, gui=True)

    print('Transition graph(s) successfully generated')
    return fig

def plot_syllable_durations_command(model_fit, index_file, groups, count, max_syllable, output_file, ylim=None):

    fig = plot_syllable_durations_wrapper(index_file, model_fit, groups, count, max_syllable, output_file, ylim=ylim, gui=True)

    return fig