'''

'''

import joblib
import numpy as np
import pandas as pd
import ruamel.yaml as yaml
from collections import OrderedDict
from IPython.display import display
from moseq2_viz.interactive.widgets import *
from moseq2_viz.util import index_to_dataframe
from ipywidgets import fixed, interactive_output
from moseq2_viz.interactive.view import graph_dendrogram
from moseq2_viz.helpers.wrappers import init_wrapper_function
from moseq2_viz.interactive.controller import SyllableLabeler, InteractiveSyllableStats, CrowdMovieComparison
from moseq2_viz.model.util import relabel_by_usage, get_syllable_usages, parse_model_results

def interactive_group_setting_wrapper(index_filepath):
    '''

    Parameters
    ----------
    index_filepath

    Returns
    -------

    '''

    index_dict, df = index_to_dataframe(index_filepath)
    qgrid_widget = qgrid.show_grid(df[['SessionName', 'SubjectName', 'group', 'uuid']], column_options=col_opts,
                                   column_definitions=col_defs, show_toolbar=False)

    def update_table(b):
        '''

        Parameters
        ----------
        b

        Returns
        -------

        '''

        update_index_button.button_style = 'info'
        update_index_button.icon = 'none'

        selected_rows = qgrid_widget.get_selected_df()
        x = selected_rows.index

        for i in x:
            qgrid_widget.edit_cell(i, 'group', group_input.value)

    def update_clicked(b):
        '''

        Parameters
        ----------
        b

        Returns
        -------

        '''

        files = index_dict['files']
        meta = [f['metadata'] for f in files]
        meta_cols = pd.DataFrame(meta).columns

        latest_df = qgrid_widget.get_changed_df()
        df.update(latest_df)

        updated_index = {'files': list(df.drop(meta_cols, axis=1).to_dict(orient='index').values()),
                         'pca_path': index_dict['pca_path']}

        with open(index_filepath, 'w+') as f:
            yaml.safe_dump(updated_index, f)

        update_index_button.button_style = 'success'
        update_index_button.icon = 'check'

    update_index_button.on_click(update_clicked)

    save_button.on_click(update_table)

    display(group_set)
    display(qgrid_widget)

def interactive_syllable_labeler_wrapper(model_path, crowd_movie_dir, output_file, max_syllables=None):
    '''

    Parameters
    ----------
    model_path
    crowd_movie_dir
    output_file
    max_syllables

    Returns
    -------

    '''

    # Load the model
    model = parse_model_results(joblib.load(model_path))

    # Compute the sorted labels
    model['labels'] = relabel_by_usage(model['labels'], count='usage')[0]

    # Get Maximum number of syllables to include
    if max_syllables == None:
        syllable_usages = get_syllable_usages(model, 'usage')
        cumulative_explanation = 100 * np.cumsum(syllable_usages)
        max_sylls = np.argwhere(cumulative_explanation >= 90)[0][0]
    else:
        max_sylls = max_syllables

    # Make initial syllable information dict
    labeler = SyllableLabeler(max_sylls=max_sylls, save_path=output_file)
    labeler.get_crowd_movie_paths(crowd_movie_dir)

    syll_select.options = labeler.syll_info

def interactive_syllable_stat_wrapper(index_path, model_path, info_path, max_syllables=None):
    '''

    Parameters
    ----------
    index_path
    model_path
    info_path
    max_syllables

    Returns
    -------

    '''

    istat = InteractiveSyllableStats(index_path=index_path, model_path=model_path, info_path=info_path,
                                     max_sylls=max_syllables)

    istat.interactive_stat_helper()

    session_sel.options = list(istat.df.SessionName.unique())
    ctrl_dropdown.options = list(istat.df.group.unique())
    exp_dropdown.options = list(istat.df.group.unique())

    out = interactive_output(istat.interactive_syll_stats_grapher, {'df': fixed(istat.df),
                                                                    'obj': fixed(istat),
                                                                    'stat': stat_dropdown,
                                                                    'sort': sorting_dropdown,
                                                                    'groupby': grouping_dropdown,
                                                                    'sessions': session_sel,
                                                                    'ctrl_group': ctrl_dropdown,
                                                                    'exp_group': exp_dropdown
                                                                    })

    display(widget_box, out)
    graph_dendrogram(istat)

    def show_mutation_group_select(change):
        '''

        Parameters
        ----------
        change

        Returns
        -------

        '''

        if change.new == 'mutation':
            ctrl_dropdown.layout.display = "block"
            exp_dropdown.layout.display = "block"
        elif sorting_dropdown.value != 'mutation':
            ctrl_dropdown.layout.display = "none"
            exp_dropdown.layout.display = "none"

    def show_session_select(change):
        '''

        Parameters
        ----------
        change

        Returns
        -------

        '''

        if change.new == 'SessionName':
            session_sel.layout = layout_visible
        elif change.new == 'group':
            session_sel.layout = layout_hidden

    grouping_dropdown.observe(show_session_select)
    sorting_dropdown.observe(show_mutation_group_select)

def interactive_crowd_movie_comparison_preview(config_data, index_path, model_path, syll_info_path, output_dir):
    '''
    Wrapper function that launches an interactive crowd movie comparison application.
    Uses ipywidgets and Bokeh to facilitate real time user interaction.

    Parameters
    ----------
    config_data (dict): dict containing crowd movie creation parameters
    index_path (str): path to index file with paths to all the extracted sessions
    model_path (str): path to trained model containing syllable labels.
    syll_info_path (str): path to syllable information file containing syllable labels
    output_dir (str): path to directory to store crowd movies

    Returns
    -------
    '''

    with open(syll_info_path, 'r') as f:
        syll_info = yaml.safe_load(f)

    syll_select.options = syll_info
    index, sorted_index, model_fit = init_wrapper_function(index_file=index_path, model_fit=model_path, output_dir=output_dir)

    sessions = list(set(model_fit['metadata']['uuids']))
    session_sel.options = [sorted_index['files'][s]['metadata']['SessionName'] for s in sessions]

    cm_compare = CrowdMovieComparison(config_data=config_data, index_path=index_path,
                                      model_path=model_path, syll_info=syll_info, output_dir=output_dir)

    cm_compare.get_session_mean_syllable_info_df(model_fit, sorted_index)

    out = interactive_output(cm_compare.crowd_movie_preview, {'config_data': fixed(cm_compare.config_data),
                                                   'syllable': syll_select,
                                                   'groupby': cm_sources_dropdown,
                                                   'sessions': session_sel,
                                                   'nexamples': num_examples})
    display(out)

    session_sel.observe(cm_compare.select_session)
    cm_sources_dropdown.observe(cm_compare.show_session_select)