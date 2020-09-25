'''

Wrapper functions for all the interactive functionality in moseq2-viz.

'''

import os
import qgrid
import shutil
import joblib
import numpy as np
import pandas as pd
from bokeh.io import show
import ruamel.yaml as yaml
import ipywidgets as widgets
from ipywidgets import interactive_output
from moseq2_viz.util import index_to_dataframe
from IPython.display import display, clear_output
from moseq2_viz.interactive.widgets import GroupSettingWidgets
from moseq2_viz.interactive.controller import SyllableLabeler, InteractiveSyllableStats
from moseq2_viz.model.util import relabel_by_usage, get_syllable_usages, parse_model_results

def interactive_group_setting_wrapper(index_filepath):
    '''

    Interactive wrapper function that launches a qgrid object, which is a table
     that has excel-like interactive functionality.

    Users will select multiple rows in the displayed table, enter their desired group name,
     update the entries and finally save the file.

    Qgrid also affords column filtering via mouse-click interactivity and entering strings to filter by
     in the pop-up menu.

    Parameters
    ----------

    index_filepath (str): Path to index file to read and update.

    Returns
    -------
    '''

    index_grid = GroupSettingWidgets()

    index_dict, df = index_to_dataframe(index_filepath)
    qgrid_widget = qgrid.show_grid(df[['SessionName', 'SubjectName', 'group', 'uuid']], column_options=index_grid.col_opts,
                                   column_definitions=index_grid.col_defs, show_toolbar=False)

    def update_table(b):
        '''

        Callback function for when the user clicks the "Set Group" button.
         On click, the table will be updated with the string value inside the text box.

        Parameters
        ----------

        b (ipywidgets.Button event): Callback event.

        Returns
        -------
        '''

        index_grid.update_index_button.button_style = 'info'
        index_grid.update_index_button.icon = 'none'

        selected_rows = qgrid_widget.get_selected_df()
        x = selected_rows.index

        for i in x:
            qgrid_widget.edit_cell(i, 'group', index_grid.group_input.value)

    def update_clicked(b):
        '''

        Button click callback function that writes the updated table values
        to the given index file path.

        Parameters
        ----------

        b (ipywidgets.Button event): Callback event.

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

        index_grid.update_index_button.button_style = 'success'
        index_grid.update_index_button.icon = 'check'

    display(index_grid.group_set, qgrid_widget)

    index_grid.update_index_button.on_click(update_clicked)
    index_grid.save_button.on_click(update_table)

def interactive_syllable_labeler_wrapper(model_path, config_file, index_file, crowd_movie_dir, output_file, max_syllables=None, n_explained=90):
    '''
    Wrapper function to launch a syllable crowd movie preview and interactive labeling application.
    Parameters
    ----------
    model_path (str): Path to trained model.
    crowd_movie_dir (str): Path to crowd movie directory
    output_file (str): Path to syllable label information file
    max_syllables (int): Maximum number of syllables to preview and label.
    Returns
    -------
    '''

    # Load the config file
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # Copy index file to modeling session directory
    modeling_session_dir = os.path.dirname(model_path)
    new_index_path = os.path.join(modeling_session_dir, os.path.basename(index_file))
    shutil.copy2(index_file, new_index_path)

    # Load the model
    model = parse_model_results(joblib.load(model_path))

    # Compute the sorted labels
    model['labels'] = relabel_by_usage(model['labels'], count='usage')[0]

    # Get Maximum number of syllables to include
    if max_syllables == None:
        syllable_usages = get_syllable_usages(model, 'usage')
        cumulative_explanation = 100 * np.cumsum(syllable_usages)
        max_sylls = np.argwhere(cumulative_explanation >= n_explained)[0][0]
        print(f'Number of syllables explaining {n_explained}% variance: {max_sylls}')
    else:
        max_sylls = max_syllables

    # Make initial syllable information dict
    labeler = SyllableLabeler(model_fit=model, index_file=index_file, max_sylls=max_sylls, save_path=output_file)

    # Populate syllable info dict with relevant syllable information
    labeler.get_crowd_movie_paths(index_file, model_path, config_data, crowd_movie_dir)
    labeler.get_mean_syllable_info()

    # Set the syllable dropdown options
    labeler.syll_select.options = labeler.syll_info

    # Launch and display interactive API
    output = widgets.interactive_output(labeler.interactive_syllable_labeler, {'syllables': labeler.syll_select})
    display(labeler.syll_select, output)

    def on_syll_change(change):
        '''
        Callback function for when user selects a different syllable number
        from the Dropdown menu
        Parameters
        ----------
        change (ipywidget DropDown select event): User changes current value of DropDownMenu
        Returns
        -------
        '''

        clear_output()
        display(labeler.syll_select, output)

    # Update view when user selects new syllable from DropDownMenu
    output.observe(on_syll_change, names='value')

def interactive_syllable_stat_wrapper(index_path, model_path, info_path, df_path=None, max_syllables=None):
    '''
    Wrapper function to launch the interactive syllable statistics API. Users will be able to view different
    syllable statistics, sort them according to their metric of choice, and dynamically group the data to
    view individual sessions or group averages.

    Parameters
    ----------
    index_path (str): Path to index file.
    model_path (str): Path to trained model file.
    info_path (str): Path to syllable information file.
    max_syllables (int): Maximum number of syllables to plot.

    Returns
    -------
    '''

    # Initialize the statistical grapher context
    istat = InteractiveSyllableStats(index_path=index_path, model_path=model_path, df_path=df_path,
                                     info_path=info_path, max_sylls=max_syllables)

    # Compute the syllable dendrogram values
    istat.compute_dendrogram()

    # Plot the Bokeh graph with the currently selected data.
    out = interactive_output(istat.interactive_syll_stats_grapher, {
                                                      'stat': istat.stat_dropdown,
                                                      'sort': istat.sorting_dropdown,
                                                      'groupby': istat.grouping_dropdown,
                                                      'errorbar': istat.errorbar_dropdown,
                                                      'sessions': istat.session_sel,
                                                      'ctrl_group': istat.ctrl_dropdown,
                                                      'exp_group': istat.exp_dropdown
                                                      })


    display(istat.stat_widget_box, out)
    show(istat.cladogram)
