'''

Interactive applications for labeling syllables, plotting syllable stats, comparing crowd movies
    and plotting transition graphs.
Interactivity functionality is facilitated via IPyWidget and Bokeh.

'''
import os
import glob
import time
import joblib
import shutil
import warnings
import numpy as np
import pandas as pd
from bokeh.io import show
import ruamel.yaml as yaml
import ipywidgets as widgets
from bokeh.models import Div
from bokeh.layouts import column
from bokeh.plotting import figure
from moseq2_viz.util import parse_index
from moseq2_viz.io.video import get_video_info
from IPython.display import display, clear_output
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from moseq2_viz.helpers.wrappers import make_crowd_movies_wrapper
from moseq2_viz.interactive.view import bokeh_plotting, graph_dendrogram, display_crowd_movies
from moseq2_viz.model.label_util import get_sorted_syllable_stat_ordering, get_syllable_mutation_ordering
from moseq2_viz.interactive.widgets import SyllableLabelerWidgets, SyllableStatWidgets, CrowdMovieCompareWidgets
from moseq2_viz.model.util import results_to_dataframe, parse_model_results, relabel_by_usage, get_syllable_usages
from moseq2_viz.scalars.util import (scalars_to_dataframe, compute_session_centroid_speeds, compute_mean_syll_scalar,
                                     compute_syllable_position_heatmaps, get_syllable_pdfs)


class SyllableLabeler(SyllableLabelerWidgets):
    '''
    Class that contains functionality for previewing syllable crowd movies and
     user interactions with buttons and menus.
    '''

    def __init__(self, model_fit, index_file, max_sylls, save_path):
        '''
        Initializes class context parameters, reads and creates the syllable information dict.

        Parameters
        ----------
        model_fit (dict): Loaded trained model dict.
        index_file (str): Path to saved index file.
        max_sylls (int): Maximum number of syllables to preview and label.
        save_path (str): Path to save syllable label information dictionary.
        '''

        super().__init__()
        self.save_path = save_path
        self.max_sylls = max_sylls

        self.model_fit = model_fit
        self.sorted_index = parse_index(index_file)[1]

        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                self.syll_info = yaml.safe_load(f)
        else:
            self.syll_info = {str(i): {'label': '', 'desc': '', 'crowd_movie_path': '', 'group_info': {}} for i in
                              range(max_sylls)}

        # Initialize button callbacks
        self.next_button.on_click(self.on_next)
        self.prev_button.on_click(self.on_prev)
        self.set_button.on_click(self.on_set)

        # Syllable Info DataFrame path
        output_dir = os.path.dirname(save_path)
        self.df_output_file = os.path.join(output_dir, 'syll_df.parquet')
        self.label_df_output_file = os.path.join(output_dir, 'label_time_df.parquet')

    def on_next(self, event):
        '''
        Callback function to trigger an view update when the user clicks the "Next" button.

        Parameters
        ----------
        event (ipywidgets.ButtonClick): User clicks next button.

        Returns
        -------
        '''

        # Updating dict
        self.syll_info[str(self.syll_select.index)]['label'] = self.lbl_name_input.value
        self.syll_info[str(self.syll_select.index)]['desc'] = self.desc_input.value

        # Handle cycling through syllable labels
        if self.syll_select.index != int(list(self.syll_select.options.keys())[-1]):
            # Updating selection to trigger update
            self.syll_select.index += 1
        else:
            self.syll_select.index = 0

        # Updating input values with current dict entries
        self.lbl_name_input.value = self.syll_info[str(self.syll_select.index)]['label']
        self.desc_input.value = self.syll_info[str(self.syll_select.index)]['desc']

    def on_prev(self, event):
        '''
        Callback function to trigger an view update when the user clicks the "Previous" button.

        Parameters
        ----------
        event (ipywidgets.ButtonClick): User clicks 'previous' button.

        Returns
        -------
        '''

        # Update syllable information dict
        self.syll_info[str(self.syll_select.index)]['label'] = self.lbl_name_input.value
        self.syll_info[str(self.syll_select.index)]['desc'] = self.desc_input.value

        # Handle cycling through syllable labels
        if self.syll_select.index != 0:
            # Updating selection to trigger update
            self.syll_select.index -= 1
        else:
            self.syll_select.index = int(list(self.syll_select.options.keys())[-1])

        # Reloading previously inputted text area string values
        self.lbl_name_input.value = self.syll_info[str(self.syll_select.index)]['label']
        self.desc_input.value = self.syll_info[str(self.syll_select.index)]['desc']

    def on_set(self, event):
        '''
        Callback function to save the dict to syllable information file.

        Parameters
        ----------
        event (ipywidgets.ButtonClick): User clicks the 'Save' button.

        Returns
        -------
        '''

        # Update dict
        self.syll_info[str(self.syll_select.index)]['label'] = self.lbl_name_input.value
        self.syll_info[str(self.syll_select.index)]['desc'] = self.desc_input.value

        yml = yaml.YAML()
        yml.indent(mapping=3, offset=2)

        # Write to file
        with open(self.save_path, 'w+') as f:
            yml.dump(self.syll_info, f)

        # Update button style
        self.set_button.button_type = 'success'

    def get_mean_syllable_info(self):
        '''
        Populates syllable information dict with usage and scalar information.

        Returns
        -------
        '''

        # Load scalar Dataframe to compute syllable speeds
        scalar_df = scalars_to_dataframe(self.sorted_index)

        # Compute a syllable summary Dataframe containing usage-based
        # sorted/relabeled syllable usage and duration information from [0, max_syllable) inclusive
        df, label_df = results_to_dataframe(self.model_fit, self.sorted_index, count='usage',
                                            max_syllable=self.max_sylls, sort=True, compute_labels=True)

        # Compute syllable speed and average distance to bucket center
        scalar_df['centroid_speed_mm'] = compute_session_centroid_speeds(scalar_df)
        df = compute_mean_syll_scalar(df, scalar_df, label_df, max_sylls=self.max_sylls)
        df = compute_mean_syll_scalar(df, scalar_df, label_df, scalar='dist_to_center_px', max_sylls=self.max_sylls)

        print('Writing main syllable info to parquet')
        df.to_parquet(self.df_output_file, engine='fastparquet', compression='gzip')
        tmp = label_df.copy()
        tmp.columns = tmp.columns.astype(str)
        print('Writing session syllable labels to parquet')
        tmp.to_parquet(self.label_df_output_file, engine='fastparquet', compression='gzip')

        # Get all unique groups in df
        self.groups = df.group.unique()

        # Get grouped DataFrame
        group_df = df.groupby(['group', 'syllable'], as_index=False).mean()

        # Get array of grouped syllable info
        group_dicts = []
        for group in self.groups:
            group_dict = {
                group: group_df[group_df['group'] == group].drop('group', axis=1).reset_index(drop=True).to_dict()}
            group_dicts.append(group_dict)

        # Update syllable info dict
        for gd in group_dicts:
            group_name = list(gd.keys())[0]
            for syll in range(self.max_sylls):
                self.syll_info[str(syll)]['group_info'][group_name] = {
                    'usage': gd[group_name]['usage'][syll],
                    'speed': gd[group_name]['speed'][syll],
                    'dist_to_center': gd[group_name]['dist_to_center'][syll],
                }

    def set_group_info_widgets(self, group_info):
        '''
        Display function that reads the syllable information into a pandas DataFrame, converts it
        to an HTML table and displays it in a Bokeh Div facilitated via the Output() widget.

        Parameters
        ----------
        group_info (dict): Dictionary of grouped current syllable information

        Returns
        -------
        '''

        output_table = Div(text=pd.DataFrame(group_info).to_html())

        ipy_output = widgets.Output()
        with ipy_output:
            show(output_table)

        self.info_boxes.children = [self.syll_info_lbl, ipy_output, ]

    def interactive_syllable_labeler(self, syllables):
        '''
        Helper function that facilitates the interactive view. Function will create a Bokeh Div object
        that will display the current video path.

        Parameters
        ----------
        syllables (int or ipywidgets.DropDownMenu): Current syllable to label

        Returns
        -------
        '''

        self.set_button.button_type = 'primary'

        # Set current widget values
        if len(syllables['label']) > 0:
            self.lbl_name_input.value = syllables['label']

        if len(syllables['desc']) > 0:
            self.desc_input.value = syllables['desc']

        # Update label
        self.cm_lbl.text = f'Crowd Movie {self.syll_select.index + 1}/{len(self.syll_select.options)}'

        # Update scalar values
        group_info = self.syll_info[str(self.syll_select.index)]['group_info']
        self.set_group_info_widgets(group_info)

        # Get current movie path
        cm_path = syllables['crowd_movie_path']

        video_dims = get_video_info(cm_path)['dims']

        # Create syllable crowd movie HTML div to embed
        video_div = f'''
                        <h2>{self.syll_select.index}: {syllables['label']}</h2>
                        <video
                            src="{cm_path}"; alt="{cm_path}"; height="{video_dims[1]}"; width="{video_dims[0]}"; preload="true";
                            style="float: left; type: "video/mp4"; margin: 0px 10px 10px 0px;
                            border="2"; autoplay controls loop>
                        </video>
                    '''

        # Create embedded HTML Div and view layout
        div = Div(text=video_div, style={'width': '100%'})
        layout = column([div, self.cm_lbl])

        # Insert Bokeh div into ipywidgets Output widget to display
        vid_out = widgets.Output(height='500px')
        with vid_out:
            show(layout)

        # Create grid layout to display all the widgets
        grid = widgets.GridspecLayout(1, 2, height='500px', layout=widgets.Layout(align_items='flex-start'))
        grid[0, 0] = vid_out
        grid[0, 1] = self.data_box

        # Display all widgets
        display(grid, self.button_box)

    def set_default_cm_parameters(self, config_data):
        '''
        Sets necessary default values if they are not found.

        Parameters
        ----------
        config_data (dict): Crowd movie writing configuration parameter dict.

        Returns
        -------
        config_data (dict): Updating configuration parameter dict.
        '''

        config_data['separate_by'] = None
        config_data['specific_syllable'] = None
        config_data['max_syllable'] = self.max_sylls
        config_data['max_examples'] = 20

        config_data['gaussfilter_space'] = [0, 0]
        config_data['medfilter_space'] = [0]
        config_data['sort'] = True
        config_data['dur_clip'] = 300
        config_data['raw_size'] = (512, 424)
        config_data['scale'] = 1
        config_data['legacy_jitter_fix'] = False
        config_data['cmap'] = 'jet'
        config_data['count'] = 'usage'

        return config_data

    def get_crowd_movie_paths(self, index_path, model_path, config_data, crowd_movie_dir):
        '''
        Populates the syllable information dict with the respective crowd movie paths.

        Parameters
        ----------
        crowd_movie_dir (str): Path to directory containing all the generated crowd movies

        Returns
        -------
        '''

        if not os.path.exists(crowd_movie_dir):
            print('Crowd movies not found. Generating movies...')
            config_data = self.set_default_cm_parameters(config_data)

            # Generate movies if directory does not exist
            _ = make_crowd_movies_wrapper(index_path, model_path, config_data, crowd_movie_dir)

        # Get movie paths
        crowd_movie_paths = [f for f in glob(crowd_movie_dir + '*') if '.mp4' in f]

        if len(crowd_movie_paths) < self.max_sylls:
            print('Crowd movie list is incomplete. Generating movies...')
            config_data = self.set_default_cm_parameters(config_data)

            # Generate movies if directory does not exist
            _ = make_crowd_movies_wrapper(index_path, model_path, config_data, crowd_movie_dir)

        crowd_movie_paths = [f for f in glob(crowd_movie_dir + '*') if '.mp4' in f]

        for cm in crowd_movie_paths:
            # Parse paths to get corresponding syllable number
            syll_num = cm.split('sorted-id-')[1].split()[0]
            if syll_num in self.syll_info.keys():
                self.syll_info[syll_num]['crowd_movie_path'] = cm

class InteractiveSyllableStats(SyllableStatWidgets):
    '''

    Interactive Syllable Statistics grapher class that holds the context for the current
     inputted session.

    '''

    def __init__(self, index_path, model_path, df_path, info_path, max_sylls):
        '''
        Initialize the main data inputted into the current context

        Parameters
        ----------
        index_path (str): Path to index file.
        model_path (str): Path to trained model file.
        info_path (str): Path to syllable information file.
        max_sylls (int): Maximum number of syllables to plot.
        '''

        super().__init__()

        self.model_path = model_path
        self.info_path = info_path
        self.max_sylls = max_sylls
        self.index_path = index_path
        self.df_path = df_path

        if df_path != None:
            if os.path.exists(df_path):
                self.label_df_path = df_path.replace('syll_df', 'label_time_df')
            else:
                self.df_path = None

        self.df = None

        self.ar_mats = None
        self.results = None
        self.icoord, self.dcoord = None, None
        self.cladogram = None

        # Load all the data
        self.interactive_stat_helper()

        # Update the widget values
        self.session_sel.options = sorted(list(self.df.SessionName.unique()))
        self.session_sel.value = [self.session_sel.options[0]]

        self.ctrl_dropdown.options = list(self.df.group.unique())
        self.exp_dropdown.options = list(self.df.group.unique())
        self.exp_dropdown.value = self.ctrl_dropdown.options[-1]

    def compute_dendrogram(self):
        '''
        Computes the pairwise distances between the included model AR-states, and
        generates the graph information to be plotted after the stats.

        Returns
        -------
        '''

        # Get Pairwise distances
        X = pairwise_distances(self.ar_mats, metric='euclidean')
        Z = linkage(X, 'ward')

        # Get Dendrogram Metadata
        self.results = dendrogram(Z, distance_sort=True, no_plot=True, get_leaves=True)

        # Get Graph layout info
        icoord, dcoord = self.results['icoord'], self.results['dcoord']

        icoord = pd.DataFrame(icoord) - 5
        icoord = icoord * (self.df['syllable'].max() / icoord.max().max())
        self.icoord = icoord.values

        dcoord = pd.DataFrame(dcoord)
        dcoord = dcoord * (self.df['usage'].max() / dcoord.max().max())
        self.dcoord = dcoord.values

    def interactive_stat_helper(self):
        '''
        Computes and saves the all the relevant syllable information to be displayed.
         Loads the syllable information dict and merges it with the syllable statistics DataFrame.

        Returns
        -------
        '''
        warnings.filterwarnings('ignore')

        # Read syllable information dict
        with open(self.info_path, 'r') as f:
            syll_info = yaml.safe_load(f)

        # Getting number of syllables included in the info dict
        max_sylls = len(list(syll_info.keys()))
        for k in range(max_sylls):
            del syll_info[str(k)]['group_info']

        info_df = pd.DataFrame(list(syll_info.values()), index=[int(k) for k in list(syll_info.keys())]).sort_index()
        info_df['syllable'] = info_df.index

        # Load the model
        model_data = parse_model_results(joblib.load(self.model_path))

        # Relabel the models, and get the order mapping
        labels, mapping = relabel_by_usage(model_data['labels'], count='usage')

        # Get max syllables if None is given
        syllable_usages = get_syllable_usages({'labels': labels}, count='usage')
        cumulative_explanation = 100 * np.cumsum(syllable_usages)
        if self.max_sylls == None:
            self.max_sylls = np.argwhere(cumulative_explanation >= 90)[0][0]

        # Read AR matrices and reorder according to the syllable mapping
        ar_mats = np.array(model_data['model_parameters']['ar_mat'])
        self.ar_mats = np.reshape(ar_mats, (100, -1))[mapping][:self.max_sylls]

        if self.df_path != None:
            print('Loading parquet files')
            df = pd.read_parquet(self.df_path, engine='fastparquet')
            label_df = pd.read_parquet(self.label_df_path, engine='fastparquet')
            label_df.columns = label_df.columns.astype(int)
        else:
            print('Syllable DataFrame not found. Computing syllable statistics...')
            # Read index file
            index, sorted_index = parse_index(self.index_path)

            # Load scalar Dataframe to compute syllable speeds
            scalar_df = scalars_to_dataframe(sorted_index)

            # Compute a syllable summary Dataframe containing usage-based
            # sorted/relabeled syllable usage and duration information from [0, max_syllable) inclusive
            df, label_df = results_to_dataframe(model_data, index, count='usage',
                                                max_syllable=self.max_sylls, sort=True, compute_labels=True)

            # Compute centroid speeds
            scalar_df['centroid_speed_mm'] = compute_session_centroid_speeds(scalar_df)

            # Compute and append additional syllable scalar data
            df = compute_mean_syll_scalar(df, scalar_df, label_df, groups=None, max_sylls=self.max_sylls)
            df = compute_mean_syll_scalar(df, scalar_df, label_df, scalar='dist_to_center_px', groups=None, max_sylls=self.max_sylls)

        self.df = df.merge(info_df, on='syllable')

    def interactive_syll_stats_grapher(self, stat, sort, groupby, errorbar, sessions, ctrl_group, exp_group):
        '''
        Helper function that is responsible for handling ipywidgets interactions and updating the currently
         displayed Bokeh plot.

        Parameters
        ----------
        stat (list or ipywidgets.DropDown): Statistic to plot: ['usage', 'speed', 'distance to center']
        sort (list or ipywidgets.DropDown): Statistic to sort syllables by (in descending order).
            ['usage', 'speed', 'distance to center', 'similarity', 'difference'].
        groupby (list or ipywidgets.DropDown): Data to plot; either group averages, or individual session data.
        sessions (list or ipywidgets.MultiSelect): List of selected sessions to display data from.
        ctrl_group (str or ipywidgets.DropDown): Name of control group to compute group difference sorting with.
        exp_group (str or ipywidgets.DropDown): Name of comparative group to compute group difference sorting with.

        Returns
        -------
        '''

        # Get current dataFrame to plot
        df = self.df

        # Handle names to query DataFrame with
        if stat == 'distance to center':
            stat = 'dist_to_center'
        if sort == 'distance to center':
            sort = 'dist_to_center'

        # Get selected syllable sorting
        if sort == 'difference':
            # display Text for groups to input experimental groups
            ordering = get_syllable_mutation_ordering(df, ctrl_group, exp_group, stat=stat)
        elif sort == 'similarity':
            ordering = self.results['leaves']
        elif sort != 'usage':
            ordering, _ = get_sorted_syllable_stat_ordering(df, stat=sort)
        else:
            ordering = range(len(df.syllable.unique()))

        # Handle selective display for whether mutation sort is selected
        if sort == 'difference':
            self.mutation_box.layout.display = "block"
        else:
            self.mutation_box.layout.display = "none"

        # Handle selective display to select included sessions to graph
        if groupby == 'SessionName':
            self.session_sel.layout.display = "flex"
            self.session_sel.layout.align_items = 'stretch'

            mean_df = df.copy()
            df = df[df['SessionName'].isin(self.session_sel.value)]

        else:
            self.session_sel.layout.display = "none"
            mean_df = None

        # Compute cladogram if it does not already exist
        if self.cladogram == None:
            self.cladogram = graph_dendrogram(self)
            self.results['cladogram'] = self.cladogram

        self.stat_fig = bokeh_plotting(df, stat, ordering, mean_df=mean_df, groupby=groupby,
                                       errorbar=errorbar, syllable_families=self.results)

class CrowdMovieComparison(CrowdMovieCompareWidgets):
    '''
    Crowd Movie Comparison application class. Contains all the user inputted parameters
    within its context.

    '''

    def __init__(self, config_data, index_path, df_path, model_path, syll_info, output_dir, get_pdfs):
        '''
        Initializes class object context parameters.

        Parameters
        ----------
        config_data (dict): Configuration parameters for creating crowd movies.
        index_path (str): Path to loaded index file.
        model_path (str): Path to loaded model.
        syll_info (dict): Dict object containing labeled syllable information.
        output_dir (str): Path to directory to store crowd movies.
        '''
        super().__init__()

        self.config_data = config_data
        self.index_path = index_path
        self.model_path = model_path
        self.df_path = df_path
        self.get_pdfs = get_pdfs

        if df_path != None:
            if os.path.exists(df_path):
                self.label_df_path = df_path.replace('syll_df', 'label_time_df')
            else:
                self.df_path = None

        self.syll_info = syll_info
        self.output_dir = output_dir
        self.max_sylls = len(syll_info.keys())

        # Prepare current context's base session syllable info dict
        self.session_dict = {str(i): {'session_info': {}} for i in range(self.max_sylls)}

        # Set widget callbacks
        self.cm_session_sel.observe(self.select_session)
        self.cm_sources_dropdown.observe(self.show_session_select)
        self.cm_trigger_button.on_click(self.on_click_trigger_button)

        self.set_default_cm_parameters()

    def set_default_cm_parameters(self):

        self.config_data['max_syllable'] = self.max_sylls
        self.config_data['separate_by'] = 'groups'

        self.config_data['gaussfilter_space'] = [0, 0]
        self.config_data['medfilter_space'] = [0]
        self.config_data['sort'] = True
        self.config_data['dur_clip'] = 300
        self.config_data['raw_size'] = (512, 424)
        self.config_data['scale'] = 1
        self.config_data['legacy_jitter_fix'] = False
        self.config_data['cmap'] = 'jet'
        self.config_data['count'] = 'usage'

    def show_session_select(self, change):
        '''
        Callback function to change current view to show session selector when user switches
        DropDownMenu selection to 'SessionName', and hides it if the user
        selects 'groups'.

        Parameters
        ----------
        change (event): User switches their DropDownMenu selection

        Returns
        -------
        '''

        # Handle display syllable selection and update config_data crowd movie generation
        # source selector.
        if change.new == 'SessionName':
            # Show session selector
            self.cm_session_sel.layout = self.layout_visible
            self.cm_trigger_button.layout.display = 'block'
            self.config_data['separate_by'] = 'sessions'
        elif change.new == 'group':
            # Hide session selector
            self.cm_session_sel.layout = self.layout_hidden
            self.cm_trigger_button.layout.display = 'none'
            self.config_data['separate_by'] = 'groups'

    def select_session(self, event):
        '''
        Callback function to save the list of selected sessions to config_data,
         and get session syllable info to pass to crowd_movie_wrapper and create the
         accompanying syllable scalar metadata table.

        Parameters
        ----------
        event (event): User clicks on multiple sessions in the SelectMultiple widget

        Returns
        -------
        '''

        # Set currently selected sessions
        self.config_data['session_names'] = list(self.cm_session_sel.value)

        # Update session_syllable info dict
        self.get_selected_session_syllable_info(self.config_data['session_names'])

    def get_session_mean_syllable_info_df(self, model_fit, sorted_index):
        '''
        Populates session-based syllable information dict with usage and scalar information.
        Parameters
        ----------
        model_fit (dict): dict containing trained model syllable data
        sorted_index (dict): sorted index file containing paths to extracted session h5s
        Returns
        -------
        '''
        warnings.filterwarnings('ignore')

        if self.df_path != None:
            print('Loading parquet files')
            df = pd.read_parquet(self.df_path, engine='fastparquet')
            label_df = pd.read_parquet(self.label_df_path, engine='fastparquet')
            label_df.columns = label_df.columns.astype(int)

            # Load scalar Dataframe to compute syllable speeds
            scalar_df = scalars_to_dataframe(sorted_index)
        else:
            print('Syllable DataFrame not found. Computing syllable statistics...')

            # Load scalar Dataframe to compute syllable speeds
            scalar_df = scalars_to_dataframe(sorted_index)

            # Compute a syllable summary Dataframe containing usage-based
            # sorted/relabeled syllable usage and duration information from [0, max_syllable) inclusive
            df, label_df = results_to_dataframe(model_fit, sorted_index, count='usage',
                                                max_syllable=self.max_sylls, sort=True, compute_labels=True)

            # Compute syllable speed
            scalar_df['centroid_speed_mm'] = compute_session_centroid_speeds(scalar_df)
            df = compute_mean_syll_scalar(df, scalar_df, label_df, groups=None, max_sylls=self.max_sylls)

        if self.get_pdfs:
            # Compute syllable position PDFs
            self.df = compute_syllable_position_heatmaps(df, scalar_df, label_df, syllables=range(self.max_sylls))

        # Get grouped DataFrame
        self.session_df = df.groupby(['SessionName', 'syllable'], as_index=False).mean()

    def get_selected_session_syllable_info(self, sel_sessions):
        '''
        Prepares dict of session-based syllable information to display.

        Parameters
        ----------
        sel_sessions (list): list of selected session names.

        Returns
        -------
        '''

        # Get array of grouped syllable info
        session_dicts = []
        for sess in sel_sessions:
            session_dict = {
                sess: self.session_df[self.session_df['SessionName'] == sess].drop('SessionName', axis=1).reset_index(
                    drop=True).to_dict()}
            session_dicts.append(session_dict)

        # Update syllable data with session info
        for sd in session_dicts:
            session_name = list(sd.keys())[0]
            for syll in range(self.max_sylls):
                self.session_dict[str(syll)]['session_info'][session_name] = {
                    'usage': sd[session_name]['usage'][syll],
                    'speed': sd[session_name]['speed'][syll],
                    'duration': sd[session_name]['duration'][syll]
                }

    def generate_crowd_movie_divs(self):
        '''
        Generates HTML divs containing crowd movies and syllable metadata tables
         from the given syllable dict file.

        Returns
        -------
        divs (list of Bokeh.models.Div): Divs of HTML videos and metadata tables.
        '''

        # Compute paths to crowd movies
        path_dict = make_crowd_movies_wrapper(self.index_path, self.model_path, self.config_data, self.output_dir)

        time.sleep(1)

        if self.get_pdfs:
            # Get corresponding syllable position PDF
            group_syll_pdfs, groups = get_syllable_pdfs(self.df, syllables=[self.cm_syll_select.index],
                                                        groupby=self.cm_sources_dropdown.value)

            if self.cm_sources_dropdown.value == 'group':
                g_iter = groups
            else:
                g_iter = self.cm_session_sel.value

            for i, group in enumerate(g_iter):
                self.grouped_syll_dict[group]['pdf'] = group_syll_pdfs[groups.index(group)]

        # Remove previously displayed data
        clear_output()

        # Create syllable info DataFrame
        syll_info_df = pd.DataFrame(self.grouped_syll_dict)

        # Get currently selected syllable name info
        self.curr_label = self.syll_info[str(self.cm_syll_select.index)]['label']
        self.curr_desc = self.syll_info[str(self.cm_syll_select.index)]['desc']

        # Create video divs including syllable metadata
        divs = []
        bk_plots = []
        for group_name, cm_path in path_dict.items():
            # Convert crowd movie metadata to HTML table
            if self.get_pdfs:
                group_info = pd.DataFrame(syll_info_df.drop('pdf', axis=0)[group_name]).to_html()

                group_syllable_pdf = syll_info_df[group_name]['pdf'][0]

                pdf_fig = figure(height=350, width=350, title=f'{group_name}')
                pdf_fig.x_range.range_padding = pdf_fig.y_range.range_padding = 0
                pdf_fig.image(image=[group_syllable_pdf],
                              x=0,
                              y=0,
                              dw=group_syllable_pdf.shape[1],
                              dh=group_syllable_pdf.shape[0],
                              palette="Viridis256")
                bk_plots.append(pdf_fig)
            else:
                group_info = pd.DataFrame(syll_info_df[group_name]).to_html()

            # Copy generated movie to temporary directory
            cm_dir = os.path.dirname(cm_path[0])
            tmp_path = os.path.join(cm_dir, 'tmp', f'{np.random.randint(0, 99999)}_{os.path.basename(cm_path[0])}')
            tmp_dirname = os.path.dirname(tmp_path)

            self.base_tmpdir = os.path.join(cm_dir, 'tmp')

            if not os.path.exists(tmp_dirname):
                os.makedirs(tmp_dirname)

            shutil.copy2(cm_path[0], tmp_path)

            video_dims = get_video_info(tmp_path)['dims']

            # Insert paths and table into HTML div
            group_txt = '''
                {group_info}
                <video
                    src="{src}"; alt="{alt}"; height="{height}"; width="{width}"; preload="auto";
                    style="float: center; type: "video/mp4"; margin: 0px 10px 10px 0px;
                    border="2"; autoplay controls loop>
                </video>
            '''.format(group_info=group_info, src=tmp_path, alt=tmp_path, height=int(video_dims[1] * 0.8),
                       width=int(video_dims[0] * 0.8))

            divs.append(group_txt)

        return divs, bk_plots

    def on_click_trigger_button(self, b):
        '''
        Generates crowd movies and displays them when the user clicks the trigger button

        Parameters
        ----------

        b (ipywidgets.Button click event): User clicks "Generate Movies" button

        Returns
        -------
        '''

        # Compute current selected syllable's session dict.
        self.grouped_syll_dict = self.session_dict[str(self.cm_syll_select.index)]['session_info']

        # Get Crowd Movie Divs
        divs, self.bk_plots = self.generate_crowd_movie_divs()

        # Display generated movies
        display_crowd_movies(self.widget_box, self.curr_label, self.curr_desc, divs, self.bk_plots)

    def crowd_movie_preview(self, syllable, groupby, nexamples):
        '''
        Helper function that triggers the crowd_movie_wrapper function and creates the HTML
        divs containing the generated crowd movies.
        Function is triggered whenever any of the widget function inputs are changed.

        Parameters
        ----------
        syllable (int or ipywidgets.DropDownMenu): Currently displayed syllable.
        nexamples (int or ipywidgets.IntSlider): Number of mice to display per crowd movie.

        Returns
        -------

        '''

        # Update current config data with widget values
        self.config_data['specific_syllable'] = int(self.cm_syll_select.index)
        self.config_data['max_examples'] = nexamples

        # Get group info based on selected DropDownMenu item
        if groupby == 'group':
            self.grouped_syll_dict = self.syll_info[str(self.cm_syll_select.index)]['group_info']

            # Get Crowd Movie Divs
            divs, self.bk_plots = self.generate_crowd_movie_divs()

            # Display generated movies
            display_crowd_movies(self.widget_box, self.curr_label, self.curr_desc, divs, self.bk_plots)
        else:
            # Display widget box until user clicks button to generate session-based crowd movies
            display(self.widget_box)