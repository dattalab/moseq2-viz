'''

'''

import joblib
import numpy as np
import pandas as pd
from glob import glob
from bokeh.io import show
import ruamel.yaml as yaml
from bokeh.layouts import column
from bokeh.models.widgets import Div
from moseq2_viz.util import parse_index
from moseq2_viz.interactive.widgets import *
from IPython.display import display, clear_output
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from moseq2_viz.helpers.wrappers import make_crowd_movies_wrapper
from moseq2_viz.interactive.view import bokeh_plotting, display_crowd_movies
from moseq2_viz.model.label_util import get_sorted_syllable_stat_ordering, get_syllable_muteness_ordering
from moseq2_viz.scalars.util import scalars_to_dataframe, compute_session_centroid_speeds, compute_mean_syll_speed
from moseq2_viz.model.util import parse_model_results, results_to_dataframe, get_syllable_usages, relabel_by_usage

class SyllableLabeler:
    '''

    '''

    def __init__(self, max_sylls, save_path):
        '''

        Parameters
        ----------
        max_sylls
        save_path
        '''

        self.save_path = save_path
        self.max_sylls = max_sylls
        self.syll_info = {str(i): {'label': '', 'desc': '', 'crowd_movie_path': ''} for i in range(max_sylls)}

    def on_next(self, event):
        '''

        Parameters
        ----------
        event

        Returns
        -------

        '''

        # Updating dict
        self.syll_info[str(syll_select.index)]['label'] = lbl_name_input.value
        self.syll_info[str(syll_select.index)]['desc'] = desc_input.value

        # Updating selection to trigger update
        if syll_select.index != int(list(syll_select.options.keys())[-1]):
            syll_select.index += 1
        else:
            syll_select.index = 0

        # Updating input values with current dict entries
        lbl_name_input.value = self.syll_info[str(syll_select.index)]['label']
        desc_input.value = self.syll_info[str(syll_select.index)]['desc']

    def on_prev(self, event):
        '''

        Parameters
        ----------
        event

        Returns
        -------

        '''

        self.syll_info[str(syll_select.index)]['label'] = lbl_name_input.value
        self.syll_info[str(syll_select.index)]['desc'] = desc_input.value

        if syll_select.index != 0:
            syll_select.index -= 1
        else:
            syll_select.index = int(list(syll_select.options.keys())[-1])

        lbl_name_input.value = self.syll_info[str(syll_select.index)]['label']
        desc_input.value = self.syll_info[str(syll_select.index)]['desc']

    def on_set(self, event):
        '''

        Parameters
        ----------
        event

        Returns
        -------

        '''

        self.syll_info[str(syll_select.index)]['label'] = lbl_name_input.value
        self.syll_info[str(syll_select.index)]['desc'] = desc_input.value

        with open(self.save_path, 'w+') as f:
            yaml.safe_dump(self.syll_info, f)

        set_button.button_type = 'success'

    def interactive_syllable_labeler(self, syllables):
        '''

        Parameters
        ----------
        syllables

        Returns
        -------

        '''

        set_button.button_type = 'primary'

        if len(syllables['label']) > 0:
            lbl_name_input.value = syllables['label']

        if len(syllables['desc']) > 0:
            desc_input.value = syllables['desc']

        # update label
        cm_lbl.text = f'Crowd Movie {syll_select.index + 1}/{len(syll_select.options)}'

        # get movie path
        cm_path = syllables['crowd_movie_path']

        video_div = f'''
                        <h2>{syll_select.index}: {syllables['label']}</h2>
                        <video
                            src="{cm_path}"; alt="{cm_path}"; height="450"; width="450"; preload="true";
                            style="float: left; type: "video/mp4"; margin: 0px 10px 10px 0px;
                            border="2"; autoplay controls loop>
                        </video>
                    '''

        div = Div(text=video_div, style={'width': '100%'})
        out = widgets.Output(height='500px')
        layout = column([div, cm_lbl])

        with out:
            show(layout)

        grid = widgets.GridspecLayout(2, 2)
        grid[0, 0] = out
        grid[0, 1] = data_box
        grid[1, :2] = button_box

        display(grid)

    def get_crowd_movie_paths(self, crowd_movie_dir):
        '''

        Parameters
        ----------
        syll_info
        crowd_movie_dir

        Returns
        -------

        '''

        crowd_movie_paths = [f for f in glob(crowd_movie_dir + '*') if '.mp4' in f]

        for cm in crowd_movie_paths:
            syll_num = cm.split('sorted-id-')[1].split()[0]
            if syll_num in self.syll_info.keys():
                self.syll_info[syll_num]['crowd_movie_path'] = cm

class InteractiveSyllableStats:
    '''

    '''

    def __init__(self, index_path, model_path, info_path, max_sylls):
        '''

        Parameters
        ----------
        index_path
        model_path
        info_path
        max_sylls
        '''

        self.model_path = model_path
        self.info_path = info_path
        self.max_sylls = max_sylls
        self.index_path = index_path
        self.df = None

        self.ar_mats = None
        self.results = None
        self.icoord, self.dcoord = None, None

    def compute_dendrogram(self):
        '''

        Returns
        -------

        '''

        # Get Pairwise distances
        X = pairwise_distances(self.ar_mats, metric='euclidean')
        Z = linkage(X, 'ward')

        # Get Dendogram Metadata
        self.results = dendrogram(Z, distance_sort=True, no_plot=True, get_leaves=True)

        # Get Graph Info
        icoord, dcoord = self.results['icoord'], self.results['dcoord']

        icoord = pd.DataFrame(icoord) - 5
        icoord = icoord * (self.df['syllable'].max() / icoord.max().max())
        self.icoord = icoord.values

        dcoord = pd.DataFrame(dcoord)
        dcoord = dcoord * (self.df['usage'].max() / dcoord.max().max())
        self.dcoord = dcoord.values

    def interactive_stat_helper(self):
        '''

        Returns
        -------

        '''

        with open(self.info_path, 'r') as f:
            syll_info = yaml.safe_load(f)

        info_df = pd.DataFrame(list(syll_info.values()), index=[int(k) for k in list(syll_info.keys())]).sort_index()
        info_df['syllable'] = info_df.index

        model_data = parse_model_results(joblib.load(self.model_path))

        labels, mapping = relabel_by_usage(model_data['labels'], count='usage')

        ar_mats = np.array(model_data['model_parameters']['ar_mat'])
        self.ar_mats = np.reshape(ar_mats, (100, -1))[mapping][:self.max_sylls]

        syllable_usages = get_syllable_usages({'labels': labels}, count='usage')
        cumulative_explanation = 100 * np.cumsum(syllable_usages)
        if self.max_sylls == None:
            self.max_sylls = np.argwhere(cumulative_explanation >= 90)[0][0]

        sorted_index = parse_index(self.index_path)[1]

        # Load scalar Dataframe to compute syllable speeds
        scalar_df = scalars_to_dataframe(sorted_index)

        # Compute a syllable summary Dataframe containing usage-based
        # sorted/relabeled syllable usage and duration information from [0, max_syllable) inclusive
        df, label_df = results_to_dataframe(model_data, sorted_index, count='usage',
                                            max_syllable=self.max_sylls, sort=True, compute_labels=True)

        scalar_df['centroid_speed_mm'] = compute_session_centroid_speeds(scalar_df)
        df = compute_mean_syll_speed(df, scalar_df, label_df, groups=None, max_sylls=self.max_sylls)

        self.df = df.merge(info_df, on='syllable')

    def interactive_syll_stats_grapher(self, df, obj, stat, sort, groupby, sessions, ctrl_group, exp_group):
        '''

        Parameters
        ----------
        df
        obj
        stat
        sort
        groupby
        sessions
        ctrl_group
        exp_group

        Returns
        -------

        '''

        if sort == 'mutation':
            # display Text for groups to input experimental groups
            ordering = get_syllable_muteness_ordering(df, ctrl_group, exp_group, stat=stat)
        elif sort == 'similarity':
            ordering = self.results['leaves']
        elif sort != 'usage':
            ordering, _ = get_sorted_syllable_stat_ordering(df, stat=sort)
        else:
            ordering = range(len(df.syllable.unique()))

        if groupby == 'SessionName':
            session_sel.layout.display = "block"
            df = df[df['SessionName'].isin(session_sel.value)]
        else:
            session_sel.layout.display = "none"

        bokeh_plotting(df, stat, ordering, groupby)

class CrowdMovieComparison:
    '''
    Crowd Movie Comparison application class. Contains all the user inputted parameters
    within its context.

    '''

    def __init__(self, config_data, index_path, model_path, syll_info, output_dir):
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

        self.config_data = config_data
        self.index_path = index_path
        self.model_path = model_path
        self.syll_info = syll_info
        self.output_dir = output_dir
        self.max_sylls = config_data['max_syllable']
        self.session_dict = {str(i): {'session_info': {}} for i in range(self.max_sylls)}

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

        if change.new == 'SessionName':
            session_sel.layout = layout_visible
            self.config_data['separate_by'] = 'sessions'
        elif change.new == 'group':
            session_sel.layout = layout_hidden
            self.config_data['separate_by'] = 'groups'

    def select_session(self, event):
        '''
        Callback function to save the list of selected sessions to config_data
        to pass to crowd_movie_wrapper.

        Parameters
        ----------
        event (event): User clicks on multiple sessions in the SelectMultiple widget

        Returns
        -------
        '''

        self.config_data['session_names'] = list(session_sel.value)

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

        # Load scalar Dataframe to compute syllable speeds
        scalar_df = scalars_to_dataframe(sorted_index)

        # Compute a syllable summary Dataframe containing usage-based
        # sorted/relabeled syllable usage and duration information from [0, max_syllable) inclusive
        df, label_df = results_to_dataframe(model_fit, sorted_index, count='usage',
                                            max_syllable=self.max_sylls, sort=True, compute_labels=True)

        # Compute syllable speed
        scalar_df['centroid_speed_mm'] = compute_session_centroid_speeds(scalar_df)
        df = compute_mean_syll_speed(df, scalar_df, label_df, groups=None, max_sylls=self.max_sylls)

        # Get grouped DataFrame
        self.session_df = df.groupby(('SessionName', 'syllable'), as_index=False).mean()

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


    def crowd_movie_preview(self, config_data, syllable, groupby, sessions, nexamples):
        '''
        Helper function that triggers the crowd_movie_wrapper function and creates the HTML
        divs containing the generated crowd movies.
        Function is triggered whenever any of the widget function inputs are changed.

        Parameters
        ----------
        config_data (dict): Configuration parameters for creating crowd movies.
        syllable (int or ipywidgets.DropDownMenu): Currently displayed syllable.
        groupby (str or ipywidgets.DropDownMenu): Indicates source selection for crowd movies.
        sessions (list or ipywidgets.SelectMultiple): Specific session sources to show.
        nexamples (int or ipywidgets.IntSlider): Number of mice to display per crowd movie.

        Returns
        -------
        '''

        # Update current config data with widget values
        self.config_data['specific_syllable'] = int(syll_select.index)
        self.config_data['max_examples'] = nexamples

        # Compute paths to crowd movies
        path_dict = make_crowd_movies_wrapper(self.index_path, self.model_path, self.config_data, self.output_dir)
        clear_output()

        # Get group info based on selected DropDownMenu item
        if groupby == 'SessionName':
            self.get_selected_session_syllable_info(sessions)
            syll_dict = self.session_dict[str(syll_select.index)]['session_info']
        else:
            syll_dict = self.syll_info[str(syll_select.index)]['group_info']

        # Create video divs including syllable metadata
        divs = []
        for group_name, cm_path in path_dict.items():
            group_txt = '''
                <h2>{group_name}</h2>
                <video
                    src="{src}"; alt="{alt}"; height="350"; width="350"; preload="true";
                    style="float: center; type: "video/mp4"; margin: 0px 10px 10px 0px;
                    border="2"; autoplay controls loop>
                </video>
                <table style="display: inline-block;">
                    <tr style="text-align:center;">
                        <th> Usage: </th>
                        <th>{usage:.3f}</th>
                    </tr>
                    <tr>
                        <th> Speed: </th>
                        <th>{speed:.3f} mm/s</th>
                    </tr>
                    <tr>
                        <th> Duration: </th>
                        <th>{duration:.3f} ms</th>
                    </tr>
                </table>
            '''.format(group_name=group_name, usage=syll_dict[group_name]['usage'], speed=syll_dict[group_name]['speed'],
                       duration=syll_dict[group_name]['duration'], src=cm_path[0], alt=cm_path[0])

            divs.append(group_txt)

        # Display generated movies
        display_crowd_movies(divs)
