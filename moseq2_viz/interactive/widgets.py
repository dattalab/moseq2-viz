'''

This file contains classes for all the widgets that facilitate the interactive
 functionality in their extended (child) classes.

'''
import qgrid
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from bokeh.models.widgets import PreText

class GroupSettingWidgets:

    def __init__(self):
        style = {'description_width': 'initial', 'display': 'flex-grow', 'align_items': 'stretch'}

        self.col_opts = {
            'editable': False,
            'toolTip': "Not editable"
        }

        self.col_defs = {
            'group': {
                'editable': True,
                'toolTip': 'editable'
            }
        }

        self.group_input = widgets.Text(value='', placeholder='Enter Group Name to Set', style=style,
                                        description='Desired Group Name', continuous_update=False, disabled=False)
        self.save_button = widgets.Button(description='Set Group', style=style,
                                          disabled=False, tooltip='Set Group')
        self.update_index_button = widgets.Button(description='Update Index File', style=style,
                                                  disabled=False, tooltip='Save Parameters')

        self.group_set = HBox([self.group_input, self.save_button, self.update_index_button])
        qgrid.set_grid_option('forceFitColumns', False)
        qgrid.set_grid_option('enableColumnReorder', True)
        qgrid.set_grid_option('highlightSelectedRow', True)
        qgrid.set_grid_option('highlightSelectedCell', False)

class SyllableLabelerWidgets:

    def __init__(self):

        self.syll_select = widgets.Dropdown(options={}, description='Syllable #:', disabled=False)

        # labels
        self.cm_lbl = PreText(text="Crowd Movie") # current crowd movie number

        self.syll_lbl = widgets.Label(value="Syllable Name") # name user prompt label
        self.desc_lbl = widgets.Label(value="Short Description") # description label

        self.syll_info_lbl = widgets.Label(value="Syllable Info", font_size=24)

        self.syll_usage_value_lbl = widgets.Label(value="")
        self.syll_speed_value_lbl = widgets.Label(value="")
        self.syll_duration_value_lbl = widgets.Label(value="")

        # text input widgets
        self.lbl_name_input = widgets.Text(value='',
                                    placeholder='Syllable Name',
                                    tooltip='2 word name for syllable')

        self.desc_input = widgets.Text(value='',
                                placeholder='Short description of behavior',
                                tooltip='Describe the behavior.',
                                layout=widgets.Layout(height='260px'),
                                disabled=False)

        # buttons
        self.prev_button = widgets.Button(description='Prev', disabled=False, tooltip='Previous Syllable', layout=widgets.Layout(flex='2 1 0', width='auto', height='40px'))
        self.set_button = widgets.Button(description='Save Setting', disabled=False, tooltip='Save current inputs.', button_style='primary', layout=widgets.Layout(flex='3 1 0', width='auto', height='40px'))
        self.next_button = widgets.Button(description='Next', disabled=False, tooltip='Next Syllable', layout=widgets.Layout(flex='2 1 0', width='auto', height='40px'))

        # Box Layouts
        self.label_layout = widgets.Layout(flex_flow='column', height='75%')
        self.input_layout = widgets.Layout(height='200px')

class SyllableStatWidgets:

    def __init__(self):

        self.layout_hidden = widgets.Layout(display='none')
        self.layout_visible = widgets.Layout(display='block')

        self.stat_dropdown = widgets.Dropdown(options=['usage', 'speed', 'distance to center'], description='Stat to Plot:', disabled=False)

        # add dist to center
        self.sorting_dropdown = widgets.Dropdown(options=['usage', 'speed', 'distance to center', 'similarity', 'difference'], description='Sorting:', disabled=False)
        self.ctrl_dropdown = widgets.Dropdown(options=[], description='Group 1:', disabled=False)
        self.exp_dropdown = widgets.Dropdown(options=[], description='Group 2:', disabled=False)

        self.grouping_dropdown = widgets.Dropdown(options=['group', 'SessionName'], description='Grouping:', disabled=False)
        self.session_sel = widgets.SelectMultiple(options=[], description='Sessions:', rows=10,
                                                  layout=self.layout_hidden, disabled=False)

        self.errorbar_dropdown = widgets.Dropdown(options=['SEM', 'STD'], description='Error Bars:', disabled=False)

        ## boxes
        self.stat_box = VBox([self.stat_dropdown, self.errorbar_dropdown])
        self.mutation_box = VBox([self.ctrl_dropdown, self.exp_dropdown])

        self.sorting_box = VBox([self.sorting_dropdown, self.mutation_box])
        self.session_box = VBox([self.grouping_dropdown, self.session_sel])

class CrowdMovieCompareWidgets:

    def __init__(self):
        style = {'description_width': 'initial'}

        self.label_layout = widgets.Layout(flex_flow='column', max_height='100px')
        self.layout_hidden = widgets.Layout(display='none', align_items='stretch')
        self.layout_visible = widgets.Layout(display='flex',  align_items='stretch', justify_items='center')

        self.cm_syll_select = widgets.Dropdown(options=[], description='Syllable #:', disabled=False)
        self.num_examples = widgets.IntSlider(value=20, min=1, max=40, step=1, description='# of Example Mice:',
                                              disabled=False, continuous_update=False, style=style,
                                              layout=widgets.Layout(display='flex', align_items='stretch'))

        self.cm_sources_dropdown = widgets.Dropdown(options=['group', 'SessionName'], style=style,
                                                    description='Movie Sources:')

        self.cm_session_sel = widgets.SelectMultiple(options=[], description='Sessions:', rows=10,
                                                     style=style, layout=self.layout_hidden)

        self.cm_trigger_button = widgets.Button(description='Generate Movies',
                                                tooltip='Make Crowd Movies',
                                                layout=widgets.Layout(display='none', width='100%',
                                                                      align_items='stretch'))

        self.syllable_box = VBox([self.cm_syll_select, self.num_examples])

        self.session_box = VBox([self.cm_sources_dropdown, self.cm_session_sel, self.cm_trigger_button])

        self.widget_box = HBox([self.syllable_box, self.session_box],
                               layout=widgets.Layout(flex_flow='row',
                                                     border='solid',
                                                     width='100%',
                                                     justify_content='space-around'))