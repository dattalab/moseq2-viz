'''

This file contains classes for all the widgets that facilitate the interactive
 functionality in their extended (child) classes.

'''
import qgrid
from ipywidgets import HBox
import ipywidgets as widgets

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