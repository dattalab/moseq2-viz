'''

Wrapper functions for all the interactive functionality in moseq2-viz.

'''

import qgrid
import pandas as pd
import ruamel.yaml as yaml
from IPython.display import display
from moseq2_viz.util import index_to_dataframe
from moseq2_viz.interactive.widgets import GroupSettingWidgets

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