import itertools
import ipywidgets as widgets
from bokeh.models.tickers import FixedTicker
from bokeh.palettes import Dark2_5 as palette

from bokeh.models import Div
from IPython.display import display
from bokeh.plotting import figure, show
from moseq2_viz.interactive.widgets import widget_box
from bokeh.models import ColumnDataSource, HoverTool, TapTool, BoxSelectTool

color_dict = {'b': 'blue',
              'r': 'red',
              'g': 'green',
              'c': 'cyan'}

def graph_dendrogram(obj):
    '''

    Parameters
    ----------
    obj

    Returns
    -------

    '''

    ## Cladogram figure
    cladogram = figure(title='Syllable Stat',
                       width=525,
                       height=350,
                       output_backend="webgl")

    # Show syllable info on hover
    cladogram.add_tools(HoverTool(tooltips=[('label', '@labels')]), TapTool(), BoxSelectTool())

    # Compute dendrogram
    obj.compute_dendrogram()

    # Get distance sorted label ordering
    labels = list(map(int, obj.results['ivl']))
    sources = []
    for ii, (i, d) in enumerate(zip(obj.icoord, obj.dcoord)):
        d = list(map(lambda x: x, d))
        tmp = [[x, y] for x, y in zip(i, d)]
        lbls = []
        groups = {c: i for i, c in enumerate(list(color_dict.keys()))}

        # Get labels
        for t in tmp:
            if t[1] == 0:
                lbls.append(labels[int(t[0])])
            else:
                lbls.append(f'group {groups[obj.results["color_list"][ii]]}')

        # Set coordinate DataSource
        source = ColumnDataSource(dict(x=i,
                                       y=d,
                                       labels=lbls))
        sources.append(source)

        # Draw glyphs
        cladogram.line(x='x', y='y', source=source, line_color=color_dict[obj.results['color_list'][ii]])
        cladogram.circle(x='x', y='y', source=source, line_color=color_dict[obj.results['color_list'][ii]])

    # Set x-axis ticks
    cladogram.xaxis.ticker = FixedTicker(ticks=labels)
    cladogram.xaxis.major_label_overrides = {i: str(l) for i, l in enumerate(labels)}

    show(cladogram)


def bokeh_plotting(df, stat, sorting, groupby='group'):
    '''

    Parameters
    ----------
    df
    stat
    sorting
    groupby

    Returns
    -------

    '''

    tools = 'pan, box_zoom, wheel_zoom, hover, reset'

    # Hover tool to display crowd movie
    cm_tooltip = """
        <div>
            <div>
                <video
                    src="@movies" height="260" alt="@movies" width="260"; preload="true";
                    style="float: left; type: "video/mp4"; "margin: 0px 15px 15px 0px;"
                    border="2"; autoplay loop
                ></video>
            </div>
        </div>
        """

    # Instantiate Bokeh figure with the HoverTool data
    p = figure(title='Syllable Stat',
               width=850,
               height=500,
               tools=tools,
               tooltips=[
                         ("syllable", "@number{0}"),
                         (f'{stat}', "$y{0.000}"),
                         ('SEM', '@sem{0.000}'),
                         ('label', '@label'),
                         ('description', '@desc'),
                         ('crowd movie', cm_tooltip)
                         ],
               output_backend="webgl")

    # TODO: allow users to set their own colors
    colors = itertools.cycle(palette)

    # Set grouping variable to plot separately
    if groupby == 'group':
        groups = list(df.group.unique())
    else:
        groups = list(df.SessionName.unique())

    ## Line Plot
    for i, color in zip(range(len(groups)), colors):
        # Get mean syllable data
        data_df = df[df[groupby] == groups[i]][['syllable', stat]].groupby('syllable', as_index=False).mean().reindex(sorting)

        # Get SEM values
        sem = df.groupby('syllable')[[stat]].sem().reindex(sorting)
        miny = data_df[stat] - sem[stat]
        maxy = data_df[stat] + sem[stat]

        errs_x = [(i, i) for i in range(len(data_df.index))]
        errs_y = [(min_y, max_y) for min_y, max_y in zip(miny, maxy)]

        # Get Labeled Syllable Information
        desc_data = df.groupby(['syllable', 'label', 'desc', 'crowd_movie_path'], as_index=False).mean()[
            ['syllable', 'label', 'desc', 'crowd_movie_path']].reindex(sorting)

        labels = desc_data['label'].to_numpy()
        desc = desc_data['desc'].to_numpy()
        cm_paths = desc_data['crowd_movie_path'].to_numpy()

        # stat data source
        source = ColumnDataSource(data=dict(
            x=range(len(data_df.index)),
            y=data_df[stat].to_numpy(),
            sem=sem[stat].to_numpy(),
            number=sem.index,
            label=labels,
            desc=desc,
            movies=cm_paths,
        ))

        # sem data source
        err_source = ColumnDataSource(data=dict(
            x=errs_x,
            y=errs_y,
            sem=sem[stat].to_numpy(),
            number=sem.index,
            label=labels,
            desc=desc,
            movies=cm_paths,
        ))

        # Draw glyphs
        p.line('x', 'y', source=source, alpha=0.8, muted_alpha=0.2, legend_label=groups[i], color=color)
        p.circle('x', 'y', source=source, alpha=0.8, muted_alpha=0.2, legend_label=groups[i], color=color, size=6)
        p.multi_line('x', 'y', source=err_source, alpha=0.8, muted_alpha=0.2, color=color)

    # Setting dynamics xticks
    p.xaxis.ticker = FixedTicker(ticks=list(sorting))
    p.xaxis.major_label_overrides = {i: str(l) for i, l in enumerate(list(sorting))}

    # Setting interactive legend
    p.legend.click_policy = "mute"
    p.legend.location = "top_right"

    ## Display
    show(p)

def display_crowd_movies(divs):
    '''
    Crowd movie comparison helper function that displays the widgets and
    embedded HTML divs to a running jupyter notebook cell or HTML webpage.

    Parameters
    ----------
    divs (list of bokeh.models.Div): list of HTML Div objects containing videos to display

    Returns
    -------
    '''

    # Set HTML formats
    movie_table = '''
                    <head>
                    <style>
                        .output {
                            display: contents;
                            height: auto;
                        }
                        .row {
                            display: flex;
                            flex-wrap: wrap;
                            vertical-align: center;
                            width: 900px;
                            text-align: center;
                        }
        
                        .column {
                            width: 50%;
                            text-align: center;
                        }
        
                        .column video {
                          vertical-align: center;
                        }
                    </style>
                    </head>
                    <div class="row"; style="background-color:#ffffff; height:auto;">
                  '''

    # Create div grid
    for i, div in enumerate(divs):
        if (i % 2 == 0) and i > 0:
            # make a new row
            movie_table += '</div>'
            col = f'''
                      <div class="row"; style="background-color:#ffffff; height:auto;">
                          <div class="column"; style="height:auto;">
                              {div}
                          </div>
                    '''
        else:
            # put movie in column
            col = f'''
                      <div class="column"; style="height:auto;">
                          {div}
                      </div>
                    '''
        movie_table += col
    # Close last div
    movie_table += '</div>'

    div2 = Div(text=movie_table, height_policy='auto')

    # Display
    div_out = widgets.Output(height='auto')
    with div_out:
        show(div2)

    display(widget_box, div_out)
    #print(movie_table)