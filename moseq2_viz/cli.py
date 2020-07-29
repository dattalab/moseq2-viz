'''

CLI front-end operations. This module contains all the functionality and configurable parameters
users can alter to most accurately process their data.

Note: These functions simply read all the parameters into a dictionary,
 and then call the corresponding wrapper function with the given input parameters.

'''

import os
import click
from moseq2_viz.helpers.wrappers import add_group_wrapper, plot_syllable_stat_wrapper, plot_scalar_summary_wrapper, \
        plot_transition_graph_wrapper, copy_h5_metadata_to_yaml_wrapper, make_crowd_movies_wrapper, \
        plot_verbose_pdfs_wrapper, plot_mean_group_position_pdf_wrapper

orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init


@click.group()
@click.version_option()
def cli():
    pass

@cli.command(name="add-group", help='Change group name in index file given a key-value pair')
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.option('--key', '-k', type=str, default='SubjectName', help='Key to search for value')
@click.option('--value', '-v', type=str, default='Mouse', help='Value to search for', multiple=True)
@click.option('--group', '-g', type=str, default='Group1', help='Group name to map to')
@click.option('--exact', '-e', type=bool, is_flag=True, help='Exact match only')
@click.option('--lowercase', type=bool, is_flag=True, help='Lowercase text filter')
@click.option('-n', '--negative', type=bool, is_flag=True, help='Negative match (everything that does not match is included)')
def add_group(index_file, key, value, group, exact, lowercase, negative):

    click_data = click.get_current_context().params
    add_group_wrapper(index_file, click_data)



# recurse through directories, find h5 files with completed extractions, make a manifest
# and copy the contents to a new directory
@cli.command(name="copy-h5-metadata-to-yaml", help='Copies metadata within an h5 file to a yaml file.')
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--h5-metadata-path', default='/metadata/acquisition', type=str, help='Path to acquisition metadata in h5 files')
def copy_h5_metadata_to_yaml(input_dir, h5_metadata_path):
    copy_h5_metadata_to_yaml_wrapper(input_dir, h5_metadata_path)


@cli.command(name='make-crowd-movies', help='Writes movies of overlaid examples of the rodent perform a given syllable')
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.argument('model-path', type=click.Path(exists=True, resolve_path=True))
@click.option('--max-syllable', type=int, default=40, help="Index of max syllable to render")
@click.option('--max-examples', '-m', type=int, default=40, help="Number of examples to show")
@click.option('--threads', '-t', type=int, default=-1, help="Number of threads to use for rendering crowd movies")
@click.option('--sort', type=bool, default=True, help="Sort syllables by usage")
@click.option('--count', type=click.Choice(['usage', 'frames']), default='usage', help='How to quantify syllable usage')
@click.option('--output-dir', '-o', type=click.Path(), default=os.path.join(os.getcwd(), 'crowd_movies'), help="Path to store files")
@click.option('--gaussfilter-space', default=(0, 0), type=(float, float), help="Spatial filter for data (Gaussian)")
@click.option('--medfilter-space', default=[0], type=int, help="Median spatial filter", multiple=True)
@click.option('--min-height', type=int, default=5, help="Minimum height for scaling videos")
@click.option('--max-height', type=int, default=80, help="Minimum height for scaling videos")
@click.option('--raw-size', type=(int, int), default=(512, 424), help="Size of original videos")
@click.option('--scale', type=float, default=1, help="Scaling from pixel units to mm")
@click.option('--cmap', type=str, default='jet', help="Name of valid Matplotlib colormap for false-coloring images")
@click.option('--dur-clip', default=300, help="Exclude syllables more than this number of frames (None for no limit)")
@click.option('--legacy-jitter-fix', default=False, type=bool, help="Set to true if you notice jitter in your crowd movies")
@click.option('--frame-path', default='frames', type=str, help='Path to depth frames in h5 file')
def make_crowd_movies(index_file, model_path, max_syllable, max_examples, threads, sort, count,
                      output_dir, min_height, max_height, raw_size, scale, cmap, dur_clip,
                      legacy_jitter_fix, frame_path, gaussfilter_space, medfilter_space):

    click_data = click.get_current_context().params
    make_crowd_movies_wrapper(index_file, model_path, click_data, output_dir)

    print(f'Crowd movies successfully generated in {output_dir}.')


@cli.command(name='plot-scalar-summary', help="Plots a scalar summary of the index file data.")
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.option('--output-file', type=click.Path(), default=os.path.join(os.getcwd(), 'scalars'))
@click.option('-c', '--colors', type=str, default=None, help="Colors to plot groups with.", multiple=True)
def plot_scalar_summary(index_file, output_file, colors):

    plot_scalar_summary_wrapper(index_file, output_file, colors=colors)
    print('Sucessfully plotted scalar summary')


@cli.command(name='plot-group-position-heatmaps', help="Plots position heatmaps for each group in the index file")
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.option('--output-file', type=click.Path(), default=os.path.join(os.getcwd(), 'scalars'))
def plot_group_position_heatmaps(index_file, output_file):

    plot_mean_group_position_pdf_wrapper(index_file, output_file)
    print('Sucessfully plotted mean group heatmaps')

@cli.command(name='plot-verbose-position-heatmaps', help="Plots a position heatmap for each session in the index file.")
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.option('--output-file', type=click.Path(), default=os.path.join(os.getcwd(), 'scalars'))
def plot_verbose_position_heatmaps(index_file, output_file):

    plot_verbose_pdfs_wrapper(index_file, output_file)
    print('Sucessfully plotted mean group heatmaps')


@cli.command(name='plot-transition-graph', help="Plots the transition graph depicting the transition probabilities between syllables.")
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.argument('model-fit', type=click.Path(exists=True, resolve_path=True))
@click.option('--max-syllable', type=int, default=40, help="Index of max syllable to render")
@click.option('-g', '--group', type=str, default=None, help="Name of group(s) to show", multiple=True)
@click.option('--output-file', type=click.Path(), default=os.path.join(os.getcwd(), 'transitions'), help="Filename to store plot")
@click.option('--normalize', type=click.Choice(['bigram', 'rows', 'columns']), default='bigram', help="How to normalize transition probabilities")
@click.option('--edge-threshold', type=float, default=.001, help="Threshold for edges to show")
@click.option('--usage-threshold', type=float, default=0, help="Threshold for nodes to show")
@click.option('--layout', type=str, default='spring', help="Default networkx layout algorithm")
@click.option('--keep-orphans', '-k', type=bool, is_flag=True, help="Show orphaned nodes")
@click.option('--orphan-weight', type=float, default=0, help="Weight for non-existent connections")
@click.option('--arrows', type=bool, is_flag=True, help="Show arrows")
@click.option('--sort', type=bool, default=True, help="Sort syllables by usage")
@click.option('--count', type=click.Choice(['usage', 'frames']), default='usage', help='How to quantify syllable usage')
@click.option('--edge-scaling', type=float, default=250, help="Scale factor from transition probabilities to edge width")
@click.option('--node-scaling', type=float, default=1e4, help="Scale factor for nodes by usage")
@click.option('--scale-node-by-usage', type=bool, default=True, help="Scale node sizes by usages probabilities")
@click.option('--width-per-group', type=float, default=8, help="Width (in inches) for figure canvas per group")
def plot_transition_graph(index_file, model_fit, max_syllable, group, output_file,
                          normalize, edge_threshold, usage_threshold, layout,
                          keep_orphans, orphan_weight, arrows, sort, count,
                          edge_scaling, node_scaling, scale_node_by_usage, width_per_group):

    click_data = click.get_current_context().params
    plot_transition_graph_wrapper(index_file, model_fit, click_data, output_file)


@cli.command(name='plot-stats', help="Plots syllable usages with different sorting,coloring and grouping capabilities")
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.argument('model-fit', type=click.Path(exists=True, resolve_path=True))
@click.option('--stat', type=str, default='usage', help="Statistic to plot ['usage', 'speed', 'duration'")
@click.option('--output-file', type=click.Path(), default=os.path.join(os.getcwd(), 'usages'), help="Filename to store plot")
@click.option('--sort', type=bool, default=True, help="Sort syllables by usage")
@click.option('--figsize', type=tuple, default=(10, 5), help="Size dimensions of the plotted figure.")
@click.option('--count', type=click.Choice(['usage', 'frames']), default='usage', help='How to quantify syllable usage')
@click.option('--max-syllable', type=int, default=40, help="Index of max syllable to render")
@click.option('-g', '--group', type=str, default=None, help="Name of group(s) to show", multiple=True)
@click.option('-o', '--ordering', type=str, default=None,
              help="How to order the groups, ['any' for descending, 'm' for muteness]")
@click.option('--ctrl-group', type=str, default=None, help="Name of control group. Only if ordering = 'm'")
@click.option('--exp-group', type=str, default=None, help="Name of experimental group. Only if ordering = 'm'")
@click.option('-c', '--colors', type=str, default=None, help="Colors to plot groups with.", multiple=True)
@click.option('-f', '--fmt', type=str, default='o-', help="Format the scatter plot data.")
def plot_stats(index_file, model_fit, stat, output_file, sort, count, max_syllable, group,
                ordering, ctrl_group, exp_group, colors, fmt, figsize):

    plot_syllable_stat_wrapper(model_fit, index_file, output_file, stat=stat, max_syllable=max_syllable, sort=sort,
                                 count=count, group=group, ordering=ordering, ctrl_group=ctrl_group,
                                 exp_group=exp_group, colors=colors, fmt=fmt, figsize=figsize)

    print('Successfully graphed usage plots')