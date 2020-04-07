import os
import click
from moseq2_viz.helpers.wrappers import add_group_wrapper, plot_syllable_usages_wrapper, plot_scalar_summary_wrapper, \
        plot_syllable_durations_wrapper, plot_transition_graph_wrapper, copy_h5_metadata_to_yaml_wrapper, \
        make_crowd_movies_wrapper

orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init


@click.group()
def cli():
    pass


@cli.command(name="add-group")
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.option('--key', '-k', type=str, default='SubjectName', help='Key to search for value')
@click.option('--value', '-v', type=str, default='Mouse', help='Value to search for', multiple=True)
@click.option('--group', '-g', type=str, default='Group1', help='Group name to map to')
@click.option('--exact', '-e', type=bool, is_flag=True, help='Exact match only')
@click.option('--lowercase', type=bool, is_flag=True, help='Lowercase text filter')
@click.option('-n', '--negative', type=bool, is_flag=True, help='Negative match (everything that does not match is included)')
def add_group(index_file, key, value, group, exact, lowercase, negative):

    click_data = click.get_current_context().params
    cli_data = {k: v for k, v in click_data.items()}

    add_group_wrapper(index_file, cli_data)



# recurse through directories, find h5 files with completed extractions, make a manifest
# and copy the contents to a new directory
@cli.command(name="copy-h5-metadata-to-yaml")
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--h5-metadata-path', default='/metadata/acquisition', type=str, help='Path to acquisition metadata in h5 files')
def copy_h5_metadata_to_yaml(input_dir, h5_metadata_path):
    copy_h5_metadata_to_yaml_wrapper(input_dir, h5_metadata_path)


@cli.command(name='make-crowd-movies')
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
    cli_data = {k: v for k, v in click_data.items()}
    make_crowd_movies_wrapper(index_file, model_path, cli_data, output_dir)

    print(f'Crowd movies successfully generated in {output_dir}.')


@cli.command(name='plot-scalar-summary')
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.option('--output-file', type=click.Path(), default=os.path.join(os.getcwd(), 'scalars'))
def plot_scalar_summary(index_file, output_file):

    plot_scalar_summary_wrapper(index_file, output_file)
    print('Sucessfully plotted scalar summary')


@cli.command(name='plot-transition-graph')
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
    cli_data = {k: v for k, v in click_data.items()}
    plot_transition_graph_wrapper(index_file, model_fit, cli_data, output_file)


@cli.command(name='plot-usages')
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.argument('model-fit', type=click.Path(exists=True, resolve_path=True))
@click.option('--sort', type=bool, default=True, help="Sort syllables by usage")
@click.option('--count', type=click.Choice(['usage', 'frames']), default='usage', help='How to quantify syllable usage')
@click.option('--max-syllable', type=int, default=40, help="Index of max syllable to render")
@click.option('-g', '--group', type=str, default=None, help="Name of group(s) to show", multiple=True)
@click.option('--output-file', type=click.Path(), default=os.path.join(os.getcwd(), 'usages'), help="Filename to store plot")
def plot_usages(index_file, model_fit, sort, count, max_syllable, group, output_file):

    plot_syllable_usages_wrapper(index_file, model_fit, max_syllable, sort, count, group, output_file)
    print('Successfully graphed usage plots')


@cli.command(name='plot-syllable-durations')
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.argument('model-fit', type=click.Path(exists=True, resolve_path=True))
@click.option('-g', '--group', type=str, default=None, help="Name of group(s) to show", multiple=True)
@click.option('--count', type=click.Choice(['usage', 'frames']), default='usage', help='How to quantify syllable usage')
@click.option('--output-file', type=click.Path(), default=os.path.join(os.getcwd(), 'durations'), help="Filename to store plot")
@click.option('--max-syllable', type=int, default=40, help="Index of max syllable to render")
def plot_syllable_durations(index_file, model_fit, group, count, output_file, max_syllable):

    plot_syllable_durations_wrapper(index_file, model_fit, group, count, max_syllable, output_file)
