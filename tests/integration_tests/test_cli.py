import os
import shutil
from unittest import TestCase
from click.testing import CliRunner
from moseq2_viz.cli import add_group, copy_h5_metadata_to_yaml, plot_scalar_summary, \
    plot_group_position_heatmaps, plot_verbose_position_heatmaps, plot_transition_graph, plot_stats, make_crowd_movies


class TestCLI(TestCase):
    def test_add_group(self):
        input_dir = 'data/'
        input_path = os.path.join(input_dir, 'test_index.yaml')

        original_file = 'data/orig.txt'
        with open(original_file, 'w') as f:
            with open(input_path, 'r') as g:
                f.write(g.read())

        runner = CliRunner()

        group_params = ['-k', 'SubjectName',
                        '-v', 'Mouse',
                        '-g', 'Group1',
                        '-e',  # FLAG
                        '--lowercase',  # FLAG
                        '-n',  # FLAG
                        input_path]

        results = runner.invoke(add_group, group_params)
        assert (results.exit_code == 0), "CLI Command did not complete successfully"
        assert (not os.path.samefile(os.path.join(input_dir, 'orig.txt'), input_path)), "Index file was not updated."
        os.remove(original_file)

    def test_copy_h5_metadata_to_yaml(self):
        input_dir = 'data/'

        runner = CliRunner()
        results = runner.invoke(copy_h5_metadata_to_yaml, ['--h5-metadata-path', '/metadata/acquisition',
                                                           '-i', input_dir])

        assert (results.exit_code == 0)

    def test_plot_scalar_summary(self):
        input_dir = 'data/'
        gen_dir = 'data/gen_plots/'
        runner = CliRunner()

        results = runner.invoke(plot_scalar_summary, ['--output-file', gen_dir + 'scalar',
                                                      input_dir + 'test_index.yaml'])

        assert (results.exit_code == 0), "CLI Command did not complete successfully"
        assert (os.path.exists(gen_dir + 'scalar_position.png')), "Position summary PNG not found"
        assert (os.path.exists(gen_dir + 'scalar_position.pdf')), "Position summary PDF not found"
        assert (os.path.exists(gen_dir + 'scalar_summary.png')), "Scalar summary PNG not found"
        assert (os.path.exists(gen_dir + 'scalar_summary.pdf')), "Scalar summary PDF not found"
        shutil.rmtree(gen_dir)

    def plot_group_position_heatmaps(self):
        input_dir = 'data/'
        gen_dir = 'data/gen_plots/'
        runner = CliRunner()

        results = runner.invoke(plot_group_position_heatmaps, ['--output-file', gen_dir + 'group_heatmap',
                                                      input_dir + 'test_index.yaml'])

        assert (results.exit_code == 0), "CLI Command did not complete successfully"
        assert (os.path.exists(gen_dir + 'group_heatmap.png')), "Position Heatmap PNG not found"
        assert (os.path.exists(gen_dir + 'group_heatmap.pdf')), "Position Heatmap PDF file not found"
        shutil.rmtree(gen_dir)

    def test_plot_verbose_position_heatmaps(self):
        input_dir = 'data/'
        gen_dir = 'data/gen_plots/'
        runner = CliRunner()

        results = runner.invoke(plot_verbose_position_heatmaps, ['--output-file', gen_dir + 'heatmaps',
                                                      input_dir + 'test_index.yaml'])

        assert (results.exit_code == 0), "CLI Command did not complete successfully"
        assert (os.path.exists(gen_dir + 'heatmaps.png')), "Position Heatmap PNG not found"
        assert (os.path.exists(gen_dir + 'heatmaps.pdf')), "Position Heatmap PDF file not found"
        shutil.rmtree(gen_dir)

    def test_plot_transition_graph(self):
        input_dir = 'data/'
        gen_dir = 'data/gen_plots/'
        runner = CliRunner()

        trans_params = ['--output-file', gen_dir + 'transitions',
                        '--max-syllable', 40,
                        '-g', 'Group1',
                        '--arrows',  # FLAG
                        '--normalize', 'bigram',
                        '--layout', 'spring',
                        '--sort', True,
                        '--count', 'usage',
                        '--edge-scaling', 250,
                        '--node-scaling', 1e4,
                        '--scale-node-by-usage', True,
                        '--width-per-group', 8,
                        input_dir + 'test_index.yaml',
                        input_dir + 'test_model.p']

        results = runner.invoke(plot_transition_graph, trans_params)

        assert (results.exit_code == 0), "CLI Command did not complete successfully"
        assert (os.path.exists(gen_dir + 'transitions.png')), "Transition graph PNG not found"
        assert (os.path.exists(gen_dir + 'transitions.pdf')), "Transition graph PDF not found"
        shutil.rmtree(gen_dir)

    def test_plot_all_stats(self):

        for stat in ['usage', 'speed', 'duration']:
            gen_dir = 'data/gen_plots/'

            runner = CliRunner()

            use_params = ['data/test_index.yaml',
                          'data/test_model.p',
                          '--output-file', gen_dir + f'test_{stat}s',
                          '--stat', stat]

            results = runner.invoke(plot_stats, use_params)

            assert (results.exit_code == 0), "CLI Command did not complete successfully"
            assert (os.path.exists(gen_dir + f'test_{stat}s.png')), f"f'{stat} plot PNG not found"
            assert (os.path.exists(gen_dir + f'test_{stat}s.pdf')), f"{stat} plot PDF not found"
            os.remove(gen_dir + f'test_{stat}s.png')
            os.remove(gen_dir + f'test_{stat}s.pdf')
            os.removedirs(gen_dir)

    def test_make_crowd_movies(self):
        input_dir = 'data/'
        crowd_dir = input_dir + 'crowd_movies/'
        max_examples = 40
        max_syllable = 5
        runner = CliRunner()

        crowd_params = [input_dir + 'test_index_crowd.yaml',
                        input_dir + 'mock_model.p',
                        '-o', crowd_dir,
                        '--max-syllable', max_syllable,
                        '--count', 'usage',
                        '--min-height', 5,
                        '--max-height', 80,
                        '--raw-size', 512, 424,
                        '--scale', 1,
                        '--cmap', 'jet',
                        '--dur-clip', 300,
                        '--legacy-jitter-fix', False,
                        '--max-examples', max_examples]

        results = runner.invoke(make_crowd_movies, crowd_params)

        assert (results.exit_code == 0), "CLI Command did not complete successfully"
        assert (os.path.exists(crowd_dir)), "Crowd movies directory was not found"
        assert (len(os.listdir(crowd_dir)) == max_syllable + 1), "Number of crowd movies does not match max syllables"
        shutil.rmtree(crowd_dir)