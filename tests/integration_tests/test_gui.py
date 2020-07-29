import os
import shutil
import ruamel.yaml as yaml
from unittest import TestCase
from moseq2_viz.gui import get_groups_command, add_group, plot_stats_command, \
                plot_scalar_summary_command, plot_transition_graph_command, \
                plot_mean_group_position_heatmaps_command, \
                plot_verbose_position_heatmaps

class TestGUI(TestCase):

    def test_get_groups_command(self):
        index_path = 'data/test_index.yaml'

        with open(index_path, 'r') as f:
            index_data = yaml.safe_load(f)

        groups, uuids = [], []
        subjectNames, sessionNames = [], []
        for f in index_data['files']:
            if f['uuid'] not in uuids:
                uuids.append(f['uuid'])
                groups.append(f['group'])
                subjectNames.append(f['metadata']['SubjectName'])
                sessionNames.append(f['metadata']['SessionName'])

        num_groups = get_groups_command(index_path)

        assert num_groups == len(set(groups)), "Number of returned groups is incorrect"


    def test_add_group(self):
        index_path = 'data/test_index.yaml'
        tmp_yaml = 'data/tmp_copy.yaml'

        with open(index_path, 'r') as f:
            index_data = yaml.safe_load(f)

        with open(tmp_yaml, 'w') as g:
            yaml.safe_dump(index_data, g)

        key = 'SubjectName'
        value = '012517'
        group = 'test1'
        exact = False
        lowercase = False
        negative = False

        add_group(tmp_yaml, key, value, group, exact, lowercase, negative)

        assert not os.path.samefile(index_path, tmp_yaml), "Index file was not updated"

        value = ['012517']
        group = ['test12']

        add_group(tmp_yaml, key, value, group, exact, lowercase, negative)

        assert not os.path.samefile(index_path, tmp_yaml), "Index file was not updated"
        os.remove(tmp_yaml)

    def test_plot_all_stats(self):

        for stat in ['usage', 'speed', 'duration']:

            gen_dir = 'data/gen_plots/'
            index_file = 'data/test_index.yaml'
            model_path = 'data/test_model.p'
            output_file = gen_dir+f'test_{stat}s'

            plot_stats_command(model_path, index_file, output_file)

            assert (os.path.exists(gen_dir + f'test_{stat}s.png')), f"{stat} PNG plot was not saved"
            assert (os.path.exists(gen_dir + f'test_{stat}s.pdf')), f"{stat} PDF plot was not saved"
            shutil.rmtree(gen_dir)

    def test_plot_scalar_summary_command(self):
        index_file = 'data/test_index.yaml'
        gen_dir = 'data/gen_plots/'
        output_file = gen_dir + 'test_scalar'

        df = plot_scalar_summary_command(index_file, output_file)

        assert not df.empty, "Scalar DataFrame was not return correctly; is empty."
        assert (os.path.exists(gen_dir + 'test_scalar_position.png')), "Position Summary PNG plot was not saved"
        assert (os.path.exists(gen_dir + 'test_scalar_position.pdf')), "Position Summary PDF plot was not saved"
        assert (os.path.exists(gen_dir + 'test_scalar_summary.png')), "Scalar Summary PNG plot was not saved"
        assert (os.path.exists(gen_dir + 'test_scalar_summary.pdf')), "Scalar Summary PDF plot was not saved"
        shutil.rmtree(gen_dir)

    def test_plot_mean_group_position_heatmaps_command(self):
        index_file = 'data/test_index.yaml'
        gen_dir = 'data/gen_plots/'
        output_file = gen_dir + 'test_gHeatmaps'

        _ = plot_mean_group_position_heatmaps_command(index_file, output_file)

        assert (os.path.exists(gen_dir + 'test_gHeatmaps.png')), "Position Summary PNG plot was not saved"
        assert (os.path.exists(gen_dir + 'test_gHeatmaps.pdf')), "Position Summary PDF plot was not saved"
        shutil.rmtree(gen_dir)

    def test_plot_verbose_position_heatmaps(self):
        index_file = 'data/test_index.yaml'
        gen_dir = 'data/gen_plots/'
        output_file = gen_dir + 'test_vHeatmaps'

        _ = plot_verbose_position_heatmaps(index_file, output_file)

        assert (os.path.exists(gen_dir + 'test_vHeatmaps.png')), "Position Summary PNG plot was not saved"
        assert (os.path.exists(gen_dir + 'test_vHeatmaps.pdf')), "Position Summary PDF plot was not saved"
        shutil.rmtree(gen_dir)

    def test_plot_transition_graph_command(self):
        gen_dir = 'data/gen_plots/'
        index_file = 'data/test_index.yaml'
        model_path = 'data/test_model.p'
        config_file = 'data/config.yaml'
        max_syllable = 40
        group = ('Group1', 'default')
        output_file = gen_dir+'test_transitions'

        plot_transition_graph_command(index_file, model_path, config_file, max_syllable, group, output_file)
        assert (os.path.exists(output_file + '.png')), "Transition PNG graph was not saved"
        assert (os.path.exists(output_file + '.pdf')), "Transition PDF graph was not saved"
        shutil.rmtree(gen_dir)

    def test_copy_h5_metadata_to_yaml_command(self):
        print()