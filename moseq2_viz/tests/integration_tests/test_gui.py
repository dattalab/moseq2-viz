import os
import shutil
import pandas as pd
import ruamel.yaml as yaml
from unittest import TestCase
from moseq2_viz.gui import get_groups_command, add_group_by_session, add_group_by_subject, make_crowd_movies_command,\
                plot_usages_command, plot_scalar_summary_command, plot_transition_graph_command, \
                plot_syllable_durations_command

class TestGUI(TestCase):

    def test_get_groups_command(self):
        index_path = 'data/test_index.yaml'

        with open(index_path, 'r') as f:
            index_data = yaml.safe_load(f)
        f.close()

        groups, uuids = [], []
        subjectNames, sessionNames = [], []
        for f in index_data['files']:
            if f['uuid'] not in uuids:
                uuids.append(f['uuid'])
                groups.append(f['group'])
                subjectNames.append(f['metadata']['SubjectName'])
                sessionNames.append(f['metadata']['SessionName'])

        num_groups = get_groups_command(index_path)

        assert num_groups == len(set(groups))

    def test_add_group_by_session(self):
        index_path = 'data/test_index.yaml'
        tmp_yaml = 'data/tmp_copy.yaml'

        with open(index_path, 'r') as f:
            index_data = yaml.safe_load(f)
        f.close()

        with open(tmp_yaml, 'w') as g:
            yaml.safe_dump(index_data, g)


        value = 'blackStockOFA80GritSanded_012517'
        group = 'test1'
        exact = False
        lowercase = False
        negative = False

        add_group_by_session(tmp_yaml, value, group, exact, lowercase, negative)

        assert not os.path.samefile(index_path, tmp_yaml)
        os.remove(tmp_yaml)


    def test_add_group_by_subject(self):
        index_path = 'data/test_index.yaml'
        tmp_yaml = 'data/tmp_copy.yaml'

        with open(index_path, 'r') as f:
            index_data = yaml.safe_load(f)
        f.close()

        with open(tmp_yaml, 'w') as g:
            yaml.safe_dump(index_data, g)

        value = '012517'
        group = 'test1'
        exact = False
        lowercase = False
        negative = False

        add_group_by_subject(tmp_yaml, value, group, exact, lowercase, negative)

        assert not os.path.samefile(index_path, tmp_yaml)
        os.remove(tmp_yaml)

    def test_make_crowd_movies_command(self):
        index_file = 'data/test_index_crowd.yaml'
        model_path = 'data/mock_model.p'
        crowd_dir = 'data/crowd_movies/'
        max_examples = 40
        max_syllable = 5

        out = make_crowd_movies_command(index_file, model_path, crowd_dir, max_syllable, max_examples)

        assert 'Success' in out
        assert (os.path.exists(crowd_dir))
        assert (len(os.listdir(crowd_dir)) == max_syllable + 1)
        shutil.rmtree(crowd_dir)


    def test_plot_usages_command(self):
        gen_dir = 'data/gen_plots/'
        index_file = 'data/test_index.yaml'
        model_path = 'data/test_model.p'
        sort = True
        count = 'usage'
        max_syllable = 40
        group = ('default', 'Group1')
        output_file = gen_dir+'test_usages'

        plot_usages_command(index_file, model_path, sort, count, max_syllable, group, output_file)

        assert (os.path.exists(gen_dir + 'test_usages.png'))
        assert (os.path.exists(gen_dir + 'test_usages.pdf'))
        os.remove(gen_dir + 'test_usages.png')
        os.remove(gen_dir + 'test_usages.pdf')
        os.removedirs(gen_dir)

    def test_plot_scalar_summary_command(self):
        index_file = 'data/test_index.yaml'
        gen_dir = 'data/gen_plots/'
        output_file = gen_dir + 'test_scalar'

        df = plot_scalar_summary_command(index_file, output_file)
        assert not df.empty
        assert (os.path.exists(gen_dir + 'test_scalar_position.png'))
        assert (os.path.exists(gen_dir + 'test_scalar_position.pdf'))
        assert (os.path.exists(gen_dir + 'test_scalar_summary.png'))
        assert (os.path.exists(gen_dir + 'test_scalar_summary.pdf'))
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
        assert (os.path.exists(output_file + '.png'))
        assert (os.path.exists(output_file + '.pdf'))
        shutil.rmtree(gen_dir)


    def test_plot_syllable_durations_command(self):
        gen_dir = 'data/gen_plots/'
        index_file = 'data/test_index.yaml'
        model_path = 'data/test_model.p'
        group = ('Group1', 'default')
        count = 'usage'
        max_syllable = 40
        output_file = gen_dir + 'test_durations'

        plot_syllable_durations_command(model_path, index_file, group, count, max_syllable, output_file)
        assert (os.path.exists(output_file + '.png'))
        assert (os.path.exists(output_file + '.pdf'))
        shutil.rmtree(gen_dir)

    def test_copy_h5_metadata_to_yaml_command(self):
        print()