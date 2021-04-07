import os
import unittest
from os.path import exists
from unittest import TestCase
from moseq2_viz.util import parse_index
from moseq2_viz.scalars.util import scalars_to_dataframe
from moseq2_viz.model.util import compute_behavioral_statistics
from moseq2_viz.model.embed import get_Xy_values, run_2d_embedding

class TestModelEmbed(TestCase):

    def setUp(self):

        index_path = 'data/test_index.yaml'
        model_path = 'data/test_model.p'

        _, self.sorted_index = parse_index(index_path)

        # compute session scalar data
        self.scalar_df = scalars_to_dataframe(self.sorted_index, model_path=model_path)

        # compute syllable usage and scalar statistics
        self.mean_df = compute_behavioral_statistics(self.scalar_df, count='usage',
                                                     groupby=['group', 'uuid'],
                                                     usage_normalization=True)

    def test_get_Xy_values(self):

        X, y, mapping, rev_mapping = get_Xy_values(self.mean_df,
                                                   self.mean_df.group.unique(),
                                                   stat='usage')

        assert list(mapping.keys()) == list(rev_mapping.values())
        assert list(mapping.values()) == list(rev_mapping.keys())

        assert len(X) == len(y)

    def test_run_2d_embedding(self):

        stat = 'usage'
        output_file = 'data/test_embedding.pdf'
        embedding = 'PCA'

        run_2d_embedding(self.mean_df, stat=stat, output_file=output_file, embedding=embedding)

        assert exists(output_file)
        os.remove(output_file)

if __name__ == '__main__':
    unittest.main()
