import unittest
from taxonomy import GraphTaxonomy, lowest_common_ancestor
from metrics import flatness, wu_palmer_similarity

class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.taxonomy = GraphTaxonomy(
            'entity',
            {'apple': ['fruit'],
             'lemon': ['citrus'],
             'orange': ['citrus', 'color'],
             'peach': ['fruit', 'color'],
             'red': ['color'],
             'yellow': ['color'],
             'citrus': ['fruit'],
             'fruit': ['entity'],
             'color': ['entity'],
             'entity': []}
        )

    def test_num_instances(self):
        assert self.taxonomy.num_instances() == 6

    def test_flatness(self):
        assert flatness(self.taxonomy, 'fruit') == 0.6
        assert flatness(self.taxonomy, 'entity') == 2 / 9

    def test_wu_palmer(self):
        # TODO: check to make sure these numbers are correct for the
        # example fruit taxonomy.
        print(wu_palmer_similarity(self.taxonomy, "orange", "lemon"))
        print(wu_palmer_similarity(self.taxonomy, "orange", "red"))

    def test_rosenberg_descendant_sim(self):
        # TODO: write tests
        pass





