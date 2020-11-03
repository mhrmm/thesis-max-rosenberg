import unittest
from wordnet import WordnetTaxonomy
from taxonomy import lowest_common_ancestor
from solver import solve_puzzle, TaxonomySimilarity, solve_puzzles, silent_logger
from puzzle import OddOneOutPuzzle, common2_puzzles


class TestWordnet(unittest.TestCase):

    def setUp(self):
        self.taxonomy = WordnetTaxonomy()

    def test_is_instance(self):
        assert self.taxonomy.is_instance('apple')
        assert self.taxonomy.is_instance('dog')
        assert not self.taxonomy.is_instance('dog.n.01')
        assert not self.taxonomy.is_instance('dogfdsa')

    def test_is_category(self):
        assert not self.taxonomy.is_category('apple')
        assert not self.taxonomy.is_category('dog')
        assert self.taxonomy.is_category('dog.n.01')
        assert not self.taxonomy.is_category('doggo.n.01')

    def test_ancestor_categories1(self):
        result = self.taxonomy.get_ancestor_categories('animal.n.01')
        expected = ['animal.n.01', 'entity.n.01', 'living_thing.n.01',
                    'object.n.01', 'organism.n.01', 'physical_entity.n.01',
                    'whole.n.02']
        assert sorted(result) == expected

    def test_ancestor_categories2(self):
        result = self.taxonomy.get_ancestor_categories('orange')
        expected = ['abstraction.n.06', 'angiospermous_tree.n.01',
                    'attribute.n.02', 'chromatic_color.n.01',
                    'citrus.n.01', 'citrus.n.02', 'color.n.01',
                    'coloring_material.n.01', 'edible_fruit.n.01',
                    'entity.n.01', 'fruit.n.01', 'fruit_tree.n.01',
                    'living_thing.n.01', 'material.n.01', 'matter.n.03',
                    'natural_object.n.01', 'object.n.01', 'orange.n.01',
                    'orange.n.02', 'orange.n.03', 'orange.n.04',
                    'organism.n.01', 'physical_entity.n.01', 'pigment.n.01',
                    'plant.n.02', 'plant_organ.n.01', 'plant_part.n.01',
                    'property.n.02', 'reproductive_structure.n.01',
                    'substance.n.01', 'tree.n.01', 'vascular_plant.n.01',
                    'visual_property.n.01', 'whole.n.02', 'woody_plant.n.01']
        assert sorted(result) == expected

    def test_descendant_instances1(self):
        result = self.taxonomy.get_descendant_instances('apple.n.01')
        expected = ['Baldwin', "Bramley's Seedling", 'Cortland',
                    "Cox's Orange Pippin", 'Delicious', 'Empire',
                    'Golden Delicious', 'Granny Smith', "Grimes' golden",
                    'Jonathan', "Lane's Prince Albert", 'Macoun',
                    'McIntosh', 'Newtown Wonder', 'Northern Spy',
                    'Pearmain', 'Pippin', 'Prima', 'Red Delicious',
                    'Rome Beauty', 'Stayman', 'Stayman Winesap', 'Winesap',
                    'Yellow Delicious', 'cooking apple', 'crab apple',
                    'crabapple', 'dessert apple', 'eating apple']
        expected = sorted([x.lower() for x in expected])
        assert sorted(result) == expected

    def test_descendant_instances2(self):
        result = self.taxonomy.get_descendant_instances('poodle.n.01')
        expected = ['large poodle', 'miniature poodle',
                    'standard poodle', 'toy poodle']
        assert sorted(result) == expected

    def test_lowest_common_ancestor1(self):
        result = lowest_common_ancestor(self.taxonomy,
                                        ['orange', 'red', 'green', 'blue'],
                                        'apple')
        assert result == (168, 'chromatic_color.n.01')

    def test_lowest_common_ancestor2(self):
        result = lowest_common_ancestor(self.taxonomy,
                                        ['orange', 'red', 'green', 'blue'],
                                        'gold')
        assert result == (104363, 'entity.n.01')

    def test_lowest_common_ancestor3(self):
        result = lowest_common_ancestor(self.taxonomy,
                                        ['large poodle', 'miniature poodle',
                                         'standard poodle', 'toy poodle'],
                                        'beagle')
        assert result == (4, 'poodle.n.01')

    def test_solve_puzzle(self):
        similarity = TaxonomySimilarity(self.taxonomy)
        puzzle = OddOneOutPuzzle("beagle",
                                 ['large poodle', 'miniature poodle',
                                  'standard poodle', 'toy poodle'],
                                 "poodle")
        result = solve_puzzle(puzzle, similarity)
        assert result == (0.25, 'poodle.n.01', 'beagle')

    def test_solve_puzzles(self):
        similarity = TaxonomySimilarity(self.taxonomy)
        result = solve_puzzles(common2_puzzles, similarity, logger=silent_logger)
        assert result == (35, 13, 54)


class TestWordnetSubset(unittest.TestCase):
    def setUp(self):
        self.taxonomy = WordnetTaxonomy('poodle.n.01')

    def test_get_categories(self):
        assert self.taxonomy.get_categories() == {'poodle.n.01',
                                                  'standard_poodle.n.01',
                                                  'large_poodle.n.01',
                                                  'miniature_poodle.n.01',
                                                  'toy_poodle.n.01'}

    def test_is_instance(self):
        assert not self.taxonomy.is_instance('apple')
        assert self.taxonomy.is_instance('toy poodle')
        assert not self.taxonomy.is_instance('dog.n.01')
        assert not self.taxonomy.is_instance('dogfdsa')

    def test_is_category(self):
        assert not self.taxonomy.is_category('apple')
        assert not self.taxonomy.is_category('dog')
        assert self.taxonomy.is_category('standard_poodle.n.01')
        assert not self.taxonomy.is_category('apple.n.01')
        assert not self.taxonomy.is_category('doggo.n.01')


class TestWordnetSubset2(unittest.TestCase):
    def setUp(self):
        self.taxonomy = WordnetTaxonomy('dog.n.01')

    def test_ancestor_categories(self):
        result = self.taxonomy.get_ancestor_categories('toy_poodle.n.01')
        expected = ['dog.n.01', 'poodle.n.01', 'toy_poodle.n.01']
        assert sorted(result) == expected

    def test_descendant_instances(self):
        result = self.taxonomy.get_descendant_instances('poodle.n.01')
        expected = ['large poodle', 'miniature poodle',
                    'standard poodle', 'toy poodle']
        assert sorted(result) == expected

    def test_lowest_common_ancestor(self):
        result = lowest_common_ancestor(self.taxonomy,
                                        ['large poodle', 'miniature poodle',
                                         'standard poodle', 'toy poodle'],
                                        'beagle')
        assert result == (4, 'poodle.n.01')

    def test_solve_puzzle(self):
        similarity = TaxonomySimilarity(self.taxonomy)
        puzzle = OddOneOutPuzzle("beagle",
                                 ['large poodle', 'miniature poodle',
                                  'standard poodle', 'toy poodle'],
                                 "poodle")
        result = solve_puzzle(puzzle, similarity)
        assert result == (0.25, 'poodle.n.01', 'beagle')
