import unittest
import torch
from taxonomy import GraphTaxonomy
from ozone.taxonomy import WordnetTaxonomy
from puzzlegen import one_hot, make_puzzle_targets, TaxonomyPuzzleGenerator


class TestTaxonomyPuzzleGenerator(unittest.TestCase):

    def setUp(self):
        self.example = GraphTaxonomy(
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
        self.generator = TaxonomyPuzzleGenerator(self.example, 3, 2, 3)

    def test_vocab(self):
        vocab = self.generator.get_vocab()
        expected = {'apple': 0, 'lemon': 1, 'orange': 2,
                    'peach': 3, 'red': 4, 'yellow': 5}
        assert vocab == expected

    def test_batch_generate(self):
        puzzles = self.generator.batch_generate(number_of_puzzles=3)
        assert len(puzzles) == 3
        first_puzzle = puzzles[0]
        # puzzle should have the form:
        #    ('apple', "orange", 'lemon'), 0)
        ((choice1, choice2, choice3), oddman) = first_puzzle
        if oddman == 0:
            assert {choice2, choice3} == {'orange', 'lemon'}
            assert choice1 in {'apple', 'peach', 'red', 'yellow'}
        elif oddman == 1:
            assert {choice1, choice3} == {'orange', 'lemon'}
            assert choice2 in {'apple', 'peach', 'red', 'yellow'}
        else:
            assert oddman == 2
            assert {choice1, choice2} == {'orange', 'lemon'}
            assert choice3 in {'apple', 'peach', 'red', 'yellow'}

    def test_one_hot(self):
        vocab = self.generator.get_vocab()
        one_hot_vec = one_hot("orange", vocab)
        assert one_hot_vec == [0, 0, 1, 0, 0, 0]

    def test_make_puzzle_matrix(self):
        puzzles = [(('lemon', 'orange', 'red'), 2),
                   (('orange', 'lemon', 'peach'), 2),
                   (('lemon', 'yellow', 'orange'), 1)]
        matrix = self.generator.make_puzzle_matrix(puzzles)
        expected = torch.tensor([[0., 1., 0., 0., 0., 0.,
                                  0., 0., 1., 0., 0., 0.,
                                  0., 0., 0., 0., 1., 0.],
                                 [0., 0., 1., 0., 0., 0.,
                                  0., 1., 0., 0., 0., 0.,
                                  0., 0., 0., 1., 0., 0.],
                                 [0., 1., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 1.,
                                  0., 0., 1., 0., 0., 0.]])
        assert matrix.tolist() == expected.tolist()


if __name__ == "__main__":
    unittest.main()