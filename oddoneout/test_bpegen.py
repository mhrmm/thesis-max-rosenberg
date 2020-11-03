import unittest
import torch
from bpegen import BpePuzzleGenerator


class SimplePuzzleGenerator:

    def __init__(self):
        self.num_choices = 3

    def batch_generate(self, number_of_puzzles=10):
        return [(("eat", "ate", "ete", "tea", "tee"), 2)]

    def generate(self):
        return ("eat", "ate", "ete", "tea", "tee"), 2


class TestBpePuzzleGenerator(unittest.TestCase):

    def setUp(self):
        codes_path = "data/small.codes"
        vocab_path = "data/small.vocab"
        num_tokens = 5
        self.generator = BpePuzzleGenerator.from_paths(SimplePuzzleGenerator(),
                                                       codes_path,
                                                       vocab_path,
                                                       num_tokens)

    def test_new_puzzles(self):
        self.tok_puzzles = self.generator.batch_generate(1)
        assert len(self.tok_puzzles) == 1
        assert self.tok_puzzles[0] == ([['e@@', 'a@@', 't'], 
                                        ['a@@', 'te'], 
                                        ['e@@', 'te'], 
                                        ['te@@', 'a'], 
                                        ['te@@', 'e']], 2)

    def test_get_vocab(self):
        vocab = self.generator.get_vocab()
        assert vocab == {'a@@': 0, 'e@@': 1, 'te': 2, 'te@@': 3, 
                         'a': 4, 'e': 5, 't': 6}

    def test_make_matrix(self):
        tok_puzzles = self.generator.batch_generate(1)
        vec = self.generator.make_puzzle_matrix(tok_puzzles)
        assert vec.shape == torch.Size([1, 175])
        vec = vec.tolist()
        assert vec == [[0., 1., 0., 0., 0., 0., 0.,
                        1., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 1.,
                        0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0.,

                        1., 0., 0., 0., 0., 0., 0.,
                        0., 0., 1., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0.,

                        0., 1., 0., 0., 0., 0., 0.,
                        0., 0., 1., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0.,

                        0., 0., 0., 1., 0., 0., 0.,
                        0., 0., 0., 0., 1., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0.,

                        0., 0., 0., 1., 0., 0., 0.,
                        0., 0., 0., 0., 0., 1., 0.,
                        0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0.]]
    

if __name__ == "__main__":
    unittest.main()