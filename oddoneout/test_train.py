import unittest
import torch
from torch import tensor
from puzzleloader import PuzzleDataLoader
from puzzlegen import PuzzleGenerator
from train import predict, evaluate
from util import cudaify, compare_tensors


class SimpleClassifier:
    
    def to(self, x):
        pass

    def eval(self):
        pass
    
    def __call__(self, input_vec):
        result = [torch.zeros(3) for _ in range(input_vec.shape[0])]
        result = torch.stack(result)
        result[:, 0] = 1
        return result


class SimplePuzzleGenerator(PuzzleGenerator):
    
    def __init__(self):
        super().__init__()
        self.vocab = {'ate': 0, 'eat': 1, 'eta': 2, 'ete': 3}
        self._num_choices = 3
        self._puzzle1 = ("eat", "eta", "ete"), 2
        self._puzzle2 = ("ate", "ete", "eta"), 0

    def num_choices(self):
        return self._num_choices
    
    def get_vocab(self):
        return self.vocab

    def batch_generate(self, number_of_puzzles=10):
        return [self._puzzle1, self._puzzle2]

    def generate(self):
        return self._puzzle1
    

class TestTrain(unittest.TestCase):

    def test_predict(self):
        model = SimpleClassifier()
        puzzgen = SimplePuzzleGenerator()
        mgr = PuzzleDataLoader(puzzgen, 2, 2, 2, 2)
        _, loader = mgr.get_loaders(epoch=0)
        for (data, response) in loader:
            predictions = predict(model, data) 
            expected = tensor([0, 0])  # predicts choice 0 for all questions
            assert compare_tensors(predictions, expected)

    def test_evaluate(self):
        model = SimpleClassifier()
        puzzgen = SimplePuzzleGenerator()
        mgr = PuzzleDataLoader(puzzgen, 2, 2, 2, 2)
        _, loader = mgr.get_loaders(epoch=0)
        assert evaluate(model, loader) == 0.5  # gets the first one right, second wrong


if __name__ == "__main__":
    unittest.main()
