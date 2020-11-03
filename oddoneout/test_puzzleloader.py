import unittest
import torch
from torch import tensor
from puzzleloader import PuzzleDataset, PuzzleDataLoader
from puzzlegen import PuzzleGenerator, TaxonomyPuzzleGenerator
from util import cudaify, compare_tensors


class SimpleClassifier:
    
    def __init__(self):
        pass
    
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

    def test_puzzle_dataset(self):
        puzzgen = SimplePuzzleGenerator()
        dataset = PuzzleDataset(puzzgen, 2)
        assert len(dataset) == 2
        datum0 = dataset[0]
        expected = (tensor([0., 1., 0., 0.,    # eat = 1
                            0., 0., 1., 0.,    # eta = 2
                            0., 0., 0., 1.]),  # ete = 3
                    tensor(2))                 # odd-man-out = "ete"
        assert compare_tensors(datum0[0], expected[0])
        assert compare_tensors(datum0[1], expected[1])
        datum1 = dataset[1]
        expected = (tensor([1., 0., 0., 0.,    # ate = 0
                            0., 0., 0., 1.,    # ete = 3
                            0., 0., 1., 0.]),  # eta = 2
                    tensor(0))                 # odd-man-out = "ate"
        assert compare_tensors(datum1[0], expected[0])
        assert compare_tensors(datum1[1], expected[1])
        
    def test_puzzle_loader(self):
        puzzgen = SimplePuzzleGenerator()
        mgr = PuzzleDataLoader(puzzgen, 2, 1, 2, 1)
        _, loader = mgr.get_loaders(epoch=0)
        for i, (data, response) in enumerate(loader):
            if i == 0:
                expected = tensor([[0., 1., 0., 0., 
                                    0., 0., 1., 0., 
                                    0., 0., 0., 1.]])
                assert compare_tensors(data, expected)
                expected = tensor([2])
                assert compare_tensors(response, expected)
            elif i == 1:
                expected = tensor([[1., 0., 0., 0., 
                                    0., 0., 0., 1., 
                                    0., 0., 1., 0.]])
                assert compare_tensors(data, expected)
                expected = tensor([0])
                assert compare_tensors(response, expected)
            else:
                assert False, "Too many results returned by loader."
            
    def test_puzzle_loader2(self):
        puzzgen = SimplePuzzleGenerator()
        mgr = PuzzleDataLoader(puzzgen, 2, 2, 2, 2)
        _, loader = mgr.get_loaders(epoch=0)
        for i, (data, response) in enumerate(loader):
            if i == 0:
                assert data.shape == torch.Size([2, 12])
                expected = tensor([[0., 1., 0., 0., 
                                    0., 0., 1., 0., 
                                    0., 0., 0., 1.],
                                   [1., 0., 0., 0., 
                                    0., 0., 0., 1., 
                                    0., 0., 1., 0.]])
                compare_tensors(data, expected)
                assert response.shape == torch.Size([2])
                expected = tensor([2, 0])                
                assert compare_tensors(response, expected)
            else:
                assert False, "Too many results returned by loader."


if __name__ == "__main__":
    unittest.main()
