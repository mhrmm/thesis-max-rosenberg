import fastBPE
import numpy as np
from ozone.util import cudaify, FloatTensor, LongTensor
from torch.utils.data import Dataset, DataLoader
import random


def make_puzzle_targets(labels):
    return cudaify(LongTensor(labels))


class PuzzleDataset(Dataset):

    def __init__(self, puzzle_generator, num_train):
        self.num_choices = puzzle_generator.num_choices()
        puzzles = puzzle_generator.batch_generate(num_train)
        self.puzzle_generator = puzzle_generator
        self.response_vector = make_puzzle_targets([label for (_, label) in puzzles])
        print('making puzzle matrix')
        self.evidence_matrix = self.puzzle_generator.make_puzzle_matrix(puzzles)
        self.vocab = puzzle_generator.get_vocab()
        
    def input_size(self):
        input_size = (len(self.vocab) * 
                      self.num_choices * 
                      self.puzzle_generator.max_tokens_per_choice())
        return input_size

    def output_size(self):
        return self.puzzle_generator.num_choices()

    def __getitem__(self, index):
        return self.evidence_matrix[index], self.response_vector[index]

    def __len__(self):
        return len(self.evidence_matrix)   

    @staticmethod
    def compile_puzzle(generator, puzzle):
        return generator.make_puzzle_matrix([(puzzle, -1)])         


class PuzzleDataLoader:
    
    def __init__(self, puzzle_generator, num_train, train_batch_size, 
                 num_test = 1000, test_batch_size=100):
        self.puzzle_generator = puzzle_generator
        self.num_train = num_train
        self.train_batch_size = train_batch_size
        self.num_test = num_test
        self.test_batch_size = test_batch_size
        self._regenerate()

    def _regenerate(self):
        self.train_data = PuzzleDataset(self.puzzle_generator, self.num_train)
        self.train_loader = DataLoader(dataset = self.train_data,
                                       batch_size = self.train_batch_size, 
                                       shuffle=True)
        self.test_data = PuzzleDataset(self.puzzle_generator, self.num_test)
        self.test_loader = DataLoader(dataset = self.test_data, 
                                      batch_size = self.test_batch_size, 
                                      shuffle=False)

    def get_loaders(self, epoch):
        if epoch % 100 == 0:
            self._regenerate()
        return self.train_loader, self.test_loader     

    def input_size(self):
        return self.train_data.input_size()

    def output_size(self):
        return self.train_data.output_size()
