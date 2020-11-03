import fastBPE
import numpy as np
from ozone.util import cudaify, FloatTensor, LongTensor
from torch.utils.data import Dataset, DataLoader
import random


def one_hot(word, vocab):
    vec = [0]*len(vocab)
    vec[vocab[word]] = 1
    return vec


def make_puzzle_vector(puzzle, vocab):
    choices, _ = puzzle
    oneHotVec = []
    for choice in choices:
        oneHotVec += one_hot(str(choice), vocab)
    return cudaify(FloatTensor(oneHotVec).view(1, -1))


def make_puzzle_target(label):
    return cudaify(LongTensor([label]))


def make_puzzle_targets(labels):
    return cudaify(LongTensor(labels))


class PuzzleGenerator:

    def batch_generate(self, number_of_puzzles = 10):
        return [self.generate() for n in range(number_of_puzzles)]

    def generate(self):
        raise NotImplementedError('cannot call .generate() on abstract class.')

    def get_vocab(self):
        raise NotImplementedError('cannot call .get_vocab() on abstract class.')

    def num_choices(self):
        raise NotImplementedError('cannot call .num_choices() on abstract class.')

    def max_tokens_per_choice(self):
        raise NotImplementedError('cannot call on abstract class.')

    def make_puzzle_matrix(self, puzzles):
        matrix = []
        for puzzle in puzzles:
            choices, _ = puzzle
            one_hot_vec = []
            for choice in choices:
                one_hot_vec += one_hot(str(choice), self.get_vocab())
            matrix.append(one_hot_vec)
        return cudaify(FloatTensor(matrix))

    def tensorify(self, puzzles, num_choice):
        results = []
        for puzzle in puzzles:
            if len(puzzle) == int(num_choice):
                index = np.random.permutation(num_choice)
                results.append((tuple([puzzle[i] for i in index]),
                                index.tolist().index(0)))
        return results 


class TaxonomyPuzzleGenerator(PuzzleGenerator):

    def __init__(self, taxonomy, num_choices, lb, ub):
        super().__init__()
        self.taxonomy = taxonomy
        self.specificity_lb = lb
        self.specificity_ub = ub
        self.categories = [category for category in self.taxonomy.get_categories()
                           if (lb <= self.taxonomy.get_specificity(category) <= ub)]
        self.instances = sorted(self.taxonomy.get_instances())
        self.n_choices = num_choices
        words = sorted(set(taxonomy.get_descendant_instances(taxonomy.get_root())))
        self.vocab = {word: index for (index, word) in enumerate(words)}

    def num_choices(self):
        return self.n_choices

    def max_tokens_per_choice(self):
        return 1

    def get_vocab(self):
        return self.vocab

    def generate(self):
        root = random.choice(self.categories)
        descendants = self.taxonomy.get_descendant_instances(root)
        puzzle = random.sample(descendants, self.num_choices() - 1)
        random_word = random.choice(self.instances)
        while random_word in descendants:  # TODO: stronger check? e.g. make sure no spurious links?
            random_word = random.choice(self.instances)
        puzzle.append(random_word)
        result = [(str(choice), 0) for choice in puzzle[:-1]] + [(str(puzzle[-1]), 1)]
        random.shuffle(result)
        xyz = tuple([i for (i, _) in result])
        onehot = [j for (_, j) in result]
        return xyz, onehot.index(1)
