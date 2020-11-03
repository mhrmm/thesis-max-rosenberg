import fastBPE
import numpy as np
from ozone.util import cudaify, FloatTensor, LongTensor
from torch.utils.data import Dataset, DataLoader
from puzzlegen import PuzzleGenerator, one_hot
import random


class BpePuzzleGenerator(PuzzleGenerator):

    def __init__(self, base_puzzle_gen, vocab, bpe, num_tok, debugging=False):
        super(BpePuzzleGenerator, self).__init__()
        self.vocab = vocab
        self.bpe = bpe
        self.base_puzzle_gen = base_puzzle_gen
        self.num_tok = num_tok
        self.debugging = debugging
      
    def num_choices(self):
        return self.base_puzzle_gen.num_choices()
      
    @staticmethod
    def _read_vocab(vocab_file_path):
        with open(vocab_file_path) as reader:
            vocab = [(line.split()[0], i) for (i, line) in enumerate(reader)]
            tok_to_ix = dict(vocab)
        return tok_to_ix

    def max_tokens_per_choice(self):
        return 5
     
    def get_vocab(self):
        return self.vocab

    def generate(self):
        """
        e.g
        result = [([['app', 'le'], ['pea', 'r']] , 0), 
                  ([['do', 'g'], ['ca', 't']], 1), 
                  ([['low', 'er'], ['high', 'er']] 0)]

        """
        puzzle = self.base_puzzle_gen.generate()
        tok_puzzle = self.bpe.apply(list(puzzle[0]))
        new_puzzle = ([word.split(" ") for word in tok_puzzle], puzzle[1])
        return new_puzzle

    def reset_root(self, root_synset):
        self.base_puzzle_gen.reset_root(root_synset)

    def make_puzzle_matrix(self, tok_puzzles):
        """
        concatenate first 4 tokens if exist, then merge the rest tokens 
        and append it to the end
        
        TODO: Is it possible to get rid of the topmost for-loop using torch tensor ops??
        
        """
        matrix = []
        for tok_puzzle in tok_puzzles:
            choices, _ = tok_puzzle
            one_hot_vec = []
            for choice in choices:
                choice_vec_list = [one_hot(tok, self.vocab) for tok in choice]
                if len(choice_vec_list) > (self.num_tok - 1):
                    summed = [sum(vec) for vec in zip(*choice_vec_list[self.num_tok-1:])]
                    choice_vec_list[self.num_tok - 1] = summed
                    choice_vec_list = choice_vec_list[:self.num_tok]
                result = [tok for word in choice_vec_list for tok in word]
                appendix = [0] * (self.num_tok * len(self.vocab) - len(result))
                one_hot_vec += result + appendix
            matrix.append(one_hot_vec)
        result = cudaify(FloatTensor(matrix))
        return result 
    
    def tensorify(self, puzzles, num_choice):
        results = []
        for puzzle in puzzles:
            assert len(puzzle) == int(num_choice)
            index = np.array(list(range(num_choice)))
            if not self.debugging:
                index = np.random.permutation(num_choice)
            randomized = [puzzle[i] for i in index]
            tok_puzzle = self.bpe.apply(randomized)
            results.append(([word.split(" ") for word in tok_puzzle],
                            index.tolist().index(0)))
        return results 

    @staticmethod
    def from_paths(base_puzzle_gen, train_file_path, vocab_file_path, num_tok):
        vocab = BpePuzzleGenerator._read_vocab(vocab_file_path)
        bpe = fastBPE.fastBPE(train_file_path, vocab_file_path)
        return BpePuzzleGenerator(base_puzzle_gen, vocab, bpe, num_tok)
