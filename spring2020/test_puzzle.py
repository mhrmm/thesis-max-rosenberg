import unittest
from wordnet import find_lowest_common_ancestor, GetRandomSynset
from puzzle import WordnetPuzzleGenerator
from nltk.corpus import wordnet as wn


class TestWordnet(unittest.TestCase):
    
    def setUp(self):
        self.grs = GetRandomSynset('bird.n.1')
    
    def test_lowest_common_ancestor(self):
        ancestor = find_lowest_common_ancestor(['car','tank','bike'])
        assert ancestor == (120, wn.synset('self-propelled_vehicle.n.01'))
    
     
    def test_random_synset_with_specificity(self):
        assert self.grs.random_synset_with_specificity(120,140) == wn.synset('wading_bird.n.01')
    
    """
    def test_random_non_hyponym(self):
        print("random_non_hyponym test")
        print("Non-hyp of ", 'dove.n.1: ', self.grs.random_non_hyponym('dove.n.1'))
     
    def test_create_random_puzzle(self):
        print("create_random_puzzle test")
        puzzle = self.grs.create_random_puzzle()
        print("puzzle: ", puzzle)
        
    def test_generate_puzzles(self):
        print("generate_puzzles test")
        puzzles = self.grs.generate_puzzles(3)
        for i in range(len(puzzles)):
            print("puzzle ", i,": ", puzzles[i])
    """
    
class TestPuzzle(unittest.TestCase):
    
    def setUp(self):
        self.puzzle_generator = WordnetPuzzleGenerator('bird.n.1')
    
    """    
    def test_generate(self):
        print("generate test")
        puzzles = self.puzzle_generator.generate()
        for puzzle in puzzles:
            print(puzzle)
    """
       
if __name__ == "__main__":
	unittest.main()
