import unittest
from wordnet import find_lowest_common_ancestor, GetRandomSynset 
from puzzle import WordnetPuzzleGenerator
class TestWordnet(unittest.TestCase):
    
    def setUp(self):
        self.grs = GetRandomSynset('bird.n.1')
    
    def test_lowest_common_ancestor(self):
        print("lowest_common_ancestor test")
        ancestor = find_lowest_common_ancestor(['car','tank','bike'])
        print(ancestor)
        
    def test_random_synset_with_specificity(self):
        print("random_synset_with_specificity test")
        #if root spec < 5, loops infinitely
        print(self.grs.random_synset_with_specificity(5,1000))
    
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
    
class TestPuzzle(unittest.TestCase):
    
    def setUp(self):
        self.puzzle_generator = WordnetPuzzleGenerator('bird.n.1')
        
    def test_generate(self):
        print("generate test")
        puzzles = self.puzzle_generator.generate()
        for puzzle in puzzles:
            print(puzzle)
        
       
if __name__ == "__main__":
	unittest.main()
