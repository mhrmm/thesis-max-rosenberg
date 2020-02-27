import random
from wordnet import GetRandomSynset

class PuzzleGenerator:
    def __init__(self):
        pass
    
    def getTrainingData(self, N):    
        return [self.generate() for n in range(N)]


class WordnetPuzzleGenerator(PuzzleGenerator):
    
    def __init__(self, root_synset):
        super(PuzzleGenerator, self).__init__()
        self.root_synset = root_synset
        


    def generate(self):
        generate_synset = GetRandomSynset(self.root_synset)
        (w1, w2, w3, w4, w5) = generate_synset.create_random_puzzle()
        puzzle = [(str(w1), 0), (str(w2), 0), (str(w3), 0), 
                  (str(w4), 0), (str(w5), 1)]
        random.shuffle(puzzle)
        xyz = tuple([i for (i,_) in puzzle])
        onehot = [j for (_,j) in puzzle]
        return (xyz, onehot.index(1))
    