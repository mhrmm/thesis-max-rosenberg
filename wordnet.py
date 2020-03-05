
import nltk
import random
nltk.download('wordnet')

from nltk.corpus import wordnet as wn

def synsets_of_mercury():
    synsets = wn.synsets('mercury')
    for synset in synsets:
        print(synset)
        

def hypernym_chain(synset_name):
    """
    e.g. hypernym_chain('boat.n.01')
    
    """
    synset = wn.synset(synset_name)
    result = [synset]
    while len(synset.hypernyms()) > 0:
        synset = synset.hypernyms()[0]
        result.append(synset)
    return result

def get_all_hypernyms_from_sense(sense):
    result = set()
    for y in sense.hypernyms():
        result.add(y)
        for z in get_all_hypernyms_from_sense(y):
            result.add(z)
    return result

def get_all_hypernyms(word):
    """
    e.g. get_all_hypernyms('dog')
    
    """
    result = set()
    for x in wn.synsets(word):
        for y in get_all_hypernyms_from_sense(x):
            result.add(y)
    return result


def get_all_hyponyms_from_sense(sense):
    """
    e.g. get_all_hyponyms_from_sense(wn.synset('metallic_element.n.01'))
    
    """
    result = set()
    for y in sense.hyponyms():
        result.add(y)
        for z in get_all_hyponyms_from_sense(y):
            result.add(z)
    return result

def get_all_lemmas_from_sense(sense):
    result = set()
    for y in get_all_hyponyms_from_sense(sense):
        for lemma in y.lemmas():
            result.add(lemma.name())
    return result



class Specificity:
    def __init__(self):
        self.cache = dict()
        
    def evaluate(self, sense):
        if sense.name() not in self.cache:
            spec = len(get_all_hyponyms_from_sense(sense))
            self.cache[sense.name()] = spec
        return self.cache[sense.name()]


specificity = Specificity()


def find_lowest_common_ancestor(words):
    """
    find_lowest_common_ancestor(['libertarian', 'green', 'garden', 'democratic'])
    find_lowest_common_ancestor(['apple', 'banana', 'orange', 'grape'])
    
    """
    commonHypernyms = get_all_hypernyms(words[0])
    for word in words[1:]:
        commonHypernyms = commonHypernyms & get_all_hypernyms(word)
    if len(commonHypernyms) == 0:
        hyp = wn.synset('entity.n.01')
        return (specificity.evaluate(hyp), hyp)
    scoredHypernyms = [(specificity.evaluate(hyp), hyp) for hyp in commonHypernyms]
    sortedHypernyms = sorted(scoredHypernyms)
    return sortedHypernyms[0]

class GetRandomSynset:
    def __init__(self, root_synset = 'dog.n.1'):
        entity = wn.synset(root_synset)
        self.entity_hyps = get_all_hyponyms_from_sense(entity)

    def __call__(self):
        random_word = random.sample(self.entity_hyps,1)[0]
        return random_word
    
    @staticmethod    
    def factory(root_synset):
        return GetRandomSynset(root_synset)
    

    def random_synset_with_specificity(self, lower, upper):
        root = self()
        root_spec = specificity.evaluate(root)
        while ((root_spec < lower) or (root_spec > upper)):
            root = self()
            root_spec = specificity.evaluate(root)
        return root

    def random_non_hyponym(self, synset):
        while True:
            result = self()
            if synset not in get_all_hypernyms_from_sense(result):
                return result

    #Creates puzzle with random rootword between spec val of 5 and 1000
    def create_random_puzzle(self):
        root = self.random_synset_with_specificity(5, 1000)
        hyps = get_all_hyponyms_from_sense(root) # children?
        puzzle = random.sample(hyps, 4)
        random_word = self.random_non_hyponym(root)
        puzzle.append(random_word)
        return [random.choice(s.lemmas()).name() for s in puzzle]

    def generate_puzzles(self, number_of_puzzles = 10):
        puzzles = []
        i = 0
        while i < number_of_puzzles:
            new_puzzle = self.create_random_puzzle()
            puzzles.append(new_puzzle)
            i += 1
            print(new_puzzle)
        #Write to file
#        with open(output_file, "w") as f:
#            for puzzle in puzzles:
#                for word in puzzle:
#                    f.write(word.name())
#                    f.write(", ")
#                f.write("\n")
#                f.write("\n")
        return puzzles



def show_puzzles(puzzles):
    score = 0
    num_puzzles_seen = 0
    lives = 100
    #"puzzle" variableis unshuffled, answer is always at puzzle[4] 
    for puzzle in puzzles:
        num_puzzles_seen += 1 
        if lives == 0:
            print(score, num_puzzles_seen)
            print("GAME OVER! you got ",100 * score / num_puzzles_seen , "% correct")
            return 0
        shuffled_puzzle = puzzle[:]
        random.shuffle(shuffled_puzzle)
        print("\n \nPUZZLE: ",)
        for word in shuffled_puzzle:
            print(word.name()[:-5])
        print("\n you have " + str(lives) + " left.")
        
        #HUMAN PLAYER
        guess = input("Which word is the odd man out? ")
        
        #COMPUTER PLAYER 
        #guess = random.choice(puzzle).name()
        
        print("\n YOUR ANSWER: ", guess)
        print("\n CORRECT ANSWER: ", puzzle[4].name()[:-5])
        if (guess == puzzle[4].name()[:-5]):
           print("\n GOOD WORK!")
           score += 1
        else:
           print("\n INCORRECT")
           lives -= 1



if __name__ == "__main__":
    generate_synset = GetRandomSynset('dog.n.1')
    test_puzzles = generate_synset.generate_puzzles()   
#    show_puzzles(test_puzzles)
    print(len(test_puzzles))
    
    