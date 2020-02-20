


ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def createFilter(letter_index, letter_range):
    return (lambda wd: letter_index < len(wd) and 
            wd[letter_index] in letter_range)
    
class FilterTree:
    def __init__(self, root_filter, children):
        self.root_filter = root_filter
        self.children = children
        
    def run(self, wd):
        result = ['']
        if self.root_filter(wd):
            for i, child in enumerate(self.children):
                child_results = [str(i)+childStr for childStr in child.run(wd)]
                result += child_results
        return result
  
import random      

def createFilterTree(depth=3):
    first_letter = random.randint(0, len(ALPHABET) - 5)
    root_filter = createFilter(3-depth-1, ALPHABET[first_letter:first_letter+10])
    if depth == 0:
        return FilterTree(root_filter, [])
    else:
        child1 = createFilterTree(depth-1)
        child2 = createFilterTree(depth-1)
        return FilterTree(root_filter, [child1, child2])
    
    
def createWordMap():
    from collections import defaultdict
    ftree = createFilterTree()
    words = defaultdict(list)
    for x in ALPHABET:
        for y in ALPHABET:
            for z in ALPHABET:
                wd = x+y+z
                result = [category for category in ftree.run(wd) if len(category) == 3]
                for category in result:
                    words[category].append(wd)
                if len(result) > 0:
                    print(' {}: {}'.format(wd, result))
    for key in words:
        print(key)
    return words