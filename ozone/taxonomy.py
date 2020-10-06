import random
from  animals import AnimalNet
from ozone.wordnet import GetRandomSynset
from ozone.wordnet import get_all_lemmas_from_sense, hypernym_chain
from ozone.puzzle import PuzzleGenerator
from nltk.corpus import wordnet as wn

class Taxonomy:
    
    def get_vocab(self):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def get_root_synset(self):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def get_descendents(self, node):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def get_children(self, node):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def random_node(self, specificity_lb, specificity_ub):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def random_descendents(self, node, k):
        raise NotImplementedError("Cannot call this on an abstract class.")
        
    def random_non_descendent(self, node):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def hypernym_chain(self, node):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def flatness(self, node):
        num_total_hyps = len(self.get_descendents(node))
        if num_total_hyps == 0:
            return 0
        return len(self.get_children(node)) / num_total_hyps

    def repititions(self, node): 
        if node == self.get_root_synset():
            return self.get_descendents(self.get_root_synset()).count(node) + 1
        else:
            return self.get_descendents(self.get_root_synset()).count(node)

    def similarity(self, node1, node2):
        #get dicts of hypernym:distance from node to hypernym (AKA index of list)
        node1_hypernym_distances = dict()
        node1_hc = self.hypernym_chain(node1)
        for h in node1_hc:
            node1_hypernym_distances[h] = node1_hc.index(h)

        node2_hypernym_distances = dict()
        node2_hc = self.hypernym_chain(node2)
        for h in node2_hc:
            node2_hypernym_distances[h] = node2_hc.index(h)

        #find common hypernyms
        common = set(node1_hc) & set(node2_hc)

        #get sums of distances of common hypernyms, return word with minimum sum
        candidates = dict()
        for c in common:
            candidates[c] = node1_hypernym_distances[c] + node2_hypernym_distances[c]
            
        lowest_common_ancestor = min(candidates, key=candidates.get)

        node1_lca_distance = node1_hypernym_distances[lowest_common_ancestor]
        node2_lca_distance = node2_hypernym_distances[lowest_common_ancestor]
        node3_distance = len(self.hypernym_chain(lowest_common_ancestor)) - 1
        numerator = 2 * node3_distance
        denominator = node1_lca_distance + node2_lca_distance + (2 * node3_distance)
        return numerator / denominator


class BasicTaxonomy(Taxonomy):
    '''Basic Taxonomy of Animals'''
    
    def __init__(self):
        self.an = AnimalNet()
        self.root_synset = self.an.get_animal("animal")
        self.vocab = self.an.graph.vocab

    def get_root_synset(self):
        return self.root_synset.get_name()

    def get_vocab(self):
        return self.vocab

    def get_specificity(self, node):
        return len(self.get_descendents(node)) + 1

    def get_children(self, node):
        node = self.an.get_animal(node)
        return [child.name for child in self.an.graph.children(node)]

    def get_descendents(self, node):
        node = self.an.get_animal(node)
        return [d.name for d in self.an.graph.descendants(node)]

    def random_node(self,specificity_lb, specificity_ub):
        shuffled_animals = [a.name for a  in self.an.graph.vertices]
        random.shuffle(shuffled_animals)
        for animal in shuffled_animals:
            spec = self.get_specificity(animal)
            if spec < specificity_ub and spec > specificity_lb:
                return animal
        raise Exception("Couldn't find a node with specificity within the bounds")

    def random_descendents(self, node, k):
        hyps = [hypo for hypo in self.get_descendents(node)]
        return random.sample(hyps, k)

    def random_non_descendent(self, node):
        counter = 0
        while (counter < 1000):
            check_node = (self.random_node(0, 10))
            hyps = self.get_descendents(node)
            if check_node == node or check_node in hyps:
                counter += 1
            else:
                return check_node

    def repititions(self, node):
        node = self.an.get_animal(node)
        if node.name == self.get_root_synset():
            return self.get_descendents(self.get_root_synset()).count(node.get_name()) + 1

        return self.get_descendents(self.get_root_synset()).count(node.get_name())

    def ancestors(self, node):
        node_obj = self.an.get_animal(node)
        result = []
        for y in self.an.graph.ancestors(node_obj):
            result.append(y.get_name())
        return result

    def hypernym_chain(self, node):
        result = [node]
        while len(self.ancestors(node)) > 0:
            node = self.ancestors(node)[0]
            result.append(node)
        return result

    

    

class WordnetTaxonomy(Taxonomy):
    
    def __init__(self, root_synset_name):
        super().__init__()
        self.root_synset = wn.synset(root_synset_name)
        self.synset_gen = GetRandomSynset(root_synset_name)
        self.vocab = self._build_vocab()

    def get_root_synset(self):
        return self.root_synset.name()
        
    def get_vocab(self):
        return self.vocab

    def get_descendents(self, node):
        result = set()
        for y in node.hyponyms():
            result.add(y)
            for z in self.get_descendents(y):
                result.add(z)
        return result

    def get_children(self, synset_name):
        sense = wn.synset(synset_name)
        return sense.hyponyms()

    def _build_vocab(self):
        words = sorted(list(get_all_lemmas_from_sense(self.root_synset)))
        word_to_ix = dict([(v, k) for (k,v) in enumerate(words)])
        # print("vocab size: {}".format(len(word_to_ix)))
        return word_to_ix

    def random_node(self, specificity_lb, specificity_ub):
        root = self.synset_gen.random_synset_with_specificity(specificity_lb, 
                                                              specificity_ub)
        if root is None:
            return None
        while root == self.root_synset:
            root = self.synset_gen.random_synset_with_specificity(specificity_lb,
                                                                  specificity_ub)
        return root.name()
    
    def random_descendents(self, node, k):
        node = wn.synset(node)
        hyps = get_all_lemmas_from_sense(node) # children?
        return random.sample(hyps, k)
    
    def random_non_descendent(self, node):
        node = wn.synset(node)
        hyps = get_all_lemmas_from_sense(node) # children?
        counter = 0
        random_hyp_lemmas = []
        while len(random_hyp_lemmas) == 0:
            if counter > 10000:
                raise Exception('Too difficult to get a non-hyponym for {}; giving up'.format(node.name()))
            random_hyp = self.synset_gen.random_non_descendent(node.name())
            random_hyp_lemmas = set(get_all_lemmas_from_sense(random_hyp))
            random_hyp_lemmas -= hyps
            counter += 1
        random_word = random.choice(list(random_hyp_lemmas))
        return random_word

    def similarity(self, node1, node2):
        #get dicts of hypernym:distance from node to hypernym (AKA index of list)
        node1_hypernym_distances = dict()
        node1_hc = hypernym_chain(node1.name())
        for h in node1_hc:
            node1_hypernym_distances[h] = node1_hc.index(h)

        node2_hypernym_distances = dict()
        node2_hc = hypernym_chain(node2.name())
        for h in node2_hc:
            node2_hypernym_distances[h] = node2_hc.index(h)

        #find common hypernyms
        common = set(node1_hc) & set(node2_hc)

        #get sums of distances of common hypernyms, return word with minimum sum
        candidates = dict()
        for c in common:
            candidates[c] = node1_hypernym_distances[c] + node2_hypernym_distances[c]
            
        lowest_common_ancestor = min(candidates, key=candidates.get)

        node1_lca_distance = node1_hypernym_distances[lowest_common_ancestor]
        node2_lca_distance = node2_hypernym_distances[lowest_common_ancestor]
        node3_distance = len(hypernym_chain(lowest_common_ancestor.name())) - 1
        numerator = 2 * node3_distance
        denominator = node1_lca_distance + node2_lca_distance + (2 * node3_distance)
        return numerator / denominator

class TaxonomyPuzzleGenerator(PuzzleGenerator):
    
    def __init__(self, taxonomy, num_choices):
        super().__init__()
        self.taxonomy = taxonomy
        self.specificity_lb = 3
        self.specificity_ub = 10
        self.n_choices = num_choices
   
    def num_choices(self):
        return self.n_choices

    def max_tokens_per_choice(self):
        return 1
    
    def get_vocab(self):
        return self.taxonomy.get_vocab()
        
    def generate(self):
        root = self.taxonomy.random_node(self.specificity_lb, 
                                         self.specificity_ub)
        puzzle = self.taxonomy.random_descendents(root, self.num_choices() - 1)
        random_word = self.taxonomy.random_non_descendent(root)
        print('random word: ', random_word)
        puzzle.append(random_word)
        result = [(str(choice), 0) for choice in puzzle[:-1]] + [(str(puzzle[-1]), 1)]
        random.shuffle(result)
        xyz = tuple([i for (i, _) in result])
        onehot = [j for (_, j) in result]    
        return (xyz, onehot.index(1))
    
if __name__ == "__main__":
    # wnt = WordnetTaxonomy(root_synset_name="entity.n.01")
    # print(wnt.similarity(wn.synset("green.n.01"), wn.synset("ganja.n.01")))
    bt = BasicTaxonomy()