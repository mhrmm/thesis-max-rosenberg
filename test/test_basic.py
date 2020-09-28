import unittest
from ozone.taxonomy import BasicTaxonomy

class TestBasicTaxonomy(unittest.TestCase):
    
    def setUp(self):
        self.taxonomy = BasicTaxonomy()   

    def test_get_vocab(self):
        expected = {'animal': 0, 'bird': 1, 'mammal': 2, 'finch': 3, 'hummingbird': 4, 'dog': 5, 'cat': 6}
        return self.taxonomy.get_vocab == expected

    def test_get_root_synset(self):
        expected = "animal"
        assert self.taxonomy.get_root_synset() == expected

    def test_random_node(self):
        assert self.taxonomy.random_node(5,9) == 'animal'
        ani = self.taxonomy.random_node(0,2)
        assert ani == 'dog' or ani == 'cat' or ani == 'finch' or ani == 'hummingbird'

    def test_random_hyponyms(self):
        expected = set(['bird',
                        "mammal",
                        'finch',
                        'hummingbird',
                        'dog',
                        'cat'])
        result = set(self.taxonomy.random_hyponyms('animal', 6))
        assert result == expected

        expected1 = set(['finch'])
        expected2 = set(['hummingbird'])
        result = set(self.taxonomy.random_hyponyms('bird', 1))
        assert result == expected1 or result == expected2

    def test_random_non_hyponyms(self):
        non_birds = {'dog', 'cat', 'animal', 'mammal'}
        for _ in range(10):
            non_hyponym = self.taxonomy.random_non_hyponym('bird')
            assert non_hyponym in non_birds

        
if __name__ == "__main__":
	unittest.main()
