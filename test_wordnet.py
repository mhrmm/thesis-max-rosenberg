import unittest
from wordnet import find_lowest_common_ancestor

class TestWordnet(unittest.TestCase):
    
    def test_lowest_common_ancestor(self):
        print(find_lowest_common_ancestor(['republican', 'democratic']))
       
if __name__ == "__main__":
	unittest.main()
