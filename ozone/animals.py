import queue
import numpy as np

class DirectedGraph:

    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        vertex_to_ix = dict([(v.name, k) for (k,v) in enumerate(self.vertices)])
        self.vocab = vertex_to_ix
        self.name_to_vertex = dict([(v.name, v) for v in (self.vertices)])

    def get_vocab(self):
        return self.vocab

    def lookup_vertex_by_name(self, name):
        return self.name_to_vertex[name]

    def descendants(self, vertex):
        res = []
        to_check = queue.Queue()
        for e in self.edges:
            if e[0].name == vertex.name:
                res.append(e[1])
                to_check.put(e[1])

        while not to_check.empty():
            v_to_check = to_check.get()
            for e in self.edges:
                if e[0].name == v_to_check.name:
                    res.append(e[1])
                    to_check.put(e[1])

        return res

    def children(self, vertex):
        res = []
        for e in self.edges:
            if e[0].name == vertex.name:
                res.append(e[1])
        return res

    def ancestors(self, vertex):
        if vertex.name == "animal":
            return []
        res = []
        curr = vertex
        while True:
            for e in self.edges:
                if e[1].name == curr.name:
                    res.append(e[0])
                    curr = e[0]
                    if e[0].name == "animal":
                        return res

    def parent(self, vertex):
        for e in self.edges:
            if e[1].name == vertex.name:
                return e[0]
    
    def non_descendants(self, vertex, root):
        all_animals = self.descendants(root)
        vertex_descendants = self.descendants(vertex)
        return np.setdiff1d(all_animals, vertex_descendants)

class AnimalVertex:
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name

    # def __repr__(self): # repr makes bugs less obvious
    #     return self.name

    def name(self):
        return self.name

class AnimalNet:

    def __init__(self):
        animal = AnimalVertex(name="animal")
        bird = AnimalVertex(name="bird")
        mammal = AnimalVertex(name="mammal")
        reptile = AnimalVertex(name="reptile")

        finch = AnimalVertex(name="finch")
        swallow = AnimalVertex(name="swallow")
        dog = AnimalVertex(name="dog")
        cat = AnimalVertex(name="cat")
        monkey = AnimalVertex(name="monkey")
        giraffe = AnimalVertex(name="giraffe")
        iguana = AnimalVertex(name="iguana")

        bulldog = AnimalVertex(name="bulldog")
        poodle = AnimalVertex(name="poodle")

        vertices = [animal, bird, mammal, reptile, finch, swallow, dog,
                    cat, monkey, giraffe, iguana, bulldog, poodle]

        edges = [(animal, bird), (animal, mammal), (animal, reptile),
                (mammal, dog), (mammal, cat), (mammal, monkey),
                (mammal, giraffe), (bird, finch), (bird, swallow),
                (reptile, iguana), (dog, bulldog), (dog, poodle)]

        self.graph = DirectedGraph(vertices, edges)

    def get_animal(self, animal_name):
        try:
            return self.graph.lookup_vertex_by_name(animal_name)
        except:
            raise Exception("Couldn't find an animal with the name '{}'.".format(animal_name)) 
        


if __name__ == "__main__":
    a = AnimalNet()
    v = AnimalVertex("dog")
    animal = AnimalVertex("animal")
    for b in (a.graph.non_descendants(v,animal)):
        print(b)