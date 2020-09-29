class AnimalWord:
    def __init__(self, name):
        self.name = name
        self.hyponyms = []
        self.hypernyms = []

    # def __eq__(self, other):
    #     if self.name == other.name:
    #         return True
    #     else:
    #         return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_name(self):
        return self.name

class AnimalNet:
    def __init__(self):
        self.animal_list = []
        animal = AnimalWord(name="animal")

        bird = AnimalWord(name="bird")
        mammal = AnimalWord(name="mammal")
        reptile = AnimalWord(name="reptile")

        finch = AnimalWord(name="finch")
        swallow = AnimalWord(name="swallow")
        dog = AnimalWord(name="dog")
        cat = AnimalWord(name="cat")
        monkey = AnimalWord(name="monkey")
        giraffe = AnimalWord(name="giraffe")
        iguana = AnimalWord(name="iguana")

        bulldog = AnimalWord(name="bulldog")
        poodle = AnimalWord(name="poodle")

        animal.hyponyms.append(bird)
        animal.hyponyms.append(mammal)
        animal.hyponyms.append(reptile)



        mammal.hyponyms.append(dog)
        mammal.hyponyms.append(cat)
        mammal.hyponyms.append(monkey)
        mammal.hyponyms.append(giraffe)
        mammal.hypernyms.append(animal)
        
        bird.hyponyms.append(finch)
        bird.hyponyms.append(swallow)
        bird.hypernyms.append(animal)

        reptile.hyponyms.append(iguana)
        reptile.hypernyms.append(animal)



        cat.hypernyms.append(mammal)
        dog.hyponyms.append(bulldog)
        dog.hyponyms.append(poodle)
        dog.hypernyms.append(mammal)
        monkey.hypernyms.append(mammal)
        giraffe.hypernyms.append(mammal)
        swallow.hypernyms.append(bird)
        finch.hypernyms.append(bird)

        bulldog.hypernyms.append(dog)
        poodle.hypernyms.append(dog)
        
        self.animal_list.append(animal)
        self.animal_list.append(bird)
        self.animal_list.append(mammal)
        self.animal_list.append(reptile)
        self.animal_list.append(finch)
        self.animal_list.append(swallow)
        self.animal_list.append(dog)
        self.animal_list.append(cat)
        self.animal_list.append(monkey)
        self.animal_list.append(giraffe)
        self.animal_list.append(iguana)
        self.animal_list.append(bulldog)
        self.animal_list.append(poodle)
        
        word_to_ix = dict([(v.name, k) for (k,v) in enumerate(self.animal_list)])
        # print("vocab size: {}".format(len(word_to_ix)))
        self.vocab = word_to_ix

    def get_animal(self, animal_name):
        if animal_name in self.animal_list:
            return animal_name
        for a in self.animal_list:
            if animal_name == a.name:
                return a
        print("Error with word ", animal_name)
        raise Exception("Couldn't find an animal with that name.")