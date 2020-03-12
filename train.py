import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from puzzle import WordnetPuzzleGenerator
from networks import DropoutClassifier
from cuda import FloatTensor, LongTensor, cudaify


parameters = {
        'numChoices': 5,
        'batchSize': 1000,
        'numEpochs': 2000,
        'base': 10,
        'hiddenLayerSize': 300,
        'optimizer': 'adam',
        'trainingDataSize': 20000
        }

class TrainingParameters:
    
    def __init__(self, params):
        self._chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz'
        self._params = params
        
    def getNumChoices(self):
        return self._params['numChoices']

    def getBatchSize(self):
        return self._params['batchSize']
    
    def getNumEpochs(self):
        return self._params['numEpochs']

    def getBase(self):
        base = self._params['base']
        if base > 62:
            raise Exception('Base size greater than {} is not supported.'
                            .format(len(self._chars)))
        return self._chars[:base]
    
    def getHiddenLayerSize(self):
        return self._params['hiddenLayerSize']
    
    def getTrainingDataSize(self):
        return self._params['trainingDataSize']
    
    def getOptimizerFactory(self):
        optimizer = self._params['optimizer']
        if optimizer == 'adam':
            return optim.Adam
        elif optimizer == 'sgd':
            return lambda p: optim.SGD(p, lr=0.1)
        else:
            raise Exception('Optimizer "{}" is not supported.'.format(optimizer))

            




def oneHot(word, vocab):
    vec = [0]*len(vocab)
    vec[vocab[word]] = 1
    return vec

def makePuzzleVector(puzzle, vocab):
    choices, _ = puzzle
    oneHotVec = []
    for choice in choices:
        oneHotVec += oneHot(str(choice), vocab)
    return cudaify(FloatTensor(oneHotVec).view(1, -1))

def makePuzzleMatrix(puzzles, vocab):
    matrix = []
    for puzzle in puzzles:
        choices, _ = puzzle
        oneHotVec = []
        for choice in choices:
            oneHotVec += oneHot(str(choice), vocab)
        matrix.append(oneHotVec)
    return cudaify(FloatTensor(matrix))

def makePuzzleTarget(label):
    return LongTensor([label])    

def makePuzzleTargets(labels):
    return LongTensor(labels)    



    
class Trainer:
    
    def __init__(self, generator, params):
        self.generator = generator
        self.num_training_epochs = params.getNumEpochs()
        self.training_data_size = params.getTrainingDataSize()
        self.test_data_size = 100
        self.hidden_layer_size = params.getHiddenLayerSize()
        self.num_choices = params.getNumChoices()
        self.optimizerFactory = params.getOptimizerFactory()
        self.batch_size = params.getBatchSize()


    def generate_data(self):
        self.data = set(self.generator.batch_generate(self.training_data_size))
        self.test_data = set()
        while len(self.test_data) < self.test_data_size:
            puzzle = self.generator.generate()
            if puzzle not in self.data:
                self.test_data.add(puzzle)
        self.data = list(self.data)
        self.test_data = list(self.test_data)

            

    def run(self):
        model = DropoutClassifier(self.generator.get_vocab(),
                                  self.num_choices, 
                                  self.hidden_layer_size)
        cudaify(model)
        return self.batch_train(model)
        

    def batch_train(self, model):
        loss_function = nn.NLLLoss()
        batch_size = self.batch_size
        optimizer = self.optimizerFactory(model.parameters())
        for epoch in range(self.num_training_epochs):
            if epoch % 100 == 0:
                self.generate_data()
            model.train()
            model.zero_grad()
            batch = random.sample(self.data, batch_size)
            input_matrix = makePuzzleMatrix(batch, model.vocab)
            target = makePuzzleTargets([label for (_, label) in batch])
            log_probs = model(input_matrix)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print('epoch {}'.format(epoch))
                train_acc = self.evaluate(model, self.data[:200])
                test_acc = self.evaluate(model, self.test_data)
                print('train: {:.2f}; test: {:.2f}'.format(train_acc, test_acc))
        return model

    def evaluate(self, model, test_d):
        """Evaluates the trained network on test data."""
        model.eval()
        word_to_ix = model.vocab
        with torch.no_grad():
            correct = 0
            for instance, label in test_d:
                input_vec = makePuzzleVector((instance, label), word_to_ix)
                log_probs = model(input_vec)
                probs = [math.exp(log_prob) for log_prob in log_probs.tolist()[0]]
                ranked_probs = list(zip(probs, range(len(probs))))
                response = max(ranked_probs)[1]
                if response == label:
                    correct += 1            
        return correct/len(test_d)


def run(params, puzzle_gen):
    trainer = Trainer(puzzle_gen, params)
    model = trainer.run() 
    print('training accuracy = {}'.format(trainer.evaluate(model, trainer.data[:1000])))
    print('test accuracy = {}'.format(trainer.evaluate(model, trainer.test_data)))        
    return model
    
print('hi')
model = run(TrainingParameters(parameters), 
            WordnetPuzzleGenerator('vehicle.n.1'))


