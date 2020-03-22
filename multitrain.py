import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from puzzle import make_puzzle_matrix, make_puzzle_targets, WordnetPuzzleGenerator
import time



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PuzzleDataset(Dataset):

    def __init__(self, puzzles, vocab):
        self.vocab = vocab
        self.evidence_matrix = make_puzzle_matrix(puzzles, vocab)
        self.response_vector = make_puzzle_targets([label for (_, label) in puzzles])
        self.num_choices = 5

    def input_size(self):
        return len(self.vocab) * self.num_choices

    def __getitem__(self, index):
        return self.evidence_matrix[index], self.response_vector[index]

    def __len__(self):
        return len(self.evidence_matrix)   

    @staticmethod
    def generate(generator, num_train):
        data = list(set(generator.batch_generate(num_train)))
        return PuzzleDataset(data, generator.get_vocab())
   
    @staticmethod
    def create_data_loader(dataset, batch_size):
        dataloader = DataLoader(dataset = dataset, 
                                     batch_size = batch_size, 
                                     shuffle=True)
        return dataloader

     


class ThreeLayerClassifier(nn.Module): 

    def __init__(self, input_size, num_labels, hidden_size):
        super(ThreeLayerClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        nextout = input_vec
        nextout = self.linear1(nextout).clamp(min=0)
        nextout = self.linear2(nextout).clamp(min=0)
        nextout = self.linear3(nextout)
        return F.log_softmax(nextout, dim=1)

def evaluate(model, loader):
    """Evaluates the trained network on test data."""
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, response in loader:
            input_matrix = data.to(device)
            log_probs = model(input_matrix)
            predictions = log_probs.argmax(dim=1)
            total += predictions.shape[0]
            for i in range(predictions.shape[0]):
                if response[i].item() == predictions[i].item():
                    correct += 1
    return correct / total

def train(puzzle_generator, num_epochs, hidden_size, 
          num_puzzles_to_generate, batch_size, multigpu = False):
    def maybe_regenerate(puzzle_generator, epoch, prev_loader, prev_test_loader):
        if epoch % 100 == 0:
            dataset = PuzzleDataset.generate(puzzle_generator, num_puzzles_to_generate)
            loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True)
            test_dataset = PuzzleDataset.generate(puzzle_generator, 100)
            test_loader = DataLoader(dataset = test_dataset, batch_size = 100, shuffle=False)
            return loader, test_loader
        else:
            return prev_loader, prev_test_loader
    
    def maybe_evaluate(model, epoch, prev_best, prev_best_acc):
        best_model = prev_best
        best_test_acc = prev_best_acc
        if epoch % 5 == 0:
            test_acc = evaluate(model, test_loader)
            print('epoch {} test: {:.2f}'.format(epoch, test_acc))
            if test_acc > prev_best_acc:
                best_test_acc = test_acc
                best_model = model
                #print('saving new model')
                #torch.save(best_model, 'best.model')
        return best_model, best_test_acc
    
    def maybe_report_time():
        if epoch % 100 == 0 and epoch > 0:
            finish_time = time.clock()
            time_per_epoch = (finish_time - start_time) / epoch
            print('Average time per epoch: {:.2} sec'.format(time_per_epoch))


    start_time = time.clock()
    input_size = 5 * len(puzzle_generator.get_vocab())
    output_size = 5
    model = ThreeLayerClassifier(input_size, output_size, hidden_size)
    if multigpu and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        #dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    
    model.to(device)
    loader = None
    test_loader = None
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())
    best_model = None
    best_test_acc = -1.0
    for epoch in range(num_epochs):
        model.train()
        model.zero_grad()
        loader, test_loader = maybe_regenerate(puzzle_generator, epoch, 
                                               loader, test_loader)
        for data, response in loader:
            input_matrix = data.to(device)
            log_probs = model(input_matrix)
            loss = loss_function(log_probs, response)
            loss.backward()
            optimizer.step()
        best_model, best_test_acc = maybe_evaluate(model, epoch,
                                                   best_model, best_test_acc)
        maybe_report_time()
    return best_model

train(WordnetPuzzleGenerator('animal.n.1'), 
      num_epochs=30000, 
      hidden_size=300,
      num_puzzles_to_generate=2000,
      batch_size=256,
      multigpu=False)

