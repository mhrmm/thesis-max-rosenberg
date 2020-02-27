from torch import nn
import torch.nn.functional as F

class TwoLayerClassifier(nn.Module): 

    def __init__(self, vocab, num_labels, hidden_size):
        super(TwoLayerClassifier, self).__init__()
        self.vocab = vocab
        self.num_labels = num_labels
        input_size = num_labels * len(vocab)
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        nextout = self.linear1(input_vec).clamp(min=0)
        nextout = self.linear2(nextout)
        return F.log_softmax(nextout, dim=1)


class DropoutClassifier(nn.Module): 

    def __init__(self, vocab, num_labels, hidden_size):
        super(DropoutClassifier, self).__init__()
        self.vocab = vocab
        self.num_labels = num_labels
        input_size = num_labels * len(vocab)
        self.hidden_size = hidden_size
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        nextout = input_vec
        nextout = self.dropout1(nextout)
        nextout = self.linear1(nextout).clamp(min=0)
        nextout = self.dropout2(nextout)    
        nextout = self.linear2(nextout)
        nextout = self.dropout3(nextout)    
        nextout = self.linear3(nextout)
        return F.log_softmax(nextout, dim=1)

