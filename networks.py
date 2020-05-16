from torch import nn
import torch.nn.functional as F
from cuda import cudaify

class DropoutClassifier(nn.Module): 

    def __init__(self, vocab, num_labels, hidden_size, num_layers = 3):
        super(DropoutClassifier, self).__init__()
        error_msg = "Cannot initialize DropoutClassifier with num_layers={}".format(num_layers)
        assert num_layers >= 2, error_msg
        self.vocab = vocab
        self.num_labels = num_labels
        input_size = num_labels * len(vocab)
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.middle_layers = [cudaify(nn.Linear(hidden_size, hidden_size)) for
                              i in range(num_layers - 2)]
        self.linear3 = nn.Linear(hidden_size, num_labels)

    """
    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        for layer in self.middle_layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()            
        self.linear3.weight.data.uniform_(-initrange, initrange)
        self.linear3.bias.data.zero_()
    """
    
    def forward(self, input_vec):
        nextout = input_vec
        nextout = self.linear1(nextout).clamp(min=0)
        for layer in self.middle_layers:
            nextout = layer(nextout).clamp(min=0)
        nextout = self.linear3(nextout)
        return F.log_softmax(nextout, dim=1)

