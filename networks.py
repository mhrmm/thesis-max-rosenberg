import torch
import torch.nn as nn
import torch.nn.functional as F


class PhraseEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, include_dropout = True):
        super(PhraseEncoder, self).__init__()
        self.include_dropout = include_dropout
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(vocab_size, hidden_size)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
 
    def forward(self, input_vec):
        output = self.linear1(input_vec).clamp(min=0)
        if self.include_dropout:
            output = self.dropout(output)
        output = self.linear2(output).clamp(min=0)
        if self.include_dropout:
            output = self.dropout(output)
        output = self.linear3(output).clamp(min=0)
        output = self.linear4(output)
        return output
        
class TiedClassifier(nn.Module): 

    def __init__(self, input_size, num_labels, hidden_size, phrase_encoder_constructor):
        super(TiedClassifier, self).__init__()
        self.input_size = input_size
        self.vocab_size = input_size // 5
        self.hidden_size = hidden_size
        self.word_encoder = phrase_encoder_constructor(self.vocab_size, hidden_size)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.linear3 = nn.Linear(5*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.final_layer = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        t = input_vec
        output1 = self.word_encoder(t[:,0*self.vocab_size:1*self.vocab_size])
        output2 = self.word_encoder(t[:,1*self.vocab_size:2*self.vocab_size])
        output3 = self.word_encoder(t[:,2*self.vocab_size:3*self.vocab_size])
        output4 = self.word_encoder(t[:,3*self.vocab_size:4*self.vocab_size])
        output5 = self.word_encoder(t[:,4*self.vocab_size:5*self.vocab_size])
        nextout = torch.cat([output1, output2, output3, output4, output5], dim=1) 
        nextout = self.linear3(nextout).clamp(min=0)
        nextout = self.dropout(nextout)
        nextout = self.linear4(nextout).clamp(min=0)
        nextout = self.dropout(nextout)
        nextout = self.linear5(nextout).clamp(min=0)
        nextout = self.dropout(nextout)
        nextout = self.final_layer(nextout)
        return F.log_softmax(nextout, dim=1)
    
def initialize_net(config_name, input_size, output_size):
    if config_name == 'tied-phrase':
        return TiedClassifier(input_size, output_size, 300,
                              lambda vocab_size, hidden_size: PhraseEncoder(vocab_size, hidden_size))
    elif config_name == 'tied-phrasenodropout':
        return TiedClassifier(input_size, output_size, 300,
                              lambda vocab_size, hidden_size: PhraseEncoder(vocab_size, hidden_size, False))
    else:
        raise Exception("Unrecognized network configuration: {}"
                        .format(config_name))
        
        
    
    
    
    