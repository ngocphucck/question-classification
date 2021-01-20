import torch
from torch import nn

from utils import get_embedding_weight


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, json_path='./data/encode_dictionary.json'):
        super(RNN, self).__init__()
        self.embedding = self.make_embedding_layer(json_path)
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_dim)
        self.rnn = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_layers, nonlinearity='tanh')
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=output_size)
        self.classifier = nn.Softmax()

    def forward(self, X):
        out = self.embedding(X)
        out = self.linear1(out)
        out, hidden = self.rnn(out)
        out = out[:, -1, :]
        out = self.linear2(out)
        out = self.classifier(out)

        return out

    def make_embedding_layer(self, json_path='./data/encode_dictionary.json'):
        embedding_weights = get_embedding_weight(json_path)
        embedding_weights = torch.tensor(embedding_weights)

        return nn.Embedding.from_pretrained(embedding_weights)
