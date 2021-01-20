import torch
from torch.utils.data import Dataset

from utils import get_corpus, get_type, encode_sentence, make_index_dictionary


class QuestionDataset(Dataset):
    def __init__(self, corpus, types, corpus_len):
        super(QuestionDataset, self).__init__()
        self.data = corpus
        self.label = types
        self.data_dim = corpus_len
        self.data_dictionary = make_index_dictionary()

    def __getitem__(self, item):
        x = encode_sentence(self.data[item], self.data_dictionary, self.data_dim)
        x = torch.tensor(x)
        y = self.label[item]
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.data)
