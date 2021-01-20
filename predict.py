from model import RNN
import torch
from utils import get_category, make_index_dictionary, encode_sentence


def predict(question):
    model = RNN(input_size=300, output_size=6, hidden_dim=64, n_layers=1)
    model.load_state_dict(torch.load('./data/rnn_model.pt'))

    output = model(question)
    pred = torch.argmax(output)
    return pred


if __name__ == '__main__':
    question = input("Fill the question: ")
    dictionary = make_index_dictionary()
    encode = encode_sentence(question, dictionary, 33)
    categories = get_category()
    print("Predict: ", categories[predict(encode)])
