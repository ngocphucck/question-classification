import pandas as pd
import re
import json
from gensim.models import KeyedVectors


def get_type(file_path='./data/Question_Classification_Dataset.csv'):
    df = pd.read_csv(file_path)
    categories = list({type for type in df['Category0'].to_list()})
    categories = sorted(categories)
    type_map = {category: index for index, category in enumerate(categories)}
    types = [type_map[type] for type in df['Category0'].to_list()]
    return types


def get_category(file_path='./data/Question_Classification_Dataset.csv'):
    df = pd.read_csv(file_path)
    categories = list({type for type in df['Category0'].to_list()})
    categories = sorted(categories)
    type_map = dict(enumerate(categories))
    return type_map


def rm_punctual(doc):
    doc = re.sub(r'[^\w\s]', ' ', doc)
    doc = re.sub(r'[0-9]{1,4}', ' 1 ', doc)
    doc = doc.lower()
    return doc


def get_corpus(file_path='./data/Question_Classification_Dataset.csv'):
    df = pd.read_csv(file_path)
    corpus = [rm_punctual(doc) for doc in df['Questions'].to_list()]
    return corpus


def get_vocab(corpus):
    vocab = {word for word in ' '.join(corpus).split()}
    return vocab


def get_pretrained_encode(vocab, model_path='./data/GoogleNews-vectors-negative300.bin',
                    save_path='./data/encode_dictionary.json'):
    words = []
    embeddings = []
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    for word in vocab:
        # noinspection PyBroadException
        try:
            word_encode = model[word].tolist()
            embeddings.append(word_encode)
            words.append(word)
        except:
            print("Can't find word: ", word)
    with open(save_path, 'w') as f:
        json.dump((words, embeddings), f)
    print("Save complete!")


def make_index_dictionary(json_path='./data/encode_dictionary.json'):
    with open(json_path, 'r') as f:
        pretrained_dictionary = json.load(f)
    index_dictionary = {word: index + 1 for index, word in enumerate(pretrained_dictionary[0])}
    index_dictionary['<ESC>'] = 0

    return index_dictionary


def encode_sentence(doc, index_dictionary, max_len):
    encode = [index_dictionary[word] for word in doc.split() if word in index_dictionary.keys()]
    encode.reverse()
    encode.extend(0 for _ in range(max_len - len(encode)))
    encode.reverse()

    return encode


def get_embedding_weight(json_path='./data/encode_dictionary.json'):
    with open(json_path, 'r') as f:
        embeddings = json.load(f)[1]
    n_dim = len(embeddings[0])
    padding = [0 for _ in range(n_dim)]
    embedding_weights = []
    embedding_weights.extend([padding])
    embedding_weights.extend(embeddings)
    return embedding_weights
