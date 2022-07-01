import pickle

import torch

from g2p_seq2seq.cnn.G2PCNN import G2PCNN
from g2p_seq2seq.gru.G2PGRU import G2PGRU
from g2p_seq2seq.utils import encode

model_file_name = 'g2p_seq2seq_cnn10.pth'


class G2PPyTorch:
    def __init__(self):
        self.model = None

    def load_model(self, path_to_model):
        input_vocab = pickle.load(open(f'{path_to_model}/vocab.inp', 'rb'))
        output_vocab = pickle.load(open(f'{path_to_model}/vocab.out', 'rb'))
        self.model = G2PCNN(input_vocab, output_vocab)
        self.model.load_state_dict(torch.load(f'{path_to_model}/{model_file_name}', map_location=torch.device('cpu')))
        self.model.to(device=torch.device('cpu'))
        self.model.eval()

    def decode_word(self, words):
        words = encode([list(word) for word in words], self.model.input_vocab)

        with torch.no_grad():
            res = self.model.infer(words)
        return res
