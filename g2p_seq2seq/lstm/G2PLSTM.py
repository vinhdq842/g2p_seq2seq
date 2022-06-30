import torch
from torch import nn
import random


class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embed = self.dropout(self.embeddings(src))
        outputs, (hidden, cell) = self.lstm(embed)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        x = input.unsqueeze(0)
        embed = self.dropout(self.embeddings(x))
        output, (hidden, cell) = self.lstm(embed, (hidden, cell))
        logits = self.fc(output.squeeze(0))
        return logits, hidden, cell


class G2PLSTM(nn.Module):
    def __init__(self,
                 input_vocab=None,
                 output_vocab=None,
                 embedding_dim=300,
                 hidden_size=256,
                 num_layers=2, dropout=0.5):
        super(G2PLSTM, self).__init__()

        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.encoder = Encoder(len(input_vocab), embedding_dim, hidden_size, num_layers, dropout)
        self.decoder = Decoder(len(output_vocab), embedding_dim, hidden_size, num_layers, dropout)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.num_embeddings

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(src.device)

        hidden, cell = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            logits, hidden, cell = self.decoder(input, hidden, cell)

            outputs[t] = logits
            teacher_force = random.random() < teacher_forcing_ratio

            top1 = logits.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

    def infer(self, src):
        hidden, cell = self.encoder(src)
        input = torch.tensor([self.output_vocab['<sos>']])
        res = []
        rev_output_dict = dict((v, k) for k, v in self.output_vocab.items())

        while True:
            logits, hidden, cell = self.decoder(input, hidden, cell)
            input = logits.argmax(-1)
            if input.item() == self.output_vocab['<pad>']:
                break
            token = rev_output_dict[input.item()]
            res.append(token)

        return ' '.join(res)
