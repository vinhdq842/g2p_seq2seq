import torch
from torch import nn
import random

from g2p_seq2seq.utils import Beam


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
                 num_layers=3,
                 dropout=0.5):
        super(G2PLSTM, self).__init__()

        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.encoder = Encoder(len(input_vocab), embedding_dim, hidden_size, num_layers, dropout)
        self.decoder = Decoder(len(output_vocab), embedding_dim, hidden_size, num_layers, dropout)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len, batch_size = trg.shape
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
        max_len = 25
        res = []
        batch_size = src.shape[0]
        preds = torch.zeros((batch_size, max_len), dtype=torch.int64).to(src.device)
        preds[:, 0] = torch.ones(batch_size).to(src.device)
        rev_output_vocab = dict((v, k) for k, v in self.output_vocab.items())

        src = src.permute(1, 0)
        hidden, cell = self.encoder(src)
        input = torch.tensor([self.output_vocab['<sos>']] * batch_size).to(src.device)

        for i in range(1, max_len):
            logits, hidden, cell = self.decoder(input, hidden, cell)
            input = logits.argmax(-1)
            preds[:, i] = input

        for i in range(batch_size):
            pred = []
            for tkn in preds[i, 1:]:
                if tkn < 3:
                    break
                pred.append(rev_output_vocab[tkn.item()])

            res.append(' '.join(pred))

        return res

    def infer_beam(self, beam_width, src_batch):
        max_len = 25
        rev_output_vocab = dict((v, k) for k, v in self.output_vocab.items())
        res = []

        for src in src_batch:
            src = src.unsqueeze(1).expand(-1, beam_width)
            hidden, cell = self.encoder(src)

            beam = Beam(size=beam_width, pad=self.output_vocab['<pad>'], bos=self.output_vocab['<sos>'],
                        eos=self.output_vocab['<eos>'], cuda=src.is_cuda)

            for i in range(max_len):
                input = beam.get_current_state()

                logits, hidden, cell = self.decoder(input, hidden, cell)
                # k x vocab_size | n_layers * n_direction x batch x hid_dim
                if beam.advance(torch.log_softmax(logits, -1)):
                    break

                hidden = hidden[:, beam.get_current_origin(), :]
                cell = cell[:, beam.get_current_origin(), :]

            preds = torch.LongTensor(beam.get_hyp(0))
            pred = []
            for tkn in preds:
                if tkn < 3:
                    break
                pred.append(rev_output_vocab[tkn.item()])
            res.append(' '.join(pred))

        return res
