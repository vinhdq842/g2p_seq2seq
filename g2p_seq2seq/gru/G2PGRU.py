import random

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        attention = self.v(energy).squeeze(2)

        return torch.softmax(attention, dim=1)


class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, enc_hidden_size, dec_hidden_size, dropout):
        super(Encoder, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=enc_hidden_size, bidirectional=True)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embed = self.dropout(self.embeddings(src))
        outputs, hidden = self.gru(embed)

        return outputs, torch.tanh(self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1)))


class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, enc_hidden_size, dec_hidden_size, dropout, attention):
        super(Decoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.attention = attention
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.gru = nn.GRU(input_size=enc_hidden_size * 2 + embedding_dim, hidden_size=dec_hidden_size)
        self.fc = nn.Linear(enc_hidden_size * 2 + dec_hidden_size + embedding_dim, num_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        x = input.unsqueeze(0)
        embed = self.dropout(self.embeddings(x))
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        gru_input = torch.cat((embed, weighted), dim=2)
        output, hidden = self.gru(gru_input, hidden.unsqueeze(0))
        assert (output == hidden).all()

        embed = embed.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        logits = self.fc(torch.cat((output, weighted, embed), dim=1))

        return logits, hidden.squeeze(0)


class G2PGRU(nn.Module):
    def __init__(self,
                 input_vocab=None,
                 output_vocab=None,
                 embedding_dim=300,
                 hidden_size=256,
                 dropout=0.5):
        super(G2PGRU, self).__init__()

        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.encoder = Encoder(len(input_vocab), embedding_dim, hidden_size, hidden_size, dropout)
        self.decoder = Decoder(len(output_vocab), embedding_dim, hidden_size, hidden_size, dropout,
                               Attention(hidden_size, hidden_size))

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.num_embeddings

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(src.device)

        encoder_outputs, hidden = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            logits, hidden = self.decoder(input, hidden, encoder_outputs)

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
        encoder_outputs, hidden = self.encoder(src)
        input = torch.tensor([self.output_vocab['<sos>']] * batch_size).to(src.device)

        for i in range(1, max_len):
            logits, hidden = self.decoder(input, hidden, encoder_outputs)
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
