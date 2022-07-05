import torch
from torch import nn
import torch.nn.functional as F

from g2p_seq2seq.utils import Beam


class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers, kernel_size, dropout, max_length=100):
        super(Encoder, self).__init__()

        assert kernel_size % 2 == 1, "Kernel_size must be odd"

        self.scale = torch.sqrt(torch.tensor([.5]))

        self.token_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.pos_embeddings = nn.Embedding(max_length, embedding_dim)

        self.emb2hid = nn.Linear(embedding_dim, hidden_size)
        self.hid2emb = nn.Linear(hidden_size, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size,
                      out_channels=2 * hidden_size,
                      kernel_size=kernel_size,
                      padding=(kernel_size - 1) // 2)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        self.scale = self.scale.to(src.device)
        batch_size, src_len = src.shape
        pos = torch.arange(src_len).unsqueeze(0).repeat(batch_size, 1).to(src.device)

        token_embeddings = self.token_embeddings(src)
        pos_embeddings = self.pos_embeddings(pos)

        embed = self.dropout(token_embeddings + pos_embeddings)
        conv_inp = self.emb2hid(embed).permute(0, 2, 1)

        for conv in self.convs:
            conved = conv(self.dropout(conv_inp))

            conved = F.glu(conved, dim=1)
            conved = (conved + conv_inp) * self.scale
            conv_inp = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))
        combined = (conved + embed) * self.scale

        return conved, combined


class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers, kernel_size, dropout, trg_pad_idx,
                 max_length=100):
        super(Decoder, self).__init__()

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.scale = torch.sqrt(torch.tensor([.5]))

        self.token_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.pos_embeddings = nn.Embedding(max_length, embedding_dim)

        self.emb2hid = nn.Linear(embedding_dim, hidden_size)
        self.hid2emb = nn.Linear(hidden_size, embedding_dim)

        self.attn_hid2emb = nn.Linear(hidden_size, embedding_dim)
        self.attn_emb2hid = nn.Linear(embedding_dim, hidden_size)

        self.fc = nn.Linear(embedding_dim, num_embeddings)

        self.convs = nn.ModuleList(
            [nn.Conv1d(hidden_size, 2 * hidden_size, kernel_size=kernel_size) for _ in
             range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

    def calc_attention(self, embed, conved, encoder_conved, encoder_combined):
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
        combined = (conved_emb + embed) * self.scale

        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        attention = torch.softmax(energy, dim=2)

        attended_encoding = torch.matmul(attention, encoder_combined)

        attended_encoding = self.attn_emb2hid(attended_encoding)
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale

        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):
        batch_size, trg_len = trg.shape
        self.scale = self.scale.to(trg.device)

        pos = torch.arange(trg_len).unsqueeze(0).repeat(batch_size, 1).to(trg.device)

        token_embeddings = self.token_embeddings(trg)
        pos_embeddings = self.pos_embeddings(pos)

        embed = self.dropout(token_embeddings + pos_embeddings)

        conv_input = self.emb2hid(embed).permute(0, 2, 1)

        batch_size, hid_dim = conv_input.shape[0], conv_input.shape[1]

        for conv in self.convs:
            conv_input = self.dropout(conv_input)
            padding = torch.zeros(batch_size, hid_dim, self.kernel_size - 1).to(trg.device)

            padded_conv_input = torch.cat((padding, conv_input), dim=2)

            conved = conv(padded_conv_input)
            conved = F.glu(conved, dim=1)
            attention, conved = self.calc_attention(embed, conved, encoder_conved, encoder_combined)
            conved = (conved + conv_input) * self.scale

            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))

        output = self.fc(self.dropout(conved))

        return output, attention


class G2PCNN(nn.Module):
    def __init__(self,
                 input_vocab=None,
                 output_vocab=None,
                 embedding_dim=300,
                 hidden_size=256,
                 num_encoder_layers=10,
                 num_decoder_layers=10,
                 encoder_kernel_size=3,
                 decoder_kernel_size=3,
                 dropout=0.25):
        super(G2PCNN, self).__init__()
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.encoder = Encoder(len(input_vocab), embedding_dim, hidden_size, num_encoder_layers, encoder_kernel_size,
                               dropout)
        self.decoder = Decoder(len(output_vocab), embedding_dim, hidden_size, num_decoder_layers, decoder_kernel_size,
                               dropout, 0)

    def forward(self, src, trg):
        encoder_conved, encoder_combined = self.encoder(src)
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)

        return output, attention

    def infer(self, src):
        max_len = 25
        res = []
        preds = torch.zeros((src.shape[0], max_len), dtype=torch.int64)
        preds[:, 0] = torch.tensor([self.output_vocab['<sos>']] * src.shape[0])
        rev_output_vocab = dict((v, k) for k, v in self.output_vocab.items())

        encoder_conved, encoder_combined = self.encoder(src)
        for i in range(1, max_len):
            trg_tensor = torch.LongTensor(preds[:, :i]).to(src.device)
            output, _ = self.decoder(trg_tensor, encoder_conved, encoder_combined)
            preds[:, i] = output.argmax(2)[:, -1]

        for i in range(src.shape[0]):
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
            beam = Beam(size=beam_width, pad=self.output_vocab['<pad>'], bos=self.output_vocab['<sos>'],
                        eos=self.output_vocab['<eos>'], cuda=src.is_cuda)

            preds = torch.zeros((beam_width, max_len), dtype=torch.int64)
            preds[:, 0] = beam.get_current_state()
            src = src.unsqueeze(0)
            encoder_conved, encoder_combined = self.encoder(src)

            for i in range(1, max_len):
                trg_tensor = torch.LongTensor(preds[:, :i]).to(src.device)
                output, _ = self.decoder(trg_tensor, encoder_conved, encoder_combined)

                if beam.advance(torch.log_softmax(output[:, -1, :], -1)):
                    break

                preds[:, :i] = preds[beam.get_current_origin(), :i]
                preds[:, i] = beam.get_current_state()

            preds = torch.LongTensor(beam.get_hyp(0))

            pred = []
            for tkn in preds:
                if tkn < 3:
                    break
                pred.append(rev_output_vocab[tkn.item()])
            res.append(' '.join(pred))

        return res
