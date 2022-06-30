import argparse
import os
import pickle

import numpy as np
import torch
from jiwer import wer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from g2p_seq2seq.cnn.G2PCNN import G2PCNN
from g2p_seq2seq.utils import encode, stat


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_dataloader, val_dataloader, optimizer, criterion, epochs, device, patient):
    model.to(device)
    criterion.to(device)
    best_val_loss = 100000
    current_patient = 0
    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_acc = []
        for src, trg in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}'):
            src, trg = src.to(device), trg.to(device)
            output, _ = model(src, trg[:, :-1])

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += [loss.item()]
            train_acc += (output.argmax(1) == trg).tolist()

        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)

        model.eval()
        val_loss = []
        val_acc = []
        with torch.no_grad():
            for src, trg in val_dataloader:
                src, trg = src.to(device), trg.to(device)
                output, _ = model(src, trg[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                val_loss += [criterion(output, trg).item()]
                val_acc += (output.argmax(1) == trg).tolist()

        val_loss = np.mean(val_loss)
        val_acc = np.mean(val_acc)

        print(
            f'Training loss: {train_loss:.5f} | Training acc: {train_acc:.2f} | Val loss: {val_loss:.5f} | Val acc: {val_acc:.2f}'
        )

        if best_val_loss > val_loss:
            current_patient = 0
            torch.save(model.state_dict(), 'checkpoint/g2p_seq2seq.pth')
            best_val_loss = val_loss
            print('Checkpoint saved successfully!')
        else:
            current_patient += 1
            if current_patient == patient:
                print(f'Stop training since not improved in the last {patient} epochs...')
                return


def evaluate(model, test_dataloader, device, rev_output_vocab):
    model.eval()
    model.to(device)
    acc = []
    infer = []
    ground = []
    max_len = 25
    with torch.no_grad():
        for src, trg in test_dataloader:
            src, trg = src.to(device), trg.to(device)
            encoder_conved, encoder_combined = model.encoder(src)

            pred = torch.zeros((src.shape[0], max_len), dtype=torch.int64)
            pred[:, 0] = torch.ones(src.shape[0])

            for i in range(1, max_len):
                trg_tensor = torch.LongTensor(pred[:, :i]).to(device)
                print(trg_tensor.shape)
                output, attention = model.decoder(trg_tensor, encoder_conved, encoder_combined)
                pred[:, i] = output.argmax(2)[:, -1]

            for i in range(src.shape[0]):
                org = ' '.join([rev_output_vocab[o.item()] for o in trg[i, :] if o.item() > 2])

                pred1 = []
                for tk in pred[i, 1:]:
                    if tk < 2:
                        break
                    pred1.append(rev_output_vocab[tk.item()])

                pred1 = ' '.join(pred1)
                infer.append(pred1)
                ground.append(org)
                acc.append(org == pred1)

        print(f'Accuracy: {np.mean(acc):.2f}')
        print(f'WER: {wer(ground, infer):.6f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='train.dict')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--dev_file', type=str, default='dev.dict')
    parser.add_argument('--test_file', type=str, default='test.dict')
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--early_stopping_patient', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_seq_length', type=int, default=25)

    args = parser.parse_args()

    train_data = open(args.train_file).read().strip().split('\n')

    src, trg, input_vocab, output_vocab = stat(train_data)
    input_vocab = dict(zip(['<pad>'] + sorted(input_vocab), range(len(input_vocab) + 1)))
    output_vocab = dict(zip(['<pad>', '<sos>', '<eos>'] + sorted(output_vocab), range(len(output_vocab) + 3)))

    src = encode(src, input_vocab, args.max_seq_length)
    trg = encode(trg, output_vocab, args.max_seq_length, True)
    train_dataloader = DataLoader(TensorDataset(src, trg), batch_size=args.batch_size, shuffle=True)

    val_data = open(args.dev_file).read().strip().split('\n')

    src_val, trg_val, _, _ = stat(val_data)
    src_val = encode(src_val, input_vocab, args.max_seq_length)
    trg_val = encode(trg_val, output_vocab, args.max_seq_length, True)
    val_dataloader = DataLoader(TensorDataset(src_val, trg_val), batch_size=args.batch_size)

    g2p_model = G2PCNN(input_vocab, output_vocab, args.embedding_dim, args.hidden_size, args.num_encoder_layers,
                       args.num_decoder_layers, 3, 3, 0.25)

    print(f'The model has {count_parameters(g2p_model):,} trainable parameters')
    if args.model_path is not None:
        g2p_model.load_state_dict(torch.load(f'{args.model_path}/g2p_seq2seq.pth', map_location='cpu'))
        print("Checkpoint loaded...")

    path = "checkpoint/vocab.inp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(input_vocab, open('checkpoint/vocab.inp', 'wb'))
    pickle.dump(output_vocab, open('checkpoint/vocab.out', 'wb'))

    optimizer = torch.optim.Adam(g2p_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train(g2p_model, train_dataloader, val_dataloader, optimizer, criterion, args.epochs, device,
          args.early_stopping_patient)

    if args.test_file is not None:
        test_data = open(args.test_file).read().strip().split('\n')
        src_test, trg_test, _, _ = stat(test_data)
        src_test = encode(src_test, g2p_model.input_vocab, args.max_seq_length)
        trg_test = encode(trg_test, g2p_model.output_vocab, args.max_seq_length, True)
        test_dataloader = DataLoader(TensorDataset(src_test, trg_test), batch_size=args.batch_size)

        g2p_model.load_state_dict(torch.load(f'checkpoint/g2p_seq2seq.pth', map_location='cpu'))
        print('Best model loaded...')
        evaluate(g2p_model, test_dataloader, device, dict((v, k) for k, v in g2p_model.output_vocab.items()))


if __name__ == '__main__':
    main()