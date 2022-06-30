from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch


def encode(arr, vocab, maxlen=25, trg=False):
    res = []
    for row in arr:
        res.append([vocab[i] for i in row])

    res = torch.from_numpy(pad_sequences(res, padding='post', value=vocab['<pad>'], maxlen=maxlen)).to(torch.int64)
    return torch.hstack(
        (vocab['<sos>'] * torch.ones((res.shape[0], 1), dtype=torch.int64),
         res,
         vocab['<eos>'] * torch.ones((res.shape[0], 1), dtype=torch.int64))
    ) if trg else res


def stat(content):
    s1 = set()
    s2 = set()
    src = []
    trg = []
    for line in content:
        tmp = line.split('\t')
        src.append(list(tmp[0]))
        trg.append(tmp[1].split())
        s1.update(src[-1])
        s2.update(trg[-1])

    return src, trg, s1, s2
