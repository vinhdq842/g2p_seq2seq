import torch


def encode(arr, vocab, maxlen=25, trg=False):
    res = []
    for row in arr:
        res.append([vocab[i] for i in row])
        res[-1] += [vocab['<pad>']] * (maxlen - len(res[-1]) - (2 if trg else 0))
        if trg:
            res[-1] = [vocab['<sos>']] + res[-1] + [vocab['<eos>']]

    return torch.LongTensor(res)


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
