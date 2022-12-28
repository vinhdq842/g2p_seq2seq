import torch

MODEL_FILE_NAME = 'g2p_seq2seq_gru.pth'


def encode(arr, vocab, maxlen=25, trg=False):
    res = []
    for row in arr:
        res.append([vocab[i] for i in row] + [vocab['<eos>']])
        if trg:
            res[-1] = [vocab['<sos>']] + res[-1]
        res[-1] += [vocab['<pad>']] * (maxlen - len(res[-1]) - (2 if trg else 1))

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


class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, pad=0, bos=1, eos=2, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The back-pointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from

        prev_k = torch.div(bestScoresId, num_words, rounding_mode='floor')
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True
        return self.done

    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]
        return hyp[::-1]
