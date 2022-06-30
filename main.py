import numpy

from g2p_seq2seq.g2p import G2PPyTorch
from time import time

model = G2PPyTorch()
model.load_model('checkpoint')
sentences = ['Noul nostru președinte  a fost cântăreț', 'Ultimul său album a fost cu adevărat uimitor',
             'Dorim o masă lângă fereastră', 'Această țară are nevoie de cineva care să ne poată conduce',
             'ot să vin mâine târziu la serviciu', 'La ce restaurant am fost săptămâna trecută',
             'Te rog să mă trezești la ora opt']
times = []
lens = []
for sentence in sentences:
    lens.append(len(sentence.split()))
    res = ''
    a = time()
    for word in sentence.lower().split():
        res += model.decode_word(word) + ' ._. '
    print(res)
    times.append(time() - a)

print(numpy.mean(times))
print(numpy.mean(lens))
