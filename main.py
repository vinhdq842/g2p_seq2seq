from time import time

import numpy

from g2p_seq2seq.g2p import G2PPyTorch

model = G2PPyTorch()
model.load_model('checkpoint')
sentences = ['Noul nostru președinte  a fost cântăreț', 'Ultimul său album a fost cu adevărat uimitor',
             'Dorim o masă lângă fereastră', 'Această țară are nevoie de cineva care să ne poată conduce',
             'ot să vin mâine târziu la serviciu', 'La ce restaurant am fost săptămâna trecută',
             'Te rog să mă trezești la ora opt']

times = []
for sentence in sentences:
    sentence = sentence.lower().split()
    a = time()
    print(model.decode_word(sentence))
    times.append(time() - a)
    print(f'{time() - a:.4}s')

print(numpy.mean(times))