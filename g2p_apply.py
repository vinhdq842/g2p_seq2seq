import argparse

from tqdm import tqdm

from g2p_seq2seq.g2p import G2PPyTorch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoint')
    parser.add_argument('--word_list', type=str, default='test.wlist')

    args = parser.parse_args()

    model = G2PPyTorch()
    model.load_model(args.model_path)

    words = open(args.word_list, 'r').read().strip().split('\n')
    res = [word + '\t' + model.decode_word(word) for word in tqdm(words)]
    open('result.dict', 'w').write('\n'.join(res))
    print('Result successfully saved to result.dict...')