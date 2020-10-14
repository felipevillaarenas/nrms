"""**Vocabulary**.
This module extacts the pretrained vectors and creates a dictionary word to index for the top words.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import logging
import pickle
import bcolz
import numpy as np
from tqdm import tqdm
from config import hparams


logger = logging.getLogger()

def extract_glove_vocab(glove_path, word_embedding_dim, max_vocab_size):
    """Extacts the pretrained vectors and creates a dictionary word to index for the top words.

    Args:
        glove_path (str): Path to the GloVec file.
        word_embedding_dim (int): Word vector dimension.
        max_vocab_size (int): Vocabulary size.
    """
    words = []
    idx = 1
    word2idx = {}
    vectors = bcolz.carray( np.zeros(1),
                            rootdir=f'{glove_path}/6B.'+str(word_embedding_dim)+'.dat',
                            mode='w')
    vectors.append(np.zeros(word_embedding_dim))
    logger.info("Generating vocabulary dictionary...")
    vocab_size = 0
    with open(f'{glove_path}/glove.6B.'+str(word_embedding_dim)+'d.txt', 'rb') as file:
        for line in tqdm(file):
            line = line.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vector = np.array(line[1:]).astype(np.float)
            vectors.append(vector)
            vocab_size+=1
            if vocab_size >= max_vocab_size:
                break
    vectors = bcolz.carray( vectors[1:].reshape((-1, word_embedding_dim)),
                            rootdir=f'{glove_path}/6B.'+str(word_embedding_dim)+'.dat',
                            mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/6B.'+str(word_embedding_dim)+'_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.'+str(word_embedding_dim)+'_idx.pkl', 'wb'))


if __name__ == '__main__':
    print('Extracting Word2idx and GloVe vectors..')
    extract_glove_vocab(hparams['glove_path'],
                        hparams['model']['embed_size'],
                        hparams['max_vocab_size'])
