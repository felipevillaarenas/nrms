"""**Neural News Recommendation with Multi-Head Self-Attention at Inference Time**.

The Inference module is used to estimate news score from news titles.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import os
import sys
import inspect
import random
import bcolz
import pickle
import spacy

import torch
import pytorch_lightning as pl

from model.net import NRMS
from config import hparams


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
__location__ = os.path.join(os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe())))


class Model(pl.LightningModule):
    """
    NRMS for inference.
    """
    def __init__(self, hparams):
        """Init NRMS for inference.

        Args:
            hparams (dict): Configuration parameters.
        """
        super(Model, self).__init__()
        self.hparams = hparams
        self.embeddings = self.init_embedding()
        self.model = NRMS(hparams['model'], self.embeddings)
        self.indexed_vocabulary = self.load_indexer()

    def forward(self, viewed, cands, topk):
        """Forward.

        Args:
            viewed (tensor): [B, viewed_num, maxlen]
            cands (tensor): [B, cand_num, maxlen]
            topk (int): Number of top candidate news.

        Returns:
            val: [B] 0 ~ 1
            idx: [B] 
        """
        logits = self.model(viewed, cands)
        val, idx = logits.topk(topk)
        return idx, val
    
    def predict_one(self, viewed, cands, topk, news_ids):
        """Predict one user.

        Args:
            viewed (list[list]): Views news indexed.
            cands (list[List]): Candidate news indexed.
            topk (int): Number of top candidate news.

        Returns:
            {
                result: Indexed news reorder
                val: Score news.
                news_ids_reorder: News IDs ordered based on the highest score.
            }
        """
        viewed_token = torch.tensor(viewed).unsqueeze(0)
        cands_token = torch.tensor(cands).unsqueeze(0)
        idx, val = self(viewed_token, cands_token, topk)
        val = val.squeeze().detach().cpu().tolist()

        result = [cands[i] for i in idx.squeeze()]
        news_ids_reorder = [news_ids[i] for i in idx.squeeze()]
        return result, val, news_ids_reorder
    
    def init_embedding(self):
        """Load pre-trained embeddings as a constant tensor.
        
        Args:
            file_path (str): the pre-trained embeddings filename.

        Returns:
            obj: A constant tensor.
        """
        word_embedding_dim = self.hparams['model']['embed_size']
        glove_path = hparams['glove_path']
        vectors = bcolz.open(f'{glove_path}/6B.'+str(word_embedding_dim)+'.dat')[:]
        embeddings = torch.tensor(vectors).float()
        if hparams['model']['dct_size'] == 'auto':
            hparams['model']['dct_size'] = embeddings.shape[0]
        return embeddings
    
    def load_indexer(self):
        """Load the dictionary word to index.

        Returns:
            dict: word to index.
        """
        word_embedding_dim = hparams['model']['embed_size']
        glove_path = hparams['glove_path']

        indexed_vocabulary = pickle.load(open(f'{glove_path}/6B.'+str(word_embedding_dim)+'_idx.pkl', 'rb'))
        return indexed_vocabulary
    
    def word2idx(self, word): 
        """Indexer.

        Args:
            word (str): Key value.

        Retruns:
            int: Indicates the Embedding matrix position of the input word.
        """
        try:
            index = self.indexed_vocabulary[word]
        except:
            index = 0
        return index

    def get_clickhistory(self, behaviors):
        """Read click history file
        Args:
            behaviors (list[list]): History of clicked news per user.
        Returns:
            list, dict: List of user session with user_id, clicks, positive and negative interactions. Dictionary
                with user_id click history
        """
        userid_history = {}
        sessions = []
        for line in behaviors:
            userid, imp_time, click, imps = line.strip().split("\t")
            clicks = click.split(" ")
            pos = []
            neg = []
            imps = imps.split(" ")
            for imp in imps:
                if imp.split("-")[1] == "1":
                    pos.append(imp.split("-")[0])
                else:
                    neg.append(imp.split("-")[0])
            sessions.append([userid, clicks, pos, neg])
        return sessions

    def read_news(self, news, tokenizer):
        """Get tokens from sentences.

        Args:
            news (list[list]): News.
            tokenizer (object): tokenizer.
            
        Returns:
            list[list]: News tokenized.
        """
        news_words = {}
        for line in news:
            splitted = line.strip("\n").split("\t")
            news_words[splitted[0]] = [tok.text for tok in tokenizer.tokenizer(splitted[3].lower())]
        return news_words

    def get_words(self, news):
        """Load words and entities
        Args:
            news (list[list]): News.
        Returns: 
            list[list]: News tokenized.
        """
        
        tokenizer = spacy.load('en')
        news_words = self.read_news(news, tokenizer)
        return news_words

    def get_news_indexed(self, news):
        """Get news indexed.

        This function builds a dictionary with News ID as Key and the header
        list of words as values.

        Args:
            News (list[list]): News words.

        Returns:
            list[list]: News Indexed.
        """
        news_indexed = {}  
        maxlen = hparams['data']['maxlen']
        for newsid in news.keys():
            single_news_indexed = [self.word2idx(word) for word in news[newsid]]
            if len(single_news_indexed) < maxlen:
                single_news_indexed = single_news_indexed + [0 for i in range(maxlen-len(single_news_indexed))]
            else: 
                single_news_indexed = single_news_indexed[:maxlen]
            news_indexed[newsid] = single_news_indexed
        return news_indexed

def get_inference_analysis(nrms, news, news_indexed, behaviors, user_idx):
    """Inference analysis""""
    viewed_ids = behaviors[user_idx][1][:50]
    news_viewed = [news[newsid] for newsid in viewed_ids]
    viewed = [news_indexed[newsid] for newsid in viewed_ids]
    cand_news_ids = random.sample(news_indexed.keys(),200)
    cands = [news_indexed[newsid] for newsid in cand_news_ids] 
    result, val, news_ids_reorder = nrms.predict_one(viewed, cands, len(cand_news_ids), cand_news_ids)
    news_reorder = [news[newsid] for newsid in news_ids_reorder]
    return result, val, news_ids_reorder, news_reorder, news_viewed

if __name__ == '__main__':

    behaviors_path = os.path.join(hparams['path_test_data'],'behaviors.tsv')
    news_path = os.path.join(hparams['path_test_data'],'news.tsv')
    
    with open(news_path, encoding="utf-8") as f:
        news = f.readlines()
    
    with open(behaviors_path, encoding="utf-8") as f:
        behaviors = f.readlines()
    
    nrms = Model.load_from_checkpoint('models/ranger/v1/epoch=14-auroc=0.71.ckpt')

    news = nrms.get_words(news)
    news_indexed = nrms.get_news_indexed(news)
    behaviors = nrms.get_clickhistory(behaviors)        
        
    
    user_idx = 30
    result, val, news_ids_reorder, news_reorder, news_viewed = get_inference_analysis(nrms,news,news_indexed, behaviors,user_idx)
    
 