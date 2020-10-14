"""**Dataset**.
This module creates dataset by join the user click behaviors and news titles.
Aditionally, each title is tokenized and indexed.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
import pickle
import random
import spacy
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from config import hparams


class NewsDataset(Dataset):
    """
    Creates dataset from click behaviors and news titles.
    """
    def __init__(self, params, path_data):
        """Init paramaters for dataset building.
        
        Args:
            params (dict): Dictionary of configuration parameters.
            path_data (str): Path to data.
        """
        super(NewsDataset, self).__init__()
        self.hparams = params
        self.path_data = path_data
        self.tokenizer = spacy.load('en')
        self.news = self.get_words()
        self.behaviors = self.get_clickhistory()
        self.indexed_vocabulary = self.load_indexer()
        self.news_indexed = self.get_news_indexed()

    def load_indexer(self):
        """Load the dictionary word to index.

        Returns:
            dict: word to index.
        """
        path = self.hparams['glove_path']+'/6B.'+str(self.hparams['model']['embed_size'])+'_idx.pkl'
        indexed_vocabulary = pickle.load(open(path, 'rb'))
        return indexed_vocabulary

    def get_clickhistory(self):
        """Read click history file.

        Returns:
            list: List of user session with user_id, clicks, positive and negative interactions.
        """
        with open(self.path_data+'/behaviors.tsv') as file:
            lines = file.readlines()
        sessions = []
        for line in lines:
            _, userid, _, click, imps = line.strip().split("\t")
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

    def get_words(self):
        """Load words and entities.

        Returns:
            list: Words and entities dictionaries.
        """
        news_words = {}
        news_words = self.read_news(self.path_data+'/news.tsv',
                                     news_words)
        return news_words

    def read_news(self,filepath, news_words):
        """Get tokens from sentences.

        Args:
            filepath (str): Path to news.
            news_words (list): List of words.

        Returns:
            list[list]: New tokenized.
        """
        with open(filepath, encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            splitted = line.strip("\n").split("\t")
            header = splitted[3].lower()
            news_words[splitted[0]] = [tok.text for tok in self.tokenizer.tokenizer(header)]
        return news_words

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

    def get_news_indexed(self):
        """Get news indexed.

        This function builds a dictionary with News ID as Key and the header
        list of words as values.

        Returns:
            list[list]: News Indexed.
        """
        news_indexed = {}
        for newsid in self.news.keys():
            single_news_indexed = [self.word2idx(word) for word in self.news[newsid]]
            maxlen = self.hparams['data']['maxlen']
            if len(single_news_indexed) < maxlen:
                pad_news = [0 for i in range(maxlen-len(single_news_indexed))]
                single_news_indexed = single_news_indexed + pad_news
            else:
                single_news_indexed = single_news_indexed[:maxlen]
            news_indexed[newsid] = single_news_indexed
        return news_indexed

    def __len__(self):
        """Dataset length.

        Returns:
            int: Dataset length.
        """
        return len(self.behaviors)

    def __getitem__(self, idx: int):
        """Get item.

        Args:
            idx (int): Index.
        Returns:
            {
                str: User ID,
                list: [batch, num_click_docs, seq_len],
                list: [batch, num_candidate_docs, seq_len],
                bool: candidate docs label (0 or 1)
            }
        """
        click_doc = self.get_click_doc(idx)
        cand_doc = self.get_cand_doc(idx)
        cand_doc_label = self.get_cand_doc_label()
        tmp = list(zip(cand_doc, cand_doc_label))
        random.shuffle(tmp)
        cand_doc, cand_doc_label = zip(*tmp)
        return torch.tensor(click_doc), torch.tensor(cand_doc), torch.tensor(cand_doc_label)

    def get_click_doc(self, idx: int):
        """Get viewed news.

        Args:
            idx (int): Dataset index.

        Returns:
            list: News viewed by the user indexed.
        """
        maxlen = self.hparams['data']['maxlen']
        maxnews = self.hparams['data']['pos_k']
        try:
            click_doc = [self.news_indexed[newsid] for newsid in self.behaviors[idx][1][:maxlen]]
        except:
            click_doc = []
        if len(click_doc) < maxnews:
            empty_news = [0 for i in range(maxlen)]
            pad_news = [empty_news for missing_news in range(maxnews-len(click_doc))]
            click_doc = click_doc + pad_news
        return click_doc

    def get_cand_doc(self, idx: int):
        """Get Candidate News.

        Args:
            idx (int): Dataset index.

        Return:
            list: Candidate news indexed.
        """
        neg_k = self.hparams['data']['neg_k']
        pos_id = self.behaviors[idx][2]
        tmp = self.behaviors[idx][3]
        maxlen = self.hparams['data']['maxlen']
        random.shuffle(tmp)
        neg_id = tmp[:neg_k]
        cand_doc = [self.news_indexed[id] for id in pos_id+neg_id]
        if len(cand_doc) < neg_k+1:
            empty_news = [0 for i in range(maxlen)]
            cand_doc = cand_doc + [empty_news for i in range(neg_k+1 - len(cand_doc))]
        return cand_doc

    def get_cand_doc_label(self):
        """Get candidate news labels.

        Return:
            list: True label and negative sample lables.
        """
        doc_label = [1] + [0 for i in range(self.hparams['data']['neg_k'])]
        return doc_label

if __name__ == '__main__':
    ds = NewsDataset(hparams, './data/raw/MINDlarge_train')
    for i in tqdm(ds):
        pass
