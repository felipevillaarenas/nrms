"""**Train**.
This module train the News Recommendation System.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
import bcolz

import torch
from torch.utils import data

import pytorch_lightning as pl
from pytorch_ranger import Ranger

from dataset import NewsDataset
from model.net import NRMS
from metric import ndcg_score, mrr_score
from config import hparams
from vocab import extract_glove_vocab


class Model(pl.LightningModule):
    """
    Define how to train the model using LightningModule.
    """
    def __init__(self, hparams):
        """Initialization of parameters.

        Args:
            params (dict): Dictionary of configuration parameters.
        """
        super(Model, self).__init__()
        self.hparams = hparams
        self.embeddings = self.init_embedding()
        self.model = NRMS(hparams['model'], self.embeddings)

    def init_embedding(self):
        """Load pre-trained embeddings as a constant tensor.

        Args:
            file_path (str): the pre-trained embeddings filename.

        Returns:
            obj: A constant tensor.
        """
        glove_path = self.hparams['glove_path']
        embed_size = self.hparams['model']['embed_size']
        max_vocab_size = self.hparams['max_vocab_size']
        extract_glove_vocab(glove_path, embed_size, max_vocab_size)
        vectors = bcolz.open(f'{glove_path}/6B.'+str(embed_size)+'.dat')[:]
        embeddings = torch.Tensor(vectors)
        if hparams['model']['dct_size'] == 'auto':
            hparams['model']['dct_size'] = embeddings.shape[0]
        return embeddings

    def configure_optimizers(self):
        """Optimizer configuration.
        
        Returns:
            object: Optimizer.
        """
        optimizer = Ranger( self.parameters(),
                            lr=self.hparams['lr'],
                            weight_decay=1e-5)
        return optimizer

    def setup(self,stage):
        """
        Data set definition according to stage.

        Args:
            stage (str): Modeling Stage.
        """
        if stage == 'fit':
            train_ds = NewsDataset(self.hparams, self.hparams['path_train_data'])
            val_ds = NewsDataset(self.hparams, self.hparams['path_val_data'])
            self.train_ds, _ = data.random_split(train_ds, [len(train_ds)-int(len(train_ds)*0.99), int(len(train_ds)*0.99)])
            self.val_ds, _ = data.random_split(val_ds, [len(val_ds)-int(len(val_ds)*0.95), int(len(val_ds)*0.95)])

        if stage == 'test':
            self.test_ds = NewsDataset(self.hparams, self.hparams['path_val_data'])

    def train_dataloader(self):
        """Build Data loader from train dataset.

        Returns:
            dataloader: Train data loader.
        """
        train_dataloader = data.DataLoader(self.train_ds,
                                        num_workers=self.hparams['num_workers'],
                                        batch_size=self.hparams['batch_size'],
                                        shuffle=self.hparams['shuffle'])
        return train_dataloader

    def val_dataloader(self):
        """Build Data loader from validation dataset.

        Returns:
            dataloader: Validation data loader.
        """
        val_dataloader = data.DataLoader(self.val_ds,
                                        num_workers=self.hparams['num_workers'],
                                        batch_size=self.hparams['batch_size'],
                                        shuffle=self.hparams['shuffle'])
        return val_dataloader

    def test_dataloader(self):
        """Build Data loader from test dataset.

        Returns:
            dataloader: Test data loader.
        """
        test_dataloader = data.DataLoader(self.test_ds,
                                        num_workers=self.hparams['num_workers'],
                                        batch_size=self.hparams['batch_size'],
                                        shuffle=self.hparams['shuffle'])
        return test_dataloader

    def forward(self):
        """Forward.

        Define as normal pytorch model.
        """
        return None

    def training_step(self, batch, _):
        """For each step(batch).

        Args:
            batch {[type]} -- data
            batch_idx {[type]}
        """
        clicks, cands, labels = batch
        loss, _ = self.model(clicks, cands, labels)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        """For each epoch end.

        Args:
            outputs: Loss values.

        Returns:
            dict: Logs loss mean.
        """
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'train_loss': loss_mean}
        self.model.eval()
        return {'progress_bar': logs, 'log': logs}

    def validation_step(self, batch, _):
        """For each step(batch).

        Args:
            batch {[type]} -- data
            batch_idx {[type]}

        Returns:
            dict: Evaluation metrics on training step.
        """
        clicks, cands, cands_label = batch
        with torch.no_grad():
            logits = self.model(clicks, cands)
        mrr = 0.
        auc = 0.
        ndcg5, ndcg10 = 0., 0.

        for score, label in zip(logits, cands_label):
            auc += pl.metrics.functional.auroc(score, label)
            score = score.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            mrr += mrr_score(label, score)
            ndcg5 += ndcg_score(label, score, 5)
            ndcg10 += ndcg_score(label, score, 10)

        auroc = (auc / logits.shape[0]).item()
        mrr = (mrr / logits.shape[0]).item()
        ndcg5 = (ndcg5 / logits.shape[0]).item()
        ndcg10 = (ndcg10 / logits.shape[0]).item()

        return {'auroc': auroc, 'mrr': mrr, 'ndcg5': ndcg5, 'ndcg10': ndcg10}

    def validation_epoch_end(self, outputs):
        """Validation end.

        Args:
            outputs (dict): History per evaluation metric.
        Reruns:
            dict: Logs of metrics.
        """
        mrr = torch.Tensor([x['mrr'] for x in outputs])
        auroc = torch.Tensor([x['auroc'] for x in outputs])
        ndcg5 = torch.Tensor([x['ndcg5'] for x in outputs])
        ndcg10 = torch.Tensor([x['ndcg10'] for x in outputs])

        logs = {'auroc': auroc.mean(),
                'mrr': mrr.mean(),
                'ndcg@5': ndcg5.mean(),
                'ndcg@10': ndcg10.mean()}
        self.model.train()
        return {'progress_bar': logs, 'log': logs}


if __name__ == '__main__':
    from pytorch_lightning import Trainer
    from pytorch_lightning.profiler import AdvancedProfiler

    profiler = AdvancedProfiler()
    model = Model(hparams)
    trainer = Trainer(max_epochs=2, profiler=profiler)
    trainer.fit(model)
