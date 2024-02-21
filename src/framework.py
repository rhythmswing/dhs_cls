
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from .output import LinearClassificationModule
from typing import List
from typing import Iterable, Union
from collections.abc import Iterable
from torch.utils.data import Dataset

from src.modules import DHSDetector, DHSMoEDetector
from src.data import DHSDataModule

from torch.optim import AdamW

import pandas as pd
import numpy as np
import tqdm
        
class DHSTrainingModule(pl.LightningModule):
    """
    A PyTorch Lightning Module for training, validating, and testing DNA sequence data models.

    Attributes:
        model (nn.Module): The neural network model to be trained.
    """

    def __init__(self, model: nn.Module): 
        super(DHSTrainingModule, self).__init__()

        self.model = model

    def step(self, batch, batch_idx, stage='train'):
        """
        A generic step method for processing a single batch of data, applicable to training, validation, and testing stages.

        Parameters:
            batch (dict): The batch of data.
            batch_idx (int): The index of the current batch.
            stage (str, optional): The stage of processing ('train', 'val', or 'test'). Defaults to 'train'.

        Returns:
            torch.Tensor: The loss for the current batch.
        """
        pred = self.model(batch)
        labels = batch['label']
        loss = self.loss_function(pred, labels)

        if stage == 'train':
            self.log(f'{stage}_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        else:
            self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage='test')

    def configure_optimizers(self):
        """
        Configures the optimizers (and learning rate schedulers, if necessary) for training the model.

        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        optimizer = AdamW(self.model.parameters(), lr=5e-6)
        #scheduler = torch.optim.lr_scheduler.StepLR(
        #    optimizer, step_size=1, gamma=0.1)
        return optimizer

    def loss_function(self, predictions, labels):
        """
        Defines the loss function for the model.

        Parameters:
            predictions (torch.Tensor): The model's predictions.
            labels (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        return torch.nn.functional.binary_cross_entropy_with_logits(predictions, labels.float())

    # def forward(self, dna_sequences, labels):
    #     dhs_pred = self.model(dna_sequences)
    #     loss = self.loss_function(dhs_pred, labels)
    #     return loss


if __name__ == '__main__':
    
    model = DHSDetector(base_model="zhihan1996/DNABERT-S", 
                        classification_module=LinearClassificationModule(768, 1))

    datamodule = DHSDataModule(feather_path="data/filtered_dataset.ftr", label_columns=(11, 15), 
                               feature_columns=["component", "total_signal"])
    datamodule.prepare_data()