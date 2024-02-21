

from src.framework import DHSDetector, LinearClassificationModule, DHSDataModule
from src.framework import DHSMoEDetector
from src.framework import DHSTrainingModule

from src.configs import *

from pytorch_lightning import Trainer
import pytorch_lightning as pl

import wandb

pl.seed_everything(42)

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_checkpoint", type=str, required=True)
parser.add_argument("--model_type", type=str, required=True)
args = parser.parse_args()


if __name__ == '__main__':
    
    #wandb.init(project="dhs-detector")
    #model = DHSDetector(base_model="zhihan1996/DNABERT-S", 
    #                    classification_module=LinearClassificationModule(768, 4))

    match args.model_type:
        case "linear":
            framework, datamodule = filtered_dnabert_linear(4)
        case "moe_linear":
            framework, datamodule = filtered_dnabert_moe_linear(4, 20)

        case "two_layer": 
            framework, datamodule = filtered_dnabert_two_layer(4)
        case "moe_two_layer":
            framework, datamodule = filtered_dnabert_moe_two_layer(4, 20)
        case "moe_rbf":
            framework, datamodule = filtered_dnabert_moe_rbf(4, 20,
                                    feature_columns=['total_signal', 'proportion'],
                                    rbf_dimension=128,)
        
    # datamodule = DHSDataModule(feather_path="data/filtered_dataset.ftr", 
    #                            label_columns=(11, 15), batch_size=32,
    #                            feature_columns=["component", "total_signal", "proportion"])


    datamodule.prepare_data()
    datamodule.setup('fit')

    train_loader = datamodule.train_dataloader()

    callbacks = []

    import os
    os.makedirs(args.model_checkpoint, exist_ok=True)
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.model_checkpoint,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger(project="dhs-detector")
    #wandb_logger = pl.loggers.WandbLogger()
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    callbacks.append(model_checkpoint)
    callbacks.append(early_stopping)

    trainer = Trainer(max_epochs=100,
                      callbacks=callbacks,
                      logger=wandb_logger,)

    trainer.fit(framework, datamodule)

    trainer.test(framework, datamodule.test_dataloader())
