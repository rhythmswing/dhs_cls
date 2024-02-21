
from argparse import ArgumentParser
import numpy as np
from src.configs import *

parser = ArgumentParser()

parser.add_argument("model_checkpoint", type=str)
parser.add_argument("--model_type", type=str, required=True)
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)


import pytorch_lightning as pl


args = parser.parse_args()
pl.seed_everything(args.seed)

from src.framework import DHSDetector, LinearClassificationModule, DHSDataModule, DHSTrainingModule, DHSMoEDetector
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


import torch
ckpt = torch.load(args.model_checkpoint)
framework.load_state_dict(ckpt['state_dict'])


datamodule.prepare_data()
datamodule.setup('any')


labels_all = []
preds_all = []
scores_all = []

framework.cuda(3)
for t in datamodule.test_dataloader():
    for key, value in t.items():
        if isinstance(value, torch.Tensor):
            t[key] = value.cuda(3)
    scores = framework.model(t)
    labels = t['label']
    scores_all.append(scores.detach().cpu().numpy())

    preds_all.append(scores.argmax(dim=1).detach().cpu().numpy())
    labels_all.append(labels.argmax(dim=1).detach().cpu().numpy())

import numpy as np
from sklearn.metrics import classification_report
preds_all = np.concatenate(preds_all)
labels_all = np.concatenate(labels_all)
scores = np.concatenate(scores_all)

import pickle as pkl
biosample_type_map = pkl.load(open("data/biosample_type_map.pkl", "rb"))

print(classification_report(labels_all, preds_all))

cancer_preds = (preds_all == 0) | (labels_all == 2)
cancer_labels = (labels_all == 0) | (labels_all == 2)

#print(cancer_labels)

print(classification_report(cancer_labels, cancer_preds))