

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from typing import List

class IdentityDNAFeatureTransformer(torch.nn.Module):
    def __init__(self):
        super(IdentityDNAFeatureTransformer, self).__init__()

    def forward(self, x):
        return x

class GaussianSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(GaussianSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(self.cutoff_lower, self.cutoff_upper, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))
        
class DHSDetector(torch.nn.Module):
    def __init__(self, base_model="zhihan1996/DNABERT-S", dna_feature_transformer=None, classification_module=None):
        super(DHSDetector, self).__init__()

        self.dna_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.dna_encoder_model = AutoModel.from_pretrained(base_model, trust_remote_code=True)

        if dna_feature_transformer is not None:
            self.dna_feature_transformer = dna_feature_transformer
        else:
            self.dna_feature_transformer = IdentityDNAFeatureTransformer()

        if classification_module is None:
            raise ValueError("classification_module must be provided! ")

        self.classification_module = classification_module

    def encode_dna(self, dna_sequences: List[str]):
        inputs = self.dna_tokenizer(dna_sequences, return_tensors="pt", padding=True)
        device = list(self.dna_encoder_model.parameters())[0].device
        inputs = {key: value.to(device) for key, value in inputs.items()}   
        outputs = self.dna_encoder_model(**inputs)
        embeddings = outputs[0].mean(dim=1)
        return embeddings


    def forward(self, batch):
        sequences = batch['sequence']
        embeddings = self.encode_dna(sequences)
        transformed_embeddings = self.dna_feature_transformer(embeddings)
        predictions = self.classification_module(transformed_embeddings)
        return predictions

class DHSMoEDetector(nn.Module):
    def __init__(self, base_model="zhihan1996/DNABERT-S", dna_feature_transformer=None,
                 classification_cls: nn.Module =None, 
                 classification_kwargs: dict = None,
                 ncomponents=20,):

        super(DHSMoEDetector, self).__init__()
        self.dna_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.dna_encoder_model = AutoModel.from_pretrained(base_model, trust_remote_code=True)

        if dna_feature_transformer is not None:
            self.dna_feature_transformer = dna_feature_transformer
        else:
            self.dna_feature_transformer = IdentityDNAFeatureTransformer()

        self.ncomponents = ncomponents
        self.classification_modules = torch.nn.ModuleList(
            [classification_cls(**classification_kwargs) for _ in range(ncomponents)])

    def encode_dna(self, dna_sequences: List[str]):
        inputs = self.dna_tokenizer(dna_sequences, return_tensors="pt", padding=True)
        device = list(self.dna_encoder_model.parameters())[0].device
        inputs = {key: value.to(device) for key, value in inputs.items()}   
        outputs = self.dna_encoder_model(**inputs)
        embeddings = outputs[0].mean(dim=1)
        return embeddings

    def forward(self, batch):
        sequences = batch['sequence']
        component_idx = batch['component']
        embeddings = self.encode_dna(sequences)
        transformed_embeddings = self.dna_feature_transformer(embeddings)

        # predict according to component index
        # component_idx: [n], n is the number of sequences

        predictions_per_component = []

        comp_idx = []
        for i in range(self.ncomponents):
            mask = component_idx == i
            comp_idx.append(torch.where(mask)[0])
            predictions = self.classification_modules[i](transformed_embeddings[mask])
            predictions_per_component.append(predictions)
        predictions = torch.cat(predictions_per_component, dim=0)[
            torch.argsort(torch.cat(comp_idx), dim=0)
        ]


        return predictions
        
        # return predictions for each component




class DHSMoERBFDetector(nn.Module):
    def __init__(self, base_model="zhihan1996/DNABERT-S", dna_feature_transformer=None,
                 classification_cls: nn.Module =None, 
                 classification_kwargs: dict = None,
                 feature_columns: List[str] = None,
                 ncomponents=20,
                 rbf_dimension=128):

        super(DHSMoERBFDetector, self).__init__()
        self.dna_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.dna_encoder_model = AutoModel.from_pretrained(base_model, trust_remote_code=True)
        self.rbf_dimension = rbf_dimension

        assert feature_columns is not None, "feature_columns must be provided! "
        self.feature_columns = feature_columns

        self.rbf_layers = torch.nn.ModuleList(
            [GaussianSmearing(cutoff_lower=0.0, cutoff_upper=10, num_rbf=rbf_dimension, trainable=True) for _ in feature_columns]
        )


        if dna_feature_transformer is not None:
            self.dna_feature_transformer = dna_feature_transformer
        else:
            self.dna_feature_transformer = IdentityDNAFeatureTransformer()

        self.ncomponents = ncomponents
        self.classification_modules = torch.nn.ModuleList(
            [classification_cls(**classification_kwargs) for _ in range(ncomponents)])

    def encode_dna(self, dna_sequences: List[str]):
        inputs = self.dna_tokenizer(dna_sequences, return_tensors="pt", padding=True)
        device = list(self.dna_encoder_model.parameters())[0].device
        inputs = {key: value.to(device) for key, value in inputs.items()}   
        outputs = self.dna_encoder_model(**inputs)
        embeddings = outputs[0].mean(dim=1)
        return embeddings

    def forward(self, batch):
        sequences = batch['sequence']
        component_idx = batch['component']

        embeddings = self.encode_dna(sequences)
        transformed_embeddings = self.dna_feature_transformer(embeddings)

        # predict according to component index
        # component_idx: [n], n is the number of sequences

        feature_columns = [batch[col] for col in self.feature_columns]
        features_rbf = [rbf_layer(col) for rbf_layer, col in zip(self.rbf_layers, feature_columns)]
        features = torch.cat(features_rbf, dim=1).float()

        predictions_per_component = []

        comp_idx = []
        for i in range(self.ncomponents):
            mask = component_idx == i
            comp_idx.append(torch.where(mask)[0])
            predictions = self.classification_modules[i](
                transformed_embeddings, features
            )
            predictions_per_component.append(predictions)
        predictions = torch.cat(predictions_per_component, dim=0)[
            torch.argsort(torch.cat(comp_idx), dim=0)
        ]


        return predictions
        
        # return predictions for each component

