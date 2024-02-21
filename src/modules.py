

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from typing import List, Dict

class IdentityDNAFeatureTransformer(torch.nn.Module):
    """
    A pass-through transformer module for DNA feature vectors.

    This module acts as a placeholder in the pipeline, returning the input without any modification.
    """
    def __init__(self):
        super(IdentityDNAFeatureTransformer, self).__init__()

    def forward(self, x):
        return x

class GaussianSmearing(nn.Module):
    """
    Applies Gaussian smearing to input distances, transforming them into a radial basis function (RBF) space.

    Parameters:
        cutoff_lower (float): The lower bound of the distance cutoff.
        cutoff_upper (float): The upper bound of the distance cutoff.
        num_rbf (int): The number of radial basis functions.
        trainable (bool): If True, the parameters of the Gaussian functions are trainable.

    Attributes:
        coeff (Parameter or Buffer): The coefficient used in the Gaussian function, representing the width of the Gaussian.
        offset (Parameter or Buffer): The offset for each Gaussian basis function, determining its center.
    """
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
    """
    A DNA sequences classifier using a pretrained transformer model and an optional feature transformer.

    Parameters:
        base_model (str): Identifier for the pretrained transformer model.
        dna_feature_transformer (torch.nn.Module, optional): A module to transform the extracted DNA features.
        classification_module (torch.nn.Module): A module for classifying the transformed features.

    Attributes:
        dna_tokenizer (AutoTokenizer): The tokenizer for the DNA sequences.
        dna_encoder_model (AutoModel): The encoder model for generating DNA sequence embeddings.
    """
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
        """
        Encodes a list of DNA sequences into embeddings.

        Parameters:
            dna_sequences (List[str]): The DNA sequences to encode.

        Returns:
            Tensor: The embeddings of the DNA sequences.
        """
        inputs = self.dna_tokenizer(dna_sequences, return_tensors="pt", padding=True)
        device = list(self.dna_encoder_model.parameters())[0].device
        inputs = {key: value.to(device) for key, value in inputs.items()}   
        outputs = self.dna_encoder_model(**inputs)
        embeddings = outputs[0].mean(dim=1)
        return embeddings

    def forward(self, batch: Dict):
        """
        Forward pass for the DHSDetector.

        Parameters:
            batch (dict): A batch containing 'sequence' as a key with a list of DNA sequences.

        Returns:
            Tensor: The classification predictions for the batch.
        """
        sequences = batch['sequence']
        embeddings = self.encode_dna(sequences)
        transformed_embeddings = self.dna_feature_transformer(embeddings)
        predictions = self.classification_module(transformed_embeddings)
        return predictions

class DHSMoEDetector(nn.Module):
    """
    A mixture of experts (MoE) model for DNA sequence detection, utilizing a transformer for sequence encoding and multiple classification modules for MoE predictions.
    Experts are selected based on Main DHS Vocabulary Components. "component" attribute is mandatory for the input batch.

    Parameters:
        base_model (str): Identifier for the pretrained transformer model.
        dna_feature_transformer (torch.nn.Module, optional): Module to transform the extracted DNA features. Defaults to an identity transformer if None is provided.
        classification_cls (nn.Module): The class of the classification module to be instantiated for each expert.
        classification_kwargs (dict): Arguments for initializing the classification modules.
        ncomponents (int): The number of expert components in the MoE model.

    Attributes:
        dna_tokenizer (AutoTokenizer): The tokenizer for DNA sequences.
        dna_encoder_model (AutoModel): The encoder model for generating DNA sequence embeddings.
        classification_modules (torch.nn.ModuleList): A list of classification modules, one for each expert.
    """
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
        """
        Forward pass for processing a batch of DNA sequences through the MoE model.

        Parameters:
            batch (dict): A batch containing 'sequence' and 'component' keys with DNA sequences and their respective component indices.

        Returns:
            Tensor: The combined predictions from the expert components for each sequence in the batch.
        """
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
        




class DHSMoERBFDetector(nn.Module):
    """
    A mixture of experts (MoE) model for DNA sequence detection, incorporating radial basis function (RBF) feature transformation for enhanced feature representation.

    Parameters:
        base_model (str): Identifier for the pretrained transformer model.
        dna_feature_transformer (torch.nn.Module, optional): Module to transform the extracted DNA features. Defaults to an identity transformer if None is provided.
        classification_cls (nn.Module): The class of the classification module to be instantiated for each expert.
        classification_kwargs (dict): Arguments for initializing the classification modules.
        feature_columns (List[str]): List of column names for additional features to be transformed via RBF.
        ncomponents (int): The number of expert components in the MoE model.
        rbf_dimension (int): The dimensionality of the RBF-transformed feature space.

    Attributes:
        dna_tokenizer (AutoTokenizer): The tokenizer for DNA sequences.
        dna_encoder_model (AutoModel): The encoder model for generating DNA sequence embeddings.
        rbf_layers (torch.nn.ModuleList): A list of GaussianSmearing modules for RBF transformation of additional features.
        classification_modules (torch.nn.ModuleList): A list of classification modules, one for each expert.
    """
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
        """
        Forward pass for processing a batch of DNA sequences through the MoE model with RBF feature transformation.

        Parameters:
            batch (dict): A batch containing 'sequence', 'component', and additional feature columns specified in `feature_columns`.

        Returns:
            Tensor: The combined predictions from the expert components for each sequence in the batch, utilizing RBF-transformed features.
        """
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

