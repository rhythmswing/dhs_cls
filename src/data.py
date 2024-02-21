
from typing import Iterable, Union, List
from collections.abc import Iterable
from torch.utils.data import Dataset, Subset
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd


class DHSData(Dataset):
    """
    A PyTorch Dataset for handling DNA sequences with optional components.

    Attributes:
        dna_sequences (Iterable): The DNA sequences.
        labels (Iterable): Corresponding labels for the DNA sequences.
        components (Optional[Iterable]): Additional components related to each DNA sequence.
    """
    def __init__(self, dna_sequences, labels, components=None):
        """
        Initializes the DHSData dataset.

        Parameters:
            dna_sequences (Iterable): The DNA sequences.
            labels (Iterable): Corresponding labels for the sequences.
            components (Optional[Iterable]): Additional components for each sequence.
        """
        self.dna_sequences = dna_sequences
        self.labels = labels
        self.components = components

    def __len__(self):
        return len(self.dna_sequences)

    def __getitem__(self, idx):
        if self.components is not None:
            return self.dna_sequences[idx], self.labels[idx], self.components[idx]
        else:
            return self.dna_sequences[idx], self.labels[idx]


def compute_stats(dataset: Dataset, columns: List[str] = None):
    """
    Computes mean and standard deviation statistics for specified columns of a dataset.

    Parameters:
        dataset (Dataset): The dataset to compute statistics for.
        columns (List[str]): The columns to compute statistics for.

    Returns:
        dict: A dictionary where keys are column names and values are dictionaries with keys 'mean' and 'std'.
    """
    list_of_data = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    for t in loader:
        list_of_data.append(t)

    column_data = {k: torch.cat([t[k] for t in list_of_data]) for k in columns}
    column_stats = {k: {"mean": torch.mean(column_data[k]), "std": torch.std(column_data[k])} for k in columns}
    return column_stats


class DHSFeatherData(Dataset):
    """
    A PyTorch Dataset for handling DNA sequences stored in a Feather file format.

    Attributes:
        feather_data (DataFrame): Data loaded from a Feather file.
        label_columns (Union[int, Iterable[int]]): Column indices or range for label(s).
        feature_columns (Union[List[str], str], optional): Column names for features.
        dna_sequence_column (str): Column name for DNA sequences.
    """
    def __init__(self, feather_path: str, 
                 dna_sequence_column: str = 'sequence', 
                 feature_columns: Union[List[str], str] = None,
                 label_columns: Union[int, Iterable[int]] = None,
                 ) -> None:
        """
        Initializes the DHSFeatherData dataset from a Feather file.

        Parameters:
            feather_path (str): Path to the Feather file containing the dataset.
            dna_sequence_column (str, optional): Column name for DNA sequences. Defaults to 'sequence'.
            feature_columns (Union[List[str], str], optional): Column names for features.
            label_columns (Union[int, Iterable[int]], optional): Column index or indices for label(s).
        """
        super().__init__()

        self.feather_data = pd.read_feather(feather_path)
        self.label_columns = label_columns
        self.feature_columns = feature_columns
        self.dna_sequence_column = dna_sequence_column

        if isinstance(self.label_columns, int):
            self.bio_sample_labels = self.feather_data.iloc[:, self.label_columns]
        elif isinstance(self.label_columns, Iterable):
            assert len(self.label_columns) == 2
            if len(self.label_columns) == 1:
                self.bio_sample_labels = self.feather_data.iloc[:, self.label_columns[0]:]
            elif len(self.label_columns) == 2:
                self.bio_sample_labels = self.feather_data.iloc[:, self.label_columns[0]:self.label_columns[1]]
        else:
            raise ValueError("label_columns must be an int or an iterable of length 2")

        if isinstance(feature_columns, str):
            self.features = {feature_columns: np.array(self.feather_data[feature_columns])}
        elif isinstance(feature_columns, Iterable):
            self.features = {k: np.array(self.feather_data[k]) for k in feature_columns}
        elif feature_columns is None:
            self.features = None

        
        self.dna_sequences = np.array(self.feather_data[dna_sequence_column].values.tolist())
        self.bio_sample_labels = np.array(self.bio_sample_labels)

    def __len__(self):
        return len(self.dna_sequences)

    def compute_feature_stats(self):
        stats = {}
        if self.features is not None:
            for k in self.features:
                mean = np.mean(self.features[k])
                std = np.std(self.features[k])
                stats[k] = {"mean": mean, "std": std}
        return stats 
    
    def __getitem__(self, index: int) -> dict:
        if self.features is not None:
            feature_dict = {k: self.features[k][index] for k in self.features}
            return {'sequence': self.dna_sequences[index], 'label': self.bio_sample_labels[index]} | feature_dict
        else:
            return {'sequence': self.dna_sequences[index], 'label': self.bio_sample_labels[index]}
    

def apply_normalization(dataset: Dataset, column_stats: dict):
    """
    Applies normalization to the dataset using precomputed column statistics.

    Parameters:
        dataset (Dataset): The dataset to normalize.
        column_stats (dict): Precomputed statistics for columns to normalize.

    Returns:
        Dataset: A new dataset with normalized columns.
    """
    class _NormalizedDataset(Dataset):

        def __getitem__(self, index):
            retrieved = dataset[index]
            for col in column_stats:
                retrieved[col] = (retrieved[col] - column_stats[col]["mean"]) / column_stats[col]["std"]
            return retrieved
        def __len__(self):
            return len(dataset)
    return _NormalizedDataset()

class DHSDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling and processing DNA sequence data.

    Attributes:
        feather_path (str): Path to the Feather file containing the dataset.
        label_columns (Union[int, Iterable[int]]): Column indices or range for label(s).
        feature_columns (Union[str, List[str]]): Column names for features.
        batch_size (int): Batch size for DataLoader.
        normalize_features (Union[bool, List[str]]): Specifies if and which features to normalize.
    """
    
    def __init__(self, feather_path: str = None,
                 label_columns: Union[int, Iterable[int]] = None,
                 feature_columns: Union[str, List[str]] = None,
                 batch_size: int = 32,
                 normalize_features: Union[bool, List[str]] = False):
        """
        Initializes the DHSDataModule with dataset paths and processing parameters.

        Parameters:
            feather_path (str, optional): Path to the Feather file.
            label_columns (Union[int, Iterable[int]], optional): Column indices for labels.
            feature_columns (Union[str, List[str]], optional): Column names for features.
            batch_size (int, optional): Batch size for the DataLoader.
            normalize_features (Union[bool, List[str]], optional): Whether to normalize features.
        """
        super(DHSDataModule, self).__init__()

        
        self.feather_path = feather_path
        self.label_columns = label_columns
        self.batch_size = batch_size
        self.feature_columns = feature_columns
        self.normalize_features = normalize_features

    def prepare_data(self):

        self.master_dataset = DHSFeatherData(self.feather_path, 
                 label_columns=self.label_columns, feature_columns=self.feature_columns)
        
        if self.normalize_features == True:
            self.feature_to_normalize = self.feature_columns
        elif isinstance(self.normalize_features, list):
            self.feature_to_normalize = self.normalize_features
        else:
            self.feature_to_normalize = None

    def setup(self, stage: str):
            
        train_val_test = [0.7, 0.15, 0.15]
        idx = np.arange(len(self.master_dataset))
        np.random.shuffle(idx)
        train_idx, val_idx, test_idx = np.split(idx, (np.cumsum(train_val_test[:-1]) * len(idx)).astype(int))


        self.train_data = Subset(self.master_dataset, train_idx)

        self.val_data = Subset(self.master_dataset, val_idx)
        self.test_data = Subset(self.master_dataset, test_idx)

        if self.feature_to_normalize is not None:
            self.train_stats = compute_stats(self.train_data, columns=self.feature_to_normalize)
            self.train_data = apply_normalization(self.train_data, self.train_stats)
            self.val_data = apply_normalization(self.val_data, self.train_stats)
            self.test_data = apply_normalization(self.test_data, self.train_stats)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=32)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=32)
