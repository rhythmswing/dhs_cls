

from src.framework import DHSTrainingModule
from src.data import DHSDataModule

from src.modules import DHSDetector, DHSMoEDetector, DHSMoERBFDetector
from src.output import LinearClassificationModule, TwoLayerJointClassificationModule, TwoLayerClassificationModule  


"""
Defines the configuration for the various models used in the project.
"""

def filtered_dnabert_linear(n_classes):
    model = DHSDetector(base_model="zhihan1996/DNABERT-S", 
                        classification_module=LinearClassificationModule(768, n_classes))
    
    data_module = DHSDataModule(feather_path="data/filtered_dataset.ftr",
                                label_columns=(11, 15), batch_size=32)

    return DHSTrainingModule(model), data_module

    
def filtered_dnabert_moe_linear(n_classes, n_components):
    model = DHSMoEDetector(base_model="zhihan1996/DNABERT-S",
                           classification_cls=LinearClassificationModule,
                           classification_kwargs={"input_size": 768, "output_size": n_classes},
                           ncomponents=n_components)

    data_module = DHSDataModule(feather_path="data/filtered_dataset.ftr",
                                label_columns=(11, 15), batch_size=32, 
                                feature_columns=['component'])

    return DHSTrainingModule(model), data_module

def filtered_dnabert_two_layer(n_classes):
    model = DHSDetector(base_model="zhihan1996/DNABERT-S", 
                        classification_module=TwoLayerClassificationModule(768, 256, n_classes))
    
    data_module = DHSDataModule(feather_path="data/filtered_dataset.ftr",
                                label_columns=(11, 15), batch_size=32)

    return DHSTrainingModule(model), data_module

def filtered_dnabert_moe_two_layer(n_classes, n_components):
    model = DHSMoEDetector(base_model="zhihan1996/DNABERT-S",
                           classification_cls=TwoLayerClassificationModule,
                           classification_kwargs={"input_size": 768, "output_size": n_classes, "hidden_size": 256},
                           ncomponents=n_components)

    data_module = DHSDataModule(feather_path="data/filtered_dataset.ftr",
                                label_columns=(11, 15), batch_size=32, 
                                feature_columns=['component'])

    return DHSTrainingModule(model), data_module

def filtered_dnabert_moe_rbf(n_classes, n_components,
                             feature_columns=None,
                             rbf_dimension=128):

    additional_ft_size = rbf_dimension * len(feature_columns)

    data_module = DHSDataModule(feather_path="data/filtered_dataset.ftr", 
                                label_columns=(11, 15), batch_size=32,
                                feature_columns=feature_columns + ['component'], 
                                normalize_features=feature_columns)
                            
    model = DHSMoERBFDetector(base_model="zhihan1996/DNABERT-S",
                              classification_cls=TwoLayerJointClassificationModule,
                                classification_kwargs={"input_size": 768, "output_size": n_classes, "hidden_size": 256, 
                                                        "additional_feature_size": additional_ft_size},
                                ncomponents=n_components,
                                rbf_dimension=rbf_dimension, 
                                feature_columns=feature_columns)
    return DHSTrainingModule(model), data_module
