# dhs_cls

The human genome, a blueprint of biological diversity and complexity, harbors regions highly susceptible to regulatory mechanisms that dictate the expression of genes. Among these, DNase I hypersensitive sites (DHSs) stand out as pivotal markers, offering insights into the regulatory DNA landscape. In a groundbreaking study by Meuleman et al., in 2020 the creation of high-resolution maps of DHSs across a wide array of human biosamples—encompassing 438 distinct cell and tissue types and states—has marked a significant advancement in our understanding of the genome's regulatory dimensions. 

We seek to build models to predict the cell- and tissue-specificity of DHSs based on their genome sequences. To this end, we harness the power of pre-trained models like DNABERT. However, our methodology extends beyond the capabilities of DNABERT alone, incorporating a Mixer of Expert technique based on the components that the DHS sequences belong to. This significantly enhances performance. This novel approach not only outperforms traditional methods and DNABERT configurations with a linear layer but also demonstrates exceptional proficiency in detecting cancer-related biosamples.

Our method base on DNABert-S and an Mixture-of-Expert that selects prediction model based on genome sequences' main DHS vocabulary component. 


## Requirements

Mostly, pytorch + pytorch_lightning + wandb. 
See requirements.txt. 


## Instructions

### Data Preparation
Run data/master_dataset.ipynb and filter_master.ipynb to generate data/filtered_dataset.ftr.

### Training
Use run.sh MODELTYPE to train a model. MODELTYPE can be one of the following:

- linear
- two_layer
- moe_linear
- moe_two_layer
- moe_rbf


### Evaluation
The best results is achieved with moe_linear. The script will call train.py and store best model checkpoint in experiments/MODELTYPE.

For evaluation, run eval.py PATH_TO_CHECKPOINT --model_type MODELTYPE --input PATH_TO_FILTERED_DATA.
PATH_TO_FILTERED_DATA should be data/filtered_dataset.ftr by default. 


## Code
src/ contains all model code with self-explanatory file names. In notebooks/linear_model.ipynb, classical baselines are implemented. 

