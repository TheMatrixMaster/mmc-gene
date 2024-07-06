# Multimodal Contrastive
This module provides a framework for training and evaluating contrastive learning models on multimodal biological datasets. Specifically, we provide implementations of CLIP and GMC on the CellPainting dataset.

## Setup
```bash
# Set up conda environment:
conda env create -f env.yml

# Install development version of multimodal_contrastive module
pip install -e .
```

## Train a model
We use hydra to manage configurations. To train a model, you must specify a configuration file in the `configs/` directory. Before running the training script, you must set valid paths to the datasets, log directory, and other parameters in the config file. For example, to train a 2-modality (structure and morphology) GMC model on the PUMA dataset, you can use the following command:

```bash
# In project root directory
python multimodal_contrastive/pretrain.py --config-name puma_sm_gmc.yaml
```

## Evaluation
All evaluation protocols are also managed through hydra config files under the `configs/evaluations` directory. Currently, we support the following evaluation protocols:

- `latent_dist_correlation.yaml`: Measures the cross-modality latent distance correlation between a pair of samples in the test set. See the paper for more details.
- `retrieval_evaluation.yaml`: Evaluates the cross-modality retrieval performance on the test set.
- `linear_probe.yaml`: Trains a linear classifier on the learned representations and evaluates the classification accuracy on the test set.


### More about configuration files
You may add configuration files or modify/use existing ones under `configs/` to train as needed. 

some configuration files:
1. configs/puma_smg_gmc.yaml: 3modality GMC model use PUMA dataset
2. configs/puma_sm_gmc.yaml: 2modality GMC model use PUMA dataset
3. configs/puma_sm_clip.yaml: 2modality CLIP model use PUMA dataset
4. configs/jump_sm_gmc.yaml: 2modality GMC model use JUMP dataset
5. configs/jump_sm_clip.yaml: 2modality CLIP model use JUMP dataset
