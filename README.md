# Pretraining Recommenders

This repository contains all the code and instructions to execute the experiments described in the paper *"Evaluating Pre-training Strategies for Collaborative Filtering"*, to be published at the [31st ACM Conference on User Modeling, Adaptation and Personalization (UMAP)](https://www.um.org/umap2023/) in Limassol, Cyprus on June 26-30, 2023.

Also, this repository holds the [Complementary Material](./complementary_material.md) for the paper.

## Executing this code

To properly execute the code in this repository, follow the steps below:

1. Download and extract the MovieLens and Netflix datasets in their respective folders inside `data/`;
2. Set your Python version to 3.9.7. Personally recommend using Anaconda for doing this;
3. Install the dependencies using `pip install -r requirements.txt`;
4. Compile the Cython model files using `make compile`.

* Note: whenever changing environments and modifying the `.pyx` files, you must compile the models again to apply the changes.

## Replicating the experiments

After executing all the steps in the previous section, you are able to run the code and replicate the experiments. Be aware that it requires a lot of RAM to load the datasets, around 150GB of hard disk to save all the models. Considering the number of combinations to generate all models, its fair to say that requires more than a month to run all the experiments in a single 6-cores computer.

This section describes how to execute the files in the `scripts/` folder to reproduce the experiments properly.

### Preparing the data

First, split the data. The complementary material describes in detail the logic behind splitting the datasets. To do so, just execute the `generate_user_leakage_data.py` file in the `scripts/` folder. This will generate new folders inside `data/` with the proper number of interactions. For example, the folder `leakage_user_from_10m/` will contain the `ml100k.csv` and `ml1m.csv` files, each one being the sample with 100k and 1M interactions, but their counterparts will be placed in the `leakage_user_from_10m_negative/` folder, named `negative_ml100k.csv` and `negative_ml1m.csv`. To ease the cost of creating and re-creating the embeddings, especially since we have executed this experiment following a 5-fold cross validation, these folders will be populated with a few `.json` and `.npy` files which hold useful mappings and the user and item embeddings to avoid the need for evaluating them again for each fold.

* To be continued
