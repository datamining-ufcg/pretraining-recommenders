# Pretraining Recommenders

This repository contains all the code and instructions to execute the experiments described in the paper *"Evaluating Pre-training Strategies for Collaborative Filtering"*, to be published at the [31st ACM Conference on User Modeling, Adaptation and Personalization (UMAP)](https://www.um.org/umap2023/) in Limassol, Cyprus on June 26-30, 2023.

Also, this repository holds the [Supplementary Material](./supplementary_material.md) for the paper.

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

* Note: some of these scripts should be executed with parameters. Check which ones of them have use `docopt` at the beginning of the file and how to properly pass these parameters.

### Preparing the data

First, split the data. The complementary material describes in detail the logic behind splitting the datasets. To do so, just execute the `generate_user_leakage_data.py` file in the `scripts/` folder. This will generate new folders inside `data/` with the proper number of interactions. For example, the folder `leakage_user_from_10m/` will contain the `ml100k.csv` and `ml1m.csv` files, each one being the sample with 100k and 1M interactions, but their counterparts will be placed in the `leakage_user_from_10m_negative/` folder, named `negative_ml100k.csv` and `negative_ml1m.csv`. To ease the cost of creating and re-creating the embeddings, especially since we have executed this experiment following a 5-fold cross validation, these folders will be populated with a few `.json` and `.npy` files which hold useful mappings and the user and item embeddings to avoid the need for evaluating them again for each fold.

After doing so, copy the movies file to each folder generated from this source: for example, copy the `movies.csv` from `data/ml20m/` to both `data/leakage_user_from_20m/` and `data/leakage_user_from_20m_negative/` in order to properly map them when transferring.

### Generating the embeddings

To generate the embeddings for the source model might be very RAM consuming. This is why we have decided to create the `generate_source_embeddings.py` script to do so separately: we had limited access to computers with enough RAM to generate the embeddings, but we could use the already generated embeddings to train models in less capable computers. Also, its important to notice that, since these embeddings will be placed in the same folder, its important to clear or generate them again when changing the target size from 100k to 1M. 

### Training the source models

Before transferring the item embeddings, we must train the models with it before. To do so, use the `train_source_models.py` script. It takes a few parameters to run, check which ones do you want to pass in order to run properly. This will generate new folders inside the `models/` folder. Also, since these models take a long time to run, we created a mechanism that allows saving the models every 10 epochs, allowing us to continue training them in case of a unwanted halting. Also, check that we have a script called `build_catalog.py`: whenever you stop running manually, please execute this script through the `make catalog` alias so that the catalog is properly updated with all the models completed and in progress inside the `models/` folder. This script generates a pair of `.json` files, one containing a map between parameters and paths, and the other containing the number of models inside each folder.

Here, it's important to understand that the source dataset with 19M interactions extracted from ML20M, for example, will use 80% of the available data, since the other 20% are used for validation, and generate 3 source models:
1. The one that initializes the embeddings randomly;
2. The one that inputs the training data and applies PCA to them, using the output to initialize the embeddings; and
3. The one that inputs the trianing data and applies Word2Vec, using the output to initialize the embeddings.

The same idea goes for Netflix, but having a different script, `train_netflix_source.py`.

### Training the target models

Having the source models properly trained, the script `train_target_models.py` allows training the target models. Please also check the parameters in docopt to set the transfer correctly. These models also have the mechanism for saving every 10 epochs, and the catalog applies to them more hardly than to the source models. 

Considering the note from the previous subsection, its also important to highlight that for every source model, 165 target models will be trained. This number is the combination for models given (1) the 5-fold cross validation, (2) the OOV mappings from 0 to 100%, and (3) the fallback initialization for the OOV items: 5 * 11 * 3 = 165. Therefore, a single dataset (e.g. ML20M) will generate 495 target models with the 1M target size.

The same idea goes for Netflix, but having a different script, `train_netflix_target.py`.

### Evaluating results

Upon training a model, a `.json` file is added to it's folder. This file contains the RMSE for each epoch as well as the settings used to train it. A few Python scripts are then used to iterate these files, building `.csv` tables used as input to both Python and R scripts. Both images and tables are outputted to the `results/` folder.
