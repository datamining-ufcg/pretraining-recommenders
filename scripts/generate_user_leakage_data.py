import os
import sys
import numpy as np
import itertools

from tqdm import tqdm

sys.path.append('../')
from src.config import RANDOM_STATE
from src.dataset import *

SEED = RANDOM_STATE
np.random.seed(SEED)


def maybe_create_folder(path):
    try:
        os.mkdir(path)
    except:
        pass


def generate_disjoint_sets(dataset, sample_size=1e3):
    df_dataset = dataset.load_as_dataframe()

    selected_users = set()
    users = df_dataset['userId'].unique().tolist()
    np.random.shuffle(users)
    interactions = 0

    for u in tqdm(users):
        interactions += len(df_dataset[df_dataset.userId == u])
        selected_users.add(u)

        if (interactions >= sample_size):
            break

    sample = df_dataset[df_dataset.userId.isin(selected_users)]
    negative_sample = df_dataset[~df_dataset.userId.isin(selected_users)]

    return sample, negative_sample


def save_samples(sampled, not_sampled, ds_name, sample_size):
    sample_name = 'ml100k' if sample_size == 1e5 else 'ml1m'
    ds_size = ds_name.lower().replace('ml', '')

    maybe_create_folder(f'../data/leakage_user_from_{ds_size}')
    sampled.to_csv(
        f'../data/leakage_user_from_{ds_size}/{sample_name}.csv', index=False)

    maybe_create_folder(f'../data/leakage_user_from_{ds_size}_negative')
    not_sampled.to_csv(
        f'../data/leakage_user_from_{ds_size}_negative/negative_{sample_name}.csv', index=False)


ML_DATASETS = [ExplicitML1M, ExplicitML10M, ExplicitML20M, ExplicitML25M]
SAMPLE_SIZES = [1e5, 1e6]

comb = list(itertools.product(ML_DATASETS, SAMPLE_SIZES))

for ds, sample_size in tqdm(comb):
    dataset = ds()
    if (dataset.name == "ML1M" and sample_size == 1e6):
        continue

    sampled, not_sampled = generate_disjoint_sets(dataset, sample_size)
    save_samples(sampled, not_sampled, dataset.name, sample_size)
